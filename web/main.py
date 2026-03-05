"""
============================================================
🌐 FastAPI Web Server — Demo Phát hiện Cháy Sớm
============================================================

MỤC ĐÍCH:
    Server web cho phép:
    1. Upload ảnh → detect lửa/khói → trả về kết quả
    2. Upload video → detect realtime → stream kết quả
    3. Webcam live detection (qua browser)
    4. API endpoint cho integration

GIẢI THÍCH FASTAPI CHO BẠN:
    FastAPI là web framework Python hiện đại:
    - Nhanh (async support)
    - Tự tạo API docs tại /docs
    - Type hints → auto validation
    - Dễ deploy (Docker, cloud)

ENDPOINT:
    GET  /              → Trang web chính (UI)
    POST /api/detect    → Upload ảnh → trả về detections (JSON)
    POST /api/detect-image → Upload ảnh → trả về ảnh có bbox (image)
    GET  /api/health    → Kiểm tra server status

CÁCH CHẠY:
    uvicorn web.main:app --host 0.0.0.0 --port 8000
    Mở browser: http://localhost:8000
"""

import io
import os
import sys
import base64
import logging
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# Thêm project root vào path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, Config
from src.models.rtdetr_model import FireDetectionModel
from src.utils.visualization import draw_detections, draw_detections_fancy

# ============================================================
# Khởi tạo FastAPI app
# ============================================================

app = FastAPI(
    title="🔥 Hệ thống Phát hiện Cháy Sớm",
    description="Early Fire Detection System — RT-DETR + SAHI",
    version="1.0.0",
)

# CORS middleware — cho phép browser gọi API
# (Cần khi frontend và backend ở domain/port khác nhau)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả origins (dev mode)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (HTML, CSS, JS)
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ============================================================
# Global variables — model sẽ load khi startup
# ============================================================
model: Optional[FireDetectionModel] = None
config: Optional[Config] = None

logger = logging.getLogger(__name__)


@app.on_event("startup")
async def startup_event():
    """
    Khởi tạo model khi server start.

    TẠI SAO LOAD MODEL Ở ĐÂY?
        - Load model chỉ 1 lần (khi server bắt đầu)
        - Các request sau dùng chung model instance
        - Không phải load lại model cho mỗi request (chậm!)
    """
    global model, config

    try:
        config_path = os.getenv("CONFIG_PATH", "configs/default.yaml")
        weights_path = os.getenv("WEIGHTS_PATH", None)

        config = load_config(config_path)

        # Load model
        model = FireDetectionModel(config, weights_path=weights_path)
        logger.info("✅ Model loaded successfully!")

    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        logger.info("⚠️ Server sẽ chạy nhưng detect sẽ trả lỗi")
        logger.info("   Hãy đặt WEIGHTS_PATH environment variable")


# ============================================================
# ROUTES
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Trang chủ — PWA Mobile-First Web UI.
    Serve file index.html từ static/
    """
    html_path = static_dir / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding='utf-8'))
    else:
        return HTMLResponse(content="""
        <html>
        <body style="background:#1a1a2e;color:white;text-align:center;padding:50px">
            <h1>🔥 Fire Detection System</h1>
            <p>File index.html chưa tồn tại trong web/static/</p>
        </body>
        </html>
        """)


@app.get("/sw.js")
async def service_worker():
    """
    Serve Service Worker ở root path.

    TẠI SAO PHẢI Ở ROOT?
        Service Worker có "scope" — chỉ kiểm soát các URL
        cùng cấp hoặc bên dưới nó.
        Nếu serve ở /static/sw.js → scope chỉ là /static/*
        Nếu serve ở /sw.js → scope là /* (toàn bộ app)
    """
    from fastapi.responses import FileResponse
    sw_path = static_dir / "sw.js"
    if sw_path.exists():
        return FileResponse(
            sw_path,
            media_type="application/javascript",
            headers={"Service-Worker-Allowed": "/"},
        )
    return JSONResponse({"error": "sw.js not found"}, status_code=404)


@app.get("/api/health")
async def health_check():
    """
    Kiểm tra trạng thái server.

    Returns:
        JSON: status, model_loaded, device
    """
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": model.device if model else "N/A",
        "model_architecture": config.model.architecture if config else "N/A",
    }


@app.post("/api/detect")
async def detect(
    file: UploadFile = File(...),
    confidence: float = Query(default=0.35, ge=0.0, le=1.0),
    use_sahi: bool = Query(default=False),
):
    """
    Upload ảnh → Phát hiện lửa/khói → Trả về JSON.

    FLOW:
        1. Nhận file upload
        2. Decode thành numpy array
        3. Chạy model inference (hoặc SAHI)
        4. Trả về list detections

    Args:
        file: Ảnh upload (jpg, png, etc.)
        confidence: Ngưỡng confidence (0-1)
        use_sahi: Dùng SAHI inference không

    Returns:
        JSON: {detections: [...], count: int, inference_time_ms: float}
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model chưa được load. Kiểm tra config và weights."
        )

    # Đọc file upload
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Không thể đọc ảnh. Hãy upload file ảnh hợp lệ (jpg, png)."
        )

    # Inference
    import time
    start = time.time()

    if use_sahi:
        # SAHI cần file path → lưu tạm
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        detections = model.predict_with_sahi(tmp_path, conf_threshold=confidence)
        os.unlink(tmp_path)
    else:
        detections = model.predict(image, conf_threshold=confidence)

    inference_time = (time.time() - start) * 1000  # ms

    return {
        "detections": detections,
        "count": len(detections),
        "inference_time_ms": round(inference_time, 2),
        "image_size": {"width": image.shape[1], "height": image.shape[0]},
    }


@app.post("/api/detect-image")
async def detect_image(
    file: UploadFile = File(...),
    confidence: float = Query(default=0.35, ge=0.0, le=1.0),
    use_sahi: bool = Query(default=False),
    fancy: bool = Query(default=True),
):
    """
    Upload ảnh → Phát hiện → Trả về ẢNH có bbox.

    Giống /api/detect nhưng trả về image thay vì JSON.
    Dùng cho hiển thị kết quả trực quan trên web UI.

    Returns:
        JPEG image có vẽ bounding boxes
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model chưa load")

    # Đọc ảnh
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Ảnh không hợp lệ")

    # Inference
    if use_sahi:
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        detections = model.predict_with_sahi(tmp_path, conf_threshold=confidence)
        os.unlink(tmp_path)
    else:
        detections = model.predict(image, conf_threshold=confidence)

    # Vẽ bbox
    if fancy:
        result = draw_detections_fancy(image, detections)
    else:
        result = draw_detections(image, detections)

    # Encode thành JPEG
    _, buffer = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, 90])
    io_buf = io.BytesIO(buffer.tobytes())

    return StreamingResponse(io_buf, media_type="image/jpeg")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "web.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload khi code thay đổi (dev mode)
    )
