"""
============================================================
🌐 FastAPI Web Server — Demo Phát hiện Cháy Sớm
============================================================

MỤC ĐÍCH:
    Server web cho phép:
    1. Upload ảnh → detect lửa/khói → trả về kết quả
    2. Upload video → detect realtime → stream kết quả
    3. Webcam live detection (qua browser)
    4. Camera IP stream → detect + tự động ghi video
    5. API endpoint cho integration

ENDPOINT:
    GET  /              → Trang web chính (UI)
    POST /api/detect    → Upload ảnh → trả về detections (JSON)
    POST /api/detect-image → Upload ảnh → trả về ảnh có bbox (image)
    GET  /api/health    → Kiểm tra server status
    POST /api/camera-stream/start  → Bắt đầu stream từ Camera IP
    POST /api/camera-stream/stop   → Dừng stream Camera IP
    GET  /api/camera-stream/frame  → Lấy frame mới nhất (MJPEG)
    GET  /api/camera-stream/status → Trạng thái camera stream
    GET  /api/recordings           → Danh sách video đã ghi
    GET  /api/recordings/{name}    → Tải video đã ghi

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
import threading
import time as time_module
from pathlib import Path
from datetime import datetime
from collections import deque
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

# ============================================================
# Camera IP Stream + Video Recording — State
# ============================================================
# Camera stream state
camera_thread: Optional[threading.Thread] = None
camera_running = False
camera_url = ""
latest_frame: Optional[np.ndarray] = None
latest_detections: list = []
frame_lock = threading.Lock()

# Video recording state
is_recording = False
video_writer: Optional[cv2.VideoWriter] = None
recording_filename = ""
no_detection_counter = 0
NO_DETECTION_STOP_THRESHOLD = 15  # Dừng ghi sau 15 frame không detect (~30s)
RECORDINGS_DIR = Path(__file__).resolve().parent.parent / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)


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


@app.get("/{page}.html", response_class=HTMLResponse)
async def serve_html_pages(page: str):
    """
    Serve các trang HTML phụ trợ (people.html, devices.html, v.v) trực tiếp từ URL root.
    """
    html_path = static_dir / f"{page}.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding='utf-8'))
    raise HTTPException(status_code=404, detail="Page not found")


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
# CAMERA IP STREAM — Đọc stream từ camera IP (RTSP/HTTP)
# ============================================================

def camera_stream_worker(url: str):
    """
    Worker thread: đọc frame từ Camera IP → detect → ghi video và cảnh báo đa kênh.

    NÂNG CẤP:
        1. Sử dụng class VideoStream để chạy UDP FFMPEG ngầm không bị lag.
        2. Tích hợp thuật toán cửa sổ trượt (sliding window) 3/5 frame chống báo giả.
        3. Tích hợp AlertManager để tự động gọi Zalo/Telegram.
    """
    global camera_running, latest_frame, latest_detections
    global is_recording, video_writer, recording_filename, no_detection_counter

    from src.utils.camera_stream import VideoStream
    from src.utils.alert import AlertManager

    logger.info(f"Khởi tạo luồng camera: {url}")
    # Resize camera xuống 1280x720 hoặc 640x640 thay vì gốc 4K để tránh kẹt CPU
    cam = VideoStream(url, resize_dim=(1280, 720)).start()
    time_module.sleep(1.0)

    if cam.stream is not None and getattr(cam, 'ret', False) is False:
        logger.error(f"❌ Không thể kết nối camera: {url}")
        camera_running = False
        cam.stop()
        return

    logger.info(f"✅ Đã kết nối camera: {url}")
    
    alert_mgr = None
    if config:
        try:
            alert_mgr = AlertManager(config)
            # Ép ghi đè điều kiện 3 consecutive object của alert
            alert_mgr.frames_threshold = 3 
        except Exception as e:
            logger.error(f"Không thể khởi tạo AlertManager: {e}")

    # Thuật toán chống báo giả: Lưu true/false của 5 frame gần nhất
    prediction_history = deque(maxlen=5)
    
    frame_count = 0
    detect_interval = 2  # Detect mỗi 2 frame

    while camera_running:
        ret, frame = cam.read()
        if not ret or frame is None:
            time_module.sleep(0.1)
            continue

        with frame_lock:
            latest_frame = frame.copy()

        frame_count += 1

        # Detect mỗi N frame
        if model is not None and frame_count % detect_interval == 0:
            try:
                # 1. Chạy AI (Sử dụng config 0.35/0.25 tuỳ model)
                detections = model.predict(frame, conf_threshold=0.30)
                with frame_lock:
                    latest_detections = detections

                has_fire_or_smoke = len(detections) > 0
                
                # 2. Lưu lịch sử
                prediction_history.append(1 if has_fire_or_smoke else 0)

                # Cửa sổ trượt: Có lửa >=3/5 frame mới xác nhận
                is_confirmed_fire = sum(prediction_history) >= 3

                # 3. Gửi cảnh báo
                if alert_mgr:
                    if is_confirmed_fire:
                        alert_mgr.process_detections(frame, detections)
                    else:
                        alert_mgr.process_detections(frame, [])

                # === Logic ghi video ===
                if has_fire_or_smoke:
                    no_detection_counter = 0

                    if not is_recording:
                        start_recording(frame)

                    if is_recording and video_writer is not None:
                        from src.utils.visualization import draw_detections
                        annotated = draw_detections(frame, detections)
                        video_writer.write(annotated)
                else:
                    if is_recording:
                        no_detection_counter += 1
                        if video_writer is not None:
                            video_writer.write(frame)

                        if no_detection_counter >= NO_DETECTION_STOP_THRESHOLD:
                            stop_recording()

            except Exception as e:
                logger.error(f"Detect error: {e}")

        # Worker loop không bị block bởi I/O mạng nữa, sleep 0.05 để nhả CPU
        time_module.sleep(0.05)

    # Cleanup
    cam.stop()
    if is_recording:
        stop_recording()
    logger.info("📷 Camera stream stopped")


def start_recording(frame: np.ndarray):
    """Bắt đầu ghi video khi phát hiện lửa/khói."""
    global is_recording, video_writer, recording_filename, no_detection_counter

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    recording_filename = f"fire_detected_{timestamp}.mp4"
    filepath = RECORDINGS_DIR / recording_filename

    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(filepath), fourcc, 2.0, (w, h))
    is_recording = True
    no_detection_counter = 0

    logger.info(f"🔴 BẮT ĐẦU GHI VIDEO: {recording_filename}")


def stop_recording():
    """Dừng ghi video."""
    global is_recording, video_writer, recording_filename, no_detection_counter

    if video_writer is not None:
        video_writer.release()
        video_writer = None

    logger.info(f"⏹️ DỪNG GHI VIDEO: {recording_filename}")
    is_recording = False
    recording_filename = ""
    no_detection_counter = 0


@app.post("/api/camera-stream/start")
async def start_camera_stream(url: str = Query(..., description="Camera IP URL (RTSP hoặc HTTP)")):
    """
    Bắt đầu đọc stream từ Camera IP.

    Args:
        url: URL camera, ví dụ:
            - rtsp://admin:pass@192.168.1.64:554/Streaming/channels/101
            - http://192.168.1.100:8080/video
    """
    global camera_thread, camera_running, camera_url

    if camera_running:
        return {"status": "already_running", "url": camera_url}

    camera_url = url
    camera_running = True
    camera_thread = threading.Thread(target=camera_stream_worker, args=(url,), daemon=True)
    camera_thread.start()

    return {"status": "started", "url": url}


@app.post("/api/camera-stream/stop")
async def stop_camera_stream():
    """Dừng stream Camera IP."""
    global camera_running
    camera_running = False
    return {"status": "stopped"}


@app.get("/api/camera-stream/status")
async def camera_stream_status():
    """Trạng thái camera stream + recording."""
    with frame_lock:
        det_count = len(latest_detections)
        det_list = latest_detections.copy()

    return {
        "streaming": camera_running,
        "url": camera_url if camera_running else "",
        "recording": is_recording,
        "recording_file": recording_filename if is_recording else "",
        "detections": det_list,
        "detection_count": det_count,
    }


@app.get("/api/camera-stream/frame")
async def get_camera_frame():
    """Lấy frame mới nhất từ camera stream (JPEG)."""
    with frame_lock:
        frame = latest_frame

    if frame is None:
        raise HTTPException(status_code=404, detail="Chưa có frame. Camera chưa kết nối.")

    # Vẽ detections lên frame
    if latest_detections:
        from src.utils.visualization import draw_detections_fancy
        frame = draw_detections_fancy(frame, latest_detections)

    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    io_buf = io.BytesIO(buffer.tobytes())
    return StreamingResponse(io_buf, media_type="image/jpeg")


# ============================================================
# RECORDINGS — Quản lý video đã ghi
# ============================================================

@app.get("/api/recordings")
async def list_recordings():
    """Danh sách video đã ghi khi phát hiện lửa."""
    recordings = []
    for f in sorted(RECORDINGS_DIR.glob("*.mp4"), reverse=True):
        stat = f.stat()
        recordings.append({
            "filename": f.name,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        })
    return {"recordings": recordings, "count": len(recordings)}


@app.get("/api/recordings/{filename}")
async def download_recording(filename: str):
    """Tải video đã ghi."""
    from fastapi.responses import FileResponse
    filepath = RECORDINGS_DIR / filename

    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File không tồn tại")

    return FileResponse(
        filepath,
        media_type="video/mp4",
        filename=filename,
    )


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
