"""
web/main.py
-----------
FastAPI web server for the Early Fire Detection System.

Endpoints:
    POST /v1/predict       — Standard RT-DETR-L inference on uploaded image.
    POST /v1/predict_sahi  — SAHI-sliced inference for small object detection.
    GET  /health           — Health check.

TODO: Set the following environment variables (or add to .env):
    MODEL_CONFIG_PATH — path to configs/default.yaml (default: configs/default.yaml)
"""

import io
import logging
import os
import time
from typing import List

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="🔥 Hệ thống Phát hiện Cháy Sớm",
    description="Early Fire Detection System — RT-DETR-L with SAHI",
    version="1.0.0",
)

# Allow all origins for development; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

# ---------------------------------------------------------------------------
# Global model instance (loaded at startup)
# ---------------------------------------------------------------------------
_model = None
_config = None


@app.on_event("startup")
async def startup_event() -> None:
    """Load the RT-DETR-L model and configuration at server startup."""
    global _model, _config

    config_path = os.getenv("MODEL_CONFIG_PATH", "configs/default.yaml")
    logger.info("Loading configuration from: %s", config_path)

    try:
        from src.config import load_config
        from src.models.rtdetr_model import FireDetectionModel

        _config = load_config(config_path)
        _model = FireDetectionModel(_config)
        logger.info("Model loaded successfully.")
    except Exception as exc:
        logger.error("Failed to load model at startup: %s", exc)
        # Continue serving — /health will still respond


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class PredictResponse(BaseModel):
    """Response body for /v1/predict and /v1/predict_sahi."""
    question: str
    detections: List[dict]
    fire_detected: bool
    smoke_detected: bool
    processing_time_ms: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pil_to_numpy(pil_img: Image.Image) -> np.ndarray:
    """Convert a PIL image to a NumPy RGB array."""
    return np.array(pil_img.convert("RGB"), dtype=np.uint8)


def _run_prediction(image_np: np.ndarray, use_sahi: bool = False) -> dict:
    """Run the model and return a structured result dict."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    t0 = time.perf_counter()
    if use_sahi:
        detections = _model.predict_with_sahi(image_np)
    else:
        conf_thresh = _config.inference.confidence_threshold if _config else 0.35
        detections = _model.predict(image_np, confidence=conf_thresh)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    fire_detected = any(d["class"].lower() == "fire" for d in detections)
    smoke_detected = any(d["class"].lower() == "smoke" for d in detections)

    formatted = [
        {
            "class": d["class"],
            "confidence": d["confidence"],
            "bbox": d["bbox"],
        }
        for d in detections
    ]

    return {
        "question": "Fire detection result",
        "detections": formatted,
        "fire_detected": fire_detected,
        "smoke_detected": smoke_detected,
        "processing_time_ms": round(elapsed_ms, 2),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    """Serve the web UI index page."""
    index_html = os.path.join(_static_dir, "index.html")
    if os.path.exists(index_html):
        return FileResponse(index_html)
    return JSONResponse({"message": "🔥 Fire Detection API — visit /docs for API docs."})


@app.get("/health")
async def health_check():
    """Return server and model health status.

    Returns:
        JSON with ``status``, ``model``, and ``classes`` fields.
    """
    return {
        "status": "ok",
        "model": "rtdetr-l",
        "classes": ["Fire", "Smoke"],
    }


@app.post("/v1/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    """Run RT-DETR-L inference on an uploaded image.

    Args:
        file: Uploaded image file (JPEG, PNG, etc.).

    Returns:
        JSON with detections, fire/smoke flags, and processing time.
    """
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents))
        image_np = _pil_to_numpy(pil_img)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    return _run_prediction(image_np, use_sahi=False)


@app.post("/v1/predict_sahi", response_model=PredictResponse)
async def predict_sahi(file: UploadFile = File(...)):
    """Run SAHI-sliced inference for small object detection.

    Especially useful for detecting small/distant fire sources captured from
    balcony cameras (folder ``04_SAHI_Small_Objects``).

    Args:
        file: Uploaded image file.

    Returns:
        JSON with detections, fire/smoke flags, and processing time.
    """
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents))
        image_np = _pil_to_numpy(pil_img)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    return _run_prediction(image_np, use_sahi=True)
