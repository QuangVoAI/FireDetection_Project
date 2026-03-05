"""
Real-Time Inference Demo
========================
Runs RT-DETR-L + (optionally) SAHI on a live camera stream or video file
and triggers multi-channel alerts when fire/smoke is detected.

Usage
-----
    python src/inference.py --source 0
    python src/inference.py --source rtsp://...
    python src/inference.py --source path/to/video.mp4 --use-sahi
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Optional

import cv2
import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def _draw_detections(
    frame,
    detections: list,
    class_colors: dict,
) -> None:
    """Draw bounding boxes + labels on *frame* in-place."""
    for label, score, bbox in detections:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = class_colors.get(label.lower(), (0, 0, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {score:.1%}"
        cv2.putText(
            frame, text, (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
        )


def run_inference(
    weights: str = "models/weights/best.pt",
    source: Optional[str] = None,
    use_sahi: bool = False,
    config_inference: str = "config/inference_config.yaml",
    config_alert: str = "config/alert_config.yaml",
    display: bool = True,
    save_output: bool = False,
    output_path: str = "runs/inference/output.mp4",
) -> None:
    """
    Main real-time inference loop.

    Parameters
    ----------
    weights          : Path to trained weights.
    source           : Video source (int index, file path, or RTSP URL).
                       Falls back to VIDEO_SOURCE env var, then webcam (0).
    use_sahi         : Apply SAHI slicing (recommended for 04_SAHI folder).
    config_inference : Path to inference_config.yaml.
    config_alert     : Path to alert_config.yaml.
    display          : Show live annotated window.
    save_output      : Write annotated video to *output_path*.
    output_path      : Output video file path.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # ── Load configs ──────────────────────────────────────────────────────
    inf_cfg: dict = {}
    sahi_cfg: dict = {}
    if Path(config_inference).exists():
        with open(config_inference) as f:
            raw = yaml.safe_load(f)
        inf_cfg = raw.get("inference", {})
        sahi_cfg = raw.get("sahi", {})

    confidence = float(inf_cfg.get("confidence_threshold", 0.45))
    iou = float(inf_cfg.get("iou_threshold", 0.45))
    imgsz = int(inf_cfg.get("image_size", 640))
    device = str(inf_cfg.get("device", "0"))

    # ── Resolve video source ──────────────────────────────────────────────
    _source: int | str = source or os.getenv("VIDEO_SOURCE", "0")
    try:
        _source = int(_source)
    except (ValueError, TypeError):
        pass  # keep as string (file path / RTSP)

    # ── Load model ────────────────────────────────────────────────────────
    if not Path(weights).exists():
        logger.error("Weights not found: %s", weights)
        return

    from ultralytics import RTDETR

    model = RTDETR(weights)
    model_device = f"cuda:{device}" if str(device).isdigit() else device

    # ── SAHI model (lazy) ─────────────────────────────────────────────────
    sahi_model = None
    if use_sahi and sahi_cfg.get("enabled", True):
        from src.utils.sahi_integration import build_sahi_model
        sahi_model = build_sahi_model(weights, device=device, confidence_threshold=confidence)

    # ── Alert manager ─────────────────────────────────────────────────────
    from src.alerts.alert_manager import AlertManager
    alert_mgr = AlertManager(config_path=config_alert)

    # ── Open video stream ─────────────────────────────────────────────────
    cap = cv2.VideoCapture(_source)
    if not cap.isOpened():
        logger.error("Cannot open video source: %s", _source)
        return

    fps_cap = int(inf_cfg.get("fps_cap", 30))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer: Optional[cv2.VideoWriter] = None
    if save_output:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps_cap, (frame_w, frame_h))

    class_colors = {"fire": (0, 80, 255), "smoke": (180, 180, 180)}

    logger.info("Starting inference. Source=%s | SAHI=%s", _source, use_sahi)

    frame_times: list = []
    screenshot_path: Optional[str] = None

    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            logger.info("Stream ended.")
            break

        # ── Run detection ──────────────────────────────────────────────────
        if sahi_model and use_sahi:
            from src.utils.sahi_integration import predict_frame_with_sahi
            detections = predict_frame_with_sahi(sahi_model, frame, sahi_cfg)
        else:
            results = model.predict(
                source=frame,
                conf=confidence,
                iou=iou,
                imgsz=imgsz,
                device=model_device,
                verbose=False,
            )
            detections = []
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    score = float(box.conf[0])
                    bbox = box.xyxy[0].tolist()
                    detections.append((label, score, bbox))

        # ── Draw & alert ───────────────────────────────────────────────────
        _draw_detections(frame, detections, class_colors)

        fire_detections = [d for d in detections if d[0].lower() in ("fire", "smoke")]
        if fire_detections:
            max_conf = max(d[1] for d in fire_detections)
            # Save screenshot for Telegram
            if alert_mgr.confidence_threshold <= max_conf:
                import tempfile
                screenshot_path = str(Path(tempfile.gettempdir()) / "fire_alert_frame.jpg")
                cv2.imwrite(screenshot_path, frame)
            alert_mgr.on_detection(confidence=max_conf, frame_screenshot=screenshot_path)
        else:
            alert_mgr.reset()

        # ── FPS overlay ────────────────────────────────────────────────────
        elapsed = time.perf_counter() - t0
        frame_times.append(elapsed)
        if len(frame_times) > 30:
            frame_times.pop(0)
        fps = 1.0 / (sum(frame_times) / len(frame_times))
        cv2.putText(
            frame, f"FPS: {fps:.1f}", (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA,
        )

        if writer:
            writer.write(frame)

        if display:
            cv2.imshow("Fire Detection – RT-DETR-L", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("User quit.")
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    logger.info("Inference stopped.")


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fire Detection Real-Time Inference")
    p.add_argument("--weights", default="models/weights/best.pt", help="Path to best.pt")
    p.add_argument("--source", default=None, help="Video source (int, file, RTSP URL)")
    p.add_argument("--use-sahi", action="store_true", help="Enable SAHI slicing")
    p.add_argument("--no-display", action="store_true", help="Suppress live window")
    p.add_argument("--save", action="store_true", help="Save output video")
    p.add_argument("--output", default="runs/inference/output.mp4", help="Output video path")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_inference(
        weights=args.weights,
        source=args.source,
        use_sahi=args.use_sahi,
        display=not args.no_display,
        save_output=args.save,
        output_path=args.output,
    )
