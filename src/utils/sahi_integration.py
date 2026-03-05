"""
SAHI Integration
================
Wraps SAHI (Slicing Aided Hyper Inference) to detect small fire/smoke objects
in high-resolution images by slicing them into overlapping tiles and merging
the per-tile predictions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import yaml

logger = logging.getLogger(__name__)


def build_sahi_model(
    weights: str,
    device: str = "0",
    confidence_threshold: float = 0.45,
):
    """
    Build a SAHI-compatible detection model backed by Ultralytics RT-DETR.

    Parameters
    ----------
    weights               : Path to best.pt weights file
    device                : "0" for GPU 0, "cpu" for CPU
    confidence_threshold  : Minimum detection confidence
    """
    from sahi import AutoDetectionModel

    device_str = f"cuda:{device}" if device.isdigit() else device
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=weights,
        confidence_threshold=confidence_threshold,
        device=device_str,
    )
    logger.info("SAHI model loaded from %s on device=%s", weights, device_str)
    return detection_model


def predict_with_sahi(
    detection_model,
    image_path: str,
    slice_height: int = 320,
    slice_width: int = 320,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    postprocess_type: str = "NMW",
    postprocess_match_threshold: float = 0.5,
):
    """
    Run SAHI sliced inference on a single image.

    Returns a ``sahi.prediction.PredictionResult`` object.
    """
    from sahi.predict import get_sliced_prediction

    result = get_sliced_prediction(
        image=image_path,
        detection_model=detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        postprocess_type=postprocess_type,
        postprocess_match_threshold=postprocess_match_threshold,
        verbose=0,
    )
    return result


def predict_folder_with_sahi(
    folder: str,
    weights: str,
    output_dir: str = "runs/sahi_inference",
    config_path: str = "config/inference_config.yaml",
) -> None:
    """
    Run SAHI inference on all images inside *folder* and export annotated
    images + COCO JSON to *output_dir*.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load SAHI config
    sahi_cfg: dict = {}
    if Path(config_path).exists():
        with open(config_path) as f:
            sahi_cfg = yaml.safe_load(f).get("sahi", {})
        inf_cfg = yaml.safe_load(open(config_path)).get("inference", {})
    else:
        inf_cfg = {}

    detection_model = build_sahi_model(
        weights=weights,
        device=str(inf_cfg.get("device", "0")),
        confidence_threshold=float(inf_cfg.get("confidence_threshold", 0.45)),
    )

    from sahi.predict import predict

    predict(
        model_type="ultralytics",
        model_path=weights,
        model_confidence_threshold=float(inf_cfg.get("confidence_threshold", 0.45)),
        model_device=f"cuda:{inf_cfg.get('device', '0')}" if str(inf_cfg.get("device", "0")).isdigit() else "cpu",
        source=folder,
        slice_height=int(sahi_cfg.get("slice_height", 320)),
        slice_width=int(sahi_cfg.get("slice_width", 320)),
        overlap_height_ratio=float(sahi_cfg.get("overlap_height_ratio", 0.2)),
        overlap_width_ratio=float(sahi_cfg.get("overlap_width_ratio", 0.2)),
        export_pickle=False,
        export_crop=False,
        project=output_dir,
        name="sahi_result",
        verbose=1,
    )
    logger.info("SAHI folder inference complete. Output: %s/sahi_result", output_dir)


def predict_frame_with_sahi(
    detection_model,
    frame,  # np.ndarray BGR frame
    sahi_cfg: Optional[dict] = None,
):
    """
    Run SAHI sliced inference on a single video frame (numpy array).
    Returns list of (class_name, confidence, bbox_xyxy).
    """
    if sahi_cfg is None:
        sahi_cfg = {}

    from sahi.predict import get_sliced_prediction

    result = get_sliced_prediction(
        image=frame,
        detection_model=detection_model,
        slice_height=int(sahi_cfg.get("slice_height", 320)),
        slice_width=int(sahi_cfg.get("slice_width", 320)),
        overlap_height_ratio=float(sahi_cfg.get("overlap_height_ratio", 0.2)),
        overlap_width_ratio=float(sahi_cfg.get("overlap_width_ratio", 0.2)),
        postprocess_type=sahi_cfg.get("postprocess_type", "NMW"),
        postprocess_match_threshold=float(sahi_cfg.get("postprocess_match_threshold", 0.5)),
        verbose=0,
    )

    detections: List[tuple] = []
    for obj_pred in result.object_prediction_list:
        label = obj_pred.category.name
        score = float(obj_pred.score.value)
        bbox = obj_pred.bbox.to_xyxy()  # [x1, y1, x2, y2]
        detections.append((label, score, bbox))

    return detections
