"""
Evaluation Module
=================
Evaluates a trained RT-DETR model and produces:
  - mAP@50 and mAP@50-95
  - Per-class Precision / Recall / F1
  - Confusion Matrix (saved as PNG)
  - Inference speed (ms/image, FPS)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def evaluate(
    weights: str = "models/weights/best.pt",
    dataset_yaml: str = "config/dataset.yaml",
    config_path: str = "config/model_config.yaml",
    split: str = "test",
    output_dir: str = "runs/evaluate",
    save_plots: bool = True,
) -> dict:
    """
    Run validation on a trained RT-DETR-L model and return metrics.

    Parameters
    ----------
    weights      : Path to trained weights (.pt).
    dataset_yaml : YOLO-format dataset YAML.
    config_path  : Model config YAML.
    split        : Dataset split to evaluate ('val' | 'test').
    output_dir   : Directory to save evaluation artefacts.
    save_plots   : If True, saves confusion matrix + PR curve plots.

    Returns
    -------
    Dictionary with keys: map50, map50_95, precision, recall, fitness, fps.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from ultralytics import RTDETR

    # Load config
    train_cfg: dict = {}
    inf_cfg: dict = {}
    if Path(config_path).exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        train_cfg = cfg.get("training", {})
    if Path("config/inference_config.yaml").exists():
        with open("config/inference_config.yaml") as f:
            inf_cfg = yaml.safe_load(f).get("inference", {})

    if not Path(weights).exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    model = RTDETR(weights)

    results = model.val(
        data=dataset_yaml,
        split=split,
        imgsz=int(train_cfg.get("image_size", 640)),
        batch=int(train_cfg.get("batch_size", 16)),
        device=str(train_cfg.get("device", "0")),
        conf=float(inf_cfg.get("confidence_threshold", 0.001)),
        iou=float(inf_cfg.get("iou_threshold", 0.45)),
        project=output_dir,
        name="eval_result",
        save_json=True,
        plots=save_plots,
    )

    # Extract metrics
    metrics = {
        "map50": float(results.box.map50),
        "map50_95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "fitness": float(results.fitness),
    }

    # Inference speed
    speed = results.speed  # dict: preprocess, inference, postprocess (ms)
    if speed:
        total_ms = sum(speed.values())
        metrics["fps"] = round(1000.0 / total_ms, 1) if total_ms > 0 else 0.0
        metrics["speed_ms"] = speed

    logger.info(
        "Evaluation results | mAP@50=%.3f | mAP@50-95=%.3f | FPS=%.1f",
        metrics["map50"],
        metrics["map50_95"],
        metrics.get("fps", 0.0),
    )

    return metrics
