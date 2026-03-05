"""
RT-DETR-L Training Module
==========================
Wraps the Ultralytics RTDETR trainer with two stages:
  Stage 1 – Baseline training on fire / smoke images.
  Stage 2 – Hard Negative Mining: fine-tune on the negative samples folder
             to reduce false positives (LED signs, steam, etc.).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_trainer(config_path: str = "config/model_config.yaml"):
    """
    Build and return a configured Ultralytics RTDETR model object.

    Returns
    -------
    ultralytics.RTDETR
    """
    from ultralytics import RTDETR  # imported here to keep module importable without GPU

    cfg = _load_config(config_path)
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    weights = model_cfg.get("weights", "")
    if weights and Path(weights).exists():
        logger.info("Loading weights from %s", weights)
        model = RTDETR(weights)
    else:
        logger.info("No pre-trained weights found – using COCO-pretrained rtdetr-l.pt")
        model = RTDETR("rtdetr-l.pt")

    return model, train_cfg, cfg


def train(
    config_path: str = "config/model_config.yaml",
    dataset_yaml: str = "config/dataset.yaml",
    stage: str = "baseline",
    resume: bool = False,
) -> None:
    """
    Train the RT-DETR-L model.

    Parameters
    ----------
    config_path  : Path to model_config.yaml
    dataset_yaml : Path to YOLO-format dataset descriptor
    stage        : 'baseline' | 'hard_negative_mining'
    resume       : Resume from last checkpoint
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    model, train_cfg, cfg = build_trainer(config_path)

    project = train_cfg.get("project", "runs/train")
    name = f"{train_cfg.get('name', 'rtdetr_fire')}_{stage}"

    common_kwargs = dict(
        data=dataset_yaml,
        epochs=train_cfg.get("epochs", 100),
        imgsz=train_cfg.get("image_size", 640),
        batch=train_cfg.get("batch_size", 16),
        device=str(train_cfg.get("device", "0")),
        workers=train_cfg.get("workers", 8),
        lr0=train_cfg.get("learning_rate", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
        patience=train_cfg.get("patience", 20),
        save_period=train_cfg.get("save_period", 10),
        project=project,
        name=name,
        resume=resume,
        # Augmentation
        mosaic=train_cfg.get("mosaic", 1.0),
        mixup=train_cfg.get("mixup", 0.15),
        degrees=train_cfg.get("degrees", 5.0),
        translate=train_cfg.get("translate", 0.1),
        scale=train_cfg.get("scale", 0.5),
        fliplr=train_cfg.get("fliplr", 0.5),
        hsv_h=train_cfg.get("hsv_h", 0.015),
        hsv_s=train_cfg.get("hsv_s", 0.7),
        hsv_v=train_cfg.get("hsv_v", 0.4),
    )

    if stage == "hard_negative_mining":
        # Reduce LR and epochs for fine-tuning stage
        common_kwargs["epochs"] = max(10, common_kwargs["epochs"] // 5)
        common_kwargs["lr0"] = common_kwargs["lr0"] * 0.1
        logger.info("Starting Hard Negative Mining stage (reduced LR + epochs).")

    logger.info("Starting training – stage: %s", stage)
    results = model.train(**common_kwargs)
    logger.info("Training complete. Results: %s", results)

    # Save best weights path for reference
    best_weights = Path(project) / name / "weights" / "best.pt"
    if best_weights.exists():
        dest = Path("models/weights/best.pt")
        dest.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(best_weights, dest)
        logger.info("Best weights copied to %s", dest)

    return results


def train_full_pipeline(
    config_path: str = "config/model_config.yaml",
    dataset_yaml: str = "config/dataset.yaml",
) -> None:
    """
    Run both training stages sequentially:
      1. Baseline training
      2. Hard Negative Mining fine-tuning
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger.info("=== Stage 1: Baseline Training ===")
    train(config_path, dataset_yaml, stage="baseline")

    logger.info("=== Stage 2: Hard Negative Mining ===")
    train(config_path, dataset_yaml, stage="hard_negative_mining")
    logger.info("Full training pipeline finished.")
