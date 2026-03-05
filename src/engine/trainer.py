"""
src/engine/trainer.py
---------------------
3-stage training engine for the Early Fire Detection System.

Stages:
    1. Baseline Training    — standard fire/smoke samples.
    2. Hard Negative Mining — add hard negative samples to reduce false positives.
    3. SAHI Fine-tuning     — small object detection with SAHI-augmented data.
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Optional

import yaml

logger = logging.getLogger(__name__)

_DATA_DIRS = {
    "baseline": [
        "data/01_Positive_Standard",
        "data/02_Alley_Context",
    ],
    "hard_negative": [
        "data/01_Positive_Standard",
        "data/02_Alley_Context",
        "data/03_Negative_Hard_Samples",
    ],
    "sahi": [
        "data/01_Positive_Standard",
        "data/02_Alley_Context",
        "data/03_Negative_Hard_Samples",
        "data/04_SAHI_Small_Objects",
        "data/05_Real_Situation",
    ],
}


class Trainer:
    """Orchestrates the 3-stage training pipeline for RT-DETR-L.

    Args:
        model: A :class:`~src.models.rtdetr_model.FireDetectionModel` instance.
        config: ConfigNode loaded from ``configs/default.yaml``.
    """

    def __init__(self, model, config) -> None:
        self.model = model
        self.config = config
        self.checkpoints_dir = Path("checkpoints")
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public stage methods
    # ------------------------------------------------------------------

    def run_baseline_training(self) -> None:
        """Stage 1: Train on standard positive fire/smoke samples.

        Uses ``01_Positive_Standard`` and ``02_Alley_Context``.
        """
        logger.info("=== Stage 1: Baseline Training ===")
        data_yaml = self._build_data_yaml(_DATA_DIRS["baseline"], stage="baseline")
        self.model.train(data_yaml_path=data_yaml, project_name="runs/baseline")
        logger.info("Stage 1 complete.")

    def run_hard_negative_mining(self) -> None:
        """Stage 2: Retrain with hard negative samples to reduce false positives.

        Adds ``03_Negative_Hard_Samples`` (LED lights, cooking steam, etc.)
        and fine-tunes from the Stage 1 best checkpoint.
        """
        logger.info("=== Stage 2: Hard Negative Mining ===")
        self._load_best_checkpoint("runs/baseline")
        data_yaml = self._build_data_yaml(_DATA_DIRS["hard_negative"], stage="hard_negative")
        self.model.train(data_yaml_path=data_yaml, project_name="runs/hard_negative")
        logger.info("Stage 2 complete.")

    def run_sahi_finetuning(self) -> None:
        """Stage 3: Fine-tune with SAHI data for small object detection.

        Uses all 5 data folders including ``04_SAHI_Small_Objects`` and
        ``05_Real_Situation``.
        """
        logger.info("=== Stage 3: SAHI Fine-tuning ===")
        self._load_best_checkpoint("runs/hard_negative")
        data_yaml = self._build_data_yaml(_DATA_DIRS["sahi"], stage="sahi")
        self.model.train(data_yaml_path=data_yaml, project_name="runs/sahi")
        logger.info("Stage 3 complete.")

    def save_checkpoint(self, epoch: int, metrics: Dict) -> Path:
        """Save a training checkpoint with epoch number and metrics.

        Args:
            epoch: Current epoch number.
            metrics: Dictionary of metric values (e.g. mAP50, loss).

        Returns:
            Path to the saved checkpoint file.
        """
        checkpoint_path = self.checkpoints_dir / f"checkpoint_epoch_{epoch:04d}.json"
        payload = {"epoch": epoch, "metrics": metrics}
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        logger.info("Checkpoint saved: %s", checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, path: str) -> Dict:
        """Load a training checkpoint.

        Args:
            path: Path to the JSON checkpoint file.

        Returns:
            Dictionary with ``epoch`` and ``metrics`` keys.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        with open(checkpoint_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        logger.info(
            "Checkpoint loaded: epoch=%d, metrics=%s",
            payload.get("epoch"),
            payload.get("metrics"),
        )
        return payload

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_data_yaml(self, data_dirs, stage: str) -> str:
        """Generate a temporary YOLO data.yaml for the given stage.

        Combines images from multiple data directories into train/val splits
        by symlinking or listing paths.

        Args:
            data_dirs: List of data directory paths.
            stage: Stage name used for the output filename.

        Returns:
            Path to the generated YAML file.
        """
        # Collect all image paths across the combined dirs
        image_paths = []
        for d in data_dirs:
            images_dir = Path(d) / "images"
            if not images_dir.exists():
                logger.warning("Images dir not found, skipping: %s", images_dir)
                continue
            image_paths.extend(
                str(p)
                for p in images_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            )

        if not image_paths:
            logger.warning("No images found for stage: %s", stage)

        # Write a simple Ultralytics-compatible data YAML
        # Using the first available data dir as root, or current dir
        root_dir = str(Path(data_dirs[0]).parent.resolve()) if data_dirs else "."

        yaml_content = {
            "path": root_dir,
            "train": [str(Path(d) / "images") for d in data_dirs],
            "val": [str(Path(d) / "images") for d in data_dirs[:2]],
            "nc": self.config.model.num_classes,
            "names": self.config.model.class_names,
        }

        yaml_path = self.logs_dir / f"data_{stage}.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

        logger.info("Data YAML written to: %s", yaml_path)
        return str(yaml_path)

    def _load_best_checkpoint(self, run_dir: str) -> None:
        """Load the best weights from a previous training run.

        Args:
            run_dir: Directory of the previous training run (e.g. ``runs/baseline``).
        """
        best_weights = sorted(Path(run_dir).rglob("best.pt"))
        if best_weights:
            weights_path = str(best_weights[-1])
            logger.info("Loading best weights from: %s", weights_path)
            try:
                from ultralytics import RTDETR
                self.model.model = RTDETR(weights_path)
            except Exception:
                logger.warning("Could not load weights from %s — continuing with current model.", weights_path)
        else:
            logger.warning("No best.pt found in %s — continuing with current model.", run_dir)
