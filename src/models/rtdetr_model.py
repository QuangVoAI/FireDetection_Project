"""
src/models/rtdetr_model.py
--------------------------
RT-DETR-L model wrapper using Ultralytics for fire and smoke detection.

Classes:
    FireDetectionModel — wraps ultralytics.RTDETR with project-specific helpers.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class FireDetectionModel:
    """Wrapper around Ultralytics RT-DETR-L for fire and smoke detection.

    Provides a high-level interface for loading pretrained weights, running
    single-image inference, SAHI-sliced inference, and orchestrating
    the 3-stage training pipeline.

    Args:
        config: ConfigNode loaded from ``configs/default.yaml``.
    """

    def __init__(self, config) -> None:
        self.config = config
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the RT-DETR-L model with pretrained weights."""
        try:
            from ultralytics import RTDETR
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required. Install with: pip install ultralytics"
            ) from exc

        model_name = "rtdetr-l.pt" if self.config.model.pretrained else "rtdetr-l.yaml"
        logger.info("Loading RT-DETR-L model: %s", model_name)
        self.model = RTDETR(model_name)
        logger.info("Model loaded successfully.")

    def train(self, data_yaml_path: str, project_name: str = "runs/train") -> None:
        """Run a single training pass using Ultralytics trainer.

        Args:
            data_yaml_path: Path to a YOLO-format ``data.yaml`` file describing
                            the dataset split.
            project_name: Output directory for training artefacts.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call _load_model() first.")

        cfg = self.config
        logger.info("Starting training with data: %s", data_yaml_path)

        self.model.train(
            data=data_yaml_path,
            epochs=cfg.training.epochs,
            batch=cfg.training.batch_size,
            imgsz=cfg.model.img_size,
            lr0=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            warmup_epochs=cfg.training.warmup_epochs,
            label_smoothing=cfg.training.label_smoothing,
            project=project_name,
            name="fire_detection",
        )
        logger.info("Training complete. Artefacts saved to: %s", project_name)

    def predict(
        self,
        image: Union[str, np.ndarray],
        confidence: float = 0.35,
    ) -> List[Dict]:
        """Run inference on a single image.

        Args:
            image: Path to an image file or a NumPy array (H, W, 3).
            confidence: Minimum detection confidence threshold.

        Returns:
            List of detection dicts:
            ``[{"class": "Fire", "confidence": 0.92, "bbox": [x1, y1, x2, y2]}, ...]``
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        results = self.model.predict(
            source=image,
            conf=confidence,
            iou=self.config.inference.iou_threshold,
            max_det=self.config.inference.max_detections,
            verbose=False,
        )

        detections: List[Dict] = []
        class_names = self.config.model.class_names

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                xyxy = box.xyxy[0].tolist()
                detections.append(
                    {
                        "class": class_names[cls_id] if cls_id < len(class_names) else str(cls_id),
                        "confidence": round(conf, 4),
                        "bbox": [round(v, 2) for v in xyxy],
                    }
                )

        return detections

    def predict_with_sahi(
        self,
        image: Union[str, np.ndarray],
        config=None,
    ) -> List[Dict]:
        """Run inference using SAHI slicing for small object detection.

        Useful for detecting small/distant fire objects in high-resolution
        images captured from balcony cameras (folder ``04_SAHI_Small_Objects``).

        Args:
            image: Path to an image file or a NumPy array.
            config: ConfigNode. Falls back to ``self.config`` if not provided.

        Returns:
            List of detection dicts (same format as :meth:`predict`).
        """
        try:
            from sahi import AutoDetectionModel
            from sahi.predict import get_sliced_prediction
        except ImportError as exc:
            raise ImportError(
                "sahi is required for SAHI inference. Install with: pip install sahi"
            ) from exc

        cfg = config or self.config
        sahi_cfg = cfg.sahi

        # Resolve image path
        if isinstance(image, np.ndarray):
            import tempfile

            import cv2

            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            cv2.imwrite(tmp.name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            image_path = tmp.name
        else:
            image_path = str(image)

        logger.info("Running SAHI inference on: %s", image_path)

        # Build a SAHI-compatible detection model wrapper
        sahi_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=self._get_best_weights(),
            confidence_threshold=cfg.inference.confidence_threshold,
            device="cuda:0" if self._cuda_available() else "cpu",
        )

        result = get_sliced_prediction(
            image_path,
            sahi_model,
            slice_height=sahi_cfg.slice_height,
            slice_width=sahi_cfg.slice_width,
            overlap_height_ratio=sahi_cfg.overlap_height_ratio,
            overlap_width_ratio=sahi_cfg.overlap_width_ratio,
            postprocess_type=sahi_cfg.postprocess_type,
            postprocess_match_threshold=sahi_cfg.postprocess_match_threshold,
            verbose=False,
        )

        class_names = cfg.model.class_names
        detections: List[Dict] = []
        for obj in result.object_prediction_list:
            cls_id = obj.category.id
            detections.append(
                {
                    "class": class_names[cls_id] if cls_id < len(class_names) else obj.category.name,
                    "confidence": round(obj.score.value, 4),
                    "bbox": [
                        round(obj.bbox.minx, 2),
                        round(obj.bbox.miny, 2),
                        round(obj.bbox.maxx, 2),
                        round(obj.bbox.maxy, 2),
                    ],
                }
            )

        return detections

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _get_best_weights(self) -> str:
        """Return the path to the best model weights file."""
        candidates = sorted(Path("runs").rglob("best.pt"))
        if candidates:
            return str(candidates[-1])
        return "rtdetr-l.pt"

    @staticmethod
    def _cuda_available() -> bool:
        """Return True if a CUDA device is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False


def load_model(config) -> FireDetectionModel:
    """Convenience factory function to load a FireDetectionModel.

    Args:
        config: ConfigNode loaded from ``configs/default.yaml``.

    Returns:
        Initialised :class:`FireDetectionModel` instance.
    """
    return FireDetectionModel(config)
