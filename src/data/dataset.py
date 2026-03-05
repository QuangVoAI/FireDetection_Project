"""
src/data/dataset.py
-------------------
PyTorch Dataset class for loading fire and smoke detection data in YOLO format.

Supports:
- Multiple data directories (the 5 project folders)
- Train/val split
- Albumentations augmentations during training
"""

import logging
import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class FireSmokeDataset(Dataset):
    """PyTorch Dataset for fire and smoke detection using YOLO-format annotations.

    Loads images and bounding-box labels from any of the 5 project data folders:
        01_Positive_Standard, 02_Alley_Context, 03_Negative_Hard_Samples,
        04_SAHI_Small_Objects, 05_Real_Situation.

    Each data folder must contain:
        images/  — JPEG/PNG image files
        labels/  — corresponding .txt files in YOLO format
                   (class_id cx cy w h, normalised 0–1)

    Args:
        data_dirs: List of root data directories to load from.
        img_size: Target image size (square).
        split: ``"train"`` or ``"val"``.
        train_ratio: Fraction of samples used for training.
        transform: Optional Albumentations transform pipeline.
        seed: Random seed for reproducible splits.
    """

    CLASS_NAMES = ["Fire", "Smoke"]

    def __init__(
        self,
        data_dirs: List[str],
        img_size: int = 640,
        split: str = "train",
        train_ratio: float = 0.85,
        transform: Optional[Callable] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.split = split
        self.transform = transform

        self.image_paths: List[Path] = []
        self.label_paths: List[Optional[Path]] = []

        for data_dir in data_dirs:
            root = Path(data_dir)
            images_dir = root / "images"
            labels_dir = root / "labels"

            if not images_dir.exists():
                logger.warning("Images directory not found, skipping: %s", images_dir)
                continue

            for img_file in sorted(images_dir.iterdir()):
                if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                label_file = labels_dir / (img_file.stem + ".txt")
                self.image_paths.append(img_file)
                self.label_paths.append(label_file if label_file.exists() else None)

        if not self.image_paths:
            logger.warning("No images found in the provided data directories.")

        # Reproducible train/val split
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(self.image_paths))
        split_idx = int(len(indices) * train_ratio)

        if split == "train":
            selected = indices[:split_idx]
        else:
            selected = indices[split_idx:]

        self.image_paths = [self.image_paths[i] for i in selected]
        self.label_paths = [self.label_paths[i] for i in selected]

        logger.info(
            "FireSmokeDataset (%s): %d samples loaded from %d directories.",
            split,
            len(self.image_paths),
            len(data_dirs),
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (image_tensor, bboxes, labels) for a single sample.

        Returns:
            image_tensor: Float tensor of shape (3, H, W) normalised to [0, 1].
            bboxes: Float tensor of shape (N, 4) in YOLO format (cx, cy, w, h).
            labels: Long tensor of shape (N,) with class indices (0=Fire, 1=Smoke).
        """
        image = self._load_image(self.image_paths[idx])
        bboxes, labels = self._load_labels(self.label_paths[idx])

        if self.transform is not None:
            # Albumentations expects HWC uint8 image and pascal_voc / yolo bboxes
            transformed = self.transform(
                image=image,
                bboxes=bboxes.tolist(),
                class_labels=labels.tolist(),
            )
            image = transformed["image"]
            bboxes_list = transformed["bboxes"]
            labels_list = transformed["class_labels"]

            if bboxes_list:
                bboxes = torch.tensor(bboxes_list, dtype=torch.float32)
                labels = torch.tensor(labels_list, dtype=torch.long)
            else:
                bboxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros(0, dtype=torch.long)

        # Convert image (HWC uint8) to CHW float tensor
        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        else:
            image_tensor = image

        return image_tensor, bboxes, labels

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_image(self, path: Path) -> np.ndarray:
        """Load and resize an image to (img_size × img_size) using letterboxing."""
        img = Image.open(path).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        return np.array(img, dtype=np.uint8)

    def _load_labels(
        self, label_path: Optional[Path]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parse a YOLO-format label file.

        Returns:
            bboxes: Tensor (N, 4) — cx, cy, w, h (normalised).
            labels: Tensor (N,)   — class index.
        """
        if label_path is None or not label_path.exists():
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros(0, dtype=torch.long)

        bboxes_list: List[List[float]] = []
        labels_list: List[int] = []

        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
                bboxes_list.append([cx, cy, w, h])
                labels_list.append(cls_id)

        if not bboxes_list:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros(0, dtype=torch.long)

        return (
            torch.tensor(bboxes_list, dtype=torch.float32),
            torch.tensor(labels_list, dtype=torch.long),
        )


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """Custom collate function for variable-length bounding-box tensors.

    Args:
        batch: List of (image, bboxes, labels) tuples.

    Returns:
        images: Stacked image tensor (B, 3, H, W).
        bboxes: List of per-image bbox tensors.
        labels: List of per-image label tensors.
    """
    images, bboxes, labels = zip(*batch)
    return torch.stack(images, dim=0), list(bboxes), list(labels)
