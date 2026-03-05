"""
Data Preprocessing Pipeline
============================
Handles:
  - Image resizing to target size (default 640×640)
  - Near-duplicate removal using perceptual hashing (imagehash)
  - Albumentations-based augmentation (Mosaic/Mixup handled at training time
    via Ultralytics; here we apply offline light augmentation for small sets)
  - Train / Val / Test split with YOLO-format label copying
"""

from __future__ import annotations

import logging
import os
import random
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import imagehash
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── Source data folders (relative to project root) ──────────────────────────
DATA_FOLDERS = [
    "01_Positive_Standard",
    "02_Alley_Context",
    "03_Negative_Hard_Samples",
    "04_SAHI_Small_Objects",
    "05_Real_Situation",
]

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── Helpers ──────────────────────────────────────────────────────────────────

def collect_image_paths(data_root: Path) -> List[Path]:
    """Collect all image paths from the 5 source sub-folders."""
    paths: List[Path] = []
    for folder in DATA_FOLDERS:
        folder_path = data_root / folder / "images"
        if not folder_path.exists():
            logger.warning("Folder not found, skipping: %s", folder_path)
            continue
        for p in folder_path.iterdir():
            if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                paths.append(p)
    logger.info("Collected %d images from source folders.", len(paths))
    return paths


def resize_image(img: np.ndarray, size: int = 640) -> np.ndarray:
    """Resize image to size×size using letterbox padding (preserve aspect ratio)."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_top = (size - new_h) // 2
    pad_left = (size - new_w) // 2
    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
    return canvas


def compute_hash(image_path: Path) -> imagehash.ImageHash:
    """Compute perceptual hash of an image for deduplication."""
    with Image.open(image_path) as img:
        return imagehash.phash(img)


def deduplicate(
    paths: List[Path], hash_threshold: int = 8
) -> List[Path]:
    """
    Remove near-duplicate images.

    Two images are considered duplicates when their perceptual hash
    difference is ≤ hash_threshold (0 = identical, higher = more lenient).
    Returns deduplicated list of paths.
    """
    seen_hashes: List[Tuple[imagehash.ImageHash, Path]] = []
    unique: List[Path] = []

    for p in tqdm(paths, desc="Deduplicating"):
        try:
            h = compute_hash(p)
        except Exception as exc:
            logger.warning("Could not hash %s: %s – skipping.", p, exc)
            continue

        is_dup = any(abs(h - existing_h) <= hash_threshold for existing_h, _ in seen_hashes)
        if not is_dup:
            seen_hashes.append((h, p))
            unique.append(p)

    removed = len(paths) - len(unique)
    logger.info("Deduplication: removed %d duplicates, %d images remain.", removed, len(unique))
    return unique


def _label_path_for(image_path: Path) -> Optional[Path]:
    """
    Infer the YOLO label (.txt) path that corresponds to an image.
    Checks a 'labels' sibling of the 'images' directory.
    """
    label_dir = image_path.parent.parent / "labels"
    label_file = label_dir / (image_path.stem + ".txt")
    return label_file if label_file.exists() else None


def split_dataset(
    paths: List[Path],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Split paths into train / val / test lists."""
    random.seed(seed)
    shuffled = paths.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    logger.info("Split → train=%d  val=%d  test=%d", len(train), len(val), len(test))
    return train, val, test


def copy_split(
    split_paths: List[Path],
    dest_img_dir: Path,
    dest_lbl_dir: Path,
    target_size: int = 640,
) -> None:
    """
    Resize images and copy them (+ labels) to destination directories.
    """
    dest_img_dir.mkdir(parents=True, exist_ok=True)
    dest_lbl_dir.mkdir(parents=True, exist_ok=True)

    for src in tqdm(split_paths, desc=f"Copying to {dest_img_dir.parent.name}"):
        img = cv2.imread(str(src))
        if img is None:
            logger.warning("Cannot read image: %s – skipping.", src)
            continue

        img_resized = resize_image(img, target_size)
        dest_img = dest_img_dir / src.name
        cv2.imwrite(str(dest_img), img_resized)

        label = _label_path_for(src)
        if label:
            shutil.copy2(label, dest_lbl_dir / label.name)

    logger.info("Copied %d images to %s.", len(split_paths), dest_img_dir)


# ── Main pipeline ────────────────────────────────────────────────────────────

def run_preprocessing(
    data_root: str = "data",
    config_path: str = "config/model_config.yaml",
    target_size: int = 640,
    hash_threshold: int = 8,
    seed: int = 42,
) -> None:
    """
    Full preprocessing pipeline:
      1. Collect images from all 5 raw folders
      2. Deduplicate
      3. Resize + split into train/val/test
      4. Write processed split to data/processed/
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    root = Path(data_root)
    processed_root = root / "processed"

    # Load split ratios from config
    train_ratio, val_ratio = 0.80, 0.10
    if Path(config_path).exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        ds = cfg.get("dataset", {})
        train_ratio = float(ds.get("train_ratio", 0.80))
        val_ratio = float(ds.get("val_ratio", 0.10))

    # 1. Collect
    paths = collect_image_paths(root)
    if not paths:
        logger.warning("No images found in source folders. Exiting preprocessing.")
        return

    # 2. Deduplicate
    paths = deduplicate(paths, hash_threshold=hash_threshold)

    # 3. Split
    train, val, test = split_dataset(paths, train_ratio, val_ratio, seed)

    # 4. Copy to processed splits
    for split_name, split_paths in [("train", train), ("val", val), ("test", test)]:
        copy_split(
            split_paths,
            dest_img_dir=processed_root / split_name / "images",
            dest_lbl_dir=processed_root / split_name / "labels",
            target_size=target_size,
        )

    logger.info("Preprocessing complete. Output in: %s", processed_root)
