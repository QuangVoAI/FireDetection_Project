"""
src/data/preprocessing.py
--------------------------
Preprocessing utilities for the Early Fire Detection System.

Functions:
    resize_image        — Resize to square with letterboxing.
    normalize_image     — ImageNet mean/std normalisation.
    deduplicate_dataset — Remove duplicate images using perceptual hash.
    validate_annotations — Check YOLO annotation format validity.
"""

import logging
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ImageNet statistics
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resize_image(img: np.ndarray, size: int = 640) -> np.ndarray:
    """Resize an image to a square while maintaining aspect ratio with letterboxing.

    The image is scaled so that the longer side equals *size*, and the shorter
    side is padded with grey (value 114) to produce a (size × size) output.

    Args:
        img: Input image as a NumPy array of shape (H, W, 3), dtype uint8.
        size: Target side length in pixels (default: 640).

    Returns:
        Resized and padded image of shape (size, size, 3).
    """
    try:
        from PIL import Image as PILImage
    except ImportError as exc:
        raise ImportError("Pillow is required for resize_image.") from exc

    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))

    pil_img = PILImage.fromarray(img).resize((new_w, new_h), PILImage.BILINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)

    pad_top = (size - new_h) // 2
    pad_left = (size - new_w) // 2
    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = np.array(pil_img)
    return canvas


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Apply ImageNet mean/std normalisation to a float image.

    Args:
        img: Input image as a NumPy array of shape (H, W, 3).
             Values may be in [0, 255] (uint8) or [0.0, 1.0] (float32).

    Returns:
        Normalised float32 array of the same shape with values roughly in [-2, 2].
    """
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
    return img


def deduplicate_dataset(folder: str, hash_size: int = 8) -> List[str]:
    """Remove duplicate images from a dataset folder using perceptual hashing.

    Duplicate images (identical perceptual hash) are deleted from disk.
    Only the first occurrence is kept.

    Args:
        folder: Path to the ``images/`` subdirectory containing image files.
        hash_size: Size parameter for the perceptual hash (default: 8).

    Returns:
        List of removed file paths.

    Raises:
        ImportError: If the ``imagehash`` library is not installed.
    """
    try:
        import imagehash
        from PIL import Image as PILImage
    except ImportError as exc:
        raise ImportError(
            "imagehash and Pillow are required for deduplicate_dataset. "
            "Install with: pip install imagehash Pillow"
        ) from exc

    images_dir = Path(folder)
    seen_hashes: dict = {}
    removed: List[str] = []

    image_files = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    for img_path in image_files:
        try:
            with PILImage.open(img_path) as pil_img:
                h = imagehash.phash(pil_img, hash_size=hash_size)
        except Exception:
            logger.warning("Could not hash image: %s — skipping.", img_path)
            continue

        if h in seen_hashes:
            logger.info("Duplicate found: %s (matches %s) — removing.", img_path, seen_hashes[h])
            img_path.unlink(missing_ok=True)
            # Also remove corresponding label if it exists
            labels_dir = img_path.parent.parent / "labels"
            label_file = labels_dir / (img_path.stem + ".txt")
            if label_file.exists():
                label_file.unlink()
            removed.append(str(img_path))
        else:
            seen_hashes[h] = img_path

    logger.info("Deduplication complete. Removed %d duplicate(s).", len(removed))
    return removed


def validate_annotations(folder: str) -> Tuple[int, int, List[str]]:
    """Validate YOLO-format annotation files in a dataset folder.

    Checks that each label file:
    - Has exactly 5 space-separated values per line.
    - Class index is a non-negative integer.
    - Bounding-box coordinates (cx, cy, w, h) are in [0, 1].

    Args:
        folder: Path to the dataset root containing ``images/`` and ``labels/``
                subdirectories.

    Returns:
        Tuple of (valid_count, invalid_count, error_messages).
    """
    labels_dir = Path(folder) / "labels"
    if not labels_dir.exists():
        logger.warning("Labels directory not found: %s", labels_dir)
        return 0, 0, [f"Labels directory not found: {labels_dir}"]

    valid_count = 0
    invalid_count = 0
    errors: List[str] = []

    for label_file in sorted(labels_dir.glob("*.txt")):
        file_valid = True
        with open(label_file, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    errors.append(
                        f"{label_file.name}:{line_no} — expected 5 values, got {len(parts)}"
                    )
                    file_valid = False
                    continue
                try:
                    cls_id = int(parts[0])
                    coords = list(map(float, parts[1:]))
                except ValueError:
                    errors.append(
                        f"{label_file.name}:{line_no} — non-numeric value"
                    )
                    file_valid = False
                    continue

                if cls_id < 0:
                    errors.append(
                        f"{label_file.name}:{line_no} — negative class id {cls_id}"
                    )
                    file_valid = False

                for i, coord in enumerate(coords):
                    if not (0.0 <= coord <= 1.0):
                        errors.append(
                            f"{label_file.name}:{line_no} — coord[{i}]={coord:.4f} out of [0,1]"
                        )
                        file_valid = False

        if file_valid:
            valid_count += 1
        else:
            invalid_count += 1

    logger.info(
        "Annotation validation: %d valid, %d invalid files.",
        valid_count,
        invalid_count,
    )
    return valid_count, invalid_count, errors
