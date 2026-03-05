"""
src/data/augmentation.py
-------------------------
Augmentation pipelines for the Early Fire Detection System using Albumentations.

Provides:
    get_train_transforms — Heavy augmentation for training.
    get_val_transforms   — Light resize + normalise for validation.
    apply_mixup          — MixUp augmentation for fire/smoke images.
"""

from typing import List, Optional, Tuple

import numpy as np


def get_train_transforms(img_size: int = 640):
    """Build an Albumentations augmentation pipeline for training.

    Includes:
    - Resize to (img_size × img_size)
    - Random horizontal flip
    - Random rotation (±15°)
    - Random brightness / contrast
    - HSV colour-space jitter (hue, saturation, value)
    - Gaussian blur (occasional)
    - ImageNet normalisation

    Args:
        img_size: Target image side length in pixels.

    Returns:
        An ``albumentations.Compose`` transform compatible with YOLO bboxes.
    """
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError as exc:
        raise ImportError(
            "albumentations is required. Install with: pip install albumentations"
        ) from exc

    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.2, p=0.7),
            A.HueSaturationValue(
                hue_shift_limit=int(0.015 * 180),
                sat_shift_limit=int(0.7 * 255),
                val_shift_limit=int(0.4 * 255),
                p=0.5,
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.2,
        ),
    )


def get_val_transforms(img_size: int = 640):
    """Build a minimal Albumentations pipeline for validation / inference.

    Only applies resize and ImageNet normalisation — no random operations.

    Args:
        img_size: Target image side length in pixels.

    Returns:
        An ``albumentations.Compose`` transform.
    """
    try:
        import albumentations as A
    except ImportError as exc:
        raise ImportError(
            "albumentations is required. Install with: pip install albumentations"
        ) from exc

    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.2,
        ),
    )


def apply_mixup(
    img1: np.ndarray,
    bboxes1: List[List[float]],
    img2: np.ndarray,
    bboxes2: List[List[float]],
    alpha: float = 0.5,
) -> Tuple[np.ndarray, List[List[float]]]:
    """Apply MixUp augmentation to two fire/smoke images.

    The output image is a weighted linear combination of the two inputs,
    and the bounding boxes from both images are concatenated.

    Args:
        img1: First image as a NumPy array (H, W, 3), uint8 or float32.
        bboxes1: Bounding boxes for img1 in YOLO format
                 [[class_id, cx, cy, w, h], ...].
        img2: Second image (must be the same shape as img1).
        bboxes2: Bounding boxes for img2 in YOLO format.
        alpha: MixUp blending factor λ ∈ (0, 1).
               The actual λ is sampled from Beta(alpha, alpha).

    Returns:
        Tuple of (mixed_image, combined_bboxes).
        combined_bboxes contains annotations from both inputs.

    Raises:
        ValueError: If img1 and img2 have different shapes.
    """
    if img1.shape != img2.shape:
        raise ValueError(
            f"Images must have the same shape for MixUp. "
            f"Got {img1.shape} and {img2.shape}."
        )

    lam = float(np.random.default_rng().beta(alpha, alpha))

    img1_f = img1.astype(np.float32)
    img2_f = img2.astype(np.float32)
    mixed = (lam * img1_f + (1.0 - lam) * img2_f).clip(0, 255).astype(np.uint8)

    combined_bboxes = list(bboxes1) + list(bboxes2)
    return mixed, combined_bboxes
