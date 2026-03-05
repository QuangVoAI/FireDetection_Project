"""
src/utils/visualization.py
---------------------------
Visualization tools for the Early Fire Detection System.

Functions:
    draw_detections        — Draw bounding boxes with confidence scores.
    plot_training_curves   — Plot loss and mAP curves from training log.
    visualize_sahi_slices  — Visualise SAHI slice grid on an image.
    create_comparison_grid — Side-by-side prediction comparison grid.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Colour palette for classes (BGR for OpenCV)
_CLASS_COLORS = [
    (0, 60, 255),    # Fire  — orange-red
    (180, 180, 180), # Smoke — light grey
]
_DEFAULT_COLOR = (0, 255, 0)


def draw_detections(
    image: np.ndarray,
    boxes: List[List[float]],
    scores: List[float],
    labels: List[int],
    class_names: List[str],
) -> np.ndarray:
    """Draw bounding boxes and confidence scores onto an image.

    Args:
        image: Input image as a NumPy array (H, W, 3), uint8. Will be copied.
        boxes: List of bounding boxes in [x1, y1, x2, y2] pixel coordinates.
        scores: List of confidence scores (one per box).
        labels: List of class indices (one per box).
        class_names: Class name strings indexed by label id.

    Returns:
        Annotated image copy as a NumPy uint8 array.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required for draw_detections.") from exc

    output = image.copy()

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [int(v) for v in box]
        color = _CLASS_COLORS[label] if label < len(_CLASS_COLORS) else _DEFAULT_COLOR
        name = class_names[label] if label < len(class_names) else str(label)

        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness=2)

        text = f"{name} {score:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(output, (x1, y1 - text_h - baseline - 4), (x1 + text_w, y1), color, -1)
        cv2.putText(
            output,
            text,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return output


def plot_training_curves(log_path: str, save_path: str) -> None:
    """Plot training loss and mAP curves from a CSV results log.

    Expects Ultralytics-style results CSV with columns like
    ``train/box_loss``, ``metrics/mAP50``, ``metrics/mAP50-95``.

    Args:
        log_path: Path to the results CSV file (Ultralytics ``results.csv``).
        save_path: File path to save the figure (PNG).
    """
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "matplotlib and pandas are required for plot_training_curves."
        ) from exc

    df = pd.read_csv(log_path)
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("RT-DETR-L Training Curves", fontsize=14)

    # Loss curves
    loss_cols = [c for c in df.columns if "loss" in c.lower()]
    for col in loss_cols:
        axes[0].plot(df.index, df[col], label=col)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss value")
    axes[0].legend(fontsize=8)
    axes[0].grid(True)

    # mAP curves
    map_cols = [c for c in df.columns if "map" in c.lower() or "mAP" in c]
    for col in map_cols:
        axes[1].plot(df.index, df[col], label=col)
    axes[1].set_title("mAP")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mAP")
    axes[1].legend(fontsize=8)
    axes[1].grid(True)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Training curves saved to: %s", save_path)


def visualize_sahi_slices(
    image: np.ndarray,
    slice_coords: List[Tuple[int, int, int, int]],
    save_path: str,
) -> None:
    """Draw SAHI slice boundaries on an image and save the result.

    Args:
        image: Input image as a NumPy array (H, W, 3), uint8.
        slice_coords: List of (x1, y1, x2, y2) pixel coordinates for each slice.
        save_path: File path to save the annotated image (PNG).
    """
    try:
        import cv2
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("opencv-python and matplotlib are required.") from exc

    output = image.copy()
    for i, (x1, y1, x2, y2) in enumerate(slice_coords):
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(
            output,
            f"S{i}",
            (x1 + 4, y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 255),
            1,
            cv2.LINE_AA,
        )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    # Convert RGB for matplotlib display
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB) if output.shape[2] == 3 else output)
    plt.title(f"SAHI Slices ({len(slice_coords)} patches)", fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("SAHI slice visualisation saved to: %s", save_path)


def create_comparison_grid(
    images: List[np.ndarray],
    predictions: List[List[Dict]],
    save_path: str,
    class_names: Optional[List[str]] = None,
) -> None:
    """Create a side-by-side grid of images with prediction overlays.

    Each image is shown with its bounding-box predictions. The grid is
    automatically arranged to be as square as possible.

    Args:
        images: List of input images (NumPy arrays, uint8).
        predictions: List of prediction lists, one per image.
                     Each list contains dicts with ``class``, ``confidence``,
                     ``bbox``.
        save_path: File path to save the comparison grid (PNG).
        class_names: Class name strings. Defaults to ``["Fire", "Smoke"]``.
    """
    try:
        import cv2
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("opencv-python and matplotlib are required.") from exc

    if class_names is None:
        class_names = ["Fire", "Smoke"]

    n = len(images)
    if n == 0:
        logger.warning("No images provided for comparison grid.")
        return

    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for ax_idx, (img, preds) in enumerate(zip(images, predictions)):
        annotated = img.copy()
        for det in preds:
            cls_name = det.get("class", "")
            conf = det.get("confidence", 0.0)
            bbox = det.get("bbox", [0, 0, 0, 0])
            cls_id = class_names.index(cls_name) if cls_name in class_names else 0
            annotated = draw_detections(
                annotated,
                [bbox],
                [conf],
                [cls_id],
                class_names,
            )

        # Convert BGR → RGB for matplotlib
        if annotated.ndim == 3 and annotated.shape[2] == 3:
            display = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        else:
            display = annotated

        axes[ax_idx].imshow(display)
        axes[ax_idx].axis("off")
        axes[ax_idx].set_title(f"Image {ax_idx + 1}", fontsize=9)

    # Hide unused axes
    for ax in axes[n:]:
        ax.axis("off")

    plt.suptitle("Detection Comparison Grid", fontsize=13)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Comparison grid saved to: %s", save_path)
