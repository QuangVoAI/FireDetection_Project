"""
src/engine/evaluator.py
-----------------------
Evaluation utilities for the Early Fire Detection System.

Functions:
    compute_map            — mAP@50.
    compute_map_95         — mAP@50-95.
    plot_confusion_matrix  — Save confusion matrix figure.
    measure_fps            — Inference speed benchmark.
    generate_evaluation_report — Export metrics to CSV/JSON.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute IoU between two bounding boxes in [x1, y1, x2, y2] format."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def _average_precision(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float,
    class_id: int,
) -> float:
    """Compute Average Precision for a single class at a given IoU threshold.

    Args:
        predictions: List of dicts with keys ``class``, ``confidence``, ``bbox``.
        ground_truth: List of dicts with keys ``class``, ``bbox``.
        iou_threshold: IoU threshold for a true positive.
        class_id: Integer class index to evaluate.

    Returns:
        Average Precision value in [0, 1].
    """
    class_preds = [p for p in predictions if p.get("class_id", p.get("class")) == class_id]
    class_gts = [g for g in ground_truth if g.get("class_id", g.get("class")) == class_id]

    if not class_gts:
        return 0.0

    class_preds.sort(key=lambda x: x["confidence"], reverse=True)
    matched_gt = set()
    tp, fp = [], []

    for pred in class_preds:
        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt in enumerate(class_gts):
            if gt_idx in matched_gt:
                continue
            iou = _compute_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold:
            tp.append(1)
            fp.append(0)
            matched_gt.add(best_gt_idx)
        else:
            tp.append(0)
            fp.append(1)

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    n_gt = len(class_gts)

    precision = tp_cum / (tp_cum + fp_cum + 1e-9)
    recall = tp_cum / (n_gt + 1e-9)

    # Compute AP using 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = precision[recall >= t].max() if (recall >= t).any() else 0.0
        ap += p / 11.0

    return float(ap)


def compute_map(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.5,
    num_classes: int = 2,
) -> float:
    """Compute mAP@IoU_threshold averaged over all classes.

    Args:
        predictions: List of detection dicts with keys ``class_id``, ``confidence``,
                     ``bbox`` ([x1, y1, x2, y2]).
        ground_truth: List of ground-truth dicts with keys ``class_id``, ``bbox``.
        iou_threshold: IoU threshold (default 0.5 → mAP@50).
        num_classes: Total number of detection classes.

    Returns:
        mAP value in [0, 1].
    """
    aps = [
        _average_precision(predictions, ground_truth, iou_threshold, cls_id)
        for cls_id in range(num_classes)
    ]
    return float(np.mean(aps))


def compute_map_95(
    predictions: List[Dict],
    ground_truth: List[Dict],
    num_classes: int = 2,
) -> float:
    """Compute mAP@50-95 (averaged over IoU thresholds 0.50–0.95 in steps of 0.05).

    Args:
        predictions: List of detection dicts.
        ground_truth: List of ground-truth dicts.
        num_classes: Total number of detection classes.

    Returns:
        mAP@50-95 value in [0, 1].
    """
    thresholds = np.arange(0.50, 1.00, 0.05)
    maps = [
        compute_map(predictions, ground_truth, iou_threshold=t, num_classes=num_classes)
        for t in thresholds
    ]
    return float(np.mean(maps))


def plot_confusion_matrix(
    predictions: List[Dict],
    ground_truth: List[Dict],
    class_names: List[str],
    save_path: str,
    iou_threshold: float = 0.5,
) -> None:
    """Plot and save a confusion matrix for object detection results.

    Args:
        predictions: List of detection dicts with ``class_id``, ``confidence``, ``bbox``.
        ground_truth: List of ground-truth dicts with ``class_id``, ``bbox``.
        class_names: List of class name strings.
        save_path: File path to save the confusion matrix image (PNG).
        iou_threshold: IoU threshold used to match predictions to GT boxes.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:
        raise ImportError(
            "matplotlib and seaborn are required. "
            "Install with: pip install matplotlib seaborn"
        ) from exc

    n = len(class_names)
    matrix = np.zeros((n + 1, n + 1), dtype=int)  # +1 for background

    matched_gt = set()

    for pred in predictions:
        pred_cls = pred.get("class_id", 0)
        best_iou = 0.0
        best_gt_idx = -1
        best_gt_cls = -1

        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in matched_gt:
                continue
            iou = _compute_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
                best_gt_cls = gt.get("class_id", 0)

        if best_iou >= iou_threshold:
            matrix[best_gt_cls][pred_cls] += 1
            matched_gt.add(best_gt_idx)
        else:
            matrix[n][pred_cls] += 1  # False positive (background → pred_cls)

    for gt_idx, gt in enumerate(ground_truth):
        if gt_idx not in matched_gt:
            matrix[gt.get("class_id", 0)][n] += 1  # False negative

    labels = class_names + ["Background"]
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Oranges",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix — Fire Detection", fontsize=14)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved to: %s", save_path)


def measure_fps(
    model,
    test_images: List,
    device: str = "cpu",
    warmup_runs: int = 5,
) -> float:
    """Measure inference speed (frames per second) for a detection model.

    Args:
        model: An object with a callable ``predict(image)`` method.
        test_images: List of images (paths or NumPy arrays) to benchmark.
        device: Device identifier (ignored here; handled inside the model).
        warmup_runs: Number of warmup inference calls before timing.

    Returns:
        Average FPS over the test images.
    """
    if not test_images:
        logger.warning("No test images provided for FPS measurement.")
        return 0.0

    # Warmup
    for img in test_images[:warmup_runs]:
        model.predict(img)

    start = time.perf_counter()
    for img in test_images:
        model.predict(img)
    elapsed = time.perf_counter() - start

    fps = len(test_images) / (elapsed + 1e-9)
    logger.info("FPS benchmark: %.2f FPS over %d images.", fps, len(test_images))
    return fps


def generate_evaluation_report(
    metrics: Dict,
    save_path: str,
) -> None:
    """Export evaluation metrics to both JSON and CSV formats.

    Args:
        metrics: Dictionary of metric names to values.
                 E.g. ``{"mAP50": 0.87, "mAP50_95": 0.66, "fps": 28.4}``.
        save_path: Base file path (without extension) for the report files.
                   Two files will be created: ``<save_path>.json``
                   and ``<save_path>.csv``.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required. Install with: pip install pandas") from exc

    base = Path(save_path)
    base.parent.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = base.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Evaluation report (JSON) saved to: %s", json_path)

    # CSV
    csv_path = base.with_suffix(".csv")
    df = pd.DataFrame([metrics])
    df.to_csv(csv_path, index=False)
    logger.info("Evaluation report (CSV) saved to: %s", csv_path)
