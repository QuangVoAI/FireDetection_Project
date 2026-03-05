"""
============================================================
📊 Evaluator Module — Đánh giá hiệu suất model
============================================================

MỤC ĐÍCH:
    Đánh giá model sau training bằng nhiều metrics:
    1. mAP@50, mAP@50-95 — metric chính cho object detection
    2. Precision, Recall, F1 — đánh giá chi tiết từng class
    3. Confusion Matrix — ma trận nhầm lẫn
    4. PR Curve — đường cong Precision-Recall
    5. FPS — tốc độ inference

GIẢI THÍCH CÁC METRIC CHO BẠN:

    📌 Precision (Độ chính xác):
       = True Positive / (True Positive + False Positive)
       "Trong số những cái model nói là lửa, bao nhiêu % thật sự là lửa?"
       Precision cao → ít false alarm (cảnh báo giả)

    📌 Recall (Độ phủ / Độ nhạy):
       = True Positive / (True Positive + False Negative)
       "Trong số những đám cháy thật, model phát hiện được bao nhiêu %?"
       Recall cao → ít bỏ sót

    📌 F1 Score:
       = 2 × (Precision × Recall) / (Precision + Recall)
       Là trung bình hài hòa của Precision và Recall.
       F1 cao → cả Precision lẫn Recall đều tốt.

    📌 mAP (Mean Average Precision):
       Metric chuẩn cho object detection.
       - mAP@50: detection đúng khi IoU ≥ 0.50 (dễ)
       - mAP@50-95: trung bình mAP ở nhiều IoU (0.50, 0.55, ..., 0.95)
         → Đánh giá khắt khe hơn

    📌 IoU (Intersection over Union):
       Đo "overlap" giữa predicted bbox và ground truth bbox.
       IoU = Diện tích giao / Diện tích hợp
       IoU = 1.0 → perfect match
       IoU ≈ 0.5 → overlap 50%
       IoU = 0.0 → không overlap

    📌 Confusion Matrix (Ma trận nhầm lẫn):
       Bảng cho thấy model nhầm class nào với class nào.
       Vd: Fire → Fire (đúng), Fire → Smoke (nhầm), Fire → Background (bỏ sót)
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Dùng backend không cần GUI
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import Config
from src.models.rtdetr_model import FireDetectionModel


class Evaluator:
    """
    Đánh giá model fire detection.

    CHỨC NĂNG:
        1. Chạy model trên validation set
        2. Tính các metrics
        3. Vẽ biểu đồ (confusion matrix, PR curve)
        4. Đo FPS
        5. Lưu báo cáo

    VÍ DỤ:
        >>> evaluator = Evaluator(model, config)
        >>> results = evaluator.evaluate('runs/prepared_data/data.yaml')
        >>> evaluator.plot_results(results)
    """

    def __init__(self, model: FireDetectionModel, config: Config):
        self.model = model
        self.config = config
        self.class_names = list(config.data.class_names)
        self.output_dir = Path(config.output.save_dir) / "evaluate"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, data_yaml: str) -> Dict:
        """
        Đánh giá model trên validation set.

        FLOW:
            1. Gọi model.val() → Ultralytics tự tính metrics
            2. Extract kết quả
            3. Format thành dict dễ đọc

        Args:
            data_yaml: Đường dẫn file data.yaml

        Returns:
            Dict chứa tất cả metrics
        """
        print(f"\n{'='*60}")
        print(f"📊 ĐÁNH GIÁ MODEL")
        print(f"{'='*60}")

        # Chạy validation
        results = self.model.model.val(
            data=data_yaml,
            imgsz=self.config.model.img_size,
            batch=self.config.training.batch_size,
            device=self.model.device,
            project=str(self.output_dir),
            name="val_results",
            exist_ok=True,
        )

        # Extract metrics
        metrics = {
            'mAP50': round(float(results.box.map50), 4),
            'mAP50_95': round(float(results.box.map), 4),
            'precision': round(float(results.box.mp), 4),
            'recall': round(float(results.box.mr), 4),
            'f1': round(
                2 * float(results.box.mp) * float(results.box.mr)
                / max(float(results.box.mp) + float(results.box.mr), 1e-6),
                4
            ),
        }

        # Per-class metrics
        metrics['per_class'] = {}
        for i, name in enumerate(self.class_names):
            if i < len(results.box.ap50):
                metrics['per_class'][name] = {
                    'AP50': round(float(results.box.ap50[i]), 4),
                    'AP50_95': round(float(results.box.ap[i]), 4),
                }

        # In kết quả
        print(f"\n📊 KẾT QUẢ ĐÁNH GIÁ:")
        print(f"   ┌──────────────────┬──────────┐")
        print(f"   │ Metric           │ Value    │")
        print(f"   ├──────────────────┼──────────┤")
        print(f"   │ mAP@50           │ {metrics['mAP50']:.4f}   │")
        print(f"   │ mAP@50-95        │ {metrics['mAP50_95']:.4f}   │")
        print(f"   │ Precision        │ {metrics['precision']:.4f}   │")
        print(f"   │ Recall           │ {metrics['recall']:.4f}   │")
        print(f"   │ F1               │ {metrics['f1']:.4f}   │")
        print(f"   └──────────────────┴──────────┘")

        for name, cls_metrics in metrics.get('per_class', {}).items():
            print(f"   {name}: AP50={cls_metrics['AP50']:.4f}, AP50-95={cls_metrics['AP50_95']:.4f}")

        return metrics

    def plot_confusion_matrix(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict],
        save_path: Optional[str] = None,
    ):
        """
        Vẽ confusion matrix.

        CONFUSION MATRIX GIẢI THÍCH:
            Mỗi ô (i, j) cho biết: số lần class thật là i, model dự đoán là j

            Đường chéo = CÁC Ô ĐÚNG (model đoán đúng)
            Ngoài đường chéo = SAI (model nhầm)

            VÍ DỤ:
                         Predicted
                      Fire  Smoke  Background
            Fire    [ 80    10     10  ]  ← 80/100 đúng
            Smoke   [  5    85     10  ]  ← 85/100 đúng
            BG      [ 15     5      -  ]  ← 20 false positive

        Args:
            predictions: List of predicted detections
            ground_truths: List of ground truth labels
            save_path: Đường dẫn lưu ảnh (None → tự tạo)
        """
        if save_path is None:
            save_path = str(self.output_dir / "confusion_matrix.png")

        num_classes = len(self.class_names) + 1  # +1 cho "Background"
        labels = self.class_names + ["Background"]

        # Tính confusion matrix
        matrix = np.zeros((num_classes, num_classes), dtype=int)

        # TODO: Implement matching logic giữa predictions và ground truths
        # Đây là placeholder — Ultralytics đã tự tính confusion matrix

        # Vẽ heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title('Confusion Matrix — Fire Detection', fontsize=14)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   📊 Confusion matrix saved: {save_path}")

    def plot_pr_curve(
        self,
        precisions: List[float],
        recalls: List[float],
        save_path: Optional[str] = None,
    ):
        """
        Vẽ đường cong Precision-Recall (PR Curve).

        PR CURVE GIẢI THÍCH:
            Trục X = Recall (càng phải càng phát hiện nhiều)
            Trục Y = Precision (càng cao càng ít false alarm)

            Model tốt → đường cong gần góc trên-phải (1.0, 1.0)
            Model dở → đường cong sát trục X

            Diện tích dưới PR curve = AP (Average Precision)

        Args:
            precisions: List precision values tại các thresholds
            recalls: List recall values tại các thresholds
            save_path: Đường dẫn lưu ảnh
        """
        if save_path is None:
            save_path = str(self.output_dir / "pr_curve.png")

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(recalls, precisions, 'b-', linewidth=2, label='PR Curve')
        ax.fill_between(recalls, precisions, alpha=0.2, color='blue')

        # Annotations
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve', fontsize=14)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   📈 PR curve saved: {save_path}")

    def save_report(self, metrics: Dict, filename: str = "evaluation_report.json"):
        """
        Lưu báo cáo đánh giá dạng JSON.

        Args:
            metrics: Dict chứa metrics
            filename: Tên file output
        """
        report_path = self.output_dir / filename
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"   📄 Report saved: {report_path}")

    def compare_stages(self, stage_results: Dict[str, Dict]):
        """
        So sánh kết quả giữa 3 stages.

        DÙNG ĐỂ:
            Kiểm tra xem mỗi stage có cải thiện so với stage trước không.
            Vd: Stage 2 phải giảm false positive so với Stage 1.

        Args:
            stage_results: Dict mapping stage_name → metrics dict
        """
        print(f"\n{'='*60}")
        print(f"📊 SO SÁNH KẾT QUẢ GIỮA CÁC STAGES")
        print(f"{'='*60}")

        headers = ['Stage', 'mAP50', 'mAP50-95', 'Precision', 'Recall', 'F1']
        print(f"\n{'│'.join(f'{h:>12}' for h in headers)}")
        print(f"{'─'*75}")

        for stage_name, metrics in stage_results.items():
            values = [
                stage_name[:12],
                f"{metrics.get('mAP50', 0):.4f}",
                f"{metrics.get('mAP50_95', 0):.4f}",
                f"{metrics.get('precision', 0):.4f}",
                f"{metrics.get('recall', 0):.4f}",
                f"{metrics.get('f1', 0):.4f}",
            ]
            print(f"{'│'.join(f'{v:>12}' for v in values)}")

        # Vẽ biểu đồ so sánh
        self._plot_stage_comparison(stage_results)

    def _plot_stage_comparison(self, stage_results: Dict[str, Dict]):
        """Vẽ bar chart so sánh metrics giữa các stages."""
        metrics_to_plot = ['mAP50', 'mAP50_95', 'precision', 'recall', 'f1']
        stage_names = list(stage_results.keys())

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(metrics_to_plot))
        width = 0.25

        for i, stage in enumerate(stage_names):
            values = [stage_results[stage].get(m, 0) for m in metrics_to_plot]
            ax.bar(x + i * width, values, width, label=stage, alpha=0.8)

        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Stage Comparison — Fire Detection Model', fontsize=14)
        ax.set_xticks(x + width)
        ax.set_xticklabels(metrics_to_plot, fontsize=10)
        ax.legend(fontsize=11)
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        save_path = self.output_dir / "stage_comparison.png"
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   📊 Stage comparison chart saved: {save_path}")
