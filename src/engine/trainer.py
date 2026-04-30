"""
============================================================
🏋️ Trainer Module — 3-Stage Training Pipeline
============================================================

MỤC ĐÍCH:
    Quản lý pipeline training 3 giai đoạn:
    1. Baseline: Học nhận biết lửa/khói cơ bản
    2. Hard Negative Mining: Giảm false positive
    3. SAHI Fine-tuning: Cải thiện phát hiện vật thể nhỏ

GIẢI THÍCH 3-STAGE TRAINING CHO BẠN:

    Tại sao cần 3 stages? Tại sao không train 1 lần?

    Tưởng tượng bạn học tiếng Anh:
    - Stage 1 (Baseline): Học từ vựng cơ bản, ngữ pháp đơn giản
      → Tương tự: model học nhận biết lửa/khói rõ ràng
    - Stage 2 (Hard Negative): Học phân biệt từ dễ nhầm (affect/effect)
      → Tương tự: model học phân biệt lửa vs đèn đỏ, khói vs hơi nước
    - Stage 3 (SAHI): Luyện nghe giọng khó (slang, accent)
      → Tương tự: model học phát hiện lửa rất nhỏ, rất xa

    Nếu trộn lẫn 3 giai đoạn → model bị "overwhelmed", không học tốt.
    Chia 3 stages → progressive learning, mỗi stage tập trung 1 vấn đề.

TRANSFER LEARNING:
    Mỗi stage kế thừa weights từ stage trước:
    Stage 1 weights → Stage 2 → Stage 3

    Giống bạn học: kiến thức cũ là nền tảng cho kiến thức mới.
"""

import os
import argparse
from pathlib import Path
from typing import Optional

from src.config import Config, load_config
from src.data.dataset import FireSmokeDataset
from src.models.rtdetr_model import FireDetectionModel


class Trainer:
    """
    3-Stage Training Engine cho Fire Detection.

    FLOW TỔNG QUAN:
        1. run_baseline_training():
           - Data: 01_Positive_Standard + 02_Alley_Context
           - Mục tiêu: model nhận biết lửa/khói cơ bản
           - Output: best.pt (checkpoint tốt nhất)

        2. run_hard_negative_mining():
           - Data: thêm 03_Negative_Hard_Samples
           - Load weights từ Stage 1
           - Mục tiêu: giảm false positive (đèn đỏ, hơi phở)
           - Output: best.pt mới

        3. run_sahi_finetuning():
           - Data: 04_SAHI_Small_Objects + 05_Ambient_Context_Null
           - Load weights từ Stage 2
           - Mục tiêu: phát hiện lửa/khói nhỏ/xa
           - Output: best.pt cuối cùng (deploy)

    VÍ DỤ SỬ DỤNG:
        >>> config = load_config('configs/default.yaml')
        >>> model = FireDetectionModel(config)
        >>> trainer = Trainer(model, config)
        >>> trainer.run_baseline_training()
        >>> trainer.run_hard_negative_mining()
        >>> trainer.run_sahi_finetuning()
    """

    def __init__(self, model: FireDetectionModel, config: Config):
        """
        Khởi tạo Trainer.

        Args:
            model: FireDetectionModel instance
            config: Config object
        """
        self.model = model
        self.config = config
        # Thư mục đầu ra phân tách theo model variant (vd: runs/baseline, runs/sota)
        variant_name = config.model.get('active_variant', 'default')
        self.output_dir = Path(config.output.save_dir) / variant_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Lưu đường dẫn weights tốt nhất sau mỗi stage
        self.best_weights = {
            'baseline': None,
            'hard_negative': None,
            'sahi': None,
        }

    def run_baseline_training(self) -> str:
        """
        🥇 Stage 1: Baseline Training
        ─────────────────────────────

        MỤC TIÊU:
            Model học nhận biết lửa và khói cơ bản từ ảnh rõ ràng.

        DATA:
            - 01_Positive_Standard: Ảnh lửa/khói từ dataset mở (COCO, Kaggle)
            - 02_Alley_Context: Ảnh hẻm TPHCM (bối cảnh thực tế)

        THAM SỐ:
            - Learning rate: 1e-4 (cao, vì train từ pretrained COCO)
            - Epochs: 80 (đủ để converge)
            - Pretrained: COCO weights (transfer learning)

        Returns:
            Đường dẫn tới best weights file (.pt)
        """
        stage_config = self.config.training.stages.baseline

        print(f"\n{'='*60}")
        print(f"🥇 STAGE 1: BASELINE TRAINING")
        print(f"{'='*60}")
        print(f"   Mục tiêu: Học nhận biết lửa/khói cơ bản")
        print(f"   Data folders: {list(stage_config.data_folders)}")
        print(f"   Epochs: {stage_config.epochs}")
        print(f"   Learning rate: {stage_config.learning_rate}")

        # Chuẩn bị dataset
        dataset = FireSmokeDataset(self.config, list(stage_config.data_folders))
        data_yaml = dataset.prepare()

        # In thống kê dataset
        stats = dataset.get_stats()
        print(f"\n   📊 Dataset stats:")
        for folder, info in stats['folder_stats'].items():
            print(f"      {folder}: {info['images']} ảnh, {info['objects']} objects")
        print(f"      Class counts: {stats['class_counts']}")

        # Training
        self.model.train(
            data_yaml=data_yaml,
            epochs=stage_config.epochs,
            learning_rate=stage_config.learning_rate,
            project=str(self.output_dir / "train"),
            name="stage1_baseline",
        )

        # Lưu đường dẫn weights
        weights_path = str(
            self.output_dir / "train" / "stage1_baseline" / "weights" / "best.pt"
        )
        self.best_weights['baseline'] = weights_path

        print(f"\n   ✅ Stage 1 hoàn thành!")
        print(f"   📦 Best weights: {weights_path}")

        return weights_path

    def run_hard_negative_mining(self) -> str:
        """
        🥈 Stage 2: Hard Negative Mining
        ──────────────────────────────────

        MỤC TIÊU:
            Giảm false positive — model phải KHÔNG phát hiện lửa/khói ở:
            - Hơi phở, hơi nước nóng (giống khói)
            - Đèn đuôi xe máy ban đêm (giống lửa)
            - Bảng đèn LED đỏ (giống lửa)
            - Quần áo đỏ, bạt xe máy đỏ (giống lửa)

        HARD NEGATIVE LÀ GÌ?
            "Hard negative" = mẫu âm tính khó phân biệt.
            Ví dụ: model nhìn hơi phở bốc lên → tưởng là khói.
            Ta cho model xem nhiều ảnh hơi phở (với label RỖNG = không có lửa)
            → model học rằng: "hơi phở ≠ khói"

        DATA:
            - Giữ nguyên data Stage 1 (01 + 02)
            - Thêm 03_Negative_Hard_Samples (hard negatives)

        THAM SỐ:
            - Learning rate: 5e-5 (nhỏ hơn Stage 1, vì fine-tuning)
            - Epochs: 50 (ít hơn vì chỉ cần tinh chỉnh)
            - Weights: load từ Stage 1 best.pt

        Returns:
            Đường dẫn tới best weights file
        """
        stage_config = self.config.training.stages.hard_negative

        print(f"\n{'='*60}")
        print(f"🥈 STAGE 2: HARD NEGATIVE MINING")
        print(f"{'='*60}")
        print(f"   Mục tiêu: Giảm false positive (hơi phở, đèn đỏ, etc.)")
        print(f"   Data folders: {list(stage_config.data_folders)}")
        print(f"   Epochs: {stage_config.epochs}")
        print(f"   Learning rate: {stage_config.learning_rate}")

        # Load weights từ Stage 1
        if self.best_weights['baseline']:
            print(f"   📦 Loading Stage 1 weights: {self.best_weights['baseline']}")
            self.model = FireDetectionModel(
                self.config,
                weights_path=self.best_weights['baseline']
            )
        else:
            print(f"   ⚠️  Không có Stage 1 weights, dùng pretrained")

        # Chuẩn bị dataset
        dataset = FireSmokeDataset(self.config, list(stage_config.data_folders))
        data_yaml = dataset.prepare()

        # Training
        self.model.train(
            data_yaml=data_yaml,
            epochs=stage_config.epochs,
            learning_rate=stage_config.learning_rate,
            project=str(self.output_dir / "train"),
            name="stage2_hard_negative",
        )

        weights_path = str(
            self.output_dir / "train" / "stage2_hard_negative" / "weights" / "best.pt"
        )
        self.best_weights['hard_negative'] = weights_path

        print(f"\n   ✅ Stage 2 hoàn thành!")
        print(f"   📦 Best weights: {weights_path}")

        return weights_path

    def run_sahi_finetuning(self) -> str:
        """
        🥉 Stage 3: SAHI Fine-tuning
        ──────────────────────────────

        MỤC TIÊU:
            Cải thiện phát hiện vật thể nhỏ/xa:
            - Lửa nhỏ nhìn từ ban công (high-angle camera)
            - Khói mỏng ở xa, bị tường che khuất
            - Tình huống cháy thực tế từ tin tức

        TẠI SAO CẦN STAGE RIÊNG?
            Vật thể nhỏ chiếm rất ít pixel (vd: 10x10 trong ảnh 640x640).
            Training trên ảnh gốc → model chỉ học vật thể lớn.
            Stage này dùng ảnh crop/zoom + SAHI augmentation
            để model quen với vật thể nhỏ.

        DATA:
            - 04_SAHI_Small_Objects: Ảnh vật thể nhỏ (crop từ balcony cam)
            - 05_Ambient_Context_Null: Ảnh bối cảnh bình thường (không có lửa/khói)

        THAM SỐ:
            - Learning rate: 2e-5 (rất nhỏ, tinh chỉnh cuối)
            - Epochs: 30 (ít vì chỉ fine-tune nhẹ)
            - Weights: load từ Stage 2 best.pt

        Returns:
            Đường dẫn tới best weights file (FINAL model)
        """
        stage_config = self.config.training.stages.sahi

        print(f"\n{'='*60}")
        print(f"🥉 STAGE 3: SAHI FINE-TUNING")
        print(f"{'='*60}")
        print(f"   Mục tiêu: Phát hiện vật thể nhỏ/xa")
        print(f"   Data folders: {list(stage_config.data_folders)}")
        print(f"   Epochs: {stage_config.epochs}")
        print(f"   Learning rate: {stage_config.learning_rate}")

        # Load weights từ Stage 2
        if self.best_weights['hard_negative']:
            print(f"   📦 Loading Stage 2 weights: {self.best_weights['hard_negative']}")
            self.model = FireDetectionModel(
                self.config,
                weights_path=self.best_weights['hard_negative']
            )
        else:
            print(f"   ⚠️  Không có Stage 2 weights, dùng pretrained")

        # Chuẩn bị dataset
        dataset = FireSmokeDataset(self.config, list(stage_config.data_folders))
        data_yaml = dataset.prepare()

        # Training
        self.model.train(
            data_yaml=data_yaml,
            epochs=stage_config.epochs,
            learning_rate=stage_config.learning_rate,
            project=str(self.output_dir / "train"),
            name="stage3_sahi",
        )

        weights_path = str(
            self.output_dir / "train" / "stage3_sahi" / "weights" / "best.pt"
        )
        self.best_weights['sahi'] = weights_path

        print(f"\n   ✅ Stage 3 hoàn thành!")
        print(f"   📦 FINAL weights: {weights_path}")
        print(f"   🎉 Training pipeline hoàn tất!")

        return weights_path

    def run_full_pipeline(self):
        """
        Chạy toàn bộ 3-stage pipeline.

        Thứ tự:
            Stage 1 → Stage 2 → Stage 3
            Mỗi stage kế thừa weights từ stage trước.
        """
        print(f"\n{'#'*60}")
        print(f"🔥 BẮT ĐẦU 3-STAGE TRAINING PIPELINE")
        print(f"{'#'*60}")

        self.run_baseline_training()
        self.run_hard_negative_mining()
        self.run_sahi_finetuning()

        print(f"\n{'#'*60}")
        print(f"🎉 TRAINING PIPELINE HOÀN TẤT!")
        print(f"   Final weights: {self.best_weights['sahi']}")
        print(f"{'#'*60}")


# ============================================================
# CLI INTERFACE — Chạy training từ command line
# ============================================================
# Cách dùng:
#   python -m src.engine.trainer --stage baseline
#   python -m src.engine.trainer --stage hard_negative
#   python -m src.engine.trainer --stage sahi
#   python -m src.engine.trainer --stage all
# ============================================================

def main():
    """Entry point cho CLI."""
    parser = argparse.ArgumentParser(
        description="🔥 Fire Detection Training Pipeline"
    )
    parser.add_argument(
        '--stage',
        type=str,
        choices=['baseline', 'hard_negative', 'sahi', 'all'],
        default='all',
        help='Training stage: baseline, hard_negative, sahi, hoặc all'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Đường dẫn file config YAML'
    )
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='Đường dẫn tới file weights (.pt) để load'
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Tạo model
    model = FireDetectionModel(config, weights_path=args.weights)

    # Tạo trainer
    trainer = Trainer(model, config)

    # Chạy stage tương ứng
    if args.stage == 'baseline':
        trainer.run_baseline_training()
    elif args.stage == 'hard_negative':
        trainer.run_hard_negative_mining()
    elif args.stage == 'sahi':
        trainer.run_sahi_finetuning()
    elif args.stage == 'all':
        trainer.run_full_pipeline()


if __name__ == '__main__':
    main()
