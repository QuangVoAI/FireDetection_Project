"""
============================================================
📦 Dataset Module — Load và quản lý dataset Fire/Smoke
============================================================

MỤC ĐÍCH:
    Tạo PyTorch Dataset cho bài toán phát hiện lửa/khói.
    Module này:
    1. Quét tất cả ảnh và labels từ các thư mục data/
    2. Chia train/val theo tỷ lệ config
    3. Tạo file data.yaml cho Ultralytics (RT-DETR dùng format này)

GIẢI THÍCH CHO BẠN:
    - Ultralytics (thư viện chạy RT-DETR) cần file data.yaml
      chứa đường dẫn tới train/val images + class names
    - Module này tự động tạo file data.yaml từ config
    - Hỗ trợ 3-stage training: mỗi stage dùng folders khác nhau

FORMAT ANNOTATION (YOLO):
    Mỗi ảnh có 1 file .txt cùng tên chứa labels:
    <class_id> <cx> <cy> <width> <height>
    Tất cả giá trị normalized (0-1).
    class_id: 0=Fire, 1=Smoke
"""

import os
import shutil
import random
from pathlib import Path
from typing import List, Tuple, Optional

import yaml

from src.config import Config


class FireSmokeDataset:
    """
    Dataset manager cho dữ liệu Fire/Smoke.

    CHỨC NĂNG:
        1. Quét ảnh và labels từ các data folders
        2. Chia train/val split
        3. Tạo file data.yaml cho Ultralytics training

    TẠI SAO KHÔNG DÙNG PyTorch Dataset?
        Vì Ultralytics đã có sẵn data loader riêng.
        Module này chỉ cần chuẩn bị file data.yaml là đủ.
        Ultralytics sẽ tự load ảnh, augment, và batch.

    FLOW:
        1. __init__: Nhận config + danh sách folders
        2. prepare(): Quét + chia + tạo data.yaml
        3. Trả về path tới data.yaml → truyền vào model.train()
    """

    def __init__(self, config: Config, data_folders: List[str]):
        """
        Khởi tạo dataset manager.

        Args:
            config: Config object
            data_folders: Danh sách tên folders sử dụng
                          Vd: ["01_Positive_Standard", "02_Alley_Context"]
        """
        self.config = config
        self.data_folders = data_folders
        self.base_dir = Path(config.data.base_dir)
        self.train_split = config.data.train_split
        self.class_names = list(config.data.class_names)

        # Thư mục tạm để Ultralytics đọc (train/val split)
        self.prepared_dir = Path("runs") / "prepared_data"

    def prepare(self) -> str:
        """
        Chuẩn bị dataset: quét, chia, tạo data.yaml.

        FLOW:
            1. Thu thập tất cả cặp (image, label) từ các folders
            2. Shuffle ngẫu nhiên (để train/val đa dạng)
            3. Chia train/val theo tỷ lệ config
            4. Copy/symlink vào thư mục prepared
            5. Tạo file data.yaml

        Returns:
            Đường dẫn tới file data.yaml
        """
        print(f"\n{'='*60}")
        print(f"📦 Chuẩn bị dataset...")
        print(f"   Folders: {self.data_folders}")
        print(f"   Train/Val split: {self.train_split}/{1-self.train_split}")
        print(f"{'='*60}")

        # --- Bước 1: Thu thập tất cả cặp (image, label) ---
        all_pairs = self._collect_image_label_pairs()
        print(f"   ✅ Tìm thấy {len(all_pairs)} cặp ảnh-label")

        if len(all_pairs) == 0:
            raise ValueError(
                f"Không tìm thấy ảnh nào trong các folders: {self.data_folders}\n"
                f"Hãy kiểm tra lại thư mục data/ và đảm bảo có ảnh."
            )

        # --- Bước 2: Shuffle ---
        random.seed(42)  # Seed cố định để reproducible
        random.shuffle(all_pairs)

        # --- Bước 3: Chia train/val ---
        split_idx = int(len(all_pairs) * self.train_split)
        train_pairs = all_pairs[:split_idx]
        val_pairs = all_pairs[split_idx:]
        print(f"   📊 Train: {len(train_pairs)} | Val: {len(val_pairs)}")

        # --- Bước 4: Tạo thư mục và copy files ---
        self._create_split_dirs(train_pairs, val_pairs)

        # --- Bước 5: Tạo data.yaml ---
        yaml_path = self._create_data_yaml()
        print(f"   📄 data.yaml: {yaml_path}")
        print(f"{'='*60}\n")

        return str(yaml_path)

    def _collect_image_label_pairs(self) -> List[Tuple[Path, Path]]:
        """
        Quét tất cả cặp (image_path, label_path) từ các data folders.

        LOGIC:
            Với mỗi folder trong data_folders:
                1. Tìm tất cả ảnh trong folder/images/
                2. Với mỗi ảnh, tìm file label tương ứng trong folder/labels/
                3. Nếu có cả ảnh + label → thêm vào danh sách

        SUPPORTED IMAGE FORMATS: .jpg, .jpeg, .png, .bmp, .webp

        Returns:
            List of (image_path, label_path) tuples
        """
        pairs = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

        for folder_name in self.data_folders:
            folder_path = self.base_dir / folder_name
            images_dir = folder_path / "images"
            labels_dir = folder_path / "labels"

            if not images_dir.exists():
                print(f"   ⚠️  Bỏ qua thư mục (không tồn tại): {folder_name}")
                continue

            # Duyệt tất cả ảnh
            for img_path in sorted(images_dir.iterdir()):
                if img_path.suffix.lower() not in image_extensions:
                    continue

                # Tìm file label tương ứng (cùng tên, đuôi .txt)
                label_path = labels_dir / f"{img_path.stem}.txt"

                if label_path.exists():
                    pairs.append((img_path, label_path))
                else:
                    # Ảnh negative (không có label) → tạo file label rỗng
                    # Điều này quan trọng cho Hard Negative Mining!
                    pairs.append((img_path, None))

        return pairs

    def _create_split_dirs(
        self,
        train_pairs: List[Tuple[Path, Optional[Path]]],
        val_pairs: List[Tuple[Path, Optional[Path]]]
    ):
        """
        Tạo thư mục train/val và copy files vào.

        CẤU TRÚC TẠO RA:
            runs/prepared_data/
            ├── train/
            │   ├── images/    ← Ảnh training
            │   └── labels/    ← Labels training
            └── val/
                ├── images/    ← Ảnh validation
                └── labels/    ← Labels validation
        """
        # Xóa thư mục cũ nếu tồn tại
        if self.prepared_dir.exists():
            shutil.rmtree(self.prepared_dir)

        # Tạo thư mục mới
        for split in ["train", "val"]:
            (self.prepared_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (self.prepared_dir / split / "labels").mkdir(parents=True, exist_ok=True)

        # Copy files
        for split_name, pairs in [("train", train_pairs), ("val", val_pairs)]:
            for img_path, label_path in pairs:
                # Copy ảnh
                dest_img = self.prepared_dir / split_name / "images" / img_path.name
                shutil.copy2(img_path, dest_img)

                # Copy label (hoặc tạo file rỗng cho negative samples)
                dest_label = self.prepared_dir / split_name / "labels" / f"{img_path.stem}.txt"
                if label_path and label_path.exists():
                    shutil.copy2(label_path, dest_label)
                else:
                    # Tạo file label rỗng (negative sample)
                    dest_label.touch()

    def _create_data_yaml(self) -> Path:
        """
        Tạo file data.yaml cho Ultralytics.

        Ultralytics cần file này để biết:
            - Đường dẫn tới train/val images
            - Số classes
            - Tên classes

        FORMAT data.yaml:
            path: runs/prepared_data
            train: train/images
            val: val/images
            nc: 2
            names: ['Fire', 'Smoke']
        """
        data_yaml = {
            'path': str(self.prepared_dir.resolve()),
            'train': 'train/images',
            'val': 'val/images',
            'nc': len(self.class_names),
            'names': self.class_names,
        }

        yaml_path = self.prepared_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        return yaml_path

    def get_stats(self) -> dict:
        """
        Thống kê dataset: số ảnh, số labels, phân bố classes.

        Returns:
            dict chứa thống kê chi tiết
        """
        stats = {
            'total_images': 0,
            'total_labels': 0,
            'class_counts': {name: 0 for name in self.class_names},
            'folder_stats': {},
        }

        for folder_name in self.data_folders:
            folder_path = self.base_dir / folder_name
            images_dir = folder_path / "images"
            labels_dir = folder_path / "labels"

            folder_info = {'images': 0, 'labels': 0, 'objects': 0}

            if images_dir.exists():
                folder_info['images'] = len(list(images_dir.glob("*")))

            if labels_dir.exists():
                for label_file in labels_dir.glob("*.txt"):
                    folder_info['labels'] += 1
                    with open(label_file) as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                class_id = int(line.split()[0])
                                if class_id < len(self.class_names):
                                    class_name = self.class_names[class_id]
                                    stats['class_counts'][class_name] += 1
                                folder_info['objects'] += 1

            stats['total_images'] += folder_info['images']
            stats['total_labels'] += folder_info['labels']
            stats['folder_stats'][folder_name] = folder_info

        return stats
