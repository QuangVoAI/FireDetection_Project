"""
============================================================
🔄 Preprocessing Module — Tiền xử lý và làm sạch dữ liệu
============================================================

MỤC ĐÍCH:
    Chuẩn bị dataset trước khi training:
    1. Resize ảnh về kích thước chuẩn
    2. Validate labels (kiểm tra format YOLO đúng chưa)
    3. Deduplication (loại bỏ ảnh trùng lặp)
    4. Kiểm tra quality (ảnh bị lỗi, quá nhỏ, etc.)

GIẢI THÍCH CHO BẠN:
    Dữ liệu thu thập từ nhiều nguồn (Kaggle, camera, tin tức)
    nên có thể bị:
    - Trùng lặp ảnh (cùng ảnh download nhiều lần)
    - Labels sai format (coordinate ngoài [0,1])
    - Ảnh bị lỗi (không đọc được)
    - Ảnh quá nhỏ (không đủ chi tiết)
    Module này giúp làm sạch tất cả những vấn đề đó.
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np


def validate_yolo_labels(labels_dir: str, num_classes: int = 2) -> Dict:
    """
    Kiểm tra tất cả file labels có đúng format YOLO không.

    FORMAT YOLO HỢP LỆ:
        <class_id> <cx> <cy> <width> <height>
        - class_id: số nguyên, 0 <= class_id < num_classes
        - cx, cy, width, height: float, giá trị trong [0, 1]

    Args:
        labels_dir: Đường dẫn thư mục chứa file .txt labels
        num_classes: Số lớp phát hiện (2: Fire, Smoke)

    Returns:
        dict với keys: 'valid', 'invalid', 'errors'
            - valid: list file labels hợp lệ
            - invalid: list file labels có lỗi
            - errors: dict mapping filename → error message
    """
    labels_path = Path(labels_dir)
    result = {'valid': [], 'invalid': [], 'errors': {}}

    if not labels_path.exists():
        print(f"⚠️  Thư mục labels không tồn tại: {labels_dir}")
        return result

    for label_file in sorted(labels_path.glob("*.txt")):
        errors = _validate_single_label(label_file, num_classes)
        if errors:
            result['invalid'].append(str(label_file))
            result['errors'][label_file.name] = errors
        else:
            result['valid'].append(str(label_file))

    # In kết quả
    total = len(result['valid']) + len(result['invalid'])
    print(f"📋 Validation: {len(result['valid'])}/{total} labels hợp lệ")
    if result['invalid']:
        print(f"   ❌ {len(result['invalid'])} labels có lỗi:")
        for fname, errs in result['errors'].items():
            for err in errs:
                print(f"      - {fname}: {err}")

    return result


def _validate_single_label(label_path: Path, num_classes: int) -> List[str]:
    """
    Kiểm tra 1 file label.

    Args:
        label_path: Path tới file .txt
        num_classes: Số lớp

    Returns:
        List of error messages (rỗng nếu hợp lệ)
    """
    errors = []

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue  # Bỏ qua dòng trống

        parts = line.split()

        # Kiểm tra số trường (phải có 5: class cx cy w h)
        if len(parts) != 5:
            errors.append(f"Dòng {i}: Cần 5 giá trị, tìm thấy {len(parts)}")
            continue

        try:
            class_id = int(parts[0])
            cx, cy, w, h = [float(x) for x in parts[1:]]
        except ValueError:
            errors.append(f"Dòng {i}: Không thể convert sang số")
            continue

        # Kiểm tra class_id
        if class_id < 0 or class_id >= num_classes:
            errors.append(
                f"Dòng {i}: class_id={class_id} ngoài phạm vi [0, {num_classes-1}]"
            )

        # Kiểm tra tọa độ trong [0, 1]
        for name, val in [('cx', cx), ('cy', cy), ('width', w), ('height', h)]:
            if val < 0 or val > 1:
                errors.append(f"Dòng {i}: {name}={val} ngoài phạm vi [0, 1]")

        # Kiểm tra width, height > 0
        if w <= 0 or h <= 0:
            errors.append(f"Dòng {i}: width hoặc height phải > 0")

    return errors


def check_image_quality(
    images_dir: str,
    min_width: int = 64,
    min_height: int = 64
) -> Dict:
    """
    Kiểm tra chất lượng ảnh: có đọc được không, có quá nhỏ không.

    TẠI SAO CẦN?
        - Ảnh bị corrupt (download lỗi) → OpenCV không đọc được
        - Ảnh quá nhỏ (< 64x64) → không đủ chi tiết để train
        - Ảnh grayscale trong khi model cần RGB

    Args:
        images_dir: Thư mục chứa ảnh
        min_width: Chiều rộng tối thiểu
        min_height: Chiều cao tối thiểu

    Returns:
        dict với 'valid', 'corrupt', 'too_small'
    """
    images_path = Path(images_dir)
    result = {'valid': [], 'corrupt': [], 'too_small': []}
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    if not images_path.exists():
        print(f"⚠️  Thư mục ảnh không tồn tại: {images_dir}")
        return result

    for img_file in sorted(images_path.iterdir()):
        if img_file.suffix.lower() not in extensions:
            continue

        # Thử đọc ảnh
        img = cv2.imread(str(img_file))

        if img is None:
            result['corrupt'].append(str(img_file))
            continue

        h, w = img.shape[:2]
        if w < min_width or h < min_height:
            result['too_small'].append(str(img_file))
        else:
            result['valid'].append(str(img_file))

    total = len(result['valid']) + len(result['corrupt']) + len(result['too_small'])
    print(f"🖼️  Image quality: {len(result['valid'])}/{total} ảnh OK")
    if result['corrupt']:
        print(f"   ❌ {len(result['corrupt'])} ảnh bị hỏng (corrupt)")
    if result['too_small']:
        print(f"   ⚠️  {len(result['too_small'])} ảnh quá nhỏ (< {min_width}x{min_height})")

    return result


def find_duplicates(images_dir: str, hash_size: int = 8) -> List[List[str]]:
    """
    Tìm ảnh trùng lặp bằng perceptual hashing.

    PERCEPTUAL HASH LÀ GÌ?
        - Mỗi ảnh được convert thành 1 "fingerprint" (hash)
        - 2 ảnh giống nhau → hash giống nhau (hoặc gần giống)
        - Khác với MD5: perceptual hash nhận biết ảnh giống nhau
          kể cả khi resize, nén JPEG, hay thay đổi nhẹ màu sắc

    THUẬT TOÁN:
        1. Resize ảnh về kích thước nhỏ (8x8)
        2. Chuyển sang grayscale
        3. Tính DCT (Discrete Cosine Transform)
        4. Lấy bit dựa trên median
        → 64-bit hash cho mỗi ảnh

    Args:
        images_dir: Thư mục chứa ảnh
        hash_size: Kích thước hash (8 → 64-bit hash)

    Returns:
        List of groups, mỗi group là list ảnh trùng nhau
    """
    try:
        import imagehash
        from PIL import Image
    except ImportError:
        print("⚠️  Cần cài imagehash: pip install imagehash Pillow")
        return []

    images_path = Path(images_dir)
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    # Tính hash cho từng ảnh
    hash_dict = {}  # hash → list of image paths
    for img_file in sorted(images_path.iterdir()):
        if img_file.suffix.lower() not in extensions:
            continue

        try:
            img = Image.open(img_file)
            img_hash = str(imagehash.phash(img, hash_size=hash_size))

            if img_hash not in hash_dict:
                hash_dict[img_hash] = []
            hash_dict[img_hash].append(str(img_file))
        except Exception as e:
            print(f"   ⚠️  Lỗi hash {img_file.name}: {e}")

    # Tìm nhóm trùng lặp (>1 ảnh cùng hash)
    duplicates = [paths for paths in hash_dict.values() if len(paths) > 1]

    if duplicates:
        total_dups = sum(len(group) - 1 for group in duplicates)
        print(f"🔍 Tìm thấy {total_dups} ảnh trùng lặp trong {len(duplicates)} nhóm")
    else:
        print(f"🔍 Không tìm thấy ảnh trùng lặp")

    return duplicates


def resize_images(
    images_dir: str,
    output_dir: str,
    target_size: int = 640,
    keep_aspect_ratio: bool = True
):
    """
    Resize tất cả ảnh về kích thước chuẩn.

    TẠI SAO CẦN RESIZE?
        RT-DETR nhận input 640x640. Nếu ảnh gốc lớn hơn,
        Ultralytics sẽ tự resize, nhưng pre-resize giúp:
        - Giảm thời gian load data khi training
        - Giảm dung lượng lưu trữ
        - Đảm bảo consistency

    Args:
        images_dir: Thư mục ảnh gốc
        output_dir: Thư mục lưu ảnh đã resize
        target_size: Kích thước đích (pixels)
        keep_aspect_ratio: Giữ tỷ lệ khung hình (True=padding, False=stretch)
    """
    src_path = Path(images_dir)
    dst_path = Path(output_dir)
    dst_path.mkdir(parents=True, exist_ok=True)

    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    count = 0

    for img_file in sorted(src_path.iterdir()):
        if img_file.suffix.lower() not in extensions:
            continue

        img = cv2.imread(str(img_file))
        if img is None:
            continue

        if keep_aspect_ratio:
            # Resize giữ tỷ lệ + padding (letterbox)
            resized = _letterbox_resize(img, target_size)
        else:
            # Resize stretch (không giữ tỷ lệ)
            resized = cv2.resize(img, (target_size, target_size))

        output_path = dst_path / img_file.name
        cv2.imwrite(str(output_path), resized)
        count += 1

    print(f"✅ Đã resize {count} ảnh → {target_size}x{target_size}")


def _letterbox_resize(img: np.ndarray, target_size: int) -> np.ndarray:
    """
    Resize ảnh giữ tỷ lệ + padding (letterbox).

    LETTERBOX LÀ GÌ?
        Thay vì stretch ảnh (làm méo vật thể), ta:
        1. Scale ảnh cho vừa target_size (giữ tỷ lệ)
        2. Padding 2 bên (hoặc trên/dưới) bằng màu xám (114)

    VÍ DỤ:
        Ảnh 1920x1080 → resize về 640x640:
        1. Scale 1920→640 (ratio=0.333) → 640x360
        2. Padding trên/dưới mỗi bên 140 pixels
        → Kết quả 640x640, không bị méo

    Args:
        img: Ảnh numpy array (BGR)
        target_size: Kích thước đích

    Returns:
        Ảnh đã resize + padding
    """
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)

    # Resize
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h))

    # Padding
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2
    canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

    return canvas
