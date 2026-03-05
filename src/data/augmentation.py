"""
============================================================
🎨 Augmentation Module — Tăng cường dữ liệu (Data Augmentation)
============================================================

MỤC ĐÍCH:
    Tạo thêm dữ liệu training bằng cách biến đổi ảnh gốc.
    Giúp model:
    1. Tránh overfitting (học thuộc thay vì học đặc trưng)
    2. Robust hơn với các biến đổi (sáng/tối, góc chụp, etc.)
    3. Tăng đa dạng dữ liệu mà không cần thu thập thêm

GIẢI THÍCH CHO BẠN:
    "Augmentation" = "tăng cường". Thay vì cần 10000 ảnh,
    bạn chỉ cần 2000 ảnh rồi augment thành 10000.

    Tưởng tượng bạn chụp 1 ảnh lửa, rồi:
    - Lật ngang → ảnh mới (lửa vẫn là lửa)
    - Tăng sáng → ảnh mới (ban ngày)
    - Giảm sáng → ảnh mới (ban đêm)
    - Thêm noise → ảnh mới (camera chất lượng thấp)
    → Từ 1 ảnh ra 4-5 ảnh, model học đa dạng hơn!

THƯ VIỆN:
    Dùng Albumentations — thư viện augmentation mạnh nhất Python.
    Ưu điểm: nhanh (dùng OpenCV), nhiều kỹ thuật, tự biến đổi
    cả bounding box theo ảnh.

LƯU Ý QUAN TRỌNG:
    Ultralytics đã có sẵn augmentation trong training loop.
    Module này cung cấp augmentation pipeline TÙY CHỈNH
    nếu bạn muốn kiểm soát chi tiết hơn hoặc áp dụng offline.
"""

from typing import Dict, Tuple, Optional

import cv2
import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform

from src.config import Config


def get_train_augmentation(config: Config) -> A.Compose:
    """
    Tạo augmentation pipeline cho training.

    PIPELINE GỒM CÁC BƯỚC:
        1. HorizontalFlip: Lật ảnh ngang
        2. RandomBrightnessContrast: Thay đổi sáng/tối
        3. HueSaturationValue: Thay đổi màu sắc
        4. GaussianBlur: Mờ nhẹ
        5. GaussNoise: Thêm noise
        6. ShiftScaleRotate: Dịch, scale, xoay

    BBOX_PARAMS:
        Khi augment ảnh, bounding box cũng phải biến đổi theo!
        Vd: lật ngang ảnh → bbox cũng phải lật ngang
        Albumentations tự làm điều này nếu ta truyền bbox_params.

    Args:
        config: Config object chứa augmentation params

    Returns:
        Albumentations Compose pipeline
    """
    aug_config = config.augmentation

    transforms = [
        # --- 1. Lật ảnh ngang ---
        # Lửa lật ngang vẫn là lửa → augmentation an toàn
        A.HorizontalFlip(p=aug_config.horizontal_flip),

        # --- 2. Xoay + dịch chuyển ---
        # Mô phỏng camera lắp nghiêng hoặc góc chụp khác nhau
        A.ShiftScaleRotate(
            shift_limit=0.05,         # Dịch tối đa 5%
            scale_limit=0.1,          # Scale ±10%
            rotate_limit=aug_config.rotation_limit,  # Xoay ±15°
            border_mode=cv2.BORDER_CONSTANT,
            value=114,                # Padding xám (giống letterbox)
            p=0.5,
        ),

        # --- 3. Thay đổi ánh sáng ---
        # Mô phỏng: ban ngày, ban đêm, đèn đường, trong nhà
        A.RandomBrightnessContrast(
            brightness_limit=aug_config.brightness_limit,  # ±30%
            contrast_limit=aug_config.contrast_limit,      # ±30%
            p=0.5,
        ),

        # --- 4. Thay đổi màu sắc ---
        # Lửa có thể cam, đỏ, vàng → model cần nhận biết tất cả
        A.HueSaturationValue(
            hue_shift_limit=aug_config.hue_shift_limit,    # ±10
            sat_shift_limit=aug_config.saturation_limit,   # ±30
            val_shift_limit=20,
            p=0.5,
        ),

        # --- 5. Mờ ảnh ---
        # Mô phỏng camera chất lượng thấp trong hẻm
        A.GaussianBlur(
            blur_limit=(1, aug_config.blur_limit),  # Kernel 1-3
            p=0.2,
        ),

        # --- 6. Thêm noise ---
        # Mô phỏng camera giám sát rẻ tiền, ban đêm
        A.GaussNoise(
            # Đã deprecated var_limit, dùng std_range thay thế
            # Noise với standard deviation từ 5 đến 30
            p=aug_config.noise_prob,
        ),

        # --- 7. CLAHE: Cân bằng histogram ---
        # Tăng chi tiết trong vùng tối (quan trọng cho ban đêm)
        A.CLAHE(
            clip_limit=2.0,
            tile_grid_size=(8, 8),
            p=0.2,
        ),

        # --- 8. Random Shadow ---
        # Mô phỏng bóng đổ từ tường/mái trong hẻm
        A.RandomShadow(
            shadow_roi=(0, 0.5, 1, 1),  # Vùng bóng đổ: nửa dưới ảnh
            num_shadows_limit=(1, 2),
            shadow_dimension=5,
            p=0.2,
        ),
    ]

    # BboxParams: cấu hình cho bounding box augmentation
    # format='yolo': cx, cy, w, h (normalized)
    # min_visibility=0.3: chỉ giữ bbox nếu ≥30% vẫn visible sau augment
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='yolo',
            min_visibility=0.3,
            label_fields=['class_labels'],
        ),
    )


def get_val_augmentation() -> A.Compose:
    """
    Augmentation pipeline cho validation/test.

    QUAN TRỌNG:
        Validation KHÔNG augment! Chỉ resize.
        Vì khi đánh giá model, ta cần data nguyên gốc
        để kết quả chính xác và so sánh công bằng.

    Returns:
        Albumentations pipeline (chỉ resize)
    """
    return A.Compose(
        [
            # Không augment gì cả, chỉ normalize nếu cần
        ],
        bbox_params=A.BboxParams(
            format='yolo',
            min_visibility=0.3,
            label_fields=['class_labels'],
        ),
    )


def apply_augmentation(
    image: np.ndarray,
    bboxes: list,
    class_labels: list,
    transform: A.Compose
) -> Tuple[np.ndarray, list, list]:
    """
    Áp dụng augmentation lên 1 cặp (image, bboxes).

    VÍ DỤ SỬ DỤNG:
        >>> transform = get_train_augmentation(config)
        >>> aug_img, aug_bboxes, aug_labels = apply_augmentation(
        ...     image, bboxes, class_labels, transform
        ... )

    Args:
        image: Ảnh numpy (H, W, C) — BGR hoặc RGB
        bboxes: List of [cx, cy, w, h] (normalized YOLO format)
        class_labels: List of class_id tương ứng với mỗi bbox
        transform: Augmentation pipeline

    Returns:
        (augmented_image, augmented_bboxes, augmented_labels)
    """
    result = transform(
        image=image,
        bboxes=bboxes,
        class_labels=class_labels,
    )

    return (
        result['image'],
        result['bboxes'],
        result['class_labels'],
    )


class SimulateLowLight(ImageOnlyTransform):
    """
    Custom augmentation: mô phỏng điều kiện ánh sáng yếu.

    TẠI SAO CẦN?
        Camera trong hẻm thường chụp ban đêm hoặc trong nhà tối.
        Augmentation sẵn có (RandomBrightness) chỉ giảm đều,
        nhưng thực tế ban đêm có:
        - Vùng sáng (đèn đường) và vùng tối (góc hẻm)
        - Noise nhiều hơn khi tối
        - Màu sắc ít rõ ràng hơn

    THUẬT TOÁN:
        1. Giảm brightness ngẫu nhiên (0.3-0.6 lần)
        2. Thêm Gaussian noise (mô phỏng ISO cao)
        3. Giảm saturation (màu sắc nhạt đi khi tối)
    """

    def __init__(self, always_apply=False, p=0.3):
        super().__init__(always_apply, p)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """Áp dụng hiệu ứng low-light."""
        # Giảm brightness
        factor = np.random.uniform(0.3, 0.6)
        result = (img.astype(np.float32) * factor).clip(0, 255).astype(np.uint8)

        # Thêm noise
        noise = np.random.normal(0, 15, result.shape).astype(np.float32)
        result = (result.astype(np.float32) + noise).clip(0, 255).astype(np.uint8)

        # Giảm saturation
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= np.random.uniform(0.5, 0.8)  # Giảm saturation
        hsv = hsv.clip(0, 255).astype(np.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return result

    def get_transform_init_args_names(self):
        return ()
