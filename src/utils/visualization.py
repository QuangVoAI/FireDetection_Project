"""
============================================================
🎨 Visualization Module — Vẽ kết quả detection
============================================================

MỤC ĐÍCH:
    Trực quan hóa kết quả phát hiện lửa/khói:
    1. Vẽ bounding box lên ảnh
    2. Hiển thị class name + confidence
    3. Vẽ training curves (loss, mAP qua epochs)
    4. Tạo video kết quả detection

GIẢI THÍCH CHO BẠN:
    Visualization rất quan trọng trong deep learning:
    - Kiểm tra model có detect đúng vị trí không
    - Phát hiện lỗi (bbox quá to, quá nhỏ, nhầm class)
    - Demo cho giảng viên/khách hàng
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Màu sắc cho từng class
# Fire: đỏ cam (BGR format cho OpenCV)
# Smoke: xám xanh
CLASS_COLORS = {
    'Fire': (0, 69, 255),     # Đỏ cam (BGR)
    'Smoke': (200, 200, 100),  # Xám xanh (BGR)
}

# Màu mặc định nếu class không có trong dict
DEFAULT_COLOR = (0, 255, 0)  # Xanh lá


def draw_detections(
    image: np.ndarray,
    detections: List[Dict],
    show_confidence: bool = True,
    line_thickness: int = 2,
    font_scale: float = 0.7,
) -> np.ndarray:
    """
    Vẽ bounding boxes lên ảnh.

    FLOW:
        Với mỗi detection:
        1. Vẽ rectangle (bounding box)
        2. Vẽ text box (nền) + text (class + confidence)
        3. Tô màu theo class (Fire=đỏ, Smoke=xám)

    VÍ DỤ:
        >>> detections = model.predict('image.jpg')
        >>> img = cv2.imread('image.jpg')
        >>> result = draw_detections(img, detections)
        >>> cv2.imwrite('result.jpg', result)

    Args:
        image: Ảnh numpy (BGR format)
        detections: List of detection dicts (từ model.predict())
        show_confidence: Hiện % confidence không
        line_thickness: Độ dày viền bbox
        font_scale: Kích thước chữ

    Returns:
        Ảnh đã vẽ bbox (numpy array, BGR)
    """
    # Copy ảnh để không sửa ảnh gốc
    result = image.copy()

    for det in detections:
        class_name = det['class_name']
        confidence = det['confidence']
        bbox = det['bbox']  # [x1, y1, x2, y2]

        # Tọa độ pixel
        x1, y1, x2, y2 = [int(v) for v in bbox]

        # Chọn màu theo class
        color = CLASS_COLORS.get(class_name, DEFAULT_COLOR)

        # --- 1. Vẽ bounding box ---
        cv2.rectangle(result, (x1, y1), (x2, y2), color, line_thickness)

        # --- 2. Vẽ label text ---
        if show_confidence:
            label = f"{class_name} {confidence:.0%}"
        else:
            label = class_name

        # Tính kích thước text
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )

        # Vẽ nền text (hình chữ nhật filled)
        cv2.rectangle(
            result,
            (x1, y1 - text_h - baseline - 4),
            (x1 + text_w + 4, y1),
            color,
            -1,  # -1 = filled
        )

        # Vẽ text
        cv2.putText(
            result,
            label,
            (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),  # Trắng
            1,
            cv2.LINE_AA,
        )

    return result


def draw_detections_fancy(
    image: np.ndarray,
    detections: List[Dict],
    alpha: float = 0.3,
) -> np.ndarray:
    """
    Vẽ bounding boxes kiểu fancy (có hiệu ứng trong suốt).

    KHÁC GÌ VỚI draw_detections()?
        - draw_detections: bbox đơn giản (viền + text)
        - draw_detections_fancy: bbox + vùng tô bán trong suốt
          → Dễ nhìn hơn, đẹp hơn khi demo

    Args:
        image: Ảnh numpy (BGR)
        detections: List detections
        alpha: Độ trong suốt (0=trong suốt, 1=đặc)

    Returns:
        Ảnh đã vẽ (BGR)
    """
    result = image.copy()
    overlay = image.copy()

    for det in detections:
        class_name = det['class_name']
        confidence = det['confidence']
        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
        color = CLASS_COLORS.get(class_name, DEFAULT_COLOR)

        # Vẽ filled rectangle trên overlay
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

    # Blend overlay với ảnh gốc
    cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)

    # Vẽ viền + text (trên ảnh đã blend)
    result = draw_detections(result, detections)

    return result


def plot_training_curves(
    log_dir: str,
    save_path: Optional[str] = None,
):
    """
    Vẽ đồ thị loss và metrics theo epochs.

    METRICS THƯỜNG CÓ:
        - Train loss: giảm dần → model đang học
        - Val loss: giảm rồi tăng → overfitting
        - mAP: tăng dần → model đang giỏi hơn
        - Precision/Recall: cần cân bằng

    CÁCH ĐỌC BIỂU ĐỒ:
        Loss (mất mát):
        - Giảm liên tục = TỐT (model học được)
        - Tăng đột ngột = XẤU (learning rate quá lớn)
        - Train loss giảm nhưng Val loss tăng = OVERFITTING

        mAP (độ chính xác):
        - Tăng liên tục = TỐT
        - Dao động = LR quá lớn hoặc data ít
        - Bão hòa (không tăng nữa) = model đã hết khả năng

    Args:
        log_dir: Thư mục chứa results.csv từ Ultralytics
        save_path: Đường dẫn lưu ảnh
    """
    import pandas as pd

    results_csv = Path(log_dir) / "results.csv"

    if not results_csv.exists():
        print(f"⚠️  Không tìm thấy {results_csv}")
        return

    # Đọc training log
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()  # Remove trailing spaces

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Curves — Fire Detection', fontsize=16)

    # --- Plot 1: Loss ---
    ax = axes[0, 0]
    if 'train/box_loss' in df.columns:
        ax.plot(df['epoch'], df['train/box_loss'], label='Box Loss', alpha=0.8)
    if 'train/cls_loss' in df.columns:
        ax.plot(df['epoch'], df['train/cls_loss'], label='Cls Loss', alpha=0.8)
    if 'train/dfl_loss' in df.columns:
        ax.plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 2: Validation Loss ---
    ax = axes[0, 1]
    if 'val/box_loss' in df.columns:
        ax.plot(df['epoch'], df['val/box_loss'], label='Box Loss', alpha=0.8)
    if 'val/cls_loss' in df.columns:
        ax.plot(df['epoch'], df['val/cls_loss'], label='Cls Loss', alpha=0.8)
    if 'val/dfl_loss' in df.columns:
        ax.plot(df['epoch'], df['val/dfl_loss'], label='DFL Loss', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 3: mAP ---
    ax = axes[1, 0]
    if 'metrics/mAP50(B)' in df.columns:
        ax.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@50', linewidth=2)
    if 'metrics/mAP50-95(B)' in df.columns:
        ax.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@50-95', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP')
    ax.set_title('Mean Average Precision')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # --- Plot 4: Precision & Recall ---
    ax = axes[1, 1]
    if 'metrics/precision(B)' in df.columns:
        ax.plot(df['epoch'], df['metrics/precision(B)'], label='Precision', linewidth=2)
    if 'metrics/recall(B)' in df.columns:
        ax.plot(df['epoch'], df['metrics/recall(B)'], label='Recall', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Precision & Recall')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path is None:
        save_path = str(Path(log_dir) / "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📈 Training curves saved: {save_path}")


def create_detection_video(
    video_path: str,
    model,
    output_path: str,
    use_sahi: bool = False,
    show_fps: bool = True,
):
    """
    Tạo video kết quả detection.

    FLOW:
        1. Đọc video frame by frame
        2. Chạy detection trên mỗi frame
        3. Vẽ bbox lên frame
        4. Ghi ra video mới

    Args:
        video_path: Đường dẫn video input
        model: FireDetectionModel instance
        output_path: Đường dẫn video output
        use_sahi: Dùng SAHI inference không
        show_fps: Hiển thị FPS trên video
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Không thể mở video: {video_path}")
        return

    # Lấy thông tin video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Tạo video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"🎬 Processing video: {video_path}")
    print(f"   Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")

    frame_count = 0
    import time

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        # Detection
        if use_sahi:
            # SAHI cần file path, save frame tạm
            temp_path = "/tmp/sahi_frame.jpg"
            cv2.imwrite(temp_path, frame)
            detections = model.predict_with_sahi(temp_path)
        else:
            detections = model.predict(frame)

        inference_time = time.time() - start_time

        # Vẽ bbox
        result_frame = draw_detections_fancy(frame, detections)

        # Hiển thị FPS
        if show_fps:
            current_fps = 1.0 / max(inference_time, 1e-6)
            cv2.putText(
                result_frame,
                f"FPS: {current_fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )

        out.write(result_frame)
        frame_count += 1

        # Progress
        if frame_count % 100 == 0:
            print(f"   Processed: {frame_count}/{total_frames} frames")

    cap.release()
    out.release()
    print(f"   ✅ Video saved: {output_path}")
