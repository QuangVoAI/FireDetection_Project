"""
============================================================
🤖 RT-DETR Model Module — Wrapper cho model phát hiện lửa/khói
============================================================

MỤC ĐÍCH:
    Bọc (wrap) model RT-DETR-L từ Ultralytics thành class
    dễ sử dụng cho project. Cung cấp API thống nhất cho:
    1. Load model (pretrained hoặc custom weights)
    2. Training (qua Ultralytics API)
    3. Inference (detect lửa/khói)
    4. SAHI inference (phát hiện vật thể nhỏ)

GIẢI THÍCH RT-DETR CHO BẠN:
    RT-DETR = Real-Time DEtection TRansformer

    TỔNG QUAN:
    ┌──────┐   ┌──────────┐   ┌──────────┐   ┌─────────┐
    │ Ảnh  │──►│ Backbone │──►│ Encoder  │──►│ Decoder │──► Detections
    │640x  │   │ ResNet50 │   │Transformer│   │6 layers │
    └──────┘   └──────────┘   └──────────┘   └─────────┘

    SO SÁNH VỚI YOLO:
    - YOLO: Dùng anchor boxes + NMS (Non-Maximum Suppression)
    - RT-DETR: End-to-end, KHÔNG cần NMS → nhanh hơn, chính xác hơn

    CÁC VARIANT:
    - RT-DETR-S: ResNet-18 backbone, nhỏ nhất, nhanh nhất
    - RT-DETR-M: ResNet-34 backbone
    - RT-DETR-L: ResNet-50 backbone ← TA DÙNG CÁI NÀY
    - RT-DETR-X: ResNet-101 backbone, lớn nhất, chính xác nhất

    TẠI SAO CHỌN L (Large)?
    - Cân bằng giữa tốc độ và độ chính xác
    - Đạt ≥25 FPS trên NVIDIA T4 (đủ real-time)
    - Pretrained trên COCO → transfer learning tốt cho fire/smoke
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Optional, Union

import cv2
import numpy as np
import torch
from ultralytics import RTDETR

from src.config import Config, get_device


class FireDetectionModel:
    """
    Wrapper class cho RT-DETR-L model.

    THUỘC TÍNH CHÍNH:
        model: Ultralytics RTDETR object
        config: Config object
        device: Device string ('cuda', 'cpu', 'mps')
        class_names: ['Fire', 'Smoke']

    VÍ DỤ SỬ DỤNG:
        >>> config = load_config('configs/default.yaml')
        >>> model = FireDetectionModel(config)
        >>>
        >>> # Training
        >>> model.train(data_yaml='runs/prepared_data/data.yaml')
        >>>
        >>> # Inference
        >>> results = model.predict('image.jpg')
        >>> print(results)
    """

    def __init__(self, config: Config, weights_path: Optional[str] = None):
        """
        Khởi tạo model.

        FLOW:
            1. Đọc config → lấy architecture, img_size, etc.
            2. Xác định device (CUDA, MPS, CPU)
            3. Load model:
               - Nếu có weights_path → load custom weights
               - Nếu không → load pretrained từ Ultralytics

        Args:
            config: Config object
            weights_path: Đường dẫn tới file .pt weights (nếu có)
        """
        self.config = config
        self.device = get_device(config)
        self.class_names = list(config.data.class_names)

        print(f"\n{'='*60}")
        print(f"🤖 Khởi tạo RT-DETR Model")
        print(f"   Architecture: {config.model.architecture}")
        print(f"   Input size:   {config.model.img_size}x{config.model.img_size}")
        print(f"   Num classes:  {config.model.num_classes}")
        print(f"   Device:       {self.device}")
        print(f"{'='*60}")

        # --- Load model ---
        if weights_path and Path(weights_path).exists():
            # Load custom weights (đã train trước đó)
            print(f"   📦 Loading weights: {weights_path}")
            self.model = RTDETR(weights_path)
        else:
            # Load pretrained weights từ Ultralytics
            # RT-DETR-L pretrained trên COCO (80 classes)
            # Ta sẽ fine-tune lại cho 2 classes (Fire, Smoke)
            model_name = config.model.weights_path  # 'rtdetr-l.pt'
            print(f"   📦 Loading pretrained: {model_name}")
            self.model = RTDETR(model_name)

        print(f"   ✅ Model loaded successfully!\n")

    def train(
        self,
        data_yaml: str,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        project: str = "runs/train",
        name: str = "fire_detection",
        resume: bool = False,
    ) -> dict:
        """
        Huấn luyện model trên dataset.

        ULTRALYTICS TRAINING:
            Ultralytics handle toàn bộ training loop:
            - Data loading + augmentation
            - Forward pass + loss computation
            - Backward pass + optimizer step
            - Validation + metrics logging
            - Checkpoint saving + early stopping
            Ta chỉ cần truyền đúng tham số.

        CÁC THAM SỐ QUAN TRỌNG:
            - data: đường dẫn data.yaml
            - epochs: số vòng lặp training
            - batch: batch size (giảm nếu thiếu VRAM)
            - imgsz: kích thước ảnh input
            - lr0: learning rate ban đầu
            - patience: early stopping patience
            - device: GPU/CPU

        Args:
            data_yaml: Đường dẫn file data.yaml
            epochs: Số epochs (None → dùng từ config)
            batch_size: Batch size (None → dùng từ config)
            learning_rate: Learning rate (None → dùng từ config)
            project: Thư mục lưu kết quả
            name: Tên experiment
            resume: Tiếp tục training từ checkpoint cuối

        Returns:
            dict chứa kết quả training (metrics)
        """
        # Lấy tham số từ config nếu không truyền
        _epochs = epochs or self.config.training.epochs
        _batch = batch_size or self.config.training.batch_size
        _lr = learning_rate or self.config.training.learning_rate

        print(f"\n🏋️ Bắt đầu Training")
        print(f"   Epochs:   {_epochs}")
        print(f"   Batch:    {_batch}")
        print(f"   LR:       {_lr}")
        print(f"   Data:     {data_yaml}")
        print(f"   Device:   {self.device}")

        # Gọi Ultralytics training API
        results = self.model.train(
            data=data_yaml,
            epochs=_epochs,
            batch=_batch,
            imgsz=self.config.model.img_size,
            lr0=_lr,
            optimizer=self.config.training.optimizer,
            weight_decay=self.config.training.weight_decay,
            warmup_epochs=self.config.training.warmup_epochs,
            patience=self.config.training.early_stopping_patience,
            device=self.device,
            project=project,
            name=name,
            exist_ok=True,
            resume=resume,
            verbose=True,
            # Augmentation params
            hsv_h=self.config.augmentation.hue_shift_limit / 360,
            hsv_s=self.config.augmentation.saturation_limit / 100,
            hsv_v=self.config.augmentation.brightness_limit,
            flipud=self.config.augmentation.vertical_flip,
            fliplr=self.config.augmentation.horizontal_flip,
            mosaic=self.config.augmentation.mosaic_prob,
            degrees=self.config.augmentation.rotation_limit,
        )

        return results

    def predict(
        self,
        source: Union[str, np.ndarray],
        conf_threshold: Optional[float] = None,
        save: bool = False,
        save_dir: str = "runs/predict",
    ) -> List[Dict]:
        """
        Phát hiện lửa/khói trong ảnh hoặc video.

        FLOW:
            1. Load ảnh (nếu truyền path)
            2. Chạy model inference
            3. Lọc kết quả theo confidence threshold
            4. Format output

        Args:
            source: Đường dẫn ảnh/video hoặc numpy array
            conf_threshold: Ngưỡng confidence (None → dùng config)
            save: Có lưu ảnh kết quả không
            save_dir: Thư mục lưu kết quả

        Returns:
            List of detections, mỗi detection là dict:
            {
                'class_id': int,        # 0 (Fire) hoặc 1 (Smoke)
                'class_name': str,      # 'Fire' hoặc 'Smoke'
                'confidence': float,    # 0.0 → 1.0
                'bbox': [x1, y1, x2, y2],  # Tọa độ pixel
                'bbox_normalized': [cx, cy, w, h],  # YOLO format
            }
        """
        _conf = conf_threshold or self.config.inference.confidence_threshold

        # Chạy inference
        results = self.model.predict(
            source=source,
            conf=_conf,
            iou=self.config.inference.iou_threshold,
            imgsz=self.config.model.img_size,
            device=self.device,
            max_det=self.config.inference.max_detections,
            save=save,
            project=save_dir if save else None,
            half=self.config.inference.half_precision,
            verbose=False,
        )

        # Format kết quả
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                # Lấy thông tin detection
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                x1, y1, x2, y2 = [float(x) for x in xyxy.tolist()]

                # Tính normalized bbox (YOLO format)
                if result.orig_shape:
                    img_h, img_w = result.orig_shape
                    cx = ((x1 + x2) / 2) / img_w
                    cy = ((y1 + y2) / 2) / img_h
                    w = (x2 - x1) / img_w
                    h = (y2 - y1) / img_h
                else:
                    cx = cy = w = h = 0.0

                detections.append({
                    'class_id': cls_id,
                    'class_name': self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}",
                    'confidence': round(conf, 4),
                    # Canonical bbox format used throughout backend
                    'bbox': [x1, y1, x2, y2],
                    # Convenience fields for web UI (some frontends expect xmin/ymin/xmax/ymax)
                    'xmin': x1,
                    'ymin': y1,
                    'xmax': x2,
                    'ymax': y2,
                    'bbox_normalized': [float(cx), float(cy), float(w), float(h)],
                })

        return detections

    def predict_with_sahi(
        self,
        image_path: str,
        conf_threshold: Optional[float] = None,
    ) -> List[Dict]:
        """
        Phát hiện vật thể nhỏ bằng SAHI (Slicing Aided Hyper Inference).

        SAHI HOẠT ĐỘNG NHƯ THẾ NÀO?
            Với ảnh lớn (vd: 1920x1080), vật thể nhỏ (vd: lửa xa)
            chiếm rất ít pixel → model khó phát hiện.

            SAHI chia ảnh thành nhiều patch nhỏ (320x320),
            chạy model trên TỪNG patch, rồi gộp kết quả:

            1. Slice ảnh → nhiều patch nhỏ (có overlap)
            2. Chạy model trên mỗi patch → detect trong patch
            3. Map tọa độ từ patch về ảnh gốc
            4. NMS để loại bỏ duplicate detections
            5. (Optional) Chạy full-image inference + merge

        TẠI SAO CẦN OVERLAP?
            Nếu vật thể nằm đúng ranh giới 2 patch,
            cả 2 patch đều chỉ thấy nửa vật thể → miss.
            Overlap 20% đảm bảo vật thể luôn nằm trọn trong ≥1 patch.

        Args:
            image_path: Đường dẫn tới ảnh
            conf_threshold: Ngưỡng confidence

        Returns:
            List of detections (giống format predict())
        """
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction

        _conf = conf_threshold or self.config.inference.confidence_threshold
        sahi_cfg = self.config.sahi

        # Tạo SAHI detection model từ Ultralytics model
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model=self.model,
            confidence_threshold=_conf,
            device=self.device,
        )

        # Chạy sliced prediction
        result = get_sliced_prediction(
            image=image_path,
            detection_model=detection_model,
            slice_height=sahi_cfg.slice_height,
            slice_width=sahi_cfg.slice_width,
            overlap_height_ratio=sahi_cfg.overlap_height_ratio,
            overlap_width_ratio=sahi_cfg.overlap_width_ratio,
            postprocess_type=sahi_cfg.postprocess_type,
            postprocess_match_threshold=sahi_cfg.postprocess_match_threshold,
            postprocess_class_agnostic=sahi_cfg.postprocess_class_agnostic,
        )

        # Format kết quả
        detections = []
        for pred in result.object_prediction_list:
            bbox = pred.bbox  # SAHI BoundingBox object
            cls_id = pred.category.id
            conf = pred.score.value
            x1, y1, x2, y2 = float(bbox.minx), float(bbox.miny), float(bbox.maxx), float(bbox.maxy)

            detections.append({
                'class_id': cls_id,
                'class_name': self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}",
                'confidence': round(conf, 4),
                'bbox': [x1, y1, x2, y2],
                'xmin': x1,
                'ymin': y1,
                'xmax': x2,
                'ymax': y2,
                'bbox_normalized': [],  # SAHI trả về pixel coords
            })

        return detections

    def export(self, format: str = "onnx", output_dir: str = "runs/export"):
        """
        Export model sang format khác để deploy.

        CÁC FORMAT HỖ TRỢ:
            - onnx: ONNX format (cross-platform)
            - torchscript: TorchScript (PyTorch deploy)
            - engine: TensorRT (NVIDIA GPU, nhanh nhất)
            - openvino: OpenVINO (Intel CPU/GPU)

        Args:
            format: Format xuất
            output_dir: Thư mục lưu
        """
        print(f"\n📦 Exporting model → {format.upper()}")
        self.model.export(
            format=format,
            imgsz=self.config.model.img_size,
            half=self.config.inference.half_precision,
        )
        print(f"   ✅ Export thành công!")

    def benchmark(self, image_path: str, num_runs: int = 100) -> dict:
        """
        Đo tốc độ inference (FPS).

        TẠI SAO CẦN ĐO FPS?
            Hệ thống real-time cần ≥25 FPS.
            Nếu chậm hơn → delay phát hiện → nguy hiểm!

        CÁCH ĐO:
            1. Warm up (chạy 10 lần trước)
            2. Đo thời gian chạy num_runs lần
            3. Tính trung bình → FPS

        Args:
            image_path: Ảnh test
            num_runs: Số lần chạy để đo trung bình

        Returns:
            dict: {'fps': float, 'avg_ms': float, 'total_ms': float}
        """
        print(f"\n⏱️ Benchmark: {num_runs} runs on {self.device}")

        # Warm up (GPU cần "khởi động" vài lần đầu)
        for _ in range(10):
            self.predict(image_path)

        # Đo thời gian
        start = time.time()
        for _ in range(num_runs):
            self.predict(image_path)
        total_time = time.time() - start

        avg_ms = (total_time / num_runs) * 1000
        fps = num_runs / total_time

        result = {
            'fps': round(fps, 1),
            'avg_ms': round(avg_ms, 2),
            'total_ms': round(total_time * 1000, 2),
            'device': self.device,
            'num_runs': num_runs,
        }

        print(f"   📊 FPS: {result['fps']}")
        print(f"   ⏱️  Avg latency: {result['avg_ms']}ms")
        print(f"   🖥️  Device: {result['device']}")

        return result
