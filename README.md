<div align="center">

# 🔥 Hệ thống Phát hiện Cháy Sớm
### Early Fire Detection System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-RT--DETR-orange)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-Academic-green)](LICENSE)
[![SAHI](https://img.shields.io/badge/SAHI-Slicing%20Inference-purple)](https://github.com/obss/sahi)

*This project focuses on building an AI surveillance system for the early detection of fires in small alleys—areas characterized by high population density and limited access for traditional fire trucks in Ho Chi Minh City*

[Live Demo](#live-demo) • [Dataset](#dataset-structure) • [Architecture](#model-architecture) • [Getting Started](#getting-started) • [Authors](#authors)

</div>

---

## 📌 Overview

The **Early Fire Detection System** is an AI-powered solution designed to detect fire (`Fire`) and smoke (`Smoke`) in real-time, with a focus on deployment in **narrow alleyways (hẻm)** in Ho Chi Minh City. These environments present unique challenges:

- Low-light and confined spaces
- Steam and cooking smoke from street vendors (false positives)
- Small/distant fire objects partially occluded by walls
- High-angle camera placements on balconies

The system uses **RT-DETR-L** (Real-Time Detection Transformer, Large variant) for high-accuracy detection combined with **SAHI** (Slicing Aided Hyper Inference) for small object detection, and supports **multi-channel alerting** (audio alarm, Telegram, Zalo, SMS).

| Feature | Detail |
|---|---|
| Core Model | RT-DETR-L (ResNet-50 backbone) |
| Framework | PyTorch + Ultralytics |
| Detection Classes | `Fire` (Lửa), `Smoke` (Khói) |
| Input Size | 640×640 |
| Target FPS | ≥ 25 FPS on NVIDIA T4 |
| Deployment | Edge devices + Cloud API |

---

## 🌐 Live Demo

> **Demo URL:** `https://fire-detection-demo.example.com` *(placeholder — update after deployment)*

```bash
# Quick local demo
uvicorn web.main:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000
```

---

## 📁 Dataset Structure

The dataset is organized into **5 purposeful folders** to support the 3-stage training pipeline:

```
data/
├── 01_Positive_Standard/      # Ảnh lửa và khói rõ ràng
│   ├── images/
│   └── labels/
├── 02_Alley_Context/          # Cảnh hẻm thực tế TPHCM
│   ├── images/
│   └── labels/
├── 03_Negative_Hard_Samples/  # Mẫu âm tính khó (giảm false positive)
│   ├── images/
│   └── labels/
├── 04_SAHI_Small_Objects/     # Vật thể nhỏ/xa, góc cao
│   ├── images/
│   └── labels/
└── 05_Real_Situation/         # Ảnh thực tế từ tin tức
    ├── images/
    └── labels/
```

### Folder Descriptions

| Folder | Description | Source |
|---|---|---|
| `01_Positive_Standard/` | Clear fire & smoke images under various conditions | Open datasets (COCO, Kaggle) |
| `02_Alley_Context/` | Real alley scenes from HCMC (Quận 7, Quận 4, Tân Bình) | Street cameras, manual collection |
| `03_Negative_Hard_Samples/` | Hard negatives: phở smoke, steam, motorbike tail lights, red LED signs, red clothing | Manual collection |
| `04_SAHI_Small_Objects/` | Small/distant fire objects, high-angle balcony shots | Augmented + manual |
| `05_Real_Situation/` | Real fire incident images from news (VTV, VnExpress) | Public news sources |

### Annotation Format (YOLO)

```
# labels/<image_name>.txt
# class_id cx cy width height  (all normalized 0–1)
0 0.512 0.334 0.245 0.312   # Fire
1 0.701 0.221 0.190 0.280   # Smoke
```

**Class mapping:** `0 = Fire (Lửa)`, `1 = Smoke (Khói)`

---

## 🏗️ Model Architecture

RT-DETR (Real-Time Detection Transformer) eliminates the need for NMS post-processing by using an end-to-end transformer decoder with IoU-aware query selection.

```
Input Image (640×640)
        │
        ▼
┌─────────────────┐
│  Backbone       │  ResNet-50 (pretrained on ImageNet)
│  (ResNet-50)    │  Multi-scale features: C3, C4, C5
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Hybrid Encoder │  CNN + Transformer hybrid
│  (Transformer)  │  Intra-scale & cross-scale feature fusion
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  IoU-aware      │  Select top-K queries based on IoU scores
│  Query Selection│  Eliminates low-quality anchors early
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Decoder        │  6-layer transformer decoder
│  (Transformer)  │  Cross-attention with encoder features
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Prediction     │  Class logits + Bounding box regression
│  Heads          │  No NMS required (end-to-end)
└─────────────────┘
```

---

## 🚀 Training Pipeline

The system uses a **3-stage training pipeline** to progressively improve robustness:

### Stage 1: Baseline Training
Train on `01_Positive_Standard` + `02_Alley_Context` to learn basic fire/smoke detection patterns.

```bash
python -m src.engine.trainer --stage baseline
```

### Stage 2: Hard Negative Mining
Add `03_Negative_Hard_Samples` to reduce false positives from:
- Cooking smoke from street vendors (hơi phở, hơi nước)
- Motorbike tail lights at night
- Red LED signs (đèn quảng cáo đỏ)
- Red clothing / motorbike covers

```bash
python -m src.engine.trainer --stage hard_negative
```

### Stage 3: SAHI Fine-tuning
Fine-tune on `04_SAHI_Small_Objects` + `05_Real_Situation` with SAHI slicing to improve detection of small/distant fires.

```bash
python -m src.engine.trainer --stage sahi
```

---

## 🔍 SAHI Integration

**SAHI (Slicing Aided Hyper Inference)** is used for detecting small fire/smoke objects in high-resolution images, particularly:
- Distant fires viewed from balcony cameras
- Small smoke plumes partially occluded by buildings

```
Original Image (1920×1080)
        │
        ▼  Slice into overlapping patches
┌──────┬──────┬──────┐
│ 320  │ 320  │ 320  │  ← slice_height=320, slice_width=320
├──────┼──────┼──────┤     overlap_ratio=0.2
│ 320  │ 320  │ 320  │
└──────┴──────┴──────┘
        │
        ▼  Run RT-DETR on each slice
        │
        ▼  Merge predictions with NMS
   Final Detections
```

SAHI config (`configs/default.yaml`):
```yaml
sahi:
  slice_height: 320
  slice_width: 320
  overlap_height_ratio: 0.2
  overlap_width_ratio: 0.2
  postprocess_type: NMS
  postprocess_match_threshold: 0.5
```

---

## 📊 Evaluation Metrics

| Metric | Description | Target |
|---|---|---|
| mAP@50 | Mean Average Precision at IoU=0.50 | ≥ 0.85 |
| mAP@50-95 | mAP averaged over IoU 0.50–0.95 | ≥ 0.65 |
| Precision | True positives / (TP + FP) | ≥ 0.88 |
| Recall | True positives / (TP + FN) | ≥ 0.82 |
| FPS | Frames per second on NVIDIA T4 | ≥ 25 |
| False Positive Rate | LED / steam misdetection rate | ≤ 5% |

---

## 🚨 Demo System — Intelligent Alert

When fire or smoke is detected for `consecutive_frames_to_alert` consecutive frames:

```
Detection Triggered
        │
        ├──► 🔊 Audio Alarm (siren/buzzer via pygame)
        │
        ├──► 📱 Telegram Bot
        │       └─ Sends snapshot + location + confidence
        │
        ├──► 💬 Zalo OA
        │       └─ Sends alert message in Vietnamese
        │
        ├──► 📞 Twilio SMS / Voice Call
        │       └─ Sends SMS to registered numbers
        │
        └──► 📍 Vietmap API
                └─ Reverse geocode GPS → Vietnamese address
                   "123 Hẻm 45, Phường Tân Thuận, Quận 7, TP.HCM"
```

**Alert message example (Vietnamese):**
```
🔥 CẢNH BÁO CHÁY!
Thời gian: 14:32:15 05/03/2026
Địa điểm: 123 Hẻm 45, P. Tân Thuận, Q.7, TP.HCM
Độ tin cậy: Lửa 92% | Khói 87%
Vui lòng liên hệ: 114 (Cứu hỏa)
```

---

## 📂 Project Structure

```
FireDetection_Project/
├── README.md
├── Dockerfile
├── requirements.txt
├── .gitignore
│
├── configs/
│   └── default.yaml           # Training & inference hyperparameters
│
├── data/
│   ├── README.md              # Dataset setup instructions
│   ├── 01_Positive_Standard/
│   ├── 02_Alley_Context/
│   ├── 03_Negative_Hard_Samples/
│   ├── 04_SAHI_Small_Objects/
│   └── 05_Real_Situation/
│
├── src/
│   ├── __init__.py
│   ├── config.py              # Config loader (dot-notation access)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py         # FireSmokeDataset (PyTorch)
│   │   ├── preprocessing.py   # Resize, normalize, dedup, validate
│   │   └── augmentation.py    # Albumentations pipeline
│   ├── models/
│   │   ├── __init__.py
│   │   └── rtdetr_model.py    # RT-DETR-L wrapper
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── trainer.py         # 3-stage training engine
│   │   └── evaluator.py       # mAP, confusion matrix, FPS
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py   # Bounding box drawing, plot curves
│       └── alert.py           # Multi-channel alert system
│
├── web/
│   ├── main.py                # FastAPI server
│   └── static/
│       └── index.html         # Dark-themed web UI
│
└── notebooks/
    └── FireDetection_Training.ipynb
```

---

## ⚡ Getting Started

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (recommended for training)
- NVIDIA GPU with ≥ 8GB VRAM (for RT-DETR-L training)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/QuangVoAI/FireDetection_Project.git
cd FireDetection_Project

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env with your API keys (Telegram, Twilio, Vietmap)
```

### Dataset Setup

Follow the instructions in [`data/README.md`](data/README.md) to organize your dataset.

### Training

```bash
# Stage 1: Baseline training
python -c "
from src.config import load_config
from src.models.rtdetr_model import FireDetectionModel
from src.engine.trainer import Trainer

config = load_config('configs/default.yaml')
model = FireDetectionModel(config)
trainer = Trainer(model, config)
trainer.run_baseline_training()
trainer.run_hard_negative_mining()
trainer.run_sahi_finetuning()
"

# Or use the notebook
jupyter notebook notebooks/FireDetection_Training.ipynb
```

### Inference

```bash
# Single image inference
python -c "
from src.config import load_config
from src.models.rtdetr_model import FireDetectionModel

config = load_config('configs/default.yaml')
model = FireDetectionModel(config)
detections = model.predict('path/to/image.jpg')
print(detections)
"
```

### Start Web Server

```bash
uvicorn web.main:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000
```

### Docker

```bash
docker build -t fire-detection .
docker run -p 8000:8000 fire-detection
```

---

## ⚙️ Configuration

Key hyperparameters in `configs/default.yaml`:

| Parameter | Default | Description |
|---|---|---|
| `model.architecture` | `rtdetr-l` | Model variant |
| `model.img_size` | `640` | Input image size |
| `model.num_classes` | `2` | Fire + Smoke |
| `training.epochs` | `100` | Total training epochs |
| `training.batch_size` | `16` | Batch size |
| `training.learning_rate` | `1e-4` | Initial learning rate |
| `training.optimizer` | `AdamW` | Optimizer |
| `training.early_stopping_patience` | `15` | Early stopping epochs |
| `sahi.slice_height` | `320` | SAHI slice height |
| `sahi.slice_width` | `320` | SAHI slice width |
| `inference.confidence_threshold` | `0.35` | Detection confidence cutoff |
| `alert.consecutive_frames_to_alert` | `3` | Frames before triggering alert |

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `torch` | ≥ 2.0.0 | Deep learning framework |
| `ultralytics` | ≥ 8.0.0 | RT-DETR implementation |
| `sahi` | ≥ 0.11.0 | Slicing aided inference |
| `fastapi` | latest | Web API server |
| `opencv-python` | ≥ 4.8.0 | Image/video processing |
| `albumentations` | ≥ 1.3.0 | Data augmentation |
| `python-telegram-bot` | ≥ 20.0 | Telegram notifications |
| `twilio` | ≥ 8.0.0 | SMS/Voice alerts |
| `imagehash` | latest | Dataset deduplication |

---

## 👥 Authors

| Name | Student ID | Contact |
|---|---|---|
| Vo Xuan Quang | 523H0173 | 523H0173@student.tdtu.edu.vn |
| *(teammate name)* | *(ID)* | *(email)* |

**Institution:** Ton Duc Thang University (TDTU) — Trường Đại học Tôn Đức Thắng

---

## 📄 License

This project is developed for **academic purposes** at Ton Duc Thang University.  
*Dự án được phát triển cho mục đích học thuật tại Trường ĐH Tôn Đức Thắng.*

```
Copyright (c) 2026 Vo Xuan Quang & Team
Academic use only — not for commercial redistribution.
```

---

<div align="center">
  <sub>Built with ❤️ and 🔥 detection for the safety of HCMC communities</sub>
</div>
