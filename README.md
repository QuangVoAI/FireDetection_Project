# 🔥 FireDetection_Project

Hệ thống phát hiện cháy sớm sử dụng kiến trúc **RT-DETR-L** (Real-Time Detection Transformer) kết hợp **SAHI** (Slicing Aided Hyper Inference) cho phát hiện đám cháy nhỏ ở xa, và hệ thống cảnh báo đa kênh thời gian thực.

> **Early Fire Detection System** – Built with RT-DETR-L, SAHI, Telegram / Zalo / Twilio SMS alerts, and Vietmap reverse-geocoding for precise location reporting.

---

## 📂 Project Structure

```
FireDetection_Project/
├── config/
│   ├── model_config.yaml        # RT-DETR training hyperparameters
│   ├── inference_config.yaml    # Inference + SAHI settings
│   ├── alert_config.yaml        # Multi-channel alert settings
│   └── dataset.yaml             # YOLO dataset descriptor
├── data/
│   ├── 01_Positive_Standard/    # Clear fire / smoke images
│   ├── 02_Alley_Context/        # HCMC alley real-world context
│   ├── 03_Negative_Hard_Samples/# Hard negatives (LED signs, steam, …)
│   ├── 04_SAHI_Small_Objects/   # Distant/small fire (SAHI required)
│   ├── 05_Real_Situation/       # News media fire scenes (VTV, VnExpress)
│   └── processed/               # Pre-processed train / val / test splits
├── models/weights/              # Trained weights (best.pt)
├── src/
│   ├── data_preprocessing.py    # Resize, dedup (pHash), split pipeline
│   ├── model_training.py        # RT-DETR-L training (baseline + HNM)
│   ├── evaluation.py            # mAP, confusion matrix, FPS evaluation
│   ├── inference.py             # Real-time inference + alert loop
│   ├── alerts/
│   │   ├── alert_manager.py     # Central alert dispatcher
│   │   ├── sound_alert.py       # Local siren / buzzer
│   │   ├── telegram_alert.py    # Telegram Bot notification
│   │   ├── zalo_alert.py        # Zalo OA notification
│   │   └── twilio_alert.py      # SMS + Voice Call via Twilio
│   └── utils/
│       ├── sahi_integration.py  # SAHI slicing utilities
│       └── vietmap_api.py       # Reverse-geocoding (GPS → address)
├── scripts/
│   ├── preprocess.py            # CLI: run preprocessing pipeline
│   ├── train.py                 # CLI: train model
│   ├── evaluate.py              # CLI: evaluate model
│   └── infer_demo.py            # CLI: run live demo
├── tests/                       # pytest test suite
├── .env.example                 # Environment variable template
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and fill in API keys / phone numbers
```

### 3. Prepare Data

Place annotated images in the 5 raw data folders (see `data/README.md`), then run:

```bash
python scripts/preprocess.py
```

This will resize all images to 640×640, remove near-duplicates via perceptual hashing,
and split them into train / val / test sets under `data/processed/`.

### 4. Train the Model

```bash
# Full pipeline: baseline training + hard-negative mining fine-tuning
python scripts/train.py

# Baseline only
python scripts/train.py --stage baseline

# Hard-negative mining only (requires baseline weights)
python scripts/train.py --stage hard_negative_mining
```

### 5. Evaluate

```bash
python scripts/evaluate.py --split test
```

Reports mAP@50, mAP@50-95, Precision, Recall, FPS, and saves confusion matrix + PR curves.

### 6. Run Inference Demo

```bash
# Webcam (default)
python scripts/infer_demo.py

# Video file with SAHI (for small/distant fire)
python scripts/infer_demo.py --source path/to/video.mp4 --use-sahi

# RTSP stream
python scripts/infer_demo.py --source rtsp://your.camera/stream

# Batch inference on SAHI folder
python scripts/infer_demo.py --folder data/04_SAHI_Small_Objects
```

---

## 🗂️ Task Checklist

### Task 1 – Data Acquisition

| Sub-folder | Content |
|---|---|
| `01_Positive_Standard` | Obvious fire & smoke (black/white smoke, visible flames) |
| `02_Alley_Context` | HCMC alley scenes (District 7, District 4, Tan Binh) |
| `03_Negative_Hard_Samples` | Confusables: pho steam, water vapour, tail lights, LED signs, red clothing |
| `04_SAHI_Small_Objects` | Distant fire/smoke shot from high balconies deep into alleys |
| `05_Real_Situation` | Real fire scenes from VTV / VnExpress news |

**Annotation**: YOLO format (`Fire = 0`, `Smoke = 1`) using Roboflow or LabelImg.

### Task 2 – Preprocessing

- Letterbox resize to **640×640**
- Near-duplicate removal via **perceptual hashing** (imagehash pHash)
- 80/10/10 train/val/test split
- Online augmentation during training: **Mosaic**, **Mixup**, Random HSV, flips

### Task 3 – Training (RT-DETR-L)

Architecture: `Image → ResNet Backbone → Hybrid Encoder (Transformer) → IoU-aware Query Selection → Decoder → Predictions`

Two-stage training:
1. **Baseline**: General fire/smoke detection on all positive data
2. **Hard Negative Mining**: Fine-tune on `03_Negative_Hard_Samples` to reduce false positives

**SAHI Integration**: Applied at inference time for `04_SAHI_Small_Objects` — slices image into 320×320 tiles, detects per-tile, merges with NMS/NMW.

### Task 4 – Evaluation

| Metric | Description |
|---|---|
| mAP@50 | Primary accuracy metric |
| mAP@50-95 | Stricter localisation quality |
| Confusion Matrix | Fire vs Smoke confusion + False Positives |
| FPS | Real-time capability (target ≥ 25 FPS on GPU) |

---

## 🚨 Alert System

When fire/smoke is detected with sufficient confidence across N consecutive frames:

| Channel | Description |
|---|---|
| 🔊 Sound | Local siren WAV playback |
| 📱 Telegram Bot | Photo + message to homeowner group |
| 💬 Zalo OA | Message via Zalo Official Account API |
| 📲 Twilio SMS | SMS to homeowner, fire department |
| 📞 Twilio Voice | Automated voice call (vi-VN TTS) |
| 📍 Vietmap | GPS → Vietnamese street address lookup |

Configure all keys in `.env` (see `.env.example`).

---

## 🛠️ Configuration

All settings are in `config/`:

| File | Controls |
|---|---|
| `model_config.yaml` | Epochs, batch size, learning rate, augmentation |
| `inference_config.yaml` | Confidence threshold, SAHI slicing parameters |
| `alert_config.yaml` | Alert thresholds, channel enable/disable, message templates |
| `dataset.yaml` | YOLO dataset paths |

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `ultralytics >= 8.3` | RT-DETR-L model (train + inference) |
| `torch >= 2.6` | PyTorch backend |
| `sahi >= 0.11.19` | Sliced inference for small objects |
| `opencv-python` | Video capture & drawing |
| `albumentations` | Offline augmentation |
| `imagehash` | Perceptual hashing for deduplication |
| `python-telegram-bot` | Telegram Bot API |
| `twilio` | SMS & voice calls |
| `python-dotenv` | Secure credential loading |

---

## 📜 License

MIT

---

*Đồ án tốt nghiệp – Hệ thống phát hiện cháy sớm sử dụng RT-DETR*