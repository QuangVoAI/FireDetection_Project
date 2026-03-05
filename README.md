<div align="center">

# 🔥 Hệ thống Phát hiện Cháy Sớm
### Early Fire Detection System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-RT--DETR-orange)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-Academic-green)](LICENSE)
[![SAHI](https://img.shields.io/badge/SAHI-Slicing%20Inference-purple)](https://github.com/obss/sahi)

*Hệ thống giám sát AI phát hiện cháy sớm tại các hẻm nhỏ — khu vực mật độ dân cư cao và xe cứu hỏa khó tiếp cận tại TP.HCM*

[Live Demo](#live-demo) • [Dataset](#dataset-structure) • [Architecture](#model-architecture) • [Getting Started](#getting-started) • [Authors](#authors)

</div>

---

## 📌 Tổng quan

**Hệ thống Phát hiện Cháy Sớm** là giải pháp AI phát hiện lửa (`Fire`) và khói (`Smoke`) trong thời gian thực, tập trung triển khai trong **các hẻm nhỏ (hẻm)** tại TP.HCM. Môi trường này đặt ra nhiều thách thức:

- Không gian chật hẹp, ánh sáng yếu
- Hơi nước từ quán phở, xe máy (false positive)
- Vật thể nhỏ/xa bị che khuất
- Camera lắp góc cao trên ban công

| Đặc tính | Chi tiết |
|---|---|
| Model chính | RT-DETR-L (ResNet-50 backbone) |
| Framework | PyTorch + Ultralytics |
| Classes | `Fire` (Lửa), `Smoke` (Khói) |
| Input | 640×640 |
| Mục tiêu FPS | ≥ 25 FPS trên NVIDIA T4 |

---

## 🚀 Bắt đầu nhanh

```bash
# 1. Clone & cài đặt
git clone https://github.com/QuangVoAI/FireDetection_Project.git
cd FireDetection_Project
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Cấu hình API keys
cp .env.example .env  # Điền Telegram, Twilio, etc.

# 3. Training (3 stages)
python -m src.engine.trainer --stage all

# 4. Web demo
uvicorn web.main:app --host 0.0.0.0 --port 8000
```

---

## 📁 Cấu trúc Project

```
FireDetection_Project/
├── configs/default.yaml       # Cấu hình hyperparameters
├── data/                      # Dataset (5 folders)
├── src/
│   ├── config.py              # Config loader
│   ├── data/                  # Dataset, preprocessing, augmentation
│   ├── models/rtdetr_model.py # RT-DETR wrapper + SAHI
│   ├── engine/                # 3-stage trainer + evaluator
│   └── utils/                 # Visualization + Alert system
├── web/                       # FastAPI + Dark UI
├── notebooks/                 # Training notebook
├── Dockerfile
└── requirements.txt
```

---

## 🏗️ Kiến trúc Model

```
Input (640×640) → ResNet-50 → Hybrid Encoder → IoU Query Selection → Decoder → Predictions
                  (backbone)  (Transformer)    (chọn queries tốt)   (6 layers) (no NMS!)
```

---

## 🏋️ Training Pipeline (3 Stages)

| Stage | Data | Mục tiêu |
|---|---|---|
| 1. Baseline | Positive + Alley | Nhận biết lửa/khói cơ bản |
| 2. Hard Negative | + Hard Samples | Giảm false positive (hơi phở, đèn đỏ) |
| 3. SAHI | Small Objects + Real | Phát hiện vật thể nhỏ/xa |

---

## 👥 Tác giả

| Tên | MSSV | Email |
|---|---|---|
| Võ Xuân Quang | 523H0173 | 523H0173@student.tdtu.edu.vn |
| Hoàng Xuân Thành | 523H0178 | 523H0178@student.tdtu.edu.vn |

**Trường:** Đại học Tôn Đức Thắng (TDTU)

---

<div align="center">
  <sub>Built with ❤️ for community safety</sub>
</div>
