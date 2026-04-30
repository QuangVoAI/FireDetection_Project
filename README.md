<div align="center">

# 🔥 Hệ thống Phát hiện Cháy Sớm
### Early Fire Detection System — TP.HCM Alley-focused

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-RT--DETR-orange)](https://github.com/ultralytics/ultralytics)
[![SAHI](https://img.shields.io/badge/SAHI-Small%20Object-purple)](https://github.com/obss/sahi)
[![Roboflow](https://img.shields.io/badge/Roboflow-Dataset-blue)](https://roboflow.com/)
[![License](https://img.shields.io/badge/License-Academic-green)](LICENSE)

*Hệ thống giám sát AI phát hiện lửa và khói từ sớm tại các hẻm nhỏ — nơi mật độ dân cư cao và xe cứu hỏa khó tiếp cận tại TP.HCM*

[Tổng quan](#-tổng-quan) • [Kiến trúc](#-kiến-trúc-mô-hình) • [Dataset](#-cấu-trúc-dataset) • [Training](#️-training-pipeline) • [Đánh giá](#-đánh-giá) • [Triển khai](#-triển-khai--demo) • [Tác giả](#-tác-giả)

</div>

---

## 📌 Tổng quan

**Hệ thống Phát hiện Cháy Sớm** là giải pháp AI phát hiện lửa (`Fire`) và khói (`Smoke`) trong thời gian thực, tập trung triển khai tại các **hẻm nhỏ (hẻm)** TP.HCM — môi trường đặt ra nhiều thách thức đặc thù:

| Thách thức | Mô tả |
|---|---|
| 🏙️ Không gian chật hẹp | Hẻm 1–3m, góc camera hạn chế, vật thể che khuất |
| 💡 Nguồn sáng hỗn tạp | Đèn LED, bảng hiệu neon, đèn hậu xe máy dễ gây false positive |
| 🍜 Nhiễu môi trường | Khói từ quán phở/hủ tiếu, hơi nước, khói xe máy |
| 📷 Camera góc cao | Lắp trên ban công → lửa/khói nhỏ bé ở xa hẻm sâu |
| 🌙 Điều kiện ban đêm | Ánh sáng yếu, độ tương phản thấp |

### Thông số hệ thống

| Đặc tính | Chi tiết |
|---|---|
| Model chính | RT-DETR-L (ResNet-50 backbone) |
| Framework | PyTorch + Ultralytics |
| Classes | `Fire` (Lửa), `Smoke` (Khói) |
| Input Resolution | 640 × 640 |
| Mục tiêu FPS | ≥ 25 FPS (NVIDIA T4) |
| Inference Mode | RT-DETR + SAHI (Sliced) |
| Alert Channels | Telegram Bot, Zalo OA, SMS/Voice (Twilio) |

---

## 🏗️ Kiến trúc Mô hình

```
Input Frame (640×640)
       │
       ▼
┌──────────────┐
│  ResNet-50   │  ← Backbone: trích xuất đặc trưng không gian
│  (Backbone)  │
└──────┬───────┘
       │  FPN Feature Maps (C3, C4, C5)
       ▼
┌──────────────────────┐
│   Hybrid Encoder     │  ← Transformer: học tương quan toàn cục
│   (AIFI + CCFM)     │     giữa các vùng ảnh, nhận biết hình
└──────┬───────────────┘     dạng khói loãng không rõ viền
       │
       ▼
┌──────────────────────┐
│  IoU-aware Query     │  ← Chọn lọc top-K queries tốt nhất
│  Selection           │     trước khi đưa vào Decoder
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│   Decoder            │  ← 6 lớp Transformer Decoder
│   (6 layers)         │     tinh chỉnh vị trí + phân loại
└──────┬───────────────┘
       │
       ▼
  Predictions          ← Không cần NMS — output trực tiếp
  (Boxes + Labels)        (lợi thế so với YOLO)
```

### Tại sao RT-DETR thay vì YOLO?

- **Không dùng NMS:** RT-DETR loại bỏ Non-Maximum Suppression — vốn là bước dễ bỏ sót các đốm lửa gần nhau trong hẻm chật
- **Transformer Encoder:** Học được tương quan toàn cục, nhận biết khói loãng không có viền rõ ràng tốt hơn CNN thuần
- **Tương thích SAHI:** Hoạt động tốt với Slicing Inference cho vật thể nhỏ ở xa

---

## 📊 Chiến lược A/B Testing — 3 Mô hình So sánh

Đồ án áp dụng phương pháp thực nghiệm khoa học để chứng minh tác động của từng yếu tố:

| | Mô hình 1 (Baseline) | Mô hình 2 (Data++) | Mô hình 3 (SOTA) |
|---|---|---|---|
| **Kiến trúc** | RT-DETR-L | RT-DETR-L | RF-DETR |
| **Backbone** | ResNet-50 | ResNet-50 | DINOv2 |
| **Pretrained trên** | COCO (118K ảnh) | Objects365 (2M+ ảnh) | Objects365 + SSL |
| **Mục đích** | Benchmark chuẩn tối thiểu | Chứng minh sức mạnh của data | Kiến trúc SOTA hiện đại nhất |
| **Kỳ vọng** | mAP baseline | ↑ mAP, ↓ FP | ↑↑ mAP_s, tốt nhất |

> 💡 **Hypothesis:** Objects365 chứa nhiều vật thể đa dạng hơn COCO → đặc trưng backbone phong phú hơn → phân biệt khói/lửa vs. nhiễu tốt hơn. DINOv2 với Self-Supervised Learning → bóc tách tốt các vật thể có hình dạng không xác định như khói loãng.

---

## 📁 Cấu trúc Dataset

Toàn bộ dataset (~15.000 ảnh) được tổ chức theo nguyên tắc **Curriculum Learning** — từ dễ đến khó, từ rõ ràng đến tinh tế:

```
data/
├── 01_Positive_Standard/     # ~9.100 ảnh (60%) ✅ Hoàn thành
├── 02_Alley_Context/         # ~600 ảnh   (4%)  🚀 Sắp xong
├── 03_Negative_Hard_Samples/ # ~800 ảnh   (5%)  🚀 Sắp xong
├── 04_SAHI_Small_Objects/    # ~1.000 ảnh (7%)  🚀 Sắp xong
└── 05_Ambient_Context_Null/  # ~3.500 ảnh (24%) ✅ Hoàn thành
```

### Chi tiết từng tập

#### `01_Positive_Standard` — ~9.100 ảnh
Ảnh lửa bùng phát, khói đen/trắng rõ ràng.
- Trạng thái: ✅ **Hoàn thành 100%**

#### `02_Alley_Context` — ~600 ảnh
Bối cảnh thực tế hẻm nhỏ TP.HCM: nhà cửa sát nhau, xe máy dày đặc.
- Trạng thái: 🚀 **Đã có 408/600** (Chỉ cần thêm ~200 ảnh)

#### `03_Negative_Hard_Samples` — ~800 ảnh
Các nguồn gây false positive: đèn LED đỏ, sương mù, khói bếp, đèn hậu xe máy.
- Trạng thái: 🚀 **Đang thu thập** (Dùng bộ DetectiumFire là xong ngay)

#### `04_SAHI_Small_Objects` — ~1.000 ảnh
Đốm lửa/khói cực nhỏ, chụp từ ban công cao hoặc góc flycam.
- Trạng thái: 🚀 **Đang thu thập** (Dùng bộ Wildfire-Smoke là xong ngay)

#### `05_Ambient_Context_Null` — ~3.500 ảnh
Cảnh sinh hoạt bình thường: rừng, đường phố, nhà cửa — **không có lửa/khói**.
- Trạng thái: ✅ **Hoàn thành 100%**

### Công cụ Annotation
- **Platform:** Roboflow (project `FireDetection_Master`, phân loại bằng Tags)
- **Tracking:** GitHub Actions + `roboflow_metrics_tracker.py` tự động theo dõi tiến độ hàng ngày, nhắc nhở qua Discord

---

## ⚙️ Cấu trúc Project

```
FireDetection_Project/
├── configs/
│   └── default.yaml           # Hyperparameters, paths, thresholds
├── data/                      # 5 thư mục dataset (xem Dataset section)
├── src/
│   ├── config.py              # Config loader
│   ├── data/
│   │   ├── dataset.py         # FireDataset class
│   │   ├── preprocessing.py   # Deduping (Perceptual Hash), validation
│   │   └── augmentation.py    # Albumentations pipeline on-the-fly
│   ├── models/
│   │   └── rtdetr_model.py    # RT-DETR wrapper + SAHI integration
│   ├── engine/
│   │   ├── trainer.py         # 3-stage training controller
│   │   └── evaluator.py       # mAP, Recall, FPS evaluation
│   └── utils/
│       ├── visualization.py   # Annotated frame rendering
│       └── alert.py           # Multi-channel alert system
├── web/                       # FastAPI backend + Dark UI frontend
├── notebooks/
│   └── training_analysis.ipynb  # Loss curves, confusion matrix
├── tests/                     # Unit tests cho các module chính
├── Dockerfile
├── requirements.txt
└── .env.example               # API keys template
```

---

## 🏋️ Training Pipeline (3 Stages)

```bash
# Chạy toàn bộ 3 stages
python -m src.engine.trainer --stage all

# Hoặc chạy từng stage
python -m src.engine.trainer --stage 1   # Baseline
python -m src.engine.trainer --stage 2   # Hard Negative
python -m src.engine.trainer --stage 3   # Robustness + SAHI
```

### Chi tiết từng Stage

| Stage | Dữ liệu sử dụng | Mục tiêu | Epochs | Learning Rate |
|---|---|---|---|---|
| **1. Baseline** | `01` + `02` | Nhận biết lửa/khói cơ bản trong bối cảnh thực tế | 80 | 1e-4 |
| **2. Hard Negative** | Stage 1 + `03` | Giảm false positive (đèn đỏ, hơi phở, LED) | 50 | 5e-5 |
| **3. SAHI Fine-tuning** | Stage 2 + `04` + `05` | Tối ưu phát hiện lửa nhỏ/xa & giảm nhiễu nền | 30 | 2e-5 |

> **Lý do dùng Curriculum Learning:** Đưa hard negative quá sớm (Stage 1) khiến model chưa học được đặc trưng cơ bản đã bị "confuse". Tiến trình tịnh tiến giúp model xây nền vững chắc trước khi gặp các trường hợp khó.

### 🛡️ Chiến lược Huấn luyện Dữ liệu Nhỏ (Small Dataset Optimization)

Để tối ưu cho tập dữ liệu hẻm TPHCM đặc thù (folder `02_Alley_Context`), hệ thống áp dụng chiến lược **Strong Augmentation** cực hạn:

- **Mosaic 1.0 (Extreme):** Luôn ghép 4 ảnh ngẫu nhiên thành 1 khung hình. Điều này ép model phải học cách phát hiện lửa trong các không gian bị chia cắt, che khuất — mô phỏng hoàn hảo sự chật hẹp và vật cản trong hẻm nhỏ.
- **Mixup (Alpha 0.15):** Trộn các lớp ảnh đè lên nhau với độ mờ khác nhau. Kỹ thuật này giúp model nhận diện khói loãng và lửa trong điều kiện ánh sáng chồng chéo phức tạp (như đèn xe máy ban đêm).
- **Random Scaling (±50%):** Tự động thay đổi kích cỡ vật thể, giúp model linh hoạt với cả đám cháy lớn ở gần và đốm lửa nhỏ ở cuối hẻm sâu.
- **Transfer Learning:** Khởi tạo từ trọng số `rtdetr-l.pt` (huấn luyện trên Objects365/COCO) để kế thừa tri thức về hình khối và cấu trúc không gian, giúp model hội tụ nhanh dù tập dữ liệu đích chỉ có vài trăm ảnh.

---

## 📈 Đánh giá

### Metrics chính

| Metric | Ý nghĩa | Mục tiêu |
|---|---|---|
| **mAP@50** | Độ chính xác tổng thể | ≥ 0.75 |
| **mAP_s** | Độ chính xác vật thể nhỏ (SAHI) | ≥ 0.60 |
| **Recall** | Tỷ lệ phát hiện đúng — **ưu tiên cao nhất** | ≥ 0.85 |
| **False Positive Rate** | Tỷ lệ báo động giả | ≤ 0.10 |
| **FPS** | Tốc độ thời gian thực | ≥ 25 FPS |

> ⚠️ **Tại sao Recall quan trọng hơn Precision?**  
> Trong bài toán phát hiện cháy, **bỏ sót 1 đám cháy nguy hiểm hơn rất nhiều** so với 1 báo động giả. Hệ thống được thiết kế ưu tiên Recall cao — có thể chấp nhận thêm một số false positive, nhưng không được bỏ sót cháy thật.

### Chạy đánh giá

```bash
# Đánh giá cả 3 mô hình với SAHI
python -m src.engine.evaluator --model all --sahi --split test

# Xuất kết quả so sánh
python -m src.engine.evaluator --export-report --output reports/comparison.xlsx
```

### Phân tích lỗi (Error Analysis)

```bash
# Xuất các ảnh predict sai để phân tích
python -m src.engine.evaluator --error-analysis --top-k 50
```

Sau mỗi lần training, bắt buộc phân tích ít nhất 20–30 ảnh predict sai để hiểu pattern lỗi — không chỉ nhìn con số mAP.

---

## 🚀 Bắt đầu nhanh

### Yêu cầu hệ thống

| Thành phần | Tối thiểu | Khuyến nghị |
|---|---|---|
| GPU | NVIDIA 8GB VRAM | NVIDIA T4 / RTX 3080 |
| RAM | 16 GB | 32 GB |
| Storage | 50 GB | 100 GB |
| Python | 3.9+ | 3.10+ |
| CUDA | 11.8 | 12.1 |

### Cài đặt

```bash
# 1. Clone repository
git clone https://github.com/QuangVoAI/FireDetection_Project.git
cd FireDetection_Project

# 2. Tạo virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 3. Cài đặt dependencies
pip install -r requirements.txt

# 4. Cấu hình API keys
cp .env.example .env
# Điền: TELEGRAM_TOKEN, ZALO_OA_KEY, TWILIO_SID, VIETMAP_API_KEY, ROBOFLOW_API_KEY
```

### Tải Dataset từ Roboflow

```bash
python scripts/download_dataset.py \
    --project FireDetection_Master \
    --version latest \
    --format yolov8
```

### Chạy Training

```bash
# Full pipeline (khuyến nghị)
python -m src.engine.trainer --stage all --model rt-detr-l

# A/B test: Chạy cả 3 mô hình
python scripts/run_ab_test.py --models baseline objects365 rfdetr
```

### Chạy Web Demo

```bash
uvicorn web.main:app --host 0.0.0.0 --port 8000
# Truy cập: http://localhost:8000
```

---

## 🚒 Triển khai & Demo

### Inference Pipeline

```
Camera Feed → Frame Buffer → RT-DETR + SAHI → Confidence Filter → Alert Logic → Notification
```

### Logic Lọc Cảnh báo Thông minh

Hệ thống không báo động ngay lập tức sau 1 frame để tránh false trigger:

```python
# Exponential Moving Average trên confidence score
ema_score = alpha * current_conf + (1 - alpha) * ema_score

# Chỉ trigger alert khi:
# 1. EMA score vượt ngưỡng liên tục (≥ 3 frames)
# 2. Đã qua thời gian cooldown (60s)
if ema_score > THRESHOLD and frames_above >= 3 and cooldown_elapsed:
    trigger_alert()
```

> **Tại sao dùng EMA thay vì "3 frame cứng"?** EMA mượt hơn khi khói loãng xuất hiện/biến mất không đều — tránh flicker cảnh báo và phản ứng tốt hơn với khói tản dần.

### Hệ thống Báo động Đa kênh

| Kênh | Nội dung | Thời gian gửi |
|---|---|---|
| 🔊 Còi cục bộ | Âm thanh tại chỗ | Ngay lập tức |
| 📱 Telegram Bot | Ảnh + địa chỉ + tọa độ | < 2 giây |
| 📲 Zalo OA | Thông báo đẩy | < 2 giây |
| 📞 SMS/Voice | Twilio auto-call | < 5 giây |

### Định vị chính xác

Tích hợp **Vietmap API** để đính kèm địa chỉ số nhà + tọa độ GPS vào mỗi cảnh báo:

```json
{
  "timestamp": "2026-04-30T14:32:11",
  "location": "Hẻm 123/4 Nguyễn Trãi, Quận 5, TP.HCM",
  "coordinates": { "lat": 10.7537, "lng": 106.6624 },
  "confidence": 0.87,
  "classes": ["Fire", "Smoke"],
  "snapshot_url": "https://..."
}
```

---

## ⚠️ Giới hạn đã biết (Known Limitations)

> Hiểu rõ giới hạn hệ thống quan trọng không kém biết điểm mạnh.

| Giới hạn | Mô tả | Hướng khắc phục |
|---|---|---|
| 🌧️ **Thời tiết xấu** | Mưa lớn, sương mù dày có thể che khuất lửa/khói | Tích hợp camera IR |
| 🔆 **Camera chất lượng thấp** | Độ phân giải < 720p giảm đáng kể khả năng phát hiện vật nhỏ | Yêu cầu tối thiểu 1080p |
| 📷 **Góc khuất** | Camera bị che bởi mái tôn, dây điện → blind spot | Layout camera đa góc |
| 🌙 **Ban đêm không có IR** | Camera thường (không IR) gần như mù trong hẻm tối | Camera hồng ngoại bắt buộc |
| ⚡ **Edge deployment** | Chưa tối ưu cho Jetson Nano/Xavier — cần quantization | TensorRT export (roadmap) |
| 🔁 **Không có temporal context** | Mỗi frame xử lý độc lập, chưa tận dụng chuyển động khói | Optical Flow (v2.0) |

---

## 🗺️ Roadmap

| Giai đoạn | Trạng thái | Mô tả |
|---|---|---|
| **Task 1:** Data Acquisition | ✅ Hoàn thành | Hoàn tất tập Positive, Negative và tối ưu hóa tập hẻm thực tế |
| **Task 2:** Preprocessing Pipeline | ✅ Hoàn thành | On-the-fly augmentation, deduping |
| **Task 3:** Training 3 Models | ⏳ Sắp tới | A/B Test: RT-DETR COCO → O365 → RF-DETR |
| **Task 4:** SAHI Evaluation | ⏳ Sắp tới | mAP_s, FPS benchmark trên T4 |
| **Task 5:** Deployment & Demo | ⏳ Sắp tới | FastAPI + Alert system + Vietmap |

---

## 📦 Dependencies chính

```
ultralytics>=8.0.0      # RT-DETR, training engine
torch>=2.0.0            # Deep learning framework
sahi>=0.11.0            # Slicing Aided Hyper Inference
albumentations>=1.3.0   # On-the-fly augmentation
fastapi>=0.104.0        # Web API server
roboflow>=1.1.0         # Dataset management
python-telegram-bot     # Telegram alert
twilio                  # SMS/Voice alert
```

---

## 👥 Tác giả

| Tên | MSSV | Email | Vai trò |
|---|---|---|---|
| Võ Xuân Quang | 523H0173 | 523H0173@student.tdtu.edu.vn | Model Architecture, Training Pipeline |
| Hoàng Xuân Thành | 523H0178 | 523H0178@student.tdtu.edu.vn | Data Collection, Alert System |

**Trường:** Đại học Tôn Đức Thắng (TDTU)  
**Môn học:** Đồ án chuyên ngành  
**Năm học:** 2025–2026

---

## 📄 License

Dự án này được phát triển cho mục đích học thuật. Mọi sử dụng thương mại cần có sự đồng ý của tác giả.

---

<div align="center">
  <sub>Built with ❤️ for community safety — TP.HCM, 2026</sub>
</div>