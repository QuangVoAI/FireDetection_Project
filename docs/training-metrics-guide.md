# 📖 Hướng dẫn đọc hiểu chỉ số Training YOLOv8-seg

> **Dự án:** Phát hiện Cháy trong Hẻm (Fire Detection in Urban Alleys)
> **Model:** YOLOv8-seg (Instance Segmentation)
> **Tác giả:** 523H0173 & 523H0178

---

## 1. Tổng quan

Khi train model YOLOv8-seg, mỗi epoch sẽ hiển thị 2 nhóm thông số:

- **Training Losses** (bên trái) — Model đang học tốt đến đâu
- **Validation Metrics** (bên phải) — Model đánh giá trên dữ liệu chưa từng thấy

```
Epoch   GPU_mem  box_loss  seg_loss  cls_loss  dfl_loss  sem_loss  Instances    Size
5/100    3.44G    1.54      3.709     2.314     1.662       0         8       640: 100% ━━━━━━
         Class   Images  Instances   Box(P     R      mAP50  mAP50-95)         Mask(P    R     mAP50  mAP50-95)
           all      777      1286    0.396   0.374    0.282    0.131           0.365   0.335   0.239     0.102
```

---

## 2. Training Losses — Các hàm mất mát

> 💡 **Nguyên tắc chung:** Loss càng **thấp** càng tốt. Loss giảm dần qua các epoch = model đang học.

### 2.1. `box_loss` — Bounding Box Loss

- **Đo gì:** Độ chính xác của **khung hình chữ nhật** bao quanh đối tượng (fire/smoke)
- **Ví dụ dễ hiểu:** Nếu có 1 đám cháy trong ảnh, `box_loss` đo xem model vẽ khung bao quanh đám cháy đó sát đến đâu
- **Kỳ vọng:** Giảm dần từ ~1.5 xuống < 0.5

### 2.2. `seg_loss` — Segmentation Loss

- **Đo gì:** Độ chính xác của **mask polygon** (đường viền chính xác quanh vùng cháy)
- **Ví dụ dễ hiểu:** Thay vì chỉ vẽ khung vuông, model còn vẽ đường viền ôm sát hình dạng ngọn lửa — `seg_loss` đo xem đường viền đó khớp bao nhiêu
- **Kỳ vọng:** Giảm dần từ ~3.8 xuống < 1.5
- **⚠️ Lưu ý:** Loss này thường cao hơn `box_loss` vì vẽ mask chính xác khó hơn vẽ khung

### 2.3. `cls_loss` — Classification Loss

- **Đo gì:** Model phân biệt đúng **loại** đối tượng đến đâu (fire vs smoke)
- **Ví dụ dễ hiểu:** Model tìm thấy 1 vùng sáng — `cls_loss` đo xem nó gán nhãn "fire" hay "smoke" có đúng không
- **Kỳ vọng:** Giảm dần từ ~2.5 xuống < 0.5

### 2.4. `dfl_loss` — Distribution Focal Loss

- **Đo gì:** Loss phụ trợ giúp tinh chỉnh **biên** của bounding box chính xác hơn
- **Ví dụ dễ hiểu:** Sau khi xác định vùng cháy, loss này giúp model tinh chỉnh đúng cạnh trái/phải/trên/dưới của khung
- **Kỳ vọng:** Giảm dần từ ~1.7 xuống < 1.0

### 2.5. `sem_loss` — Semantic Loss

- **Đo gì:** Loss bổ sung cho semantic segmentation (nếu sử dụng)
- **Giá trị = 0** nghĩa là tính năng này không được bật — **bỏ qua**

---

## 3. Validation Metrics — Chỉ số đánh giá

> 💡 **Nguyên tắc chung:** Metrics càng **cao** càng tốt (thang 0 → 1). Tăng dần qua các epoch = model đang cải thiện.

### 3.1. `P` — Precision (Độ chính xác)

- **Đo gì:** Trong những vùng model **phát hiện ra**, bao nhiêu % là đúng?
- **Ví dụ dễ hiểu:** Model phát hiện 10 vùng "fire" → 7 cái đúng là fire, 3 cái sai → P = 0.7
- **Precision thấp** = Model hay bị **báo nhầm** (False Positive)
- **Mục tiêu:** > 0.7

```
                    ┌──────────────────────┐
                    │  Model phát hiện ra  │
                    │                      │
                    │   ✅ Đúng = 7/10     │ ← Precision = 70%
                    │   ❌ Sai  = 3/10     │
                    └──────────────────────┘
```

### 3.2. `R` — Recall (Độ phủ)

- **Đo gì:** Trong tất cả fire/smoke **thật sự có** trong ảnh, model tìm được bao nhiêu %?
- **Ví dụ dễ hiểu:** Có 10 đám cháy thật → model tìm được 6 → R = 0.6
- **Recall thấp** = Model hay bị **bỏ sót** (False Negative)
- **Mục tiêu:** > 0.7

```
                    ┌──────────────────────┐
                    │  Fire thật trong ảnh │
                    │                      │
                    │   ✅ Tìm thấy = 6/10│ ← Recall = 60%
                    │   😶 Bỏ sót  = 4/10 │
                    └──────────────────────┘
```

### 3.3. `mAP50` — Mean Average Precision @ IoU=0.5 ⭐

- **Đo gì:** Độ chính xác trung bình khi yêu cầu vùng phát hiện trùng ≥50% với vùng thật
- **Ví dụ dễ hiểu:** Model vẽ 1 khung quanh đám cháy → nếu khung đó trùng ≥50% diện tích với khung đáp án → tính là đúng
- **⭐ Đây là CHỈ SỐ QUAN TRỌNG NHẤT** để đánh giá model
- **Mục tiêu:** > 0.5 (tốt), > 0.7 (rất tốt)

```
    ┌─────────┐
    │ Đáp án  │
    │    ┌────┼────┐
    │    │ ≥50│    │   ← IoU ≥ 50% → Tính là ĐÚNG ✅
    └────┼────┘    │
         │ Model  │
         └────────┘
```

### 3.4. `mAP50-95` — Mean Average Precision @ IoU=0.5:0.95

- **Đo gì:** Trung bình mAP ở nhiều mức IoU khắt khe (50%, 55%, 60%, ..., 95%)
- **Ví dụ dễ hiểu:** Không chỉ trùng 50% mà phải trùng rất chính xác (lên đến 95%) → đánh giá nghiêm ngặt hơn
- **Luôn thấp hơn mAP50** — điều này bình thường
- **Mục tiêu:** > 0.3 (tốt), > 0.5 (rất tốt)

### 3.5. Box vs Mask Metrics

Mỗi metric trên đều có **2 phiên bản**:

| | Box | Mask |
|---|---|---|
| **Đánh giá dựa trên** | Bounding box (khung chữ nhật) | Segmentation mask (đường viền polygon) |
| **Thường cao hơn** | ✅ Có | |
| **Khó hơn** | | ✅ Có — vì đòi hỏi chính xác từng pixel |

---

## 4. Các thông số khác

| Thông số | Ý nghĩa |
|---|---|
| **Epoch** | Vòng lặp training hiện tại / tổng số (vd: 5/100) |
| **GPU_mem** | Bộ nhớ GPU đang sử dụng (vd: 3.44G = 3.44 GB) |
| **Images** | Số ảnh trong tập validation |
| **Instances** | Tổng số đối tượng (fire/smoke) trong tập validation |
| **Size 640** | Kích thước ảnh đầu vào (640×640 pixels) |
| **Thanh tiến trình** | 378/378 = số batch đã xử lý / tổng batch |

---

## 5. Cách đọc xu hướng training

### ✅ Dấu hiệu model đang học TỐT:
- Loss **giảm dần** qua các epoch
- mAP50 **tăng dần** qua các epoch
- Precision và Recall **cùng tăng**

### ⚠️ Dấu hiệu CẦN CHÚ Ý:
- Loss **tăng lại** sau khi đã giảm → có thể bị **overfitting**
- Precision cao nhưng Recall thấp → model **quá thận trọng**, bỏ sót nhiều
- Recall cao nhưng Precision thấp → model **phát hiện bừa**, báo nhầm nhiều
- Loss **không giảm** sau 10-20 epoch → cần điều chỉnh learning rate hoặc dữ liệu

### 📊 Bảng theo dõi mẫu

| Epoch | box_loss | seg_loss | mAP50 (Box) | mAP50 (Mask) | Nhận xét |
|---|---|---|---|---|---|
| 1 | 1.518 | 3.849 | 0.129 | 0.0995 | Mới bắt đầu |
| 2 | 1.598 | 3.819 | 0.212 | 0.175 | Đang cải thiện ✅ |
| 3 | 1.615 | 3.829 | 0.182 | 0.158 | Dao động nhẹ ⚡ |
| 4 | 1.598 | 3.876 | 0.23 | 0.184 | Phục hồi ✅ |
| 5 | 1.54 | 3.709 | 0.282 | 0.239 | Tiến triển tốt ✅ |

---

## 6. Mục tiêu chất lượng

| Chỉ số | Yếu | Trung bình | Tốt | Rất tốt |
|---|---|---|---|---|
| **mAP50 (Box)** | < 0.3 | 0.3 – 0.5 | 0.5 – 0.7 | > 0.7 |
| **mAP50 (Mask)** | < 0.2 | 0.2 – 0.4 | 0.4 – 0.6 | > 0.6 |
| **mAP50-95** | < 0.15 | 0.15 – 0.3 | 0.3 – 0.5 | > 0.5 |
| **Precision** | < 0.4 | 0.4 – 0.6 | 0.6 – 0.8 | > 0.8 |
| **Recall** | < 0.4 | 0.4 – 0.6 | 0.6 – 0.8 | > 0.8 |

> 💡 **Với dự án phát hiện cháy**, Recall quan trọng hơn Precision — vì **bỏ sót đám cháy** nguy hiểm hơn **báo nhầm**.

---

## 7. Tóm tắt nhanh

```
🔴 Loss (bên trái)    → Càng THẤP càng tốt → Model đang học
🟢 Metrics (bên phải) → Càng CAO càng tốt  → Model đang giỏi lên

⭐ Chỉ số quan trọng nhất: mAP50
🔥 Với bài toán phát hiện cháy: ưu tiên Recall > Precision
```
