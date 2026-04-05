
# 🗺️ Kế hoạch triển khai Đồ án Phát hiện Cháy Sớm (Roadmap)

Tài liệu này hệ thống lại quy trình thực hiện đồ án, tập trung vào việc tận dụng tối đa 05 cấu trúc thư mục dữ liệu đã nêu để xây dựng một mô hình AI có độ chính xác cao và thực tiễn bằng kiến trúc **RT-DETR**.

---

## 🟢 Task 1: Xây dựng Tập dữ liệu Chuẩn Production (Data Acquisition) - [ĐANG THỰC HIỆN]
Đây là giai đoạn quan trọng nhất, quyết định thành bại của đồ án. Áp dụng chuẩn **Data-centric AI**, gán nhãn ở cấp độ **Pixel (Polygon Mask)** để đạt độ tinh khiết tối đa thay vì dạng Bounding Box thông thường. Toàn bộ tập dữ liệu hướng tới quy mô: **~29.500 ảnh** *(cân đối lại do thư mục 01 vượt mục tiêu)*.

* **Quản trị Data trên Roboflow:** Sử dụng duy nhất 1 Project (`FireDetection_Master`), dùng tính năng **Tags** (vd: `01_positive_standard`) để phân loại ảnh rạch ròi.
* **Phân bổ cấu trúc 5 lớp dữ liệu (Curriculum Learning):**
  1. `01_Positive_Standard` **(12.000 ảnh / 41%):** Ảnh lửa bùng phát, khói rõ ràng. Đây là lớp nền tảng để model học hình khối/màu sắc cơ bản của ngọn lửa/khói. *(✅ Đã hoàn thành — gấp 1.6x mục tiêu ban đầu 7.500).*
  2. `02_Alley_Context` **(~5.000 ảnh / 17%):** Bối cảnh thực tế nhà cửa, dây điện chằng chịt, hẻm nhỏ hẹp đặc thù tại TP.HCM. Giúp model làm quen không gian đô thị phức tạp.
  3. `03_Negative_Hard_Samples` **(~5.500 ảnh / 19%):** Ánh đèn Neon đỏ, sương mù, khói bún bò xe hủ tiếu, đèn hậu xe máy. *Lưu ý: Không vẽ nhãn Polygon nào ở tập này.* Nhằm trừng phạt hàm Loss, triệt tiêu hoàn toàn khả năng báo động giả (False Positive). *(⬆️ Tăng từ 3.500 → 5.500 để cân đối tỷ lệ negative/positive khi 01 tăng lên 12K).*
  4. `04_SAHI_Small_Objects` **(~5.000 ảnh / 17%):** Đốm lửa/khói cực kỳ nhỏ bé, chụp từ góc cao hoặc flycam. Lớp dữ liệu bản lề giúp mô hình phát huy sức mạnh Cảnh Báo Sớm (Early Warning) khi kết hợp cùng công nghệ SAHI.
  5. `05_Real_Situation` **(~2.000 ảnh / 6%):** Hiện trường thực chiến ban đêm, khói mịt mù, vòi rồng cứu hỏa, nước phun, lính cứu hỏa. Bài Test khắt khe nhất để kiểm chứng tính chống chịu (Robustness).
* **Tracking tự động:** Đã tích hợp API Github Actions & Roboflow (`roboflow_metrics_tracker.py`) để theo dõi tiến độ gán nhãn hàng ngày, tự động nhắc nhở tiến độ qua Discord.

---

## 🟡 Task 2: Tiền xử lý & Tạo dữ liệu tổng hợp (Preprocessing) - [HOÀN THÀNH 80%]
* **Chiến lược:** Không xử lý "chết" dữ liệu ra ổ cứng, toàn bộ Resize và Augmentation được thực hiện qua **Pipline On-the-fly** bằng Albumentations và Dataloader của Ultralytics.
* **Chuẩn hóa:**
  - Resize ảnh tự động (Letterbox) về kích thước 640x640 bảo toàn hoàn hảo bounding box.
  - Loại bỏ ảnh trùng lặp (Deduping - sử dụng Perceptual Hashing) và ảnh lỗi.
* **Augmentation (Tăng cường dữ liệu):**
  - Mô phỏng ban đêm/hẻm thiếu sáng (`SimulateLowLight`, `CLAHE`, `RandomBrightnessContrast`).
  - Thêm nhiễu camera rẻ tiền (`GaussianBlur`, `GaussNoise`).
  - Lật, xoay và tạo bóng râm (`HorizontalFlip`, `ShiftScaleRotate`, `RandomShadow`).

---

## 🔵 Task 3: Huấn luyện Hệ 3 Mô hình So sánh (Core AI) - [SẮP TỚI]
Áp dụng chiến lược A/B Testing chuyên sâu để đánh giá sức mạnh của Dữ liệu (Objects365) và Kiến trúc (DINOv2 Backbone).

* **Framework & Kiến trúc:** PyTorch, Ultralytics, và hệ sinh thái Roboflow (Roboflow-DETR). Cả 3 đều là kiến trúc Transformer không dùng thuật toán NMS phức tạp.
* **Chiến lược Huấn luyện (Tiến trình tịnh tiến):**
  1. **Mô hình 1 (Đường cơ sở - Baseline): `RT-DETR (Pretrained COCO)`**
     - Đặt ra mức tiêu chuẩn tối thiểu (Benchmark) cho đồ án bằng tập weights phổ biến nhất.
  2. **Mô hình 2 (Sức mạnh dữ liệu): `RT-DETR (Pretrained Objects365)`**
     - Chứng minh việc nâng cấp dữ liệu huấn luyện (từ 118k ảnh của COCO lên hơn 2 triệu ảnh của O365) giúp trích xuất đặc trưng Lửa/Khói tốt hơn và giảm False Positives rõ rệt.
  3. **Mô hình 3 (Sức mạnh nền tảng - SOTA): `RF-DETR (Backbone DINOv2)`**
     - Mô hình chốt hạ. Áp dụng kiến trúc Self-Supervised Learning (DINOv2) xịn nhất hiện nay, bóc tách hoàn hảo ánh sáng và độ mờ nhiễu của Lửa/Khói.
* **Theo dõi:** Hàm Loss (Classification, Bounding box) để xem tốc độ hội tụ của RF-DETR so với RT-DETR.

---

## ⚪ Task 4: Đánh giá Cực hạn với SAHI (Evaluation & Testing) - [SẮP TỚI]
* **Tích hợp SAHI (Slicing Aided Hyper Inference):** Bắt buộc chạy SAHI trên tập Validation/Test cho CẢ 3 MÔ HÌNH để có sự so sánh công bằng (Fair Comparison). Cắt các bức ảnh lớn (4K/8K) thành các patch nhỏ để không bỏ sót đốm lửa/khói nào.
* **Chỉ số đánh giá độ chính xác:**
  - Trích xuất mAP@50 và đặc biệt là **mAP_s (mAP for Small objects)** để chứng minh hiệu quả của thiết kế.
  - Confusion Matrix để đánh giá khả năng loại bỏ nhiễu/cảnh báo giả (False Positive Rate).
* **Đánh giá Hiệu năng (Performance vs Accuracy trade-off):** 
  - Đo lường FPS (Tốc độ khung hình/giây) và so sánh thời gian Inference giữa RT-DETR và RF-DETR trên cùng cài đặt SAHI.

---

## ⚪ Task 5: Hệ thống Demo & Triển khai (Deployment) - [SẮP TỚI]
* **Inference Pipeline:** Chạy mô hình RT-DETR + SAHI trên luồng Video/Camera giám sát. Thuật toán lọc báo động thông minh (Chỉ báo khi có lửa 3 frame liên tục, cooldown 60s để chống spam).
* **Báo động tại chỗ (Local Alert):** Kích hoạt còi báo động / audio khi phát hiện sự cố.
* **Thông báo đa kênh (Multi-channel Notification):** Triển khai tự động đẩy thông báo khẩn qua Telegram Bot, Zalo OA (đã lên khung trong `alert.py`).
* **Định vị chính xác:** Tích hợp Vietmap API đính kèm địa chỉ/tọa độ vào cảnh báo, hỗ trợ công tác cứu hộ nhanh chóng nhất.

---
*Cập nhật lần cuối: 04/04/2026 — Thư mục 01 đã đạt 12.000+ ảnh (vượt mục tiêu 7.500). Label Bot (YOLOv8s-seg) đang được train lại với cấu hình cải tiến.*

---

## 📌 Ghi chú Kỹ thuật cốt lõi (Technical Memory)
> [!IMPORTANT]
> **Về việc giữ nhãn Polygon (Mặt nạ chóp đa giác):**
> - Khi tải dữ liệu từ Roboflow, bắt buộc chọn **YOLOv8 Instance Segmentation**.
> - Khi viết Code Python huấn luyện qua Ultralytics, **TUYỆT ĐỐI BẮT BUỘC** khai báo model có vần `-seg.pt` (Ví dụ: `yolov8n-seg.pt` hoặc `yolo11n-seg.pt`). 
> - Nếu nạp nhầm model `.pt` thường (Object Detection), toàn bộ công sức gán nhãn Polygon viền khít của hơn 29.500 ảnh sẽ bị thư viện ép vuông thành Bounding Box (Mất hoàn toàn lợi thế độ tinh khiết ảnh).
