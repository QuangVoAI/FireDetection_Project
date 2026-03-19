import os
import requests
import time

TOKEN = os.environ.get("GITHUB_TOKEN")
REPO = os.environ.get("GITHUB_REPOSITORY")
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github.v3+json"}
URL = f"https://api.github.com/repos/{REPO}/issues"

def create_issue(title, body, labels):
    payload = {"title": title, "body": body, "labels": labels}
    resp = requests.post(URL, json=payload, headers=HEADERS)
    if resp.status_code in [201, 200]:
        print(f"✅ Đã tạo thành công Ticket: {title}")
    else:
        print(f"❌ Lỗi tạo Ticket ({resp.status_code}): {resp.text}")
    time.sleep(1) # Tránh bị dính Rate Limit của GitHub API

tickets = [
    # EPIC 1: DATA ENGINEERING (Tối ưu để cán mốc mAP > 90%)
    {
        "title": "🎫 [GĐ 1] Data - Positive Standard (Rừng, Ban đêm, Đô thị)",
        "body": "### 🎯 KPI Đạt Chuẩn >90% mAP: 4,000 Ảnh\n- **Thư mục:** `data/01_Positive_Standard`\n- **Nhiệm vụ (Trị chứng Overfitting của RT-DETR):** Transformer cần sự đa dạng khổng lồ. Thu thập ảnh cháy rừng, cháy nhà xưởng.\n- **ĐIỀU KIỆN 9.0:** BẮT BUỘC 1/3 số ảnh (tầm 1500 tấm) phải là chụp BAN ĐÊM (Night-time fires) để chống lỗi lóe sáng Cam (Lens flare).\n- **Hành động:** Nạp lên Roboflow, khoanh Bounding Box thật sát viền.",
        "labels": ["epic-1-data", "todo", "high-priority"]
    },
    {
        "title": "🎫 [GĐ 1] Data - Alley Context (Bối cảnh Đô thị cốt lõi)",
        "body": "### 🎯 KPI (Bảo vệ nội dung Đề tài): 3,000 Ảnh\n- **Thư mục:** `data/02_Alley_Context`\n- **Nhiệm vụ:** Đây là 'linh hồn' của đề tài Hẻm. Không có tệp này, AI ra đời thực gặp dây điện chằng chịt sẽ bị mù hoàn toàn.\n- **Hành động:** Tìm ảnh các con hẻm chật hẹp, xe cộ đông đúc, nhà ống, chung cư cũ chuồng cọp. Chấp nhận kỹ thuật Photoshop: Dán lửa giả vào cục nóng máy lạnh, ban công để Model học bối cảnh.",
        "labels": ["epic-1-data", "todo"]
    },
    {
        "title": "🎫 [GĐ 1] Data - Hard Negatives (Lá Chắn Mạng - Diệt False Positive)",
        "body": "### 🎯 KPI (Quyết định điểm 10 Khóa luận): 2,000 Ảnh\n- **Thư mục:** `data/03_Negative_Hard_Samples`\n- **Nhiệm vụ:** Chống báo giả (False Positives). Đào tạo RT-DETR phân biệt rạch ròi cục diện 'Cảnh giống cháy' vs 'Cháy thật'.\n- **Hành động:** Tải ảnh Sương mù Đà Lạt, mây trời bù mịt, hoàng hôn đỏ rực, đèn LED đỏ gắt, đèn đuôi xe máy kẹt xe ban đêm. \n- **LƯU Ý TINH MẠNG:** TUYỆT ĐỐI KHÔNG KHOANH BOX NÀO CẢ (Lưu file TXT trống không).",
        "labels": ["epic-1-data", "todo", "high-priority"]
    },
    {
        "title": "🎫 [GĐ 1] Data - SAHI Small Objects (Lửa đốm siêu nhỏ)",
        "body": "### 🎯 KPI (Đột Phá Kỹ thuật AI): 1,000 Ảnh\n- **Thư mục:** `data/04_SAHI_Small_Objects`\n- **Nhiệm vụ:** Transformer có khả năng bù trừ mờ cực tồi với vật thể bé hơn 3% diện tích bức ảnh do bị Resize ép xuống 640x640. Data này dùng kỹ thuật SAHI.\n- **Hành động:** Tải ảnh Flycam cháy rừng, ảnh quay lén từ ban công tòa nhà đối diện cách 500m. Đốm lửa phải bé chỉ bằng hạt đậu.",
        "labels": ["epic-1-data", "todo"]
    },
    {
        "title": "🎫 [GĐ 1] Data - Real Situation (Tập Kiểm định Thực Tế CCTV)",
        "body": "### 🎯 KPI (TẬP ĐÁNH GIÁ THESIS): 1,000 Ảnh\n- **Thư mục:** `data/05_Real_Situation`\n- **Nhiệm vụ:** Giữ bí mật 100% (Không dùng Train). Dùng để Test xem mô hình có ảo tưởng không.\n- **Hành động:** Lên báo VnExpress, Tuổi Trẻ, VTV cắt ảnh Camera an ninh (CCTV) mờ nhòe bão hạt của các vụ cháy thật ở quán Karaoke, nhà dân trong đêm.",
        "labels": ["epic-1-data", "todo"]
    },
    
    # EPIC 2: MODEL RESEARCH
    {
        "title": "🔬 [GĐ 2] Model - Train Baseline (Huấn luyện Model Mốc YOLO)",
        "body": "### 🎯 Chỉ tiêu: YOLOv8 / YOLOv10\n- **Nhiệm vụ:** Chạy huấn luyện 100 Epochs trên phiên bản Model CNN truyền thống để lấy mốc (Baseline) so sánh.\n- **Hành động:** Chụp lại các mốc mAP50, F1-Score vào Excel.",
        "labels": ["epic-2-model", "todo"]
    },
    {
        "title": "🔬 [GĐ 2] Model - Huấn luyện RT-DETR Chính (Transformer)",
        "body": "### 🎯 Chỉ tiêu: RT-DETR-L 150-200 Epochs\n- **Nhiệm vụ:** Vắt kiệt sức mạnh Attention của Transformer trên 11,000 Data đã cào.\n- **Hành động:** Theo dõi đồ thị Loss, ép Model chạy để cán mốc > 90% mAP. Thực hiện **Ablation Study**: Chạy 1 bản có Ảnh lừa, 1 bản không có ảnh Lừa để lập biểu đồ đối chứng Khóa luận.",
        "labels": ["epic-2-model", "todo", "high-priority"]
    },
    
    # EPIC 3: PRODUCTION DEPLOYMENT
    {
        "title": "🚀 [GĐ 3] Deploy - Tối ưu hóa trọng lượng mô hình (TensorRT/ONNX)",
        "body": "### 🎯 Chỉ tiêu: 60 FPS Real-Time\n- **Nhiệm vụ:** Transformer rất NẶNG. Bạn cần dịch mã gốc file `.pt` (66MB) sang TensorRT (`.engine`) hoặc ONNX (Sử dụng Float16).\n- **Kỳ vọng:** Phải chạy mượt Camera luồng mà không bị giật lag.",
        "labels": ["epic-3-deploy", "todo"]
    },
    {
        "title": "🚀 [GĐ 3] Deploy - Backend Cảnh báo Streaming Hệ thống",
        "body": "### 🎯 Chỉ tiêu: Debounce Logic \n- **Nhiệm vụ:** Viết Python lấy trực tiếp luồng Camera. Chặn Báo giả cực mạnh bằng logic: Cháy liên tục 30 Frames (1 Giây) với Độ Tin Cậy > 0.8 MỚI ĐƯỢC CHỐT.\n- **Thông báo:** Gửi hình ảnh Crop cận cảnh ngọn lửa qua Zalo/Discord/Telegram.",
        "labels": ["epic-3-deploy", "todo"]
    }
]

if __name__ == "__main__":
    print("🚀 Bắt đầu tự động tạo Agile Tickets trên GitHub Issues...")
    for t in tickets:
        create_issue(t["title"], t["body"], t["labels"])
    print("🎉 HOÀN TẤT! Bạn hãy vào mục Issues/Projects trên Github để quản lý Board nhiệm vụ nhé!")
