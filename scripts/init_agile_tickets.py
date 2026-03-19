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
    # EPIC 1: DATA ENGINEERING
    {
        "title": "🎫 [GĐ 1] Data - Positive Standard (Ảnh Lửa/Khói Rõ Nét)",
        "body": "### 🎯 Chỉ tiêu (KPI): 2,500 Ảnh\n- **Thư mục:** `data/01_Positive_Standard`\n- **Nhiệm vụ:** Thu thập hình ảnh cháy nổ cơ bản để model học hình thái gốc của ngọn lửa và cột khói.\n- **Hành động:** Nạp lên Roboflow, khoanh Bounding Box (Lửa, Khói) thật sát viền.",
        "labels": ["epic-1-data", "todo"]
    },
    {
        "title": "🎫 [GĐ 1] Data - Alley Context (Bối cảnh Hẻm/Chung cư TPHCM)",
        "body": "### 🎯 Chỉ tiêu (KPI TỐI THIỂU): 1,000 Ảnh\n- **Thư mục:** `data/02_Alley_Context`\n- **Nhiệm vụ:** Đây là vùng không gian cốt lõi của Đồ án. Giúp chống điểm mù khi lửa cháy chìm trong rừng dây điện và ban công hẹp.\n- **Hành động:** Tìm ảnh các con hẻm, cắt ghép (Photoshop) ngọn lửa vào các ổ điện, nạp lên Roboflow và khoanh Box.",
        "labels": ["epic-1-data", "todo"]
    },
    {
        "title": "🎫 [GĐ 1] Data - Hard Negatives (Chống Báo Giả / Sai số)",
        "body": "### 🎯 Chỉ tiêu (KPI BẢO VỆ ĐIỂM 9): 1,000 Ảnh\n- **Thư mục:** `data/03_Negative_Hard_Samples`\n- **Nhiệm vụ:** Giải quyết nhược điểm AI hay nhận diện nhầm Sương mù thành Khói, Hoàng hôn/Đèn đỏ thành Lửa.\n- **Hành động:** Thu thập ảnh Sương mù Đà Lạt, hoàng hôn, còi xe, khói bún bò. **TUYỆT ĐỐI KHÔNG KHOANH BOX (Lưu File Txt rỗng)**.",
        "labels": ["epic-1-data", "todo"]
    },
    {
        "title": "🎫 [GĐ 1] Data - SAHI Small Objects (Lửa đốm siêu nhỏ xa)",
        "body": "### 🎯 Chỉ tiêu (KPI CỘNG ĐIỂM): 500 Ảnh\n- **Thư mục:** `data/04_SAHI_Small_Objects`\n- **Nhiệm vụ:** Giải quyết điểm mù khi lửa cháy từ khoảng cách Flycam/Camera chung cư chọc trời, quá nhỏ so với khung hình 640x640.\n- **Hành động:** Thu thập ảnh có đốm lửa chiếm dới 3% diện tích ảnh.",
        "labels": ["epic-1-data", "todo"]
    },
    {
        "title": "🎫 [GĐ 1] Data - Real Situation (Tập Kiểm định Thực Tế)",
        "body": "### 🎯 Chỉ tiêu (TẬP TEST): 500 Ảnh\n- **Thư mục:** `data/05_Real_Situation`\n- **Nhiệm vụ:** Nguồn Test mù cho biểu đồ mAP trong tiểu luận.\n- **Hành động:** Cắt hình từ báo đài thời sự các vụ cháy thật sự.",
        "labels": ["epic-1-data", "todo"]
    },
    
    # EPIC 2: MODEL RESEARCH
    {
        "title": "🔬 [GĐ 2] Model - Train Baseline (Huấn luyện Model Mẫu so sánh)",
        "body": "### 🎯 Chỉ tiêu: YOLOv8 / YOLOv10\n- **Nhiệm vụ:** Chạy huấn luyện 100 Epochs trên phiên bản Model nhẹ (CNN) để lấy mốc thông số (Baseline) so sánh.\n- **Hành động:** Lưu mAP50, F1-Score vào Excel.",
        "labels": ["epic-2-model", "todo"]
    },
    {
        "title": "🔬 [GĐ 2] Model - Train Transformer (Huấn luyện RT-DETR Chính)",
        "body": "### 🎯 Chỉ tiêu: RT-DETR-L 150 Epochs\n- **Nhiệm vụ:** Dùng sức mạnh Self-Attention của Transformer trên trọn bộ 5000+ data.\n- **Hành động:** Theo dõi đồ thị Loss hội tụ, đánh giá khả năng bao quát bối cảnh so với YOLO.",
        "labels": ["epic-2-model", "todo", "high-priority"]
    },
    
    # EPIC 3: PRODUCTION DEPLOYMENT
    {
        "title": "🚀 [GĐ 3] Deploy - Tối ưu hóa trọng lượng mô hình (ONNX/TensorRT)",
        "body": "### 🎯 Chỉ tiêu: > 60 FPS Realtime\n- **Nhiệm vụ:** Chuyển đổi file `.pt` (66MB) sang định dạng nhẹ TensorRT (`.engine`) hoặc ONNX.\n- **Kỳ vọng:** Xác thực Camera stream chạy mượt trên các thiết bị Edge.",
        "labels": ["epic-3-deploy", "todo"]
    },
    {
        "title": "🚀 [GĐ 3] Deploy - Xây dựng Backend Cảnh báo Streaming",
        "body": "### 🎯 Chỉ tiêu: Chống nhiễu Debounce 30 Frames\n- **Nhiệm vụ:** Viết Python FastAPI hứng Camera Hẻm. Áp dụng Logic: Cháy 30 Frames liên tiếp với Độ Tin Cậy > 0.8 mới gửi tín hiệu báo hỏa hoạn qua Zalo/Discord.\n- **Đóng gói:** Triển khai thử nghiệm.",
        "labels": ["epic-3-deploy", "todo"]
    }
]

if __name__ == "__main__":
    print("🚀 Bắt đầu tự động tạo Agile Tickets trên GitHub Issues...")
    for t in tickets:
        create_issue(t["title"], t["body"], t["labels"])
    print("🎉 HOÀN TẤT! Bạn hãy vào mục Issues/Projects trên Github để quản lý Board nhiệm vụ nhé!")
