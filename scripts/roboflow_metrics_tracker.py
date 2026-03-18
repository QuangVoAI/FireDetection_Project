import os
import requests
import json
from datetime import datetime

# ==========================================
# MLOps Dashboard: Dataset Progress Tracker
# ==========================================

RF_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
WORKSPACE = os.environ.get("ROBOFLOW_WORKSPACE")
PROJECT = os.environ.get("ROBOFLOW_PROJECT")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO = os.environ.get("GITHUB_REPOSITORY")

def get_roboflow_stats():
    """Lấy dữ liệu số lượng ảnh, classes từ dự án."""
    url = f"https://api.roboflow.com/{WORKSPACE}/{PROJECT}?api_key={RF_API_KEY}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json().get("project", {})
    return data

def get_or_create_issue():
    """Tìm Bảng tin (Issue) Dashboard, nếu chưa có thì tạo mới."""
    url = f"https://api.github.com/repos/{REPO}/issues"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}", 
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Truy vấn xem đã có Issue nào dán nhãn mlops-dashboard chưa
    resp = requests.get(url + "?state=open&labels=mlops-dashboard", headers=headers)
    issues = resp.json()
    if issues:
        return issues[0]["number"]
    
    # Chế tạo Issue mới keng
    payload = {
        "title": "📊 Báo cáo Tiến độ Gán nhãn Dataset (Roboflow)",
        "body": "Issue này được sinh ra tự động bởi MLOps Bot. Mỗi ngày bot sẽ tự động điểm danh tiến độ cày cuốc của team (SpringWang & Thành) vào bên dưới bình luận để tránh làm rác nhánh Code `main`.",
        "labels": ["mlops-dashboard"]
    }
    resp = requests.post(url, headers=headers, json=payload)
    return resp.json()["number"]

def post_comment(issue_number, body):
    """Bình luận tiến độ vào bảng Issue."""
    url = f"https://api.github.com/repos/{REPO}/issues/{issue_number}/comments"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}", 
        "Accept": "application/vnd.github.v3+json"
    }
    resp = requests.post(url, headers=headers, json={"body": body})
    resp.raise_for_status()

def main():
    if not RF_API_KEY:
        print("LỖI: Chưa cấu hình Secret ROBOFLOW_API_KEY trên GitHub.")
        return

    print("Đang cào dữ liệu từ Roboflow API của team...")
    try:
        stats = get_roboflow_stats()
    except Exception as e:
        print(f"Lỗi nối API: {e}")
        return
    
    total_images = stats.get("images", 0)
    unannotated = stats.get("unannotated", 0)
    annotated = total_images - unannotated
    percentage = (annotated / total_images * 100) if total_images > 0 else 0
    
    classes = stats.get("classes", {})
    fire_count = classes.get("fire", 0)
    smoke_count = classes.get("smoke", 0)
    
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    
    # Soạn văn bản báo cáo chuyên nghiệp
    report = f"""### 🕒 Cập nhật ngày {now}

**Tiến độ Dự án (Fire Detection):**
Toàn bộ dự án đang có vướng bận **{total_images}** ảnh.
- 🟢 Đã gán nhãn: **{annotated}** ảnh ({percentage:.1f}% hoàn thành)
- 🔴 Chưa gán nhãn (Tồn đọng): **{unannotated}** ảnh cần team xử lý tiếp.

**Thống kê Kho Bounding Box:**
- 🔥 Lửa (Fire): `{fire_count}` boxes
- 💨 Khói (Smoke): `{smoke_count}` boxes

*(Note: Vì gói API miễn phí của Roboflow không phân tích chẻ nhỏ được số liệu của User gán nhãn, tiến độ trên là hiệu suất đếm lùi tổng cộng mồ hôi công sức của cả team SpringWang và Thanh281105).* 
"""
    
    print("Truy tìm Bảng tin Dashboard GitHub Issues...")
    issue_number = get_or_create_issue()
    
    print(f"Bắn báo cáo vào Issue #{issue_number}...")
    post_comment(issue_number, report)
    print("🎉 Hoàn tất MLOps Pipeline! Bảng tin đã được dán lên Issue.")

if __name__ == "__main__":
    main()
