import os
import requests
from datetime import datetime, timedelta, timezone

# ==========================================================
# MLOps Dashboard: ROBOFLOW API DATASET TRACKER
# ==========================================================
# Bật lại chế độ gọi API: Cào trực tiếp dữ liệu từ toàn bộ Workspace Roboflow

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO = os.environ.get("GITHUB_REPOSITORY")
DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL")

# Đọc API Key từ Secret của Github
RF_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
WORKSPACE = os.environ.get("ROBOFLOW_WORKSPACE", "springwangs-workspace")

def get_roboflow_projects():
    """Lấy dữ liệu của TẤT CẢ các dự án nằm trong Workspace."""
    url = f"https://api.roboflow.com/{WORKSPACE}?api_key={RF_API_KEY}"
    resp = requests.get(url)
    resp.raise_for_status()
    # Móc mảng 'projects' chứa danh sách các tập dữ liệu
    return resp.json().get("workspace", {}).get("projects", [])

def get_or_create_issue():
    """Tìm Bảng tin (Issue) Dashboard, nếu chưa có thì tạo mới."""
    url = f"https://api.github.com/repos/{REPO}/issues"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}", 
        "Accept": "application/vnd.github.v3+json"
    }
    
    resp = requests.get(url + "?state=open&labels=mlops-dashboard", headers=headers)
    issues = resp.json()
    if isinstance(issues, list) and issues:
        return issues[0]["number"]
    
    payload = {
        "title": "📊 MLOps: Báo cáo Tiến độ Gán nhãn Tổng hợp (Multi-Datasets)",
        "body": "Bảng giám sát trực quan tự động đếm tiến trình cho TẤT CẢ các Datasets trong Workspace Roboflow của team thông qua khóa API chính thức. Được cập nhật bởi GitHub Actions hằng ngày.",
        "labels": ["mlops-dashboard"]
    }
    resp = requests.post(url, headers=headers, json=payload)
    return resp.json()["number"]

def main():
    if not RF_API_KEY:
        print("LỖI: Chưa cấu hình Secret ROBOFLOW_API_KEY trên nhánh GitHub.")
        return

    print("🚀 Đang rà quét và cào dữ liệu từ Roboflow API của team...")
    try:
        projects = get_roboflow_projects()
    except Exception as e:
        print(f"Lỗi nối API Roboflow: {e}")
        return

    now = (datetime.now(timezone.utc) + timedelta(hours=7)).strftime("%d/%m/%Y %H:%M:%S")
    report_lines = [f"## 🛠️ Dashboard Dữ liệu: Roboflow API - {now}\n"]
    report_lines.append("> [!TIP]\n> Hệ thống đã được Mở Khóa chuyển sang chế độ **Cloud API Tracking**. Bot sẽ cào trực tiếp dữ liệu chuẩn xác nhất từ Server Web Roboflow của nhóm!\n")
    
    report_lines.append("### 🦉 Nhiệm vụ KPI Hôm nay:")
    report_lines.append("- [ ] @springwang_08 (150 Ảnh)")
    report_lines.append("- [ ] @hoangxuanthanh2811 (150 Ảnh)\n")
    
    total_images_all = 0
    total_annotated_all = 0
    
    for proj in projects:
        name = proj.get("name", "Unknown Project")
        total_images = proj.get("images", 0)
        unannotated = proj.get("unannotated", 0)
        annotated = total_images - unannotated
        pct = (annotated / total_images * 100) if total_images > 0 else 0
        
        total_images_all += total_images
        total_annotated_all += annotated
        
        # Thống kê Classes
        classes = proj.get("classes", {})
        classes_str = ", ".join([f"**{k}**: {v}" for k, v in classes.items()]) if classes else "Chưa phân loại/Chưa gán nhãn"
        
        report_lines.append(f"#### 🌩️ Project: {name}")
        report_lines.append(f"- **Tiến độ:** {annotated} / {total_images} ảnh ({pct:.1f}%) | *Tồn đọng chờ duyệt: {unannotated} ảnh*")
        report_lines.append(f"- **Chi tiết Bounding Box:** {classes_str}")
        if pct == 100 and total_images > 0:
            report_lines.append("- 🚀 *Ready for Export Version (Đã Hoàn Thành 100%)*")
        report_lines.append("")

    summary_pct = (total_annotated_all / total_images_all * 100) if total_images_all > 0 else 0
    report_lines.insert(1, f"### 📈 Tổng tiến độ Cloud Workspace: **{total_annotated_all}/{total_images_all}** ({summary_pct:.1f}%)\n")
    
    final_body = "\n".join(report_lines)
    
    print("Truy tìm Bảng tin Dashboard GitHub Issues...")
    issue_num = get_or_create_issue()
    
    print(f"Bắn báo cáo vào Issue #{issue_num}...")
    url = f"https://api.github.com/repos/{REPO}/issues/{issue_num}/comments"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    requests.post(url, headers=headers, json={"body": final_body})
    
    # Kích hoạt bắn tin nhắn sang Zalo / Discord
    if DISCORD_WEBHOOK:
        discord_msg = final_body + f"\n\n👉 **Link Xem Báo Cáo:** https://github.com/{REPO}/issues/{issue_num}"
        discord_msg = discord_msg.replace("@springwang_08", "<@770639864760631296>")
        discord_msg = discord_msg.replace("@hoangxuanthanh2811", "<@1256982686145183785>")
        try:
            requests.post(DISCORD_WEBHOOK, json={"content": discord_msg, "username": "API MLOps Bot 🌩️"})
        except:
            pass

    print(f"✅ Đã cập nhật Dashboard API! Tổng Server đang có {total_annotated_all}/{total_images_all} ảnh đã gán trọn vẹn.")

if __name__ == "__main__":
    main()
