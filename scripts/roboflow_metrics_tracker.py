import os
import requests
from datetime import datetime
from roboflow import Roboflow

# ==========================================
# MLOps Dashboard: Multi-Dataset Tracker
# ==========================================

RF_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
WORKSPACE = os.environ.get("ROBOFLOW_WORKSPACE", "springwangs-workspace")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO = os.environ.get("GITHUB_REPOSITORY")

def get_all_projects_stats():
    """Lấy dữ liệu của TẤT CẢ các dự án nằm trong Workspace."""
    url = f"https://api.roboflow.com/{WORKSPACE}?api_key={RF_API_KEY}"
    resp = requests.get(url)
    resp.raise_for_status()
    # Móc ra mảng projects chứa danh sách vô số các dataset
    return resp.json().get("workspace", {}).get("projects", [])

def send_zalo_alert(message):
    """Gửi cảnh báo qua nền tảng Zalo OA nếu có cài đặt."""
    access_token = os.environ.get("ZALO_ACCESS_TOKEN")
    user_id = os.environ.get("ZALO_USER_ID")
    if access_token and user_id:
        try:
            url = "https://openapi.zalo.me/v3.0/oa/message/text"
            headers = {
                "access_token": access_token,
                "Content-Type": "application/json"
            }
            payload = {
                "recipient": {"user_id": user_id},
                "message": {"text": message}
            }
            requests.post(url, headers=headers, json=payload)
        except Exception as e:
            print("Zalo send error:", e)

def get_or_create_issue():
    """Tìm Bảng tin (Issue) Dashboard, nếu chưa có thì tạo mới."""
    url = f"https://api.github.com/repos/{REPO}/issues"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}", 
        "Accept": "application/vnd.github.v3+json"
    }
    
    resp = requests.get(url + "?state=open&labels=mlops-dashboard", headers=headers)
    issues = resp.json()
    if issues:
        return issues[0]["number"]
    
    payload = {
        "title": "📊 MLOps: Báo cáo Tiến độ Gán nhãn Tổng hợp (Multi-Datasets)",
        "body": "Bảng giám sát trực quan tự động đếm tiến trình cho TẤT CẢ các Datasets trong Workspace Roboflow của team. Được cập nhật bởi GitHub Actions hằng ngày.",
        "labels": ["mlops-dashboard"]
    }
    resp = requests.post(url, headers=headers, json=payload)
    return resp.json()["number"]

def post_comment(issue_number, body):
    """Bình luận báo cáo vào bảng Issue."""
    url = f"https://api.github.com/repos/{REPO}/issues/{issue_number}/comments"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}", 
        "Accept": "application/vnd.github.v3+json"
    }
    resp = requests.post(url, headers=headers, json={"body": body})
    resp.raise_for_status()

def create_progress_bar(percentage, length=25):
    """Vẽ thanh trạng thái biểu đồ Ascii cực đẹp."""
    filled = int(length * percentage // 100)
    bar = '█' * filled + '░' * (length - filled)
    return f"`[{bar}] {percentage:.1f}%`"

def main():
    if not RF_API_KEY:
        print("LỖI: Chưa cấu hình Secret ROBOFLOW_API_KEY trên GitHub.")
        return

    print("Đang cào dữ liệu toàn bộ Workspace từ Roboflow API...")
    try:
        projects = get_all_projects_stats()
    except Exception as e:
        print(f"Lỗi nối API: {e}")
        return
    
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    
    # Soạn thảo Format chuyên nghiệp
    report_lines = [f"## 🎯 Cập nhật Ngày {now}\n"]
    report_lines.append(f"**Tổng số Datasets đang theo dõi:** {len(projects)}")
    report_lines.append("---\n")
    
    for proj in projects:
        name = proj.get("name", "Unknown Project")
        total_images = proj.get("images", 0)
        unannotated = proj.get("unannotated", 0)
        annotated = total_images - unannotated
        percentage = (annotated / total_images * 100) if total_images > 0 else 0
        
        # Biểu đồ thanh tiến trình
        progress_bar = create_progress_bar(percentage)
        
        # Thống kê Classes và Cảnh báo MẤT CÂN BẰNG DATA
        classes = proj.get("classes", {})
        classes_str = ", ".join([f"**{k}**: {v}" for k, v in classes.items()]) if classes else "Chưa phân loại"
        
        warnings_str = ""
        total_instances = sum(classes.values()) if classes else 0
        if total_instances > 0:
            for cls_name, count in classes.items():
                ratio = count / total_instances
                if ratio < 0.15: # Nếu nhỏ hơn 15% tổng data
                    warnings_str += f"\n- ⚠️ **CẢNH BÁO MẤT CÂN BẰNG**: Nhãn `{cls_name}` chiếm tỷ trọng quá thấp ({ratio:.1%}). Cần bổ sung thêm ảnh chứa đối tượng này để model không bị hội tụ lệch (Bias)!"
        
        # Nếu đạt 100% (Tiến hành Auto-Generate Version Tự Động!)
        if percentage == 100 and unannotated == 0 and total_images > 0:
            warnings_str += f"\n- 🚀 **HOÀN THÀNH 100%**: Tập dữ liệu đã sẵn sàng."
            
            # Gửi lệnh API Đóng Gói (Generate Version)
            project_id_slug = proj.get("id", "").split("/")[-1]
            if project_id_slug:
                try:
                    rf = Roboflow(api_key=RF_API_KEY)
                    rf_proj = rf.workspace(WORKSPACE).project(project_id_slug)
                    
                    print(f"Đang gửi lệnh kích hoạt Đóng gói Version cho {name}...")
                    # Generate dựa trên config default (512x512) có sẵn của workspace
                    new_version = rf_proj.generate_version(settings={})
                    v_num = new_version.version
                    warnings_str += f" Bot MLOps đã GỬI LỆNH THÀNH CÔNG đúc ra phiên bản **Version {v_num}** trên máy chủ Roboflow! Bạn có thể sử dụng tính năng Export để tải về."
                except Exception as e:
                    err_msg = str(e).lower()
                    if "no changes" in err_msg or "identical" in err_msg:
                        warnings_str += f" Phiên bản chót đã được đóng gói trước đó, không có ảnh/nhãn nào mới được thêm vào nên Bot không sinh thêm Version rác."
                    else:
                        warnings_str += f" Thử kích hoạt đúc Version lỗi: {e}"

        # Markdown Block cho mỗi project
        proj_report = f"""### 📁 Tiêu điểm: **{name}**
{progress_bar}
- 🟢 **Hoàn thành:** {annotated} / {total_images} ảnh
- 🔴 **Tồn đọng:** {unannotated} ảnh chờ xử lý
- 🏷 **Nhãn dữ liệu (Classes):** {classes_str}{warnings_str}
"""
        report_lines.append(proj_report)
    
    final_report = "\n".join(report_lines)
    
    print("Truy tìm Bảng tin Dashboard GitHub Issues...")
    issue_number = get_or_create_issue()
    
    print(f"Bắn báo cáo vào Issue #{issue_number}...")
    post_comment(issue_number, final_report)
    
    # Bắn Zalo sau khi xong (Nếu có cài đặt Bot)
    short_summary = f"🤖 [MLOps Bot] Cập nhật tiến độ Team:\nĐã thẩm định {len(projects)} dự án dataset. Vui lòng vào tab Issues trên GitHub để đọc báo cáo Cảnh báo Mất Cân Bằng Data (nếu có) nhé Xuân Thành và Quang!"
    send_zalo_alert(short_summary)
    
    print("🎉 Hoàn tất MLOps Pipeline! Bảng tin đã được dán lên Issue.")

if __name__ == "__main__":
    main()
