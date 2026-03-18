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

def send_discord_alert(message):
    """Gửi cảnh báo qua nền tảng Discord (Siêu dễ và nhanh)."""
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    if webhook_url:
        try:
            payload = {
                "content": message,
                "username": "MLOps Bot 🔥"
            }
            requests.post(webhook_url, json=payload)
        except Exception as e:
            print("Discord send error:", e)

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
    
    report_lines.append(f"### 🦉 Nhiệm vụ (KPI) Ngày {now.split(' ')[0]}:")
    report_lines.append("Mỗi sếp gán nhãn ít nhất **100 ảnh**. Xong việc thì tick vào ô vuông bên dưới để Cú Xanh Duolingo tha mạng:\n")
    report_lines.append("- [ ] @springwang_08")
    report_lines.append("- [ ] @hoangxuanthanh2811")
    report_lines.append("---\n")
    
    report_lines.append(f"**Tổng số Datasets đang theo dõi:** {len(projects)}")
    report_lines.append("---\n")
    
    for proj in projects:
        name = proj.get("name", "Unknown Project")
        total_images = proj.get("images", 0)
        unannotated = proj.get("unannotated", 0)
        annotated = total_images - unannotated
        percentage = (annotated / total_images * 100) if total_images > 0 else 0
        
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
                    # Giải quyết triệt để lỗi thiếu dictionary dict() của Roboflow SDK
                    new_version = rf_proj.generate_version(settings={"augmentation": {}, "preprocessing": {}})
                    v_num = new_version.version
                    warnings_str += f"\n- Bot MLOps đã GỬI LỆNH THÀNH CÔNG đúc ra phiên bản **Version {v_num}** trên máy chủ Roboflow!\n"
                    
                    # Tự động thực thi Luồng 2 (Export & Zip) ngay tại đây
                    print(f"🤖 100% Kích hoạt cơ chế Clone! Tải Version {v_num} chuẩn YOLOv8 về kho...")
                    warnings_str += f" - 📦 **Tự động trích xuất:** Đang tải dataset rễ YOLOv8 về Github Actions Artifacts..."
                    folder_name = f"dataset_{project_id_slug}_v{v_num}"
                    new_version.download("yolov8", location=folder_name)
                    os.system(f"zip -r {folder_name}.zip {folder_name}/")
                    warnings_str += f" ✅ Đã nén ZIP và niêm phong thành công tệp gốc vào kho Github!"
                    print(f"✅ Nén xong ZIP: {folder_name}.zip")
                    
                except Exception as e:
                    err_msg = str(e).lower()
                    if "no changes" in err_msg or "identical" in err_msg:
                        warnings_str += f" Phiên bản chót đã được đóng gói trước đó, không có ảnh/nhãn nào mới được thêm vào nên Bot không sinh thêm Version rác."
                    else:
                        warnings_str += f" Thử kích hoạt đúc Version lỗi: {e}"

        # Markdown Block cho mỗi project
        proj_report = f"""### 📁 Tiêu điểm: **{name}**
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
    
    # Bắn Discord sau khi xong (Nếu có kênh)
    discord_msg = final_report + "\n\n👉 *Vào Github Issues tick xanh để hoàn thành chấm công nhé!*"
    send_discord_alert(discord_msg)
    
    print("🎉 Hoàn tất MLOps Pipeline! Bảng tin đã được dán lên Issue.")

if __name__ == "__main__":
    main()
