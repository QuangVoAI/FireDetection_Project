import os
import requests
from pathlib import Path
from datetime import datetime, timedelta, timezone

# ==========================================================
# MLOps Dashboard: LOCAL DATASET TRACKER (Free Migration)
# ==========================================================
# Hệ thống này không dùng API Roboflow nữa (để tiết kiệm Credit)
# Bot sẽ quét trực tiếp file nhãn .txt trong Folder Repo của sếp.

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO = os.environ.get("GITHUB_REPOSITORY")
DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL")

# Cấu hình KPI Khóa luận (Denominator)
DATASET_KPIS = {
    "01_Positive_Standard": 4000,
    "02_Alley_Context": 3000,
    "03_Negative_Hard_Samples": 2000,
    "04_SAHI_Small_Objects": 1000,
    "05_Real_Situation": 1000
}

CLASS_MAP = {0: "Lửa (Fire)", 1: "Khói (Smoke)"}

def get_or_create_issue():
    """Tìm Bảng tin Dashboard GitHub, nếu chưa có thì tạo mới."""
    url = f"https://api.github.com/repos/{REPO}/issues"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    resp = requests.get(url + "?state=open&labels=mlops-dashboard", headers=headers)
    issues = resp.json()
    if isinstance(issues, list) and issues:
        return issues[0]["number"]
    
    payload = {
        "title": "📊 MLOps Dashboard: Giám sát Dữ liệu Nội bộ (Local Data)",
        "body": "Bảng tin tự động theo dõi tiến độ gán nhãn dựa trên file .txt trong Repo. Ảnh gốc được lưu an toàn trên DVC.",
        "labels": ["mlops-dashboard"]
    }
    resp = requests.post(url, headers=headers, json=payload)
    return resp.json()["number"]

def scan_local_data():
    """Quét thư mục data/ để hạch toán tiến độ."""
    stats = {}
    data_path = Path("data")
    
    for folder, target in DATASET_KPIS.items():
        label_path = data_path / folder / "labels"
        # Loại trừ classes.txt khỏi danh sách file đếm (để không bị đếm nhầm là 1 ảnh)
        if label_path.exists():
            annotated_files = [f for f in label_path.glob("*.txt") if f.name != "classes.txt"]
        else:
            annotated_files = []
        
        # Đếm chi tiết Class (Class Balance)
        class_counts = {0: 0, 1: 0}
        for f in annotated_files:
            try:
                content = f.read_text().strip()
                if content:
                    for line in content.split("\n"):
                        cls_id = int(line.split()[0])
                        if cls_id in class_counts:
                            class_counts[cls_id] += 1
            except: pass
            
        stats[folder] = {
            "annotated": len(annotated_files),
            "target": target,
            "classes": class_counts
        }
    return stats

def main():
    print("🚀 Bắt đầu hạch toán dữ liệu Local...")
    stats = scan_local_data()
    
    now = (datetime.now(timezone.utc) + timedelta(hours=7)).strftime("%d/%m/%Y %H:%M:%S")
    
    report = [f"## 🛠️ Dashboard Dữ liệu Local - {now}\n"]
    report.append("> [!TIP]\n> Hệ thống đã chuyển sang chế độ **Local Tracking**. Bot sẽ đếm các tệp `.txt` đã được sếp Push lên GitHub.\n")
    
    report.append("### 🦉 Nhiệm vụ KPI Hôm nay:")
    report.append("- [ ] @springwang_08 (150 Ảnh)")
    report.append("- [ ] @hoangxuanthanh2811 (150 Ảnh)\n")
    
    total_ann = 0
    total_goal = sum(DATASET_KPIS.values())
    
    for name, data in stats.items():
        ann = data["annotated"]
        goal = data["target"]
        total_ann += ann
        pct = (ann / goal * 100) if goal > 0 else 0
        
        class_info = ", ".join([f"**{CLASS_MAP[k]}**: {v}" for k, v in data["classes"].items()])
        
        report.append(f"#### 📂 {name}")
        report.append(f"- **Tiến độ:** {ann} / {goal} ({pct:.1f}%)")
        report.append(f"- **Chi tiết nhãn:** {class_info}")
        if pct == 100: report.append("- 🚀 *Ready for DVC Tagging*")
        report.append("")

    summary_pct = (total_ann / total_goal * 100)
    report.insert(1, f"### 📈 Tổng tiến độ Khóa luận: **{total_ann}/{total_goal}** ({summary_pct:.1f}%)\n")
    
    final_body = "\n".join(report)
    
    # 1. Update GitHub
    issue_num = get_or_create_issue()
    url = f"https://api.github.com/repos/{REPO}/issues/{issue_num}/comments"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    requests.post(url, headers=headers, json={"body": final_body})
    
    # 2. Update Discord
    if DISCORD_WEBHOOK:
        discord_msg = final_body + f"\n\n👉 **Link Dashboard:** https://github.com/{REPO}/issues/{issue_num}"
        discord_msg = discord_msg.replace("@springwang_08", "<@770639864760631296>")
        discord_msg = discord_msg.replace("@hoangxuanthanh2811", "<@1256982686145183785>")
        requests.post(DISCORD_WEBHOOK, json={"content": discord_msg, "username": "Local MLOps Bot 📂"})

    print(f"✅ Đã cập nhật Dashboard Local! (Tổng: {total_ann} ảnh)")

if __name__ == "__main__":
    main()
