import os
import requests
from datetime import datetime

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO = os.environ.get("GITHUB_REPOSITORY")
WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL")

def get_latest_issue_report():
    url = f"https://api.github.com/repos/{REPO}/issues"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    resp = requests.get(url + "?state=open&labels=mlops-dashboard", headers=headers)
    issues = resp.json()
    if not issues:
        return None
    
    issue_number = issues[0]["number"]
    
    # Lấy danh sách Comments của Bảng Báo Cáo
    comments_url = f"https://api.github.com/repos/{REPO}/issues/{issue_number}/comments"
    resp = requests.get(comments_url, headers=headers)
    comments = resp.json()
    if not comments:
        return None
    
    # Rút trích Dòng trạng thái (Comment) mới nhất do MLOps Bot viết
    latest_comment = comments[-1]["body"]
    return latest_comment

def send_discord_ping(targets):
    if not WEBHOOK or not targets: return
    
    mentions = " ".join(targets)
    msg = f"🦉 **[CÚ XANH DUOLINGO]** Éc éc! Tới giờ kiểm điểm!\nCác đồng chí {mentions} chưa hoàn thành KPI 100 ảnh ngày hôm nay! Mau rủ nhau vào Roboflow gán nhãn, gán xong nhớ vào Github Issue bấm ✅ (Tick) chấm công kẻo Cú mổ nhé!"
    payload = {"content": msg, "username": "Duolingo Cú Xanh 🦉", "avatar_url": "https://www.duolingo.com/images/facebook/duo200.png"}
    requests.post(WEBHOOK, json=payload)

def main():
    body = get_latest_issue_report()
    if not body:
        print("Không có báo cáo nào để kiểm tra.")
        return
        
    # Máy soi kĩ thuật kiểm tra ô vuông
    slackers = []
    
    # Nếu ô vuông chưa có dấu [x]
    if "- [ ] @springwang_08" in body:
        slackers.append("@springwang_08")
    if "- [ ] @hoangxuanthanh2811" in body:
        slackers.append("@hoangxuanthanh2811")
        
    if slackers:
        print(f"Phát hiện trốn việc: {slackers}. Gửi Cú Xanh gõ đầu...")
        send_discord_ping(slackers)
    else:
        print("Ai cũng đã tick xong nhiệm vụ. Cú Xanh đi ngủ hiền hoà.")

if __name__ == "__main__":
    main()
