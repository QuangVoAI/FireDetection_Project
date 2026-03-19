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
    
    # GitHub API có thể trả về Error Dictionary thay vì List array (VD: Rate limit)
    if not isinstance(issues, list):
        print(f"Lỗi API từ Github: {issues}")
        return None
        
    if not issues:
        return None
    
    latest_issue = issues[0]["number"]
    
    # Lấy danh sách Comments của Bảng Báo Cáo
    comments_url = f"https://api.github.com/repos/{REPO}/issues/{latest_issue}/comments"
    resp = requests.get(comments_url, headers=headers)
    comments = resp.json()
    if not comments:
        return None
    
    # Rút trích Dòng trạng thái (Comment) mới nhất do MLOps Bot viết
    latest_comment = comments[-1]["body"]
    return latest_comment, latest_issue

def send_discord_ping(targets, issue_number):
    if not WEBHOOK or not targets: return
    
    mentions = " ".join(targets)
    # Bỏ mặt nạ Github, lột xác thành Discord Tags để réo điện thoại
    mentions = mentions.replace("@springwang_08", "<@770639864760631296>")
    mentions = mentions.replace("@hoangxuanthanh2811", "<@1256982686145183785>")
    
    msg = f"🦉 **[CÚ XANH DUOLINGO]** Éc éc! Trễ deadline khóa luận rồi!\nCác đồng chí {mentions} chưa hoàn thành KPI **150 ảnh** ngày hôm nay!\n\n🏃‍♂️ **CÁCH TRỐN KHỎI CÚ XANH (Chiêu lười):**\n1. Mở web Roboflow gán đủ số lượng (Đừng quên nạp Ảnh Lừa chống False Positive).\n2. Bấm Link này vào thẳng Github: https://github.com/{REPO}/issues/{issue_number}\n3. Lấy chuột **NHẤN VÀO CÁI Ô VUÔNG** cạnh tên bạn cho nó hiện dấu Check ✅ (Hệ thống sẽ tự Save). Xong!"
    payload = {"content": msg, "username": "Duolingo Cú Xanh 🦉", "avatar_url": "https://www.duolingo.com/images/facebook/duo200.png"}
    
    if WEBHOOK and WEBHOOK.startswith("http"):
        try:
            requests.post(WEBHOOK, json=payload)
        except Exception as e:
            print("Lỗi kết nối Webhook Discord:", e)
    else:
        print("CẢNH BÁO: Chưa cấu hình DISCORD_WEBHOOK_URL hợp lệ, không thể réo tên!")

def main():
    result = get_latest_issue_report()
    if not result:
        print("Không có báo cáo nào để kiểm tra.")
        return
        
    body, issue_number = result
        
    # Máy soi kĩ thuật kiểm tra ô vuông
    slackers = []
    
    # Nếu ô vuông chưa có dấu [x]
    if "- [ ] @springwang_08" in body:
        slackers.append("@springwang_08")
    if "- [ ] @hoangxuanthanh2811" in body:
        slackers.append("@hoangxuanthanh2811")
        
    if slackers:
        print(f"Phát hiện trốn việc: {slackers}. Gửi Cú Xanh gõ đầu...")
        send_discord_ping(slackers, issue_number)
    else:
        print("Ai cũng đã tick xong nhiệm vụ. Cú Xanh đi ngủ hiền hoà.")

if __name__ == "__main__":
    main()
