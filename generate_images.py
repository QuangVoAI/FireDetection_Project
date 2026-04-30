import os
import random
import time
import requests
import urllib.parse
from datetime import datetime, timedelta

# 1. Không cần API Key cho bản Miễn phí này
print("🚀 Đang khởi động bản Miễn phí (Pollinations AI)...")

# 2. Các danh sách biến số để tạo sự đa dạng cho 2000 bức ảnh
neon_signs = ["PHỞ CỔ", "HỦ TIẾU NAM VANG", "TIỆM LÀM TÓC", "TẠP HÓA", "KARAOKE", "BÁNH MÌ", "CƠM TẤM", "SỬA XE", "BIDA", "TRÀ ĐÁ"]
weather_conditions = ["mưa lất phất", "mưa rào", "mưa tầm tã", "mưa bóng mây", "mưa phùn ướt sũng"]
fire_intensities = ["bốc cháy dữ dội", "phát nổ với tia lửa lớn", "cháy chập điện với nhiều khói đen", "xuyệt tia lửa liên tục rực sáng"]

# Thư mục lưu ảnh - Lưu thẳng vào dataset
output_dir = "data/02_alley_context/images"
os.makedirs(output_dir, exist_ok=True)

# Mốc thời gian bắt đầu
start_date = datetime(2023, 1, 1)

# 3. Vòng lặp tạo 2000 ảnh
for i in range(1, 101):
    # Trộn các biến ngẫu nhiên
    current_time = start_date + timedelta(days=random.randint(0, 365), hours=random.randint(18, 23), minutes=random.randint(0, 59), seconds=random.randint(0, 59))
    timestamp_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    
    sign_1 = random.choice(neon_signs)
    sign_2 = random.choice([s for s in neon_signs if s != sign_1]) # Đảm bảo biển 2 không trùng biển 1
    weather = random.choice(weather_conditions)
    fire = random.choice(fire_intensities)

    # Lắp ghép prompt hoàn chỉnh
    prompt = f"Một góc nhìn camera an ninh (CCTV) từ trên cao xuống một con hẻm hẹp ở Sài Gòn về đêm. Con hẻm hẹp, lát bê tông ướt nhẫy vì {weather}. Một cột điện lộn xộn dây điện ở phía bên phải đang {fire}. Ngọn lửa và mớ dây điện bị đứt và rơi xuống tạo ra tia lửa, với mây khói đen cuồn cuộn. Người dân đứng dưới hẻm mặc áo mưa đang hoảng loạn, có người cầm bình chữa cháy và người khác đang dùng điện thoại gọi. Biển hiệu neon ở phía sau sáng rực với các chữ '{sign_1}' và '{sign_2}'. Dấu thời gian ở góc trên bên phải là '{timestamp_str}'. Toàn cảnh hẻm đầy xe máy và các mái tôn lụp xụp."

    print(f"🖼️ Đang tạo ảnh {i}/100...")
    
    try:
        # Tạo URL cho Pollinations (Model Flux hoặc SDXL)
        encoded_prompt = urllib.parse.quote(prompt)
        # Thêm seed ngẫu nhiên để mỗi ảnh mỗi khác
        seed = random.randint(0, 999999)
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?seed={seed}&width=1280&height=720&nologo=true&model=flux"

        response = requests.get(url)
        
        if response.status_code == 200:
            image_path = os.path.join(output_dir, f"alley_{i:04d}.jpg")
            with open(image_path, "wb") as f:
                f.write(response.content)
            print(f"   ✅ Đã lưu: {image_path}")
        else:
            print(f"   ❌ Lỗi API: {response.status_code}")

    except Exception as e:
        print(f"   ❌ Gặp lỗi ở ảnh thứ {i}: {e}")
        
    # Giảm thời gian chờ xuống vì Pollinations rất nhanh
    time.sleep(1) 

print("Đã hoàn tất quá trình tạo 2.000 ảnh!")