# ============================================================
# 🐳 Dockerfile — Đóng gói ứng dụng thành Docker container
# ============================================================
# Build:  docker build -t fire-detection .
# Run:    docker run -p 8000:8000 fire-detection
# ============================================================

# --- Stage 1: Base image với Python + CUDA ---
# Dùng Python 3.11 slim để giảm kích thước image
FROM python:3.11-slim

# Đặt thư mục làm việc trong container
WORKDIR /app

# Cài đặt system dependencies
# libgl1: cần cho OpenCV (xử lý ảnh)
# libglib2.0-0: thư viện hệ thống cho OpenCV
# libsndfile1: cần cho pygame (phát âm thanh)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements trước (tận dụng Docker cache)
# Nếu requirements.txt không đổi, Docker sẽ dùng lại layer cũ
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ source code
COPY . .

# Expose port 8000 cho FastAPI
EXPOSE 8000

# Lệnh chạy khi container khởi động
CMD ["uvicorn", "web.main:app", "--host", "0.0.0.0", "--port", "8000"]
