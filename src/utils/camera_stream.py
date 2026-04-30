import os
import cv2
import threading
import time
import logging

logger = logging.getLogger(__name__)

# Optimize FFMPEG settings to minimize latency and drop corrupted packets
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|fflags;nobuffer|flags;low_delay|max_delay;500000"

class VideoStream:
    """
    Class đọc Stream Camera IP chạy trên Thread nền.
    Tự động thử lại khi mất mạng và giữ frame mới nhất ở trạng thái O(1).
    """
    def __init__(self, src, resize_dim=None):
        """
        Khởi tạo luồng camera.
        
        Args:
            src: Đường dẫn RTSP/HTTP (VD: rtsp://192.168.1.87:554/onvif1)
            resize_dim: Kích thước tuple để resize frame ngay lúc đọc (VD: (640, 640))
        """
        self.src = src
        self.resize_dim = resize_dim
        self.stream = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.ret, self.frame = self.stream.read()
        
        if self.ret and self.resize_dim:
            self.frame = cv2.resize(self.frame, self.resize_dim)
            
        self.stopped = False

    def start(self):
        # Bắt đầu luồng đọc frame trong background
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.stream.read()
            if not ret:
                logger.warning(f"Connection lost to {self.src}. Reconnecting...")
                self.stream.release()
                time.sleep(2)
                self.stream = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
                continue
            
            if self.resize_dim:
                frame = cv2.resize(frame, self.resize_dim)
                
            self.ret = ret
            self.frame = frame

    def read(self):
        # Luôn trả về frame mới nhất trong biến instance
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        if self.stream.isOpened():
            self.stream.release()
