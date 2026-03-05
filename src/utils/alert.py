"""
============================================================
🚨 Alert Module — Hệ thống cảnh báo đa kênh
============================================================

MỤC ĐÍCH:
    Gửi cảnh báo khi phát hiện lửa/khói qua nhiều kênh:
    1. 🔊 Âm thanh (còi báo động)
    2. 📱 Telegram Bot (gửi ảnh + thông tin)
    3. 💬 Zalo OA (tin nhắn tiếng Việt)
    4. 📞 SMS (qua Twilio)
    5. 📍 Geocoding (chuyển GPS → địa chỉ VN)

GIẢI THÍCH LOGIC CẢNH BÁO CHO BẠN:

    KHÔNG cảnh báo ngay khi detect 1 frame!
    (Vì có thể false positive → quấy rầy người dùng)

    Logic:
    1. Model detect lửa/khói → đếm consecutive_frames
    2. Nếu ≥ 3 frames liên tiếp → BẬT cảnh báo
    3. Sau khi cảnh báo → cooldown 60 giây (tránh spam)
    4. Nếu hết cooldown + vẫn detect → cảnh báo lại

    VÍ DỤ:
    Frame 1: Fire (confidence=0.8)  → count=1 → chưa cảnh báo
    Frame 2: Fire (confidence=0.9)  → count=2 → chưa cảnh báo
    Frame 3: Fire (confidence=0.85) → count=3 → 🚨 CẢNH BÁO!
    Frame 4: Fire (confidence=0.7)  → đang cooldown → bỏ qua
    ...
    Frame 100: không detect        → count=0 → reset
"""

import os
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import cv2
import numpy as np

from src.config import Config

# Thiết lập logging
logger = logging.getLogger(__name__)


class AlertManager:
    """
    Quản lý hệ thống cảnh báo đa kênh.

    THUỘC TÍNH:
        config: Config chứa alert settings
        consecutive_count: Số frame liên tiếp có detection
        last_alert_time: Thời điểm cảnh báo gần nhất
        is_alert_active: Đang trong trạng thái cảnh báo không

    VÍ DỤ:
        >>> alert_mgr = AlertManager(config)
        >>> # Trong detection loop:
        >>> for frame in video:
        ...     detections = model.predict(frame)
        ...     alert_mgr.process_detections(frame, detections)
    """

    def __init__(self, config: Config):
        self.config = config.alert
        self.consecutive_count = 0
        self.last_alert_time = 0
        self.is_alert_active = False
        self.frames_threshold = self.config.consecutive_frames_to_alert
        self.cooldown = self.config.cooldown_seconds

        # Khởi tạo các channel alerts
        self._init_channels()

    def _init_channels(self):
        """Khởi tạo các kênh cảnh báo."""
        self.channels = []

        if self.config.audio.enabled:
            self.channels.append(AudioAlert(self.config.audio))
            logger.info("🔊 Audio alert: ENABLED")

        if self.config.telegram.enabled:
            self.channels.append(TelegramAlert(self.config.telegram))
            logger.info("📱 Telegram alert: ENABLED")

        if self.config.zalo.enabled:
            self.channels.append(ZaloAlert(self.config.zalo))
            logger.info("💬 Zalo alert: ENABLED")

        if self.config.twilio.enabled:
            self.channels.append(TwilioSMSAlert(self.config.twilio))
            logger.info("📞 Twilio SMS alert: ENABLED")

        if not self.channels:
            logger.warning("⚠️ Không có kênh cảnh báo nào được bật!")

    def process_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        location: Optional[str] = None,
        gps: Optional[tuple] = None,
    ):
        """
        Xử lý detections và quyết định có cảnh báo không.

        LOGIC:
            1. Kiểm tra có detection nào không
            2. Nếu có → tăng consecutive_count
            3. Nếu count >= threshold → kiểm tra cooldown → cảnh báo
            4. Nếu không có → reset count

        Args:
            frame: Frame video hiện tại
            detections: List detections từ model
            location: Địa chỉ (vd: "123 Hẻm 45, Q.7, TP.HCM")
            gps: Tuple (latitude, longitude) cho reverse geocoding
        """
        if not self.config.enabled:
            return

        current_time = time.time()

        if detections:
            self.consecutive_count += 1

            # Kiểm tra đủ consecutive frames chưa
            if self.consecutive_count >= self.frames_threshold:
                # Kiểm tra cooldown
                time_since_last = current_time - self.last_alert_time
                if time_since_last >= self.cooldown:
                    self._trigger_alert(frame, detections, location, gps)
                    self.last_alert_time = current_time
                    self.consecutive_count = 0
        else:
            # Không detect → reset
            self.consecutive_count = 0
            self.is_alert_active = False

    def _trigger_alert(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        location: Optional[str],
        gps: Optional[tuple],
    ):
        """
        Gửi cảnh báo qua tất cả kênh đã bật.

        Args:
            frame: Frame chứa fire/smoke
            detections: Thông tin detection
            location: Địa chỉ
            gps: Tọa độ GPS
        """
        self.is_alert_active = True
        now = datetime.now().strftime("%H:%M:%S %d/%m/%Y")

        # Tìm detection confidence cao nhất cho mỗi class
        fire_conf = max(
            (d['confidence'] for d in detections if d['class_name'] == 'Fire'),
            default=0
        )
        smoke_conf = max(
            (d['confidence'] for d in detections if d['class_name'] == 'Smoke'),
            default=0
        )

        # Reverse geocoding nếu có GPS
        if gps and not location and self.config.vietmap.enabled:
            location = self._reverse_geocode(gps)

        # Tạo message
        message = self._create_alert_message(
            now, location, fire_conf, smoke_conf
        )

        # Lưu ảnh tạm
        temp_image_path = "/tmp/fire_alert_frame.jpg"
        cv2.imwrite(temp_image_path, frame)

        # Gửi qua tất cả kênh
        logger.warning(f"🚨 FIRE ALERT TRIGGERED at {now}")
        print(f"\n{'!'*60}")
        print(f"🚨🔥 CẢNH BÁO CHÁY!")
        print(message)
        print(f"{'!'*60}\n")

        for channel in self.channels:
            try:
                channel.send(message, temp_image_path)
            except Exception as e:
                logger.error(f"❌ Lỗi gửi cảnh báo qua {channel.__class__.__name__}: {e}")

    def _create_alert_message(
        self,
        timestamp: str,
        location: Optional[str],
        fire_conf: float,
        smoke_conf: float,
    ) -> str:
        """
        Tạo tin nhắn cảnh báo bằng tiếng Việt.

        Format:
            🔥 CẢNH BÁO CHÁY!
            Thời gian: 14:32:15 05/03/2026
            Địa điểm: 123 Hẻm 45, P. Tân Thuận, Q.7, TP.HCM
            Độ tin cậy: Lửa 92% | Khói 87%
            Vui lòng liên hệ: 114 (Cứu hỏa)
        """
        parts = [
            "🔥 CẢNH BÁO CHÁY!",
            f"Thời gian: {timestamp}",
        ]

        if location:
            parts.append(f"Địa điểm: {location}")

        conf_parts = []
        if fire_conf > 0:
            conf_parts.append(f"Lửa {fire_conf:.0%}")
        if smoke_conf > 0:
            conf_parts.append(f"Khói {smoke_conf:.0%}")
        if conf_parts:
            parts.append(f"Độ tin cậy: {' | '.join(conf_parts)}")

        parts.append("Vui lòng liên hệ: 114 (Cứu hỏa)")

        return "\n".join(parts)

    def _reverse_geocode(self, gps: tuple) -> str:
        """
        Chuyển tọa độ GPS thành địa chỉ Việt Nam (qua Vietmap API).

        Args:
            gps: (latitude, longitude)

        Returns:
            Địa chỉ tiếng Việt
        """
        try:
            import requests

            lat, lon = gps
            api_key = self.config.vietmap.api_key
            url = (
                f"https://maps.vietmap.vn/api/reverse/v3"
                f"?apikey={api_key}"
                f"&lat={lat}&lng={lon}"
            )

            response = requests.get(url, timeout=5)
            data = response.json()

            if data and len(data) > 0:
                return data[0].get('display', f"GPS: {lat}, {lon}")
        except Exception as e:
            logger.error(f"Lỗi reverse geocode: {e}")

        return f"GPS: {gps[0]}, {gps[1]}"


# ============================================================
# CÁC KÊNH CẢNH BÁO RIÊNG LẺ
# ============================================================

class AudioAlert:
    """
    🔊 Cảnh báo bằng âm thanh (còi/chuông).

    Dùng pygame để phát file .wav/.mp3.
    Nếu chạy trên server headless (không có loa) → skip.
    """

    def __init__(self, config):
        self.alarm_file = config.alarm_file
        self._initialized = False

        try:
            import pygame
            pygame.mixer.init()
            self._initialized = True
        except Exception:
            logger.warning("⚠️ Không thể khởi tạo pygame audio")

    def send(self, message: str, image_path: str = None):
        if not self._initialized:
            return

        try:
            import pygame

            if Path(self.alarm_file).exists():
                pygame.mixer.music.load(self.alarm_file)
                pygame.mixer.music.play()
                logger.info("🔊 Đang phát còi báo động...")
            else:
                # Nếu không có file alarm, tạo beep đơn giản
                logger.warning(f"⚠️ Không tìm thấy file: {self.alarm_file}")
        except Exception as e:
            logger.error(f"❌ Lỗi phát âm thanh: {e}")


class TelegramAlert:
    """
    📱 Cảnh báo qua Telegram Bot.

    CÀI ĐẶT:
        1. Tạo bot tại @BotFather trên Telegram
        2. Lấy bot token
        3. Thêm bot vào nhóm hoặc chat trực tiếp
        4. Lấy chat_id
        5. Điền vào .env file

    FLOW:
        1. Upload ảnh fire frame lên Telegram
        2. Gửi kèm caption (message cảnh báo)
    """

    def __init__(self, config):
        self.bot_token = config.bot_token
        self.chat_id = config.chat_id

    def send(self, message: str, image_path: str = None):
        try:
            import requests

            if image_path and Path(image_path).exists():
                # Gửi ảnh kèm caption
                url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
                with open(image_path, 'rb') as photo:
                    response = requests.post(
                        url,
                        data={'chat_id': self.chat_id, 'caption': message},
                        files={'photo': photo},
                        timeout=10,
                    )
            else:
                # Chỉ gửi text
                url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
                response = requests.post(
                    url,
                    data={'chat_id': self.chat_id, 'text': message},
                    timeout=10,
                )

            if response.status_code == 200:
                logger.info("📱 Telegram alert sent!")
            else:
                logger.error(f"❌ Telegram error: {response.text}")

        except Exception as e:
            logger.error(f"❌ Telegram error: {e}")


class ZaloAlert:
    """
    💬 Cảnh báo qua Zalo OA (Official Account).

    CÀI ĐẶT:
        1. Tạo OA tại https://oa.zalo.me/
        2. Lấy access_token
        3. Người dùng phải follow OA để nhận tin

    LƯU Ý:
        Zalo OA API chỉ cho gửi tin nhắn cho user đã follow.
        Cần refresh access_token định kỳ.
    """

    def __init__(self, config):
        self.access_token = config.access_token
        self.user_id = config.user_id

    def send(self, message: str, image_path: str = None):
        try:
            import requests

            url = "https://openapi.zalo.me/v2.0/oa/message"
            headers = {
                "access_token": self.access_token,
                "Content-Type": "application/json",
            }
            payload = {
                "recipient": {"user_id": self.user_id},
                "message": {"text": message},
            }

            response = requests.post(url, json=payload, headers=headers, timeout=10)

            if response.status_code == 200:
                logger.info("💬 Zalo alert sent!")
            else:
                logger.error(f"❌ Zalo error: {response.text}")

        except Exception as e:
            logger.error(f"❌ Zalo error: {e}")


class TwilioSMSAlert:
    """
    📞 Cảnh báo qua SMS (Twilio).

    CÀI ĐẶT:
        1. Đăng ký tại https://www.twilio.com/
        2. Mua số điện thoại
        3. Lấy account_sid + auth_token
        4. Điền vào .env

    CHI PHÍ:
        Twilio tính phí mỗi SMS (~$0.0075/SMS cho VN).
        Nên dùng Telegram miễn phí cho testing.
    """

    def __init__(self, config):
        self.account_sid = config.account_sid
        self.auth_token = config.auth_token
        self.from_number = config.from_number
        self.to_numbers = list(config.to_numbers) if config.to_numbers else []

    def send(self, message: str, image_path: str = None):
        try:
            from twilio.rest import Client

            client = Client(self.account_sid, self.auth_token)

            for to_number in self.to_numbers:
                client.messages.create(
                    body=message,
                    from_=self.from_number,
                    to=to_number,
                )
                logger.info(f"📞 SMS sent to {to_number}")

        except Exception as e:
            logger.error(f"❌ Twilio error: {e}")
