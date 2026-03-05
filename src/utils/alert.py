"""
src/utils/alert.py
------------------
Multi-channel alert system for the Early Fire Detection System.

When fire or smoke is detected, this module can:
    - Play an on-site audio alarm
    - Send a Telegram Bot notification with image snapshot
    - Send a Zalo OA notification
    - Send an SMS/voice call via Twilio
    - Reverse-geocode GPS coordinates via Vietmap API
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class AlertSystem:
    """Multi-channel alert system for fire/smoke detection events.

    Reads alert configuration from a ConfigNode loaded via
    :func:`~src.config.load_config`. API credentials are loaded from
    environment variables (use a ``.env`` file).

    Args:
        config: ConfigNode with an ``alert`` sub-node.
    """

    def __init__(self, config) -> None:
        self.config = config
        self.alert_cfg = config.alert

        # TODO: Load credentials from environment variables (set in .env)
        self.telegram_bot_token: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id: Optional[str] = os.getenv("TELEGRAM_CHAT_ID")
        self.zalo_access_token: Optional[str] = os.getenv("ZALO_ACCESS_TOKEN")
        self.twilio_sid: Optional[str] = os.getenv("TWILIO_ACCOUNT_SID")
        self.twilio_token: Optional[str] = os.getenv("TWILIO_AUTH_TOKEN")
        self.twilio_from_number: Optional[str] = os.getenv("TWILIO_FROM_NUMBER")
        self.twilio_to_number: Optional[str] = os.getenv("TWILIO_TO_NUMBER")
        self.vietmap_api_key: Optional[str] = os.getenv("VIETMAP_API_KEY")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def trigger_alarm(self) -> None:
        """Play an on-site audio alarm using pygame.

        The alarm sound file path should be set via the ``ALARM_SOUND_PATH``
        environment variable. Falls back to a simple beep if not set.
        """
        if not self.alert_cfg.audio_alarm:
            return

        sound_path = os.getenv("ALARM_SOUND_PATH", "assets/alarm.wav")

        try:
            import pygame

            pygame.mixer.init()
            if Path(sound_path).exists():
                pygame.mixer.music.load(sound_path)
                pygame.mixer.music.play()
                logger.info("Audio alarm triggered: %s", sound_path)
            else:
                # Fallback: generate a simple beep tone
                import array
                import math

                sample_rate = 22050
                duration = 2  # seconds
                frequency = 880  # Hz

                n_samples = sample_rate * duration
                buf = array.array(
                    "h",
                    [
                        int(32767 * math.sin(2 * math.pi * frequency * t / sample_rate))
                        for t in range(n_samples)
                    ],
                )
                sound = pygame.sndarray.make_sound(
                    __import__("numpy").array(buf).reshape(-1, 1).repeat(2, axis=1)
                )
                sound.play()
                logger.info("Audio alarm (generated beep) triggered.")
        except ImportError:
            logger.warning("pygame not installed — audio alarm skipped.")
        except Exception as exc:
            logger.error("Audio alarm failed: %s", exc)

    def send_telegram(
        self,
        message: str,
        image_path: Optional[str] = None,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ) -> bool:
        """Send a Telegram notification with an optional image snapshot.

        Args:
            message: Alert text message.
            image_path: Optional path to a snapshot image to attach.
            bot_token: Telegram Bot API token. Falls back to ``TELEGRAM_BOT_TOKEN``.
            chat_id: Telegram chat ID. Falls back to ``TELEGRAM_CHAT_ID``.

        Returns:
            True if the message was sent successfully, False otherwise.

        TODO: Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in your .env file.
        """
        if not self.alert_cfg.telegram_enabled:
            logger.debug("Telegram alerts are disabled in config.")
            return False

        token = bot_token or self.telegram_bot_token
        cid = chat_id or self.telegram_chat_id

        if not token or not cid:
            logger.warning(
                "Telegram credentials missing. "
                "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env"
            )
            return False

        try:
            import requests

            if image_path and Path(image_path).exists():
                url = f"https://api.telegram.org/bot{token}/sendPhoto"
                with open(image_path, "rb") as photo:
                    resp = requests.post(
                        url,
                        data={"chat_id": cid, "caption": message},
                        files={"photo": photo},
                        timeout=10,
                    )
            else:
                url = f"https://api.telegram.org/bot{token}/sendMessage"
                resp = requests.post(
                    url,
                    data={"chat_id": cid, "text": message},
                    timeout=10,
                )

            if resp.ok:
                logger.info("Telegram alert sent successfully.")
                return True
            else:
                logger.error("Telegram API error: %s — %s", resp.status_code, resp.text)
                return False

        except Exception as exc:
            logger.error("Failed to send Telegram alert: %s", exc)
            return False

    def send_zalo(
        self,
        message: str,
        image_path: Optional[str] = None,
        access_token: Optional[str] = None,
    ) -> bool:
        """Send a Zalo OA notification.

        Args:
            message: Alert text message.
            image_path: Optional path to a snapshot image.
            access_token: Zalo OA access token. Falls back to ``ZALO_ACCESS_TOKEN``.

        Returns:
            True if sent successfully, False otherwise.

        TODO: Set ZALO_ACCESS_TOKEN in your .env file.
              Refer to https://developers.zalo.me/docs/api/official-account-api
        """
        if not self.alert_cfg.zalo_enabled:
            logger.debug("Zalo alerts are disabled in config.")
            return False

        token = access_token or self.zalo_access_token
        if not token:
            logger.warning("Zalo access token missing. Set ZALO_ACCESS_TOKEN in .env")
            return False

        try:
            import requests

            # TODO: Replace with actual Zalo OA broadcast API endpoint and payload
            url = "https://openapi.zalo.me/v2.0/oa/message"
            headers = {"access_token": token, "Content-Type": "application/json"}
            payload = {
                "recipient": {"message_tag": "CONFIRMED_EVENT_UPDATE"},
                "message": {"text": message},
            }
            resp = requests.post(url, json=payload, headers=headers, timeout=10)

            if resp.ok:
                logger.info("Zalo OA alert sent successfully.")
                return True
            else:
                logger.error("Zalo API error: %s — %s", resp.status_code, resp.text)
                return False

        except Exception as exc:
            logger.error("Failed to send Zalo alert: %s", exc)
            return False

    def send_sms_twilio(
        self,
        message: str,
        to_number: Optional[str] = None,
        from_number: Optional[str] = None,
        sid: Optional[str] = None,
        token: Optional[str] = None,
    ) -> bool:
        """Send an SMS alert via Twilio.

        Args:
            message: Alert text message (max 160 chars for single SMS).
            to_number: Destination phone number (E.164 format, e.g. ``+84901234567``).
            from_number: Twilio sender number. Falls back to ``TWILIO_FROM_NUMBER``.
            sid: Twilio Account SID. Falls back to ``TWILIO_ACCOUNT_SID``.
            token: Twilio Auth Token. Falls back to ``TWILIO_AUTH_TOKEN``.

        Returns:
            True if the SMS was queued successfully, False otherwise.

        TODO: Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER,
              and TWILIO_TO_NUMBER in your .env file.
        """
        if not self.alert_cfg.twilio_enabled:
            logger.debug("Twilio alerts are disabled in config.")
            return False

        account_sid = sid or self.twilio_sid
        auth_token = token or self.twilio_token
        sender = from_number or self.twilio_from_number
        recipient = to_number or self.twilio_to_number

        if not all([account_sid, auth_token, sender, recipient]):
            logger.warning(
                "Twilio credentials incomplete. "
                "Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, "
                "TWILIO_FROM_NUMBER, TWILIO_TO_NUMBER in .env"
            )
            return False

        try:
            from twilio.rest import Client

            client = Client(account_sid, auth_token)
            sms = client.messages.create(body=message, from_=sender, to=recipient)
            logger.info("Twilio SMS sent. SID: %s", sms.sid)
            return True

        except ImportError:
            logger.warning("twilio library not installed — SMS skipped.")
            return False
        except Exception as exc:
            logger.error("Failed to send Twilio SMS: %s", exc)
            return False

    def get_location_vietmap(
        self,
        lat: float,
        lon: float,
        api_key: Optional[str] = None,
    ) -> str:
        """Reverse-geocode GPS coordinates to a Vietnamese address via Vietmap API.

        Args:
            lat: Latitude (WGS84).
            lon: Longitude (WGS84).
            api_key: Vietmap API key. Falls back to ``VIETMAP_API_KEY``.

        Returns:
            Vietnamese address string, or a fallback coordinate string on failure.

        TODO: Obtain a Vietmap API key from https://maps.vietmap.vn/ and set
              VIETMAP_API_KEY in your .env file.
        """
        if not self.alert_cfg.vietmap_enabled:
            return f"GPS: {lat:.6f}, {lon:.6f}"

        key = api_key or self.vietmap_api_key
        if not key:
            logger.warning("Vietmap API key missing. Set VIETMAP_API_KEY in .env")
            return f"GPS: {lat:.6f}, {lon:.6f}"

        try:
            import requests

            # TODO: Confirm the exact Vietmap reverse-geocode endpoint URL
            url = "https://maps.vietmap.vn/api/reverse"
            params = {"apikey": key, "lat": lat, "lng": lon}
            resp = requests.get(url, params=params, timeout=5)

            if resp.ok:
                data = resp.json()
                # TODO: Parse the actual Vietmap response structure
                address = data.get("display", f"GPS: {lat:.6f}, {lon:.6f}")
                logger.info("Vietmap reverse geocode: %s", address)
                return address
            else:
                logger.warning("Vietmap API error: %s", resp.status_code)
                return f"GPS: {lat:.6f}, {lon:.6f}"

        except Exception as exc:
            logger.error("Vietmap geocode failed: %s", exc)
            return f"GPS: {lat:.6f}, {lon:.6f}"

    def build_alert_message(
        self,
        location: str,
        confidence: float,
        timestamp: Optional[datetime] = None,
        smoke_confidence: Optional[float] = None,
    ) -> str:
        """Format an alert message in Vietnamese.

        Args:
            location: Location string (address or GPS coordinates).
            confidence: Fire detection confidence score in [0, 1].
            timestamp: Alert timestamp. Defaults to ``datetime.now()``.
            smoke_confidence: Optional smoke detection confidence.

        Returns:
            Formatted Vietnamese alert message string.
        """
        if timestamp is None:
            timestamp = datetime.now()

        time_str = timestamp.strftime("%H:%M:%S %d/%m/%Y")
        fire_pct = f"{confidence * 100:.0f}%"

        lines = [
            "🔥 CẢNH BÁO CHÁY!",
            f"Thời gian: {time_str}",
            f"Địa điểm: {location}",
            f"Độ tin cậy: Lửa {fire_pct}",
        ]
        if smoke_confidence is not None:
            lines.append(f"             Khói {smoke_confidence * 100:.0f}%")

        lines.append("Vui lòng liên hệ: 114 (Cứu hỏa)")
        return "\n".join(lines)
