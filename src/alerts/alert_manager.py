"""
Alert Manager
=============
Central hub that aggregates all notification channels and enforces
alert cooldown to avoid notification storms.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Manages multi-channel fire alerts with configurable thresholds.

    Usage
    -----
    >>> mgr = AlertManager()
    >>> mgr.on_detection(confidence=0.82, frame_screenshot="frame.jpg",
    ...                  lat=10.7769, lon=106.7009)
    """

    def __init__(self, config_path: str = "config/alert_config.yaml") -> None:
        self._cfg = {}
        if Path(config_path).exists():
            with open(config_path) as f:
                self._cfg = yaml.safe_load(f)

        alert_cfg = self._cfg.get("alert", {})
        self.confidence_threshold: float = float(
            os.getenv("ALERT_CONFIDENCE_THRESHOLD",
                      alert_cfg.get("confidence_threshold", 0.45))
        )
        self.consecutive_required: int = int(
            os.getenv("ALERT_CONSECUTIVE_FRAMES",
                      alert_cfg.get("consecutive_frames", 3))
        )
        self.cooldown_seconds: int = int(alert_cfg.get("cooldown_seconds", 60))

        self._consecutive_count: int = 0
        self._last_alert_time: float = 0.0

    # ── Internal helpers ───────────────────────────────────────────────────

    def _format_message(self, template: str, **kwargs) -> str:
        try:
            return template.format(**kwargs)
        except KeyError:
            return template

    def _build_context(
        self,
        confidence: float,
        lat: float,
        lon: float,
        address: str,
    ) -> dict:
        return dict(
            confidence=confidence,
            lat=lat,
            lon=lon,
            address=address,
            time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def on_detection(
        self,
        confidence: float,
        frame_screenshot: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> bool:
        """
        Call this on every frame where a detection occurs.

        Returns True when an alert was actually fired, False otherwise.
        """
        if confidence < self.confidence_threshold:
            self._consecutive_count = 0
            return False

        self._consecutive_count += 1
        if self._consecutive_count < self.consecutive_required:
            return False

        now = time.time()
        if now - self._last_alert_time < self.cooldown_seconds:
            logger.debug("Alert cooldown active – suppressing.")
            return False

        self._last_alert_time = now
        self._consecutive_count = 0

        # Resolve address
        _lat = lat if lat is not None else float(os.getenv("CAMERA_LATITUDE", "10.7769"))
        _lon = lon if lon is not None else float(os.getenv("CAMERA_LONGITUDE", "106.7009"))
        default_address = os.getenv("CAMERA_ADDRESS", f"{_lat}, {_lon}")
        address = default_address

        if self._cfg.get("vietmap", {}).get("enabled", False):
            try:
                from src.utils.vietmap_api import reverse_geocode
                address = reverse_geocode(_lat, _lon)
            except Exception as exc:
                logger.warning("Vietmap lookup failed: %s", exc)

        ctx = self._build_context(confidence, _lat, _lon, address)

        self._fire_alerts(ctx, frame_screenshot)
        return True

    def reset(self) -> None:
        """Reset consecutive counter (call when no detection in a frame)."""
        self._consecutive_count = 0

    # ── Channel dispatch ───────────────────────────────────────────────────

    def _fire_alerts(self, ctx: dict, photo_path: Optional[str]) -> None:
        """Dispatch to all enabled alert channels."""

        # 1. Sound
        if self._cfg.get("sound", {}).get("enabled", True):
            try:
                from src.alerts.sound_alert import play_alert
                wav = self._cfg.get("sound", {}).get("wav_file", "assets/siren.wav")
                play_alert(wav)
            except Exception as exc:
                logger.error("Sound alert error: %s", exc)

        # 2. Telegram
        tg_cfg = self._cfg.get("telegram", {})
        if tg_cfg.get("enabled", False):
            try:
                from src.alerts.telegram_alert import send_alert as tg_send
                msg = self._format_message(
                    tg_cfg.get("message_template", "🔥 FIRE detected at {address}"),
                    **ctx,
                )
                tg_send(msg, photo_path=photo_path)
            except Exception as exc:
                logger.error("Telegram alert error: %s", exc)

        # 3. Zalo
        zalo_cfg = self._cfg.get("zalo", {})
        if zalo_cfg.get("enabled", False):
            try:
                from src.alerts.zalo_alert import send_alert as zalo_send
                msg = self._format_message(
                    zalo_cfg.get("message_template", "🔥 FIRE at {address}"),
                    **ctx,
                )
                zalo_send(msg)
            except Exception as exc:
                logger.error("Zalo alert error: %s", exc)

        # 4. Twilio SMS
        twilio_cfg = self._cfg.get("twilio", {})
        if twilio_cfg.get("enabled", False) and twilio_cfg.get("sms", {}).get("enabled", False):
            try:
                from src.alerts.twilio_alert import send_sms
                msg = self._format_message(
                    twilio_cfg.get("sms", {}).get(
                        "message_template", "FIRE at {address} ({lat},{lon}) [{time}]"
                    ),
                    **ctx,
                )
                send_sms(msg)
            except Exception as exc:
                logger.error("Twilio SMS alert error: %s", exc)

        # 5. Twilio Voice Call
        if twilio_cfg.get("enabled", False) and twilio_cfg.get("voice", {}).get("enabled", False):
            try:
                from src.alerts.twilio_alert import make_voice_call
                msg = self._format_message(
                    twilio_cfg.get("voice", {}).get(
                        "twiml_message", "Fire detected at {address}"
                    ),
                    **ctx,
                )
                make_voice_call(msg)
            except Exception as exc:
                logger.error("Twilio voice alert error: %s", exc)

        logger.info("🔥 Alert dispatched | confidence=%.2f | address=%s", ctx["confidence"], ctx["address"])
