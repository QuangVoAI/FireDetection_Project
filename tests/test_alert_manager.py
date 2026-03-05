"""
Tests for the alert manager.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.alerts.alert_manager import AlertManager


@pytest.fixture
def alert_mgr(tmp_path):
    """AlertManager with all channels disabled (no real I/O)."""
    cfg_file = tmp_path / "alert_config.yaml"
    cfg_file.write_text(
        "alert:\n"
        "  confidence_threshold: 0.5\n"
        "  consecutive_frames: 2\n"
        "  cooldown_seconds: 1\n"
        "sound:\n"
        "  enabled: false\n"
        "telegram:\n"
        "  enabled: false\n"
        "zalo:\n"
        "  enabled: false\n"
        "twilio:\n"
        "  enabled: false\n"
        "vietmap:\n"
        "  enabled: false\n"
    )
    return AlertManager(config_path=str(cfg_file))


class TestAlertManagerThreshold:
    def test_low_confidence_does_not_trigger(self, alert_mgr):
        fired = alert_mgr.on_detection(confidence=0.3)
        assert fired is False

    def test_high_confidence_below_consecutive_does_not_trigger(self, alert_mgr):
        fired = alert_mgr.on_detection(confidence=0.8)
        assert fired is False  # only 1 detection, need 2

    def test_triggers_after_consecutive_frames(self, alert_mgr):
        with patch.object(alert_mgr, "_fire_alerts") as mock_fire:
            alert_mgr.on_detection(confidence=0.8)
            fired = alert_mgr.on_detection(confidence=0.8)
        assert fired is True
        mock_fire.assert_called_once()


class TestAlertManagerCooldown:
    def test_cooldown_suppresses_repeated_alerts(self, alert_mgr):
        with patch.object(alert_mgr, "_fire_alerts"):
            # First burst
            alert_mgr.on_detection(confidence=0.9)
            alert_mgr.on_detection(confidence=0.9)  # triggers
            # Immediately again – should be suppressed by cooldown
            alert_mgr._consecutive_count = 2
            result = alert_mgr.on_detection(confidence=0.9)
        assert result is False

    def test_alert_fires_again_after_cooldown(self, alert_mgr):
        alert_mgr.cooldown_seconds = 0  # no cooldown
        with patch.object(alert_mgr, "_fire_alerts") as mock_fire:
            alert_mgr.on_detection(confidence=0.9)
            alert_mgr.on_detection(confidence=0.9)  # fires #1
            alert_mgr.on_detection(confidence=0.9)
            alert_mgr.on_detection(confidence=0.9)  # fires #2
        assert mock_fire.call_count == 2


class TestAlertManagerReset:
    def test_reset_clears_consecutive_count(self, alert_mgr):
        alert_mgr.on_detection(confidence=0.8)
        assert alert_mgr._consecutive_count == 1
        alert_mgr.reset()
        assert alert_mgr._consecutive_count == 0


class TestAlertManagerMessageFormat:
    def test_format_message_interpolates_fields(self, alert_mgr):
        template = "Fire at {address} with {confidence:.0%}"
        result = alert_mgr._format_message(template, address="Test St", confidence=0.85)
        assert "Test St" in result
        assert "85%" in result

    def test_format_message_handles_missing_keys(self, alert_mgr):
        template = "Alert: {missing_key}"
        # Should not raise
        result = alert_mgr._format_message(template)
        assert "Alert:" in result
