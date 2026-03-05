"""
Tests for Vietmap API integration.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.vietmap_api import reverse_geocode


class TestReverseGeocode:
    def test_returns_raw_coords_when_no_api_key(self):
        result = reverse_geocode(10.7769, 106.7009, api_key="")
        assert "10.7769" in result
        assert "106.7009" in result

    def test_uses_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("VIETMAP_API_KEY", "testkey")
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"display": "123 Nguyen Van Linh, Q7"}]
        mock_resp.raise_for_status = MagicMock()
        with patch("src.utils.vietmap_api.requests.get", return_value=mock_resp) as mock_get:
            result = reverse_geocode(10.7769, 106.7009)
        mock_get.assert_called_once()
        assert "Nguyen Van Linh" in result

    def test_returns_coords_on_request_failure(self, monkeypatch):
        monkeypatch.setenv("VIETMAP_API_KEY", "testkey")
        with patch("src.utils.vietmap_api.requests.get", side_effect=Exception("timeout")):
            result = reverse_geocode(10.7769, 106.7009)
        assert "10.7769" in result

    def test_handles_empty_api_response(self, monkeypatch):
        monkeypatch.setenv("VIETMAP_API_KEY", "testkey")
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status = MagicMock()
        with patch("src.utils.vietmap_api.requests.get", return_value=mock_resp):
            result = reverse_geocode(10.7769, 106.7009)
        assert "10.7769" in result
