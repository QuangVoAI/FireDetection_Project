"""
Vietmap API Integration
========================
Provides reverse-geocoding (GPS coordinates → human-readable address)
using the Vietmap REST API.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)

VIETMAP_REVERSE_URL = "https://maps.vietmap.vn/api/reverse/v3"


def reverse_geocode(
    lat: float,
    lon: float,
    api_key: Optional[str] = None,
    timeout: int = 5,
) -> str:
    """
    Look up a human-readable address for the given GPS coordinates.

    Parameters
    ----------
    lat     : Latitude (decimal degrees)
    lon     : Longitude (decimal degrees)
    api_key : Vietmap API key (falls back to VIETMAP_API_KEY env var)
    timeout : Request timeout in seconds

    Returns
    -------
    A formatted address string, or a fallback "lat, lon" string on failure.
    """
    key = api_key or os.getenv("VIETMAP_API_KEY", "")
    if not key:
        logger.warning("VIETMAP_API_KEY not set – returning raw coordinates.")
        return f"{lat}, {lon}"

    params = {"apikey": key, "lat": lat, "lng": lon}
    try:
        resp = requests.get(VIETMAP_REVERSE_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        # Vietmap returns a list; pick the first result's display field
        if isinstance(data, list) and data:
            return data[0].get("display", f"{lat}, {lon}")
        return f"{lat}, {lon}"
    except Exception as exc:
        logger.warning("Vietmap reverse-geocode failed: %s", exc)
        return f"{lat}, {lon}"
