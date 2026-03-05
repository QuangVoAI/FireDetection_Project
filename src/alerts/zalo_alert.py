"""
Zalo OA Alert
=============
Sends a fire alert to a user via Zalo Official Account (OA) Open API.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)

ZALO_SEND_MESSAGE_URL = "https://openapi.zalo.me/v3.0/oa/message/cs"


def send_alert(text: str, user_id: Optional[str] = None) -> None:
    """
    Send a text alert message via Zalo OA.

    Parameters
    ----------
    text    : Alert message text.
    user_id : Zalo user ID to notify (falls back to ZALO_USER_ID env var).
    """
    access_token = os.getenv("ZALO_ACCESS_TOKEN", "")
    recipient_id = user_id or os.getenv("ZALO_USER_ID", "")

    if not access_token or not recipient_id:
        logger.warning(
            "Zalo alert skipped: ZALO_ACCESS_TOKEN / ZALO_USER_ID not set."
        )
        return

    headers = {
        "access_token": access_token,
        "Content-Type": "application/json",
    }
    payload = {
        "recipient": {"user_id": recipient_id},
        "message": {"text": text},
    }

    try:
        resp = requests.post(
            ZALO_SEND_MESSAGE_URL, json=payload, headers=headers, timeout=10
        )
        resp.raise_for_status()
        logger.info("Zalo alert sent. Response: %s", resp.json())
    except requests.RequestException as exc:
        logger.error("Failed to send Zalo alert: %s", exc)
