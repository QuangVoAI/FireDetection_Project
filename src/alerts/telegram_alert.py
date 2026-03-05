"""
Telegram Alert
==============
Sends a fire alert message (and optional photo) to a Telegram chat via Bot API.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _get_credentials() -> tuple[str, str]:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        raise EnvironmentError(
            "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in environment / .env"
        )
    return token, chat_id


async def _send_message_async(
    token: str,
    chat_id: str,
    text: str,
    photo_path: Optional[str] = None,
) -> None:
    from telegram import Bot

    bot = Bot(token=token)
    if photo_path and Path(photo_path).exists():
        with open(photo_path, "rb") as f:
            await bot.send_photo(chat_id=chat_id, photo=f, caption=text)
    else:
        await bot.send_message(chat_id=chat_id, text=text)


def send_alert(
    text: str,
    photo_path: Optional[str] = None,
) -> None:
    """
    Send a Telegram alert.

    Parameters
    ----------
    text       : Message body (supports HTML/Markdown per bot config).
    photo_path : Optional path to a screenshot to attach.
    """
    try:
        token, chat_id = _get_credentials()
    except EnvironmentError as exc:
        logger.warning("Telegram alert skipped: %s", exc)
        return

    try:
        asyncio.run(_send_message_async(token, chat_id, text, photo_path))
        logger.info("Telegram alert sent.")
    except Exception as exc:
        logger.error("Failed to send Telegram alert: %s", exc)
