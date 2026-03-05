"""
Twilio Alert
============
Sends an SMS (and optionally places a voice call) via Twilio to notify
homeowners, fire departments, and local authorities.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def _get_twilio_client():
    """Return an authenticated Twilio REST client."""
    from twilio.rest import Client  # type: ignore

    account_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN", "")
    if not account_sid or not auth_token:
        raise EnvironmentError(
            "TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN must be set in environment / .env"
        )
    return Client(account_sid, auth_token)


def send_sms(
    body: str,
    to: Optional[str] = None,
    from_: Optional[str] = None,
) -> None:
    """
    Send an SMS alert via Twilio.

    Parameters
    ----------
    body   : SMS message body.
    to     : Recipient phone number (falls back to TWILIO_TO_NUMBER env var).
    from_  : Sender phone number (falls back to TWILIO_FROM_NUMBER env var).
    """
    to_number = to or os.getenv("TWILIO_TO_NUMBER", "")
    from_number = from_ or os.getenv("TWILIO_FROM_NUMBER", "")

    if not to_number or not from_number:
        logger.warning("Twilio SMS skipped: TWILIO_TO_NUMBER / TWILIO_FROM_NUMBER not set.")
        return

    try:
        client = _get_twilio_client()
    except EnvironmentError as exc:
        logger.warning("Twilio SMS skipped: %s", exc)
        return

    try:
        message = client.messages.create(body=body, from_=from_number, to=to_number)
        logger.info("Twilio SMS sent. SID: %s", message.sid)
    except Exception as exc:
        logger.error("Failed to send Twilio SMS: %s", exc)


def make_voice_call(
    twiml_say: str,
    to: Optional[str] = None,
    from_: Optional[str] = None,
) -> None:
    """
    Place a voice call via Twilio that reads out *twiml_say*.

    Parameters
    ----------
    twiml_say : Text for Twilio to speak (plain TwiML <Say> verb).
    to        : Recipient phone number.
    from_     : Caller phone number.
    """
    to_number = to or os.getenv("TWILIO_TO_NUMBER", "")
    from_number = from_ or os.getenv("TWILIO_FROM_NUMBER", "")

    if not to_number or not from_number:
        logger.warning("Twilio voice call skipped: phone numbers not set.")
        return

    try:
        client = _get_twilio_client()
    except EnvironmentError as exc:
        logger.warning("Twilio voice call skipped: %s", exc)
        return

    twiml = f"<Response><Say language='vi-VN'>{twiml_say}</Say></Response>"
    try:
        call = client.calls.create(
            twiml=twiml,
            from_=from_number,
            to=to_number,
        )
        logger.info("Twilio voice call initiated. SID: %s", call.sid)
    except Exception as exc:
        logger.error("Failed to place Twilio voice call: %s", exc)
