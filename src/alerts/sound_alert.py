"""
Sound Alert
===========
Plays a local siren/buzzer sound file when fire is detected.
Falls back to a system beep if the configured WAV file is not found.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


def _play_async(wav_path: str) -> None:
    """Play audio in a background thread so it never blocks the main loop."""
    try:
        from playsound import playsound  # type: ignore
        playsound(wav_path, block=True)
    except Exception as exc:
        logger.warning("playsound failed (%s) – trying OS beep.", exc)
        _system_beep()


def _system_beep() -> None:
    """Cross-platform terminal bell."""
    try:
        import sys
        sys.stdout.write("\a")
        sys.stdout.flush()
    except Exception:
        pass


def play_alert(wav_file: str = "assets/siren.wav") -> None:
    """
    Trigger the siren sound in a non-blocking background thread.

    Parameters
    ----------
    wav_file : Path to the WAV/MP3 siren file.
    """
    if not Path(wav_file).exists():
        logger.warning("Alert sound file not found: %s – using system beep.", wav_file)
        _system_beep()
        return

    t = threading.Thread(target=_play_async, args=(wav_file,), daemon=True)
    t.start()
    logger.info("Sound alert triggered: %s", wav_file)
