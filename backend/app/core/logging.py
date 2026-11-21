"""Stdlib logging setup: stderr always, a dated file when LOG_TO_FILE is on."""

from __future__ import annotations

import datetime
import logging
import sys
from pathlib import Path

from backend.app.core.config import Settings

_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_NOISY = ("httpx", "httpcore", "google_genai", "google.genai", "asyncio")


def setup_logging(settings: Settings) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if settings.log_to_file:
        try:
            directory = Path(settings.log_path)
            directory.mkdir(parents=True, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%d-%m-%Y")
            handlers.append(logging.FileHandler(directory / f"logger_{stamp}.log", encoding="utf-8"))
        except OSError as exc:
            print(f"Could not open log file: {exc}", file=sys.stderr)
    logging.basicConfig(level=logging.INFO, format=_FORMAT, handlers=handlers, force=True)
    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)
