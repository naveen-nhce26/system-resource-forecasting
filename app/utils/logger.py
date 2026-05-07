"""
Reusable logging utilities.

Provides console + rotating file logging suitable for production services.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


_CONFIGURED_LOGGERS: set[str] = set()


def get_logger(
    name: str,
    *,
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    log_file: str = "app.log",
) -> logging.Logger:
    """
    Create (or return) a configured logger with console + rotating file handlers.

    Args:
        name: Logger name (typically `__name__`).
        level: Logging level string (e.g. "INFO", "DEBUG").
        log_dir: Directory for file logs. If None, file logging is skipped.
        log_file: File name for the rotating log file (within `log_dir`).
    """

    logger = logging.getLogger(name)
    resolved_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(resolved_level)

    # Prevent duplicate handlers if get_logger is called multiple times.
    if name in _CONFIGURED_LOGGERS:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(resolved_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_path = log_dir / log_file
        file_handler = RotatingFileHandler(
            filename=str(file_path),
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(resolved_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Avoid double logging via root handlers.
    logger.propagate = False
    _CONFIGURED_LOGGERS.add(name)
    return logger

