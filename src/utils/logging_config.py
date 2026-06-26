"""
Production logging configuration.

Provides specialized loggers for different application concerns:
- Application logger: general application events
- API logger: API-specific events
- Request logger: HTTP request/response details
- Error logger: error events with stack traces

All loggers use RotatingFileHandler to prevent unbounded log growth.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)

_LOG_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"

_DEFAULT_LOG_DIR = "logs"
_DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_DEFAULT_BACKUP_COUNT = 5


def _create_rotating_logger(
    name: str,
    log_file: str,
    log_dir: str = _DEFAULT_LOG_DIR,
    level: int = logging.INFO,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    backup_count: int = _DEFAULT_BACKUP_COUNT,
    propagate: bool = False,
) -> logging.Logger:
    """Create a logger with rotating file handler and stream handler.

    Args:
        name: Logger name.
        log_file: Log file name (e.g. 'app.log').
        log_dir: Directory for log files.
        level: Logging level.
        max_bytes: Max file size before rotation.
        backup_count: Number of backup files to keep.
        propagate: Whether to propagate to parent loggers.

    Returns:
        Configured logger instance.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = propagate

    formatter = logging.Formatter(
        fmt=_LOG_FORMAT,
        datefmt=_LOG_DATE_FORMAT,
    )

    # Rotating file handler
    file_handler = RotatingFileHandler(
        filename=log_path / log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream handler (stdout)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def get_app_logger(
    log_dir: str = _DEFAULT_LOG_DIR,
) -> logging.Logger:
    """Get the application logger for general events.

    Args:
        log_dir: Directory for log files.

    Returns:
        Application logger.
    """
    return _create_rotating_logger(
        name="app",
        log_file="app.log",
        log_dir=log_dir,
    )


def get_api_logger(
    log_dir: str = _DEFAULT_LOG_DIR,
) -> logging.Logger:
    """Get the API logger for API-specific events.

    Args:
        log_dir: Directory for log files.

    Returns:
        API logger.
    """
    return _create_rotating_logger(
        name="api",
        log_file="api.log",
        log_dir=log_dir,
    )


def get_request_logger(
    log_dir: str = _DEFAULT_LOG_DIR,
) -> logging.Logger:
    """Get the request logger for HTTP request details.

    Args:
        log_dir: Directory for log files.

    Returns:
        Request logger.
    """
    return _create_rotating_logger(
        name="requests",
        log_file="requests.log",
        log_dir=log_dir,
    )


def get_error_logger(
    log_dir: str = _DEFAULT_LOG_DIR,
) -> logging.Logger:
    """Get the error logger for error events with stack traces.

    Args:
        log_dir: Directory for log files.

    Returns:
        Error logger.
    """
    return _create_rotating_logger(
        name="errors",
        log_file="errors.log",
        log_dir=log_dir,
        level=logging.ERROR,
    )
