"""
Centralized logging.
Uses Loguru for structured, rotating logs.

Usage:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Starting data pipeline...")

Multi-GPU behavior:
    - All ranks log to stdout with their rank prefix
    - Only rank 0 logs to file
    - Pass local_rank when calling get_logger() if inside training process
"""

import os
import sys
from loguru import logger as _loguru_logger


def get_logger(name: str = "", local_rank: int = 0):
    """
    Configure and return Loguru logger.

    Args:
        name: Module name, used as a prefix in log messages.
              Pass __name__ from calling module.
        local_rank: GPU rank of the calling process.
                    Rank 0 logs to file, others to stdout.
    """

    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_dir = os.environ.get("LOG_DIR", "./logs")

    # Remove all default Loguru handlers
    _loguru_logger.remove()

    # Stdout handler -- all ranks, prefixed with rank for clarity
    rank_prefix = f"[rank{local_rank}] " if local_rank > 0 else ""
    stdout_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        f"{rank_prefix}"
        "<cyan>{extra[module]}</cyan> | "
        "<level>{message}</level>"
    )
    _loguru_logger.add(
        sink=sys.stdout,
        format=stdout_format,
        level=log_level,
        colorize=True,
        enqueue=True,
    )

    # File handler -- only rank 0 logs to file
    if local_rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "train.log")

        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{extra[module]} | "
            "{message}"
        )

        _loguru_logger.add(
            sink=log_file,
            format=file_format,
            level=log_level,
            rotation="50 MB",
            retention=5,
            compression="zip",
            enqueue=True,
            encoding="utf-8",
        )

    # Bind the module name so it appears in every log line from this logger
    bound_logger = _loguru_logger.bind(module=name if name else "root")
    return bound_logger


logger = get_logger(__name__)
