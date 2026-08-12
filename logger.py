from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    project_name: str,
    log_file: Optional[Path] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    force_reconfigure: bool = True,
) -> logging.Logger:
    """Configure and return a project logger.

    Reuses the structure of the previous project logger, but keeps the API
    minimal and aligned with this repository.
    """

    logger = logging.getLogger(project_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if logger.hasHandlers():
        if force_reconfigure:
            logger.handlers.clear()
        else:
            return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.info("=" * 50)
    logger.info(f"PROJECT LOGGING INITIALIZED: {project_name}")
    if log_file is not None:
        logger.info(f"Log file: {log_file}")
    logger.info(
        f"Console level: {logging.getLevelName(console_level)}, File level: {logging.getLevelName(file_level)}"
    )
    logger.info("=" * 50)

    return logger
