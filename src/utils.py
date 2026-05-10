"""Shared utility functions for the project."""

import logging
from pathlib import Path
from typing import Optional


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger with a simple stream handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parents[1]


def data_dir(subdir: Optional[str] = "raw") -> Path:
    """Return a path under the data/ directory, creating it if needed."""
    path = project_root() / "data" / (subdir or "")
    path.mkdir(parents=True, exist_ok=True)
    return path


def reports_dir() -> Path:
    """Return the reports/ directory, creating it if needed."""
    path = project_root() / "reports"
    path.mkdir(exist_ok=True)
    return path
