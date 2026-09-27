"""
Central logging configuration for Kokoro AI.

Logs are written to:
    logs/kokoro.log

The logger also writes to the console during local development.

The logging layer is kept separate from business logic so that
all Kokoro modules can use the same logging configuration.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOG_DIR = PROJECT_ROOT / "logs"

LOG_FILE = LOG_DIR / "kokoro.log"


# ============================================================
# Logger Configuration
# ============================================================

def get_logger(name: str = "kokoro") -> logging.Logger:
    """
    Return a configured Kokoro logger.

    Parameters
    ----------
    name:
        Logger name.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """

    # Create logs directory if it doesn't exist.
    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers when Streamlit reloads
    # the application or the module is imported multiple times.
    if logger.handlers:
        return logger

    # ========================================================
    # Log Format
    # ========================================================

    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ========================================================
    # File Handler
    # ========================================================

    file_handler = RotatingFileHandler(
        filename=LOG_FILE,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )

    file_handler.setLevel(logging.INFO)

    file_handler.setFormatter(formatter)

    # ========================================================
    # Console Handler
    # ========================================================

    console_handler = logging.StreamHandler()

    console_handler.setLevel(logging.INFO)

    console_handler.setFormatter(formatter)

    # ========================================================
    # Register Handlers
    # ========================================================

    logger.addHandler(file_handler)

    logger.addHandler(console_handler)

    # Prevent messages from being propagated to the
    # root logger and appearing twice.
    logger.propagate = False

    return logger


# ============================================================
# Global Kokoro Logger
# ============================================================

logger = get_logger()