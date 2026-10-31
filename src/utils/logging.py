"""One shared logger setup for the whole project.

Every module does:
    from src.utils.logging import get_logger
    logger = get_logger(__name__)

Logs go to the console and to data/logs/app.log. We use logging instead of print()
so messages have a level (INFO/WARNING/ERROR), a timestamp and the module name.
"""

import logging
from logging.handlers import RotatingFileHandler

from src.config import get_settings

_ROOT_NAME = "rag"  # all project loggers live under "rag.*"
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging() -> logging.Logger:
    """Configure the project's root logger once. Safe to call many times.

    Streamlit re-runs the script on every interaction, so we check for existing
    handlers to avoid printing every message two, three, four... times.
    """
    settings = get_settings()
    root = logging.getLogger(_ROOT_NAME)
    if root.handlers:
        return root

    root.setLevel(settings.log_level.upper())
    formatter = logging.Formatter(_FORMAT, datefmt="%H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    # Keep at most 3 files of 2 MB each, so logs never fill the disk.
    log_file = settings.logs_dir / "app.log"
    file_handler = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    root.propagate = False  # don't send our messages to third-party/root handlers too
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a child logger, e.g. get_logger(__name__) -> 'rag.src.ingestion.chunker'."""
    setup_logging()
    return logging.getLogger(f"{_ROOT_NAME}.{name}")
