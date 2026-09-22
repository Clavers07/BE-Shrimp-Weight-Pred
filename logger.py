import os
import logging
from logging.handlers import RotatingFileHandler
from config import LOG_LEVEL, LOG_FILE_PATH

def setup_logger():
    log_dir = os.path.dirname(LOG_FILE_PATH)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    log = logging.getLogger("shrimp_api")
    
    # Avoid duplicate handlers if setup_logger is called multiple times
    if log.hasHandlers():
        log.handlers.clear()

    level = getattr(logging, LOG_LEVEL.upper(), logging.DEBUG)
    log.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    )

    console = logging.StreamHandler()
    console.setFormatter(fmt)

    file_handler = RotatingFileHandler(
        LOG_FILE_PATH, maxBytes=10_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)

    log.addHandler(console)
    log.addHandler(file_handler)

    return log

logger = setup_logger()
