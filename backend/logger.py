import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from collections import deque
from typing import List, Dict, Any, Optional
from datetime import datetime

LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOGS_DIR, "backend.log")

class RingBufferLogHandler(logging.Handler):
    """Stores the most recent log records in memory for quick querying via API."""
    def __init__(self, capacity: int = 1000):
        super().__init__()
        self.buffer = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord):
        try:
            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "level": record.levelname,
                "module": record.module,
                "filename": record.filename,
                "lineno": record.lineno,
                "message": record.getMessage()
            }
            self.buffer.append(log_entry)
        except Exception:
            self.handleError(record)

    def get_logs(self, limit: int = 100, min_level: Optional[str] = None) -> List[Dict[str, Any]]:
        level_values = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        min_val = level_values.get((min_level or "").upper(), logging.DEBUG)
        
        logs = []
        for entry in reversed(self.buffer):
            entry_level_val = level_values.get(entry["level"], logging.INFO)
            if entry_level_val >= min_val:
                logs.append(entry)
                if len(logs) >= limit:
                    break
        return logs[::-1]

    def clear(self):
        self.buffer.clear()


# Global ring buffer handler
ring_handler = RingBufferLogHandler(capacity=2000)
ring_handler.setLevel(logging.DEBUG)

# Logger setup
logger = logging.getLogger("rf_backend")
logger.setLevel(logging.DEBUG)
logger.propagate = False

# Avoid duplicate handlers if reloaded
if not logger.handlers:
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)-7s] [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # Rotating File Handler (20 MB max per file, 5 backups)
    file_handler = RotatingFileHandler(
        LOG_FILE_PATH,
        maxBytes=20 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)-7s] [%(name)s] [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    # In-memory buffer handler
    logger.addHandler(ring_handler)

def get_recent_logs(limit: int = 100, min_level: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve the most recent log entries from the ring buffer."""
    return ring_handler.get_logs(limit=limit, min_level=min_level)

def clear_recent_logs():
    """Clear memory log buffer."""
    ring_handler.clear()
