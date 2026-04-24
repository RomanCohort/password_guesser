"""
Structured logging utilities.

Provides a JSON-formatted log formatter and project-level logging setup.
"""

import logging
import json
import sys
import os
from datetime import datetime
from typing import Optional, Dict, Any


class JSONFormatter(logging.Formatter):
    """Log formatter that emits one JSON object per line.

    Each log record is serialised as a JSON object containing timestamp,
    level, logger name, message, source location, and optional extra data
    or exception traceback.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        if record.exc_info and record.exc_info[0] is not None:
            log_entry['exception'] = self.formatException(record.exc_info)

        if hasattr(record, 'extra_data'):
            log_entry['data'] = record.extra_data

        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging(
    level: str = 'INFO',
    log_file: Optional[str] = None,
    json_format: bool = True,
    log_dir: str = 'logs',
) -> logging.Logger:
    """Configure the project root logger.

    Sets up both console (stderr) and optional file output.  When
    ``json_format`` is ``True`` the :class:`JSONFormatter` is used;
    otherwise a human-readable format is applied.

    Args:
        level: Logging level string (``'DEBUG'``, ``'INFO'``, etc.).
        log_file: Optional filename under *log_dir* for file output.
        json_format: Use JSON formatting if ``True``.
        log_dir: Directory for log files.

    Returns:
        The configured root logger for the ``'password_guesser'`` namespace.
    """
    root_logger = logging.getLogger('password_guesser')
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicate handlers on repeated calls
    if root_logger.handlers:
        root_logger.handlers.clear()

    # Choose formatter
    if json_format:
        formatter: logging.Formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, log_file)
        file_handler = logging.FileHandler(file_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str = 'password_guesser') -> logging.Logger:
    """Return a logger under the ``password_guesser`` namespace.

    If *name* does not already start with ``'password_guesser'`` it is
    prepended automatically.

    Args:
        name: Logger name or suffix.

    Returns:
        A :class:`logging.Logger` instance.
    """
    if not name.startswith('password_guesser'):
        name = f'password_guesser.{name}'
    return logging.getLogger(name)
