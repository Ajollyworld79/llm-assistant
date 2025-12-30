"""
Project-local Functions module (logging helper)
Provides:
- setup_logger(): config-driven logger setup that prefers `settings.LOG_TO_FILE`.

This is a lightweight replacement of the external Support_agent Functions_module logging behavior,
but focused on being robust and dependency-light for this project.
"""

import os
import sys
import socket
import datetime
from loguru import logger as loguru_logger
from typing import Any

try:
    from concurrent_log_handler import ConcurrentRotatingFileHandler
except Exception:
    ConcurrentRotatingFileHandler = None

# Try to import project settings if available
try:
    from backend.app.config import settings as app_settings
except Exception:
    try:
        from ..config import settings as app_settings
    except Exception:
        app_settings = None

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"



def setup_logger(log_dir: str | None = None) -> Any:
    """Setup and return a configured loguru Logger.

    Behavior:
    - Looks for `app_settings.LOG_TO_FILE` (if available) or env `LOG_TO_FILE` to decide whether to create a file handler.
    - If file logging is enabled, creates a dated log file using loguru.
    - Always installs a stderr handler so logs appear in the terminal.

    Returns a loguru Logger instance.
    """

    # Server name and script name for context
    try:
        server = socket.gethostname()
    except Exception:
        server = "localhost"

    try:
        if sys.argv and ("uvicorn" in sys.argv[0] or "gunicorn" in sys.argv[0]):
            script_name = "app.py"
        else:
            file_path = getattr(sys.modules.get("__main__", None), "__file__", None)
            if file_path:
                script_name = os.path.basename(file_path)
                if script_name == "__main__.py":
                    try:
                        argv_name = os.path.basename(sys.argv[0])
                        if argv_name and argv_name.endswith('.py') and not argv_name.startswith('<'):
                            script_name = argv_name
                    except Exception:
                        pass
            else:
                script_name = os.path.basename(sys.argv[0]) if sys.argv else "unknown"
    except Exception:
        script_name = "unknown"

    # Decide whether to log to file
    log_to_file = True
    try:
        if app_settings is not None and hasattr(app_settings, 'LOG_TO_FILE'):
            log_to_file = bool(getattr(app_settings, 'LOG_TO_FILE'))
        else:
            log_to_file = os.getenv('LOG_TO_FILE', '1').lower() not in ('0', 'false', 'f', 'no', 'n')
    except Exception:
        log_to_file = os.getenv('LOG_TO_FILE', '1').lower() not in ('0', 'false', 'f', 'no', 'n')

    # Resolve log directory
    if log_dir is None:
        log_dir = os.getenv('LOG_PATH') or os.path.join(os.path.dirname(__file__), '..', '..', 'logs')
    log_dir = os.path.abspath(log_dir)

    # Configure loguru logger (preferred) and add handlers
    loguru_logger.remove()

    # Prepare a log format that mirrors the standard format
    logger_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{extra[server_name]}</cyan> | <cyan>{extra[script_name]}</cyan> | <level>{message}</level>"

    if log_to_file:
        try:
            os.makedirs(log_dir, exist_ok=True)
            date_str = datetime.datetime.now().strftime('%d-%m-%Y')
            log_path = os.path.join(log_dir, f"logger_{date_str}.log")

            # Use loguru for file logging
            loguru_logger.add(log_path, level="INFO", format=logger_format, encoding="utf-8", enqueue=True, backtrace=False, diagnose=False)
            print(f"Logger file handler added: {log_path}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Could not create log file handler: {e}", file=sys.stderr)

    # Always add stderr/console handler
    loguru_logger.add(sys.stderr, level="INFO", format=logger_format, colorize=True)

    # Return a bound loguru logger to allow `.bind()` and consistent extra fields
    return loguru_logger.bind(server_name=server, script_name=script_name)
