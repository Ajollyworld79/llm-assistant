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
import logging
import datetime
from loguru import logger as loguru_logger
from typing import Any

# Try to import project settings if available
try:
    from backend.app.config import settings as app_settings
except Exception:
    try:
        from ..config import settings as app_settings
    except Exception:
        app_settings = None

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


class StdlibStyleLogger:
    """Loguru logger that accepts the stdlib call convention used across the codebase.

    Call sites write `logger.info("model=%s", name)` and `exc_info=True`. Loguru
    formats with str.format, so without this adapter the literal "%s" is logged
    and the arguments are dropped.
    """

    def __init__(self, inner: Any):
        self._inner = inner

    def _log(self, level: str, msg: Any, args: tuple, exc_info: Any = None) -> None:
        text = str(msg)
        if args:
            try:
                text = text % args
            except (TypeError, ValueError):
                text = " ".join([text, *map(str, args)])
        # "{}" + positional arg keeps braces inside `text` from being treated as format fields.
        self._inner.opt(depth=2, exception=exc_info or None).log(level, "{}", text)

    def debug(self, msg: Any, *args: Any, exc_info: Any = None, **_: Any) -> None:
        self._log("DEBUG", msg, args, exc_info)

    def info(self, msg: Any, *args: Any, exc_info: Any = None, **_: Any) -> None:
        self._log("INFO", msg, args, exc_info)

    def warning(self, msg: Any, *args: Any, exc_info: Any = None, **_: Any) -> None:
        self._log("WARNING", msg, args, exc_info)

    warn = warning

    def error(self, msg: Any, *args: Any, exc_info: Any = None, **_: Any) -> None:
        self._log("ERROR", msg, args, exc_info)

    def critical(self, msg: Any, *args: Any, exc_info: Any = None, **_: Any) -> None:
        self._log("CRITICAL", msg, args, exc_info)

    def exception(self, msg: Any, *args: Any, exc_info: Any = True, **_: Any) -> None:
        self._log("ERROR", msg, args, exc_info)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _InterceptHandler(logging.Handler):
    """Route stdlib `logging` records (rate_limiter, usage_db, crypto, ...) into loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: Any = loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        loguru_logger.opt(exception=record.exc_info or None).log(level, "{}", record.getMessage())


_configured = False


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

    global _configured
    if not _configured:
        # Every module calls setup_logger(); handlers are installed only once.
        _configured = True
        loguru_logger.remove()
        loguru_logger.configure(extra={"server_name": server, "script_name": script_name})

        # Prepare a log format that mirrors the standard format
        logger_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{extra[server_name]}</cyan> | <cyan>{extra[script_name]}</cyan> | <level>{message}</level>"

        if log_to_file:
            try:
                os.makedirs(log_dir, exist_ok=True)
                date_str = datetime.datetime.now().strftime('%d-%m-%Y')
                log_path = os.path.join(log_dir, f"logger_{date_str}.log")
                loguru_logger.add(log_path, level="INFO", format=logger_format, encoding="utf-8", enqueue=True, backtrace=False, diagnose=False)
                print(f"Logger file handler added: {log_path}", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Could not create log file handler: {e}", file=sys.stderr)

        # Always add stderr/console handler
        loguru_logger.add(sys.stderr, level="INFO", format=logger_format, colorize=True)

        logging.basicConfig(handlers=[_InterceptHandler()], level=logging.INFO, force=True)
        for noisy in ("httpx", "httpcore", "google_genai"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    return StdlibStyleLogger(loguru_logger.bind(server_name=server, script_name=script_name))
