import logging
import sys
import os
from rich.logging import RichHandler

# Project root: obstacle_avoidance_mission/
# Dùng realpath để luôn resolve đúng bất kể CWD khi chạy script.
_THIS_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.dirname(_THIS_DIR)

def _env_flag_enabled(name: str) -> bool:
    value = os.getenv(name, "")
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class _PrefixAllowFilter(logging.Filter):
    def __init__(self, prefixes):
        super().__init__()
        self.prefixes = tuple(str(p) for p in (prefixes or []))

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        if record.levelno < logging.INFO:
            return False
        message = record.getMessage()
        return any(message.startswith(prefix) for prefix in self.prefixes)


def setup_logger(
    name,
    log_file=None,
    level=logging.DEBUG,
    console_level=logging.INFO,
    force_console_debug=False,
    info_log_file=None,
    debug_log_file=None,
    file_prefix_allowlist=None,
):
    """
    Sets up a logger that outputs INFO and above to the console using rich,
    and DEBUG and above to the specified log_file (if provided).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent propagation to root logger to avoid duplicate prints
    logger.propagate = False
    
    # Remove existing handlers to avoid duplicates if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    console_debug_enabled = bool(force_console_debug or _env_flag_enabled("DRONE_RL_CONSOLE_DEBUG"))
    effective_console_level = logging.DEBUG if console_debug_enabled else max(int(console_level), int(logging.INFO))

    # Console Handler (Rich)
    rich_handler = RichHandler(rich_tracebacks=True, show_time=True, show_path=False, markup=True)
    rich_handler.setLevel(effective_console_level)
    # RichHandler has its own formatter
    console_format = logging.Formatter("%(message)s")
    rich_handler.setFormatter(console_format)
    logger.addHandler(rich_handler)
    
    # Legacy single file handler
    if log_file and not info_log_file and not debug_log_file:
        # Ensure log_file is absolute relative to project root
        if not os.path.isabs(log_file):
            log_file = os.path.join(PROJECT_ROOT, log_file)
            
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(level)
        file_format = logging.Formatter("[%(asctime)s][%(levelname)s][%(name)s] %(message)s")
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    # Split main/debug file handlers
    if info_log_file:
        if not os.path.isabs(info_log_file):
            info_log_file = os.path.join(PROJECT_ROOT, info_log_file)
        os.makedirs(os.path.dirname(os.path.abspath(info_log_file)), exist_ok=True)
        info_handler = logging.FileHandler(info_log_file, mode='a', encoding='utf-8')
        info_handler.setLevel(logging.INFO)
        info_handler.setFormatter(
            logging.Formatter("[%(asctime)s][%(levelname)s][%(name)s] %(message)s")
        )
        if file_prefix_allowlist:
            info_handler.addFilter(_PrefixAllowFilter(file_prefix_allowlist))
        logger.addHandler(info_handler)

    if debug_log_file:
        if not os.path.isabs(debug_log_file):
            debug_log_file = os.path.join(PROJECT_ROOT, debug_log_file)
        os.makedirs(os.path.dirname(os.path.abspath(debug_log_file)), exist_ok=True)
        debug_handler = logging.FileHandler(debug_log_file, mode='a', encoding='utf-8')
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(
            logging.Formatter("[%(asctime)s][%(levelname)s][%(name)s] %(message)s")
        )
        logger.addHandler(debug_handler)
        
    return logger
