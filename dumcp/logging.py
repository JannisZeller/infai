import logging
import os
from pathlib import Path


def configure_logging():
    """
    Configure logging for the MCP subprocess.

    - All logs (INFO+) go to the log file
    - ERROR and CRITICAL logs also go to stdout (console)
    """
    log_level_str = os.getenv("MCP_LOG_LEVEL", "INFO")
    log_file = os.getenv("MCP_LOG_FILE", None)

    # Parse log level
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if log_file:
        # Running as subprocess - set up dual logging
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # File handler for all logs
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        # Console handler for ERROR and above
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    else:
        # Running standalone - normal console logging
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
