import logging
from pathlib import Path
from typing import Literal

from src.config.factory import get_config
from src.config.models import Config


def configure_module_logging(config: Config):
    """
    Configure logging based on the configuration.

    This function handles:
    1. Module-specific logging (e.g., sqlalchemy, fastmcp) to files when UI is console
    2. Creating loggers that can log to console, file, or both
    3. Redirecting all logs to console when UI is not "console"

    Args:
        config: The configuration containing UI settings, logging paths, and module configurations.
    """
    cfg = config.logging
    logging_path = cfg.base_path

    if config.ui != "console":
        _configure_console_logging()
        return

    logging_path.mkdir(parents=True, exist_ok=True)

    # Configure the root logger to show warnings/errors but not info
    # This ensures uncaught exceptions are visible
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", force=True
    )

    for module_name, filename in cfg.module_logging_filename_dict.items():
        _configure_module_file_logging(
            module_name=module_name, log_file_path=logging_path / filename, level=logging.INFO
        )


def _configure_module_file_logging(module_name: str, log_file_path: Path | str, level: int = logging.INFO):
    """
    Configure a specific module's logger to write to a file.

    INFO and DEBUG logs go to file only.
    WARNING, ERROR, and CRITICAL logs go to both file and console.

    Args:
        module_name: The name of the module/logger to configure (e.g., "sqlalchemy.engine")
        log_file_path: Path to the log file
        level: Logging level (default: INFO)
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Add file handler for all levels
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Add console handler for WARNING and above (errors/exceptions)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Prevent propagation to root logger
    logger.propagate = False


def _configure_console_logging():
    """Configure root logger for console output only."""
    logging.basicConfig(
        level=logging.WARNING,  # Show warnings and errors by default
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )


def get_logger(
    name: str,
    output: Literal["console", "file", "both"] = "console",
    level: int = logging.INFO,
    config: Config | None = None,
    simple_format: bool = False,
) -> logging.Logger:
    """
    Create and configure a logger with the specified output destination.

    This function respects the UI setting: if UI is not "console", all loggers
    will output to console regardless of the output parameter.

    Args:
        name: Name for the logger (typically __name__)
        config: The application configuration
        output: Where to send logs - "console", "file", or "both"
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance

    Example:
        # Console only (default)
        logger = get_logger(__name__, config)

        # File only (when UI is console)
        logger = get_logger(__name__, config, output="file")

        # Both console and file
        logger = get_logger(__name__, config, output="both")
    """

    config = config or get_config()
    cfg = config.logging

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    if simple_format:
        formatter = logging.Formatter("%(message)s")
    else:
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if output in ("console", "both"):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if output in ("file", "both"):
        if cfg.base_path:
            cfg.base_path.mkdir(parents=True, exist_ok=True)
            log_file = cfg.base_path / cfg.main_logging_filename

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        else:
            # No logging path configured, fall back to console
            if output == "file":
                console_handler = logging.StreamHandler()
                console_handler.setLevel(level)
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)

    return logger
