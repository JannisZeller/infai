import logging
import os

from src.config.models import Config


def configure_logging(config: Config):
    """
    Configure logging to suppress SQLAlchemy console output.

    Args:
        config: The configuration. If the logging path is set, logs SQL to the path.
        If the logging path is not set, logs SQL to the console.
    """
    logging_path = config.logging_path
    if logging_path:
        # Create logs directory if it doesn't exist
        os.makedirs(logging_path, exist_ok=True)

        # Configure SQLAlchemy logger to write to file only
        sqlalchemy_logger = logging.getLogger("sqlalchemy.engine")
        sqlalchemy_logger.setLevel(logging.INFO)
        sqlalchemy_logger.handlers.clear()

        # Add file handler
        file_handler = logging.FileHandler(logging_path / "sqlalchemy.log")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        sqlalchemy_logger.addHandler(file_handler)

        # Prevent propagation to root logger (suppresses console output)
        sqlalchemy_logger.propagate = False
    else:
        # Just suppress SQLAlchemy logging entirely
        logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
