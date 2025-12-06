import logging
import os


def configure_logging(enable_sql_file_logging: bool = False):
    """
    Configure logging to suppress SQLAlchemy console output.

    Args:
        enable_sql_file_logging: If True, logs SQL to data/logs/sqlalchemy.log
    """
    if enable_sql_file_logging:
        # Create logs directory if it doesn't exist
        os.makedirs("data/logs", exist_ok=True)

        # Configure SQLAlchemy logger to write to file only
        sqlalchemy_logger = logging.getLogger("sqlalchemy.engine")
        sqlalchemy_logger.setLevel(logging.INFO)
        sqlalchemy_logger.handlers.clear()

        # Add file handler
        file_handler = logging.FileHandler("data/logs/sqlalchemy.log")
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
