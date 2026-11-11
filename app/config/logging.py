import logging
import sys
from app.config.settings import settings


# ------------------------------------------------------------------
# Log format
# ------------------------------------------------------------------
# Example output:
# 2026-09-06 16:10:23,412 | INFO     | app.services.pricing | Price calculated: 299.99
# ------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging() -> None:
    """
    Configure the root logger once at application startup.

    How it works
    ------------
    Python's logging system is hierarchical.
    Every logger (e.g. 'app.services.pricing') is a child of the root logger.
    By configuring the root logger here, every module in the project
    automatically inherits the same format and level — no per-module setup needed.

    Call this function ONCE from main.py before anything else runs.
    """

    # Choose log level based on environment
    # DEBUG in development gives verbose output; INFO in production is clean
    log_level = logging.DEBUG if settings.debug else logging.INFO

    # ------------------------------------------------------------------
    # Handler — where logs go
    # ------------------------------------------------------------------
    # StreamHandler writes to stdout (terminal / container logs)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    handler.setFormatter(logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT))

    # ------------------------------------------------------------------
    # Root logger — parent of every logger in the project
    # ------------------------------------------------------------------
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Avoid adding duplicate handlers if setup_logging() is called more than once
    if not root_logger.handlers:
        root_logger.addHandler(handler)

    # Silence noisy third-party libraries in production
    if not settings.debug:
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    root_logger.info(
        "Logging configured | level=%s | environment=%s",
        logging.getLevelName(log_level),
        settings.environment,
    )
