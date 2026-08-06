"""Rich-formatted logging utilities."""

from typing import Optional
from rich.logging import RichHandler
from rich.console import Console
import logging


_console = Console()


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Get a Rich-formatted logger.

    Args:
        name: Logger name, typically __name__ of the calling module.
        level: Optional logging level. Defaults to INFO.

    Returns:
        A configured logger with Rich output formatting.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = RichHandler(
            console=_console,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            markup=True,
        )
        formatter = logging.Formatter(
            "%(message)s",
            datefmt="[%X]",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if level is None:
        level = logging.INFO
    logger.setLevel(level)

    return logger
