"""Logging utilities with Rich formatting."""

from __future__ import annotations

import logging
from typing import Optional

from rich.logging import RichHandler

_LOG_FORMAT = "%(message)s"
_DATE_FORMAT = "[%X]"

# Registry so we don't add duplicate handlers on repeated calls.
_configured: set[str] = set()


def get_logger(
    name: str,
    level: int = logging.INFO,
    *,
    rich_tracebacks: bool = True,
    markup: bool = True,
    show_path: bool = False,
) -> logging.Logger:
    """Return (and lazily configure) a Rich-formatted logger.

    Parameters
    ----------
    name:
        Logger name, typically ``__name__`` of the calling module.
    level:
        Logging level (default ``logging.INFO``).
    rich_tracebacks:
        Whether Rich should render exception tracebacks.
    markup:
        Allow Rich markup in log messages.
    show_path:
        Show the source file path next to log records.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Examples
    --------
    >>> from world_model_lens.utils.logging import get_logger
    >>> log = get_logger(__name__)
    >>> log.info("Initialised [bold green]world_model_lens[/]")
    """
    logger = logging.getLogger(name)

    if name not in _configured:
        logger.setLevel(level)

        handler = RichHandler(
            level=level,
            rich_tracebacks=rich_tracebacks,
            markup=markup,
            show_path=show_path,
            log_time_format=_DATE_FORMAT,
        )
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)

        # Don't propagate to the root logger to avoid double output.
        logger.propagate = False
        _configured.add(name)

    return logger
