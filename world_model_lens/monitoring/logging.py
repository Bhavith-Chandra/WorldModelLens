"""Structured logging with OpenTelemetry support."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from typing import Any, Optional
import json

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False


class JSONFormatter(logging.Formatter):
    """Format logs as JSON for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra"):
            log_data.update(record.extra)

        return json.dumps(log_data)


class StructuredLogger:
    """Structured logger with context and OpenTelemetry integration."""

    def __init__(
        self,
        name: str,
        level: int = logging.INFO,
        json_format: bool = False,
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.name = name
        self._context: dict[str, Any] = {}

        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            if json_format:
                handler.setFormatter(JSONFormatter())
            else:
                handler.setFormatter(
                    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
                )
            self.logger.addHandler(handler)

    def set_context(self, **kwargs: Any) -> None:
        """Set context for all log messages."""
        self._context.update(kwargs)

    def clear_context(self) -> None:
        """Clear all context."""
        self._context.clear()

    def _format_message(
        self,
        msg: str,
        extra: Optional[dict[str, Any]] = None,
    ) -> tuple[str, dict[str, Any]]:
        """Format message with context."""
        context = self._context.copy()
        if extra:
            context.update(extra)
        return msg, context

    def debug(self, msg: str, **kwargs: Any) -> None:
        """Log debug message."""
        msg, extra = self._format_message(msg, kwargs)
        self.logger.debug(msg, extra=extra)

    def info(self, msg: str, **kwargs: Any) -> None:
        """Log info message."""
        msg, extra = self._format_message(msg, kwargs)
        self.logger.info(msg, extra=extra)

    def warning(self, msg: str, **kwargs: Any) -> None:
        """Log warning message."""
        msg, extra = self._format_message(msg, kwargs)
        self.logger.warning(msg, extra=extra)

    def error(self, msg: str, **kwargs: Any) -> None:
        """Log error message."""
        msg, extra = self._format_message(msg, kwargs)
        self.logger.error(msg, extra=extra)

    def critical(self, msg: str, **kwargs: Any) -> None:
        """Log critical message."""
        msg, extra = self._format_message(msg, kwargs)
        self.logger.critical(msg, extra=extra)


_loggers: dict[str, StructuredLogger] = {}


def setup_logging(
    level: int = logging.INFO,
    json_format: bool = False,
    service_name: str = "world-model-lens",
) -> None:
    """Setup structured logging with optional OpenTelemetry.

    Args:
        level: Logging level
        json_format: Whether to output JSON format
        service_name: Service name for OpenTelemetry
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        if json_format:
            handler.setFormatter(JSONFormatter())
        else:
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
        root_logger.addHandler(handler)

    if OTEL_AVAILABLE:
        resource = Resource.create(
            {
                "service.name": service_name,
                "service.version": "0.2.0",
            }
        )
        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)


def get_logger(
    name: str,
    level: int = logging.INFO,
    json_format: bool = False,
) -> StructuredLogger:
    """Get or create a structured logger.

    Args:
        name: Logger name
        level: Logging level
        json_format: Whether to output JSON format

    Returns:
        StructuredLogger instance
    """
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name, level, json_format)
    return _loggers[name]
