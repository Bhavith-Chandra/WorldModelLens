"""Monitoring module for World Model Lens.

Provides:
- Structured logging with OpenTelemetry
- Prometheus metrics
- Distributed tracing
"""

from world_model_lens.monitoring.logging import setup_logging, get_logger
from world_model_lens.monitoring.metrics import MetricsCollector
from world_model_lens.monitoring.tracing import setup_tracing, trace_function

__all__ = [
    "setup_logging",
    "get_logger",
    "MetricsCollector",
    "setup_tracing",
    "trace_function",
]
