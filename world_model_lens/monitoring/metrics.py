"""Prometheus metrics collection for World Model Lens."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional
from functools import wraps
import threading

try:
    from prometheus_client import Counter, Gauge, Histogram, Summary, generate_latest
    from prometheus_client import CollectorRegistry

    PROMETHUES_AVAILABLE = True
except ImportError:
    PROMETHUES_AVAILABLE = False


@dataclass
class MetricSnapshot:
    """Snapshot of metrics at a point in time."""

    timestamp: datetime
    analysis_count: Counter
    analysis_duration_seconds: Histogram
    memory_usage_bytes: Gauge
    gpu_memory_usage_bytes: Gauge
    active_analyses: Gauge
    cache_hits: Counter
    cache_misses: Counter


class MetricsCollector:
    """Collects and exports Prometheus metrics.

    Note: each MetricsCollector instance uses its own Prometheus
    CollectorRegistry. This prevents duplicate metric registration when
    multiple instances (or tests) create collectors in the same process.
    """

    def __init__(self) -> None:
        if not PROMETHUES_AVAILABLE:
            self._available = False
            return

        self._available = True
        self._lock = threading.Lock()
        # Use a dedicated registry per collector instance to avoid global
        # duplicate metric registration when tests or multiple instances run.
        self._registry = CollectorRegistry()

        self.analysis_count = Counter(
            "wml_analysis_total",
            "Total number of analyses",
            ["analysis_type", "status"],
            registry=self._registry,
        )

        self.analysis_duration = Histogram(
            "wml_analysis_duration_seconds",
            "Analysis duration in seconds",
            ["analysis_type"],
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
            registry=self._registry,
        )

        self.memory_usage = Gauge(
            "wml_memory_usage_bytes",
            "Memory usage in bytes",
            ["type"],
            registry=self._registry,
        )

        self.gpu_memory_usage = Gauge(
            "wml_gpu_memory_usage_bytes",
            "GPU memory usage in bytes",
            ["device_id"],
            registry=self._registry,
        )

        self.active_analyses = Gauge(
            "wml_active_analyses",
            "Number of active analyses",
            registry=self._registry,
        )

        self.cache_hits = Counter(
            "wml_cache_hits_total",
            "Cache hit count",
            ["cache_name"],
            registry=self._registry,
        )

        self.cache_misses = Counter(
            "wml_cache_misses_total",
            "Cache miss count",
            ["cache_name"],
            registry=self._registry,
        )

        self.request_count = Counter(
            "wml_requests_total",
            "Total API requests",
            ["endpoint", "method", "status"],
            registry=self._registry,
        )

        self.request_duration = Histogram(
            "wml_request_duration_seconds",
            "Request duration in seconds",
            ["endpoint"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self._registry,
        )

        self.probe_accuracy = Gauge(
            "wml_probe_accuracy",
            "Probe accuracy score",
            ["concept", "task"],
            registry=self._registry,
        )

        self.trajectory_length = Histogram(
            "wml_trajectory_length",
            "Trajectory length in timesteps",
            ["source"],
            buckets=(10, 50, 100, 500, 1000, 5000, 10000),
            registry=self._registry,
        )

    def record_analysis(
        self,
        analysis_type: str,
        duration: float,
        success: bool,
    ) -> None:
        """Record an analysis metric."""
        if not self._available:
            return

        status = "success" if success else "failure"
        self.analysis_count.labels(
            analysis_type=analysis_type,
            status=status,
        ).inc()

        self.analysis_duration.labels(
            analysis_type=analysis_type,
        ).observe(duration)

    def record_cache_access(
        self,
        cache_name: str,
        hit: bool,
    ) -> None:
        """Record cache access."""
        if not self._available:
            return

        if hit:
            self.cache_hits.labels(cache_name=cache_name).inc()
        else:
            self.cache_misses.labels(cache_name=cache_name).inc()

    def record_memory_usage(
        self,
        memory_type: str,
        bytes_used: int,
    ) -> None:
        """Record memory usage."""
        if not self._available:
            return

        self.memory_usage.labels(type=memory_type).set(bytes_used)

    def record_gpu_memory_usage(
        self,
        device_id: int,
        bytes_used: int,
    ) -> None:
        """Record GPU memory usage."""
        if not self._available:
            return

        self.gpu_memory_usage.labels(device_id=device_id).set(bytes_used)

    def set_active_analyses(self, count: int) -> None:
        """Set number of active analyses."""
        if not self._available:
            return

        self.active_analyses.set(count)

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        if not PROMETHUES_AVAILABLE:
            return "# Metrics not available"

        return generate_latest(self._registry).decode("utf-8")

    def start(self) -> None:
        """Start background metric collection."""
        pass

    def stop(self) -> None:
        """Stop background metric collection."""
        pass


def track_request_latency(
    metrics: MetricsCollector,
) -> Callable:
    """Decorator to track request latency."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start
                metrics.request_duration.labels(
                    endpoint=func.__name__,
                ).observe(duration)

        return wrapper

    return decorator


def track_memory_usage(metrics: MetricsCollector) -> Callable:
    """Decorator to track memory usage."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import psutil
            import os

            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss
            result = func(*args, **kwargs)
            mem_after = process.memory_info().rss
            metrics.record_memory_usage("process", mem_after)
            return result

        return wrapper

    return decorator
