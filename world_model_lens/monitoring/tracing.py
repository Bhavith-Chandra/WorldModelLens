"""Distributed tracing with OpenTelemetry."""

from __future__ import annotations

from typing import Any, Callable, Optional
from functools import wraps

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.trace import Status, StatusCode

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False


_tracer: Optional[Any] = None


def setup_tracing(
    service_name: str = "world-model-lens",
    endpoint: Optional[str] = None,
) -> None:
    """Setup OpenTelemetry tracing.

    Args:
        service_name: Name of the service
        endpoint: OTLP endpoint (optional)
    """
    global _tracer

    if not OTEL_AVAILABLE:
        return

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": "0.2.0",
            "deployment.environment": "production",
        }
    )

    provider = TracerProvider(resource=resource)

    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            exporter = OTLPSpanExporter(endpoint=endpoint)
            processor = BatchSpanProcessor(exporter)
            provider.add_span_processor(processor)
        except Exception:
            pass

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(service_name)


def get_tracer() -> Optional[Any]:
    """Get the global tracer."""
    global _tracer
    if _tracer is None and OTEL_AVAILABLE:
        _tracer = trace.get_tracer("world-model-lens")
    return _tracer


def trace_function(
    name: Optional[str] = None,
    attributes: Optional[dict[str, Any]] = None,
) -> Callable:
    """Decorator to trace a function.

    Args:
        name: Span name (defaults to function name)
        attributes: Additional span attributes

    Example:
        @trace_function("my_analysis")
        def my_analysis(data):
            ...
    """

    def decorator(func: Callable) -> Callable:
        span_name = name or func.__name__

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            if tracer is None:
                return func(*args, **kwargs)

            with tracer.start_as_current_span(span_name) as span:
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)

                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper

    return decorator


def trace_async(
    name: Optional[str] = None,
    attributes: Optional[dict[str, Any]] = None,
) -> Callable:
    """Decorator to trace an async function."""

    def decorator(func: Callable) -> Callable:
        span_name = name or func.__name__

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            if tracer is None:
                return await func(*args, **kwargs)

            with tracer.start_as_current_span(span_name) as span:
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)

                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper

    return decorator


class SpanContext:
    """Context manager for creating spans."""

    def __init__(
        self,
        name: str,
        attributes: Optional[dict[str, Any]] = None,
    ):
        self.name = name
        self.attributes = attributes or {}
        self.tracer = get_tracer()
        self.span = None

    def __enter__(self):
        if self.tracer:
            self.span = self.tracer.start_span(self.name)
            for key, value in self.attributes.items():
                self.span.set_attribute(key, value)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            if exc_type:
                self.span.set_status(Status(StatusCode.ERROR, str(exc_val)))
                self.span.record_exception(exc_val)
            else:
                self.span.set_status(Status(StatusCode.OK))
            self.span.end()
        return False

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        if self.span:
            self.span.set_attribute(key, value)

    def add_event(self, name: str, attributes: Optional[dict[str, Any]] = None) -> None:
        """Add an event to the span."""
        if self.span:
            self.span.add_event(name, attributes or {})
