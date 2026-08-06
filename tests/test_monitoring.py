"""Tests for monitoring modules."""

import pytest
from world_model_lens.monitoring.logging import get_logger


class TestLogging:
    """Tests for logging module."""

    def test_get_logger(self):
        """Test getting a logger."""
        logger = get_logger("test")
        assert logger is not None
        assert logger.name == "test"

    def test_get_logger_different_names(self):
        """Test getting loggers with different names."""
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")
        assert logger1.name == "module1"
        assert logger2.name == "module2"
        assert logger1 is not logger2

    def test_logger_has_standard_methods(self):
        """Test logger has standard logging methods."""
        logger = get_logger("test.methods")
        assert hasattr(logger, "debug")
        assert hasattr(logger, "info")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")
        assert hasattr(logger, "critical")


class TestMonitoringModuleExports:
    """Tests for monitoring module exports."""

    def test_logging_exports(self):
        """Test logging module exports."""
        from world_model_lens.monitoring import logging as monitoring_logging

        assert hasattr(monitoring_logging, "get_logger")
        assert hasattr(monitoring_logging, "setup_logging")

    def test_metrics_exports(self):
        """Test metrics module exports."""
        from world_model_lens.monitoring import metrics as monitoring_metrics

        assert hasattr(monitoring_metrics, "MetricsCollector")

    def test_tracing_exports(self):
        """Test tracing module exports."""
        from world_model_lens.monitoring import tracing as monitoring_tracing

        assert hasattr(monitoring_tracing, "setup_tracing")
        assert hasattr(monitoring_tracing, "trace_function")


class TestMetricsCollectorInstantiation:
    """Tests for MetricsCollector class."""

    def test_collector_can_be_created(self):
        """Test MetricsCollector can be instantiated."""
        from world_model_lens.monitoring.metrics import MetricsCollector

        collector = MetricsCollector()
        assert collector is not None

    def test_collector_has_expected_methods(self):
        """Test MetricsCollector has expected methods."""
        from world_model_lens.monitoring.metrics import MetricsCollector

        collector = MetricsCollector()
        assert hasattr(collector, "start")
        assert hasattr(collector, "stop")
        assert hasattr(collector, "record_analysis")
        assert hasattr(collector, "export_prometheus")


class TestTracingSetup:
    """Tests for tracing module."""

    def test_trace_function_decorator_exists(self):
        """Test trace_function decorator exists."""
        from world_model_lens.monitoring.tracing import trace_function

        assert callable(trace_function)
