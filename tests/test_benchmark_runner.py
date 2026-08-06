"""Tests for the benchmark runner."""

import torch
import pytest

from world_model_lens.benchmarks.perf_runner import BenchmarkResult


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_result_creation(self):
        """Test creating a benchmark result."""
        result = BenchmarkResult(
            benchmark_type="latency",
            latency_ms=1.5,
            memory_mb=100.0,
            throughput_samples_per_sec=1000.0,
            metadata={"device": "cpu"},
        )
        assert result.benchmark_type == "latency"
        assert result.latency_ms == 1.5
        assert result.memory_mb == 100.0
        assert result.throughput_samples_per_sec == 1000.0

    def test_result_with_empty_metadata(self):
        """Test creating result with empty metadata."""
        result = BenchmarkResult(
            benchmark_type="memory",
            latency_ms=0.0,
            memory_mb=50.0,
            throughput_samples_per_sec=500.0,
        )
        assert result.benchmark_type == "memory"
        assert result.metadata == {}
