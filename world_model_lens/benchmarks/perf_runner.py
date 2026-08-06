"""Benchmark runner for world model analysis.

Measures latency, memory usage, and throughput for different analysis types.
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch

from world_model_lens import HookedWorldModel
from world_model_lens.monitoring.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    benchmark_type: str
    latency_ms: float
    memory_mb: float
    throughput_samples_per_sec: float
    metadata: dict[str, Any] = field(default_factory=dict)


class BenchmarkRunner:
    """Runs benchmarks on world model analysis operations."""

    def __init__(
        self,
        checkpoint: str,
        dataset_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialize benchmark runner.

        Args:
            checkpoint: Path to model checkpoint
            dataset_path: Optional path to benchmark dataset
            device: Device to run on (cpu/cuda)
        """
        self.checkpoint = checkpoint
        self.dataset_path = dataset_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.wm: Optional[HookedWorldModel] = None

    def _setup_model(self) -> HookedWorldModel:
        """Load the world model for benchmarking."""
        if self.wm is None:
            self.wm = HookedWorldModel.from_checkpoint(self.checkpoint)
            self.wm = self.wm.to(self.device)
            self.wm.eval()
        return self.wm

    def run(
        self,
        benchmarks: list[str],
        num_samples: int = 1000,
        profile: bool = False,
    ) -> dict[str, Any]:
        """Run specified benchmarks.

        Args:
            benchmarks: List of benchmarks to run (latency, memory, throughput)
            num_samples: Number of samples to use
            profile: Whether to enable profiling

        Returns:
            Dictionary containing benchmark results
        """
        wm = self._setup_model()
        results: dict[str, Any] = {"benchmarks": {}, "num_samples": num_samples}

        if "latency" in benchmarks:
            results["benchmarks"]["latency"] = self._benchmark_latency(wm, num_samples)

        if "memory" in benchmarks:
            results["benchmarks"]["memory"] = self._benchmark_memory(wm, num_samples)

        if "throughput" in benchmarks:
            results["benchmarks"]["throughput"] = self._benchmark_throughput(wm, num_samples)

        if profile:
            results["profiles"] = self._get_profiles(wm)

        return results

    def _benchmark_latency(self, wm: HookedWorldModel, num_samples: int) -> dict[str, float]:
        """Benchmark single operation latency."""
        latencies = []

        observations = torch.randn(1, wm.config.d_obs, device=self.device)

        for _ in range(min(num_samples, 100)):
            start = time.perf_counter()
            _ = wm.forward(observations=observations)
            torch.cuda.synchronize() if self.device != "cpu" else None
            latencies.append((time.perf_counter() - start) * 1000)

        return {
            "mean_ms": sum(latencies) / len(latencies),
            "p50_ms": sorted(latencies)[len(latencies) // 2],
            "p95_ms": sorted(latencies)[int(len(latencies) * 0.95)],
            "p99_ms": sorted(latencies)[int(len(latencies) * 0.99)],
            "min_ms": min(latencies),
            "max_ms": max(latencies),
        }

    def _benchmark_memory(self, wm: HookedWorldModel, num_samples: int) -> dict[str, float]:
        """Benchmark memory usage."""
        if self.device == "cpu":
            gc.collect()
            import psutil

            process = psutil.Process()
            start_mem = process.memory_info().rss / 1024 / 1024

            observations = torch.randn(32, wm.config.d_obs, device=self.device)
            for _ in range(10):
                _ = wm.forward(observations=observations)

            end_mem = process.memory_info().rss / 1024 / 1024
            return {
                "memory_used_mb": end_mem - start_mem,
                "total_memory_mb": end_mem,
            }

        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated() / 1024 / 1024

        observations = torch.randn(32, wm.config.d_obs, device=self.device)
        for _ in range(10):
            _ = wm.forward(observations=observations)

        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
        end_mem = torch.cuda.memory_allocated() / 1024 / 1024

        return {
            "memory_used_mb": end_mem - start_mem,
            "peak_memory_mb": peak_mem,
            "total_memory_mb": torch.cuda.get_device_properties(0).total_memory / 1024 / 1024,
        }

    def _benchmark_throughput(self, wm: HookedWorldModel, num_samples: int) -> dict[str, float]:
        """Benchmark throughput (samples per second)."""
        batch_sizes = [1, 4, 8, 16, 32]
        results = {}

        for batch_size in batch_sizes:
            if batch_size > num_samples:
                continue

            observations = torch.randn(batch_size, wm.config.d_obs, device=self.device)

            for _ in range(3):
                _ = wm.forward(observations=observations)
            torch.cuda.synchronize() if self.device != "cpu" else None

            start = time.perf_counter()
            iterations = max(1, num_samples // batch_size)
            for _ in range(iterations):
                _ = wm.forward(observations=observations)
            torch.cuda.synchronize() if self.device != "cpu" else None
            elapsed = time.perf_counter() - start

            samples_processed = iterations * batch_size
            results[f"batch_{batch_size}"] = {
                "samples_per_sec": samples_processed / elapsed,
                "elapsed_sec": elapsed,
            }

        return results

    def _get_profiles(self, wm: HookedWorldModel) -> dict[str, Any]:
        """Get profiling information."""
        return {
            "model_parameters": sum(p.numel() for p in wm.named_weights.values()),
            "device": self.device,
        }
