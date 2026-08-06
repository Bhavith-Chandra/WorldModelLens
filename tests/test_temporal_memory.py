"""Tests for the temporal memory prober."""

import torch
import pytest

from world_model_lens import HookedWorldModel, WorldModelConfig, ActivationCache
from world_model_lens.probing.temporal_memory import TemporalMemoryProber, TemporalMemoryResult
from tests.conftest import SimpleTestAdapter


class TestTemporalMemoryProber:
    """Tests for TemporalMemoryProber class."""

    @pytest.fixture
    def prober(self):
        """Create a temporal memory prober."""
        config = WorldModelConfig(d_h=32, d_action=4, d_obs=64)
        adapter = SimpleTestAdapter(config)
        wm = HookedWorldModel(adapter=adapter, config=config)
        return TemporalMemoryProber(wm)

    @pytest.fixture
    def sample_cache(self):
        """Create a sample activation cache."""
        cache = ActivationCache()
        for t in range(10):
            cache["state", t] = torch.randn(32)
        return cache

    def test_prober_creation(self, prober):
        """Test prober creates correctly."""
        assert prober is not None

    def test_capabilities_property(self, prober):
        """Test capabilities property."""
        caps = prober.capabilities
        assert caps is not None

    def test_memory_retention_returns_temporal_result(self, prober, sample_cache):
        """Test memory_retention returns TemporalMemoryResult."""
        result = prober.memory_retention(sample_cache)
        assert isinstance(result, TemporalMemoryResult)

    def test_temporal_dependencies_returns_dict(self, prober, sample_cache):
        """Test temporal_dependencies returns dict."""
        result = prober.temporal_dependencies(sample_cache)
        assert isinstance(result, dict)

    def test_sequential_patterns_returns_dict(self, prober, sample_cache):
        """Test sequential_patterns returns dict."""
        result = prober.sequential_patterns(sample_cache)
        assert isinstance(result, dict)


class TestTemporalMemoryResult:
    """Tests for TemporalMemoryResult dataclass."""

    def test_result_creation(self):
        """Test creating a TemporalMemoryResult."""
        result = TemporalMemoryResult(
            retention_scores=torch.randn(10),
            memory_capacity=0.8,
            temporal_dependencies={"lag_1": 0.5},
            working_memory_estimate=5.0,
        )
        assert result.memory_capacity == 0.8
        assert result.working_memory_estimate == 5.0
        assert len(result.retention_scores) == 10

    def test_result_to_dict(self):
        """Test converting result to dict."""
        result = TemporalMemoryResult(
            retention_scores=torch.randn(5),
            memory_capacity=0.9,
            temporal_dependencies={},
            working_memory_estimate=3.0,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "memory_capacity" in d
        assert "working_memory_estimate" in d
