"""Tests for the geometry analyzer."""

import torch
import pytest

from world_model_lens import HookedWorldModel, WorldModelConfig, ActivationCache
from world_model_lens.probing.geometry import GeometryAnalyzer, GeometryResult
from tests.conftest import SimpleTestAdapter


class TestGeometryAnalyzer:
    """Tests for GeometryAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a geometry analyzer."""
        config = WorldModelConfig(d_h=32, d_action=4, d_obs=64)
        adapter = SimpleTestAdapter(config)
        wm = HookedWorldModel(adapter=adapter, config=config)
        return GeometryAnalyzer(wm)

    @pytest.fixture
    def sample_cache(self):
        """Create a sample activation cache."""
        cache = ActivationCache()
        for t in range(10):
            cache["state", t] = torch.randn(32)
            cache["encoding", t] = torch.randn(32)
        return cache

    def test_analyzer_creation(self, analyzer):
        """Test analyzer creates correctly."""
        assert analyzer is not None

    def test_capabilities_property(self, analyzer):
        """Test capabilities property."""
        caps = analyzer.capabilities
        assert caps is not None

    def test_pca_projection_returns_geometry_result(self, analyzer, sample_cache):
        """Test PCA projection returns GeometryResult."""
        result = analyzer.pca_projection(sample_cache)
        assert isinstance(result, GeometryResult)

    def test_trajectory_metrics_returns_geometry_result(self, analyzer, sample_cache):
        """Test trajectory metrics returns GeometryResult."""
        result = analyzer.trajectory_metrics(sample_cache)
        assert isinstance(result, GeometryResult)

    def test_clustering_returns_dict(self, analyzer, sample_cache):
        """Test clustering returns dict."""
        result = analyzer.clustering(sample_cache, n_clusters=3)
        assert isinstance(result, dict)

    def test_manifold_analysis_returns_dict(self, analyzer, sample_cache):
        """Test manifold analysis returns dict."""
        result = analyzer.manifold_analysis(sample_cache)
        assert isinstance(result, dict)


class TestGeometryResult:
    """Tests for GeometryResult dataclass."""

    def test_result_creation(self):
        """Test creating a GeometryResult."""
        result = GeometryResult(
            pca_components=torch.randn(10, 32),
            mean_trajectory_distance=1.5,
            temporal_coherence=0.8,
        )
        assert result.mean_trajectory_distance == 1.5
        assert result.temporal_coherence == 0.8
        assert result.pca_components is not None

    def test_result_to_dict(self):
        """Test converting result to dict."""
        result = GeometryResult(
            mean_trajectory_distance=1.5,
            temporal_coherence=0.8,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "mean_trajectory_distance" in d
        assert "temporal_coherence" in d
