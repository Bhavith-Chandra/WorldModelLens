"""Tests for faithfulness analysis module (AOPC metric)."""

import torch
import pytest

from world_model_lens import HookedWorldModel
from world_model_lens.analysis.faithfulness import (
    FaithfulnessAnalyzer,
    AOPCResult,
    PerturbationResult,
)
from tests.conftest import SimpleTestAdapter
from world_model_lens.backends.toy_video_model import ToyVideoAdapter


def make_video_config():
    """Create a config for ToyVideoAdapter."""
    return type(
        "Config",
        (),
        {
            "latent_dim": 16,
            "hidden_dim": 64,
            "obs_channels": 3,
        },
    )()


class TestAOPCResult:
    """Tests for AOPCResult dataclass."""

    def test_creation(self):
        """Test AOPCResult creates correctly."""
        result = AOPCResult(
            aopc_score=0.5,
            mses=[0.1, 0.2, 0.3],
            k_values=[1, 2, 3],
            component="z_posterior",
        )
        assert result.aopc_score == 0.5
        assert result.mses == [0.1, 0.2, 0.3]
        assert result.k_values == [1, 2, 3]
        assert result.component == "z_posterior"

    def test_plot_returns_figure(self):
        """Test plot method returns matplotlib figure."""
        result = AOPCResult(
            aopc_score=0.5,
            mses=[0.1, 0.2, 0.3],
            k_values=[1, 2, 3],
            component="z_posterior",
        )
        matplotlib = pytest.importorskip("matplotlib")

        matplotlib.use("Agg")
        fig = result.plot()
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestPerturbationResult:
    """Tests for PerturbationResult dataclass."""

    def test_creation(self):
        """Test PerturbationResult creates correctly."""
        result = PerturbationResult(
            k=5,
            mse_delta=0.25,
            ablated_dims=[0, 1, 2, 3, 4],
            component="z_posterior",
        )
        assert result.k == 5
        assert result.mse_delta == 0.25
        assert result.ablated_dims == [0, 1, 2, 3, 4]
        assert result.component == "z_posterior"


class TestFaithfulnessAnalyzer:
    """Tests for FaithfulnessAnalyzer class."""

    @pytest.fixture
    def video_wm(self):
        """Create a ToyVideoModel world model."""
        config = make_video_config()
        adapter = ToyVideoAdapter(config=config, latent_dim=16, hidden_dim=64)
        wm = HookedWorldModel(adapter=adapter, config=config, name="toy_video")
        return wm

    @pytest.fixture
    def analyzer(self, video_wm):
        """Create a faithfulness analyzer."""
        return FaithfulnessAnalyzer(video_wm)

    def test_analyzer_creation(self, analyzer):
        """Test analyzer creates correctly."""
        assert analyzer is not None

    def test_has_aopc_method(self, analyzer):
        """Test aopc method exists."""
        assert hasattr(analyzer, "aopc")
        assert callable(analyzer.aopc)

    def test_has_perturbation_curve_method(self, analyzer):
        """Test perturbation_curve method exists."""
        assert hasattr(analyzer, "perturbation_curve")
        assert callable(analyzer.perturbation_curve)

    def test_aopc_returns_result(self, analyzer, video_wm):
        """Test aopc returns AOPCResult."""
        obs = torch.randn(10, 3, 64, 64)

        def predictor_fn(cache):
            return cache["reconstruction"]

        result = analyzer.aopc(
            obs,
            target_component="z_posterior",
            predictor_fn=predictor_fn,
            max_k=3,
        )

        assert isinstance(result, AOPCResult)
        assert hasattr(result, "aopc_score")
        assert hasattr(result, "mses")
        assert hasattr(result, "k_values")
        assert result.component == "z_posterior"

    def test_aopc_with_different_components(self, analyzer, video_wm):
        """Test AOPC with different target components."""
        obs = torch.randn(10, 3, 64, 64)

        def predictor_fn(cache):
            return cache["reconstruction"]

        result_z = analyzer.aopc(
            obs, target_component="z_posterior", predictor_fn=predictor_fn, max_k=3
        )
        assert result_z.component == "z_posterior"

        result_h = analyzer.aopc(obs, target_component="h", predictor_fn=predictor_fn, max_k=3)
        assert result_h.component == "h"

    def test_aopc_with_custom_dim_importance(self, analyzer, video_wm):
        """Test AOPC with custom dimension importance."""
        obs = torch.randn(10, 3, 64, 64)

        def predictor_fn(cache):
            return cache["reconstruction"]

        dim_importance = torch.ones(16)
        result = analyzer.aopc(
            obs,
            target_component="z_posterior",
            predictor_fn=predictor_fn,
            dim_importance=dim_importance,
            max_k=3,
        )

        assert isinstance(result, AOPCResult)

    def test_aopc_normalize_param(self, analyzer, video_wm):
        """Test AOPC normalize parameter."""
        obs = torch.randn(10, 3, 64, 64)

        def predictor_fn(cache):
            return cache["reconstruction"]

        result_norm = analyzer.aopc(
            obs,
            target_component="z_posterior",
            predictor_fn=predictor_fn,
            max_k=3,
            normalize=True,
        )

        result_unnorm = analyzer.aopc(
            obs,
            target_component="z_posterior",
            predictor_fn=predictor_fn,
            max_k=3,
            normalize=False,
        )

        assert isinstance(result_norm, AOPCResult)
        assert isinstance(result_unnorm, AOPCResult)

    def test_perturbation_curve_returns_list(self, analyzer, video_wm):
        """Test perturbation_curve returns list of PerturbationResults."""
        obs = torch.randn(10, 3, 64, 64)

        def predictor_fn(cache):
            return cache["reconstruction"]

        results = analyzer.perturbation_curve(
            obs,
            target_component="z_posterior",
            predictor_fn=predictor_fn,
            k_values=[1, 2, 3],
        )

        assert isinstance(results, list)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, PerturbationResult)

    def test_perturbation_curve_custom_k_values(self, analyzer, video_wm):
        """Test perturbation_curve with custom k_values."""
        obs = torch.randn(10, 3, 64, 64)

        def predictor_fn(cache):
            return cache["reconstruction"]

        results = analyzer.perturbation_curve(
            obs,
            target_component="z_posterior",
            predictor_fn=predictor_fn,
            k_values=[1, 5, 10],
        )

        assert len(results) == 3
        assert results[0].k == 1
        assert results[1].k == 5
        assert results[2].k == 10

    def test_aopc_mono_increasing_k(self, analyzer, video_wm):
        """Test that MSE increases with K (more ablation = more disruption)."""
        obs = torch.randn(10, 3, 64, 64)

        def predictor_fn(cache):
            return cache["reconstruction"]

        result = analyzer.aopc(
            obs,
            target_component="z_posterior",
            predictor_fn=predictor_fn,
            max_k=5,
        )

        for i in range(len(result.mses) - 1):
            if result.k_values[i + 1] > result.k_values[i]:
                assert result.mses[i + 1] >= result.mses[i] - 1e-6

    def test_aopc_with_max_k_none(self, analyzer, video_wm):
        """Test AOPC with max_k=None uses all dimensions."""
        obs = torch.randn(10, 3, 64, 64)

        def predictor_fn(cache):
            return cache["reconstruction"]

        _, cache = video_wm.run_with_cache(obs)
        z = cache["z_posterior"]
        num_dims = z.shape[-1] if z is not None else 16

        result = analyzer.aopc(
            obs,
            target_component="z_posterior",
            predictor_fn=predictor_fn,
            max_k=None,
        )

        assert result.k_values[-1] == num_dims

    def test_perturbation_curve_empty_on_missing_component(self, analyzer, video_wm):
        """Test perturbation_curve returns empty list for missing component."""
        obs = torch.randn(10, 3, 64, 64)

        def predictor_fn(cache):
            return cache["reconstruction"]

        results = analyzer.perturbation_curve(
            obs,
            target_component="nonexistent_component",
            predictor_fn=predictor_fn,
        )

        assert results == []


class TestAOPCIntegration:
    """Integration tests for AOPC with different model types."""

    def test_aopc_with_simple_adapter(self):
        """Test AOPC with SimpleTestAdapter."""
        from dataclasses import dataclass

        @dataclass
        class SimpleConfig:
            d_h: int = 32
            d_action: int = 4
            d_obs: int = 64

        config = SimpleConfig()
        adapter = SimpleTestAdapter(config)
        wm = HookedWorldModel(adapter=adapter, config=config)

        obs = torch.randn(10, 64)

        analyzer = FaithfulnessAnalyzer(wm)

        _, cache = wm.run_with_cache(obs)

        result = analyzer.aopc(obs, target_component="h", max_k=3)
        assert isinstance(result, AOPCResult)


class TestAOPCComparison:
    """Tests for comparing AOPC across different setups."""

    @pytest.fixture
    def video_wm(self):
        """Create a ToyVideoModel world model."""
        config = make_video_config()
        adapter = ToyVideoAdapter(config=config, latent_dim=16, hidden_dim=64)
        wm = HookedWorldModel(adapter=adapter, config=config, name="toy_video")
        return wm

    def test_aopc_is_deterministic(self, video_wm):
        """Test that AOPC returns same result on repeated runs."""
        obs = torch.randn(10, 3, 64, 64)

        def predictor_fn(cache):
            return cache["reconstruction"]

        analyzer = FaithfulnessAnalyzer(video_wm)

        result1 = analyzer.aopc(obs, max_k=3, predictor_fn=predictor_fn)
        result2 = analyzer.aopc(obs, max_k=3, predictor_fn=predictor_fn)

        assert result1.aopc_score == result2.aopc_score
        assert result1.mses == result2.mses
        assert result1.k_values == result2.k_values

    def test_aopc_different_max_k(self, video_wm):
        """Test AOPC with different max_k values."""
        obs = torch.randn(10, 3, 64, 64)

        def predictor_fn(cache):
            return cache["reconstruction"]

        analyzer = FaithfulnessAnalyzer(video_wm)

        result_k1 = analyzer.aopc(obs, max_k=1, predictor_fn=predictor_fn)
        result_k5 = analyzer.aopc(obs, max_k=5, predictor_fn=predictor_fn)

        assert len(result_k1.mses) == 1
        assert len(result_k5.mses) == 5
        assert result_k5.k_values[-1] == 5
