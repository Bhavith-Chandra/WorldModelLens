"""Integration tests for non-RL world model adapters.

These tests ensure that WorldModelLens works correctly with non-RL models
(video world models, scientific latent dynamics models) without requiring
reward heads, value heads, or actions.
"""

import pytest
import torch
from typing import Tuple

from world_model_lens import HookedWorldModel
from world_model_lens.backends.toy_video_model import ToyVideoAdapter, create_toy_video_adapter
from world_model_lens.backends.toy_scientific_model import (
    ToyScientificAdapter,
    create_toy_scientific_adapter,
    generate_lorenz_trajectory,
    generate_pendulum_trajectory,
)
from world_model_lens.analysis.belief_analyzer import BeliefAnalyzer
from world_model_lens.probing.geometry import GeometryAnalyzer
from world_model_lens.probing.temporal_memory import TemporalMemoryProber


class TestToyVideoAdapter:
    """Tests for the Toy Video World Model Adapter."""

    def test_adapter_creation(self):
        """Test that the adapter can be created."""
        adapter = create_toy_video_adapter(latent_dim=16, hidden_dim=64)
        assert adapter is not None
        assert adapter.latent_dim == 16
        assert adapter.hidden_dim == 64

    def test_capabilities(self):
        """Test that capabilities are correctly set."""
        adapter = create_toy_video_adapter()

        caps = adapter.capabilities
        assert caps.has_decoder is True
        assert caps.has_reward_head is False
        assert caps.has_continue_head is False
        assert caps.has_actor is False
        assert caps.has_critic is False
        assert caps.uses_actions is False
        assert caps.is_rl_trained is False

    def test_initial_state(self):
        """Test initial state generation."""
        adapter = create_toy_video_adapter(latent_dim=16)
        h_0, z_0 = adapter.initial_state(batch_size=4)

        assert h_0.shape == (4, 16)
        assert z_0.shape == (4, 16)

    def test_encode(self):
        """Test encoding observations to latents."""
        adapter = create_toy_video_adapter(latent_dim=16)

        obs = torch.randn(3, 64, 64)
        h_prev = torch.zeros(16)

        z_post, z_prior = adapter.encode(obs, h_prev)

        assert z_post.shape == (1, 16) or z_post.dim() == 1
        assert z_prior.shape == z_post.shape

    def test_transition(self):
        """Test latent transition without actions."""
        adapter = create_toy_video_adapter(latent_dim=16)

        z = torch.randn(16)
        h = torch.randn(16)

        z_next = adapter.transition(h, z, action=None)

        assert z_next.shape == z.shape

    def test_decode(self):
        """Test decoding latents to observations."""
        adapter = create_toy_video_adapter(latent_dim=16, hidden_dim=32)

        z = torch.randn(16)
        h = torch.randn(32)

        recon = adapter.decode(h, z)

        assert recon is not None
        assert recon.shape[0] == 3 or recon.dim() >= 2

    def test_named_parameters(self):
        """Test that named_parameters returns all model parameters."""
        adapter = create_toy_video_adapter()

        params = adapter.named_parameters()

        assert len(params) > 0
        assert all(isinstance(p, torch.Tensor) for p in params.values())


class TestToyScientificAdapter:
    """Tests for the Toy Scientific Latent Dynamics Adapter."""

    def test_adapter_creation(self):
        """Test that the adapter can be created."""
        adapter = create_toy_scientific_adapter(latent_dim=16, obs_dim=10)
        assert adapter is not None
        assert adapter.latent_dim == 16
        assert adapter.obs_dim == 10

    def test_capabilities(self):
        """Test that capabilities are correctly set."""
        adapter = create_toy_scientific_adapter()

        caps = adapter.capabilities
        assert caps.has_decoder is False
        assert caps.has_reward_head is False
        assert caps.has_continue_head is False
        assert caps.has_actor is False
        assert caps.has_critic is False
        assert caps.uses_actions is False
        assert caps.is_rl_trained is False

    def test_initial_state(self):
        """Test initial state generation."""
        adapter = create_toy_scientific_adapter(latent_dim=16)
        h_0, z_0 = adapter.initial_state(batch_size=4)

        assert h_0.shape == (4, 16)
        assert z_0.shape == (4, 16)

    def test_encode(self):
        """Test encoding observations to latents."""
        adapter = create_toy_scientific_adapter(latent_dim=16, obs_dim=10)

        obs = torch.randn(10)
        h_prev = torch.zeros(16)

        z_post, z_prior = adapter.encode(obs, h_prev)

        assert z_post.shape[-1] == 16

    def test_transition(self):
        """Test latent transition without actions."""
        adapter = create_toy_scientific_adapter(latent_dim=16)

        z = torch.randn(16)
        h = torch.randn(16)

        z_next = adapter.transition(h, z, action=None)

        assert z_next.shape == z.shape

    def test_named_parameters(self):
        """Test that named_parameters returns all model parameters."""
        adapter = create_toy_scientific_adapter()

        params = adapter.named_parameters()

        assert len(params) > 0
        assert all(isinstance(p, torch.Tensor) for p in params.values())


class TestHookedWorldModelWithVideo:
    """Test HookedWorldModel with video adapter."""

    @pytest.fixture
    def video_wm(self) -> HookedWorldModel:
        """Create a hooked world model with video adapter."""
        config = type(
            "Config",
            (),
            {
                "latent_dim": 16,
                "hidden_dim": 64,
                "obs_channels": 3,
            },
        )()
        adapter = ToyVideoAdapter(config=config, latent_dim=16, hidden_dim=64)
        return HookedWorldModel(adapter=adapter, config=config, name="toy_video")

    def test_capabilities_property(self, video_wm):
        """Test that capabilities property works."""
        caps = video_wm.capabilities

        assert caps.has_decoder is True
        assert caps.has_reward_head is False
        assert caps.is_rl_model() is False

    def test_run_with_cache(self, video_wm):
        """Test run_with_cache with video sequence."""
        frames = torch.randn(10, 3, 64, 64)

        world_traj, latent_traj, cache = video_wm.run_with_cache(frames)

        assert world_traj is not None
        assert latent_traj is not None
        assert cache is not None
        assert len(world_traj.states) == 10
        assert len(latent_traj.states) == 10

    def test_run_with_cache_no_actions(self, video_wm):
        """Test that run_with_cache works without actions."""
        frames = torch.randn(10, 3, 64, 64)
        actions = None

        world_traj, _, cache = video_wm.run_with_cache(frames, actions=actions)

        assert world_traj is not None
        for state in world_traj.states:
            assert state.action is None


class TestHookedWorldModelWithScientific:
    """Test HookedWorldModel with scientific adapter."""

    @pytest.fixture
    def scientific_wm(self) -> HookedWorldModel:
        """Create a hooked world model with scientific adapter."""
        config = type(
            "Config",
            (),
            {
                "obs_dim": 10,
                "latent_dim": 16,
                "hidden_dim": 64,
            },
        )()
        adapter = ToyScientificAdapter(config=config, obs_dim=10, latent_dim=16)
        return HookedWorldModel(adapter=adapter, config=config, name="toy_scientific")

    def test_capabilities_property(self, scientific_wm):
        """Test that capabilities property works."""
        caps = scientific_wm.capabilities

        assert caps.has_decoder is False
        assert caps.has_reward_head is False
        assert caps.is_rl_model() is False

    def test_run_with_cache(self, scientific_wm):
        """Test run_with_cache with observation sequence."""
        observations = torch.randn(50, 10)

        world_traj, latent_traj, cache = scientific_wm.run_with_cache(observations)

        assert world_traj is not None
        assert latent_traj is not None
        assert cache is not None
        assert len(world_traj.states) == 50
        assert len(latent_traj.states) == 50


class TestBeliefAnalyzerWithNonRL:
    """Test BeliefAnalyzer with non-RL models."""

    @pytest.fixture
    def video_wm(self) -> HookedWorldModel:
        """Create video world model."""
        config = type("Config", (), {"latent_dim": 16, "hidden_dim": 64})()
        adapter = ToyVideoAdapter(config=config, latent_dim=16, hidden_dim=64)
        return HookedWorldModel(adapter=adapter, config=config)

    @pytest.fixture
    def scientific_wm(self) -> HookedWorldModel:
        """Create scientific world model."""
        config = type("Config", (), {"obs_dim": 10, "latent_dim": 16})()
        adapter = ToyScientificAdapter(config=config, obs_dim=10, latent_dim=16)
        return HookedWorldModel(adapter=adapter, config=config)

    def test_surprise_timeline_video(self, video_wm):
        """Test surprise timeline with video model."""
        frames = torch.randn(20, 3, 64, 64)
        _, _, cache = video_wm.run_with_cache(frames)

        analyzer = BeliefAnalyzer(video_wm)
        result = analyzer.surprise_timeline(cache)

        assert result is not None
        assert hasattr(result, "kl_sequence")
        assert len(result.kl_sequence) == 20 or len(result.kl_sequence) == 0

    def test_surprise_timeline_scientific(self, scientific_wm):
        """Test surprise timeline with scientific model."""
        observations = torch.randn(30, 10)
        _, _, cache = scientific_wm.run_with_cache(observations)

        analyzer = BeliefAnalyzer(scientific_wm)
        result = analyzer.surprise_timeline(cache)

        assert result is not None

    def test_reward_attribution_returns_unavailable(self, scientific_wm):
        """Test that reward attribution returns unavailable for non-RL models."""
        observations = torch.randn(20, 10)
        world_traj, _, cache = scientific_wm.run_with_cache(observations)

        analyzer = BeliefAnalyzer(scientific_wm)
        result = analyzer.reward_attribution(world_traj, cache)

        assert result.is_available is False

    def test_value_analysis_returns_unavailable(self, scientific_wm):
        """Test that value analysis returns unavailable for non-RL models."""
        observations = torch.randn(20, 10)
        _, _, cache = scientific_wm.run_with_cache(observations)

        analyzer = BeliefAnalyzer(scientific_wm)
        result = analyzer.value_analysis(cache)

        assert result.get("is_available", True) is False


class TestGeometryAnalyzerWithNonRL:
    """Test GeometryAnalyzer with non-RL models."""

    @pytest.fixture
    def scientific_wm(self) -> HookedWorldModel:
        """Create scientific world model."""
        config = type("Config", (), {"obs_dim": 10, "latent_dim": 16})()
        adapter = ToyScientificAdapter(config=config, obs_dim=10, latent_dim=16)
        return HookedWorldModel(adapter=adapter, config=config)

    def test_pca_projection(self, scientific_wm):
        """Test PCA projection with scientific model."""
        observations = torch.randn(30, 10)
        _, _, cache = scientific_wm.run_with_cache(observations)

        analyzer = GeometryAnalyzer(scientific_wm)
        result = analyzer.pca_projection(cache)

        assert result is not None
        assert hasattr(result, "mean_trajectory_distance")

    def test_trajectory_metrics(self, scientific_wm):
        """Test trajectory metrics with scientific model."""
        observations = torch.randn(30, 10)
        _, _, cache = scientific_wm.run_with_cache(observations)

        analyzer = GeometryAnalyzer(scientific_wm)
        result = analyzer.trajectory_metrics(cache)

        assert result is not None
        assert hasattr(result, "mean_trajectory_distance")
        assert hasattr(result, "temporal_coherence")

    def test_clustering(self, scientific_wm):
        """Test clustering with scientific model."""
        observations = torch.randn(30, 10)
        _, _, cache = scientific_wm.run_with_cache(observations)

        analyzer = GeometryAnalyzer(scientific_wm)
        result = analyzer.clustering(cache, n_clusters=5)

        assert result is not None
        assert "clusters" in result
        assert "centroids" in result

    def test_manifold_analysis(self, scientific_wm):
        """Test manifold analysis with scientific model."""
        observations = torch.randn(30, 10)
        _, _, cache = scientific_wm.run_with_cache(observations)

        analyzer = GeometryAnalyzer(scientific_wm)
        result = analyzer.manifold_analysis(cache)

        assert result is not None
        assert "intrinsic_dimensionality_estimate" in result
        assert "local_linearity" in result


class TestTemporalMemoryProberWithNonRL:
    """Test TemporalMemoryProber with non-RL models."""

    @pytest.fixture
    def scientific_wm(self) -> HookedWorldModel:
        """Create scientific world model."""
        config = type("Config", (), {"obs_dim": 10, "latent_dim": 16})()
        adapter = ToyScientificAdapter(config=config, obs_dim=10, latent_dim=16)
        return HookedWorldModel(adapter=adapter, config=config)

    def test_memory_retention(self, scientific_wm):
        """Test memory retention analysis."""
        observations = torch.randn(50, 10)
        _, _, cache = scientific_wm.run_with_cache(observations)

        prober = TemporalMemoryProber(scientific_wm)
        result = prober.memory_retention(cache)

        assert result is not None
        assert hasattr(result, "retention_scores")
        assert hasattr(result, "memory_capacity")

    def test_temporal_dependencies(self, scientific_wm):
        """Test temporal dependency analysis."""
        observations = torch.randn(30, 10)
        _, _, cache = scientific_wm.run_with_cache(observations)

        prober = TemporalMemoryProber(scientific_wm)
        result = prober.temporal_dependencies(cache)

        assert result is not None
        assert "autocorrelations" in result

    def test_sequential_patterns(self, scientific_wm):
        """Test sequential pattern detection."""
        observations = torch.randn(30, 10)
        _, _, cache = scientific_wm.run_with_cache(observations)

        prober = TemporalMemoryProber(scientific_wm)
        result = prober.sequential_patterns(cache)

        assert result is not None
        assert "patterns" in result


class TestScientificTrajectoryGenerators:
    """Test synthetic trajectory generators."""

    def test_generate_lorenz_trajectory(self):
        """Test Lorenz attractor trajectory generation."""
        traj = generate_lorenz_trajectory(n_steps=100)

        assert traj.shape == (100, 3)
        assert traj.dtype == torch.float32

    def test_generate_pendulum_trajectory(self):
        """Test pendulum trajectory generation."""
        traj = generate_pendulum_trajectory(n_steps=100)

        assert traj.shape == (100, 2)
        assert traj.dtype == torch.float32

    def test_lorenz_with_adapter(self):
        """Test using Lorenz trajectory with scientific adapter."""
        config = type("Config", (), {"obs_dim": 3, "latent_dim": 16})()
        adapter = ToyScientificAdapter(config=config, obs_dim=3, latent_dim=16)
        wm = HookedWorldModel(adapter=adapter, config=config)

        traj = generate_lorenz_trajectory(n_steps=50)
        observations = traj

        result_traj, _, cache = wm.run_with_cache(observations)

        assert result_traj is not None
        assert len(result_traj.states) == 50


class TestImaginationWithNonRL:
    """Test imagination with non-RL models."""

    @pytest.fixture
    def scientific_wm(self) -> HookedWorldModel:
        """Create scientific world model."""
        config = type("Config", (), {"obs_dim": 10, "latent_dim": 16})()
        adapter = ToyScientificAdapter(config=config, obs_dim=10, latent_dim=16)
        return HookedWorldModel(adapter=adapter, config=config)

    def test_imagine_without_actions(self, scientific_wm):
        """Test imagination without actions (non-RL model)."""
        observations = torch.randn(10, 10)
        world_traj, latent_traj, cache = scientific_wm.run_with_cache(observations)

        start_state = latent_traj.states[0]
        imagined_world, imagined_latent = scientific_wm.imagine(start_state, actions=None, horizon=20)

        assert imagined_world is not None
        assert imagined_latent is not None
        assert len(imagined_world.states) == 20
        assert len(imagined_latent.states) == 20

        for state in imagined_world.states:
            assert state.action is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
