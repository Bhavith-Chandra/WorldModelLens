"""Library entry points that call HookedWorldModel.run_with_cache must accept its three outputs."""

import numpy as np
import pytest
import torch

from world_model_lens import HookedWorldModel, WorldTrajectory
from world_model_lens.analysis.continual_learning import (
    ContinualLearningAuditor,
    RepresentationTracker,
)
from world_model_lens.analysis.metrics import LatentMetrics
from world_model_lens.backends.dreamerv3 import DreamerV3Adapter, default_arch
from world_model_lens.benchmarks.cartpole import CartPoleBenchmark
from world_model_lens.benchmarks.probing import ProbingBenchmarkSuite
from world_model_lens.causal.counterfactual import CounterfactualEngine, Intervention
from world_model_lens.visualization.temporal_maps import TemporalAttributionMap

T, D, A = 24, 4, 3


@pytest.fixture
def wm():
    torch.manual_seed(0)
    arch = default_arch(
        deter=32, hidden=8, units=8, stoch=4, classes=4, action_dim=A, img_shape=None, vec_dim=D
    )
    adapter = DreamerV3Adapter(arch=arch).eval()
    return HookedWorldModel(adapter, adapter.config)


@pytest.fixture
def obs():
    return torch.randn(T, D, generator=torch.Generator().manual_seed(1))


@pytest.fixture
def actions():
    return torch.randint(0, A, (T,), generator=torch.Generator().manual_seed(2))


def ablate_state(t):
    return Intervention(target_timestep=t, target_type="state")


@torch.no_grad()
def test_counterfactual_engine(wm, obs, actions):
    engine = CounterfactualEngine(wm)

    tree = engine.build_branch_tree(obs, [ablate_state(3)], actions)
    comparison = engine.compare_interventions(obs, [ablate_state(3)], actions)
    critical = engine.find_critical_timestep(obs, torch.zeros(32))

    assert isinstance(tree.root_trajectory, WorldTrajectory)
    assert len(comparison) == 1
    assert 0 <= critical < T


@torch.no_grad()
def test_latent_metrics_reconstruction_error(wm, obs, actions):
    _, _, cache = wm.run_with_cache(obs, actions)
    expected = torch.nn.functional.mse_loss(cache["reconstruction"], obs)

    assert LatentMetrics.reconstruction_error(wm, obs, actions) == pytest.approx(expected.item())


@torch.no_grad()
def test_continual_learning(wm, obs, actions, tmp_path):
    path = tmp_path / "epoch0.pt"
    torch.save(wm.adapter.state_dict(), path)
    auditor = ContinualLearningAuditor(wm)
    auditor.register_checkpoint(str(path), 0)

    trajectory = auditor.run_on_checkpoint(0, obs, actions)
    RepresentationTracker(wm).snapshot_epoch(0, obs)

    assert isinstance(trajectory, WorldTrajectory)
    assert trajectory.length == T


@torch.no_grad()
def test_probing_benchmarks(wm, obs, actions):
    suite = ProbingBenchmarkSuite()
    assert np.isfinite(suite.reward_prediction(wm, obs, actions, torch.randn(T)))
    assert np.isfinite(suite.temporal_position(wm, obs, actions))


@torch.no_grad()
def test_cartpole_velocity_encoding(wm, obs):
    found, dims = CartPoleBenchmark(wm).find_velocity_encoding([obs, obs + 1.0])
    assert isinstance(found, bool)
    assert isinstance(dims, list)


@torch.no_grad()
def test_temporal_attribution_map(wm, obs, actions):
    mapper = TemporalAttributionMap(wm)
    short_obs, short_actions = obs[:6], actions[:6]

    matrix = mapper.compute_influence_matrix(short_obs, short_actions)
    from_t = mapper.influence_from_timestep(short_obs, short_actions, 1)
    to_t = mapper.influence_to_timestep(short_obs, short_actions, 4)
    flow = mapper.trace_causal_flow(short_obs, short_actions, 0, 4)

    assert matrix.matrix.shape == (6, 6)
    assert np.all(matrix.matrix[1, :1] == 0)
    assert matrix.matrix[1, 2:].sum() > 0
    assert from_t.shape == (6,)
    assert to_t.shape == (6,)
    assert set(flow) <= set(range(6))
    assert len(mapper.wm.hook_registry) == 0
