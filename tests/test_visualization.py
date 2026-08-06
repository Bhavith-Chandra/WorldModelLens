import torch
import numpy as np
import pytest

from world_model_lens.core.latent_state import LatentState
from world_model_lens.core.latent_trajectory import LatentTrajectory
from world_model_lens.core.activation_cache import ActivationCache

from world_model_lens.visualization.latent_plots import LatentTrajectoryPlotter, SurpriseHeatmap
from world_model_lens.visualization.intervention_plots import InterventionVisualizer
from world_model_lens.visualization.prediction_plots import PredictionVisualizer
from world_model_lens.visualization.cache_plots import CacheSignalPlotter

def create_mock_trajectory(length: int = 10, d_h: int = 16, n_cat: int = 8, n_cls: int = 8):
    states = []
    for t in range(length):
        states.append(LatentState(
            h_t=torch.randn(d_h),
            z_posterior=torch.randn(n_cat, n_cls),
            z_prior=torch.randn(n_cat, n_cls),
            timestep=t,
            reward_pred=torch.tensor(np.random.rand()),
            reward_real=torch.tensor(np.random.rand()),
            value_pred=torch.tensor(np.random.rand()),
        ))
    return LatentTrajectory(states=states, env_name="MockEnv", episode_id="mock_id")

def test_latent_trajectory_plotter_project_pca():
    traj = create_mock_trajectory()
    plotter = LatentTrajectoryPlotter(world_model=None)
    proj = plotter.project_pca(traj, n_components=2)
    assert proj.x.shape == (10,)
    assert proj.y.shape == (10,)
    assert proj.labels.shape == (10,)

def test_latent_trajectory_plotter_project_tsne():
    traj = create_mock_trajectory()
    plotter = LatentTrajectoryPlotter(world_model=None)
    proj = plotter.project_tsne(traj, perplexity=2, n_iter=250)
    assert proj.x.shape == (10,)
    assert proj.y.shape == (10,)

def test_latent_trajectory_plotter_color_by_reward():
    traj = create_mock_trajectory()
    plotter = LatentTrajectoryPlotter(world_model=None)
    colors = plotter.color_by_reward(traj)
    assert colors.shape == (10,)
    assert colors.dtype == np.float32

def test_intervention_visualizer_divergence_curve():
    t_before = create_mock_trajectory()
    t_after = create_mock_trajectory()
    viz = InterventionVisualizer(world_model=None)
    curve = viz.divergence_curve(before_trajectory=t_before, after_trajectory=t_after)
    assert len(curve) == 10
    assert all(isinstance(v, float) for v in curve.values())

def test_intervention_visualizer_intervention_heatmap():
    t_before = create_mock_trajectory()
    t_after = create_mock_trajectory()
    viz = InterventionVisualizer(world_model=None)
    heatmap = viz.intervention_heatmap(before_trajectory=t_before, after_trajectory=t_after)
    d_z = t_before.states[0].flat.shape[0]
    assert heatmap.shape == (10, d_z)

def test_prediction_visualizer_reward_timeline():
    traj = create_mock_trajectory()
    viz = PredictionVisualizer(world_model=None)
    res = viz.reward_timeline(traj)
    assert "predicted" in res
    assert "ground_truth" in res
    assert res["predicted"].shape == (10,)
    assert res["ground_truth"].shape == (10,)

def test_surprise_heatmap():
    cache = ActivationCache()
    for t in range(5):
        cache[("z_posterior", t)] = torch.randn(8, 8)
        cache[("z_prior", t)] = torch.randn(8, 8)
    
    heatmap = SurpriseHeatmap.compute(cache)
    assert heatmap["matrix"].shape == (5, 8)
    assert heatmap["timesteps"].shape == (5,)

def test_cache_signal_plotter_surprise():
    cache = ActivationCache()
    for t in range(5):
        cache[('kl', t)] = torch.tensor(t * 0.1)
    
    res = CacheSignalPlotter.plot_surprise_timeline(cache)
    assert res["timesteps"].shape == (5,)
    assert res["kl_values"].shape == (5,)
    assert np.allclose(res["kl_values"], [0.0, 0.1, 0.2, 0.3, 0.4])

def test_cache_signal_plotter_reward():
    traj = create_mock_trajectory()
    res = CacheSignalPlotter.plot_reward_timeline(traj)
    assert res["timesteps"].shape == (10,)
    assert res["predicted"].shape == (10,)
    assert res["actual"].shape == (10,)
