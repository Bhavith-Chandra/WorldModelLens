"""Test fixtures for World Model Lens."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from world_model_lens import WorldModelConfig
from world_model_lens import HookedWorldModel
from world_model_lens.backends.base_adapter import WorldModelCapabilities


class SimpleTestAdapter(nn.Module):
    """Simple adapter for testing without dimension issues."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder_net = nn.Sequential(
            nn.Linear(config.d_obs, config.d_h),
            nn.ReLU(),
            nn.Linear(config.d_h, config.d_h),
        )
        self.dynamics_net = nn.Sequential(
            nn.Linear(config.d_h, config.d_h),
            nn.ReLU(),
            nn.Linear(config.d_h, config.d_h),
        )
        self._capabilities = WorldModelCapabilities(
            has_reward_head=False,
            has_continue_head=False,
            has_actor=False,
            has_decoder=False,
            has_critic=False,
            uses_actions=False,
        )

    @property
    def capabilities(self):
        return self._capabilities

    def encode(self, obs, state=None):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.encoder_net(obs), self.encoder_net(obs)

    def dynamics(self, state, action=None):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.dynamics_net(state)

    def transition(self, state, action=None, input_=None):
        return self.dynamics(state, action)

    def initial_state(self, batch_size=1, device=None):
        if device is None:
            device = torch.device("cpu")
        return torch.zeros(batch_size, self.config.d_h, device=device)

    def sample_z(self, logits_or_repr, temperature=1.0, sample=True):
        return logits_or_repr

    def sample_state(self, prior, temperature=1.0):
        return prior


@pytest.fixture
def tiny_cfg():
    """Small config for fast tests."""
    return WorldModelConfig(d_h=32, d_action=4, d_obs=64)


@pytest.fixture
def mock_adapter(tiny_cfg):
    """Simple adapter with random weights."""
    return SimpleTestAdapter(tiny_cfg)


@pytest.fixture
def hooked_wm(mock_adapter, tiny_cfg):
    """HookedWorldModel wrapper."""
    return HookedWorldModel(adapter=mock_adapter, config=tiny_cfg)


@pytest.fixture
def fake_obs_seq(tiny_cfg):
    """Random observation sequence (vector format)."""
    return torch.randn(10, tiny_cfg.d_obs)


@pytest.fixture
def fake_action_seq(tiny_cfg):
    """Random action sequence."""
    return torch.randn(10, tiny_cfg.d_action)


@pytest.fixture
def fake_trajectory(tiny_cfg):
    """Fake world trajectory."""
    from world_model_lens import WorldState, WorldTrajectory

    states = []
    for t in range(10):
        state = WorldState(
            state=torch.randn(tiny_cfg.d_h),
            timestep=t,
            action=torch.randn(tiny_cfg.d_action) if hasattr(tiny_cfg, "d_action") else None,
            reward=torch.tensor(1.0),
        )
        states.append(state)

    return WorldTrajectory(states=states)


@pytest.fixture
def fake_cache(tiny_cfg):
    """Fake activation cache."""
    from world_model_lens import ActivationCache

    cache = ActivationCache()
    for t in range(10):
        cache["state", t] = torch.randn(tiny_cfg.d_h)
        cache["posterior", t] = torch.randn(tiny_cfg.d_h)
        cache["prior", t] = torch.randn(tiny_cfg.d_h)
        cache["reward_pred", t] = torch.randn(1)
    return cache
