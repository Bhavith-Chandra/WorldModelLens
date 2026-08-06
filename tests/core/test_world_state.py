"""Tests for WorldState."""

import pytest
import torch


def test_world_state_properties(fake_trajectory):
    """Test WorldState properties."""
    state = fake_trajectory.states[0]

    assert state.state.shape[0] == 32
    assert state.timestep == 0


def test_world_state_has_reward(fake_trajectory):
    """Test has_reward method."""
    state = fake_trajectory.states[0]
    assert state.has_reward() is True


def test_world_state_has_action(fake_trajectory):
    """Test has_action method."""
    state = fake_trajectory.states[0]
    assert state.has_action() is True


def test_to_device(fake_trajectory):
    """Test moving state to device."""
    state = fake_trajectory.states[0]
    device = torch.device("cpu")
    moved = state.to_device(device)

    assert moved.state.device == device


def test_detach(fake_trajectory):
    """Test detaching state from graph."""
    state = fake_trajectory.states[0]
    state.state.requires_grad = True

    detached = state.detach()
    assert not detached.state.requires_grad


def test_world_state_optional_fields():
    """Test WorldState with minimal fields."""
    from world_model_lens import WorldState

    state = WorldState(
        state=torch.randn(32),
        timestep=0,
    )
    assert state.has_reward() is False
    assert state.has_value() is False
    assert state.has_action() is False


def test_world_state_action_source():
    """Test WorldState with action source tracking."""
    from world_model_lens import WorldState
    from world_model_lens.core.world_state import ActionSource

    # Test with externally provided action
    action_source = ActionSource(
        source_type="externally_provided",
        temperature=None,
    )

    state = WorldState(
        state=torch.randn(32),
        timestep=0,
        action=torch.randn(4),
        action_source=action_source,
    )

    assert state.has_action() is True
    assert state.action_source.source_type == "externally_provided"
    assert state.action_source.temperature is None

    # Test to_device with action_source
    device = torch.device("cpu")
    moved = state.to_device(device)
    assert moved.action_source is not None
    assert moved.action_source.source_type == "externally_provided"

    # Test detach with action_source
    state.action_source.policy_logits = torch.randn(4, requires_grad=True)
    detached = state.detach()
    assert detached.action_source.policy_logits is not None
    assert not detached.action_source.policy_logits.requires_grad
