"""Tests for WorldTrajectory."""

import pytest
import torch


def test_trajectory_length(fake_trajectory):
    """Test trajectory length property."""
    assert fake_trajectory.length == 10
    assert len(fake_trajectory) == 10


def test_state_sequence_shape(fake_trajectory):
    """Test state_sequence shape."""
    state_seq = fake_trajectory.state_sequence
    assert state_seq.shape[0] == 10


def test_reward_sequence(fake_trajectory):
    """Test reward_sequence."""
    rewards = fake_trajectory.reward_sequence
    assert rewards is not None
    assert rewards.shape[0] == 10


def test_action_sequence(fake_trajectory):
    """Test action_sequence."""
    actions = fake_trajectory.action_sequence
    assert actions is not None


def test_surprise_peaks(fake_trajectory):
    """Test surprise peaks detection."""
    peaks = fake_trajectory.surprise_peaks(threshold=100.0)
    assert isinstance(peaks, list)


def test_slice(fake_trajectory):
    """Test trajectory slicing."""
    sliced = fake_trajectory.slice(2, 5)
    assert sliced.length == 3
    assert sliced[0].timestep == 2


def test_to_device(fake_trajectory):
    """Test moving trajectory to device."""
    moved = fake_trajectory.to_device(torch.device("cpu"))
    assert moved.states[0].state.device == torch.device("cpu")


def test_fork_at(fake_trajectory):
    """Test forking trajectory."""
    forked = fake_trajectory.fork_at(5)
    assert forked.fork_point == 5
    assert forked.is_real is True


def test_source_property(fake_trajectory):
    """Test source property."""
    assert fake_trajectory.source == "real"
    assert fake_trajectory.is_real is True
    assert fake_trajectory.is_imagined is False


def test_filter_states(fake_trajectory):
    """Test filter_states method."""
    filtered = fake_trajectory.filter_states(lambda s: s.timestep < 5)
    assert filtered.length == 5


def test_map_states(fake_trajectory):
    """Test map_states method."""
    mapped = fake_trajectory.map_states(lambda s: s)
    assert mapped.length == fake_trajectory.length
