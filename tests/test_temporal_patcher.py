"""TemporalPatcher on a linear world model where every answer is known in closed form."""

import math
from types import SimpleNamespace

import pytest
import torch

from world_model_lens import HookedWorldModel
from world_model_lens.backends.base_adapter import BaseModelAdapter
from world_model_lens.core.hooks import HookPoint
from world_model_lens.patching.patcher import CorruptedCacheFactory, TemporalPatcher

T = 5
D = 3


class LinearWorldModel(BaseModelAdapter):
    """h_{t+1} = 0.5 * h_t + w * obs_t."""

    def __init__(self):
        super().__init__(SimpleNamespace(d_h=D, d_z=D))
        self.w = torch.tensor([1.0, -2.0, 3.0])

    def encode(self, obs, h_prev):
        z = obs * self.w
        return z, z

    def transition(self, h, z, action=None):
        return 0.5 * h + z


def final_state_sum(cache):
    return float(cache["state", T - 1].sum())


@pytest.fixture
def wm():
    adapter = LinearWorldModel()
    return HookedWorldModel(adapter, adapter.config)


@pytest.fixture
def patcher(wm):
    return TemporalPatcher(wm, device=torch.device("cpu"))


@pytest.fixture
def runs(wm):
    clean_obs = torch.arange(T * D, dtype=torch.float32).reshape(T, D) / 10
    corrupted_obs = clean_obs.clone()
    corrupted_obs[0] += 1.0
    actions = torch.zeros(T, 1)
    _, _, clean_cache = wm.run_with_cache(clean_obs, actions)
    _, _, corrupted_cache = wm.run_with_cache(corrupted_obs, actions)
    return clean_obs, corrupted_obs, actions, clean_cache, corrupted_cache


def patch(patcher, runs, t, **kwargs):
    clean_obs, corrupted_obs, actions, clean_cache, corrupted_cache = runs
    kwargs.setdefault("corrupted_obs_seq", corrupted_obs)
    return patcher.patch_state(
        clean_cache, corrupted_cache, "state", t, final_state_sum, clean_obs, actions, **kwargs
    )


def test_metrics_come_from_the_runs(patcher, runs):
    *_, clean_cache, corrupted_cache = runs
    result = patch(patcher, runs, 1)

    assert result.metric_clean == pytest.approx(final_state_sum(clean_cache))
    assert result.metric_corrupted == pytest.approx(final_state_sum(corrupted_cache))
    assert result.metric_clean != result.metric_corrupted


def test_denoising_a_mediating_site_fully_recovers(patcher, runs):
    for t in range(1, T):
        result = patch(patcher, runs, t)
        assert result.patch_mode == "denoise"
        assert result.recovery_rate == pytest.approx(1.0)
        assert result.metric_patched == pytest.approx(result.metric_clean)


def test_patching_before_the_corruption_has_no_effect(patcher, runs):
    result = patch(patcher, runs, 0)
    assert result.recovery_rate == pytest.approx(0.0)


def test_noising_a_mediating_site_reproduces_the_corruption(patcher, runs):
    result = patch(patcher, runs, 2, corrupted_obs_seq=None)
    assert result.patch_mode == "noise"
    assert result.recovery_rate == pytest.approx(0.0)
    assert result.metric_patched == pytest.approx(result.metric_corrupted)


def test_recovery_is_not_clipped(patcher, runs):
    *_, clean_cache, corrupted_cache = runs
    overshoot = 2 * clean_cache["state", 2] - corrupted_cache["state", 2]
    result = patch(patcher, runs, 2, patch_value=overshoot)
    assert result.recovery_rate == pytest.approx(2.0)


def test_recovery_is_nan_without_a_clean_corrupted_difference(patcher, runs):
    clean_obs, _, actions, clean_cache, _ = runs
    result = patcher.patch_state(
        clean_cache,
        clean_cache,
        "state",
        2,
        final_state_sum,
        clean_obs,
        actions,
        corrupted_obs_seq=clean_obs,
    )
    assert math.isnan(result.recovery_rate)


def test_user_hooks_survive_patching(wm, patcher, runs):
    calls = []
    user_hook = HookPoint(name="z_posterior", fn=lambda x, ctx: calls.append(ctx.timestep) or x)
    wm.add_hook(user_hook)

    patch(patcher, runs, 2)

    assert len(wm.hook_registry) == 1
    assert calls == list(range(T))


def test_full_sweep_ranks_measured_patches(patcher, runs):
    clean_obs, corrupted_obs, actions, clean_cache, corrupted_cache = runs
    sweep = patcher.full_sweep(
        clean_cache,
        corrupted_cache,
        ["state"],
        final_state_sum,
        clean_obs_seq=clean_obs,
        clean_action_seq=actions,
        corrupted_obs_seq=corrupted_obs,
    )

    assert [p.timestep for p in sweep.top_k_patches(k=T)][: T - 1] == [1, 2, 3, 4]
    assert sweep.get_timestep_rankings()[0] == pytest.approx(0.0)
    torch.testing.assert_close(sweep.recovery_matrix(), torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0]]))


def test_restoring_one_timestep_after_corrupting_all(patcher, runs):
    clean_obs, _, actions, *_ = runs

    def final_state(traj):
        return float(traj.states[-1].state.sum())

    def zero(x):
        return torch.zeros_like(x)

    last = patcher.patch_state_with_corruption(
        clean_obs, actions, zero, "state", T - 1, final_state
    )
    earlier = patcher.patch_state_with_corruption(clean_obs, actions, zero, "state", 1, final_state)

    assert last.metric_clean != last.metric_corrupted
    assert last.recovery_rate == pytest.approx(1.0)
    assert earlier.recovery_rate == pytest.approx(0.0)


def test_shuffle_permutes_timesteps(runs):
    *_, clean_cache, _ = runs
    torch.manual_seed(0)
    shuffled = CorruptedCacheFactory.shuffle(clean_cache, components=["state"])

    original = sorted(clean_cache["state", t].tolist() for t in clean_cache.timesteps)
    permuted = sorted(shuffled["state", t].tolist() for t in shuffled.timesteps)
    assert permuted == original
    assert shuffled["state"].shape == clean_cache["state"].shape


def test_patching_an_unreached_site_raises(patcher, runs):
    clean_obs, _, actions, clean_cache, corrupted_cache = runs
    with pytest.raises(ValueError, match="not computed"):
        patcher.patch_state(
            clean_cache,
            corrupted_cache,
            "pos",
            2,
            final_state_sum,
            clean_obs,
            actions,
            patch_value=torch.zeros(D),
        )
