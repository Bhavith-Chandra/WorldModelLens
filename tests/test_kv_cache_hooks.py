"""Tests for KV-cache hook point (editing transformer KV memory)."""

import torch
import pytest

from world_model_lens import HookedWorldModel, HookPoint, WorldModelConfig
from world_model_lens.backends.base_adapter import BaseModelAdapter, WorldModelCapabilities
from world_model_lens.core.hooks import HookContext


class KVAdapter(BaseModelAdapter):
    """Simple adapter to run through run_with_cache; exposes (h, z) API.

    Nothing fancy — latents are small deterministic tensors so tests are
    easy to reason about.
    """

    def __init__(self, config):
        super().__init__(config)
        self._capabilities = WorldModelCapabilities(has_decoder=False)

    def encode(self, obs, h_prev):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        # return a small deterministic posterior and an obs encoding
        batch = obs.shape[0]
        out = (
            torch.arange(batch * self.config.d_h, dtype=torch.float32).reshape(
                batch, self.config.d_h
            )
            * 0.0
            + 1.0
        )
        return out, out

    def dynamics(self, h, action=None):
        if h.dim() == 1:
            h = h.unsqueeze(0)
        return torch.zeros_like(h)

    def transition(self, h, z, action=None):
        return h + z

    def initial_state(self, batch_size=1, device=None):
        return torch.zeros(batch_size, self.config.d_h)


def make_wm_and_obs(T: int = 3):
    cfg = WorldModelConfig(d_h=6, d_obs=4, has_decoder=False)
    adapter = KVAdapter(cfg)
    wm = HookedWorldModel(adapter=adapter, config=cfg)
    obs = torch.randn(T, cfg.d_obs)
    return wm, obs


def test_kv_hook_creates_entries():
    wm, obs = make_wm_and_obs(T=3)

    def create_kv(cache, ctx: HookContext):
        # create simple KV entries at this timestep
        cache.set_kv(0, "k", ctx.timestep, torch.full((2,), float(ctx.timestep)))
        cache.set_kv(0, "v", ctx.timestep, torch.full((2,), float(ctx.timestep) + 0.5))

    hook = HookPoint(name="kv_cache", fn=create_kv, timestep=1)
    wm.add_hook(hook)

    traj, cache = wm.run_with_cache(obs)

    assert torch.equal(cache.get_kv(0, "k", 1), torch.tensor([1.0, 1.0]))
    assert torch.equal(cache.get_kv(0, "v", 1), torch.tensor([1.5, 1.5]))


def test_kv_hook_modifies_past_entry():
    wm, obs = make_wm_and_obs(T=4)

    def create_kv(cache, ctx: HookContext):
        cache.set_kv(0, "k", ctx.timestep, torch.full((2,), 0.0))

    def modify_prev(cache, ctx: HookContext):
        # modify the kv entry from the previous timestep
        prev_t = ctx.timestep - 1
        val = cache.get_kv(0, "k", prev_t, None)
        if val is not None:
            cache.set_kv(0, "k", prev_t, val + 10.0)

    wm.add_hook(HookPoint(name="kv_cache", fn=create_kv, timestep=1))
    wm.add_hook(HookPoint(name="kv_cache", fn=modify_prev, timestep=2))

    traj, cache = wm.run_with_cache(obs)

    # created at t=1 then modified at t=2
    assert torch.equal(cache.get_kv(0, "k", 1), torch.tensor([10.0, 10.0]))


def test_kv_hook_removes_entry():
    wm, obs = make_wm_and_obs(T=4)

    def create_kv(cache, ctx: HookContext):
        cache.set_kv(0, "k", ctx.timestep, torch.tensor([9.0, 9.0]))

    def delete_prev(cache, ctx: HookContext):
        prev_t = ctx.timestep - 1
        cache.delete_kv(0, "k", prev_t)

    wm.add_hook(HookPoint(name="kv_cache", fn=create_kv, timestep=1))
    wm.add_hook(HookPoint(name="kv_cache", fn=delete_prev, timestep=2))

    traj, cache = wm.run_with_cache(obs)

    # entry at t=1 should have been deleted by hook at t=2
    assert cache.get_kv(0, "k", 1, None) is None


def test_kv_hook_time_slice_applies_across_range():
    wm, obs = make_wm_and_obs(T=5)

    calls = []

    def slice_hook(cache, ctx: HookContext):
        # record that the hook fired and write a marker
        calls.append(ctx.timestep)
        cache.set_kv(0, "v", ctx.timestep, torch.tensor([ctx.timestep]))

    # time_slice from 1 (inclusive) to 4 (exclusive) should fire at 1,2,3
    wm.add_hook(HookPoint(name="kv_cache", fn=slice_hook, time_slice=[1, 4]))

    traj, cache = wm.run_with_cache(obs)

    assert calls == [1, 2, 3]
    for t in calls:
        assert torch.equal(cache.get_kv(0, "v", t), torch.tensor([t]))
