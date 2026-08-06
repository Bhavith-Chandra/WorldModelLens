"""Ensure removing one hook preserves others in HookedWorldModel."""

import torch

from world_model_lens import HookedWorldModel, HookPoint


def test_remove_one_hook_preserves_other(hooked_wm):
    """Register two hooks for the same component, remove one, and ensure the other remains."""

    calls = []

    def hook_a(tensor, ctx):
        calls.append("a")
        return tensor

    def hook_b(tensor, ctx):
        calls.append("b")
        return tensor

    h_a = HookPoint(name="state", fn=hook_a)
    h_b = HookPoint(name="state", fn=hook_b)

    hooked_wm.add_hook(h_a)
    hooked_wm.add_hook(h_b)

    # run a single step to trigger hooks
    obs = torch.zeros(1, hooked_wm.config.d_obs)
    hooked_wm.run_with_hooks(obs.unsqueeze(0))

    assert "a" in calls and "b" in calls

    # clear calls and remove hook_a
    calls.clear()
    hooked_wm.remove_hook(h_a)

    # run again
    hooked_wm.run_with_hooks(obs.unsqueeze(0))

    # only hook_b should fire
    assert calls == ["b"]
