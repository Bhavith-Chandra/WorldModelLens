import pytest
import torch

from world_model_lens.core.hooks import HookPoint, HookContext
from world_model_lens import HookedWorldModel, WorldModelConfig


def test_temp_hooks_are_cleaned_up(hooked_wm, fake_obs_seq, fake_action_seq):
    calls = []

    def patch_z(tensor, ctx: HookContext):
        calls.append((ctx.timestep, ctx.component))
        return tensor * 0.0

    hp = HookPoint(name="z_posterior", fn=patch_z, timestep=2)

    # run with hook and request cache
    traj, cache = hooked_wm.run_with_hooks(fake_obs_seq, fake_action_seq, [hp], return_cache=True)

    # Hook should have been called at t=2
    assert (2, "z_posterior") in calls

    # After run_with_hooks returns the registry should not retain the hook
    # Registering the same HookPoint again should result in one call only
    calls.clear()
    traj2, cache2 = hooked_wm.run_with_hooks(fake_obs_seq, fake_action_seq, [hp], return_cache=True)
    assert calls.count((2, "z_posterior")) == 1


def test_temp_hooks_cleanup_on_exception(hooked_wm, fake_obs_seq, fake_action_seq):
    def bad_hook(tensor, ctx: HookContext):
        raise RuntimeError("boom")

    hp = HookPoint(name="z_posterior", fn=bad_hook, timestep=0)

    with pytest.raises(RuntimeError):
        hooked_wm.run_with_hooks(fake_obs_seq, fake_action_seq, [hp], return_cache=False)

    # After exception, hook should not remain registered — run again without it
    # Should not raise
    hooked_wm.run_with_hooks(fake_obs_seq, fake_action_seq, [], return_cache=False)
