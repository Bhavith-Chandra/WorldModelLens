import torch

from world_model_lens.core.hooks import HookRegistry, HookPoint, HookContext


def test_prepend_and_remove_semantics():
    calls = []

    def make_hook(tag):
        def hook(tensor, ctx):
            calls.append(tag)
            return tensor

        return hook

    reg = HookRegistry()

    hp1 = HookPoint(name="comp", fn=make_hook("first"))
    hp2 = HookPoint(name="comp", fn=make_hook("prepended"))

    # register first normally, then prepend second
    reg.register(hp1)
    reg.register(hp2, prepend=True)

    ctx = HookContext(timestep=0, component="test", trajectory_so_far=[])
    _ = reg.apply("comp", 0, torch.tensor(1.0), ctx)

    # prepended hook should run before the originally registered one
    assert calls == ["prepended", "first"]

    # remove the prepended hook and ensure only the first remains
    reg.remove(hp2)
    calls.clear()
    _ = reg.apply("comp", 0, torch.tensor(1.0), ctx)
    assert calls == ["first"]


def test_timestep_specific_and_clear_name():
    calls = []

    def make_hook(tag):
        def hook(tensor, ctx):
            calls.append((tag, ctx.timestep))
            return tensor

        return hook

    reg = HookRegistry()

    hp_global = HookPoint(name="x", fn=make_hook("g"))
    hp_t2 = HookPoint(name="x", fn=make_hook("t2"), timestep=2)

    reg.register(hp_global)
    reg.register(hp_t2)

    ctx1 = HookContext(timestep=1, component="test", trajectory_so_far=[])
    _ = reg.apply("x", 1, torch.tensor(0.0), ctx1)
    # only global hook runs at t=1
    assert calls == [("g", 1)]

    calls.clear()
    ctx2 = HookContext(timestep=2, component="test", trajectory_so_far=[])
    _ = reg.apply("x", 2, torch.tensor(0.0), ctx2)
    # both global and timestep-specific hooks run at t=2
    assert ("g", 2) in calls and ("t2", 2) in calls

    # clear by name removes both
    reg.clear(name="x")
    calls.clear()
    _ = reg.apply("x", 2, torch.tensor(0.0), ctx2)
    assert calls == []
