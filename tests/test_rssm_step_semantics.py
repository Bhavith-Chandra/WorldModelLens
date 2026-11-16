"""run_with_cache and imagine must execute the adapter's own RSSM step."""

from types import SimpleNamespace

import pytest
import torch
from torch.distributions import Categorical, kl_divergence

from world_model_lens import HookedWorldModel
from world_model_lens.backends.base_adapter import BaseModelAdapter
from world_model_lens.backends.dreamerv3 import DreamerV3Adapter, default_arch
from world_model_lens.core.hooks import HookPoint

T = 5
SMALL = {
    "deter": 64,
    "hidden": 16,
    "units": 16,
    "depth": 2,
    "stoch": 8,
    "classes": 4,
    "action_dim": 6,
    "img_shape": (32, 32, 1),
}


@pytest.fixture
def adapter():
    torch.manual_seed(0)
    return DreamerV3Adapter(arch=default_arch(**SMALL)).eval()


@pytest.fixture
def inputs():
    g = torch.Generator().manual_seed(1)
    obs = torch.randint(0, 255, (T, 32, 32, 1), generator=g).float()
    actions = torch.randint(0, SMALL["action_dim"], (T,), generator=g)
    return obs, actions


def reference_rollout(adapter, obs, actions, sample=False):
    h, _ = adapter.initial_state(1)
    steps = []
    for t in range(T):
        post, _ = adapter.encode(obs[t : t + 1], h)
        z = adapter.sample_z(post, sample=sample)
        steps.append(
            {
                "h": h[0],
                "post": post[0],
                "prior": adapter.dynamics(h)[0],
                "z": z[0],
                "reward": adapter.predict_reward(h, z)[0],
                "value": adapter.critic_forward(h, z)[0],
            }
        )
        h = adapter.transition(h, z, actions[t : t + 1])
    return steps


@torch.no_grad()
def test_matches_reference_rollout(adapter, inputs):
    obs, actions = inputs
    ref = reference_rollout(adapter, obs, actions)
    world, latent, cache = HookedWorldModel(adapter, adapter.config).run_with_cache(obs, actions)

    for t, step in enumerate(ref):
        torch.testing.assert_close(cache["h", t], step["h"])
        torch.testing.assert_close(cache["z_posterior", t], step["post"])
        torch.testing.assert_close(cache["z_prior", t], step["prior"])
        torch.testing.assert_close(cache["z", t], step["z"])
        torch.testing.assert_close(cache["reward", t], step["reward"])
        torch.testing.assert_close(cache["value", t], step["value"])
        torch.testing.assert_close(latent.states[t].h_t, step["h"])
        torch.testing.assert_close(world.states[t].state, step["h"])


@torch.no_grad()
def test_sampled_rollout_matches_reference(adapter, inputs):
    obs, actions = inputs
    torch.manual_seed(7)
    ref = reference_rollout(adapter, obs, actions, sample=True)
    torch.manual_seed(7)
    _, _, cache = HookedWorldModel(adapter, adapter.config).run_with_cache(
        obs, actions, sample_latents=True
    )

    for t, step in enumerate(ref):
        torch.testing.assert_close(cache["z", t], step["z"])
        torch.testing.assert_close(cache["h", t], step["h"])


@torch.no_grad()
def test_latent_states_hold_probabilities_and_true_kl(adapter, inputs):
    obs, actions = inputs
    _, latent, cache = HookedWorldModel(adapter, adapter.config).run_with_cache(obs, actions)

    for t in range(T):
        post_logp = adapter.latent_logprobs(cache["z_posterior", t])
        prior_logp = adapter.latent_logprobs(cache["z_prior", t])
        expected = kl_divergence(Categorical(logits=post_logp), Categorical(logits=prior_logp))
        torch.testing.assert_close(cache["kl", t], expected)
        torch.testing.assert_close(latent.states[t].kl, expected.sum())
        torch.testing.assert_close(latent.states[t].z_posterior.sum(-1), torch.ones(SMALL["stoch"]))


@pytest.mark.parametrize("component", ["h", "z_prior", "z", "reward"])
@torch.no_grad()
def test_hooks_change_downstream_computation(adapter, inputs, component):
    obs, actions = inputs
    wm = HookedWorldModel(adapter, adapter.config)
    _, clean, _ = wm.run_with_cache(obs, actions)

    seen = []

    def perturb(tensor, ctx):
        seen.append(ctx.component)
        return 2.0 * tensor + 1.0

    wm.add_hook(HookPoint(name=component, fn=perturb, timestep=2))
    _, patched, _ = wm.run_with_cache(obs, actions)

    assert seen == [component]
    torch.testing.assert_close(patched.states[1].h_t, clean.states[1].h_t)
    if component == "z_prior":
        assert not torch.allclose(patched.states[2].z_prior, clean.states[2].z_prior)
    elif component == "reward":
        assert not torch.allclose(patched.states[2].reward_pred, clean.states[2].reward_pred)
    else:
        assert not torch.allclose(patched.states[3].h_t, clean.states[3].h_t)


def reference_imagine(adapter, h, z, actions):
    steps = []
    for action in actions:
        h = adapter.transition(h, z, action[None])
        z = adapter.sample_z(adapter.dynamics(h))
        steps.append({"h": h[0], "z": z[0], "reward": adapter.predict_reward(h, z)[0]})
    return steps


@torch.no_grad()
def test_imagine_matches_reference(adapter, inputs):
    obs, actions = inputs
    wm = HookedWorldModel(adapter, adapter.config)
    _, latent, _ = wm.run_with_cache(obs, actions)
    start = latent.states[1]
    plan = actions[2:]

    torch.manual_seed(3)
    ref = reference_imagine(adapter, start.h_t[None], start.z[None], plan)
    torch.manual_seed(3)
    world, imagined = wm.imagine(start, actions=plan, horizon=len(plan))

    for k, step in enumerate(ref):
        torch.testing.assert_close(imagined.states[k].h_t, step["h"])
        torch.testing.assert_close(imagined.states[k].z, step["z"])
        torch.testing.assert_close(imagined.states[k].reward_pred, step["reward"])
        torch.testing.assert_close(world.states[k].state, step["h"])
        assert imagined.states[k].timestep == start.timestep + k + 1


@torch.no_grad()
def test_imagine_continues_from_observed_state(adapter, inputs):
    obs, actions = inputs
    wm = HookedWorldModel(adapter, adapter.config)
    _, latent, _ = wm.run_with_cache(obs, actions)

    for t in range(T - 1):
        _, imagined = wm.imagine(latent.states[t], actions=actions[t : t + 1], horizon=1)
        torch.testing.assert_close(imagined.states[0].h_t, latent.states[t + 1].h_t)


@torch.no_grad()
def test_imagine_samples_policy_actions(adapter, inputs):
    obs, actions = inputs
    wm = HookedWorldModel(adapter, adapter.config)
    _, latent, _ = wm.run_with_cache(obs, actions)

    world, imagined = wm.imagine(latent.states[-1], horizon=4)

    assert imagined.length == 4
    for state in world.states:
        assert state.action_source.source_type == "policy_sampled"
        assert 0 <= int(state.action) < SMALL["action_dim"]


@torch.no_grad()
def test_imagine_requires_sampled_start_latent(adapter, inputs):
    obs, actions = inputs
    wm = HookedWorldModel(adapter, adapter.config)
    world, latent, _ = wm.run_with_cache(obs, actions)

    with pytest.raises(ValueError):
        wm.imagine(world.states[0], horizon=2)
    latent.states[0].z = None
    with pytest.raises(ValueError):
        wm.imagine(latent.states[0], horizon=2)


@torch.no_grad()
def test_latent_state_keeps_z_when_moved(adapter, inputs):
    obs, actions = inputs
    _, latent, _ = HookedWorldModel(adapter, adapter.config).run_with_cache(obs, actions)
    state = latent.states[0]

    torch.testing.assert_close(state.to_device(torch.device("cpu")).z, state.z)
    torch.testing.assert_close(state.detach().z, state.z)


class ContinuousWorldModel(BaseModelAdapter):
    """h_{t+1} = 0.5 * h_t + z_t with z_t = obs_t and no categorical latents."""

    def __init__(self):
        super().__init__(SimpleNamespace(d_h=3, d_z=3))

    def encode(self, obs, h_prev):
        return obs, obs

    def transition(self, h, z, action=None):
        return 0.5 * h + z

    def sample_z(self, logits_or_repr, temperature=1.0, sample=True):
        return logits_or_repr


@torch.no_grad()
def test_imagine_uses_edited_posterior_for_continuous_latents():
    adapter = ContinuousWorldModel()
    wm = HookedWorldModel(adapter, adapter.config)
    _, latent, _ = wm.run_with_cache(torch.ones(T, 3))
    start = latent.states[2]
    assert start.z is None

    _, unchanged = wm.imagine(start, horizon=1)
    start.z_posterior = start.z_posterior + 1.0
    _, edited = wm.imagine(start, horizon=1)

    torch.testing.assert_close(edited.states[0].h_t, unchanged.states[0].h_t + 1.0)
