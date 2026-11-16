"""run_with_cache must execute the adapter's own RSSM step and let hooks intervene."""

import pytest
import torch
from torch.distributions import Categorical, kl_divergence

from world_model_lens import HookedWorldModel
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
