import torch
from torch import Tensor

from world_model_lens.core.forward_runner import ForwardRunner
from world_model_lens.core.activation_cache import ActivationCache
from world_model_lens.core.latent_trajectory import LatentTrajectory


class FakeHooked:
    """Minimal object exposing the helpers ForwardRunner expects."""

    def __init__(self, T: int):
        self.name = "fake"
        self.T = T

        # simple adapter with initial_state
        class A:
            def initial_state(inner_self):
                return torch.zeros(self.T) if False else torch.zeros(4)

        self._adapter = A()

    def _encode(self, obs: Tensor, t: int, ctx, cache, names_filter):
        return obs + 1.0

    def _apply_and_cache(self, name, t, tensor, ctx, cache, names_filter):
        # mimic hook pass-through and caching
        if cache is not None and (names_filter is None or name in names_filter):
            cache[name, t] = tensor.detach()
        return tensor

    def _dynamics_and_prior(self, h, z, a_prev, t, ctx, cache, names_filter):
        # return next h and a fake prior logits
        h_next = h + 0.1
        logits = torch.tensor([0.6, 0.4])
        probs = logits.softmax(dim=-1)
        if cache is not None:
            cache["z_prior.logits", t] = logits.detach()
            cache["z_prior", t] = probs.detach()
        return h_next, logits, probs

    def _posterior(self, h, obs_emb, t, ctx, cache, names_filter):
        logits = obs_emb + 0.5
        probs = logits.softmax(dim=-1)
        if cache is not None:
            cache["z_posterior.logits", t] = logits.detach()
            cache["z_posterior", t] = probs.detach()
        return logits, probs

    def _compute_kl_and_cache(self, z_post_prob, z_prior_prob, t, ctx, cache, names_filter):
        # small deterministic scalar
        kl = ((z_post_prob - z_prior_prob).abs()).sum()
        if cache is not None:
            cache["kl", t] = kl.detach()
        return kl

    def _compute_heads(self, h, z, t, ctx, cache, names_filter):
        # return known scalars
        return 0.0, 1.0, None, 2.0

    def _build_state(
        self,
        h,
        z_post_prob,
        z_prior_prob,
        t,
        action_seq,
        reward_val,
        cont_val,
        actor_logits_out,
        value_val,
    ):
        # simple object with attributes used in assertions
        class S:
            pass

        s = S()
        s.h_t = h.detach()
        s.z_posterior = z_post_prob.detach()
        s.z_prior = z_prior_prob.detach()
        s.timestep = t
        s.reward_pred = reward_val
        s.cont_pred = cont_val
        s.value_pred = value_val
        return s


def test_forward_runner_basic():
    T = 3
    fake = FakeHooked(T)
    runner = ForwardRunner(fake)

    obs = torch.randn(T, 2)
    actions = torch.zeros(T, 1)
    cache = ActivationCache()

    traj = runner.run_forward(obs, actions, cache, names_filter=None, no_grad=True)
    assert isinstance(traj, LatentTrajectory)
    assert traj.length == T
    # check cache keys exist for a few expected components
    assert cache.get("kl", 0) is not None
    assert cache.get("z_posterior", 0) is not None
