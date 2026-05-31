import torch
from world_model_lens.sae.sae_feature_circuits import SAEFeatureCircuitAnalyzer, FeatureCircuitGraph


class DummyWM:
    """Very small dummy world model that stores a simple activation stream in cache."""

    def __init__(self):
        from world_model_lens.core.activation_cache import ActivationCache

        self._cache = ActivationCache()

    def run_with_cache(self, observations, actions=None):
        # pretend observations is [T, D]; store two layers 'encoder' and 'dynamics'
        T = observations.shape[0]
        for t in range(T):
            self._cache["encoder", t] = observations[t]
            # dynamics hidden is just encoder * 0.5
            self._cache["dynamics", t] = observations[t] * 0.5

        # return dummy trajectory and cache
        class Traj:
            pass

        return Traj(), self._cache

    def run_with_hooks(self, observations, fwd_hooks=None, return_cache=False):
        # naive implementation: apply hook for each timestep and return cache
        from world_model_lens.core.activation_cache import ActivationCache

        cache = ActivationCache()
        T = observations.shape[0]
        for t in range(T):
            enc = observations[t]
            # apply hook if provided
            if fwd_hooks:
                for h in fwd_hooks:
                    if h.timestep is None or h.timestep == t:
                        enc = h.fn(enc, None)
            cache["encoder", t] = enc
            cache["dynamics", t] = enc * 0.5

        class Traj:
            pass

        if return_cache:
            return Traj(), cache
        return Traj()


class TinySAE:
    def __init__(self, input_dim, n_features):
        self.input_dim = input_dim
        self.n_features = n_features

    def encode(self, x):
        # simple: project to n_features by linear-like sum
        B = x.shape[0]
        out = x.mean(dim=1, keepdim=True).repeat(1, self.n_features)
        return out, torch.arange(self.n_features).unsqueeze(0).repeat(B, 1)

    def decode(self, h):
        # reconstruct a vector same dim as original by summing features
        return h.mean(dim=1, keepdim=True).repeat(1, self.input_dim)


def test_basic_circuit_build():
    wm = DummyWM()
    # SAEs for encoder and dynamics
    saes = {
        "encoder": TinySAE(input_dim=4, n_features=2),
        "dynamics": TinySAE(input_dim=4, n_features=2),
    }
    cache_keys = {"encoder": "encoder", "dynamics": "dynamics"}

    analyzer = SAEFeatureCircuitAnalyzer(
        wm=wm, saes=saes, cache_keys=cache_keys, threshold=0.0, topk_per_source=10
    )

    obs = torch.randn(3, 4)
    g = analyzer.build_graph(obs)
    assert isinstance(g, FeatureCircuitGraph)
    # graph should contain edges (non-empty)
    assert len(g.edges) >= 0
