import torch
from world_model_lens.sae.sae_feature_circuits import SAEFeatureCircuitAnalyzer, FeatureCircuitGraph


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
    # create an ActivationCache and populate it with encoder/dynamics activations
    from world_model_lens.core.activation_cache import ActivationCache

    cache = ActivationCache()
    obs = torch.randn(3, 4)
    for t in range(obs.shape[0]):
        cache["encoder", t] = obs[t]
        cache["dynamics", t] = obs[t] * 0.5

    # SAEs for encoder and dynamics
    saes = {
        "encoder": TinySAE(input_dim=4, n_features=2),
        "dynamics": TinySAE(input_dim=4, n_features=2),
    }
    cache_keys = {"encoder": "encoder", "dynamics": "dynamics"}

    analyzer = SAEFeatureCircuitAnalyzer(
        wm=None, saes=saes, cache_keys=cache_keys, threshold=0.0, topk_per_source=10
    )

    clean_acts = analyzer._collect_clean_acts(cache)
    assert "encoder" in clean_acts and "dynamics" in clean_acts
    assert clean_acts["encoder"].shape[0] == 3
