import torch
from world_model_lens.sae.sae_feature_circuits import SAEFeatureCircuitAnalyzer


def test_collect_clean_acts_with_various_cache_types():
    from world_model_lens.core.activation_cache import ActivationCache
    import torch.distributions as dist

    cache = ActivationCache()

    # timestep 0: plain tensor 1D
    cache["layerA", 0] = torch.randn(4)

    # timestep 1: tensor with batch dim
    cache["layerA", 1] = torch.randn(1, 4)

    # timestep 2: distribution (Normal) -> should use mean
    cache["layerA", 2] = dist.Normal(loc=torch.zeros(4), scale=torch.ones(4))

    # timestep 3: dict containing 'mean' tensor
    cache["layerA", 3] = {"mean": torch.ones(4) * 2.0}

    # timestep 4: dict containing a tensor under explicit 'tensor' key (ActivationCache expects 'tensor' or 'mean')
    cache["layerA", 4] = {"tensor": torch.arange(4).float()}

    # SAE that works on real tensors; ensure no .to() required
    class RealSAE:
        def __init__(self, input_dim, n_features):
            self.input_dim = input_dim
            self.n_features = n_features

        def encode(self, x):
            B = x.shape[0]
            out = x.mean(dim=1, keepdim=True).repeat(1, self.n_features)
            return out, torch.arange(self.n_features).unsqueeze(0).repeat(B, 1)

        def decode(self, h):
            return h.mean(dim=1, keepdim=True).repeat(1, self.input_dim)

    saes = {"layerA": RealSAE(input_dim=4, n_features=2)}
    cache_keys = {"layerA": "layerA"}

    analyzer = SAEFeatureCircuitAnalyzer(wm=None, saes=saes, cache_keys=cache_keys)
    clean_acts = analyzer._collect_clean_acts(cache)

    assert "layerA" in clean_acts
    # stacked over 5 timesteps -> shape[0] == 5
    assert clean_acts["layerA"].shape[0] == 5
