# SAE Developer Guide

This short guide shows the minimal API a new Sparse Autoencoder (SAE)
implementation should provide so it works cleanly with `SAETrainer`.

Key points
- Prefer implementing `from_config(config, device=None)` — the trainer will
  call this when available to construct your model in a single standardized way.
- `encode(x, ...)` should return either `(h, mask)` or `(h, indices)` where
  `h` is the sparse latent tensor of shape `[B, n_features]`.
- `decode(h)` should return reconstructions of shape `[B, input_dim]`.
- `forward` need not be implemented if you inherit from `SAEBase` — the
  provided base `forward` composes `encode` + `decode` and returns either
  `(recon, h)` or `(recon, h, mask)` depending on what `encode` returns.
- Optionally implement `compute_l0(h)` to return the mean L0 estimate per
  example; the trainer will fall back to a safe approximation if missing.

Minimal example

```python
from torch import nn
import torch.nn.functional as F

from world_model_lens.sae.sae import SAEBase

class MyCustomSAE(SAEBase):
    """Minimal SAE example compatible with SAETrainer."""

    def __init__(self, input_dim: int, n_features: int, k: int = 1, tie_weights: bool = False):
        super().__init__()
        self.encoder = nn.Linear(input_dim, n_features)
        self.decoder = nn.Linear(n_features, input_dim, bias=not tie_weights)
        self.k = int(k)

    @classmethod
    def from_config(cls, config, device=None):
        # config is the SAEConfig used by SAETrainer (has d_input, n_boj, k, tied_weights)
        inst = cls(input_dim=config.d_input, n_features=config.n_boj, k=config.k, tie_weights=config.tied_weights)
        if device is not None:
            inst.to(device)
        return inst

    def encode(self, x, **kwargs):
        h = self.encoder(x)
        # keep only top-k per row
        vals, idx = torch.topk(h, self.k, dim=-1)
        mask = torch.zeros_like(h).scatter_(-1, idx, 1.0)
        h = F.relu(h * mask)
        return h, mask

    def decode(self, h):
        return self.decoder(h)

    def compute_l0(self, h):
        # return mean number of non-zero features per example
        return (h.abs() > 1e-8).float().sum(dim=-1).mean()
```

Using your SAE with the trainer

```python
from world_model_lens.sae.trainer import SAETrainer
# pass the class to the trainer; trainer will call `from_config`
trainer = SAETrainer(d_input=64, n_boj=32, k=4, sae_class=MyCustomSAE)
result = trainer.train(torch.randn(256, 64), epochs=5)
```

Notes
- The trainer uses a normalized sparsity penalty: it multiplies `l1_coefficient`
  by `h.abs().mean()` (mean across batch and features). When porting code that
  used a summed L1 penalty (`.sum()`), retune the coefficient.
- If your implementation returns `(recon, h)` or `(recon, h, mask)` from
  `forward`, trainer already supports both shapes.
