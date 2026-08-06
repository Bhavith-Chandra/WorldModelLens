import pytest
import torch

from world_model_lens.sae.sae import TopKSparseAutoencoder, SAEBase
from world_model_lens.sae.trainer import SAETrainer


def test_topk_mask_behavior():
    # small controlled example where encoder is identity so encoded values == inputs
    sae = TopKSparseAutoencoder(input_dim=4, n_features=4, k=2)
    with torch.no_grad():
        sae.encoder.weight.copy_(torch.eye(4))
        if sae.encoder.bias is not None:
            sae.encoder.bias.zero_()

    x = torch.tensor([[0.1, 0.9, 0.2, 0.8], [1.0, 0.5, 0.4, 0.3]])

    recon, h, mask = sae(x)

    # h should have exactly k nonzero entries per row
    nnz_per_row = (h.abs() > 0).sum(dim=1).tolist()
    assert nnz_per_row == [2, 2]

    # mask positions should match top-k indices of the original pre-ReLU values
    for row_vals, row_mask in zip(x, mask):
        mask_idxs = (row_mask == 1).nonzero(as_tuple=True)[0].tolist()
        topk_idxs = torch.topk(row_vals, 2).indices.tolist()
        assert set(mask_idxs) == set(topk_idxs)


def test_compute_loss_and_diagnostics():
    sae = TopKSparseAutoencoder(input_dim=4, n_features=4, k=2)
    x = torch.randn(8, 4)

    total, info = sae.compute_loss(x, l1_weight=1e-2)

    assert isinstance(total, torch.Tensor)
    assert isinstance(info, dict)
    assert set(["reconstruction", "sparsity", "total"]).issubset(set(info.keys()))
    # numeric consistency
    assert info["total"] == pytest.approx(float(total.item()), rel=1e-6)


def test_tied_weights_copy():
    sae = TopKSparseAutoencoder(input_dim=3, n_features=3, k=1, tie_weights=True)
    # decoder weight should be a copy of encoder.weight.T at initialization
    assert torch.allclose(sae.decoder.weight.detach(), sae.encoder.weight.T.detach())


def test_sae_trainer_with_gated_type_smoke():
    d_input = 8
    n_boj = 4
    k = 2

    activations = torch.randn(16, d_input)

    trainer = SAETrainer(d_input=d_input, n_boj=n_boj, k=k, sae_type="gated")

    # run one-epoch quick training
    result = trainer.train(activations, epochs=1, batch_size=8, lr=1e-3, progress=False)

    assert hasattr(result, "sae")
    assert isinstance(result.losses, list)
    assert len(result.losses) == 1


def test_sae_trainer_with_compact_topk_class_smoke():
    d_input = 8
    n_boj = 6
    k = 3

    activations = torch.randn(12, d_input)

    # pass the compact TopKSparseAutoencoder class directly
    trainer = SAETrainer(d_input=d_input, n_boj=n_boj, k=k, sae_class=TopKSparseAutoencoder)

    result = trainer.train(activations, epochs=1, batch_size=6, lr=1e-3, progress=False)

    assert hasattr(result, "sae")
    assert isinstance(result.losses, list)
    assert len(result.losses) == 1


def test_sae_trainer_tied_weights_keyword_smoke():
    """Ensure trainer constructs compact SAE with tied-weights when requested."""

    torch.manual_seed(0)

    d_input = 10
    n_boj = 5
    k = 2

    activations = torch.randn(8, d_input)

    # request tied_weights via trainer parameter (mapped to sae ctor)
    trainer = SAETrainer(
        d_input=d_input, n_boj=n_boj, k=k, sae_class=TopKSparseAutoencoder, tied_weights=True
    )

    # model should have been constructed
    assert hasattr(trainer, "sae")

    sae = trainer.sae
    # check shapes
    assert sae.encoder.weight.shape == (n_boj, d_input)
    assert sae.decoder.weight.shape == (d_input, n_boj)

    # values should be close to the transpose (copied at init)
    assert torch.allclose(sae.decoder.weight, sae.encoder.weight.t(), atol=1e-6)


def test_trainer_constructs_from_config_minimal_sae_smoke():
    class MinimalFromConfigSAE(SAEBase):
        def __init__(self, input_dim: int, n_features: int, k: int = 1, tie_weights: bool = False):
            super().__init__()
            self.encoder = torch.nn.Linear(input_dim, n_features)
            self.decoder = torch.nn.Linear(n_features, input_dim)
            self.k = int(k)

        @classmethod
        def from_config(cls, config, device=None):
            inst = cls(input_dim=config.d_input, n_features=config.n_boj, k=config.k)
            if device is not None:
                inst.to(device)
            return inst

        def encode(self, x, **kwargs):
            h = self.encoder(x)
            h = torch.relu(h)
            mask = (h > 0).float()
            return h, mask

        def decode(self, h):
            return self.decoder(h)

    d_input = 6
    n_boj = 3
    k = 2

    activations = torch.randn(10, d_input)

    trainer = SAETrainer(d_input=d_input, n_boj=n_boj, k=k, sae_class=MinimalFromConfigSAE)

    result = trainer.train(activations, epochs=1, batch_size=5, lr=1e-3, progress=False)

    assert hasattr(result, "sae")
    assert isinstance(result.sae, MinimalFromConfigSAE)
    assert isinstance(result.losses, list)
    assert len(result.losses) == 1


def test_sae_trainer_variants_grouped_smoke():
    # reuse small cases from the separate variants test file and ensure they pass
    d_input = 8
    n_boj = 4
    k = 2

    activations = torch.randn(16, d_input)

    trainer = SAETrainer(d_input=d_input, n_boj=n_boj, k=k, sae_type="gated")
    result = trainer.train(activations, epochs=1, batch_size=8, lr=1e-3, progress=False)
    assert hasattr(result, "sae")

    d_input = 8
    n_boj = 6
    k = 3
    activations = torch.randn(12, d_input)
    trainer = SAETrainer(d_input=d_input, n_boj=n_boj, k=k, sae_class=TopKSparseAutoencoder)
    result = trainer.train(activations, epochs=1, batch_size=6, lr=1e-3, progress=False)
    assert hasattr(result, "sae")

    # tied-weights construction smoke
    d_input = 10
    n_boj = 5
    k = 2
    trainer = SAETrainer(
        d_input=d_input, n_boj=n_boj, k=k, sae_class=TopKSparseAutoencoder, tied_weights=True
    )
    sae = trainer.sae
    assert sae.encoder.weight.shape == (n_boj, d_input)
