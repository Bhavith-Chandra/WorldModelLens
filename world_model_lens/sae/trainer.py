"""Sparse Autoencoder trainer for world model feature discovery."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm


def _get_device() -> torch.device:
    """Get the best available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TopKReLU(nn.Module):
    """Top-k ReLU activation function.

    Keeps only the top-k largest values, sets others to zero.
    """

    def __init__(self, k: int):
        """Initialize TopK ReLU.

        Args:
            k: Number of active neurons.
        """
        super().__init__()
        self.k = k

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass with top-k sparsity.

        Args:
            x: Input tensor.

        Returns:
            Tuple of (output tensor, mask tensor).
        """
        if self.k >= x.shape[-1]:
            return x, torch.ones_like(x)

        values, indices = torch.topk(x, self.k, dim=-1)
        mask = torch.zeros_like(x).scatter_(-1, indices, 1.0)

        output = x * mask
        return output, mask

    def extra_repr(self) -> str:
        return f"k={self.k}"


@dataclass
class SAEConfig:
    """Configuration for Sparse Autoencoder."""

    d_input: int
    n_boj: int
    k: int
    l1_coefficient: float = 1e-3
    tied_weights: bool = True
    initialization: str = "xavier"


class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder with Top-k ReLU activation.

    Learns sparse, interpretable latent features from activations.
    """

    def __init__(self, config: SAEConfig, device: Optional[torch.device] = None):
        """Initialize SAE.

        Args:
            config: SAE configuration.
            device: Device for tensors.
        """
        super().__init__()
        self.config = config
        self.device = device or _get_device()

        if config.initialization == "xavier":
            init_fn = nn.init.xavier_uniform_
        elif config.initialization == "kaiming":
            init_fn = nn.init.kaiming_normal_
        else:
            init_fn = nn.init.normal_

        self.encoder = nn.Linear(config.d_input, config.n_boj, bias=False)
        init_fn(self.encoder.weight)

        if config.tied_weights:
            self.decoder = None
        else:
            self.decoder = nn.Linear(config.n_boj, config.d_input, bias=False)
            init_fn(self.decoder.weight)

        self.topk = TopKReLU(config.k)

        self.to(self.device)

    @property
    def decoder_weight(self) -> Tensor:
        """Get decoder weights (tied or separate)."""
        if self.config.tied_weights:
            return self.encoder.weight.T
        return self.decoder.weight

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode input to sparse latent.

        Args:
            x: Input tensor [batch, d_input].

        Returns:
            Tuple of (sparse latents, mask).
        """
        h = self.encoder(x)
        sparse_h, mask = self.topk(h)
        return sparse_h, mask

    def decode(self, h: Tensor) -> Tensor:
        """Decode sparse latent to reconstruction.

        Args:
            h: Sparse latent tensor [batch, n_boj].

        Returns:
            Reconstructed tensor.
        """
        return F.linear(h, self.decoder_weight)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Tuple of (reconstruction, sparse_latent, mask).
        """
        sparse_h, mask = self.encode(x)
        reconstruction = self.decode(sparse_h)
        return reconstruction, sparse_h, mask

    def compute_l0(self, h: Tensor) -> Tensor:
        """Compute L0 norm (number of non-zero elements).

        Args:
            h: Latent tensor.

        Returns:
            L0 norm.
        """
        return (h.abs() > 1e-8).float().sum(dim=-1).mean()

    def get_feature_weights(self) -> Tensor:
        """Get decoder weights as feature vectors.

        Returns:
            Feature weight matrix [n_boj, d_input].
        """
        return self.decoder_weight


@dataclass
class SAETrainingResult:
    """Results from SAE training."""

    sae: SparseAutoencoder
    losses: List[float]
    l0_values: List[float]
    reconstruction_losses: List[float]
    l1_losses: List[float]
    final_l0: float
    final_reconstruction_loss: float
    epochs: int
    device: torch.device


class SAETrainer:
    """Trainer for Sparse Autoencoders."""

    def __init__(
        self,
        d_input: int,
        n_boj: int,
        k: int,
        l1_coefficient: float = 1e-3,
        device: Optional[torch.device] = None,
        tied_weights: bool = True,
        sae_class: Optional[Callable] = None,
        sae_kwargs: Optional[Dict[str, Any]] = None,
        sae_type: Optional[str] = None,
    ):
        """Initialize trainer.

        Args:
            d_input: Input dimension.
            n_boj: Number of bottleneck neurons.
            k: Top-k sparsity parameter.
            l1_coefficient: L1 regularization strength.
            device: Device for training.
            tied_weights: Whether to tie encoder/decoder weights.
        """
        config = SAEConfig(
            d_input=d_input,
            n_boj=n_boj,
            k=k,
            l1_coefficient=l1_coefficient,
            tied_weights=tied_weights,
        )
        # keep config on trainer so we don't rely on model having .config
        self.config = config
        # allow injecting alternate SAE implementations via sae_class
        self.device = device or _get_device()
        sae_kwargs = sae_kwargs or {}

        # allow selecting common SAE implementations by name for simplicity
        if sae_type is not None and sae_class is None:
            # keep imports local to avoid circular import at module load
            try:
                from world_model_lens.sae.sae import (
                    GatedSparseAutoencoder,
                    TopKSparseAutoencoder,
                )
            except Exception:
                GatedSparseAutoencoder = None
                TopKSparseAutoencoder = None

            name = sae_type.lower()
            if name in ("trainer", "sparse", "default"):
                sae_class = None
            elif name in ("gated",):
                sae_class = GatedSparseAutoencoder
            elif name in ("topk", "compact", "compact_topk"):
                sae_class = TopKSparseAutoencoder
            else:
                # leave sae_class as provided (possibly None)
                sae_class = sae_class

        if sae_class is None:
            # default trainer-local SparseAutoencoder
            self.sae = SparseAutoencoder(config, self.device)
        else:
            # Prefer standardized constructor if available: from_config(config, device)
            constructed = False
            if hasattr(sae_class, "from_config"):
                try:
                    self.sae = sae_class.from_config(config, self.device)
                    constructed = True
                except Exception:
                    constructed = False

            if not constructed:
                # Try common explicit (config, device) constructor next
                try:
                    self.sae = sae_class(config, self.device)
                    constructed = True
                except Exception:
                    constructed = False

            if not constructed:
                # Try keyword-style constructor that many compact SAEs support
                try:
                    self.sae = sae_class(
                        input_dim=d_input,
                        n_features=n_boj,
                        k=k,
                        tie_weights=tied_weights,
                        **sae_kwargs,
                    )
                    constructed = True
                except Exception:
                    constructed = False

            if not constructed:
                # try simple positional constructors: (d_input, n_boj, k)
                try:
                    self.sae = sae_class(d_input, n_boj, k)
                    constructed = True
                except Exception:
                    constructed = False

            if not constructed:
                # try simplest positional (d_input, n_boj)
                try:
                    self.sae = sae_class(d_input, n_boj)
                    constructed = True
                except Exception as e:
                    # give up and raise a helpful error
                    raise TypeError(
                        f"Unable to construct SAE from {sae_class}."
                        " Expected one of: class.from_config(config, device),"
                        " (config, device), (input_dim=.., n_features=.., k=..),"
                        " or positional (d_input, n_boj[, k])."
                    ) from e

        # ensure model lives on trainer device
        self.sae = self.sae.to(self.device)
        # store class/kwargs for introspection
        self.sae_class = sae_class
        self.sae_kwargs = sae_kwargs

    def train(
        self,
        activations: Tensor,
        batch_size: int = 512,
        epochs: int = 100,
        lr: float = 1e-3,
        scheduler: Optional[Any] = None,
        progress: bool = True,
    ) -> SAETrainingResult:
        """Train the SAE on activations.

        Args:
            activations: Training activations [N, d_input].
            batch_size: Batch size.
            epochs: Number of training epochs.
            lr: Learning rate.
            scheduler: Optional learning rate scheduler.
            progress: Show progress bar.

        Returns:
            SAETrainingResult with trained SAE and metrics.
        """
        self.sae.train()

        optimizer = torch.optim.Adam(self.sae.parameters(), lr=lr)
        if scheduler is not None:
            scheduler = scheduler(optimizer)

        activations = activations.to(self.device)
        n_samples = len(activations)
        n_batches = (n_samples + batch_size - 1) // batch_size

        losses = []
        l0_values = []
        reconstruction_losses = []
        l1_losses = []

        iterator = tqdm(range(epochs), desc="Training SAE") if progress and tqdm else range(epochs)

        for epoch in iterator:
            epoch_loss = 0.0
            epoch_l0 = 0.0
            epoch_recon = 0.0
            epoch_l1 = 0.0

            indices = torch.randperm(n_samples, device=self.device)

            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, n_samples)
                batch = activations[indices[start:end]]

                optimizer.zero_grad()

                out = self.sae(batch)

                # support SAEs that return (recon, sparse_h, mask) or (recon, sparse_h)
                if isinstance(out, (tuple, list)):
                    if len(out) == 3:
                        recon, sparse_h, mask = out
                    elif len(out) == 2:
                        recon, sparse_h = out
                        mask = None
                    else:
                        raise RuntimeError("Unexpected SAE forward output shape")
                else:
                    # single tensor returned (assume reconstruction) - not supported
                    raise RuntimeError(
                        "SAE forward must return (recon, sparse_h) or (recon, sparse_h, mask)"
                    )

                recon_loss = F.mse_loss(recon, batch)
                # normalize sparsity penalty by taking mean over batch+features
                l1_loss = self.config.l1_coefficient * sparse_h.abs().mean()
                loss = recon_loss + l1_loss

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                # compute l0 using available API or approximate
                if hasattr(self.sae, "compute_l0"):
                    try:
                        epoch_l0 += self.sae.compute_l0(sparse_h).item()
                    except Exception:
                        epoch_l0 += (sparse_h.abs() > 1e-8).float().sum(dim=-1).mean().item()
                else:
                    epoch_l0 += (sparse_h.abs() > 1e-8).float().sum(dim=-1).mean().item()
                epoch_recon += recon_loss.item()
                epoch_l1 += l1_loss.item()

            if scheduler is not None:
                scheduler.step()

            n_batches = max(n_batches, 1)
            losses.append(epoch_loss / n_batches)
            l0_values.append(epoch_l0 / n_batches)
            reconstruction_losses.append(epoch_recon / n_batches)
            l1_losses.append(epoch_l1 / n_batches)

            if progress and tqdm:
                iterator.set_postfix(
                    {
                        "loss": f"{losses[-1]:.4f}",
                        "l0": f"{l0_values[-1]:.1f}",
                        "recon": f"{reconstruction_losses[-1]:.4f}",
                    }
                )

        self.sae.eval()

        return SAETrainingResult(
            sae=self.sae,
            losses=losses,
            l0_values=l0_values,
            reconstruction_losses=reconstruction_losses,
            l1_losses=l1_losses,
            final_l0=l0_values[-1] if l0_values else 0.0,
            final_reconstruction_loss=reconstruction_losses[-1] if reconstruction_losses else 0.0,
            epochs=epochs,
            device=self.device,
        )

    def encode(self, activations: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode activations using trained SAE.

        Args:
            activations: Input activations.

        Returns:
            Tuple of (sparse latents, mask).
        """
        self.sae.eval()
        with torch.no_grad():
            activations = activations.to(self.device)
            return self.sae.encode(activations)

    def decode(self, sparse_latents: Tensor) -> Tensor:
        """Decode sparse latents to reconstructions.

        Args:
            sparse_latents: Sparse latents.

        Returns:
            Reconstructed activations.
        """
        self.sae.eval()
        with torch.no_grad():
            sparse_latents = sparse_latents.to(self.device)
            return self.sae.decode(sparse_latents)


class SAESweeper:
    """Hyperparameter sweeper for SAE training."""

    def __init__(
        self,
        d_input: int,
        k: int,
        l1_coefficient: float = 1e-3,
        device: Optional[torch.device] = None,
    ):
        """Initialize SAESweeper.

        Args:
            d_input: Input dimension.
            k: Top-k sparsity parameter.
            l1_coefficient: L1 regularization strength.
            device: Device for training.
        """
        self.d_input = d_input
        self.k = k
        self.l1_coefficient = l1_coefficient
        self.device = device or _get_device()

    def sweep(
        self,
        activations: Tensor,
        n_boj_values: List[int],
        batch_size: int = 512,
        epochs: int = 50,
        lr: float = 1e-3,
    ) -> Dict[int, SAETrainingResult]:
        """Sweep over number of bottleneck neurons.

        Args:
            activations: Training activations.
            n_boj_values: List of n_boj values to try.
            batch_size: Batch size.
            epochs: Training epochs.
            lr: Learning rate.

        Returns:
            Dictionary mapping n_boj to training results.
        """
        results = {}

        for n_boj in n_boj_values:
            trainer = SAETrainer(
                d_input=self.d_input,
                n_boj=n_boj,
                k=self.k,
                l1_coefficient=self.l1_coefficient,
                device=self.device,
            )

            result = trainer.train(
                activations=activations,
                batch_size=batch_size,
                epochs=epochs,
                lr=lr,
                progress=False,
            )

            results[n_boj] = result

        return results


def compute_dead_features(sae: SparseAutoencoder, test_activations: Tensor) -> float:
    """Compute fraction of dead features.

    Args:
        sae: Trained SparseAutoencoder.
        test_activations: Test activations.

    Returns:
        Fraction of dead features (never activated).
    """
    sae.eval()
    with torch.no_grad():
        test_activations = test_activations.to(sae.device)
        _, sparse_h, _ = sae(test_activations)
        activation_rate = (sparse_h.abs() > 1e-8).float().mean(dim=0)
        dead_fraction = (activation_rate < 0.01).float().mean().item()
        return dead_fraction


def compute_feature_uncertainty(
    sae: SparseAutoencoder, activations: Tensor, n_samples: int = 10
) -> Tensor:
    """Compute uncertainty in feature activations via dropout sampling.

    Args:
        sae: Trained SparseAutoencoder.
        activations: Input activations.
        n_samples: Number of dropout samples.

    Returns:
        Uncertainty scores per feature.
    """
    sae.eval()

    if not hasattr(sae, "decoder"):
        return torch.zeros(sae.config.n_boj)

    original_decoder = sae.decoder_weight.data.clone()

    uncertainties = []

    with torch.no_grad():
        for _ in range(n_samples):
            noise = torch.randn_like(sae.decoder_weight) * 0.1
            sae.decoder_weight.data = original_decoder + noise

            _, sparse_h, _ = sae(activations)
            uncertainties.append(sparse_h.abs())

    sae.decoder_weight.data = original_decoder

    uncertainties = torch.stack(uncertainties)
    uncertainty = uncertainties.var(dim=0).mean(dim=0)

    return uncertainty
