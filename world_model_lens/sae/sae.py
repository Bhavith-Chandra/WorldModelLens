"""Sparse Autoencoder (SAE) for mechanistic interpretability.

SAEs decompose polysemantic latent spaces into monosemantic features.
This is the gold standard in LLM interpretability, now adapted for world models.

Features:
- Standard SAE with top-k activation
- Gated SAE for better feature learning
- JumpReLU for improved sparsity
- Feature attribution analysis
"""

from typing import Any, Dict, List, Optional, Tuple, Callable
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import numpy as np


@dataclass
class SAEResult:
    """Result of SAE encoding."""

    latent_features: torch.Tensor  # [B, n_features] sparse activations
    feature_indices: torch.Tensor  # Indices of active features
    reconstruction: torch.Tensor  # Reconstructed input
    loss: float
    sparsity_loss: float
    reconstruction_loss: float


class SAEBase(nn.Module, ABC):
    """Standardized SAE interface for trainer compatibility.

    Implementations should provide `from_config(config, device)` to allow the
    trainer to construct models in a single, standard way. Forward should
    return `(reconstruction, sparse_h)` or `(reconstruction, sparse_h, mask)`.
    """

    @classmethod
    @abstractmethod
    def from_config(cls, config, device: Optional[torch.device] = None):
        raise NotImplementedError

    @abstractmethod
    def encode(
        self, x: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, h: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def compute_l0(self, h: torch.Tensor) -> torch.Tensor:
        """Default L0 approximation (can be overridden)."""
        return (h.abs() > 1e-8).float().sum(dim=-1).mean()

    def forward(self, x: torch.Tensor, *args, **kwargs):
        out = self.encode(x, *args, **kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            h, mask = out
        elif isinstance(out, tuple) and len(out) == 1:
            h = out[0]
            mask = None
        else:
            # if encode returns unexpected shape, assume encode already returned h
            h = out
            mask = None

        recon = self.decode(h)
        if mask is None:
            return recon, h
        return recon, h, mask


class SparseAutoencoder(SAEBase):
    """Sparse Autoencoder for decomposing latent spaces.

    Learns a dictionary of features that can be activated sparsely.

    Example:
        sae = SparseAutoencoder(input_dim=512, n_features=4096)

        features, recon = sae.encode(x)
        loss = sae.compute_loss(x)
    """

    def __init__(
        self,
        input_dim: int,
        n_features: int = 4096,
        hidden_dim: Optional[int] = None,
        activation: str = "relu",
        tie_weights: bool = False,
        dead_threshold: float = 1e-6,
    ):
        """Initialize SAE.

        Args:
            input_dim: Input dimension
            n_features: Number of dictionary features
            hidden_dim: Hidden dimension (default: 2 * input_dim)
            activation: Activation function
            tie_weights: Tie encoder/decoder weights
            dead_threshold: Threshold for dead feature detection
        """
        super().__init__()

        self.input_dim = input_dim
        self.n_features = n_features
        self.hidden_dim = hidden_dim or input_dim * 2
        self.dead_threshold = dead_threshold

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Linear(self.hidden_dim, n_features),
        )

        if tie_weights:
            self.decoder = nn.Linear(n_features, input_dim, bias=False)
            self.decoder.weight = nn.Parameter(self.encoder[0].weight.T.clone())
        else:
            self.decoder = nn.Linear(n_features, input_dim)

        self.encoder[2].weight.data *= 0.1
        self.decoder.weight.data *= 0.1

        self.bottleneck_scale = nn.Parameter(torch.ones(n_features))
        self.bottleneck_bias = nn.Parameter(torch.zeros(n_features))

    @classmethod
    def from_config(cls, config, device: Optional[torch.device] = None):
        """Construct from trainer SAEConfig-like object.

        This provides a standardized constructor used by SAETrainer.
        """
        inst = cls(
            input_dim=config.d_input,
            n_features=config.n_boj,
            hidden_dim=None,
            tie_weights=config.tied_weights,
        )
        if device is not None:
            inst.to(device)
        return inst

    def encode(
        self,
        x: torch.Tensor,
        k: Optional[int] = None,
        threshold: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to sparse features.

        Args:
            x: Input tensor [B, input_dim]
            k: Top-k features to keep (default: all above threshold)
            threshold: Minimum activation threshold

        Returns:
            Tuple of (sparse_features [B, n_features], feature_indices)
        """
        h = self.encoder(x)
        h = h * self.bottleneck_scale + self.bottleneck_bias

        if k is not None:
            topk_vals, topk_idx = torch.topk(h, k, dim=-1)
            mask = torch.zeros_like(h).scatter_(-1, topk_idx, topk_vals)
            h = F.relu(mask)
        else:
            h = F.relu(h - threshold)

        h = F.relu(h)

        return h, h.nonzero(as_tuple=True)[1]

    def decode(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Decode features to reconstruction.

        Args:
            features: Sparse features [B, n_features]

        Returns:
            Reconstructed input [B, input_dim]
        """
        return self.decoder(features)

    def forward(
        self,
        x: torch.Tensor,
        k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor
            k: Top-k features

        Returns:
            Tuple of (reconstruction, features)
        """
        features, indices = self.encode(x, k)
        reconstruction = self.decode(features)
        return reconstruction, features

    def compute_loss(
        self,
        x: torch.Tensor,
        k: Optional[int] = None,
        l1_weight: float = 1e-3,
        death_penalty: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute SAE loss.

        Args:
            x: Input tensor
            k: Top-k features
            l1_weight: L1 sparsity weight
            death_penalty: Penalty for dead features

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        reconstruction, features = self.forward(x, k)

        recon_loss = F.mse_loss(reconstruction, x)

        l1_loss = features.abs().mean()

        death_rate = (features.sum(dim=0) < self.dead_threshold).float().mean()
        death_loss = death_rate * death_penalty

        total_loss = recon_loss + l1_weight * l1_loss + death_loss

        return total_loss, {
            "reconstruction": recon_loss.item(),
            "sparsity": l1_loss.item(),
            "death_rate": death_rate.item(),
            "total": total_loss.item(),
        }

    def get_active_features(
        self,
        x: torch.Tensor,
        k: int = 50,
    ) -> Dict[int, float]:
        """Get most active features for input.

        Args:
            x: Input tensor
            k: Number of top features

        Returns:
            Dict mapping feature index to activation value
        """
        features, indices = self.encode(x, k=k)

        result = {}
        for idx, val in zip(indices[0].tolist(), features[0, indices[0]].tolist()):
            result[idx] = val

        return result

    def compute_feature_importance(
        self,
        dataloader: torch.utils.data.DataLoader,
        n_samples: int = 1000,
    ) -> np.ndarray:
        """Compute feature importance scores.

        Args:
            dataloader: DataLoader with inputs
            n_samples: Number of samples

        Returns:
            Array of importance scores [n_features]
        """
        self.eval()
        total_activation = torch.zeros(self.n_features, device=self.device)
        total_count = 0

        with torch.no_grad():
            for i, (x,) in enumerate(dataloader):
                if i >= n_samples:
                    break

                features, _ = self.encode(x.to(self.device))
                total_activation += features.sum(dim=0).cpu()
                total_count += x.shape[0]

        return (total_activation / total_count).numpy()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


class GatedSparseAutoencoder(SAEBase):
    """Gated SAE with better dead feature handling.

    Uses a gating mechanism to better handle feature activation.
    """

    def __init__(
        self,
        input_dim: int,
        n_features: int = 4096,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.n_features = n_features
        self.hidden_dim = hidden_dim or input_dim * 2

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, n_features * 2),
        )

        self.decoder = nn.Linear(n_features, input_dim)

        nn.init.xavier_uniform_(self.encoder[0].weight)
        nn.init.xavier_uniform_(self.decoder.weight)

    @classmethod
    def from_config(cls, config, device: Optional[torch.device] = None):
        inst = cls(input_dim=config.d_input, n_features=config.n_boj)
        if device is not None:
            inst.to(device)
        return inst

    def compute_l0(self, h: torch.Tensor) -> torch.Tensor:
        """Approximate L0 from activations for compatibility."""
        return (h.abs() > 1e-8).float().sum(dim=-1).mean()

    def encode(
        self,
        x: torch.Tensor,
        k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode with gating."""
        h = self.encoder(x)

        gate, value = h.chunk(2, dim=-1)
        gate = F.sigmoid(gate)
        value = F.relu(value)

        features = gate * value

        if k is not None:
            topk_vals, topk_idx = torch.topk(features, k, dim=-1)
            mask = torch.zeros_like(features).scatter_(-1, topk_idx, topk_vals)
            features = mask

        return features, features.nonzero(as_tuple=True)[1]

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features)

    def forward(self, x: torch.Tensor, k: Optional[int] = None):
        features, indices = self.encode(x, k)
        reconstruction = self.decode(features)
        return reconstruction, features


class SAELayer:
    """SAE integration layer for world model hooks.

    Wraps a world model with SAE analysis.

    Example:
        sae_layer = SAELayer(world_model, input_dim=512, n_features=4096)

        traj, cache = sae_layer.run_with_cache(observations)

        # Analyze features
        feature_activity = sae_layer.get_feature_activations(cache)
        important_features = sae_layer.find_important_features('collision', cache)
    """

    def __init__(
        self,
        world_model: Any,
        input_dim: int,
        n_features: int = 4096,
        layer_name: str = "z",
        device: str = "cuda",
    ):
        """Initialize SAE layer.

        Args:
            world_model: HookedWorldModel instance
            input_dim: Dimension of activations
            n_features: Number of SAE features
            layer_name: Layer to extract for SAE
            device: Device
        """
        self.wm = world_model
        self.layer_name = layer_name
        self.device = device

        self.sae = SparseAutoencoder(
            input_dim=input_dim,
            n_features=n_features,
        ).to(device)

        self._feature_cache = {}

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encode through SAE."""
        x = x.to(self.device)
        features, _ = self.sae.encode(x, **kwargs)
        return features

    def train_sae(
        self,
        dataloader: torch.utils.data.DataLoader,
        l1_weight: float = 1e-3,
        n_epochs: int = 10,
        lr: float = 1e-4,
    ) -> List[float]:
        """Train SAE on activations.

        Args:
            dataloader: DataLoader with activation tensors
            l1_weight: L1 sparsity weight
            n_epochs: Training epochs
            lr: Learning rate

        Returns:
            List of losses per epoch
        """
        optimizer = torch.optim.Adam(self.sae.parameters(), lr=lr)

        losses = []

        for epoch in range(n_epochs):
            epoch_loss = 0.0

            for batch in dataloader:
                x = (
                    batch[0].to(self.device)
                    if isinstance(batch, (list, tuple))
                    else batch.to(self.device)
                )

                loss, loss_dict = self.sae.compute_loss(x, l1_weight=l1_weight)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss_dict["total"]

            losses.append(epoch_loss / len(dataloader))

        return losses

    def get_feature_activations(
        self,
        cache: Any,
        k: Optional[int] = None,
    ) -> Dict[int, torch.Tensor]:
        """Get feature activations from cache.

        Args:
            cache: ActivationCache
            k: Top-k features

        Returns:
            Dict mapping timestep to feature activations
        """
        activations = {}

        for t in cache.timesteps:
            try:
                act = cache[self.layer_name, t]
                if act.dim() > 1:
                    act = act.flatten(1)

                features, _ = self.sae.encode(act, k=k)
                activations[t] = features.cpu()

            except (KeyError, Exception):
                continue

        return activations

    def find_concept_features(
        self,
        concept_name: str,
        positive_examples: torch.Tensor,
        negative_examples: torch.Tensor,
        threshold: float = 0.5,
    ) -> List[int]:
        """Find features that discriminate a concept.

        Args:
            concept_name: Name for concept
            positive_examples: Positive samples
            negative_examples: Negative samples
            threshold: Difference threshold

        Returns:
            List of feature indices
        """
        self.sae.eval()

        with torch.no_grad():
            pos_features, _ = self.sae.encode(positive_examples.to(self.device))
            neg_features, _ = self.sae.encode(negative_examples.to(self.device))

            pos_mean = pos_features.mean(dim=0)
            neg_mean = neg_features.mean(dim=0)

            diff = pos_mean - neg_mean

            important = (diff > threshold).nonzero(as_tuple=True)[0]

        return important.tolist()

    def run_with_cache(
        self,
        observations: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Any, Any]:
        """Run world model with SAE analysis.

        Args:
            observations: Input observations
            actions: Optional actions
            **kwargs: Additional args

        Returns:
            Tuple of (trajectory, cache)
        """
        _, traj, cache = self.wm.run_with_cache(observations, actions)

        feature_activations = self.get_feature_activations(cache)
        self._feature_cache = feature_activations

        return traj, cache


# New feature: compact TopK ReLU Sparse Autoencoder class.
class TopKSparseAutoencoder(SAEBase):
    """Top-k ReLU Sparse Autoencoder.

    Encoder: Linear(input_dim -> n_features)
    Bottleneck: keep top-k activations per example, ReLU
    Decoder: Linear(n_features -> input_dim)

    compute_loss returns reconstruction MSE + l1 sparsity penalty.
    """

    def __init__(self, input_dim: int, n_features: int, k: int = 1, tie_weights: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.n_features = n_features
        self.k = int(k)

        # initialize encoder weight first
        self.encoder = nn.Linear(input_dim, n_features, bias=True)
        nn.init.xavier_uniform_(self.encoder.weight)

        if tie_weights:
            # decoder uses a copied transpose of encoder weight at init
            self.decoder = nn.Linear(n_features, input_dim, bias=False)
            with torch.no_grad():
                self.decoder.weight.copy_(self.encoder.weight.t())
        else:
            self.decoder = nn.Linear(n_features, input_dim, bias=True)
            nn.init.xavier_uniform_(self.decoder.weight)

    def topk_relu(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.k <= 0 or self.k >= h.shape[-1]:
            out = F.relu(h)
            mask = (out > 0).float()
            return out, mask

        vals, idx = torch.topk(h, self.k, dim=-1)
        mask = torch.zeros_like(h).scatter_(-1, idx, 1.0)
        out = h * mask
        out = F.relu(out)
        return out, mask

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.topk_relu(h)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, mask = self.encode(x)
        recon = self.decode(h)
        return recon, h, mask

    def compute_loss(
        self, x: torch.Tensor, l1_weight: float = 1e-3
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        recon, h, mask = self.forward(x)
        recon_loss = F.mse_loss(recon, x)
        sparsity = h.abs().mean()
        total = recon_loss + l1_weight * sparsity
        return total, {
            "reconstruction": float(recon_loss.item()),
            "sparsity": float(sparsity.item()),
            "total": float(total.item()),
        }

    @classmethod
    def from_config(cls, config, device: Optional[torch.device] = None):
        inst = cls(
            input_dim=config.d_input,
            n_features=config.n_boj,
            k=config.k,
            tie_weights=config.tied_weights,
        )
        if device is not None:
            inst.to(device)
        return inst

    def compute_l0(self, h: torch.Tensor) -> torch.Tensor:
        return (h.abs() > 1e-8).float().sum(dim=-1).mean()
