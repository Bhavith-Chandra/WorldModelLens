"""Latent state representation for a single timestep."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class LatentState:
    """Represents the belief state at a single timestep.

    Attributes:
        h_t: Recurrent hidden state tensor of shape [d_h].
        z_posterior: Categorical posterior distribution [n_cat, n_cls].
        z_prior: Categorical prior distribution [n_cat, n_cls].
        timestep: Current timestep index.
        action: Action taken at this timestep.
        reward_pred: Predicted reward distribution.
        reward_real: Actual reward received.
        cont_pred: Predicted continuation/probability of episode continuing.
        value_pred: Predicted value estimate.
        actor_logits: Policy logits.
        obs_encoding: Observation encoding from encoder.
        metadata: Additional arbitrary data.
        multimodal_channels: Dict mapping channel name to tensor for multimodal models.
            Supports 'vision', 'proprio', 'language', 'audio', and custom channels.
    """

    h_t: torch.Tensor
    z_posterior: torch.Tensor
    z_prior: torch.Tensor
    timestep: int = 0
    action: torch.Tensor | None = None
    reward_pred: torch.Tensor | None = None
    reward_real: torch.Tensor | None = None
    cont_pred: torch.Tensor | None = None
    value_pred: torch.Tensor | None = None
    actor_logits: torch.Tensor | None = None
    obs_encoding: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    multimodal_channels: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def flat(self) -> torch.Tensor:
        """Concatenated flattened representation [h_t + z_flat].

        The z_posterior is flattened to [n_cat * n_cls] via one-hot expansion.
        """
        z_onehot = self.z_posterior.flatten()
        return torch.cat([self.h_t, z_onehot])

    @property
    def kl(self) -> torch.Tensor:
        """Numerically stable KL divergence between posterior and prior.

        Returns:
            Scalar tensor: sum of KL over all categorical variables.
        """
        p = self.z_posterior.clamp(min=1e-8)
        q = self.z_prior.clamp(min=1e-8)
        p = p / p.sum(dim=-1, keepdim=True)
        q = q / q.sum(dim=-1, keepdim=True)
        kl = p * (p.log() - q.log())
        return kl.sum()

    @property
    def surprise(self) -> torch.Tensor:
        """Alias for kl (KL divergence = surprise = information gain)."""
        return self.kl

    @property
    def z_flat(self) -> torch.Tensor:
        """Flattened z_posterior tensor [n_cat * n_cls]."""
        return self.z_posterior.flatten()

    @property
    def z_indices(self) -> torch.Tensor:
        """Argmax indices of z_posterior [n_cat]."""
        return self.z_posterior.argmax(dim=-1)

    @property
    def reward(self) -> torch.Tensor | None:
        """Backward-compatible alias for the actual reward received (reward_real)."""
        return self.reward_real

    def to_device(self, device: torch.device) -> "LatentState":
        """Move all tensors to the specified device.

        Args:
            device: Target torch device.

        Returns:
            New LatentState on target device.
        """
        import copy

        new_state = copy.copy(self)
        for attr in [
            "h_t",
            "z_posterior",
            "z_prior",
            "action",
            "reward_pred",
            "reward_real",
            "cont_pred",
            "value_pred",
            "actor_logits",
            "obs_encoding",
        ]:
            val = getattr(self, attr)
            if isinstance(val, torch.Tensor):
                setattr(new_state, attr, val.to(device))
        new_state.multimodal_channels = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in self.multimodal_channels.items()
        }
        return new_state

    def detach(self) -> "LatentState":
        """Detach all tensors from computation graph."""
        import copy

        new_state = copy.copy(self)
        for attr in [
            "h_t",
            "z_posterior",
            "z_prior",
            "action",
            "reward_pred",
            "reward_real",
            "cont_pred",
            "value_pred",
            "actor_logits",
            "obs_encoding",
        ]:
            val = getattr(self, attr)
            if isinstance(val, torch.Tensor):
                setattr(new_state, attr, val.detach())
        new_state.multimodal_channels = {
            k: v.detach() if isinstance(v, torch.Tensor) else v
            for k, v in self.multimodal_channels.items()
        }
        return new_state

    def get_multimodal_channel(self, channel_name: str) -> torch.Tensor | None:
        """Get a specific multimodal channel tensor.

        Args:
            channel_name: Name of channel ('vision', 'proprio', 'language', etc.).

        Returns:
            Tensor for the channel, or None if not present.
        """
        return self.multimodal_channels.get(channel_name)

    def has_channel(self, channel_name: str) -> bool:
        """Check if a multimodal channel exists."""
        return channel_name in self.multimodal_channels

    @property
    def channel_names(self) -> list[str]:
        """List of available multimodal channel names."""
        return list(self.multimodal_channels.keys())
