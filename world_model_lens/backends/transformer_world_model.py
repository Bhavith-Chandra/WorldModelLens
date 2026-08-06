"""Simple Transformer World Model.

A minimal transformer-based world model for benchmarking.
"""

from typing import Any, Dict, Optional, Set
import torch
import torch.nn as nn
import math


class TransformerWorldModelAdapter(nn.Module):
    """Simple transformer-based world model.

    Architecture:
    - Transformer encoder for observations
    - Causal transformer decoder for dynamics
    - Optional prediction heads

    This is a reference implementation for benchmarking.
    """

    def __init__(
        self,
        d_obs: int = 64,
        d_action: int = 4,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 512,
        max_seq_len: int = 100,
    ):
        """Initialize transformer world model.

        Args:
            d_obs: Observation dimension
            d_action: Action dimension
            d_model: Transformer model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feedforward dimension
            max_seq_len: Maximum sequence length
        """
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        # Observation embedding
        self.obs_embedding = nn.Linear(d_obs, d_model)

        # Action embedding
        self.action_embedding = nn.Linear(d_action, d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.1)

        # Encoder (for observation processing)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Decoder (for dynamics prediction)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Latent projection
        self.latent_proj = nn.Linear(d_model, d_model)

        # Prediction heads
        self.reward_head = nn.Linear(d_model, 1)
        self.value_head = nn.Linear(d_model, 1)

    def encode(
        self,
        obs: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode observation to latent.

        Args:
            obs: Observation [B, d_obs] or [B, T, d_obs]
            hidden: Hidden state (ignored, for API compatibility)

        Returns:
            Tuple of (latent, encoding)
        """
        # Add batch dimension if needed
        squeeze = False
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            squeeze = True

        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # [B, 1, d_obs]

        # Embed
        x = self.obs_embedding(obs)

        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]

        # Encode
        encoded = self.encoder(x)

        # Project to latent
        latent = self.latent_proj(encoded)

        if squeeze:
            latent = latent.squeeze(1)
            encoded = encoded.squeeze(1)

        return latent, latent

    def dynamics(
        self,
        hidden: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute prior/dynamics.

        Args:
            hidden: Hidden state [B, d_model]
            action: Action (optional)

        Returns:
            Prior latent
        """
        if hidden.dim() == 1:
            hidden = hidden.unsqueeze(0)

        # Add action if provided
        if action is not None:
            if action.dim() == 1:
                action = action.unsqueeze(0)
            action_emb = self.action_embedding(action)
            hidden = hidden + action_emb

        # Use as decoder query
        query = hidden.unsqueeze(1)  # [B, 1, d_model]

        # Memory is just the hidden state repeated
        memory = hidden.unsqueeze(1)

        # Decode
        decoded = self.decoder(query, memory)

        # Project
        prior = self.latent_proj(decoded)

        return prior.squeeze(1)

    def transition(
        self,
        hidden: torch.Tensor,
        latent: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Transition to next hidden state.

        Args:
            hidden: Current hidden state
            latent: Current latent
            action: Action (optional)

        Returns:
            Next hidden state
        """
        if hidden.dim() == 1:
            hidden = hidden.unsqueeze(0)
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)

        # Combine hidden and latent
        combined = hidden + latent

        # Add action
        if action is not None:
            if action.dim() == 1:
                action = action.unsqueeze(0)
            action_emb = self.action_embedding(action)
            combined = combined + action_emb

        # Simple MLP transition
        return combined

    def decode(
        self,
        hidden: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        """Decode latent to observation reconstruction.

        Args:
            hidden: Hidden state
            latent: Latent state

        Returns:
            Reconstructed observation
        """
        if hidden.dim() == 1:
            hidden = hidden.unsqueeze(0)
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)

        combined = hidden + latent

        return combined

    def predict_reward(
        self,
        hidden: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        """Predict reward.

        Args:
            hidden: Hidden state
            latent: Latent state

        Returns:
            Predicted reward
        """
        if hidden.dim() == 1:
            hidden = hidden.unsqueeze(0)
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)

        combined = hidden + latent
        return self.reward_head(combined)

    def predict_value(
        self,
        hidden: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        """Predict value.

        Args:
            hidden: Hidden state
            latent: Latent state

        Returns:
            Predicted value
        """
        if hidden.dim() == 1:
            hidden = hidden.unsqueeze(0)
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)

        combined = hidden + latent
        return self.value_head(combined)

    def initial_state(self, batch_size: int = 1, device: torch.device = None) -> torch.Tensor:
        """Get initial hidden state.

        Args:
            batch_size: Batch size
            device: Device

        Returns:
            Initial hidden state
        """
        if device is None:
            device = torch.device("cpu")

        return torch.zeros(batch_size, self.d_model, device=device)

    def sample_z(
        self, logits: torch.Tensor, temperature: float = 1.0, sample: bool = True
    ) -> torch.Tensor:
        """Sample from latent distribution.

        Args:
            logits: Latent logits
            temperature: Sampling temperature
            sample: Whether to sample

        Returns:
            Sampled latent
        """
        if temperature == 0 or not sample:
            return logits

        # Add noise
        noise = torch.randn_like(logits) * temperature
        return logits + noise


def create_transformer_world_model(
    obs_dim: int = 64,
    action_dim: int = 4,
    latent_dim: int = 128,
    **kwargs,
) -> TransformerWorldModelAdapter:
    """Create a transformer world model with sensible defaults.

    Args:
        obs_dim: Observation dimension
        action_dim: Action dimension
        latent_dim: Latent dimension
        **kwargs: Additional arguments

    Returns:
        TransformerWorldModelAdapter
    """
    return TransformerWorldModelAdapter(
        d_obs=obs_dim,
        d_action=action_dim,
        d_model=latent_dim,
        **kwargs,
    )
