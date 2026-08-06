"""RSSM (Recurrent State Space Model) Adapter.

RSSM is the core of Dreamer models. This is a standalone implementation
for benchmarking and comparison.
"""

from typing import Any, Dict, Optional, Set
import torch
import torch.nn as nn
import torch.nn.functional as F


class RSSMAdapter(nn.Module):
    """Recurrent State Space Model (RSSM).

    RSSM consists of:
    - Deterministic hidden state (h_t)
    - Stochastic latent state (z_t)
    - Recurrent transition: h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
    - Posterior: z_t ~ q(z_t | h_t, x_t)
    - Prior: p(z_t | h_t)

    This is the core of Dreamer-style world models.
    """

    def __init__(
        self,
        d_obs: int = 64,
        d_action: int = 4,
        d_embed: int = 256,
        d_hidden: int = 256,
        d_latent: int = 32,
        d_recurrent: int = 256,
        n_classes: int = 32,
    ):
        """Initialize RSSM.

        Args:
            d_obs: Observation dimension
            d_action: Action dimension
            d_embed: Embedding dimension
            d_hidden: Hidden dimension
            d_latent: Latent dimension
            d_recurrent: Recurrent (GRU) dimension
            n_classes: Number of classes for discrete latent
        """
        super().__init__()

        self.d_obs = d_obs
        self.d_action = d_action
        self.d_embed = d_embed
        self.d_hidden = d_hidden
        self.d_latent = d_latent
        self.d_recurrent = d_recurrent
        self.n_classes = n_classes

        # Embedding network
        self.encoder = nn.Sequential(
            nn.Linear(d_obs, d_embed),
            nn.ReLU(),
            nn.Linear(d_embed, d_embed),
        )

        # Recurrent transition (GRU)
        self.gru = nn.GRUCell(d_embed + d_action + d_latent, d_recurrent)

        # Prior (determines what the model expects)
        self.prior_net = nn.Sequential(
            nn.Linear(d_recurrent, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_latent * n_classes),
        )

        # Posterior (what the model actually sees)
        self.posterior_net = nn.Sequential(
            nn.Linear(d_recurrent + d_embed, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_latent * n_classes),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(d_recurrent + d_latent, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_embed),
        )

        # Reward head
        self.reward_head = nn.Sequential(
            nn.Linear(d_recurrent + d_latent, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(d_recurrent + d_latent, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        hidden: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            obs: Observation [B, d_obs]
            action: Action [B, d_action]
            hidden: Hidden state [B, d_recurrent]

        Returns:
            Dict with keys: posterior, prior, hidden, reward, value
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if action is not None and action.dim() == 1:
            action = action.unsqueeze(0)

        batch_size = obs.shape[0]

        # Initialize hidden
        if hidden is None:
            hidden = torch.zeros(batch_size, self.d_recurrent, device=obs.device)

        # Embed observation
        embed = self.encoder(obs)

        # Get previous latent (z_{t-1}), zero for first step
        if not hasattr(self, "_prev_z"):
            self._prev_z = torch.zeros(batch_size, self.d_latent, device=obs.device)

        # GRU input: embed + action + previous latent
        gru_input = (
            torch.cat([embed, self._prev_z, action], dim=-1)
            if action is not None
            else torch.cat([embed, self._prev_z], dim=-1)
        )

        # Update hidden
        hidden = self.gru(gru_input, hidden)

        # Prior: what the model expects
        prior_logits = self.prior_net(hidden)
        prior_dist = torch.distributions.OneHotCategorical(
            logits=prior_logits.view(-1, self.n_classes)
        )
        prior = prior_dist.sample().view(batch_size, -1)[:, : self.d_latent]
        prior_mean = prior_dist.probs.view(-1, self.n_classes)[:, : self.d_latent]

        # Posterior: what the model sees
        posterior_input = torch.cat([hidden, embed], dim=-1)
        posterior_logits = self.posterior_net(posterior_input)
        posterior_dist = torch.distributions.OneHotCategorical(
            logits=posterior_logits.view(-1, self.n_classes)
        )
        posterior = posterior_dist.sample().view(batch_size, -1)[:, : self.d_latent]
        posterior_mean = posterior_dist.probs.view(-1, self.n_classes)[:, : self.d_latent]

        # Store for next step
        self._prev_z = posterior

        # Decode
        decoder_input = torch.cat([hidden, posterior], dim=-1)
        reconstruction = self.decoder(decoder_input)

        # Reward and value
        reward = self.reward_head(decoder_input)
        value = self.value_head(decoder_input)

        return {
            "posterior": posterior,
            "posterior_mean": posterior_mean,
            "prior": prior,
            "prior_mean": prior_mean,
            "hidden": hidden,
            "embedding": embed,
            "reconstruction": reconstruction,
            "reward": reward,
            "value": value,
        }

    def encode(
        self,
        obs: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode observation to latent.

        Args:
            obs: Observation
            hidden: Hidden state (optional)

        Returns:
            Tuple of (latent, embedding)
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        result = self.forward(obs, hidden=hidden)

        return result["posterior"], result["embedding"]

    def dynamics(
        self,
        hidden: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute prior (dynamics model).

        Args:
            hidden: Hidden state
            action: Action (optional)

        Returns:
            Prior latent
        """
        if hidden.dim() == 1:
            hidden = hidden.unsqueeze(0)

        # Get prior
        prior_logits = self.prior_net(hidden)
        prior_dist = torch.distributions.OneHotCategorical(
            logits=prior_logits.view(-1, self.n_classes)
        )
        prior = prior_dist.sample().view(hidden.size(0), -1)[:, : self.d_latent]

        return prior

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

        # Combine
        if action is not None:
            if action.dim() == 1:
                action = action.unsqueeze(0)
            gru_input = torch.cat([latent, action], dim=-1)
        else:
            gru_input = latent

        # Update hidden
        next_hidden = self.gru(gru_input, hidden)

        return next_hidden

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
            Reconstructed observation embedding
        """
        if hidden.dim() == 1:
            hidden = hidden.unsqueeze(0)
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)

        decoder_input = torch.cat([hidden, latent], dim=-1)
        reconstruction = self.decoder(decoder_input)

        return reconstruction

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

        decoder_input = torch.cat([hidden, latent], dim=-1)
        return self.reward_head(decoder_input)

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

        decoder_input = torch.cat([hidden, latent], dim=-1)
        return self.value_head(decoder_input)

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

        return torch.zeros(batch_size, self.d_recurrent, device=device)

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
            # Return mean
            probs = F.softmax(logits, dim=-1)
            return probs

        # Gumbel-softmax sampling
        gumbel = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
        sampled = F.softmax((logits + gumbel) / temperature, dim=-1)

        return sampled


def create_rssm(
    obs_dim: int = 64,
    action_dim: int = 4,
    latent_dim: int = 32,
    hidden_dim: int = 256,
    **kwargs,
) -> RSSMAdapter:
    """Create RSSM with sensible defaults.

    Args:
        obs_dim: Observation dimension
        action_dim: Action dimension
        latent_dim: Latent dimension
        hidden_dim: Hidden dimension
        **kwargs: Additional arguments

    Returns:
        RSSMAdapter
    """
    return RSSMAdapter(
        d_obs=obs_dim,
        d_action=action_dim,
        d_latent=latent_dim,
        d_hidden=hidden_dim,
        **kwargs,
    )
