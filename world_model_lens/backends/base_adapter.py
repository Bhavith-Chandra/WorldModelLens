"""Abstract base class for world model adapters.

This module defines the interface that all world model backends must implement,
providing a unified API for interpretability analysis across all world model types.

Key Design Principles:
1. Required methods MUST be implemented for every world model
2. Optional methods are only required for models that actually have those heads
3. Capabilities descriptor tells which optional features are available
4. Works with RL models (Dreamer, TD-MPC) AND non-RL models (video, scientific)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


@dataclass
class AdapterConfig:
    """Backend-facing adapter config.

    This intentionally spans both the older backend-local fields and the more
    Dreamer-style fields used across the concrete adapters in this folder.
    """

    d_h: int = 512
    d_z: Optional[int] = None
    d_state: Optional[int] = None
    d_action: int = 0
    d_obs: int = 0
    is_discrete: bool = True
    n_cat: int = 32
    n_cls: int = 32
    n_categories: Optional[int] = None
    n_classes: Optional[int] = None
    encoder_type: str = "cnn"
    n_gru_layers: int = 1
    reward_head: str = "twohot"
    decoder_type: str = "cnn"
    activation: str = "silu"
    n_encoder_channels: int = 48
    encoder_depth: int = 4
    continue_head: str = "logistic"
    discount: float = 0.99
    free_nats: float = 0.0
    kl_scale: float = 1.0
    actor_entropy_scale: float = 1e-4
    imagination_horizon: int = 50
    seed: Optional[int] = None
    has_decoder: bool = True
    has_reward_head: bool = True
    has_value_head: bool = True
    has_policy_head: bool = True
    has_done_head: bool = True
    d_embed: int = 256
    n_layers: int = 4
    n_heads: int = 4
    vocab_size: int = 512
    d_latent: Optional[int] = None
    name: str = "world_model"
    model_type: str = "rssm"
    backend: str = "custom"

    def __post_init__(self) -> None:
        if self.n_categories is None:
            self.n_categories = self.n_cat
        if self.n_classes is None:
            self.n_classes = self.n_cls

        if self.d_state is None:
            self.d_state = self.d_h
        elif self.d_h == 512:
            self.d_h = self.d_state

        if self.d_z is None:
            if self.is_discrete:
                self.d_z = self.n_cat * self.n_cls
            else:
                self.d_z = self.d_state

        if self.d_latent is None:
            self.d_latent = self.d_h + self.d_z


@dataclass
class WorldModelCapabilities:
    """Describes which optional capabilities a world model adapter exposes.

    All fields default to False - adapters should set only what they actually have.
    """

    has_decoder: bool = False
    has_reward_head: bool = False
    has_continue_head: bool = False
    has_actor: bool = False
    has_critic: bool = False
    uses_actions: bool = False  # True for RL/control, False for pure video/science models
    is_rl_trained: bool = False  # RL vs self-supervised vs supervised etc.

    def __post_init__(self):
        """Validate capability consistency."""
        if self.has_actor and not self.uses_actions:
            self.uses_actions = True
        if self.has_reward_head and not self.is_rl_trained:
            self.is_rl_trained = True

    def requires_actions(self) -> bool:
        """Whether this model needs action inputs."""
        return self.uses_actions

    def is_rl_model(self) -> bool:
        """Whether this is an RL-trained model with reward/value heads."""
        return self.is_rl_trained and self.has_reward_head and self.has_critic


class BaseModelAdapter(ABC, nn.Module):
    """Abstract base class for world model architectures.

    Universal interface that works with ANY world model type:
    - RL world models (DreamerV3, TD-MPC2, IRIS)
    - Video prediction models (non-RL)
    - Scientific latent dynamics models (non-RL)
    - World foundation models (non-RL)

    Required vs Optional Methods:
    - REQUIRED: encode, transition, initial_state, sample_z, named_parameters, to, eval, train
    - OPTIONAL: decode, predict_reward, predict_continue, actor_forward, critic_forward
    """

    def __init__(self, config: Any):
        nn.Module.__init__(self)
        self.config = config
        self._capabilities = WorldModelCapabilities()

    @property
    def capabilities(self) -> WorldModelCapabilities:
        """Return which optional features this adapter exposes."""
        return self._capabilities

    @property
    def hook_point_names(self) -> List[str]:
        """Standard hook point names across all components."""
        return [
            "h",
            "z_posterior",
            "z_prior",
            "z_posterior_logits",
            "z_prior_logits",
            "reward_pred",
            "cont_pred",
            "actor_logits",
            "value_pred",
            "obs_reconstruction",
        ]

    # ==================== REQUIRED METHODS ====================
    # These MUST be implemented for every world model

    @abstractmethod
    def encode(
        self,
        obs: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode observation into posterior latent distribution.

        REQUIRED for ALL world models.

        Args:
            obs: Observation tensor [..., *obs_shape]
            h_prev: Previous hidden state [..., d_h]

        Returns:
            Tuple of (z_posterior, z_prior_or_repr).
            - z_posterior: Posterior logits or representation (for RL models: [batch, n_cat, n_cls])
            - z_prior_or_repr: Prior logits (for RL) or just representation (for non-RL video models)
        """
        ...

    def transition(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Transition hidden state using latent and optionally action.

        Base implementation raises NotImplementedError.
        Subclasses should override this method.

        Args:
            h: Current hidden state [..., d_h]
            z: Latent tensor [..., d_z]
            action: Optional action tensor [..., d_action]

        Returns:
            Next hidden state [..., d_h].
        """
        raise NotImplementedError("Subclass must implement transition()")

    def initial_state(
        self, batch_size: int = 1, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden and latent state.

        Base implementation returns zeros.
        Subclasses should override this method.

        Args:
            batch_size: Number of initial states.
            device: Optional device to place tensors on.

        Returns:
            Tuple of (h_0, z_0) initial states.
        """
        if device is None:
            device = torch.device("cpu")
        h = torch.zeros(batch_size, getattr(self.config, "d_h", 512), device=device)
        z = torch.zeros(batch_size, getattr(self.config, "d_z", 512), device=device)
        return h, z

    def sample_z(
        self,
        logits_or_repr: torch.Tensor,
        temperature: float = 1.0,
        sample: bool = True,
    ) -> torch.Tensor:
        """Sample discrete latent from logits, or return representation as-is.

        For categorical latent models: samples from logits using Gumbel-softmax.
        For continuous/embedding models: returns the representation directly.

        Args:
            logits_or_repr: Either logits [..., n_cat, n_cls] or representation [..., d_z]
            temperature: Sampling temperature (only used for categorical)
            sample: If False, return argmax (deterministic)

        Returns:
            Sampled tensor [..., n_cat, n_cls] or [..., d_z]
        """
        n_cls = logits_or_repr.shape[-1]
        if not sample:
            indices = logits_or_repr.argmax(dim=-1)
            return torch.nn.functional.one_hot(indices, num_classes=n_cls).float()

        gumbels = torch.rand_like(logits_or_repr).log().neg()
        gumbels = (logits_or_repr + gumbels) / temperature
        soft = torch.nn.functional.softmax(gumbels, dim=-1)

        indices = soft.argmax(dim=-1)
        hard = torch.nn.functional.one_hot(indices, num_classes=n_cls).float()
        return (hard - soft).detach() + soft

    def named_parameters_dict(self) -> Dict[str, torch.Tensor]:
        """Return named parameters like standard nn.Module."""
        return dict(super().named_parameters())

    # ==================== OPTIONAL METHODS ====================
    # Only implement if your model actually has these heads

    def decode(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Decode latent state into observation reconstruction.

        OPTIONAL - only implement if model has a decoder.

        Args:
            h: Hidden state [..., d_h]
            z: Latent tensor [..., d_z]

        Returns:
            Reconstructed observation [..., *obs_shape] or None if not available
        """
        if self._capabilities.has_decoder:
            raise NotImplementedError("Subclass must implement decode() if has_decoder=True")
        return None

    def predict_reward(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Predict reward distribution.

        OPTIONAL - only implement for RL models with reward heads.

        Args:
            h: Hidden state [..., d_h]
            z: Latent tensor [..., d_z]

        Returns:
            Reward distribution or value, or None if not available
        """
        if self._capabilities.has_reward_head:
            raise NotImplementedError(
                "Subclass must implement predict_reward() if has_reward_head=True"
            )
        return None

    def predict_continue(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Predict episode continuation probability.

        OPTIONAL - only implement for RL models with continue heads.

        Args:
            h: Hidden state [..., d_h]
            z: Latent tensor [..., d_z]

        Returns:
            Continuation logit [..., 1], or None if not available
        """
        if self._capabilities.has_continue_head:
            raise NotImplementedError(
                "Subclass must implement predict_continue() if has_continue_head=True"
            )
        return None

    def actor_forward(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Policy forward pass.

        OPTIONAL - only implement for RL models with actor/policy.

        Args:
            h: Hidden state [..., d_h]
            z: Latent tensor [..., d_z]

        Returns:
            Action logits or distribution parameters, or None if not available
        """
        if self._capabilities.has_actor:
            raise NotImplementedError("Subclass must implement actor_forward() if has_actor=True")
        return None

    def critic_forward(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Value function forward pass.

        OPTIONAL - only implement for RL models with critic.

        Args:
            h: Hidden state [..., d_h]
            z: Latent tensor [..., d_z]

        Returns:
            Value estimate [..., 1], or None if not available
        """
        if self._capabilities.has_critic:
            raise NotImplementedError("Subclass must implement critic_forward() if has_critic=True")
        return None

    def dynamics(
        self,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """Compute prior distribution from hidden state only (for RL models).

        OPTIONAL - primarily used for RL models. Non-RL video models may not need this.

        CRITICAL: This is the dynamics predictor that takes ONLY h_t.
        It does NOT take observations.

        For non-RL models without prior/posterior distinction, this can just
        return the h_t encoding or be overridden to return h directly.

        Args:
            h: Hidden state [..., d_h]

        Returns:
            z_prior_logits [..., n_cat, n_cls] for categorical,
            or [..., d_z] for continuous/representation.
        """
        return h

    # ==================== UTILITY METHODS ====================

    def imagine(
        self,
        start_h: torch.Tensor,
        start_z: torch.Tensor,
        action_sequence: Optional[torch.Tensor] = None,
        horizon: int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run imagined rollouts using the dynamics model.

        Works for both RL and non-RL models (action_sequence can be None).

        Args:
            start_h: Initial hidden state [d_h]
            start_z: Initial latent [d_z]
            action_sequence: Optional actions to execute [horizon, d_action]
            horizon: Number of steps to imagine

        Returns:
            Tuple of (h_sequence, z_sequence) during imagination
        """
        h_t = start_h
        z_t = start_z

        h_seq = [h_t]
        z_seq = [z_t]

        for t in range(horizon):
            action = None
            if action_sequence is not None and t < len(action_sequence):
                action = action_sequence[t]
            elif self._capabilities.uses_actions:
                action = (
                    torch.zeros_like(start_h)[: self.config.d_action]
                    if self.config.d_action
                    else None
                )

            h_t = self.transition(
                h_t.unsqueeze(0) if h_t.dim() == 1 else h_t,
                z_t.unsqueeze(0) if z_t.dim() == 1 else z_t,
                action.unsqueeze(0) if action is not None and action.dim() == 1 else action,
            )
            h_t = h_t.squeeze(0)

            prior = self.dynamics(h_t.unsqueeze(0) if h_t.dim() == 1 else h_t)
            z_t = self.sample_z(prior.squeeze(0), temperature=1.0)

            h_seq.append(h_t)
            z_seq.append(z_t)

        return torch.stack(h_seq, dim=0), torch.stack(z_seq, dim=0)


    def to(self, *args: Any, device: torch.device, **kwargs: Any) -> "BaseModelAdapter":
        """Move adapter to device (standard nn.Module interface)."""
        return super().to(device)

    def _check_capabilities(self) -> None:
        """Validate capability settings are consistent.

        Override in subclasses to add validation logic.
        """
        pass
