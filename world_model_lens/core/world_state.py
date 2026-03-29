"""Generic world state representation.

World Model Lens is backend-agnostic and works with ANY world model that:
- Maintains a latent "world state"
- Has transition/dynamics dynamics
- Can encode observations into state
- Optionally predicts rewards, values, done flags, actions

Supported model types:
- DreamerV2/V3 (RSSM-based)
- TD-MPC2 (continuous latent)
- IRIS/transformer-based models
- Video prediction models
- Planning-oriented latent models
- Future architectures fitting the pattern

RL-specific components (reward, value, action, done) are OPTIONAL extensions.
Non-RL world models (video prediction, unsupervised) are first-class citizens.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from typing_extensions import TypedDict

import torch


@dataclass
class WorldState:
    """Generic world state representation.

    This is the core abstraction - a model-independent representation of
    the latent state at a single timestep.

    Optional fields (RL-specific):
    - action: Action taken to reach this state
    - reward: Reward received for reaching this state
    - value: Value estimate at this state
    - done: Episode termination flag
    - obs_encoding: Observation encoding (if available)

    For non-RL models, only `state` is required.

    Attributes:
        state: The core latent state tensor [d_state]
        timestep: Current timestep index
        action: Optional action taken [d_action]
        reward: Optional reward received (can be scalar or distribution)
        reward_pred: Optional predicted reward
        value: Optional value estimate
        value_pred: Optional predicted value
        done: Optional episode termination flag
        obs_encoding: Optional observation encoding from encoder
        metadata: Arbitrary additional data (e.g., video frames, sensor data)
    """

    state: torch.Tensor
    timestep: int = 0
    action: Optional[torch.Tensor] = None
    reward: Optional[torch.Tensor] = None
    reward_pred: Optional[torch.Tensor] = None
    value: Optional[torch.Tensor] = None
    value_pred: Optional[torch.Tensor] = None
    done: Optional[torch.Tensor] = None
    obs_encoding: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def flat(self) -> torch.Tensor:
        """Flattened core state tensor."""
        return self.state.flatten()

    @property
    def device(self) -> torch.device:
        """Device the state tensors are on."""
        return self.state.device

    @property
    def shape(self) -> torch.Size:
        """Shape of the core state tensor."""
        return self.state.shape

    def has_reward(self) -> bool:
        """Check if reward is available."""
        return self.reward is not None or self.reward_pred is not None

    def has_value(self) -> bool:
        """Check if value is available."""
        return self.value is not None or self.value_pred is not None

    def has_action(self) -> bool:
        """Check if action is available."""
        return self.action is not None

    def is_terminal(self) -> Optional[bool]:
        """Check if this is a terminal state."""
        if self.done is None:
            return None
        return bool(self.done.item()) if self.done.numel() == 1 else None

    def to_device(self, device: torch.device) -> "WorldState":
        """Move all tensors to a device."""
        import copy

        new_state = copy.copy(self)
        for attr in [
            "state",
            "action",
            "reward",
            "reward_pred",
            "value",
            "value_pred",
            "done",
            "obs_encoding",
        ]:
            val = getattr(self, attr)
            if isinstance(val, torch.Tensor):
                setattr(new_state, attr, val.to(device))
        return new_state

    def detach(self) -> "WorldState":
        """Detach all tensors from computation graph."""
        import copy

        new_state = copy.copy(self)
        for attr in [
            "state",
            "action",
            "reward",
            "reward_pred",
            "value",
            "value_pred",
            "done",
            "obs_encoding",
        ]:
            val = getattr(self, attr)
            if isinstance(val, torch.Tensor):
                setattr(new_state, attr, val.detach())
        return new_state

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "timestep": self.timestep,
            "state_shape": list(self.state.shape),
            "has_action": self.has_action(),
            "has_reward": self.has_reward(),
            "has_value": self.has_value(),
            "is_terminal": self.is_terminal(),
        }
        for key, val in self.metadata.items():
            result[f"meta_{key}"] = val
        return result


@dataclass
class WorldDynamics:
    """Dynamics prediction output.

    Represents the model's prediction of future states,
    independent of whether this is a posterior (observed) or
    prior (imagined) prediction.
    """

    prior_state: Optional[torch.Tensor] = None
    posterior_state: Optional[torch.Tensor] = None
    kl_divergence: Optional[torch.Tensor] = None
    surprise: Optional[torch.Tensor] = None
    dynamics_logits: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def compute_surprise(self) -> torch.Tensor:
        """Compute surprise (KL divergence) between prior and posterior."""
        if self.kl_divergence is not None:
            return self.kl_divergence
        if self.prior_state is not None and self.posterior_state is not None:
            p = self.posterior_state.clamp(min=1e-8)
            q = self.prior_state.clamp(min=1e-8)
            p = p / p.sum(dim=-1, keepdim=True)
            q = q / q.sum(dim=-1, keepdim=True)
            self.surprise = (p * (p.log() - q.log())).sum(dim=-1)
            return self.surprise
        return torch.tensor(0.0)


class ObservationType:
    """Type alias for flexible observation formats."""

    pass


ObservationType = Union[
    torch.Tensor,  # Single frame [C, H, W] or [D]
    Dict[str, torch.Tensor],  # Multi-modal {"image": ..., "state": ...}
    List[torch.Tensor],  # Video sequence [T, C, H, W]
    Dict[str, Any],  # Complex observations with metadata
]


@dataclass
class WorldModelOutput:
    """Output from a world model forward pass.

    Generic container that holds all outputs from a world model.
    Only `next_state` is required; all others are optional.
    """

    next_state: WorldState
    observation_reconstruction: Optional[torch.Tensor] = None
    reward_prediction: Optional[torch.Tensor] = None
    value_prediction: Optional[torch.Tensor] = None
    action_prediction: Optional[torch.Tensor] = None
    done_prediction: Optional[torch.Tensor] = None
    dynamics: Optional[WorldDynamics] = None
    hidden_state: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
