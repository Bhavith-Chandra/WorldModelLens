"""Custom adapter template.

This template provides a heavily documented skeleton for implementing
new world model adapters. Fill in each method to integrate with World Model Lens.
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn

from world_model_lens.backends.base_adapter import BaseModelAdapter, AdapterConfig


class CustomAdapter(BaseModelAdapter):
    """Template for implementing custom world model adapters.

    Tips and Common Mistakes:
    1. Dynamics (prior) takes ONLY h_t - no observation.
       The encoder (posterior) takes BOTH obs AND h_t.
    2. sample_z must return one-hot for categorical or raw vector for continuous.
    3. named_parameters must return structured names like 'component.layer.weight'.
    4. All methods must handle both batched (B,) and unbatched (1,) inputs.
    5. Don't forget to set self._device in __init__ for initial_state.

    Checklist:
    - [ ] Implement all abstract methods
    - [ ] Add custom components in __init__
    - [ ] Handle batched/unbatched inputs
    - [ ] Implement named_parameters with structured names
    - [ ] Set self._device in __init__
    - [ ] Implement from_checkpoint or infer_config classmethods
    - [ ] Register in backends/__init__.py and backends/registry.py
    """

    def __init__(self, config: AdapterConfig):
        """Initialize your custom adapter.

        Args:
            config: AdapterConfig with architecture hyperparameters.
        """
        super().__init__(config)
        self.config = config

        # TODO: Add your model components here
        # Example:
        # self.encoder = YourEncoder(config)
        # self.transition = YourTransition(config)
        # self.reward_head = YourRewardHead(config)

        # IMPORTANT: Set device for initial_state
        self._device = torch.device("cpu")

    @property
    def hook_point_names(self) -> List[str]:
        """Return the list of hook point names this adapter exposes.

        Extend the standard list with model-specific points if needed.
        """
        names = super().hook_point_names
        # TODO: Add model-specific hook points
        # names.append('my_custom_activation')
        return names

    def encode(
        self,
        obs: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode observation into latent distribution.

        Args:
            obs: Observation tensor [..., *obs_shape]
            h_prev: Previous hidden state [..., d_h]

        Returns:
            Tuple of (z_posterior_logits, obs_encoding)
        """
        # TODO: Implement encoding
        # Handle both batched and unbatched inputs
        # obs: [B, *obs_shape] or [*obs_shape]
        # h_prev: [B, d_h] or [d_h]
        raise NotImplementedError("Implement encode method")

    def dynamics(self, h: torch.Tensor) -> torch.Tensor:
        """Compute prior distribution from hidden state only.

        CRITICAL: This is the dynamics/transition predictor that takes
        ONLY the hidden state h_t. It does NOT take observations.

        Args:
            h: Hidden state [..., d_h]

        Returns:
            z_prior_logits [..., n_cat, n_cls] for categorical,
            or [..., d_z] for continuous.
        """
        # TODO: Implement dynamics predictor
        raise NotImplementedError("Implement dynamics method")

    def transition(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Transition hidden state using latent and action.

        Args:
            h: Current hidden state [..., d_h]
            z: Latent tensor [..., d_z]
            action: Action tensor [..., d_action]

        Returns:
            Next hidden state [..., d_h]
        """
        # TODO: Implement transition (e.g., GRU cell)
        raise NotImplementedError("Implement transition method")

    def decode(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Decode latent state into observation reconstruction.

        Args:
            h: Hidden state [..., d_h]
            z: Latent tensor [..., d_z]

        Returns:
            Reconstructed observation [..., *obs_shape]
        """
        # TODO: Implement decoder
        raise NotImplementedError("Implement decode method")

    def predict_reward(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Predict reward distribution.

        Args:
            h: Hidden state [..., d_h]
            z: Latent tensor [..., d_z]

        Returns:
            Reward distribution (format depends on your model)
        """
        # TODO: Implement reward prediction
        raise NotImplementedError("Implement predict_reward method")

    def predict_continue(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Predict episode continuation probability.

        Args:
            h: Hidden state [..., d_h]
            z: Latent tensor [..., d_z]

        Returns:
            Continuation logit [..., 1]
        """
        # TODO: Implement continue prediction
        raise NotImplementedError("Implement predict_continue method")

    def actor_forward(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Policy forward pass.

        Args:
            h: Hidden state [..., d_h]
            z: Latent tensor [..., d_z]

        Returns:
            Action logits or distribution parameters.
        """
        # TODO: Implement actor
        raise NotImplementedError("Implement actor_forward method")

    def critic_forward(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Value function forward pass.

        Args:
            h: Hidden state [..., d_h]
            z: Latent tensor [..., d_z]

        Returns:
            Value estimate [..., 1]
        """
        # TODO: Implement critic
        raise NotImplementedError("Implement critic_forward method")

    def initial_state(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden and latent state.

        Args:
            batch_size: Number of initial states to create.

        Returns:
            Tuple of (h_0, z_0) initial states.
        """
        # TODO: Implement initial state creation
        # h_0: [batch_size, d_h]
        # z_0: [batch_size, n_cat, n_cls] for categorical
        #     or [batch_size, d_z] for continuous
        raise NotImplementedError("Implement initial_state method")

    def named_parameters(self) -> Dict[str, torch.Tensor]:
        """Return all model parameters with structured names.

        Returns:
            Dict mapping parameter names to tensors.
            Names should follow pattern: 'component.layer.weight'
            Example: 'encoder.conv1.weight', 'transition.gru.weight_ih'
        """
        # TODO: Return named parameters
        # Use self.named_parameters(full=True) from nn.Module
        params = {}
        for name, param in super().named_parameters(full=True):
            params[name] = param
        return params

    @classmethod
    def from_checkpoint(
        cls, path: str, config: Optional[AdapterConfig] = None
    ) -> "CustomAdapter":
        """Load adapter from checkpoint file.

        Args:
            path: Path to checkpoint (.pt file).
            config: Optional config. If None, infer from checkpoint.

        Returns:
            Loaded CustomAdapter instance.
        """
        # TODO: Implement checkpoint loading
        raise NotImplementedError("Implement from_checkpoint classmethod")

    @classmethod
    def infer_config(cls, state_dict: Dict) -> AdapterConfig:
        """Infer config from state dict shapes.

        Args:
            state_dict: Model state dictionary.

        Returns:
            AdapterConfig with inferred parameters.
        """
        # TODO: Infer config from state dict
        # Example: examine weight shapes to determine dimensions
        raise NotImplementedError("Implement infer_config classmethod")

    def to(self, device: torch.device) -> "CustomAdapter":
        """Move model to device."""
        super().to(device)
        self._device = device
        return self

    def eval(self) -> "CustomAdapter":
        """Set to evaluation mode."""
        super().eval()
        return self

    def train(self, mode: bool = True) -> "CustomAdapter":
        """Set training mode."""
        super().train(mode)
        return self
