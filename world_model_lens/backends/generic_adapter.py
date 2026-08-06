"""Abstract base class for world model adapters.

This is the GENERIC interface that ANY world model can implement.
It is backend-agnostic and supports:

- DreamerV1/V2/V3 (RSSM-based)
- PlaNet (latent RSSM)
- Ha & Schmidhuber World Models (VAE + MDN-RNN)
- TD-MPC/TD-MPC2 (continuous latent, JEPA-style)
- IRIS/transformer-based models
- Decision/Trajectory Transformers
- Contrastive/Predictive models (CWM, SPR)
- Video prediction models (WorldDreamer)
- Autonomous driving world models
- Robotics/embodied world models
- Future architectures

RL-specific components (reward, value, action, done) are OPTIONAL.
Non-RL world models need only implement the core methods.

Key Design Principles:
1. Core methods are REQUIRED: encode(), dynamics(), transition()
2. RL methods are OPTIONAL: reward prediction, value estimation, policy
3. Decoders are OPTIONAL: not all world models reconstruct
4. Everything assumes batched tensors but works with unbatched
5. Adapters are categorized by WorldModelFamily for registry lookup

The adapter interface enforces:
- Backend-agnostic core abstractions (h_t, z_t, actions, rewards optional)
- Support for continuous/discrete/VQ/hybrid latents
- Support for recurrent/transformer/convolutional dynamics
- Optional reward/value/action fields for non-RL models
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import torch
import torch.nn as nn

from world_model_lens.core.types import LatentType, DynamicsType, WorldModelFamily, ModelPurpose

if TYPE_CHECKING:
    from world_model_lens.core.world_state import WorldState, WorldDynamics, WorldModelOutput
    from world_model_lens.core.world_trajectory import WorldTrajectory


@dataclass
class AdapterConfig:
    """Configuration for any world model.

    This is intentionally minimal. Model-specific configs should
    extend or specialize this.
    """

    d_state: int = 512
    d_action: int = 0
    d_obs: int = 0
    is_discrete: bool = True
    n_categories: int = 32
    n_classes: int = 32
    name: str = "world_model"
    model_type: str = "rssm"

    @property
    def d_latent(self) -> int:
        """Total latent dimension."""
        if self.is_discrete:
            return self.n_categories * self.n_classes
        return self.d_state


class WorldModelAdapter(ABC, nn.Module):
    """Abstract base for ANY world model.

    This interface is designed to be:
    - Backend-agnostic: Works with any world model architecture
    - Minimal: Only core methods are required
    - Extensible: Optional methods for RL-specific functionality
    - Future-proof: New architectures fit naturally

    REQUIRED methods (core functionality):
    - encode(): Convert observation to state representation
    - dynamics(): Predict next state without observation (imagination)
    - transition(): Transition state using action/input

    OPTIONAL methods (RL-specific):
    - predict_reward(): Reward prediction head
    - predict_value(): Value estimation head
    - predict_done(): Episode termination prediction
    - actor_forward(): Policy/action prediction
    - decode(): Observation reconstruction (if applicable)

    All methods accept both batched [B, ...] and unbatched [...] tensors.
    """

    def __init__(self, config: AdapterConfig):
        nn.Module.__init__(self)
        self.config = config
        self._device = torch.device("cpu")

    @property
    def hook_point_names(self) -> List[str]:
        """Hook points exposed by this model.

        Override to add model-specific hook points.
        Standard points:
        - state: Core latent state
        - prior: Prior prediction (dynamics)
        - posterior: Posterior (observation-conditioned)
        - observation: Encoded observation
        - reward: Reward prediction
        - value: Value prediction
        """
        return [
            "state",
            "prior",
            "posterior",
            "observation",
            "reward",
            "value",
            "action",
            "done",
        ]

    @abstractmethod
    def encode(
        self,
        observation: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode observation into state representation.

        This is the core method - ALL world models must implement this.

        Args:
            observation: Observation tensor [..., *obs_shape]
            context: Optional context (e.g., previous hidden state)

        Returns:
            Tuple of (posterior_state, observation_encoding)
            - posterior_state: Observation-conditioned state [..., d_state] or [..., n_cat, n_cls]
            - observation_encoding: Optional intermediate encoding

        Example implementations:
        - Dreamer: CNN encoder -> RSSM posterior
        - TD-MPC2: ResNet encoder -> continuous latent
        - IRIS: VQVAE encoder -> discrete token
        - Video: Video encoder -> state sequence
        """
        ...

    @abstractmethod
    def dynamics(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict next state without observation (imagination/prior).

        This is the dynamics model - the core of imagination.

        Args:
            state: Current state [..., d_state]
            action: Optional action to condition on [..., d_action]

        Returns:
            Prior/predicted next state [..., d_state] or [..., n_cat, n_cls]

        Note: For models without actions (video prediction), action can be None.
        """
        ...

    def transition(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        input_: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Transition state using action or other input.

        For most RSSM-style models, this is the GRU/transition step.
        For other models, this may be a simple pass-through or
        learned transition function.

        Args:
            state: Current state [..., d_state]
            action: Optional action [..., d_action]
            input_: Optional additional input [..., *]

        Returns:
            Next state [..., d_state]
        """
        return self.dynamics(state, action)

    def predict_reward(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Predict reward for a state (OPTIONAL).

        Only implement for RL applications.

        Args:
            state: Current state [..., d_state]
            action: Optional action [..., d_action]

        Returns:
            Reward prediction [..., 1] or [..., n_classes] or None
        """
        return None

    def predict_value(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Predict value estimate for a state (OPTIONAL).

        Only implement for RL applications with value estimation.

        Args:
            state: Current state [..., d_state]
            action: Optional action [..., d_action]

        Returns:
            Value estimate [..., 1] or None
        """
        return None

    def predict_done(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Predict episode termination (OPTIONAL).

        Only implement for environments with episode boundaries.

        Args:
            state: Current state [..., d_state]
            action: Optional action [..., d_action]

        Returns:
            Done probability/logit [..., 1] or None
        """
        return None

    def decode(
        self,
        state: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Decode state to observation (OPTIONAL).

        Only implement if the model has observation reconstruction.

        Args:
            state: Current state [..., d_state]

        Returns:
            Reconstructed observation [..., *obs_shape] or None
        """
        return None

    def actor_forward(
        self,
        state: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Policy/action prediction (OPTIONAL).

        Only implement for RL policy extraction.

        Args:
            state: Current state [..., d_state]

        Returns:
            Action logits/distribution or None
        """
        return None

    def sample_state(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        sample: bool = True,
    ) -> torch.Tensor:
        """Sample from state distribution.

        For discrete states: Gumbel-softmax sampling
        For continuous states: passthrough

        Args:
            logits: State logits [..., n_cat, n_cls] or [..., d_state]
            temperature: Sampling temperature
            sample: If False, return argmax (deterministic)

        Returns:
            Sampled state tensor
        """
        if not self.config.is_discrete:
            return logits

        if not sample:
            indices = logits.argmax(dim=-1)
            return torch.nn.functional.one_hot(indices, num_classes=logits.shape[-1]).float()

        gumbels = torch.rand_like(logits).log().neg()
        gumbels = (logits + gumbels) / temperature
        soft = torch.nn.functional.softmax(gumbels, dim=-1)
        indices = soft.argmax(dim=-1)
        hard = torch.nn.functional.one_hot(indices, num_classes=logits.shape[-1]).float()
        return (hard - soft).detach() + soft

    def initial_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Initialize starting state.

        Args:
            batch_size: Number of initial states
            device: Optional device specification

        Returns:
            Initial state tensor [batch_size, d_state]
        """
        if self.config.is_discrete:
            state = torch.zeros(
                batch_size,
                self.config.n_categories,
                self.config.n_classes,
                device=device or self._device,
            )
            state[:, :, 0] = 1.0
        else:
            state = torch.zeros(
                batch_size,
                self.config.d_state,
                device=device or self._device,
            )
        return state

    def imagine(
        self,
        start_state: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        horizon: int = 50,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]]]:
        """Run imagined rollout using dynamics model.

        Core imagination capability - works with ANY world model.

        Args:
            start_state: Starting state [d_state] or [B, d_state]
            actions: Optional action sequence [horizon, d_action]
            horizon: Number of imagination steps
            temperature: Sampling temperature for discrete states

        Returns:
            Tuple of:
            - state_sequence: [horizon+1, d_state] or [horizon+1, B, d_state]
            - reward_predictions: List of reward tensors or None
        """
        state = start_state
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        B = state.shape[0]
        state_seq = [state]
        rewards = []

        for t in range(horizon):
            action = None
            if actions is not None and t < len(actions):
                action = actions[t]
                if action.dim() == 1:
                    action = action.unsqueeze(0).expand(B, -1)

            prior = self.dynamics(state, action)
            state = self.sample_state(prior, temperature=temperature)

            reward = self.predict_reward(state, action)
            rewards.append(reward)
            state_seq.append(state)

        state_seq_tensor = torch.stack(state_seq, dim=0)

        if squeeze_output:
            state_seq_tensor = state_seq_tensor.squeeze(1)

        return state_seq_tensor, rewards

    def forward(
        self,
        observation: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
    ) -> "WorldModelOutput":
        """Full forward pass (OPTIONAL - convenience method).

        Override for model-specific forward logic.

        Args:
            observation: Current observation
            state: Previous state (for recurrent models)
            action: Action taken

        Returns:
            WorldModelOutput with all predictions
        """
        from world_model_lens.core.world_state import WorldState, WorldDynamics, WorldModelOutput

        posterior, obs_encoding = self.encode(observation, state)
        prior = self.dynamics(state if state is not None else self.initial_state(), action)

        kl = None
        if posterior.shape == prior.shape:
            p = posterior.clamp(min=1e-8)
            q = prior.clamp(min=1e-8)
            p = p / p.sum(dim=-1, keepdim=True)
            q = q / q.sum(dim=-1, keepdim=True)
            kl = (p * (p.log() - q.log())).sum(dim=-1, keepdim=True)

        next_state = self.transition(posterior, action)

        output = WorldModelOutput(
            next_state=WorldState(state=next_state, timestep=0),
            observation_reconstruction=self.decode(next_state),
            reward_prediction=self.predict_reward(next_state, action),
            value_prediction=self.predict_value(next_state, action),
            dynamics=WorldDynamics(
                prior_state=prior,
                posterior_state=posterior,
                kl_divergence=kl,
            ),
            hidden_state=obs_encoding,
        )

        return output

    def to(self, device: torch.device) -> "WorldModelAdapter":
        """Move model to device."""
        super().to(device)
        self._device = device
        return self

    @property
    def device(self) -> torch.device:
        """Current device."""
        return self._device
