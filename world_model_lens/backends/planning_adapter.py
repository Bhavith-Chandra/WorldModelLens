"""Planning Model Adapter.

Example of a planning-oriented world model that works with World Model Lens.

Planning models predict action sequences to achieve goals.
They fit the "encode + dynamics + plan" pattern.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn

from world_model_lens.backends.base_adapter import BaseModelAdapter, AdapterConfig


class PlanningAdapter(BaseModelAdapter):
    """World model adapter for planning-oriented models.

    This model:
    - Encodes observations into state
    - Uses dynamics for imagination
    - Predicts actions for planning
    - Optionally predicts goal achievement

    No reward or value predictions - planning is goal-based.
    """

    def __init__(self, config: AdapterConfig, goal_dim: int = 32):
        super().__init__(config)
        self.goal_dim = goal_dim

        self.encoder = nn.Sequential(
            nn.Linear(config.d_obs + goal_dim, 256),
            nn.ReLU(),
            nn.Linear(256, config.d_state),
        )

        self.transition_model = nn.Sequential(
            nn.Linear(config.d_state + config.d_action, 256),
            nn.ReLU(),
            nn.Linear(256, config.d_state),
        )

        self.planner = nn.Sequential(
            nn.Linear(config.d_state + goal_dim, 256),
            nn.ReLU(),
            nn.Linear(256, config.d_action * 10),
        )

        self.goal_achievement_head = nn.Sequential(
            nn.Linear(config.d_state + goal_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def _goal_from_h(
        self,
        h: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Project hidden state onto the goal-conditioning space."""
        if h is None:
            return torch.zeros(batch_size, self.goal_dim, device=device)

        if h.dim() == 1:
            h = h.unsqueeze(0)

        if h.shape[-1] == self.goal_dim:
            return h

        if h.shape[-1] > self.goal_dim:
            return h[..., : self.goal_dim]

        pad = self.goal_dim - h.shape[-1]
        return torch.nn.functional.pad(h, (0, pad))

    def encode(
        self,
        obs: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode observation plus goal context into the planning latent."""
        if obs.dim() > 2:
            obs = obs.flatten(1)
        elif obs.dim() == 1:
            obs = obs.unsqueeze(0)

        goal = self._goal_from_h(h_prev, obs.shape[0], obs.device)
        x = torch.cat([obs, goal], dim=-1)
        z = self.encoder(x)
        return z, z

    def transition(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict next planning state.

        This adapter is effectively single-state internally, so `z` is treated
        as the current planning state and the returned hidden state is the next
        planning state.
        """
        del h
        if action is None:
            action = torch.zeros(z.shape[0], self.config.d_action, device=z.device)
        x = torch.cat([z, action], dim=-1)
        return self.transition_model(x)

    def dynamics(self, h: torch.Tensor) -> torch.Tensor:
        """Planning adapter uses the hidden state itself as the next latent prior."""
        return h if h.dim() > 1 else h.unsqueeze(0)

    def sample_z(
        self,
        logits_or_repr: torch.Tensor,
        temperature: float = 1.0,
        sample: bool = True,
    ) -> torch.Tensor:
        """Planning latents are continuous representations, not categorical logits."""
        del temperature, sample
        return logits_or_repr

    def plan(
        self,
        current_state: torch.Tensor,
        goal: torch.Tensor,
        horizon: int = 10,
    ) -> torch.Tensor:
        """Plan action sequence to reach goal.

        Args:
            current_state: Current state
            goal: Goal state
            horizon: Planning horizon

        Returns:
            Planned action sequence [horizon, d_action]
        """
        x = torch.cat([current_state, goal], dim=-1)
        action_sequence = self.planner(x)
        return action_sequence.view(-1, horizon, self.config.d_action)

    def actor_forward(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Return the first action from the planned action sequence."""
        goal = self._goal_from_h(h, z.shape[0], z.device)
        planned_actions = self.plan(z, goal)
        return planned_actions[:, 0]

    def predict_goal_achievement(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        """Predict whether goal is achieved."""
        x = torch.cat([state, goal], dim=-1)
        return torch.sigmoid(self.goal_achievement_head(x))
