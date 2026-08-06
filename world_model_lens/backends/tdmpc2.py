"""TD-MPC2 adapter implementation.

TD-MPC2 has no explicit decoder and no recurrent state.
Uses ResNet encoder and continuous latent.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn

from world_model_lens.backends.base_adapter import BaseModelAdapter, AdapterConfig, WorldModelCapabilities


class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResNetEncoder(nn.Module):
    def __init__(self, d_obs: int, d_h: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_obs, d_h),
            nn.ReLU(),
            *[ResBlock(d_h) for _ in range(3)],
            nn.LayerNorm(d_h),
        )
        self.out_dim = d_h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPDynamics(nn.Module):
    def __init__(self, d_h: int, d_action: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_h + d_action, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_h),
        )

    def forward(self, h: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, action], dim=-1)
        return self.net(x)


class RewardPredictor(nn.Module):
    def __init__(self, d_h: int, d_z: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_h + d_z, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, z], dim=-1)
        return self.net(x)


class PolicyHead(nn.Module):
    def __init__(self, d_h: int, d_action: int):
        super().__init__()
        self.mean = nn.Linear(d_h, d_action)
        self.log_std = nn.Parameter(torch.zeros(d_action))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        mean = self.mean(h)
        std = torch.exp(self.log_std).unsqueeze(0).expand_as(mean)
        return torch.cat([mean, std], dim=-1)


class ValueHead(nn.Module):
    def __init__(self, d_h: int, d_z: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_h + d_z, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, z], dim=-1)
        return self.net(x)


class TDMPC2Adapter(BaseModelAdapter):
    """TD-MPC2: Temporal Difference MPC.

    Continuous latent with no recurrent state. No decoder.
    """

    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        self.config = config
        self.latent_dim = config.d_h

        self.encoder = ResNetEncoder(config.d_obs, self.latent_dim)
        self.dynamics_model = MLPDynamics(self.latent_dim, config.d_action)
        self.reward_predictor = RewardPredictor(self.latent_dim, self.latent_dim)
        self.policy = PolicyHead(self.latent_dim, config.d_action)
        self.value = ValueHead(self.latent_dim, self.latent_dim)

        self._capabilities = WorldModelCapabilities(
            has_decoder=False,
            has_reward_head=True,
            has_continue_head=True,
            has_actor=True,
            has_critic=True,
            uses_actions=True,
            is_rl_trained=True,
        )

        self._device = torch.device("cpu")

    def encode(self, obs: torch.Tensor, h_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        del h_prev
        if obs.dim() > 2:
            obs = obs.flatten(start_dim=1)
        elif obs.dim() == 1:
            obs = obs.unsqueeze(0)

        z = self.encoder(obs)
        return z, z

    def dynamics_fn(self, h: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.dynamics_model(h, action)

    def dynamics(self, h: torch.Tensor) -> torch.Tensor:
        return h if h.dim() > 1 else h.unsqueeze(0)

    def sample_z(
        self,
        logits_or_repr: torch.Tensor,
        temperature: float = 1.0,
        sample: bool = True,
    ) -> torch.Tensor:
        del temperature, sample
        return logits_or_repr

    def decode(self, h: torch.Tensor, z: torch.Tensor) -> None:
        del h, z
        return None

    def predict_reward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if h.dim() == 1:
            h = h.unsqueeze(0)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return self.reward_predictor(h, z)

    def predict_continue(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        batch_size = h.shape[0] if h.dim() > 1 else 1
        return torch.zeros(batch_size, 1, device=h.device)

    def actor_forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        del z
        if h.dim() == 1:
            h = h.unsqueeze(0)
        return self.policy(h)

    def critic_forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if h.dim() == 1:
            h = h.unsqueeze(0)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return self.value(h, z)

    def transition(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del h
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if action is None:
            action = torch.zeros(z.shape[0], self.config.d_action, device=z.device)
        elif action.dim() == 1:
            action = action.unsqueeze(0)
        return self.dynamics_model(z, action)

    def initial_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if device is None:
            device = self._device
        h = torch.zeros(batch_size, self.latent_dim, device=device)
        z = torch.zeros(batch_size, self.latent_dim, device=device)
        return h, z

    def named_parameters(self):
        return dict(nn.Module.named_parameters(self))

    def to(self, device: torch.device) -> "TDMPC2Adapter":
        super().to(device)
        self._device = device
        return self

    def eval(self) -> "TDMPC2Adapter":
        super().eval()
        return self

    def train(self, mode: bool = True) -> "TDMPC2Adapter":
        super().train(mode)
        return self

    def list_hookable_points(self) -> List[str]:
        """Return semantic hook names plus concrete module paths."""
        points = list(self.hook_point_names)
        points.extend(name for name, _ in self.named_modules() if name)
        return list(dict.fromkeys(points))
