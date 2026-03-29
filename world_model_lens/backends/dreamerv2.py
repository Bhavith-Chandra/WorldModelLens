"""DreamerV2 adapter implementation.

Differs from V3: Gaussian reward (not two-hot), ELU activations,
no symlog, discrete z without symlog.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from world_model_lens.backends.base_adapter import WorldModelAdapter
from world_model_lens.core.config import WorldModelConfig


class MLP(nn.Module):
    def __init__(self, dims: List[int], activation: str = "elu"):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
        self.activation_name = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                if self.activation_name == "elu":
                    x = torch.nn.functional.elu(x)
                elif self.activation_name == "relu":
                    x = torch.nn.functional.relu(x)
                elif self.activation_name == "silu":
                    x = torch.nn.functional.silu(x)
        return x


class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32, depth: int = 4):
        super().__init__()
        layers = []
        ch = in_channels
        for i in range(depth):
            out_ch = base_channels * (2**i)
            layers.append(nn.Conv2d(ch, out_ch, kernel_size=4, stride=2, padding=1))
            layers.append(nn.ELU())
            ch = out_ch
        self.cnn = nn.Sequential(*layers)
        self.out_dim = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            B, T = x.shape[:2]
            x = x.flatten(0, 1)
            x = self.cnn(x)
            x = x.mean(dim=[2, 3])
            x = x.view(B, T, -1)
            return x
        return self.cnn(x).mean(dim=[2, 3])


class VectorEncoder(nn.Module):
    def __init__(self, d_obs: int, d_h: int):
        super().__init__()
        self.mlp = MLP([d_obs, d_h, d_h], activation="elu")
        self.out_dim = d_h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class DreamerV2Encoder(nn.Module):
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        if config.encoder_type == "cnn":
            self.cnn = CNNEncoder(in_channels=3, base_channels=32, depth=4)
            self.out_dim = self.cnn.out_dim
        else:
            self.vector = VectorEncoder(config.d_obs, config.d_h)
            self.out_dim = config.d_h

        self.fc = nn.Linear(self.out_dim + config.d_h, config.d_z)

    def forward(self, obs: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 4:
            obs_encoding = self.cnn(obs / 255.0)
        else:
            obs_encoding = self.vector(obs)

        if obs_encoding.dim() == 2:
            obs_encoding = obs_encoding.unsqueeze(0)
        if h_prev.dim() == 1:
            h_prev = h_prev.unsqueeze(0)

        x = torch.cat([obs_encoding, h_prev], dim=-1)
        return self.fc(x)


class DreamerV2DynamicsPredictor(nn.Module):
    def __init__(self, d_h: int, d_z: int):
        super().__init__()
        self.mlp = MLP([d_h, d_h, d_z], activation="elu")

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.mlp(h)


class DreamerV2Transition(nn.Module):
    def __init__(self, d_z: int, d_action: int, d_h: int):
        super().__init__()
        self.gru = nn.GRUCell(input_size=d_z + d_action, hidden_size=d_h)

    def forward(self, h: torch.Tensor, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, action], dim=-1)
        return self.gru(x, h)


class DreamerV2Decoder(nn.Module):
    def __init__(self, d_h: int, d_z: int):
        super().__init__()
        self.mlp = MLP([d_h + d_z, 1024, 4096], activation="elu")

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, z], dim=-1)
        return self.mlp(x)


class DreamerV2RewardHead(nn.Module):
    def __init__(self, d_h: int, d_z: int):
        super().__init__()
        self.fc_mean = nn.Linear(d_h + d_z, 1)
        self.fc_std = nn.Linear(d_h + d_z, 1)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, z], dim=-1)
        mean = self.fc_mean(x)
        std = torch.exp(self.fc_std(x).clamp(-5, 2))
        return torch.cat([mean, std], dim=-1)


class DreamerV2ContinueHead(nn.Module):
    def __init__(self, d_h: int, d_z: int):
        super().__init__()
        self.fc = nn.Linear(d_h + d_z, 1)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, z], dim=-1)
        return self.fc(x)


class DreamerV2Actor(nn.Module):
    def __init__(self, d_h: int, d_z: int, d_action: int, n_classes: int = 32):
        super().__init__()
        self.n_classes = n_classes
        self.d_action = d_action
        self.mlp = MLP([d_h + d_z, 512, d_action * n_classes], activation="elu")

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, z], dim=-1)
        out = self.mlp(x)
        return out.view(*x.shape[:-1], self.d_action, self.n_classes)


class DreamerV2Critic(nn.Module):
    def __init__(self, d_h: int, d_z: int):
        super().__init__()
        self.mlp = MLP([d_h + d_z, 512, 1], activation="elu")

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, z], dim=-1)
        return self.mlp(x)


class DreamerV2Adapter(WorldModelAdapter):
    """DreamerV2 with ELU activations and Gaussian rewards."""

    def __init__(self, config: WorldModelConfig):
        super().__init__(config)
        self.config = config

        self.encoder = DreamerV2Encoder(config)
        self.dynamics = DreamerV2DynamicsPredictor(config.d_h, config.d_z)
        self.transition_layer = DreamerV2Transition(config.d_z, config.d_action, config.d_h)
        self.decoder = DreamerV2Decoder(config.d_h, config.d_z)
        self.reward_head = DreamerV2RewardHead(config.d_h, config.d_z)
        self.continue_head = DreamerV2ContinueHead(config.d_h, config.d_z)
        self.actor = DreamerV2Actor(config.d_h, config.d_z, config.d_action)
        self.critic = DreamerV2Critic(config.d_h, config.d_z)

        self._device = torch.device("cpu")

    def encode(self, obs: torch.Tensor, h_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if obs.dim() > 2:
            B = 1
        else:
            B = obs.shape[0] if obs.dim() > 1 else 1
        if h_prev.dim() == 1:
            h_prev = h_prev.unsqueeze(0)

        obs_flat = obs.reshape(B, -1) if obs.dim() > 2 else obs
        logits = self.encoder(obs_flat, h_prev[0])
        return logits, logits

    def dynamics(self, h: torch.Tensor) -> torch.Tensor:
        if h.dim() == 1:
            h = h.unsqueeze(0)
        return self.dynamics.mlp(h)

    def decode(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(h, z)

    def predict_reward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if h.dim() == 1:
            h = h.unsqueeze(0)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return self.reward_head(h, z)

    def predict_continue(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if h.dim() == 1:
            h = h.unsqueeze(0)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return self.continue_head(h, z)

    def actor_forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if h.dim() == 1:
            h = h.unsqueeze(0)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return self.actor(h, z)

    def critic_forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if h.dim() == 1:
            h = h.unsqueeze(0)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return self.critic(h, z)

    def initial_state(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.config.d_h, device=self._device)
        z_prior_logits = self.dynamics(h)
        z_onehot = self.sample_z(z_prior_logits, temperature=1.0, sample=False)
        z = z_onehot.view(batch_size, self.config.n_cat, self.config.n_cls)
        return h, z

    def named_parameters(self) -> Dict[str, torch.Tensor]:
        params = {}
        for name, param in self.named_parameters(full=True):
            params[name] = param
        return params

    def to(self, device: torch.device) -> "DreamerV2Adapter":
        super().to(device)
        self._device = device
        return self

    def eval(self) -> "DreamerV2Adapter":
        super().eval()
        return self

    def train(self, mode: bool = True) -> "DreamerV2Adapter":
        super().train(mode)
        return self
