"""DreamerV3 adapter implementation."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from world_model_lens.backends.base_adapter import WorldModelAdapter
from world_model_lens.core.config import WorldModelConfig


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric log transform."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog."""
    return torch.sign(x) * torch.expm1(torch.abs(x))


def twohot_encode(
    x: torch.Tensor, bins: int = 255, low: float = -20.0, high: float = 20.0
) -> torch.Tensor:
    """Two-hot encode a scalar value to a distribution over bins."""
    x = x.clamp(low, high)
    bin_width = (high - low) / (bins - 1)
    bin_idx = (x - low) / bin_width
    lo = bin_idx.floor().long()
    hi = (lo + 1).clamp(max=bins - 1)
    frac = bin_idx - lo.float()
    target = torch.zeros(*x.shape, bins, device=x.device, dtype=x.dtype)
    target[..., lo] = 1.0 - frac
    target[..., hi] = frac
    return target


def twohot_decode(
    probs: torch.Tensor, bins: int = 255, low: float = -20.0, high: float = 20.0
) -> torch.Tensor:
    """Decode two-hot distribution to scalar."""
    bin_width = (high - low) / (bins - 1)
    indices = torch.arange(bins, device=probs.device)
    values = low + indices.float() * bin_width
    return (probs * values).sum(dim=-1)


class MLP(nn.Module):
    def __init__(self, dims: List[int], activation: str = "silu"):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
        self.activation_name = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                if self.activation_name == "silu":
                    x = torch.nn.functional.silu(x)
                elif self.activation_name == "relu":
                    x = torch.nn.functional.relu(x)
                elif self.activation_name == "elu":
                    x = torch.nn.functional.elu(x)
        return x


class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 48, depth: int = 4):
        super().__init__()
        layers = []
        ch = in_channels
        for i in range(depth):
            out_ch = base_channels * (2**i)
            layers.append(nn.Conv2d(ch, out_ch, kernel_size=4, stride=2, padding=1))
            layers.append(nn.SiLU())
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
        self.mlp = MLP([d_obs, d_h, d_h], activation="silu")
        self.out_dim = d_h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class DreamerV3Encoder(nn.Module):
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        if config.encoder_type == "cnn":
            self.cnn = CNNEncoder(
                in_channels=3,
                base_channels=config.n_encoder_channels,
                depth=config.encoder_depth,
            )
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

        if obs_encoding.dim() == 1:
            obs_encoding = obs_encoding.unsqueeze(0)
        if h_prev.dim() == 1:
            h_prev = h_prev.unsqueeze(0)

        x = torch.cat([obs_encoding, h_prev], dim=-1)
        logits = self.fc(x)
        return logits


class DreamerV3DynamicsPredictor(nn.Module):
    def __init__(self, d_h: int, d_z: int):
        super().__init__()
        self.mlp = MLP([d_h, d_h, d_z], activation="silu")

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.mlp(h)


class DreamerV3Transition(nn.Module):
    def __init__(self, d_z: int, d_action: int, d_h: int):
        super().__init__()
        self.gru = nn.GRUCell(input_size=d_z + d_action, hidden_size=d_h)

    def forward(self, h: torch.Tensor, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, action], dim=-1)
        return self.gru(x, h)


class DreamerV3Decoder(nn.Module):
    def __init__(self, d_h: int, d_z: int, out_channels: int = 3):
        super().__init__()
        self.mlp = MLP([d_h + d_z, 1024, 4096], activation="silu")
        self.out_channels = out_channels

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, z], dim=-1)
        x = self.mlp(x)
        return x


class DreamerV3RewardHead(nn.Module):
    def __init__(self, d_h: int, d_z: int, bins: int = 255):
        super().__init__()
        self.mlp = MLP([d_h + d_z, 512, bins], activation="silu")
        self.bins = bins

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, z], dim=-1)
        return self.mlp(x)


class DreamerV3ContinueHead(nn.Module):
    def __init__(self, d_h: int, d_z: int):
        super().__init__()
        self.fc = nn.Linear(d_h + d_z, 1)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, z], dim=-1)
        return self.fc(x)


class DreamerV3Actor(nn.Module):
    def __init__(
        self, d_h: int, d_z: int, d_action: int, discrete: bool = True, n_classes: int = 32
    ):
        super().__init__()
        self.discrete = discrete
        self.n_classes = n_classes
        self.d_action = d_action
        if discrete:
            self.mlp = MLP([d_h + d_z, 512, d_action * n_classes], activation="silu")
        else:
            self.mlp = MLP([d_h + d_z, 512, d_action * 2], activation="silu")

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, z], dim=-1)
        out = self.mlp(x)
        if self.discrete:
            return out.view(*x.shape[:-1], self.d_action, self.n_classes)
        return out


class DreamerV3Critic(nn.Module):
    def __init__(self, d_h: int, d_z: int, bins: int = 255):
        super().__init__()
        self.mlp = MLP([d_h + d_z, 512, bins], activation="silu")
        self.bins = bins

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, z], dim=-1)
        return self.mlp(x)


class DreamerV3Adapter(WorldModelAdapter):
    """Full DreamerV3 implementation with RSSM."""

    def __init__(self, config: WorldModelConfig):
        super().__init__(config)
        self.config = config

        self.encoder = DreamerV3Encoder(config)
        self.dynamics_model = DreamerV3DynamicsPredictor(config.d_h, config.d_z)
        self.transition_layer = DreamerV3Transition(config.d_z, config.d_action, config.d_h)
        self.decoder = DreamerV3Decoder(config.d_h, config.d_z)
        self.reward_head = DreamerV3RewardHead(config.d_h, config.d_z)
        self.continue_head = DreamerV3ContinueHead(config.d_h, config.d_z)
        self.actor = DreamerV3Actor(config.d_h, config.d_z, config.d_action)
        self.critic = DreamerV3Critic(config.d_h, config.d_z)

        self._device = torch.device("cpu")

    @classmethod
    def from_checkpoint(
        cls, path: str, config: Optional[WorldModelConfig] = None
    ) -> "DreamerV3Adapter":
        """Load from checkpoint."""
        state_dict = torch.load(path, map_location="cpu")
        if config is None:
            config = cls.infer_config(state_dict)
        adapter = cls(config)
        adapter.load_state_dict(state_dict)
        return adapter

    @classmethod
    def infer_config(cls, state_dict: Dict) -> WorldModelConfig:
        """Infer config from state dict shapes."""
        d_z = 0
        d_h = 0
        d_action = 0

        for key in state_dict.keys():
            if "transition.gru" in key and "weight_ih" in key:
                d_z_plus_action = state_dict[key].shape[1]
            if "dynamics.mlp.layers.0.weight" in key:
                d_h = state_dict[key].shape[0]
            if "actor.mlp.layers.0.weight" in key and d_action == 0:
                d_z_plus_action = state_dict[key].shape[1] - d_h

        return WorldModelConfig(d_h=d_h or 512, d_z=d_z or 1024, d_action=d_action or 4)

    def encode(self, obs: torch.Tensor, h_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode observation to posterior logits."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if h_prev.dim() == 1:
            h_prev = h_prev.unsqueeze(0)
        if h_prev.shape[0] > 1:
            h_prev = h_prev[:1]

        obs_encoding = self.encoder(obs, h_prev[0])
        return obs_encoding, obs_encoding

    def transition(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """RSSM transition: GRU update from (h, z, action) -> h_next."""
        if h.dim() == 1:
            h = h.unsqueeze(0)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if z.dim() == 3:
            z = z.reshape(z.shape[0], -1)
        if action is None:
            action = torch.zeros(h.shape[0], self.config.d_action, device=h.device, dtype=h.dtype)
        elif action.dim() == 1:
            action = action.unsqueeze(0)
        return self.transition_layer(h, z, action)

    def dynamics(self, h: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute prior from hidden state only.

        DreamerV3 dynamics doesn't use actions for prior prediction.
        The action argument is accepted for API compatibility.
        """
        if h.dim() == 1:
            h = h.unsqueeze(0)
        return self.dynamics_model.mlp(h)

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

    def initial_state(
        self, batch_size: int = 1, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if device is None:
            device = self._device
        h = torch.zeros(batch_size, self.config.d_h, device=device)
        z_prior_logits = self.dynamics(h)
        z_onehot = self.sample_z(z_prior_logits, temperature=1.0, sample=False)
        z = z_onehot.view(batch_size, self.config.n_cat, self.config.n_cls)
        return h, z

    def named_parameters(self) -> Dict[str, torch.Tensor]:
        return dict(nn.Module.named_parameters(self))

    def sample_state(self, prior: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Sample from prior distribution.

        Args:
            prior: Prior logits [..., n_cat, n_cls]
            temperature: Sampling temperature

        Returns:
            Sampled state tensor
        """
        z_onehot = self.sample_z(prior, temperature=temperature, sample=True)
        if z_onehot.dim() > 1 and self.config.n_cat > 1:
            z_onehot = z_onehot.view(-1, self.config.n_cat, self.config.n_cls)
        return z_onehot

    def to(self, device: torch.device) -> "DreamerV3Adapter":
        super().to(device)
        self._device = device
        return self

    def eval(self) -> "DreamerV3Adapter":
        super().eval()
        return self

    def train(self, mode: bool = True) -> "DreamerV3Adapter":
        super().train(mode)
        return self
