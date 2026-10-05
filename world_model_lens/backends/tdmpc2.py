"""TD-MPC2 implicit world model architecture.

The implementation mirrors the source model's defining components for
single-task state observations: a SimNorm encoder and dynamics model,
action-conditioned discrete reward prediction, a squashed Gaussian policy
prior, and an ensemble of distributional Q-functions.

Reference implementation: https://github.com/nicklashansen/tdmpc2
"""

import copy
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as functional

from world_model_lens.backends.base_adapter import (
    AdapterConfig,
    BaseModelAdapter,
    WorldModelCapabilities,
)
from world_model_lens.backends.registry import register
from world_model_lens.core.types import WorldModelFamily


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))


class SimNorm(nn.Module):
    """Simplicial normalization over fixed-size groups of latent features."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % self.dim != 0:
            raise ValueError(
                f"SimNorm input dimension {x.shape[-1]} must be divisible by {self.dim}"
            )
        shape = x.shape
        x = x.reshape(*shape[:-1], -1, self.dim)
        return functional.softmax(x, dim=-1).reshape(shape)


class NormedLinear(nn.Linear):
    """Linear layer followed by LayerNorm, dropout, and an activation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        dropout: float = 0.0,
        activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.activation = activation if activation is not None else nn.Mish()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = functional.linear(input, self.weight, self.bias)
        return self.activation(self.norm(self.dropout(output)))


def mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    *,
    hidden_layers: int = 2,
    dropout: float = 0.0,
    output_activation: Optional[nn.Module] = None,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = in_dim
    for layer_index in range(hidden_layers):
        layers.append(
            NormedLinear(
                current_dim,
                hidden_dim,
                dropout=dropout if layer_index == 0 else 0.0,
            )
        )
        current_dim = hidden_dim
    if output_activation is None:
        layers.append(nn.Linear(current_dim, out_dim))
    else:
        layers.append(NormedLinear(current_dim, out_dim, activation=output_activation))
    return nn.Sequential(*layers)


class StateEncoder(nn.Module):
    """TD-MPC2 state encoder ending in a SimNorm latent."""

    def __init__(self, obs_dim: int, enc_dim: int, latent_dim: int, simnorm_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            NormedLinear(obs_dim, enc_dim),
            NormedLinear(enc_dim, latent_dim, activation=SimNorm(simnorm_dim)),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class QEnsemble(nn.Module):
    """Independent ensemble of distributional Q-functions."""

    def __init__(
        self,
        num_q: int,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_bins: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.members = nn.ModuleList(
            [
                mlp(
                    latent_dim + action_dim,
                    hidden_dim,
                    num_bins,
                    dropout=dropout,
                )
                for _ in range(num_q)
            ]
        )

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        features = torch.cat((z, action), dim=-1)
        return torch.stack([member(features) for member in self.members], dim=0)


class TDMPC2Adapter(BaseModelAdapter):
    """Single-task, state-observation TD-MPC2 world model adapter.

    The adapter intentionally implements the learned model rather than the
    replay, optimization, and MPPI planning loops. Use ``reward``, ``policy``,
    and ``Q`` for the native source-aligned interfaces.
    """

    value_bins: torch.Tensor

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__(config)
        self.config = config
        self.latent_dim = config.d_h
        self.enc_dim = getattr(config, "tdmpc2_enc_dim", 256)
        self.mlp_dim = getattr(config, "tdmpc2_mlp_dim", 512)
        self.simnorm_dim = getattr(config, "tdmpc2_simnorm_dim", 8)
        self.num_bins = getattr(config, "tdmpc2_num_bins", 101)
        self.num_q = getattr(config, "tdmpc2_num_q", 5)
        self.q_dropout = getattr(config, "tdmpc2_dropout", 0.01)
        self.vmin = getattr(config, "tdmpc2_vmin", -10.0)
        self.vmax = getattr(config, "tdmpc2_vmax", 10.0)
        self.log_std_min = getattr(config, "tdmpc2_log_std_min", -10.0)
        self.log_std_max = getattr(config, "tdmpc2_log_std_max", 2.0)
        self.episodic = getattr(config, "tdmpc2_episodic", False)

        if self.latent_dim % self.simnorm_dim != 0:
            raise ValueError(
                "TD-MPC2 latent dimension must be divisible by tdmpc2_simnorm_dim: "
                f"{self.latent_dim} % {self.simnorm_dim} != 0"
            )

        self.encoder = StateEncoder(
            config.d_obs,
            self.enc_dim,
            self.latent_dim,
            self.simnorm_dim,
        )
        self.dynamics_model = mlp(
            self.latent_dim + config.d_action,
            self.mlp_dim,
            self.latent_dim,
            output_activation=SimNorm(self.simnorm_dim),
        )
        self.reward_model = mlp(
            self.latent_dim + config.d_action,
            self.mlp_dim,
            self.num_bins,
        )
        self.policy_model = mlp(
            self.latent_dim,
            self.mlp_dim,
            2 * config.d_action,
        )
        self.q_ensemble = QEnsemble(
            self.num_q,
            self.latent_dim,
            config.d_action,
            self.mlp_dim,
            self.num_bins,
            self.q_dropout,
        )
        self.target_q_ensemble = copy.deepcopy(self.q_ensemble)
        self.target_q_ensemble.requires_grad_(False)
        self.termination_model = (
            mlp(self.latent_dim, self.mlp_dim, 1) if self.episodic else None
        )

        # The released implementation starts reward and Q logits at zero.
        reward_output = self.reward_model[-1]
        if not isinstance(reward_output, nn.Linear):
            raise TypeError("TD-MPC2 reward model must end in a linear layer")
        nn.init.zeros_(reward_output.weight)
        nn.init.zeros_(reward_output.bias)
        for member in self.q_ensemble.members:
            if not isinstance(member, nn.Sequential):
                raise TypeError("TD-MPC2 Q ensemble members must be sequential models")
            q_output = member[-1]
            if not isinstance(q_output, nn.Linear):
                raise TypeError("TD-MPC2 Q model must end in a linear layer")
            nn.init.zeros_(q_output.weight)
            nn.init.zeros_(q_output.bias)
        self.target_q_ensemble.load_state_dict(self.q_ensemble.state_dict())

        self.register_buffer(
            "value_bins",
            torch.linspace(self.vmin, self.vmax, self.num_bins),
        )
        self._capabilities = WorldModelCapabilities(
            has_decoder=False,
            has_reward_head=True,
            has_continue_head=self.episodic,
            has_actor=True,
            has_critic=True,
            uses_actions=True,
            is_rl_trained=True,
        )
        self._device = torch.device("cpu")

    @property
    def world_model_family(self) -> WorldModelFamily:
        return WorldModelFamily.TD_MPC

    @property
    def hook_point_names(self) -> list[str]:
        return [
            "encoder",
            "dynamics_model",
            "reward_model",
            "policy_model",
            "q_ensemble",
            "target_q_ensemble",
        ]

    def decode_distribution(self, logits: torch.Tensor) -> torch.Tensor:
        """Decode TD-MPC2 categorical regression logits to scalar values."""
        expectation = (logits.softmax(dim=-1) * self.value_bins).sum(dim=-1, keepdim=True)
        return symexp(expectation)

    def encode(
        self, obs: torch.Tensor, h_prev: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del h_prev
        if obs.dim() > 2:
            obs = obs.flatten(start_dim=1)
        elif obs.dim() == 1:
            obs = obs.unsqueeze(0)
        z = self.encoder(obs)
        return z, z

    def next(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict the next SimNorm latent from current latent and action."""
        return self.dynamics_model(torch.cat((z, action), dim=-1))

    def reward(self, z: torch.Tensor, action: torch.Tensor, *, decode: bool = False) -> torch.Tensor:
        """Return action-conditioned reward logits or their scalar decode."""
        logits = self.reward_model(torch.cat((z, action), dim=-1))
        return self.decode_distribution(logits) if decode else logits

    def policy(
        self,
        z: torch.Tensor,
        *,
        sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample from the tanh-squashed Gaussian policy prior."""
        raw_mean, raw_log_std = self.policy_model(z).chunk(2, dim=-1)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            torch.tanh(raw_log_std) + 1.0
        )
        pre_tanh = raw_mean
        if sample:
            pre_tanh = raw_mean + torch.randn_like(raw_mean) * log_std.exp()
        action = torch.tanh(pre_tanh)
        mean = torch.tanh(raw_mean)
        return action, mean, log_std

    def q_logits(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        *,
        target: bool = False,
    ) -> torch.Tensor:
        ensemble = self.target_q_ensemble if target else self.q_ensemble
        return ensemble(z, action)

    def q(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        *,
        return_type: Literal["min", "avg", "all"] = "min",
        target: bool = False,
    ) -> torch.Tensor:
        """Evaluate the Q ensemble using TD-MPC2's two-member reduction."""
        if return_type not in {"min", "avg", "all"}:
            raise ValueError(f"unsupported Q return_type: {return_type}")
        values = self.decode_distribution(self.q_logits(z, action, target=target))
        if return_type == "all":
            return values
        if self.num_q > 2:
            indices = torch.randperm(self.num_q, device=values.device)[:2]
            values = values[indices]
        if return_type == "min":
            return values.min(dim=0).values
        return values.mean(dim=0)

    @torch.no_grad()
    def soft_update_target_q(self, tau: float = 0.01) -> None:
        """Polyak-update target Q parameters from the online ensemble."""
        for target, online in zip(
            self.target_q_ensemble.parameters(),
            self.q_ensemble.parameters(),
        ):
            target.lerp_(online, tau)

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
            action = torch.zeros(
                z.shape[0],
                self.config.d_action,
                device=z.device,
                dtype=z.dtype,
            )
        elif action.dim() == 1:
            action = action.unsqueeze(0)
        return self.next(z, action)

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

    def _adapter_action(self, state: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        if candidate.shape[-1] == self.config.d_action:
            return candidate
        return self.policy(state, sample=False)[0]

    def predict_reward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if h.dim() == 1:
            h = h.unsqueeze(0)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        action = self._adapter_action(h, z)
        return self.reward(h, action)

    def predict_continue(self, h: torch.Tensor, z: torch.Tensor) -> Optional[torch.Tensor]:
        del z
        if self.termination_model is None:
            return None
        if h.dim() == 1:
            h = h.unsqueeze(0)
        # Negating the termination logit produces the corresponding continue logit.
        return -self.termination_model(h)

    def actor_forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        del z
        if h.dim() == 1:
            h = h.unsqueeze(0)
        _, mean, log_std = self.policy(h, sample=False)
        return torch.cat((mean, log_std), dim=-1)

    def critic_forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if h.dim() == 1:
            h = h.unsqueeze(0)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        action = self._adapter_action(h, z)
        return self.q(h, action, return_type="min")

    def initial_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = device or self._device
        # A uniform point on every simplex is a valid neutral SimNorm state.
        z = torch.full(
            (batch_size, self.latent_dim),
            1.0 / self.simnorm_dim,
            device=device,
        )
        return z, z.clone()

    def to(self, device: torch.device) -> "TDMPC2Adapter":
        super().to(device=device)
        self._device = device
        return self

    def eval(self) -> "TDMPC2Adapter":
        super().eval()
        return self

    def train(self, mode: bool = True) -> "TDMPC2Adapter":
        super().train(mode)
        self.target_q_ensemble.eval()
        return self

    def list_hookable_points(self) -> list[str]:
        points = list(self.hook_point_names)
        points.extend(name for name, _ in self.named_modules() if name)
        return list(dict.fromkeys(points))


register(
    name="tdmpc2",
    family=WorldModelFamily.TD_MPC,
    description="TD-MPC2 implicit world model",
    supports_rl=True,
    supports_video=False,
    supports_planning=True,
)(TDMPC2Adapter)
