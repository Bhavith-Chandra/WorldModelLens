"""Video Prediction Model Adapter.

Example of a non-RL world model that works with World Model Lens.

Video prediction models predict future frames from current frames.
They fit the "encode + dynamics + decode" pattern but have no
reward/value/action predictions.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from world_model_lens.backends.base_adapter import BaseModelAdapter, AdapterConfig, WorldModelCapabilities


class VideoEncoder(nn.Module):
    """Encode video frames into latent representation."""

    def __init__(self, d_obs: int, d_state: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_obs, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, d_state),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.flatten(1))


class VideoDecoder(nn.Module):
    """Decode latent state back to video frames."""

    def __init__(self, d_state: int, d_obs: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(d_state, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, d_obs),
        )

    def forward(self, x: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
        recon = self.decoder(x)
        return recon.view(*target_shape)


class VideoDynamics(nn.Module):
    """Predict next frame from current latent state."""

    def __init__(self, d_state: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_state, 256),
            nn.ReLU(),
            nn.Linear(256, d_state),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class VideoWorldModelAdapter(BaseModelAdapter):
    """World model adapter for video prediction."""

    def __init__(self, config: AdapterConfig, d_obs: int, frame_shape: Tuple[int, ...]):
        super().__init__(config)
        self.d_obs = d_obs
        self.frame_shape = frame_shape
        self.frame_size = 1
        for dim in frame_shape:
            self.frame_size *= dim

        self.encoder = VideoEncoder(self.frame_size, config.d_embed)
        self.dynamics_model = VideoDynamics(config.d_embed)
        self.decoder = VideoDecoder(config.d_embed, self.frame_size)
        self._capabilities = WorldModelCapabilities(
            has_decoder=True,
            has_reward_head=False,
            has_continue_head=False,
            has_actor=False,
            has_critic=False,
            uses_actions=False,
            is_rl_trained=False,
        )
        self._device = torch.device("cpu")

    @property
    def hook_point_names(self) -> List[str]:
        return ["state", "observation", "reconstruction", "dynamics"]

    def encode(
        self,
        obs: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode video frame into latent state."""
        del h_prev
        if obs.dim() == len(self.frame_shape):
            obs = obs.unsqueeze(0)
        z = self.encoder(obs)
        return z, z

    def transition(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Transition state (same as dynamics for this model)."""
        del h, action
        return self.dynamics_model(z)

    def dynamics(self, h: torch.Tensor) -> torch.Tensor:
        """Continuous video latents use the current state as the prior carrier."""
        return h if h.dim() > 1 else h.unsqueeze(0)

    def sample_z(
        self,
        logits_or_repr: torch.Tensor,
        temperature: float = 1.0,
        sample: bool = True,
    ) -> torch.Tensor:
        del temperature, sample
        return logits_or_repr

    def decode(self, h: torch.Tensor, z: torch.Tensor) -> Optional[torch.Tensor]:
        """Decode latent state to video frame."""
        del h
        if z.dim() == 1:
            z = z.unsqueeze(0)
        target_shape = (z.shape[0], *self.frame_shape)
        return self.decoder(z, target_shape)

    def forward(
        self,
        observation: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full video prediction forward pass."""
        h_prev = state if state is not None else self.initial_state(batch_size=1, device=observation.device)[0]
        posterior, obs_encoding = self.encode(observation, h_prev)
        prior = self.dynamics(h_prev)
        next_state = self.transition(h_prev, posterior, action)
        reconstruction = self.decode(next_state, posterior)
        return {
            "posterior": posterior,
            "prior": prior,
            "next_state": next_state,
            "reconstruction": reconstruction,
            "obs_encoding": obs_encoding,
        }

    def predict_next_frame(
        self,
        current_frame: torch.Tensor,
        n_frames: int = 1,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Predict multiple future frames."""
        h, z = self.initial_state(batch_size=1, device=current_frame.device)
        state, _ = self.encode(current_frame, h)
        predictions = []
        states = [state]

        for _ in range(n_frames):
            state = self.transition(state, state)
            pred_frame = self.decode(state, state)
            predictions.append(pred_frame)
            states.append(state)

        return torch.stack(predictions, dim=0), torch.stack(states, dim=0)

    def initial_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if device is None:
            device = self._device
        h = torch.zeros(batch_size, self.config.d_embed, device=device)
        z = torch.zeros(batch_size, self.config.d_embed, device=device)
        return h, z

    def to(self, device: torch.device) -> "VideoWorldModelAdapter":
        super().to(device)
        self._device = device
        return self
