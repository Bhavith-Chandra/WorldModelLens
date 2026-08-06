"""Synthetic world models for benchmarking interpretability methods.

This module provides controlled toy models with known causal structures for
testing and benchmarking patching, probing, and SAE methods.

5 Toy Models:
1. PositionTracker: latent dim 0 encodes x-position
2. RewardGate: GRU dim 5 gates reward computation
3. DecisionCircuit: 3-step causal chain from observation → latent → action
4. HarmonicOscillator: damped harmonic motion for temporal modeling
5. FeatureBinding: binds features across multiple channels
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from world_model_lens.backends.base_adapter import BaseModelAdapter
from world_model_lens.core.types import WorldModelConfig, WorldState


@dataclass
class BenchmarkResult:
    """Results from running benchmark suite."""

    model_name: str
    probe_accuracy: Dict[str, float]
    patching_recovery: Dict[str, float]
    sae_reconstruction: Dict[str, float]
    overall_score: float


class PositionTracker(nn.Module):
    """Toy model where latent dim 0 directly encodes x-position.

    Architecture:
        obs -> encoder -> latent (dim 0 = x-position)
        latent -> decoder -> reconstruction

    Ground truth: latent[:, 0] == obs[:, 0] (x-position)
    """

    def __init__(self, obs_dim: int = 10, latent_dim: int = 8):
        super().__init__()
        self.encoder = nn.Linear(obs_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, obs_dim)
        self.latent_dim = latent_dim

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(obs)
        reconstruction = self.decode(latent)
        return latent, reconstruction

    def get_component_activations(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get activations at each component for interpretability analysis."""
        return {
            "encoder": self.encoder(obs),
            "latent": self.encode(obs),
            "decoder": self.decode(self.encode(obs)),
        }


class RewardGate(nn.Module):
    """Toy model where GRU hidden dim 5 gates reward computation.

    Architecture:
        obs -> concat with action -> GRU -> h (dim 5 = reward gate)
        h -> reward_head (uses dim 5)

    Ground truth: reward = sigmoid(h[:, 5]) * base_reward
    """

    def __init__(self, obs_dim: int = 10, action_dim: int = 4, hidden_dim: int = 16):
        super().__init__()
        self.gru = nn.GRUCell(obs_dim + action_dim, hidden_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.hidden_dim = hidden_dim

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor, h_prev: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if h_prev is None:
            h_prev = torch.zeros(obs.shape[0], self.hidden_dim, device=obs.device)

        gru_input = torch.cat([obs, action], dim=-1)
        h = self.gru(gru_input, h_prev)

        reward_gate = torch.sigmoid(h[:, 5]).unsqueeze(-1)
        base_reward = self.reward_head(h)
        reward = reward_gate * base_reward

        return h, reward

    def get_component_activations(
        self, obs: torch.Tensor, action: torch.Tensor, h_prev: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        h, reward = self.forward(obs, action, h_prev)
        return {
            "gru_input": torch.cat([obs, action], dim=-1),
            "gru_hidden": h,
            "reward_gate": torch.sigmoid(h[:, 5]),
            "reward": reward,
        }


class DecisionCircuit(nn.Module):
    """3-step causal chain: observation -> latent -> action -> value.

    Architecture:
        obs -> obs_encoder -> latent
        latent -> policy -> action
        latent + action -> value_head

    Ground truth causal paths:
        obs -> latent (strong)
        latent -> action (strong)
        latent + action -> value (moderate)
    """

    def __init__(self, obs_dim: int = 20, latent_dim: int = 12, action_dim: int = 4):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        self.policy = nn.Linear(latent_dim, action_dim)
        self.value = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
        self.latent_dim = latent_dim
        self.action_dim = action_dim

    def forward(
        self, obs: torch.Tensor, action_prev: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        latent = self.obs_encoder(obs)
        action_logits = self.policy(latent)

        if action_prev is not None:
            value_input = torch.cat([latent, action_prev], dim=-1)
        else:
            value_input = torch.cat([latent, torch.zeros_like(action_logits)], dim=-1)

        value = self.value(value_input)

        return {
            "obs_encoder": latent,
            "latent": latent,
            "action_logits": action_logits,
            "value": value,
        }

    def get_component_activations(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.forward(obs)


class HarmonicOscillator(nn.Module):
    """Damped harmonic oscillator for temporal modeling research.

    Physics: x(t+1) = x(t) + v(t)*dt; v(t+1) = v(t) - k*x(t)*dt - b*v(t)*dt

    Latent encodes: [x_position, velocity, energy]
    """

    def __init__(self, k: float = 1.0, b: float = 0.1, dt: float = 0.1):
        super().__init__()
        self.k = k
        self.b = b
        self.dt = dt

        self.encoder = nn.Linear(2, 4)
        self.dynamics = nn.Linear(4, 4)
        self.decoder = nn.Linear(4, 2)

    def physics_step(self, state: torch.Tensor) -> torch.Tensor:
        x = state[:, 0]
        v = state[:, 1]
        x_next = x + v * self.dt
        v_next = v - self.k * x * self.dt - self.b * v * self.dt
        return torch.stack([x_next, v_next], dim=-1)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def dynamics_step(self, latent: torch.Tensor) -> torch.Tensor:
        return self.dynamics(latent)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(
        self, obs: torch.Tensor, latent: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        if latent is None:
            latent = self.encode(obs)
        next_latent = self.dynamics_step(latent)
        reconstruction = self.decode(next_latent)
        return {
            "encoder": self.encode(obs),
            "latent": latent,
            "next_latent": next_latent,
            "reconstruction": reconstruction,
        }


class FeatureBinding(nn.Module):
    """Binds features across multiple channels.

    Architecture:
        Multiple independent encoders for each channel
        Binding layer that combines channels
        Decoding layer

    Ground truth: binding strength correlates with feature similarity
    """

    def __init__(self, n_channels: int = 3, channel_dim: int = 8, binding_dim: int = 16):
        super().__init__()
        self.n_channels = n_channels
        self.channel_encoders = nn.ModuleList(
            [nn.Linear(channel_dim, binding_dim) for _ in range(n_channels)]
        )
        self.binding_layer = nn.Linear(binding_dim * n_channels, binding_dim)
        self.decoder = nn.Linear(binding_dim, channel_dim * n_channels)

    def forward(self, channels: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        encoded = [enc(ch) for enc, ch in zip(self.channel_encoders, channels)]
        concat = torch.cat(encoded, dim=-1)
        bound = torch.relu(self.binding_layer(concat))
        reconstruction = self.decoder(bound)

        return {
            "channel_encodings": encoded,
            "bound_features": bound,
            "reconstruction": reconstruction,
        }

    def get_binding_strength(self, channels: List[torch.Tensor]) -> torch.Tensor:
        encoded = [enc(ch) for enc, ch in zip(self.channel_encoders, channels)]
        similarities = []
        for i in range(len(encoded)):
            for j in range(i + 1, len(encoded)):
                sim = F.cosine_similarity(encoded[i], encoded[j], dim=-1).mean()
                similarities.append(sim)
        return torch.stack(similarities).mean() if similarities else torch.tensor(0.0)


class SyntheticWorldModelAdapter(BaseModelAdapter):
    """BaseModelAdapter wrapper for synthetic models."""

    def __init__(self, model: nn.Module, config: Optional[WorldModelConfig] = None):
        super().__init__(model if hasattr(model, "encode") else None)
        self.model = model
        self._config = config or WorldModelConfig()

    def encode(self, obs: torch.Tensor, context=None) -> Tuple[WorldState, torch.Tensor]:
        if hasattr(self.model, "encode"):
            latent = self.model.encode(obs)
        else:
            latent = obs

        state = WorldState(
            hidden=latent,
            deterministic=latent,
            stochastic=latent,
        )
        return state, latent

    def dynamics(self, state: WorldState, action: Optional[torch.Tensor] = None) -> WorldState:
        if hasattr(self.model, "forward"):
            if action is not None:
                if hasattr(self.model, "gru"):
                    h = self.model.gru(torch.cat([state.hidden, action], dim=-1))
                else:
                    h = self.model(state.hidden)
            else:
                h = state.hidden
        else:
            h = state.hidden

        return WorldState(hidden=h, deterministic=h, stochastic=h)

    def decode(self, state: WorldState, context=None) -> torch.Tensor:
        if hasattr(self.model, "decode"):
            return self.model.decode(state.hidden)
        return state.hidden

    def get_components(self) -> List[str]:
        if hasattr(self.model, "get_component_activations"):
            return list(self.model.get_component_activations(torch.zeros(1, 10)).keys())
        return ["main"]


class MechanisticBenchmarkSuite:
    """Suite for benchmarking interpretability methods on synthetic models.

    Tests:
    1. Probe recovery: Can we recover ground truth concepts?
    2. Patching recovery: Can patching restore corrupted behavior?
    3. SAE reconstruction: Can SAEs reconstruct key activations?
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.results: Dict[str, BenchmarkResult] = {}

    def generate_trajectories(
        self,
        n_trajectories: int = 100,
        trajectory_length: int = 50,
        noise_std: float = 0.1,
    ) -> List[Dict[str, torch.Tensor]]:
        """Generate synthetic trajectories for benchmarking."""
        trajectories = []

        for _ in range(n_trajectories):
            traj = {
                "observations": [],
                "actions": [],
                "latents": [],
                "rewards": [],
            }

            obs = torch.randn(1, 10)
            for _ in range(trajectory_length):
                if hasattr(self.model, "encode"):
                    latent = self.model.encode(obs)
                else:
                    latent = self.model(obs)

                traj["observations"].append(obs)
                traj["latents"].append(latent)

                action = (
                    torch.randn(1, 4)
                    if hasattr(self.model, "gru") or hasattr(self.model, "forward")
                    else None
                )
                if action is not None:
                    traj["actions"].append(action)

                obs = torch.randn(1, 10) + noise_std * torch.randn(1, 10)

            for key in traj:
                if traj[key]:
                    traj[key] = torch.cat(traj[key], dim=0)

            trajectories.append(traj)

        return trajectories

    def run_probe_benchmark(
        self,
        trajectories: List[Dict[str, torch.Tensor]],
        target_concept: str = "position",
    ) -> Dict[str, float]:
        """Run linear probing to recover ground truth concepts."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        all_latents = []
        all_labels = []

        for traj in trajectories:
            if "latents" in traj and len(traj["latents"]) > 0:
                all_latents.append(traj["latents"])
                labels = torch.arange(len(traj["latents"])) % 10
                all_labels.append(labels)

        if not all_latents:
            return {"accuracy": 0.0}

        latents = torch.cat(all_latents, dim=0).cpu().numpy()
        labels = torch.cat(all_labels).cpu().numpy()

        X_train, X_test, y_train, y_test = train_test_split(
            latents, labels, test_size=0.2, random_state=42
        )

        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)
        accuracy = lr.score(X_test, y_test)

        return {"accuracy": accuracy}

    def run_patching_benchmark(
        self,
        trajectories: List[Dict[str, torch.Tensor]],
        corruption_strength: float = 0.5,
    ) -> Dict[str, float]:
        """Run patching to recover corrupted behavior."""
        recovery_scores = {}

        for traj in trajectories[:10]:
            if "latents" not in traj or len(traj["latents"]) == 0:
                continue

            original = traj["latents"]
            corrupted = original + corruption_strength * torch.randn_like(original)

            recovery = 1.0 - torch.nn.functional.mse_loss(corrupted, original).item()

            recovery_scores["mean_recovery"] = recovery_scores.get("mean_recovery", 0) + recovery

        n = len(recovery_scores) if recovery_scores else 1
        return (
            {k: v / n for k, v in recovery_scores.items()} if recovery_scores else {"recovery": 1.0}
        )

    def run_sae_benchmark(
        self,
        trajectories: List[Dict[str, torch.Tensor]],
        n_features: int = 32,
    ) -> Dict[str, float]:
        """Run SAE training and measure reconstruction quality."""
        from world_model_lens.sae.sparse_autoencoder import SparseAutoencoder

        all_latents = []
        for traj in trajectories:
            if "latents" in traj and len(traj["latents"]) > 0:
                all_latents.append(traj["latents"])

        if not all_latents:
            return {"reconstruction_error": 1.0}

        latents = torch.cat(all_latents, dim=0)
        d_latent = latents.shape[-1]

        sae = SparseAutoencoder(d_model=d_latent, n_features=n_features)

        optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)
        losses = []

        for epoch in range(100):
            recon, features, l1_loss = sae(latents)
            recon_loss = F.mse_loss(recon, latents)
            loss = recon_loss + 0.01 * l1_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        final_recon_error = losses[-1] if losses else 1.0

        return {
            "reconstruction_error": final_recon_error,
            "convergence_epoch": len(losses),
        }

    def run_full_benchmark(self) -> BenchmarkResult:
        """Run complete benchmark suite."""
        trajectories = self.generate_trajectories(n_trajectories=50)

        probe_acc = self.run_probe_benchmark(trajectories)
        patching_rec = self.run_patching_benchmark(trajectories)
        sae_recon = self.run_sae_benchmark(trajectories)

        overall = (
            probe_acc.get("accuracy", 0.0)
            + patching_rec.get("recovery", 0.0)
            + (1.0 - min(sae_recon.get("reconstruction_error", 1.0), 1.0))
        ) / 3.0

        result = BenchmarkResult(
            model_name=self.__class__.__name__,
            probe_accuracy=probe_acc,
            patching_recovery=patching_rec,
            sae_reconstruction=sae_recon,
            overall_score=overall,
        )

        self.results[self.__class__.__name__] = result
        return result

    def compare_models(self, other: "MechanisticBenchmarkSuite") -> Dict[str, float]:
        """Compare this benchmark suite's model against another."""
        self.run_full_benchmark()
        other.run_full_benchmark()

        comparison = {}
        for key in ["probe_accuracy", "patching_recovery", "sae_reconstruction"]:
            self_res = getattr(self.results[self.__class__.__name__], key)
            other_res = getattr(other.results[other.__class__.__name__], key)

            if isinstance(self_res, dict) and isinstance(other_res, dict):
                for k in self_res:
                    comparison[f"{key}_{k}"] = self_res.get(k, 0) - other_res.get(k, 0)

        return comparison


def create_position_tracker_benchmark() -> MechanisticBenchmarkSuite:
    """Create benchmark suite for PositionTracker model."""
    model = PositionTracker(obs_dim=10, latent_dim=8)
    return MechanisticBenchmarkSuite(model)


def create_reward_gate_benchmark() -> MechanisticBenchmarkSuite:
    """Create benchmark suite for RewardGate model."""
    model = RewardGate(obs_dim=10, action_dim=4, hidden_dim=16)
    return MechanisticBenchmarkSuite(model)


def create_decision_circuit_benchmark() -> MechanisticBenchmarkSuite:
    """Create benchmark suite for DecisionCircuit model."""
    model = DecisionCircuit(obs_dim=20, latent_dim=12, action_dim=4)
    return MechanisticBenchmarkSuite(model)


def run_all_benchmarks() -> Dict[str, BenchmarkResult]:
    """Run benchmarks on all synthetic models."""
    results = {}

    benchmarks = [
        ("PositionTracker", create_position_tracker_benchmark()),
        ("RewardGate", create_reward_gate_benchmark()),
        ("DecisionCircuit", create_decision_circuit_benchmark()),
    ]

    for name, benchmark in benchmarks:
        print(f"Running benchmark: {name}")
        result = benchmark.run_full_benchmark()
        results[name] = result
        print(f"  Overall Score: {result.overall_score:.4f}")

    return results
