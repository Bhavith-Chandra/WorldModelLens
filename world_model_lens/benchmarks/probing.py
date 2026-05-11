"""Probing benchmarks for world model interpretability evaluation."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import torch
import numpy as np

from world_model_lens.probing.prober import LatentProber, ProbeResult


def _get_device() -> torch.device:
    """Get the best available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class BenchmarkResult:
    """Result of a benchmark evaluation."""

    benchmark_name: str
    model_name: str
    scores: Dict[str, float]
    details: Dict[str, Any] = field(default_factory=dict)


class ProbingBenchmarkSuite:
    """Suite of probing benchmarks for world model evaluation."""

    def __init__(self, device: Optional[torch.device] = None):
        """Initialize benchmark suite."""
        self.device = device or _get_device()
        self.results: Dict[str, BenchmarkResult] = {}

    def reward_prediction(
        self,
        wm: Any,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
    ) -> float:
        """Benchmark: Predict reward from latents.

        Args:
            wm: HookedWorldModel.
            observations: Observation sequence.
            actions: Action sequence.
            rewards: Reward sequence.

        Returns:
            R² score for reward prediction.
        """
        traj, cache = wm.run_with_cache(observations, actions)

        try:
            z_posterior = cache["z_posterior"]
            if z_posterior.dim() == 3:
                z_flat = z_posterior.flatten(1)
            else:
                z_flat = z_posterior
        except:
            return 0.0

        if len(z_flat) != len(rewards):
            min_len = min(len(z_flat), len(rewards))
            z_flat = z_flat[:min_len]
            rewards = rewards[:min_len]

        labels = rewards.cpu().numpy() if isinstance(rewards, torch.Tensor) else rewards

        prober = LatentProber(seed=42)
        result = prober.train_probe(z_flat, labels, "reward", "z_posterior", probe_type="ridge")

        return result.r2 if result.r2 is not None else 0.0

    def value_prediction(
        self,
        wm: Any,
        observations: torch.Tensor,
        actions: torch.Tensor,
        values: torch.Tensor,
    ) -> float:
        """Benchmark: Predict value from latents.

        Args:
            wm: HookedWorldModel.
            observations: Observation sequence.
            actions: Action sequence.
            values: Value sequence.

        Returns:
            R² score for value prediction.
        """
        traj, cache = wm.run_with_cache(observations, actions)

        try:
            h = cache["h"]
            if h.dim() == 3:
                h_flat = h.flatten(1)
            else:
                h_flat = h
        except:
            return 0.0

        if len(h_flat) != len(values):
            min_len = min(len(h_flat), len(values))
            h_flat = h_flat[:min_len]
            values = values[:min_len]

        labels = values.cpu().numpy() if isinstance(values, torch.Tensor) else values

        prober = LatentProber(seed=42)
        result = prober.train_probe(h_flat, labels, "value", "h", probe_type="ridge")

        return result.r2 if result.r2 is not None else 0.0

    def action_prediction(
        self,
        wm: Any,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> float:
        """Benchmark: Predict action from latents.

        Args:
            wm: HookedWorldModel.
            observations: Observation sequence.
            actions: Action sequence.

        Returns:
            Accuracy for action prediction.
        """
        traj, cache = wm.run_with_cache(observations, actions)

        try:
            z_prior = cache["z_prior"]
            if z_prior.dim() == 3:
                z_flat = z_prior.flatten(1)
            else:
                z_flat = z_prior
        except:
            return 0.0

        if len(z_flat) != len(actions):
            min_len = min(len(z_flat), len(actions))
            z_flat = z_flat[:min_len]
            actions = actions[:min_len]

        if actions.dim() > 1 and actions.shape[1] > 1:
            discrete_actions = actions.argmax(dim=1).cpu().numpy()
        else:
            discrete_actions = (actions.squeeze() > 0).float().cpu().numpy()

        prober = LatentProber(seed=42)
        result = prober.train_probe(
            z_flat, discrete_actions, "action", "z_prior", probe_type="logistic"
        )

        return result.accuracy

    def state_classification(
        self,
        wm: Any,
        observations: torch.Tensor,
        actions: torch.Tensor,
        state_labels: np.ndarray,
    ) -> float:
        """Benchmark: Classify state from latents.

        Args:
            wm: HookedWorldModel.
            observations: Observation sequence.
            actions: Action sequence.
            state_labels: State labels per timestep.

        Returns:
            Accuracy for state classification.
        """
        traj, cache = wm.run_with_cache(observations, actions)

        try:
            z_posterior = cache["z_posterior"]
            if z_posterior.dim() == 3:
                z_flat = z_posterior.flatten(1)
            else:
                z_flat = z_posterior
        except:
            return 0.0

        if len(z_flat) != len(state_labels):
            min_len = min(len(z_flat), len(state_labels))
            z_flat = z_flat[:min_len]
            state_labels = state_labels[:min_len]

        prober = LatentProber(seed=42)
        result = prober.train_probe(
            z_flat, state_labels, "state", "z_posterior", probe_type="logistic"
        )

        return result.accuracy

    def temporal_position(
        self,
        wm: Any,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> float:
        """Benchmark: Predict timestep from latents.

        Args:
            wm: HookedWorldModel.
            observations: Observation sequence.
            actions: Action sequence.

        Returns:
            R² score for temporal position prediction.
        """
        traj, cache = wm.run_with_cache(observations, actions)

        try:
            h = cache["h"]
            if h.dim() == 3:
                h_flat = h.flatten(1)
            else:
                h_flat = h
        except:
            return 0.0

        timesteps = torch.arange(len(h_flat)).float()

        prober = LatentProber(seed=42)
        result = prober.train_probe(h_flat, timesteps.numpy(), "timestep", "h", probe_type="ridge")

        return result.r2 if result.r2 is not None else 0.0

    def run_full_suite(
        self,
        wm: Any,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: Optional[torch.Tensor] = None,
        values: Optional[torch.Tensor] = None,
        state_labels: Optional[np.ndarray] = None,
        model_name: str = "world_model",
    ) -> BenchmarkResult:
        """Run the full benchmark suite.

        Args:
            wm: HookedWorldModel.
            observations: Observation sequence.
            actions: Action sequence.
            rewards: Optional reward sequence.
            values: Optional value sequence.
            state_labels: Optional state labels.
            model_name: Name of the model being evaluated.

        Returns:
            BenchmarkResult with all scores.
        """
        scores = {}

        scores["action_prediction"] = self.action_prediction(wm, observations, actions)
        scores["temporal_position"] = self.temporal_position(wm, observations, actions)

        if rewards is not None:
            scores["reward_prediction"] = self.reward_prediction(wm, observations, actions, rewards)

        if values is not None:
            scores["value_prediction"] = self.value_prediction(wm, observations, actions, values)

        if state_labels is not None:
            scores["state_classification"] = self.state_classification(
                wm, observations, actions, state_labels
            )

        avg_score = sum(scores.values()) / len(scores) if scores else 0.0
        scores["average"] = avg_score

        result = BenchmarkResult(
            benchmark_name="probing_suite",
            model_name=model_name,
            scores=scores,
        )

        self.results[model_name] = result
        return result


def compare_models(results: Dict[str, BenchmarkResult]) -> Dict[str, Any]:
    """Compare benchmark results across models.

    Args:
        results: Dict of model_name -> BenchmarkResult.

    Returns:
        Comparison summary.
    """
    comparison = {
        "models": list(results.keys()),
        "metrics": {},
    }

    all_metrics = set()
    for result in results.values():
        all_metrics.update(result.scores.keys())

    for metric in all_metrics:
        comparison["metrics"][metric] = {}
        for model_name, result in results.items():
            if metric in result.scores:
                comparison["metrics"][metric][model_name] = result.scores[metric]

    return comparison
