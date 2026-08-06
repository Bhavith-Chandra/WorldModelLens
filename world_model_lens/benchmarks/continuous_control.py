"""Continuous Control Benchmark.

Canonical experiment showing:
1. Long-term prediction degradation
2. Uncertainty growth over horizons
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch
import numpy as np


@dataclass
class ContinuousControlResult:
    """Results from continuous control benchmark."""

    degradation_rate: float
    uncertainty_growth_rate: float
    horizon_effects: Dict[int, float]
    prediction_errors: List[float]
    uncertainty_by_horizon: Dict[int, float]
    metadata: Dict[str, Any]


class ContinuousControlBenchmark:
    """Continuous control interpretability benchmark.

    Demonstrates:
    - Prediction quality degrades over longer horizons
    - Uncertainty grows with rollout length
    - Specific dimensions capture uncertainty

    Example:
        benchmark = ContinuousControlBenchmark(world_model)

        results = benchmark.run(
            horizon=50,
            num_trajectories=20,
        )

        print(f"Degradation rate: {results.degradation_rate}")
        print(f"Uncertainty growth: {results.uncertainty_growth_rate}")
    """

    def __init__(self, world_model: Any):
        """Initialize benchmark.

        Args:
            world_model: HookedWorldModel instance
        """
        self.wm = world_model

    def generate_control_trajectories(
        self,
        num_trajectories: int = 20,
        trajectory_length: int = 100,
        obs_dim: int = 17,
        action_dim: int = 6,
    ) -> List[torch.Tensor]:
        """Generate continuous control trajectories (Mujoco-style).

        Args:
            num_trajectories: Number of trajectories
            trajectory_length: Length of each trajectory
            obs_dim: Observation dimension
            action_dim: Action dimension

        Returns:
            List of observation tensors
        """
        trajectories = []

        for _ in range(num_trajectories):
            # Random walk in observation space
            obs = torch.randn(trajectory_length, obs_dim)

            # Smooth with temporal coherence
            for t in range(1, trajectory_length):
                obs[t] = 0.9 * obs[t - 1] + 0.1 * obs[t]

            trajectories.append(obs)

        return trajectories

    def measure_prediction_degradation(
        self,
        trajectories: List[torch.Tensor],
    ) -> tuple[float, List[float]]:
        """Measure how prediction error grows with horizon.

        Args:
            trajectories: List of trajectory tensors

        Returns:
            Tuple of (degradation_rate, errors_by_horizon)
        """
        errors_by_horizon = {}

        for horizon in [5, 10, 20, 50]:
            errors = []

            for traj in trajectories[:10]:
                if len(traj) < horizon:
                    continue

                # Use first part to predict
                obs_short = traj[:horizon]

                # Run model
                pred_traj, _ = self.wm.run_with_cache(obs_short.unsqueeze(0))

                # Compare final states
                if len(pred_traj.states) > 0:
                    final_pred = pred_traj.states[-1].state
                    final_true = (
                        obs_short[-1, : final_pred.shape[-1]]
                        if final_pred.shape[-1] <= obs_short.shape[1]
                        else final_pred.squeeze()
                    )

                    if final_pred.shape[-1] <= obs_short.shape[1]:
                        error = torch.nn.functional.mse_loss(
                            final_pred, final_true[: final_pred.shape[-1]]
                        ).item()
                        errors.append(error)

            if errors:
                errors_by_horizon[horizon] = np.mean(errors)

        # Compute degradation rate (linear fit)
        horizons = sorted(errors_by_horizon.keys())
        errors = [errors_by_horizon[h] for h in horizons]

        if len(horizons) > 1:
            # Linear regression
            x = np.array(horizons)
            y = np.array(errors)
            slope = np.polyfit(x, y, 1)[0]
            degradation_rate = float(slope)
        else:
            degradation_rate = 0.0

        return degradation_rate, errors_by_horizon

    def measure_uncertainty_growth(
        self,
        trajectories: List[torch.Tensor],
        num_samples: int = 5,
    ) -> tuple[float, Dict[int, float]]:
        """Measure how uncertainty grows with horizon.

        Uses ensemble of rollouts with noise to estimate uncertainty.

        Args:
            trajectories: List of trajectory tensors
            num_samples: Number of samples for uncertainty

        Returns:
            Tuple of (growth_rate, uncertainty_by_horizon)
        """
        uncertainty_by_horizon = {}

        for horizon in [5, 10, 20, 50]:
            final_states = []

            for traj in trajectories[:5]:
                if len(traj) < horizon:
                    continue

                obs = traj[:horizon]

                # Multiple rollouts with noise
                for _ in range(num_samples):
                    # Add noise to latent
                    def add_noise(tensor, ctx):
                        noise = torch.randn_like(tensor) * 0.1
                        return tensor + noise

                    try:
                        pred_traj, _ = self.wm.run_with_advanced_hooks(
                            obs.unsqueeze(0),
                            hook_specs={"z[0:32]": add_noise},
                        )

                        if pred_traj.states:
                            final_states.append(pred_traj.states[-1].state)
                    except Exception:
                        pass

            # Compute variance
            if len(final_states) > 1:
                final_tensor = torch.stack([s.flatten() for s in final_states])
                variance = final_tensor.var(dim=0).mean().item()
                uncertainty_by_horizon[horizon] = variance

        # Compute growth rate
        horizons = sorted(uncertainty_by_horizon.keys())
        uncertainties = [uncertainty_by_horizon[h] for h in horizons]

        if len(horizons) > 1:
            x = np.array(horizons)
            y = np.array(uncertainties)
            slope = np.polyfit(x, y, 1)[0]
            growth_rate = float(slope)
        else:
            growth_rate = 0.0

        return growth_rate, uncertainty_by_horizon

    def run(
        self,
        horizon: int = 50,
        num_trajectories: int = 20,
    ) -> ContinuousControlResult:
        """Run full continuous control benchmark.

        Args:
            horizon: Maximum horizon to test
            num_trajectories: Number of trajectories

        Returns:
            ContinuousControlResult with all findings
        """
        # Generate trajectories
        trajectories = self.generate_control_trajectories(
            num_trajectories=num_trajectories,
            trajectory_length=horizon + 10,
        )

        # Measure degradation
        degradation_rate, prediction_errors = self.measure_prediction_degradation(trajectories)

        # Measure uncertainty growth
        uncertainty_growth, uncertainty_by_horizon = self.measure_uncertainty_growth(trajectories)

        return ContinuousControlResult(
            degradation_rate=degradation_rate,
            uncertainty_growth_rate=uncertainty_growth,
            horizon_effects=prediction_errors,
            prediction_errors=list(prediction_errors.values()) if prediction_errors else [],
            uncertainty_by_horizon=uncertainty_by_horizon,
            metadata={
                "horizon": horizon,
                "num_trajectories": num_trajectories,
            },
        )
