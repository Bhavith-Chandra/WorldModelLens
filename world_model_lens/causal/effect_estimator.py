"""Causal Effect Estimation for World Models.

Formal A/B testing framework for world model interpretability.

The core causal question:
    "What is the effect of intervening on component X on outcome Y?"

Mathematical Formulation:
    Effect = E[Y | do(X = x_1)] - E[Y | do(X = x_0)]

Where:
    - X: Intervention target (latent dimension, state, trajectory segment)
    - Y: Outcome metric (reward, reconstruction error, trajectory divergence)
    - do(): Causal intervention operator

Three levels of granularity:
1. Dimension-level: Single neuron effects
2. State-level: Full latent vector effects
3. Trajectory-level: Sequence-level causal effects
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import numpy as np
from collections import defaultdict


@dataclass
class CausalEffect:
    """Result of a causal effect estimation.

    Attributes:
        effect_size: The estimated causal effect (delta between conditions)
        confidence_interval: (lower, upper) bounds at 95% confidence
        p_value: Statistical significance
        baseline_mean: Mean outcome under baseline condition
        intervention_mean: Mean outcome under intervention condition
        sample_size: Number of samples used
        effect_type: Type of effect ("dimension", "state", "trajectory")
    """

    effect_size: float
    baseline_mean: float
    intervention_mean: float
    sample_size: int
    effect_type: str
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    p_value: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InterventionSpec:
    """Specification for a causal intervention.

    Defines what to intervene on and how.

    Types of intervention:
    - ablation: Set to zero
    - constant: Set to fixed value
    - noise: Add Gaussian noise
    - swap: Replace with value from another trajectory
    - learnable: Trainable intervention (for causal discovery)
    """

    target_type: str  # "dimension", "state", "trajectory"
    target_indices: Optional[List[int]] = None  # Which dimensions/steps
    intervention_type: str = "ablation"  # "ablation", "constant", "noise", "swap"
    intervention_value: float = 0.0  # For constant intervention
    noise_std: float = 0.1  # For noise intervention
    source_trajectory: Optional[Any] = None  # For swap intervention


class CausalEffectEstimator:
    """Formal causal effect estimation for world models.

    Example:
        estimator = CausalEffectEstimator(world_model)

        # Level 1: Dimension-level effect
        dim_effect = estimator.estimate_effect(
            source=(0, 5),  # dimension 0 at timestep 5
            target="reward_pred",
            intervention=InterventionSpec(
                target_type="dimension",
                target_indices=[0],
                intervention_type="ablation",
            ),
        )

        # Level 2: State-level effect
        state_effect = estimator.estimate_effect(
            source=(None, 5),  # full state at timestep 5
            target="reconstruction_error",
            intervention=InterventionSpec(
                target_type="state",
                intervention_type="noise",
                noise_std=1.0,
            ),
        )

        # Level 3: Trajectory-level effect
        traj_effect = estimator.estimate_effect(
            source=(None, 0, 10),  # timesteps 0-10
            target="final_reward",
            intervention=InterventionSpec(
                target_type="trajectory",
                intervention_type="constant",
                intervention_value=0.0,
            ),
        )
    """

    def __init__(self, world_model: Any, num_samples: int = 100):
        """Initialize estimator.

        Args:
            world_model: HookedWorldModel instance
            num_samples: Number of samples for effect estimation
        """
        self.wm = world_model
        self.num_samples = num_samples

    def estimate_effect(
        self,
        source: Tuple[Optional[int], int],
        target: str,
        intervention: InterventionSpec,
        observations: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
    ) -> CausalEffect:
        """Estimate causal effect of intervention on outcome.

        Args:
            source: Source of intervention
                - (dim, t): Dimension-level at timestep t
                - (None, t): State-level at timestep t
                - (None, t_start, t_end): Trajectory-level
            target: Target metric ("reward_pred", "reconstruction_error", etc.)
            intervention: Specification of intervention
            observations: Input observations
            actions: Input actions

        Returns:
            CausalEffect with estimated effect size
        """
        if observations is None:
            observations = torch.randn(20, 64, 64, 3)

        # Determine effect type
        if intervention.target_type == "dimension":
            effect_type = "dimension"
            dim_idx, t = source
            effect = self._estimate_dimension_effect(
                dim_idx, t, target, intervention, observations, actions
            )
        elif intervention.target_type == "state":
            effect_type = "state"
            _, t = source
            effect = self._estimate_state_effect(t, target, intervention, observations, actions)
        else:
            effect_type = "trajectory"
            _, t_start, t_end = source
            effect = self._estimate_trajectory_effect(
                t_start, t_end, target, intervention, observations, actions
            )

        effect.effect_type = effect_type
        return effect

    def _run_baseline(
        self,
        observations: torch.Tensor,
        actions: Optional[torch.Tensor],
    ) -> List[float]:
        """Run baseline without intervention."""
        outcomes = []
        for _ in range(self.num_samples):
            traj, _, cache = self.wm.run_with_cache(observations, actions)
            outcome = self._extract_target(traj, cache, "reward_pred")
            outcomes.append(outcome)
        return outcomes

    def _run_intervention(
        self,
        observations: torch.Tensor,
        actions: Optional[torch.Tensor],
        intervention: InterventionSpec,
        t: int,
    ) -> List[float]:
        """Run with intervention at timestep t."""
        outcomes = []

        def make_intervention_fn(intervention, dim_idx):
            def intervention_fn(tensor, ctx):
                if ctx.timestep != t:
                    return tensor
                result = tensor.clone()
                if intervention.intervention_type == "ablation":
                    if dim_idx is not None:
                        result[..., dim_idx] = intervention.intervention_value
                    else:
                        result = result * 0
                elif intervention.intervention_type == "noise":
                    noise = torch.randn_like(result) * intervention.noise_std
                    result = result + noise
                elif intervention.intervention_type == "constant":
                    result = result * 0 + intervention.intervention_value
                return result

            return intervention_fn

        dim_idx = intervention.target_indices[0] if intervention.target_indices else None

        for _ in range(self.num_samples):
            hook_spec = f"t={t}.z"
            if dim_idx is not None:
                hook_spec = f"z[{dim_idx}:{dim_idx + 1}]"

            intervention_fn = make_intervention_fn(intervention, dim_idx)

            traj, cache = self.wm.run_with_advanced_hooks(
                observations,
                actions,
                hook_specs={hook_spec: intervention_fn},
            )
            outcome = self._extract_target(traj, cache, "reward_pred")
            outcomes.append(outcome)

        return outcomes

    def _estimate_dimension_effect(
        self,
        dim_idx: int,
        t: int,
        target: str,
        intervention: InterventionSpec,
        observations: torch.Tensor,
        actions: Optional[torch.Tensor],
    ) -> CausalEffect:
        """Estimate effect of dimension-level intervention."""
        baseline = self._run_baseline(observations, actions)
        intervened = self._run_intervention(observations, actions, intervention, t)

        return self._compute_effect_stats(baseline, intervened, intervention.target_type)

    def _estimate_state_effect(
        self,
        t: int,
        target: str,
        intervention: InterventionSpec,
        observations: torch.Tensor,
        actions: Optional[torch.Tensor],
    ) -> CausalEffect:
        """Estimate effect of full state intervention."""
        baseline = self._run_baseline(observations, actions)
        intervened = self._run_intervention(observations, actions, intervention, t)

        return self._compute_effect_stats(baseline, intervened, "state")

    def _estimate_trajectory_effect(
        self,
        t_start: int,
        t_end: int,
        target: str,
        intervention: InterventionSpec,
        observations: torch.Tensor,
        actions: Optional[torch.Tensor],
    ) -> CausalEffect:
        """Estimate effect of trajectory segment intervention."""
        baseline = self._run_baseline(observations, actions)

        outcomes = []
        for t in range(t_start, t_end):
            intervened = self._run_intervention(observations, actions, intervention, t)
            outcomes.extend(intervened)

        return self._compute_effect_stats(baseline, outcomes, "trajectory")

    def _extract_target(
        self,
        trajectory: Any,
        cache: Any,
        target: str,
    ) -> float:
        """Extract target metric from trajectory/cache."""
        if target == "reward_pred" or target == "reward":
            rewards = [s.predictions.get("reward", torch.tensor(0.0)) for s in trajectory.states]
            if rewards:
                return sum(r.item() if isinstance(r, torch.Tensor) else r for r in rewards) / len(
                    rewards
                )
        elif target == "reconstruction_error":
            errors = []
            for t in range(len(trajectory.states)):
                recon = cache.get(("reconstruction", t))
                obs = cache.get(("observation", t))
                if recon is not None and obs is not None:
                    err = torch.nn.functional.mse_loss(recon, obs).item()
                    errors.append(err)
            return np.mean(errors) if errors else 0.0
        elif target == "final_state_norm":
            final_state = trajectory.states[-1].state
            return final_state.norm().item()
        return 0.0

    def _compute_effect_stats(
        self,
        baseline: List[float],
        intervention: List[float],
        effect_type: str,
    ) -> CausalEffect:
        """Compute statistical effect size."""
        baseline_mean = np.mean(baseline)
        intervention_mean = np.mean(intervention)
        effect_size = intervention_mean - baseline_mean

        # Bootstrap confidence interval
        baseline_arr = np.array(baseline)
        intervention_arr = np.array(intervention)

        n_bootstrap = 1000
        bootstrap_effects = []
        for _ in range(n_bootstrap):
            b_sample = np.random.choice(baseline_arr, size=len(baseline_arr), replace=True)
            i_sample = np.random.choice(intervention_arr, size=len(intervention_arr), replace=True)
            bootstrap_effects.append(np.mean(i_sample) - np.mean(b_sample))

        ci_lower = np.percentile(bootstrap_effects, 2.5)
        ci_upper = np.percentile(bootstrap_effects, 97.5)

        # Simple t-test approximation
        pooled_std = np.sqrt(np.var(baseline_arr) + np.var(intervention_arr))
        if pooled_std > 0:
            t_stat = effect_size / (pooled_std / np.sqrt(len(baseline)))
            p_value = 2 * (1 - self._normal_cdf(abs(t_stat)))
        else:
            p_value = 1.0

        return CausalEffect(
            effect_size=effect_size,
            baseline_mean=baseline_mean,
            intervention_mean=intervention_mean,
            sample_size=len(baseline) + len(intervention),
            effect_type=effect_type,
            confidence_interval=(ci_lower, ci_upper),
            p_value=p_value,
        )

    def _normal_cdf(self, x: float) -> float:
        """Approximate normal CDF."""
        return 0.5 * (1 + torch.erf(torch.tensor(x / np.sqrt(2))).item())

    def sweep_dimension_effects(
        self,
        t: int,
        target: str,
        observations: Optional[torch.Tensor] = None,
    ) -> Dict[int, CausalEffect]:
        """Sweep all dimensions at a timestep to find most causal.

        Args:
            t: Timestep to sweep
            target: Target metric
            observations: Input observations

        Returns:
            Dict mapping dimension index to causal effect
        """
        results = {}

        # Get latent dimension from cache
        if observations is None:
            observations = torch.randn(20, 64, 64, 3)

        traj, _, cache = self.wm.run_with_cache(observations)
        z_sample = cache.get(("z_posterior", 0))
        if z_sample is None:
            z_sample = traj.states[0].state

        d_z = z_sample.shape[-1]

        for dim in range(min(d_z, 50)):  # Limit for efficiency
            intervention = InterventionSpec(
                target_type="dimension",
                target_indices=[dim],
                intervention_type="ablation",
            )
            effect = self.estimate_effect(
                source=(dim, t),
                target=target,
                intervention=intervention,
                observations=observations,
            )
            results[dim] = effect

        return results
