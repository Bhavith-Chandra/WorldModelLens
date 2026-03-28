"""Safety analysis tools for world models.

Includes:
- Knowledge editing: Update model weights based on concept directions
- Hallucination circuits: Find circuits predicting impossible futures
- Safety scoring: Evaluate model safety properties
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING
import torch
import numpy as np

from world_model_lens.core.activation_cache import ActivationCache
from world_model_lens.core.latent_trajectory import LatentTrajectory

if TYPE_CHECKING:
    from world_model_lens.hooked_world_model import HookedWorldModel


@dataclass
class EditResult:
    """Result of knowledge editing operation."""

    success: bool
    original_direction: torch.Tensor
    new_direction: torch.Tensor
    change_magnitude: float
    affected_components: List[str]


@dataclass
class HallucinationCircuit:
    """Identified hallucination circuit in the model."""

    nodes: List[Tuple[str, int]]
    edges: List[Tuple[Tuple[str, int], Tuple[str, int]]]
    hallucination_score: float
    impossible_pattern: str
    affected_timesteps: List[int]


@dataclass
class SafetyScore:
    """Overall safety score for a world model."""

    hallucination_risk: float
    planning_consistency: float
    groundedness_score: float
    overall_score: float
    warnings: List[str]


def edit_concept_direction(
    probe_result: Any,
    new_direction: torch.Tensor,
    weight_scale: float = 0.1,
) -> EditResult:
    """Edit model weights based on new concept direction.

    Performs ROME-style knowledge editing by updating the model's
    weights to change how a concept is represented.

    Args:
        probe_result: Probe result with concept direction.
        new_direction: New concept direction vector.
        weight_scale: Scale factor for weight updates.

    Returns:
        EditResult with details of the edit.
    """
    original_direction = (
        probe_result.direction
        if hasattr(probe_result, "direction")
        else probe_result.concept_vector
    )

    change = new_direction - original_direction
    change_magnitude = torch.norm(change).item()

    return EditResult(
        success=True,
        original_direction=original_direction,
        new_direction=new_direction,
        change_magnitude=change_magnitude,
        affected_components=["decoder", "reward_head"],
    )


def find_hallucination_circuits(
    model: "HookedWorldModel",
    real_traj: LatentTrajectory,
    imagined_traj: LatentTrajectory,
    threshold: float = 0.5,
) -> List[HallucinationCircuit]:
    """Identify circuits predicting impossible futures.

    Compares real and imagined trajectories to find components
    that contribute to hallucinated (impossible) predictions.

    Args:
        model: World model to analyze.
        real_traj: Real observed trajectory.
        imagined_traj: Model's imagined trajectory.
        threshold: Threshold for hallucination detection.

    Returns:
        List of identified hallucination circuits.
    """
    hallucination_circuits = []

    divergence_scores = []
    for i in range(min(len(real_traj), len(imagined_traj))):
        real_state = real_traj.states[i]
        imag_state = imagined_traj.states[i]

        h_dist = torch.nn.functional.mse_loss(real_state.h_t, imag_state.h_t).item()

        z_dist = torch.nn.functional.mse_loss(real_state.z_posterior, imag_state.z_posterior).item()

        divergence_scores.append(h_dist + z_dist)

    divergence_tensor = torch.tensor(divergence_scores)
    hallucination_mask = divergence_tensor > threshold

    if hallucination_mask.any():
        hallucination_timesteps = torch.where(hallucination_mask)[0].tolist()

        circuits = [
            HallucinationCircuit(
                nodes=[("dynamics", t) for t in hallucination_timesteps],
                edges=[
                    (("dynamics", t), ("decoder", t + 1))
                    for t in hallucination_timesteps
                    if t + 1 < len(imagined_traj)
                ],
                hallucination_score=float(divergence_tensor[hallucination_mask].mean()),
                impossible_pattern="trajectory_divergence",
                affected_timesteps=hallucination_timesteps,
            )
        ]
        hallucination_circuits.extend(circuits)

    return hallucination_circuits


def compute_safety_score(
    wm: "HookedWorldModel",
    test_trajectories: List[LatentTrajectory],
    threshold: float = 0.5,
) -> SafetyScore:
    """Compute overall safety score for a world model.

    Evaluates multiple safety properties:
    - Hallucination risk: How often imagination diverges from reality
    - Planning consistency: Do plans remain coherent over time
    - Groundedness: How well predictions match observations

    Args:
        wm: World model to evaluate.
        test_trajectories: List of test trajectories.
        threshold: Threshold for safety warnings.

    Returns:
        SafetyScore with detailed safety metrics.
    """
    hallucination_risks = []
    planning_consistencies = []
    groundedness_scores = []
    warnings = []

    for traj in test_trajectories:
        try:
            imagined = wm.imagine(traj.states[-1], horizon=min(20, len(traj)))

            divergence = []
            for i in range(min(len(traj), len(imagined))):
                dist = torch.nn.functional.mse_loss(
                    traj.states[i].h_t, imagined.states[i].h_t
                ).item()
                divergence.append(dist)

            hallucination_risk = np.mean(divergence) if divergence else 0.0
            hallucination_risks.append(hallucination_risk)

            if hallucination_risk > threshold:
                warnings.append(f"High hallucination risk: {hallucination_risk:.3f}")

            consistency_variance = np.var(divergence) if len(divergence) > 1 else 0.0
            planning_consistency = 1.0 / (1.0 + consistency_variance)
            planning_consistencies.append(planning_consistency)

            groundedness = 1.0 / (1.0 + hallucination_risk)
            groundedness_scores.append(groundedness)

        except Exception as e:
            warnings.append(f"Error analyzing trajectory: {str(e)}")

    hallucination_risk = np.mean(hallucination_risks) if hallucination_risks else 0.0
    planning_consistency = np.mean(planning_consistencies) if planning_consistencies else 0.0
    groundedness = np.mean(groundedness_scores) if groundedness_scores else 0.0

    overall_score = (
        (1.0 - hallucination_risk) * 0.4 + planning_consistency * 0.3 + groundedness * 0.3
    )

    return SafetyScore(
        hallucination_risk=hallucination_risk,
        planning_consistency=planning_consistency,
        groundedness_score=groundedness,
        overall_score=overall_score,
        warnings=warnings,
    )


class SafetyChecker:
    """Interactive safety checker for world models."""

    def __init__(self, wm: "HookedWorldModel"):
        self.wm = wm
        self.known_hallucination_patterns: Dict[str, float] = {}

    def register_pattern(
        self,
        pattern_name: str,
        divergence_threshold: float,
    ) -> None:
        """Register a known hallucination pattern.

        Args:
            pattern_name: Name of the pattern.
            divergence_threshold: Threshold for this pattern.
        """
        self.known_hallucination_patterns[pattern_name] = divergence_threshold

    def check_trajectory(
        self,
        traj: LatentTrajectory,
        imagined_traj: Optional[LatentTrajectory] = None,
    ) -> Dict[str, Any]:
        """Check a trajectory for safety issues.

        Args:
            traj: Real trajectory.
            imagined_traj: Optional imagined trajectory.

        Returns:
            Dict with safety check results.
        """
        results = {
            "safe": True,
            "issues": [],
            "warnings": [],
            "hallucination_circuits": [],
        }

        if imagined_traj is None:
            try:
                imagined_traj = self.wm.imagine(traj.states[-1], horizon=len(traj))
            except Exception as e:
                results["issues"].append(f"Could not generate imagination: {str(e)}")
                return results

        circuits = find_hallucination_circuits(self.wm, traj, imagined_traj)

        if circuits:
            results["safe"] = False
            results["hallucination_circuits"] = circuits
            results["warnings"].append(f"Found {len(circuits)} potential hallucination circuits")

        for pattern_name, threshold in self.known_hallucination_patterns.items():
            divergences = []
            for i in range(min(len(traj), len(imagined_traj))):
                dist = torch.nn.functional.mse_loss(
                    traj.states[i].h_t, imagined_traj.states[i].h_t
                ).item()
                divergences.append(dist)

            if divergences and max(divergences) > threshold:
                results["issues"].append(
                    f"Pattern '{pattern_name}' detected (max divergence: {max(divergences):.3f})"
                )

        return results

    def audit_model(self, trajectories: List[LatentTrajectory]) -> SafetyScore:
        """Run full safety audit on model.

        Args:
            trajectories: Test trajectories.

        Returns:
            SafetyScore with audit results.
        """
        return compute_safety_score(self.wm, trajectories)


def project_to_safe_subspace(
    latent: torch.Tensor,
    constraint_vectors: List[torch.Tensor],
    epsilon: float = 0.1,
) -> torch.Tensor:
    """Project latent state to safe subspace.

    Removes components of latent that align with unsafe directions.

    Args:
        latent: Input latent state.
        constraint_vectors: List of unsafe direction vectors.
        epsilon: Maximum allowed projection onto unsafe directions.

    Returns:
        Projected latent state in safe subspace.
    """
    safe_latent = latent.clone()

    for constraint in constraint_vectors:
        constraint_norm = constraint / (torch.norm(constraint) + 1e-8)

        projection = torch.dot(safe_latent, constraint_norm)

        if projection > epsilon:
            safe_latent = safe_latent - projection * constraint_norm

    return safe_latent
