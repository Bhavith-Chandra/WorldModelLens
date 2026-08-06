"""Temporal Attribution for world model predictions.

This module provides attribution methods adapted for time-series predictions:
- Integrated Gradients for temporal data
- SHAP values for trajectory contributions
- Temporal attention visualization
- Causal tracing through time
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class AttributionResult:
    """Result of temporal attribution analysis."""

    attributions: torch.Tensor  # [T, ...] attribution scores
    baseline: torch.Tensor | None  # Baseline input
    predictions: torch.Tensor  # Model predictions
    convergence_score: float | None = None

    def get_top_timesteps(self, k: int = 5) -> list[tuple[int, float]]:
        """Get top k most important timesteps.

        Returns:
            List of (timestep, importance) tuples
        """
        if self.attributions.dim() > 1:
            scores = self.attributions.abs().mean(dim=tuple(range(1, self.attributions.dim())))
        else:
            scores = self.attributions.abs()

        topk = torch.topk(scores, min(k, len(scores)))
        return list(zip(topk.indices.tolist(), topk.values.tolist(), strict=False))


class TemporalIntegratedGradients:
    """Integrated Gradients adapted for temporal/world model predictions.

    Computes attribution by integrating gradients along path from baseline
    to actual input. Particularly useful for understanding which past
    frames influenced the current prediction.

    Example:
        ig = TemporalIntegratedGradients(world_model)

        # Attribute current prediction to past observations
        result = ig.attribute(
            current_obs=obs_t,
            past_obs=obs_history,  # [T, ...]
            target='collision_detection',
            n_steps=50,
        )

        # Get most important past frames
        important_frames = result.get_top_timesteps(k=3)
    """

    def __init__(
        self,
        forward_fn: Callable,
        method: str = "ig",
    ):
        """Initialize IG.

        Args:
            forward_fn: Function that takes inputs and returns predictions
            method: 'ig' (Integrated Gradients) or 'sa' (Saliency)
        """
        self.forward_fn = forward_fn
        self.method = method

    def attribute(
        self,
        current_obs: torch.Tensor,
        past_obs: torch.Tensor | None = None,
        baseline: torch.Tensor | None = None,
        target: int | str | None = None,
        n_steps: int = 50,
        return_convergence: bool = True,
    ) -> AttributionResult:
        """Compute attributions.

        Args:
            current_obs: Current observation [*, C, H, W] or [*, D]
            past_obs: Past observations [T, C, H, W] or [T, D]
            baseline: Baseline input (zero if None)
            target: Target class/index for attribution
            n_steps: Number of integration steps
            return_convergence: Whether to compute convergence score

        Returns:
            AttributionResult
        """
        if past_obs is not None:
            past_obs = past_obs.detach().requires_grad_(True)

        current_obs = current_obs.detach().requires_grad_(True)

        if baseline is None:
            baseline = torch.zeros_like(
                current_obs
                if past_obs is None
                else torch.cat([current_obs.unsqueeze(0), past_obs], dim=0)
            )

        baseline = baseline.detach().requires_grad_(True)

        if past_obs is None:
            inputs = torch.cat([current_obs.unsqueeze(0)], dim=0)
        else:
            inputs = torch.cat([current_obs.unsqueeze(0), past_obs], dim=0)

        inputs = inputs.detach().requires_grad_(True)

        scaled_inputs = [
            baseline + (float(i) / n_steps) * (inputs - baseline) for i in range(n_steps + 1)
        ]

        total_grad = torch.zeros_like(inputs)

        for scaled_input in scaled_inputs:
            scaled_input.requires_grad_(True)

            if past_obs is not None:
                output = self.forward_fn(
                    current_obs=scaled_input[1:],
                    past_obs=scaled_input[:-1],
                )
            else:
                output = self.forward_fn(
                    current_obs=scaled_input.squeeze(0),
                )

            if target is not None:
                if isinstance(target, int):
                    score = output[target]
                elif isinstance(target, str):
                    score = output
            else:
                score = output.sum()

            score.backward()

            if scaled_input.grad is not None:
                total_grad += scaled_input.grad

        attributions = total_grad * (inputs - baseline) / n_steps

        convergence = None
        if return_convergence:
            mid = n_steps // 2
            with torch.no_grad():
                if past_obs is not None:
                    mid_output = self.forward_fn(
                        current_obs=scaled_inputs[mid][1:],
                        past_obs=scaled_inputs[mid][:-1],
                    )
                else:
                    mid_output = self.forward_fn(scaled_inputs[mid].squeeze(0))

                if target is not None and isinstance(target, int):
                    mid_score = mid_output[target].item()
                else:
                    mid_score = mid_output.sum().item()

                final_output = self.forward_fn(
                    current_obs=inputs[1:] if past_obs is not None else inputs,
                    past_obs=inputs[:1] if past_obs is not None else None,
                )
                if target is not None and isinstance(target, int):
                    final_score = final_output[target].item()
                else:
                    final_score = final_output.sum().item()

                convergence = abs(final_score - mid_score)

        return AttributionResult(
            attributions=attributions,
            baseline=baseline,
            predictions=inputs,
            convergence_score=convergence,
        )


class TemporalSHAP:
    """SHAP values for temporal/world model attributions.

    Computes Shapley values considering temporal structure.
    """

    def __init__(
        self,
        forward_fn: Callable,
        background_size: int = 100,
    ):
        """Initialize SHAP.

        Args:
            forward_fn: Forward function
            background_size: Number of background samples
        """
        self.forward_fn = forward_fn
        self.background_size = background_size

    def compute_shap_values(
        self,
        input_sequence: torch.Tensor,
        target_idx: int = 0,
        n_samples: int = 100,
    ) -> torch.Tensor:
        """Compute SHAP values for sequence.

        Args:
            input_sequence: Input sequence [T, D]
            target_idx: Index to attribute
            n_samples: Number of samples

        Returns:
            SHAP values [T]
        """
        T = input_sequence.shape[0]

        torch.cat(
            [
                input_sequence,
                torch.zeros_like(input_sequence),
            ],
            dim=0,
        )

        shap_values = torch.zeros(T)

        for t in range(T):
            differences = []

            for _ in range(n_samples):
                mask = torch.rand(T) > 0.5
                mask[t] = True

                masked_input = input_sequence.clone()
                masked_input[~mask] = 0

                with torch.no_grad():
                    output_with = self.forward_fn(masked_input)
                    output_without = self.forward_fn(torch.zeros_like(masked_input))

                if isinstance(output_with, torch.Tensor):
                    diff = output_with[target_idx].item() - output_without[target_idx].item()
                else:
                    diff = abs(output_with - output_without)

                differences.append(diff)

            shap_values[t] = np.mean(differences)

        return shap_values


class CausalTracer:
    """Causal tracing through world model components.

    Identifies which components/activations are causally responsible
    for specific behaviors.

    Example:
        tracer = CausalTracer(world_model)

        # Find causal components for collision detection
        result = tracer.trace(
            trajectory=traj,
            target_behavior='collision',
            components=['z', 'h', 'attention'],
        )
    """

    def __init__(
        self,
        world_model: Any,
    ):
        """Initialize causal tracer.

        Args:
            world_model: HookedWorldModel instance
        """
        self.wm = world_model

    def trace(
        self,
        trajectory: Any,
        target_behavior: str,
        components: list[str] | None = None,
        intervention_value: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Trace causal influence through components.

        Args:
            trajectory: WorldTrajectory
            target_behavior: Behavior to trace
            components: Components to analyze
            intervention_value: Value to intervene with

        Returns:
            Dict mapping component to causal score
        """
        components = components or ["z", "h"]

        baseline_output = self._get_behavior_score(trajectory, target_behavior)

        causal_scores = {}

        for comp in components:
            intervention_scores = []

            for t in range(len(trajectory.states)):
                intervened_traj = self._intervene_at(trajectory, comp, t, intervention_value)

                score = self._get_behavior_score(intervened_traj, target_behavior)
                intervention_scores.append(abs(score - baseline_output))

            causal_scores[comp] = np.mean(intervention_scores)

        return causal_scores

    def _get_behavior_score(
        self,
        trajectory: Any,
        behavior: str,
    ) -> float:
        """Get score for target behavior."""
        if behavior == "collision":
            if hasattr(trajectory, "metadata") and "collision" in trajectory.metadata:
                return trajectory.metadata["collision"]

        if hasattr(trajectory, "reward_sequence") and trajectory.reward_sequence is not None:
            return trajectory.reward_sequence.mean().item()

        return 0.0

    def _intervene_at(
        self,
        trajectory: Any,
        component: str,
        timestep: int,
        value: torch.Tensor | None = None,
    ) -> Any:
        """Intervene at specific component/timestep."""
        if value is None:
            value = torch.zeros_like(trajectory.states[timestep].state)

        from world_model_lens.core.world_trajectory import WorldTrajectory

        new_states = []
        for i, state in enumerate(trajectory.states):
            if i == timestep:
                new_state = state.__class__(
                    state=value,
                    timestep=state.timestep,
                    predictions=state.predictions.copy(),
                )
                new_states.append(new_state)
            else:
                new_states.append(state)

        return WorldTrajectory(
            states=new_states,
            source=trajectory.source,
            name=trajectory.name,
        )


class AttentionVisualizer:
    """Visualize temporal attention patterns in world models.

    For transformer-based world models, visualizes which past
    timesteps receive attention.
    """

    def __init__(
        self,
        world_model: Any,
    ):
        """Initialize attention visualizer.

        Args:
            world_model: HookedWorldModel
        """
        self.wm = world_model
        self._attention_cache = []

    def visualize_attention(
        self,
        sequence: torch.Tensor,
        layer_name: str = "attention",
    ) -> torch.Tensor:
        """Extract and visualize attention patterns.

        Args:
            sequence: Input sequence [T, D]
            layer_name: Layer to extract attention from

        Returns:
            Attention matrix [T, T]
        """
        attention_weights = []

        def hook_fn(module, input, output):
            if isinstance(output, tuple) and len(output) > 1:
                attention_weights.append(output[1])

        hooks = []
        for name, module in self.wm.adapter.named_modules():
            if layer_name in name:
                hooks.append(module.register_forward_hook(hook_fn))

        with torch.no_grad():
            _ = self.wm.run_with_cache(sequence.unsqueeze(0))

        for hook in hooks:
            hook.remove()

        if attention_weights:
            return attention_weights[0].squeeze(0).cpu()

        return torch.zeros(len(sequence), len(sequence))

    def get_attention_heatmap(
        self,
        sequence: torch.Tensor,
        layer_name: str = "attention",
    ) -> np.ndarray:
        """Get attention as heatmap array.

        Args:
            sequence: Input sequence
            layer_name: Layer name

        Returns:
            Attention heatmap as numpy array
        """
        attn = self.visualize_attention(sequence, layer_name)
        return attn.numpy()
