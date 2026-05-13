"""Faithfulness metrics for world model interpretability.

This module provides faithfulness metrics that measure how well latent representations
explain model predictions. AOPC (Area Over Perturbation Curve) is the primary metric:
- Ablate Top-K dimensions using hooks
- Measure MSE delta between original and ablated predictions
- Integrate area under the perturbation curve
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from world_model_lens import ActivationCache, HookedWorldModel
from world_model_lens.core.hooks import HookPoint, HookContext


@dataclass
class AOPCResult:
    """Result of AOPC (Area Over Perturbation Curve) analysis."""

    aopc_score: float
    mses: list[float]
    k_values: list[int]
    component: str

    def plot(self, figsize=(10, 6)):
        """Plot the perturbation curve."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.k_values, self.mses, marker="o")
        ax.set_xlabel("K (Number of ablated dimensions)")
        ax.set_ylabel("MSE Delta")
        ax.set_title(f"AOPC Curve (score={self.aopc_score:.4f})")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.fill_between(self.k_values, self.mses, alpha=0.3)
        return fig


@dataclass
class PerturbationResult:
    """Result of perturbation analysis at a specific K."""

    k: int
    mse_delta: float
    ablated_dims: list[int]
    component: str


class FaithfulnessAnalyzer:
    """Analyzer for computing faithfulness metrics.

    Computes how faithfully latent dimensions encode prediction-relevant
    information. Higher AOPC = more faithful representations.
    """

    def __init__(self, wm: HookedWorldModel):
        """Initialize faithfulness analyzer.

        Args:
            wm: HookedWorldModel instance.
        """
        self.wm = wm

    def aopc(
        self,
        observations: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        target_component: str = "z_posterior",
        predictor_fn: Optional[Callable[[ActivationCache], torch.Tensor]] = None,
        max_k: Optional[int] = None,
        normalize: bool = True,
        dim_importance: Optional[torch.Tensor] = None,
    ) -> AOPCResult:
        """Compute Area Over Perturbation Curve (AOPC).

        AOPC measures the area under the curve of MSE deltas as we progressively
        ablate top-K dimensions. Higher AOPC means dimensions are more faithful
        to the predictions.

        Algorithm:
        1. Run baseline forward pass to get predictions
        2. For each K = 1, 2, 3, ... max_k:
           - Create a hook that zeros top-K dimensions
           - Run forward pass with hook applied
           - Compute MSE delta
        3. Integrate (trapezoidal) the MSE deltas

        Args:
            observations: Observation sequence [T, ...].
            actions: Optional action sequence [T, d_action].
            target_component: Component to ablate (must exist in cache).
            predictor_fn: Custom predictor function. If None, uses cached predictions.
            max_k: Maximum K to ablate. If None, uses all dimensions.
            normalize: Whether to normalize by max MSE.
            dim_importance: Dimension importance scores. If None, uses magnitude.

        Returns:
            AOPCResult with score and perturbation data.
        """
        _, cache = self.wm.run_with_cache(observations, actions)

        original = cache[target_component]

        original_device = original.device
        original_flat = original.flatten(1)
        num_dims = original_flat.shape[-1]

        if max_k is None:
            max_k = num_dims

        if dim_importance is None:
            dim_importance = original_flat.abs().mean(dim=0)
        else:
            dim_importance = dim_importance.to(original_device)

        sorted_dims = dim_importance.argsort(descending=True).tolist()

        baseline_pred = self._get_predictions(cache, target_component, predictor_fn)
        if baseline_pred is None:
            return AOPCResult(
                aopc_score=0.0,
                mses=[],
                k_values=[],
                component=target_component,
            )

        mses = []
        k_values = []

        for k in range(1, max_k + 1):
            dims_to_ablated = sorted_dims[:k]

            def make_ablate_hook(dims_to_zero):
                def ablate_hook(tensor: torch.Tensor, context: HookContext) -> torch.Tensor:
                    result = tensor.clone()
                    for dim in dims_to_zero:
                        if dim < result.shape[-1]:
                            result[..., dim] = 0.0
                    return result

                return ablate_hook

            hook = HookPoint(
                name=target_component,
                fn=make_ablate_hook(dims_to_ablated),
                stage="post",
            )

            hooked_result = self.wm.run_with_hooks(
                observations,
                actions,
                fwd_hooks=[hook],
                return_cache=True,
            )
            if isinstance(hooked_result, tuple):
                _, cache_ablated = hooked_result
            else:
                cache_ablated = None

            if cache_ablated is None:
                mses.append(0.0)
                k_values.append(k)
                continue

            pred_after = self._get_predictions(cache_ablated, target_component, predictor_fn)

            if pred_after is not None and baseline_pred is not None:
                mse_delta = torch.nn.functional.mse_loss(pred_after, baseline_pred).item()
            else:
                mse_delta = 0.0

            mses.append(mse_delta)
            k_values.append(k)

        if len(mses) == 0:
            return AOPCResult(
                aopc_score=0.0,
                mses=[],
                k_values=[],
                component=target_component,
            )

        mses_normalized = mses.copy()
        if normalize:
            max_mse = max(mses) if max(mses) > 0 else 1.0
            mses_normalized = [m / max_mse for m in mses]

        aopc_score = self._integrate_area(k_values, mses_normalized)

        return AOPCResult(
            aopc_score=aopc_score,
            mses=mses,
            k_values=k_values,
            component=target_component,
        )

    def _get_predictions(
        self,
        cache: ActivationCache,
        target_component: str,
        predictor_fn: Optional[Callable],
    ) -> Optional[torch.Tensor]:
        """Get predictions from cache or compute via forward pass."""
        if predictor_fn is not None:
            try:
                return predictor_fn(cache)
            except KeyError:
                return None

        pred_keys = ["reward_pred", "value", "policy"]
        for key in pred_keys:
            try:
                pred = cache[key]
                if pred is not None:
                    return pred
            except (KeyError, TypeError):
                pass

        return None

    def _integrate_area(self, x: list[int], y: list[float]) -> float:
        """Integrate area under curve using trapezoidal rule."""
        if len(x) < 2 or len(y) < 2:
            return sum(y) if y else 0.0

        area = 0.0
        for i in range(len(x) - 1):
            dx = x[i + 1] - x[i]
            avg_height = (y[i] + y[i + 1]) / 2.0
            area += dx * avg_height

        return area

    def perturbation_curve(
        self,
        observations: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        target_component: str = "z_posterior",
        predictor_fn: Optional[Callable] = None,
        k_values: Optional[list[int]] = None,
    ) -> list[PerturbationResult]:
        """Compute detailed perturbation curve.

        Args:
            observations: Observation sequence.
            actions: Optional action sequence.
            target_component: Component to ablate (must exist in cache).
            predictor_fn: Optional predictor function.
            k_values: Specific K values to test. If None, uses [1, 2, 5, 10, 20, ...].

        Returns:
            List of PerturbationResult for each K.
        """
        _, cache = self.wm.run_with_cache(observations, actions)

        try:
            original = cache[target_component]
        except KeyError:
            return []

        original_flat = original.flatten(1)
        num_dims = original_flat.shape[-1]

        if k_values is None:
            k_values = [1, 2, 5, 10, 20]
            k_values = [k for k in k_values if k <= num_dims]
            if num_dims not in k_values:
                k_values.append(num_dims)

        dim_importance = original_flat.abs().mean(dim=0)
        sorted_dims = dim_importance.argsort(descending=True).tolist()

        baseline_pred = self._get_predictions(cache, target_component, predictor_fn)
        if baseline_pred is None:
            baseline_pred = original_flat.mean(dim=0, keepdim=True)

        results = []

        for k in k_values:
            dims_to_ablated = sorted_dims[:k]

            def make_ablate_hook(dims_to_zero):
                def ablate_hook(tensor: torch.Tensor, context: HookContext) -> torch.Tensor:
                    result = tensor.clone()
                    for dim in dims_to_zero:
                        if dim < result.shape[-1]:
                            result[..., dim] = 0.0
                    return result

                return ablate_hook

            hook = HookPoint(
                name=target_component,
                fn=make_ablate_hook(dims_to_ablated),
                stage="post",
            )

            hooked_result = self.wm.run_with_hooks(
                observations,
                actions,
                fwd_hooks=[hook],
                return_cache=True,
            )
            if isinstance(hooked_result, tuple):
                _, cache_ablated = hooked_result
            else:
                cache_ablated = None

            if cache_ablated is None:
                results.append(
                    PerturbationResult(
                        k=k,
                        mse_delta=0.0,
                        ablated_dims=dims_to_ablated,
                        component=target_component,
                    )
                )
                continue

            pred_after = self._get_predictions(cache_ablated, target_component, predictor_fn)

            if pred_after is not None and baseline_pred is not None:
                mse_delta = torch.nn.functional.mse_loss(pred_after, baseline_pred).item()
            else:
                mse_delta = 0.0

            results.append(
                PerturbationResult(
                    k=k,
                    mse_delta=mse_delta,
                    ablated_dims=dims_to_ablated,
                    component=target_component,
                )
            )

        return results
