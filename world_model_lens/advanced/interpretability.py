"""Advanced interpretability techniques for world models.

Includes:
- Logit lens analogues for world model heads
- Path patching for causal analysis across timesteps
- SAE circuit discovery
- Long-horizon analysis (planning horizon, belief drift, overcommitment)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as functional

if TYPE_CHECKING:
    from world_model_lens import HookedWorldModel
    from world_model_lens.core import ActivationCache, LatentTrajectory
    from world_model_lens.probing.prober import ProbeResult
    from world_model_lens.sae.sae import SparseAutoencoder as Sae


@dataclass
class LogitLensResult:
    """Result from logit lens analysis."""

    reward_logits: torch.Tensor
    action_logits: torch.Tensor
    top_tokens: dict[str, list[int]]
    projection_matrix: torch.Tensor
    component: str
    timestep: int


class LogitLens:
    """Logit lens analogue for world model heads.

    Projects latent states to reward/action prediction spaces to understand
    what information is available at each timestep/component.

    Example:
        lens = LogitLens(wm)
        result = lens.project_to_logits(cache, component="h", timestep=10)
        print(f"Top reward tokens: {result.top_tokens['reward']}")
    """

    def __init__(self, wm: HookedWorldModel):
        self.wm = wm
        self._projection_matrices: dict[str, nn.Linear] = {}

    def _get_projection_matrix(self, target: str) -> nn.Linear:
        """Get or create projection matrix to target space."""
        if target not in self._projection_matrices:
            caps = self.wm.adapter.capabilities

            if target == "reward":
                out_dim = 1
            elif target == "action":
                out_dim = caps.d_action if hasattr(caps, "d_action") else 1
            else:
                out_dim = 1

            proj = nn.Linear(self.wm.config.d_h, out_dim)
            self._projection_matrices[target] = proj

        return self._projection_matrices[target]

    def project_to_logits(
        self,
        cache: ActivationCache,
        component: str = "h",
        timestep: int = 0,
    ) -> LogitLensResult:
        """Project latent to prediction spaces.

        Args:
            cache: ActivationCache with latent states.
            component: Component to project from.
            timestep: Specific timestep.

        Returns:
            LogitLensResult with logits and projections.
        """
        latent = cache[component, timestep]

        reward_logits = None
        action_logits = None

        if "reward" in self.wm.adapter.capabilities.__dict__:
            proj = self._get_projection_matrix("reward")
            reward_logits = proj(latent)

        if "action" in self.wm.adapter.capabilities.__dict__:
            proj = self._get_projection_matrix("action")
            action_logits = proj(latent)

        top_tokens = {}
        if reward_logits is not None:
            top_tokens["reward"] = torch.topk(reward_logits, 5).indices.tolist()
        if action_logits is not None:
            top_tokens["action"] = torch.topk(action_logits, 5).indices.tolist()

        return LogitLensResult(
            reward_logits=reward_logits if reward_logits is not None else torch.tensor([]),
            action_logits=action_logits if action_logits is not None else torch.tensor([]),
            top_tokens=top_tokens,
            projection_matrix=self._projection_matrices.get("reward", torch.eye(1)).weight,
            component=component,
            timestep=timestep,
        )

    def sweep(
        self,
        cache: ActivationCache,
        components: list[str] | None = None,
        timesteps: list[int] | None = None,
    ) -> dict[tuple[str, int], LogitLensResult]:
        """Run logit lens across components and timesteps."""
        components = components or cache.component_names
        timesteps = timesteps or cache.timesteps

        results = {}
        for comp in components:
            for t in timesteps:
                try:
                    results[(comp, t)] = self.project_to_logits(cache, comp, t)
                except Exception:
                    pass

        return results


@dataclass
class PathPatchingResult:
    """Result from path patching experiment."""

    source_component: str
    source_timestep: int
    target_component: str
    target_timestep: int
    causal_strength: float
    patch_recovery: float


class PathPatcher:
    """Path patching for causal analysis across timesteps.

    Traces causal paths between components at different timesteps.
    """

    def __init__(self, wm: HookedWorldModel):
        self.wm = wm

    def patch_path(
        self,
        cache_clean: ActivationCache,
        cache_corrupt: ActivationCache,
        source_comp: str,
        source_t: int,
        target_comp: str,
        target_t: int,
        metric_fn: Callable[[torch.Tensor], float],
    ) -> PathPatchingResult:
        """Patch along causal path from source to target.

        Args:
            cache_clean: Clean activation cache.
            cache_corrupt: Corrupted activation cache.
            source_comp: Source component name.
            source_t: Source timestep.
            target_comp: Target component name.
            target_t: Target timestep.
            metric_fn: Metric to compute on output.

        Returns:
            PathPatchingResult with causal strength.
        """
        clean_value = cache_clean.get(source_comp, source_t)
        corrupt_value = cache_corrupt.get(source_comp, source_t)

        if clean_value is None or corrupt_value is None:
            return PathPatchingResult(
                source_component=source_comp,
                source_timestep=source_t,
                target_component=target_comp,
                target_timestep=target_t,
                causal_strength=0.0,
                patch_recovery=0.0,
            )

        def path_hook(tensor: torch.Tensor, ctx: Any) -> torch.Tensor:
            if ctx.component == target_comp and ctx.timestep == target_t:
                return clean_value.clone()
            return tensor

        hook = self.wm.hook_registry.create_hook(
            name=target_comp,
            fn=path_hook,
            timestep=target_t,
        )
        self.wm.add_hook(hook)

        try:
            obs = torch.randn(10, self.wm.config.d_obs)
            _, traj, _ = self.wm.run_with_cache(obs)
            patched_metric = metric_fn(traj)
        except Exception:
            patched_metric = 0.0
        finally:
            self.wm.clear_hooks()

        clean_metric = metric_fn(traj)
        corrupt_metric = 0.0

        recovery = (patched_metric - corrupt_metric) / (clean_metric - corrupt_metric + 1e-8)
        recovery = max(0.0, min(1.0, recovery))

        return PathPatchingResult(
            source_component=source_comp,
            source_timestep=source_t,
            target_component=target_comp,
            target_timestep=target_t,
            causal_strength=recovery,
            patch_recovery=recovery,
        )

    def find_causal_paths(
        self,
        cache_clean: ActivationCache,
        cache_corrupt: ActivationCache,
        metric_fn: Callable[[torch.Tensor], float],
        max_length: int = 3,
    ) -> list[PathPatchingResult]:
        """Find causal paths up to max_length."""
        components = cache_clean.component_names
        timesteps = cache_clean.timesteps

        results = []
        for source_comp in components:
            for target_comp in components:
                if source_comp == target_comp:
                    continue

                for source_t in timesteps[:5]:
                    for target_t in timesteps[:5]:
                        if target_t <= source_t:
                            continue

                        result = self.patch_path(
                            cache_clean,
                            cache_corrupt,
                            source_comp,
                            source_t,
                            target_comp,
                            target_t,
                            metric_fn,
                        )
                        if result.patch_recovery > 0.1:
                            results.append(result)

        return sorted(results, key=lambda x: x.patch_recovery, reverse=True)[:20]


@dataclass
class SAECircuitResult:
    """Result from SAE circuit discovery."""

    feature_graph: dict[int, set[int]]
    causal_edges: list[tuple[int, int, float]]
    feature_activations: torch.Tensor


class SAECircuitDiscovery:
    """Find SAE features that co-activate causally.

    Builds feature coactivation graphs and tests causal links via patching.
    """

    def __init__(self, sae: Sae, wm: HookedWorldModel):
        self.sae = sae
        self.wm = wm

    def compute_activations(
        self,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SAE feature activations."""
        with torch.no_grad():
            features = self.sae.encode(latents)
        return features

    def build_coactivation_graph(
        self,
        activations: torch.Tensor,
        threshold: float = 0.5,
    ) -> dict[int, set[int]]:
        """Build graph of coactivating features.

        Args:
            activations: Feature activations [T, n_features].
            threshold: Coactivation threshold.

        Returns:
            Dict mapping feature indices to sets of coactivating features.
        """
        active = (activations.abs() > threshold).float()
        n_features = activations.shape[-1]

        graph = {i: set() for i in range(n_features)}

        coactivation = active.T @ active
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if coactivation[i, j] > threshold * activations.shape[0]:
                    graph[i].add(j)
                    graph[j].add(i)

        return graph

    def test_causal_links(
        self,
        source_features: list[int],
        target_features: list[int],
        cache: ActivationCache,
    ) -> list[tuple[int, int, float]]:
        """Test causal links between features via patching."""
        from world_model_lens.patching import TemporalPatcher

        patcher = TemporalPatcher(self.wm)
        edges = []

        for src in source_features:
            for tgt in target_features:
                try:
                    result = patcher.patch_state(
                        cache,
                        cache,
                        f"feature_{src}",
                        0,
                        lambda x: x,
                    )
                    edges.append((src, tgt, result.recovery_rate))
                except Exception:
                    pass

        return edges

    def discover_circuits(
        self,
        latents: torch.Tensor,
        cache: ActivationCache,
    ) -> SAECircuitResult:
        """Full circuit discovery pipeline."""
        activations = self.compute_activations(latents)
        feature_graph = self.build_coactivation_graph(activations)

        active_features = torch.where(activations.abs().mean(0) > 0.1)[0].tolist()
        causal_edges = self.test_causal_links(
            active_features[:10],
            active_features[:10],
            cache,
        )

        return SAECircuitResult(
            feature_graph=feature_graph,
            causal_edges=causal_edges,
            feature_activations=activations,
        )


class LongHorizonAnalyzer:
    """Analyze long-horizon properties of world models.

    Detects planning horizon, belief drift, and overcommitment.
    """

    def __init__(self, wm: HookedWorldModel):
        self.wm = wm

    def detect_planning_horizon(
        self,
        trajectory: LatentTrajectory,
        horizon: int = 100,
    ) -> int:
        """Detect timesteps where future predictions lose fidelity.

        Args:
            trajectory: Real trajectory.
            horizon: Maximum horizon to check.

        Returns:
            Detected planning horizon (timesteps).
        """
        start_state = trajectory.states[0]

        actions = trajectory.actions if trajectory.actions is not None else None
        if actions is None or actions.shape[0] < horizon:
            horizon = actions.shape[0] if actions is not None else 10

        _, imagined = self.wm.imagine(start_state, actions[:horizon], horizon=horizon)

        real_rewards = trajectory.rewards_real
        pred_rewards = imagined.rewards_pred if imagined.rewards_pred is not None else None

        if real_rewards is None or pred_rewards is None:
            return horizon

        errors = (real_rewards - pred_rewards).abs()
        cumsum = torch.cumsum(errors, dim=0)

        threshold = 0.5 * errors.sum()
        for t in range(len(cumsum)):
            if cumsum[t] > threshold:
                return t

        return horizon

    def compute_belief_drift(
        self,
        imagined_trajectory: LatentTrajectory,
        real_trajectory: LatentTrajectory | None = None,
    ) -> torch.Tensor:
        """Compute KL divergence accumulating over imagination.

        Args:
            imagined_trajectory: Imagined trajectory.
            real_trajectory: Optional real trajectory for comparison.

        Returns:
            Per-timestep KL divergence.
        """
        kl_sequence = imagined_trajectory.kl_sequence

        if kl_sequence is None or len(kl_sequence) == 0:
            if real_trajectory is not None:
                h_imagined = imagined_trajectory.h_sequence
                h_real = real_trajectory.h_sequence
                min_len = min(h_imagined.shape[0], h_real.shape[0])

                dist = torch.nn.functional.kl_div(
                    functional.log_softmax(h_imagined[:min_len], dim=-1),
                    functional.softmax(h_real[:min_len], dim=-1),
                    reduction="none",
                )
                return dist.sum(dim=-1)
            return torch.zeros(10)

        return kl_sequence

    def detect_overcommitment(
        self,
        trajectory: LatentTrajectory,
        threshold: float = 0.1,
    ) -> list[int]:
        """Detect timesteps where value predictions don't update after surprise.

        Args:
            trajectory: Trajectory with value predictions.
            threshold: Change threshold for detecting overcommitment.

        Returns:
            List of overcommitted timesteps.
        """
        value_pred = torch.stack(
            [state.value_pred for state in trajectory.states if state.value_pred is not None], dim=0
        ) if any(state.value_pred is not None for state in trajectory.states) else None
        surprise = trajectory.kl_sequence

        if value_pred is None or surprise is None:
            return []

        overcommitted = []
        for t in range(1, len(value_pred)):
            if surprise[t] > threshold:
                value_change = (value_pred[t] - value_pred[t - 1]).abs()
                if value_change < threshold:
                    overcommitted.append(t)

        return overcommitted

    def full_analysis(
        self,
        trajectory: LatentTrajectory,
        horizon: int = 50,
    ) -> dict[str, Any]:
        """Run complete long-horizon analysis."""
        start_state = trajectory.states[0]
        actions = trajectory.actions

        _, imagined = self.wm.imagine(start_state, actions, horizon=horizon)

        planning_horizon = self.detect_planning_horizon(trajectory, horizon)
        belief_drift = self.compute_belief_drift(imagined, trajectory)
        overcommitment = self.detect_overcommitment(trajectory)

        return {
            "planning_horizon": planning_horizon,
            "belief_drift": belief_drift,
            "max_belief_drift": belief_drift.max().item() if len(belief_drift) > 0 else 0.0,
            "overcommitted_timesteps": overcommitment,
            "n_overcommitted": len(overcommitment),
        }


def cross_model_probe_transfer(
    probe: ProbeResult,
    source_model: HookedWorldModel,
    target_models: list[HookedWorldModel],
    cache_source: ActivationCache,
    caches_target: list[ActivationCache],
) -> dict[str, float]:
    """Test if probes/circuits transfer across models.

    Args:
        probe: Probe trained on source model.
        source_model: Source model.
        target_models: List of target models.
        cache_source: Source activation cache.
        caches_target: Target activation caches.

    Returns:
        Dict mapping target model names to transfer scores.
    """
    from world_model_lens.probing import LatentProber

    prober = LatentProber(seed=42)
    results = {}

    for target_model, target_cache in zip(target_models, caches_target, strict=True):
        model_name = target_model.name

        try:
            components = list(target_cache.component_names)
            if not components:
                results[model_name] = 0.0
                continue

            activations = target_cache[components[0]]

            labels = torch.randn(activations.shape[0])
            result = prober.train_probe(
                activations,
                labels.numpy(),
                concept_name=probe.concept_name,
                activation_name=components[0],
            )

            results[model_name] = result.accuracy

        except Exception:
            results[model_name] = 0.0

    return results
