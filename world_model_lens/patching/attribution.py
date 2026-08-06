"""Direct Attribution (DRA/DPA) for World Models.

This module provides:
- Direct Reward Attribution (DRA): Which latents contribute to reward?
- Direct Pixel Attribution ( DPA): What is the model "imagining" at each layer?
- Latent Attribution: Which z dimensions matter for predictions?

Similar to TransformerLens's Direct Logit Attribution (DLA).
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from dataclasses import dataclass
import numpy as np

from world_model_lens.core.advanced_hooks import AdvancedHookRegistry, HookBuilder


@dataclass
class AttributionResult:
    """Result of direct attribution analysis."""

    attributions: torch.Tensor  # Attribution scores [latent_dim]
    contribution_scores: torch.Tensor  # Per-head/per-layer contributions
    top_dims: List[int]  # Most important dimensions
    top_scores: List[float]  # Scores for top dims
    total_attribution: float  # Sum of all attributions


@dataclass
class ImaginedReconstruction:
    """What the model is 'imagining' at an intermediate layer."""

    layer_name: str
    reconstruction: torch.Tensor
    timestep: int
    fidelity_score: float


class DirectRewardAttribution:
    """Direct Reward Attribution (DRA).

    Answers: "At what exact layer/timestep did the model realize
    it was about to score a point?"

    Example:
        dra = DirectRewardAttribution(world_model)

        # Get attribution from latent state to reward
        result = dra.attribute_to_reward(
            latent_state=z_t,
            target_reward=reward_head,
        )

        # Which z dimensions matter most?
        print(f"Top dims: {result.top_dims}")
    """

    def __init__(
        self,
        world_model: Any,
    ):
        """Initialize DRA.

        Args:
            world_model: HookedWorldModel instance
        """
        self.wm = world_model

    def attribute_to_reward(
        self,
        latent: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        reward_head_name: str = "reward_pred",
    ) -> AttributionResult:
        """Attribute latent dimensions to reward prediction.

        Args:
            latent: Latent state [d_z] or [B, d_z]
            target: Target reward tensor (if computing error attribution)
            reward_head_name: Name of reward head

        Returns:
            AttributionResult with per-dimension contributions
        """
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)

        latent = latent.requires_grad_(True)

        # Get reward prediction
        h = torch.zeros_like(latent)
        reward_pred = self.wm.adapter.predict_reward(h, latent)

        if reward_pred is None:
            raise ValueError("Model does not have reward head")

        # Compute gradient of reward w.r.t. latent
        reward_pred.sum().backward()

        attribution = latent.grad.abs()

        # Get top dimensions
        topk = min(10, attribution.shape[-1])
        scores, indices = torch.topk(attribution, topk, dim=-1)

        return AttributionResult(
            attributions=attribution.squeeze(0),
            contribution_scores=attribution.squeeze(0),
            top_dims=indices.squeeze(0).tolist(),
            top_scores=scores.squeeze(0).tolist(),
            total_attribution=attribution.sum().item(),
        )

    def attribute_through_time(
        self,
        trajectory_cache: Any,
        target_reward: Optional[torch.Tensor] = None,
    ) -> Dict[int, AttributionResult]:
        """Attribute reward across timesteps.

        Args:
            trajectory_cache: ActivationCache with trajectory
            target_reward: Target reward sequence [T]

        Returns:
            Dict mapping timestep to AttributionResult
        """
        results = {}

        try:
            z_posterior = trajectory_cache["z_posterior"]
        except KeyError:
            raise ValueError("Cache must contain z_posterior")

        T = z_posterior.shape[0]

        for t in range(T):
            z_t = z_posterior[t]

            try:
                result = self.attribute_to_reward(z_t)
                results[t] = result
            except Exception:
                pass

        return results

    def compute_attention_weighted_attribution(
        self,
        latent: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Weight attributions by attention patterns.

        Args:
            latent: Latent state [d_z]
            attention_weights: Attention from this token [n_heads]

        Returns:
            Attention-weighted attributions
        """
        attribution = self.attribute_to_reward(latent)

        # Weight by attention
        n_heads = attention_weights.shape[0]
        d_per_head = attribution.attributions.shape[0] // n_heads

        attribution_per_head = attribution.attributions.view(n_heads, d_per_head)
        weighted = (attention_weights.view(-1, 1) * attribution_per_head).flatten()

        return weighted


class DirectPixelAttribution:
    """Direct Pixel Attribution (DPA).

    Answers: "What is the model literally imagining at this layer?"

    Passes intermediate states through decoder to see the "blurry,
    half-formed image the model is holding in its early layers."

    Example:
        dpa = DirectPixelAttribution(world_model)

        # Get imagination at each layer
        for layer in ['encoder.layer_1', 'encoder.layer_2', 'decoder.layer_1']:
            imagined = dpa.imagine_at_layer(
                state=z_t,
                layer_name=layer,
            )
            # Show the blurry image the model is "thinking" of
    """

    def __init__(
        self,
        world_model: Any,
    ):
        """Initialize DPA.

        Args:
            world_model: HookedWorldModel
        """
        self.wm = world_model

    def imagine_at_layer(
        self,
        state: torch.Tensor,
        layer_name: str,
        use_cache: bool = True,
    ) -> ImaginedReconstruction:
        """Pass latent through decoder to see what model imagines.

        Args:
            state: Latent state
            layer_name: Which layer to extract imagination from
            use_cache: Use cached activations

        Returns:
            ImaginedReconstruction with decoded image
        """
        if not hasattr(self.wm, "_layer_activations"):
            self.wm._layer_activations = {}

        h = torch.zeros_like(state).unsqueeze(0) if state.dim() == 1 else state.unsqueeze(0)
        z = state.unsqueeze(0) if state.dim() == 1 else state

        # Run partial forward pass to capture layer
        def hook_fn(name):
            def hook(module, input, output):
                if use_cache:
                    self.wm._layer_activations[name] = output.detach()
                return output

            return hook

        # Register hooks on decoder layers
        handles = []
        for name, module in self.wm.adapter.named_modules():
            if layer_name in name and hasattr(module, "register_forward_hook"):
                handles.append(module.register_forward_hook(hook_fn(name)))

        try:
            # Run decoder
            reconstruction = self.wm.adapter.decode(h.squeeze(0), z.squeeze(0))
        finally:
            for handle in handles:
                handle.remove()

        if layer_name in self.wm._layer_activations:
            layer_output = self.wm._layer_activations[layer_name]
            reconstruction = layer_output

        # Compute fidelity score (how much info preserved)
        fidelity = 1.0
        if hasattr(self.wm.adapter, "decode"):
            full_recon = self.wm.adapter.decode(h.squeeze(0), z.squeeze(0))
            if full_recon is not None and reconstruction is not None:
                fidelity = F.mse_loss(reconstruction, full_recon).item()

        return ImaginedReconstruction(
            layer_name=layer_name,
            reconstruction=reconstruction,
            timestep=0,
            fidelity_score=fidelity,
        )

    def imagine_sequence(
        self,
        latent_sequence: torch.Tensor,
        layers: Optional[List[str]] = None,
    ) -> Dict[str, List[ImaginedReconstruction]]:
        """Get imaginations for a sequence of latents.

        Args:
            latent_sequence: Sequence of latents [T, d_z]
            layers: Which layers to extract

        Returns:
            Dict mapping layer_name to list of ImaginedReconstruction
        """
        layers = layers or ["encoder.out", "decoder.layer_1", "decoder.out"]

        results = {layer: [] for layer in layers}

        for t, z_t in enumerate(latent_sequence):
            for layer in layers:
                try:
                    imagined = self.imagine_at_layer(z_t, layer)
                    imagined.timestep = t
                    results[layer].append(imagined)
                except Exception:
                    pass

        return results


class TemporalPathPatching:
    """Temporal Path Patching.

    Patch information flow between timesteps.
    Not just layer-to-layer, but TIME-to-TIME.

    Example:
        # Agent sees key at t=1, opens door at t=10
        # Patch "no key" at t=1 and see effect at t=10

        tpp = TemporalPathPatching(world_model)

        # Create corrupted trajectory
        corrupted = tpp.patch_timestep(
            original_trajectory=traj,
            patch_timestep=1,
            patch_fn=lambda z: z * 0,  # zero out key representation
        )

        # See how corruption ripples through time
        result = tpp.trace_ripple_effect(
            original_trajectory=traj,
            corrupted_trajectory=corrupted,
            target_timestep=10,
        )
    """

    def __init__(
        self,
        world_model: Any,
    ):
        """Initialize Temporal Path Patching.

        Args:
            world_model: HookedWorldModel
        """
        self.wm = world_model

    def patch_timestep(
        self,
        trajectory: Any,
        patch_timestep: int,
        patch_fn: Callable[[torch.Tensor], torch.Tensor],
        component: str = "z_posterior",
    ) -> Any:
        """Patch a specific timestep in trajectory.

        Args:
            trajectory: WorldTrajectory
            patch_timestep: Which timestep to patch
            patch_fn: Function to modify the latent
            component: Which component to patch

        Returns:
            Patched WorldTrajectory
        """
        from world_model_lens.core.world_trajectory import WorldTrajectory
        from world_model_lens.core.world_state import WorldState

        new_states = []

        for t, state in enumerate(trajectory.states):
            if t == patch_timestep:
                # Apply patch
                patched_state = state.state.clone()
                patched_state = patch_fn(patched_state)

                new_state = WorldState(
                    state=patched_state,
                    timestep=state.timestep,
                    predictions=state.predictions.copy() if hasattr(state, "predictions") else {},
                    obs_encoding=state.obs_encoding,
                )
                new_states.append(new_state)
            else:
                new_states.append(state)

        return WorldTrajectory(
            states=new_states,
            source=trajectory.source + "_patched",
            name=trajectory.name,
        )

    def patch_residual_stream(
        self,
        trajectory: Any,
        patch_timestep: int,
        residual_name: str = "h",
        patch_fn: Optional[Callable] = None,
        patch_value: Optional[torch.Tensor] = None,
    ) -> Any:
        """Patch the residual stream at a specific timestep.

        Args:
            trajectory: WorldTrajectory
            patch_timestep: Timestep to patch
            residual_name: 'h' or 'z'
            patch_fn: Function to apply, or use patch_value
            patch_value: Value to set (if patch_fn is None)

        Returns:
            Patched trajectory
        """
        from world_model_lens.core.world_trajectory import WorldTrajectory
        from world_model_lens.core.world_state import WorldState

        def default_patch(tensor):
            if patch_value is not None:
                return patch_value
            return tensor * 0  # Zero out by default

        patch_fn = patch_fn or default_patch

        new_states = []

        for t, state in enumerate(trajectory.states):
            if t == patch_timestep:
                patched_state = patch_fn(state.state)
                new_state = WorldState(
                    state=patched_state,
                    timestep=state.timestep,
                    predictions=state.predictions.copy() if hasattr(state, "predictions") else {},
                )
                new_states.append(new_state)
            else:
                new_states.append(state)

        return WorldTrajectory(
            states=new_states,
            source=trajectory.source + "_residual_patch",
            name=trajectory.name,
        )

    def trace_ripple_effect(
        self,
        original_trajectory: Any,
        corrupted_trajectory: Any,
        target_timestep: int,
        measure: str = "difference",
    ) -> Dict[str, Any]:
        """Measure how corruption at one timestep affects later timesteps.

        Args:
            original_trajectory: Clean trajectory
            corrupted_trajectory: Trajectory with patch
            target_timestep: Which timestep to measure effect on
            measure: 'difference', 'ratio', or 'cosine'

        Returns:
            Dict with ripple effect metrics
        """
        if target_timestep >= len(original_trajectory.states):
            raise ValueError(f"target_timestep {target_timestep} out of range")

        orig_state = original_trajectory.states[target_timestep]
        corr_state = corrupted_trajectory.states[target_timestep]

        orig_vec = orig_state.state.flatten()
        corr_vec = corr_state.state.flatten()

        if measure == "difference":
            diff = (orig_vec - corr_vec).abs().mean().item()
            return {
                "timestep": target_timestep,
                "ripple_magnitude": diff,
                "relative_change": diff / (orig_vec.abs().mean() + 1e-8),
            }
        elif measure == "cosine":
            cos = F.cosine_similarity(orig_vec.unsqueeze(0), corr_vec.unsqueeze(0)).item()
            return {
                "timestep": target_timestep,
                "cosine_similarity": cos,
                "angular_distance": np.arccos(cos) if abs(cos) < 1 else 0,
            }
        elif measure == "ratio":
            ratio = (corr_vec.abs().sum() / (orig_vec.abs().sum() + 1e-8)).item()
            return {
                "timestep": target_timestep,
                "magnitude_ratio": ratio,
            }
        else:
            raise ValueError(f"Unknown measure: {measure}")

    def full_ripple_analysis(
        self,
        trajectory: Any,
        patch_timestep: int,
        patch_fn: Callable,
    ) -> Dict[int, Dict[str, float]]:
        """Analyze ripple effect across ALL future timesteps.

        Args:
            trajectory: Original trajectory
            patch_timestep: Where to apply patch
            patch_fn: Patch function

        Returns:
            Dict mapping each future timestep to ripple metrics
        """
        corrupted = self.patch_timestep(trajectory, patch_timestep, patch_fn)

        results = {}

        for t in range(patch_timestep + 1, len(trajectory.states)):
            result = self.trace_ripple_effect(
                trajectory,
                corrupted,
                t,
                measure="cosine",
            )
            results[t] = result

        return results


class LatentConceptAblator:
    """Ablate specific latent dimensions (concepts).

    Example:
        # Zero out velocity-related dimensions
        ablator = LatentConceptAblator(world_model)

        ablated_traj = ablator.ablate_concept(
            trajectory=traj,
            concept_dims=[5, 12, 23],  # Dimensions for velocity
        )
    """

    def __init__(self, world_model: Any):
        self.wm = world_model
        self._hook_builder = HookBuilder()

    def ablate_concept(
        self,
        trajectory: Any,
        concept_dims: List[int],
        value: float = 0.0,
    ) -> Any:
        """Ablate specific latent dimensions.

        Args:
            trajectory: Input trajectory
            concept_dims: Which dimensions to ablate
            value: Value to set (default 0)

        Returns:
            Trajectory with ablated concepts
        """
        from world_model_lens.core.world_trajectory import WorldTrajectory
        from world_model_lens.core.world_state import WorldState

        new_states = []

        for state in trajectory.states:
            ablated = state.state.clone()
            ablated[concept_dims] = value

            new_state = WorldState(
                state=ablated,
                timestep=state.timestep,
                predictions=state.predictions.copy() if hasattr(state, "predictions") else {},
            )
            new_states.append(new_state)

        return WorldTrajectory(
            states=new_states,
            source=trajectory.source + "_ablated",
            name=trajectory.name,
        )

    def ablate_by_importance(
        self,
        trajectory: Any,
        attribution_result: AttributionResult,
        top_k: int = 10,
    ) -> Any:
        """Ablate most important dimensions.

        Args:
            trajectory: Input trajectory
            attribution_result: From DirectRewardAttribution
            top_k: Number of top dims to ablate

        Returns:
            Ablated trajectory
        """
        dims = attribution_result.top_dims[:top_k]
        return self.ablate_concept(trajectory, dims)


class AdvancedCausalTracer:
    """Advanced Causal Tracing using 3-level hook system.

    Uses spatial, temporal, and transition hooks for precise causal analysis.
    Provides finer-grained control than standard causal tracing.

    Example:
        tracer = AdvancedCausalTracer(world_model)

        # Patch specific dimensions at specific timesteps
        results = tracer.trace_spatial(
            observations=obs,
            corrupt_dims=[0, 1, 2],
            target_timestep=10,
        )

        # Trace temporal causality
        results = tracer.trace_temporal(
            observations=obs,
            source_timestep=1,
            target_timestep=10,
        )
    """

    def __init__(self, world_model: Any):
        self.wm = world_model

    def trace_spatial(
        self,
        observations: torch.Tensor,
        corrupt_dims: List[int],
        target_timestep: int = 0,
        corrupt_value: float = 0.0,
    ) -> Dict[str, Any]:
        """Trace causal effect of corrupting specific latent dimensions.

        Uses z[dim_start:dim_end] hooks for precise spatial intervention.

        Args:
            observations: Input observations [T, ...]
            corrupt_dims: Dimensions to corrupt
            target_timestep: Which timestep to corrupt
            corrupt_value: Value to replace with

        Returns:
            Dict with clean vs corrupted metrics
        """
        hook_spec = f"z[{min(corrupt_dims)}:{max(corrupt_dims) + 1}]"
        if target_timestep > 0:
            hook_spec = f"t={target_timestep}.{hook_spec}"

        corrupt_hook = self.wm.hook_builder.ablate_dims(corrupt_dims, corrupt_value)

        clean_traj, _, clean_cache = self.wm.run_with_cache(observations)

        corrupted_traj = self.wm.run_with_advanced_hooks(
            observations,
            hook_specs={hook_spec: corrupt_hook},
        )

        return {
            "hook_spec": hook_spec,
            "corrupted_dims": corrupt_dims,
            "target_timestep": target_timestep,
            "clean_trajectory": clean_traj,
            "corrupted_trajectory": corrupted_traj,
        }

    def trace_temporal(
        self,
        observations: torch.Tensor,
        source_timestep: int,
        target_timestep: int,
        patch_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Trace causal flow between timesteps.

        Uses t=T.z hooks to intercept and modify specific timesteps.

        Args:
            observations: Input observations
            source_timestep: Where corruption originates
            target_timestep: Where effect is measured
            patch_fn: Function to apply at source timestep

        Returns:
            Dict with causal flow metrics
        """
        hook_spec = f"t={source_timestep}.z"

        def default_patch(tensor, ctx):
            return tensor * 0

        patch_fn = patch_fn or default_patch

        clean_traj, _, _ = self.wm.run_with_cache(observations)

        corrupted_traj = self.wm.run_with_advanced_hooks(
            observations,
            hook_specs={hook_spec: patch_fn},
        )

        return {
            "source_timestep": source_timestep,
            "target_timestep": target_timestep,
            "clean_trajectory": clean_traj,
            "corrupted_trajectory": corrupted_traj,
        }

    def trace_transition(
        self,
        observations: torch.Tensor,
        stage: str = "pre",
        patch_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Trace effect of intervening in transition dynamics.

        Uses transition.pre/transition.post hooks.

        Args:
            observations: Input observations
            stage: "pre" or "post" transition
            patch_fn: Function to apply

        Returns:
            Dict with transition analysis
        """
        hook_spec = f"transition.{stage}"

        def default_patch(tensor, ctx):
            return tensor * 0

        patch_fn = patch_fn or default_patch

        clean_traj, _, _ = self.wm.run_with_cache(observations)

        intervened_traj = self.wm.run_with_advanced_hooks(
            observations,
            hook_specs={hook_spec: patch_fn},
        )

        return {
            "stage": stage,
            "clean_trajectory": clean_traj,
            "intervened_trajectory": intervened_traj,
        }
