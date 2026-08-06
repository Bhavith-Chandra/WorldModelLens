"""Advanced Hook System - Spatial, Temporal, and Transition Hooks.

This is the CORE differentiator for WorldModelLens. Supports ALL hook levels:

1. SPATIAL HOOKS: hook(z_t) - latent dimension level
   - neuron-level intervention
   - feature suppression
   - dimension ablation

2. TEMPORAL HOOKS: hook(z_t, t) - time-indexed
   - "only modify at timestep 5"
   - memory analysis
   - delayed causality

3. TRANSITION HOOKS: hook_pre/hook_post
   - hook_pre(z_t, a_t) - before transition
   - hook_post(z_t+1) - after transition
   - isolate encoding vs dynamics issues

4. HOOK API (TransformerLens-style):
   run_with_hooks(
       inputs,
       hooks=[
           ("z[0:10]", hook_fn),        # spatial: first 10 dims
           ("t=5.z", hook_fn),          # temporal: timestep 5
           ("transition.pre", hook_fn),  # pre-transition
           ("transition.post", hook_fn), # post-transition
       ]
   )
"""

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
import torch
import re
import numpy as np


@dataclass
class HookSpec:
    """Specification for a hook.

    Parsed from string like "z[0:10]" or "t=5.transition.post"
    """

    component: str  # 'z', 'encoder', 'transition', 'reward', etc.
    spatial_slice: Optional[slice] = None  # e.g., [0:10] for dimensions
    temporal_slice: Optional[slice] = None  # e.g., t=5 or t=0:10
    stage: str = "forward"  # "pre", "post", "forward"
    position: str = ""  # "pre_transition", "post_transition"

    @classmethod
    def parse(cls, hook_string: str) -> "HookSpec":
        """Parse hook string to specification.

        Examples:
            "z"                    -> component='z', no filtering
            "z[0:10]"              -> component='z', dims 0-10
            "t=5.z"                -> component='z', timestep 5
            "t=0:10.z"             -> component='z', timesteps 0-10
            "transition.pre"        -> component='transition', pre-transition
            "transition.post"      -> component='transition', post-transition
            "t=5.transition.pre"    -> timestep 5, pre-transition
            "encoder.layer_1"      -> encoder layer 1
            "reward"               -> reward head output
        """
        spec = cls(component="")

        # Parse stage/position
        if ".pre" in hook_string:
            hook_string = hook_string.replace(".pre", "")
            spec.stage = "pre"
            spec.position = "pre_transition"
        elif ".post" in hook_string:
            hook_string = hook_string.replace(".post", "")
            spec.stage = "post"
            spec.position = "post_transition"

        # Parse spatial slice [dim_start:dim_end]
        spatial_match = re.search(r"\[(\d+)?:(\d+)?\]", hook_string)
        if spatial_match:
            start = int(spatial_match.group(1)) if spatial_match.group(1) else 0
            end = int(spatial_match.group(2)) if spatial_match.group(2) else None
            spec.spatial_slice = slice(start, end)
            hook_string = re.sub(r"\[(\d+)?:(\d+)?\]", "", hook_string)

        # Parse temporal t=5 or t=0:10
        temporal_match = re.search(r"t=(\d+):(\d+)", hook_string)
        if temporal_match:
            start = int(temporal_match.group(1))
            end = int(temporal_match.group(2))
            spec.temporal_slice = slice(start, end)
            hook_string = re.sub(r"t=\d+:\d+", "", hook_string)
        else:
            temporal_match = re.search(r"t=(\d+)", hook_string)
            if temporal_match:
                t = int(temporal_match.group(1))
                spec.temporal_slice = slice(t, t + 1)
                hook_string = re.sub(r"t=\d+", "", hook_string)

        # Remaining is component
        component = hook_string.strip(".")
        spec.component = component

        return spec

    def matches(self, component: str, timestep: int, dim: Optional[int] = None) -> bool:
        """Check if this hook matches the given context."""
        # Component match
        if self.component and not component.startswith(self.component):
            return False

        # Temporal match
        if self.temporal_slice is not None:
            if timestep < self.temporal_slice.start or timestep >= self.temporal_slice.stop:
                return False

        # Spatial match
        if self.spatial_slice is not None and dim is not None:
            if dim < self.spatial_slice.start or dim >= self.spatial_slice.stop:
                return False

        return True


@dataclass
class Hook:
    """A registered hook function.

    Similar to TransformerLens hooks.
    """

    name: str
    spec: HookSpec
    fn: Callable[[torch.Tensor, "HookCallContext"], torch.Tensor]
    is_permanent: bool = False
    is_conditional: bool = False
    condition_fn: Optional[Callable[["HookCallContext"], bool]] = None
    priority: int = 0  # Execution order


@dataclass
class HookCallContext:
    """Context passed to hook functions at call time.

    Provides full context about where/when the hook fired.
    """

    component: str  # Component name like 'z', 'encoder', 'transition'
    timestep: int  # Current timestep t
    stage: str  # "pre_transition", "post_transition", "forward"
    dim: Optional[int]  # Dimension index if spatial hook
    forward_stack: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def t(self) -> int:
        """Shorthand for timestep."""
        return self.timestep


class AdvancedHookRegistry:
    """Advanced hook registry supporting spatial, temporal, and transition hooks.

    Example:
        registry = AdvancedHookRegistry()

        # Spatial: ablate first 10 dimensions
        registry.add_hook("z[0:10]", lambda x, ctx: x * 0)

        # Temporal: only modify at timestep 5
        registry.add_hook("t=5.z", my_hook)

        # Transition: before dynamics
        registry.add_hook("transition.pre", pre_hook)

        # After dynamics
        registry.add_hook("transition.post", post_hook)
    """

    def __init__(self):
        self.hooks: List[Hook] = []
        self._hook_counts: Dict[str, int] = {}

    def add_hook(
        self,
        hook_name: str,
        fn: Callable[[torch.Tensor, HookCallContext], torch.Tensor],
        is_permanent: bool = False,
        priority: int = 0,
    ) -> None:
        """Add a hook.

        Args:
            hook_name: Hook string like "z[0:10]" or "t=5.transition.pre"
            fn: Hook function(tensor, context) -> tensor
            is_permanent: Keep across runs
            priority: Execution order (higher runs first)
        """
        spec = HookSpec.parse(hook_name)
        hook = Hook(
            name=hook_name,
            spec=spec,
            fn=fn,
            is_permanent=is_permanent,
            priority=priority,
        )
        self.hooks.append(hook)
        self.hooks.sort(key=lambda h: -h.priority)
        self._hook_counts[hook_name] = self._hook_counts.get(hook_name, 0) + 1

    def add_hooks(
        self,
        hooks: Dict[str, Callable],
    ) -> None:
        """Add multiple hooks.

        Args:
            hooks: Dict mapping hook names to functions
        """
        for name, fn in hooks.items():
            self.add_hook(name, fn)

    def get_matching_hooks(
        self,
        component: str,
        timestep: int,
        stage: str = "forward",
        dim: Optional[int] = None,
    ) -> List[Hook]:
        """Get all hooks matching the current context.

        Args:
            component: Component name
            timestep: Current timestep
            stage: "pre_transition", "post_transition", "forward"
            dim: Dimension index for spatial hooks

        Returns:
            List of matching hooks in priority order
        """
        matching = []

        for hook in self.hooks:
            if hook.spec.matches(component, timestep, dim):
                # Check conditional
                if hook.is_conditional and hook.condition_fn:
                    ctx = HookCallContext(
                        component=component,
                        timestep=timestep,
                        stage=stage,
                        dim=dim,
                    )
                    if not hook.condition_fn(ctx):
                        continue
                matching.append(hook)

        return matching

    def apply_hooks(
        self,
        tensor: torch.Tensor,
        component: str,
        timestep: int,
        stage: str = "forward",
    ) -> torch.Tensor:
        """Apply all matching hooks to tensor.

        Args:
            tensor: Input tensor
            component: Component name
            timestep: Current timestep
            stage: "pre_transition", "post_transition", "forward"

        Returns:
            Tensor after all hooks applied
        """
        matching = self.get_matching_hooks(component, timestep, stage)

        if not matching:
            return tensor

        result = tensor
        for hook in matching:
            ctx = HookCallContext(
                component=component,
                timestep=timestep,
                stage=stage,
                dim=None,
            )
            try:
                result = hook.fn(result, ctx)
            except Exception as e:
                print(f"Hook {hook.name} failed: {e}")

        return result

    def apply_spatial_hooks(
        self,
        tensor: torch.Tensor,
        component: str,
        timestep: int,
        stage: str = "forward",
    ) -> torch.Tensor:
        """Apply hooks with spatial dimension awareness.

        Handles hooks like "z[0:10]" that target specific dimensions.

        Args:
            tensor: Input tensor [..., d_z]
            component: Component name
            timestep: Current timestep
            stage: Stage

        Returns:
            Tensor after hooks applied
        """
        result = tensor

        # First apply non-spatial hooks
        non_spatial = [
            h
            for h in self.get_matching_hooks(component, timestep, stage)
            if h.spec.spatial_slice is None
        ]

        for hook in non_spatial:
            ctx = HookCallContext(
                component=component,
                timestep=timestep,
                stage=stage,
            )
            try:
                result = hook.fn(result, ctx)
            except Exception:
                pass

        # Then apply spatial hooks
        spatial_hooks = [
            h
            for h in self.get_matching_hooks(component, timestep, stage)
            if h.spec.spatial_slice is not None
        ]

        if spatial_hooks and result.dim() > 0:
            d_z = result.shape[-1]

            # Split into spatial and non-spatial parts
            for hook in spatial_hooks:
                slc = hook.spec.spatial_slice
                slc = slice(slc.start or 0, min(slc.stop or d_z, d_z))

                ctx = HookCallContext(
                    component=component,
                    timestep=timestep,
                    stage=stage,
                    dim=slc.start,
                )

                # Get parts before, inside, after
                if slc.start > 0:
                    before = result[..., : slc.start]
                else:
                    before = None

                if slc.stop < d_z:
                    after = result[..., slc.stop :]
                else:
                    after = None

                inside = result[..., slc]

                try:
                    inside = hook.fn(inside, ctx)
                except Exception:
                    pass

                # Reassemble
                parts = []
                if before is not None:
                    parts.append(before)
                parts.append(inside)
                if after is not None:
                    parts.append(after)

                result = torch.cat(parts, dim=-1)

        return result

    def remove_hook(self, name: str) -> bool:
        """Remove hook by name."""
        for i, hook in enumerate(self.hooks):
            if hook.name == name:
                self.hooks.pop(i)
                return True
        return False

    def clear(self, permanent_only: bool = False) -> None:
        """Clear hooks.

        Args:
            permanent_only: Only clear non-permanent hooks
        """
        if permanent_only:
            self.hooks = [h for h in self.hooks if h.is_permanent]
        else:
            self.hooks.clear()

    def __len__(self) -> int:
        return len(self.hooks)

    def __repr__(self) -> str:
        return f"AdvancedHookRegistry({len(self.hooks)} hooks)"


class SpatialHookMixin:
    """Mixin providing spatial hook utilities."""

    def hook_z(
        self,
        slice_or_dim: Union[int, slice],
        fn: Callable,
    ) -> Callable:
        """Create hook for specific z dimensions.

        Args:
            slice_or_dim: Dimension index or slice like 5 or [0:10]
            fn: Hook function

        Returns:
            Hook function
        """
        if isinstance(slice_or_dim, int):
            spec = f"z[{slice_or_dim}:{slice_or_dim + 1}]"
        else:
            spec = f"z[{slice_or_dim.start}:{slice_or_dim.stop}]"

        def wrapper(tensor, ctx):
            # Apply to specific dimension
            result = tensor.clone()
            result[..., slice_or_dim] = fn(tensor[..., slice_or_dim], ctx)
            return result

        return wrapper

    def ablate_dims(
        self,
        dims: List[int],
        value: float = 0.0,
    ) -> Callable:
        """Create hook to ablate specific dimensions.

        Args:
            dims: Dimensions to ablate
            value: Value to set (default 0)

        Returns:
            Hook function
        """
        dims_set = set(dims)

        def ablate_hook(tensor, ctx):
            result = tensor.clone()
            for d in dims_set:
                if d < result.shape[-1]:
                    result[..., d] = value
            return result

        return ablate_hook

    def keep_dims(
        self,
        dims: List[int],
    ) -> Callable:
        """Create hook to zero out all but specified dimensions.

        Args:
            dims: Dimensions to keep

        Returns:
            Hook function
        """
        dims_set = set(dims)
        d_total = None

        def keep_hook(tensor, ctx):
            nonlocal d_total
            if d_total is None:
                d_total = tensor.shape[-1]
            result = torch.zeros_like(tensor)
            for d in dims_set:
                if d < d_total:
                    result[..., d] = tensor[..., d]
            return result

        return keep_hook


class TemporalHookMixin:
    """Mixin providing temporal hook utilities."""

    def hook_at_t(
        self,
        t: int,
        fn: Callable,
    ) -> Callable:
        """Create hook that only fires at specific timestep.

        Args:
            t: Timestep
            fn: Hook function

        Returns:
            Hook function with condition
        """

        def conditional_fn(ctx: HookCallContext) -> bool:
            return ctx.timestep == t

        fn._temporal_condition = conditional_fn
        return fn

    def hook_range(
        self,
        t_start: int,
        t_end: int,
        fn: Callable,
    ) -> Callable:
        """Create hook that fires for timestep range.

        Args:
            t_start: Start timestep (inclusive)
            t_end: End timestep (exclusive)
            fn: Hook function

        Returns:
            Hook function with condition
        """

        def conditional_fn(ctx: HookCallContext) -> bool:
            return t_start <= ctx.timestep < t_end

        fn._temporal_condition = conditional_fn
        return fn

    def hook_first_n(
        self,
        n: int,
        fn: Callable,
    ) -> Callable:
        """Create hook for first n timesteps."""
        return self.hook_range(0, n, fn)

    def hook_last_n(
        self,
        n: int,
        total_length: int,
        fn: Callable,
    ) -> Callable:
        """Create hook for last n timesteps."""
        return self.hook_range(total_length - n, total_length, fn)


class TransitionHookMixin:
    """Mixin providing transition hook utilities."""

    def hook_pre_transition(
        self,
        fn: Callable,
    ) -> Callable:
        """Hook before transition function.

        Use for:
        - Intervening on z_t before dynamics
        - Action inspection
        - Premature prediction
        """

        def wrapper(tensor, ctx):
            if ctx.stage == "pre_transition":
                return fn(tensor, ctx)
            return tensor

        wrapper._transition_stage = "pre"
        return wrapper

    def hook_post_transition(
        self,
        fn: Callable,
    ) -> Callable:
        """Hook after transition function.

        Use for:
        - Inspecting z_t+1
        - Detecting dynamics instability
        - Output monitoring
        """

        def wrapper(tensor, ctx):
            if ctx.stage == "post_transition":
                return fn(tensor, ctx)
            return tensor

        wrapper._transition_stage = "post"
        return wrapper

    def hook_encoder(
        self,
        fn: Callable,
    ) -> Callable:
        """Hook encoder output (z_t)."""

        def wrapper(tensor, ctx):
            if "encoder" in ctx.component or ctx.component == "z":
                return fn(tensor, ctx)
            return tensor

        return wrapper

    def hook_dynamics(
        self,
        fn: Callable,
    ) -> Callable:
        """Hook dynamics/transition output."""

        def wrapper(tensor, ctx):
            if "transition" in ctx.component or "dynamics" in ctx.component:
                return fn(tensor, ctx)
            return tensor

        return wrapper

    def hook_decoder(
        self,
        fn: Callable,
    ) -> Callable:
        """Hook decoder output."""

        def wrapper(tensor, ctx):
            if "decoder" in ctx.component or "reconstruction" in ctx.component:
                return fn(tensor, ctx)
            return tensor

        return wrapper


# Composite mixin for convenience
class HookBuilder(
    SpatialHookMixin,
    TemporalHookMixin,
    TransitionHookMixin,
):
    """Fluent API for building hooks.

    Example:
        builder = HookBuilder()

        hooks = {
            "z[0:10]": builder.ablate_dims([0,1,2,3,4]),
            "t=5.z": builder.hook_at_t(5, my_hook),
            "transition.pre": builder.hook_pre_transition(inspect_action),
        }
    """

    pass
