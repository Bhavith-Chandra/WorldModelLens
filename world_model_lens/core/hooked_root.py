"""HookedRootModule - The core hook injection system.

This is the heart of WorldModelLens's hook system. It recursively searches through
a PyTorch model, injects hooks at every layer, and provides a unified namespace.

Key concepts:
- Residual Stream: The central highway of hidden state (h_t) and latent (z_t)
- Hook Dictionary: Standardized names for every component
- HookedRootModule: Base class that wraps any world model
"""

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from collections import OrderedDict

# Adapter imports for the timestep-oriented hook API
from world_model_lens.core import hooks as core_hooks


@dataclass
class ModuleHookPoint:
    """A named hook function registered at a specific component.

    This module-level hook type is intentionally separate from the
    timestep-oriented HookPoint defined in `world_model_lens.core.hooks`.
    Keeping distinct types clarifies the different hook semantics used by
    HookedRootModule (module instrumentation) vs. timestep hooks (sequence
    interventions).
    """

    name: str
    fn: Callable[[torch.Tensor, "ModuleHookContext"], torch.Tensor]
    is_permanent: bool = False
    is_conditional: bool = False
    condition_fn: Optional[Callable[["ModuleHookContext"], bool]] = None


@dataclass
class ModuleHookContext:
    """Runtime metadata passed to module-level hook functions.

    Fields describe where in the module hierarchy the hook fired. This is
    different from the timestep-oriented `HookContext` in
    `world_model_lens.core.hooks` which focuses on sequence/timestep info.
    """

    component_type: str  # e.g., 'encoder', 'dynamics', 'reward_head'
    component_name: str  # e.g., 'encoder.layer_1'
    layer_index: Optional[int]  # Layer number if applicable
    timestep: int  # Current timestep in sequence
    forward_stack: List[str]  # Path through model tree
    metadata: Dict[str, Any] = field(default_factory=dict)


class HookedRootModule(nn.Module):
    """Base class that wraps any world model with hook injection.

    This is the TransformerLens equivalent. It provides:
    1. Recursive module search and hook injection
    2. Standardized hook namespace
    3. Residual stream tracking
    4. Easy intervention APIs

    Example:
        class MyWorldModel(HookedRootModule):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.setup_hooks()  # Must call to inject hooks

        wm = MyWorldModel(my_model)
        cache = wm.run_with_cache(observations)

        # Use standardized hook names
        z_hook = cache["dynamics.prior.hook_sample"]
    """

    # Per-instance registries and metadata. These are created in __init__ to
    # avoid accidental sharing between instances (dataclasses.field was
    # incorrectly used here previously and returns a dataclass.Field object).

    def __init__(self):
        super().__init__()
        self._hooks_installed = False
        self._hook_handles: List[Any] = []
        # Initialize per-instance storage for hooks and metadata.
        # Use a WML-specific name to avoid shadowing nn.Module internals.
        self._wml_forward_hooks: Dict[str, List[ModuleHookPoint]] = {}
        self._hook_metadata: Dict[str, ModuleHookContext] = {}
        self._residual_streams: Dict[str, torch.Tensor] = {}
        self._name_to_module: Dict[str, nn.Module] = {}
        self._module_to_name: Dict[int, str] = {}
        # Map core HookPoint instances to ModuleHookPoint wrappers so we can
        # remove temporary hooks registered via the core API.
        self._core_to_module_map: Dict[int, ModuleHookPoint] = {}

    # ==================== HOOK NAMESPACE ====================
    # Standardized naming convention (TransformerLens-style)

    STANDARDIZED_HOOK_NAMES = {
        # Observation Encoder
        "encoder.layer_{i}.hook_input": "encoder layer {i} input",
        "encoder.layer_{i}.hook_output": "encoder layer {i} output",
        "encoder.layer_{i}.hook_residual": "encoder layer {i} residual stream",
        "encoder.hook_out": "final encoder output",
        # Latent Posterior
        "posterior.hook_sample": "sampled latent",
        "posterior.hook_logits": "posterior logits before sampling",
        "posterior.hook_pre": "before sampling activation",
        # Latent Prior (dynamics)
        "dynamics.hook_hidden": "GRU/RNN hidden state",
        "dynamics.hook_prior": "prior distribution",
        "dynamics.hook_sample": "sampled prior",
        "dynamics.layer_{i}.hook_output": "dynamics layer {i} output",
        # Transition
        "transition.hook_input": "transition input (h, z, action)",
        "transition.hook_output": "transition output (next h)",
        # Decoders
        "decoder.hook_input": "decoder input",
        "decoder.hook_output": "decoder output (reconstruction)",
        "decoder.reward.hook_out": "reward prediction",
        "decoder.value.hook_out": "value prediction",
        "decoder.image.layer_{i}.hook_out": "image decoder layer {i}",
        # Residual Stream (the central highway)
        "residual.hook_h": "hidden state residual stream (h_t)",
        "residual.hook_z": "latent residual stream (z_t)",
        "residual.hook_h_before": "h before transition",
        "residual.hook_z_before": "z before transition",
        "residual.hook_h_after": "h after transition",
        "residual.hook_z_after": "z after transition",
        # Attention (for transformer-based world models)
        "attn.hook_query": "attention query",
        "attn.hook_key": "attention key",
        "attn.hook_value": "attention value",
        "attn.hook_pattern": "attention pattern",
        "attn.hook_z": "attention output (z = attention @ value)",
    }

    def setup_hooks(self) -> None:
        """Recursively search model and inject hooks.

        This must be called after model is fully constructed.
        """
        self._name_to_module.clear()
        self._module_to_name.clear()
        self._hook_metadata.clear()
        self._wml_forward_hooks.clear()

        self._recursive_register(self, name_prefix="")
        self._hooks_installed = True

    def _recursive_register(
        self,
        module: nn.Module,
        name_prefix: str,
        depth: int = 0,
    ) -> None:
        """Recursively register all submodules with standardized names."""

        for name, child in module.named_children():
            full_name = f"{name_prefix}.{name}" if name_prefix else name

            self._name_to_module[full_name] = child
            self._module_to_name[id(child)] = full_name

            # Categorize by component type
            component_type = self._categorize_component(name)

            # Create hook context for this module
            self._hook_metadata[full_name] = ModuleHookContext(
                component_type=component_type,
                component_name=full_name,
                layer_index=self._get_layer_index(name),
                timestep=0,
                forward_stack=[full_name],
            )

            # Recurse into children
            self._recursive_register(child, full_name, depth + 1)

            # Register hooks for leaf modules (layers)
            self._register_layer_hooks(child, full_name, component_type)

    def _categorize_component(self, name: str) -> str:
        """Categorize component by name pattern."""
        name_lower = name.lower()

        if "encoder" in name_lower or "embed" in name_lower:
            return "encoder"
        elif "posterior" in name_lower or "encode" in name_lower:
            return "posterior"
        elif "prior" in name_lower or "dynamics" in name_lower or "transition" in name_lower:
            return "dynamics"
        elif "decoder" in name_lower or "reconstruct" in name_lower:
            return "decoder"
        elif "reward" in name_lower:
            return "reward_head"
        elif "value" in name_lower or "critic" in name_lower:
            return "value_head"
        elif "actor" in name_lower or "policy" in name_lower:
            return "policy_head"
        elif "hidden" in name_lower or "h" in name_lower:
            return "hidden_state"
        elif "z" in name_lower or "latent" in name_lower:
            return "latent_state"
        elif "attn" in name_lower or "attention" in name_lower:
            return "attention"
        else:
            return "unknown"

    def _get_layer_index(self, name: str) -> Optional[int]:
        """Extract layer index from name like 'layer_1' or 'blocks.0'."""
        import re

        patterns = [r"layer_(\d+)", r"blocks?\.(\d+)", r"(\d+)\."]
        for pattern in patterns:
            match = re.search(pattern, name)
            if match:
                return int(match.group(1))
        return None

    def _register_layer_hooks(
        self,
        module: nn.Module,
        name: str,
        component_type: str,
    ) -> None:
        """Register forward hooks for a layer."""

        def create_hook(hook_name: str, stage: str):
            def hook(mod, input, output):
                ctx = self._hook_metadata.get(
                    name,
                    ModuleHookContext(
                        component_type=component_type,
                        component_name=name,
                        layer_index=None,
                        timestep=0,
                        forward_stack=[name],
                    ),
                )

                # Run registered hooks
                for hp in self._wml_forward_hooks.get(hook_name, []):
                    if hp.is_conditional and hp.condition_fn:
                        if not hp.condition_fn(ctx):
                            continue
                    output = hp.fn(output, ctx)

                return output

            return hook

        # Register hooks for different stages
        if isinstance(module, nn.Linear):
            handle = module.register_forward_hook(create_hook(f"{name}.hook_linear", "forward"))
            self._hook_handles.append(handle)
        elif isinstance(module, nn.Conv2d):
            handle = module.register_forward_hook(create_hook(f"{name}.hook_conv", "forward"))
            self._hook_handles.append(handle)
        elif isinstance(module, nn.GRU) or isinstance(module, nn.LSTM):
            handle = module.register_forward_hook(create_hook(f"{name}.hook_rnn", "forward"))
            self._hook_handles.append(handle)
        elif isinstance(module, nn.MultiheadAttention):
            handle = module.register_forward_hook(create_hook(f"{name}.hook_attn", "forward"))
            self._hook_handles.append(handle)
        elif isinstance(module, nn.Identity) and "hook_" in name:
            # Native support for explicitly placed HookPoints using Identity
            handle = module.register_forward_hook(create_hook(name, "forward"))
            self._hook_handles.append(handle)

    # ==================== HOOK REGISTRATION API ====================

    def add_hook(
        self,
        name: str,
        fn: Callable[[torch.Tensor, ModuleHookContext], torch.Tensor],
        is_permanent: bool = False,
    ) -> None:
        """Add a hook to a specific component.

        Args:
            name: Standardized hook name (e.g., 'dynamics.prior.hook_sample')
            fn: Hook function
            is_permanent: Keep hook across runs
        """
        hook = ModuleHookPoint(name=name, fn=fn, is_permanent=is_permanent)

        if name not in self._wml_forward_hooks:
            self._wml_forward_hooks[name] = []
        self._wml_forward_hooks[name].append(hook)

    # --- Compatibility shims for the timestep-oriented Hook API ---
    def add_core_hook(self, hook: core_hooks.HookPoint, prepend: bool = False) -> None:
        """Adapter: register a core timestep HookPoint as a module-level hook.

        This wraps the core HookPoint into a ModuleHookPoint and stores a
        mapping so it can be removed later via remove_core_hook.
        """

        # Wrap core hook function to adapt signature
        def wrapper(tensor, ctx: ModuleHookContext):
            # Convert ModuleHookContext -> core HookContext
            core_ctx = core_hooks.HookContext(
                timestep=ctx.timestep,
                component=ctx.component_name,
                trajectory_so_far=[],
                metadata={},
            )
            return hook.fn(tensor, core_ctx)

        mhook = ModuleHookPoint(name=hook.name, fn=wrapper, is_permanent=False)
        if hook.name not in self._wml_forward_hooks:
            self._wml_forward_hooks[hook.name] = []
        if prepend:
            self._wml_forward_hooks[hook.name].insert(0, mhook)
        else:
            self._wml_forward_hooks[hook.name].append(mhook)

        # remember mapping for removal
        self._core_to_module_map[id(hook)] = mhook

    def remove_core_hook(self, hook: core_hooks.HookPoint) -> None:
        """Remove a core HookPoint previously added via add_core_hook."""
        mhook = self._core_to_module_map.pop(id(hook), None)
        if mhook is None:
            return
        lst = self._wml_forward_hooks.get(mhook.name, [])
        try:
            lst.remove(mhook)
        except ValueError:
            pass

    def add_hooks(
        self,
        hooks: Dict[str, Callable],
        is_permanent: bool = False,
    ) -> None:
        """Add multiple hooks at once.

        Args:
            hooks: Dict mapping hook names to functions
            is_permanent: Keep hooks across runs
        """
        for name, fn in hooks.items():
            self.add_hook(name, fn, is_permanent)

    def remove_hook(self, name: str) -> None:
        """Remove hooks from a component."""
        # Maintain legacy API name but operate on the WML-specific registry.
        self._wml_forward_hooks.pop(name, None)

    def clear_hooks(self, permanent_only: bool = False) -> None:
        """Clear all hooks."""
        if permanent_only:
            self._wml_forward_hooks = {
                k: [h for h in v if h.is_permanent] for k, v in self._wml_forward_hooks.items()
            }
            self._wml_forward_hooks = {k: v for k, v in self._wml_forward_hooks.items() if v}
        else:
            self._wml_forward_hooks.clear()

        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    # ==================== RESIDUAL STREAM TRACKING ====================

    def track_residual(
        self,
        name: str,
        tensor: torch.Tensor,
        timestep: int,
    ) -> torch.Tensor:
        """Track a tensor in the residual stream.

        The residual stream is the "central highway" of the world model:
        - h_t: Deterministic hidden state
        - z_t: Stochastic latent state

        Args:
            name: Stream name (e.g., 'residual.hook_h')
            tensor: Tensor to track
            timestep: Current timestep

        Returns:
            The tensor unchanged (for use in forward pass)
        """
        key = f"{name}_t{timestep}"
        self._residual_streams[key] = tensor.detach()
        return tensor

    def get_residual(
        self,
        name: str,
        timestep: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """Get tracked residual stream tensor.

        Args:
            name: Stream name
            timestep: Specific timestep or None for latest

        Returns:
            Tracked tensor or None
        """
        if timestep is not None:
            return self._residual_streams.get(f"{name}_t{timestep}")
        else:
            # Return latest
            matching = [v for k, v in self._residual_streams.items() if k.startswith(name)]
            return matching[-1] if matching else None

    def get_residual_stack(
        self,
        name: str,
    ) -> torch.Tensor:
        """Get all timesteps of a residual stream.

        Args:
            name: Stream name

        Returns:
            Stacked tensor [T, ...]
        """
        tensors = []
        for key in sorted(self._residual_streams.keys()):
            if key.startswith(name):
                tensors.append(self._residual_streams[key])
        return torch.stack(tensors, dim=0) if tensors else torch.tensor([])

    # ==================== MODULE QUERY API ====================

    def get_module(self, name: str) -> Optional[nn.Module]:
        """Get module by standardized name."""
        return self._name_to_module.get(name)

    def get_hook_context(self, name: str) -> Optional[ModuleHookContext]:
        """Get metadata for a hook point."""
        return self._hook_metadata.get(name)

    def list_hooks(self) -> List[str]:
        """List all available hook points."""
        return list(self._hook_metadata.keys())

    def list_components(self, component_type: Optional[str] = None) -> List[str]:
        """List components of a specific type.

        Args:
            component_type: Filter by type (e.g., 'encoder', 'dynamics')
        """
        if component_type is None:
            return list(self._hook_metadata.keys())
        return [
            name
            for name, ctx in self._hook_metadata.items()
            if ctx.component_type == component_type
        ]

    # ==================== FORWARD PASS WITH HOOKS ====================

    def forward_with_hooks(
        self,
        *args,
        **kwargs,
    ) -> Any:
        """Run forward pass with all hooks active."""
        self._residual_streams.clear()
        return self(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> Any:
        """Override in subclass."""
        raise NotImplementedError


def standardize_name(name: str) -> str:
    """Convert PyTorch name to standardized WML name.

    Args:
        name: Original PyTorch name like 'layers.0.self_attn.q_proj'

    Returns:
        Standardized name like 'blocks.0.attn.hook_q'
    """
    name = name.lower()

    replacements = [
        (r"layers?\.(\d+)\.self_attn\.q", "blocks.{0}.attn.hook_query"),
        (r"layers?\.(\d+)\.self_attn\.k", "blocks.{0}.attn.hook_key"),
        (r"layers?\.(\d+)\.self_attn\.v", "blocks.{0}.attn.hook_value"),
        (r"layers?\.(\d+)\.self_attn", "blocks.{0}.attn"),
        (r"layers?\.(\d+)\.mlp", "blocks.{0}.mlp"),
        (r"encoder\.(\d+)", "encoder.layer_{0}"),
        (r"gru|rnn|lstm", "dynamics.rnn"),
        (r"linear|fc", "hook_linear"),
    ]

    import re

    for pattern, replacement in replacements:
        match = re.search(pattern, name)
        if match:
            return replacement.format(match.group(1)) if "{0}" in replacement else replacement

    return name
