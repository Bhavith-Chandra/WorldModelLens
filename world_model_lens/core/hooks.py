"""Hook system for intercepting and modifying computations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class HookPoint:
    """A named hook function registered at a specific component.

    Attributes:
        name: Component name (e.g., 'z_posterior', 'h', 'reward_pred').
        fn: The hook function, takes (tensor, context) and returns modified tensor.
        stage: When to apply the hook ('pre', 'post'). 'post' fires after
               computation, before downstream use.
        timestep: Optional specific timestep to hook. None means all timesteps.
        time_slice: Optional temporal range [start, end) for the hook.
            If set, the hook only fires when start <= timestep < end.
            This enables interventions at specific frames without affecting
            other timesteps. Cannot be used together with timestep (if both
            are set, timestep takes precedence).
    """

    name: str
    fn: Callable[[torch.Tensor, "HookContext"], torch.Tensor]
    stage: str = "post"
    timestep: int | None = None
    time_slice: list[int] | None = None

    def __post_init__(self):
        if self.stage not in ("pre", "post"):
            raise ValueError(f"stage must be 'pre' or 'post', got {self.stage}")


@dataclass
class HookContext:
    """Runtime metadata passed to hook functions.

    Attributes:
        timestep: Current timestep in the sequence.
        component: Component name that was just computed.
        trajectory_so_far: List of LatentState objects computed up to this point.
        metadata: Additional context dictionary.
    """

    timestep: int
    component: str
    trajectory_so_far: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class HookRegistry:
    """Registry for managing hook functions on world model components.

    Hooks are stored as a list per (component, timestep) combination and
    applied in registration order.
    """

    def __init__(self):
        self._hooks: dict[tuple[str, int | None], list[HookPoint]] = {}
        self._global_hooks: dict[str, list[HookPoint]] = {}

    def register(self, hook: HookPoint, prepend: bool = False) -> None:
        """Register a hook.

        Args:
            hook: The HookPoint to register.
        """
        if hook.timestep is not None:
            key = (hook.name, hook.timestep)
            lst = self._hooks.setdefault(key, [])
        else:
            key = (hook.name, None)
            lst = self._global_hooks.setdefault(hook.name, [])

        if prepend:
            lst.insert(0, hook)
        else:
            lst.append(hook)

    def remove(self, hook: HookPoint) -> None:
        """Remove a specific HookPoint from the registry if present.

        This is a safe no-op when the hook is not found.
        """
        # remove from global hooks
        if hook.timestep is None:
            lst = self._global_hooks.get(hook.name, [])
            try:
                lst.remove(hook)
            except ValueError:
                pass
            return

        # remove from timestep-specific hooks
        key = (hook.name, hook.timestep)
        lst = self._hooks.get(key, [])
        try:
            lst.remove(hook)
        except ValueError:
            pass

    def clear(self, name: str | None = None) -> None:
        """Remove registered hooks.

        If *name* is None (default) clears all hooks.  If *name* is provided
        removes all hooks for that component name.
        """
        if name is None:
            self._hooks.clear()
            self._global_hooks.clear()
            return

        # remove global hooks for this name
        self._global_hooks.pop(name, None)

        # remove any timestep-specific keys matching the name
        keys_to_remove = [k for k in self._hooks.keys() if k[0] == name]
        for k in keys_to_remove:
            self._hooks.pop(k, None)

    def get_hooks_for(self, component: str, timestep: int | None = None) -> list[HookPoint]:
        """Get all hooks matching a component and timestep.

        Args:
            component: Component name.
            timestep: Optional specific timestep.

        Returns:
            List of matching HookPoints.
        """
        hooks = list(self._global_hooks.get(component, []))

        if timestep is not None:
            specific = self._hooks.get((component, timestep), [])
            hooks.extend(specific)
            hooks = [h for h in hooks if self._matches_timestep(h, timestep)]

        return hooks

    def _matches_timestep(self, hook: HookPoint, timestep: int) -> bool:
        """Check if a hook matches the given timestep considering timestep and time_slice."""
        if hook.timestep is not None:
            return hook.timestep == timestep
        if hook.time_slice is not None:
            start, end = hook.time_slice
            return start <= timestep < end
        return True

    def apply(
        self,
        component: str,
        timestep: int,
        tensor: torch.Tensor,
        context: HookContext,
    ) -> torch.Tensor:
        """Apply all matching hooks to a tensor.

        Hooks fire after computation but before downstream use.

        Args:
            component: Component name.
            timestep: Current timestep.
            tensor: The activation tensor.
            context: Hook context with runtime metadata.

        Returns:
            The tensor after all hooks have been applied.
        """
        hooks = self.get_hooks_for(component, timestep)
        for hook in hooks:
            tensor = hook.fn(tensor, context)
        return tensor

    def __len__(self) -> int:
        """Total number of registered hooks."""
        return sum(len(h) for h in self._hooks.values()) + sum(
            len(h) for h in self._global_hooks.values()
        )

    def temp_hooks(self, hooks: list[HookPoint]):
        """Context manager that registers hooks for the duration of a `with` block.

        Hooks are prepended so they run before existing permanent hooks. On
        exit the exact HookPoint objects are removed (best-effort).
        """
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            # register (prepend) all hooks and remember which were actually added
            added: list[HookPoint] = []
            for h in hooks:
                self.register(h, prepend=True)
                added.append(h)
            try:
                yield
            finally:
                # best-effort removal — don't let cleanup mask original errors
                for h in added:
                    try:
                        self.remove(h)
                    except Exception:
                        pass

        return _cm()
