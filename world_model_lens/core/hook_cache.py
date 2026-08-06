from __future__ import annotations

from typing import Optional, Collection

import torch
from torch import Tensor

from world_model_lens.core.hooks import HookRegistry, HookContext
from world_model_lens.core.activation_cache import ActivationCache
from typing import Any, cast


class HookCacheManager:
    """Small helper that centralizes hook application and caching semantics.

    This wraps a :class:`HookRegistry` and provides a single point where
    hooks are applied and activations are written to an :class:`ActivationCache`.
    """

    def __init__(self, registry: HookRegistry):
        self._registry = registry

    def apply_and_cache(
        self,
        name: str,
        t: int,
        tensor: Tensor,
        ctx: HookContext,
        cache: Optional[ActivationCache],
        names_filter: Optional[Collection[str]],
    ) -> Tensor:
        # apply hooks
        tensor = self._registry.apply(name, t, tensor, ctx)
        # cache post-hook value if requested
        if cache is not None and (names_filter is None or name in names_filter):
            cache[name, t] = tensor.detach()
        return tensor

    def set_prior_equivalent(
        self,
        cache: Optional[ActivationCache],
        timestep: int,
        logits: Tensor,
        probs: Tensor,
        names_filter: Optional[Collection[str]] = None,
    ) -> None:
        if cache is None:
            return
        if names_filter is None or "z_prior.logits" in names_filter:
            cache["z_prior.logits", timestep] = logits
        if names_filter is None or "z_prior" in names_filter:
            cache["z_prior", timestep] = probs

    def apply_kv_hooks(
        self,
        cache: Optional[ActivationCache],
        timestep: int,
        ctx: HookContext,
    ) -> None:
        """Run hooks that operate on the KV-style cache.

        This provides a dedicated hook point named "kv_cache". Hooks
        registered at this component should accept the ActivationCache
        and the HookContext and may mutate the cache in-place to emulate
        editing a model's key/value memory without re-running the whole
        sequence.

        The expected hook signature is ``fn(cache: ActivationCache, ctx: HookContext) -> None``.
        HookRegistry does not enforce types, so users may register such
        functions with HookPoint(name="kv_cache", fn=..., timestep=...).
        """
        if cache is None:
            return
        hooks = self._registry.get_hooks_for("kv_cache", timestep)
        for h in hooks:
            # Allow the hook to mutate the cache in-place. HookPoint.fn is
            # typed for tensor hooks; cast to Any to allow KV-cache hooks
            # that accept (ActivationCache, HookContext).
            fn = cast(Any, h.fn)
            fn(cache, ctx)
