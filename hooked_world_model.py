"""HookedWorldModel — the central interpretability wrapper.

:class:`HookedWorldModel` wraps any :class:`WorldModelAdapter` backend and
adds a full hook + caching layer on top of it.  Every named activation is
intercepted at a well-defined point in the forward pass:

    **RULE** — hooks fire *after* an activation is computed but *before* it
    is consumed by the next computation step.

This means a patching hook on ``"z_posterior"`` at timestep *t* will cause
the downstream reward / value / actor heads at the *same* timestep to receive
the patched value, and the dynamics step at timestep *t+1* will also receive
the patched ``z_t`` as its input.

Named activation points
-----------------------
At every timestep *t* the following names are accessible via hooks and cache:

+------------------------+--------------------------------------------+
| Name                   | Shape                                      |
+========================+============================================+
| ``encoder.out``        | ``(d_emb,)``                               |
| ``rnn.h``              | ``(d_h,)``                                 |
| ``z_prior.logits``     | ``(n_cat, n_cls)``                         |
| ``z_prior``            | ``(n_cat, n_cls)`` — post-softmax          |
| ``z_posterior.logits`` | ``(n_cat, n_cls)``                         |
| ``z_posterior``        | ``(n_cat, n_cls)`` — post-softmax          |
| ``kl``                 | scalar                                     |
| ``reward_pred``        | scalar                                     |
| ``cont_pred``          | scalar                                     |
| ``actor.logits``       | ``(d_a,)``                                 |
| ``value_pred``         | scalar                                     |
+------------------------+--------------------------------------------+

``z_prior.logits`` and ``z_prior`` at *t=0* are set equal to their posterior
counterparts (KL₀ = 0 by convention, since no prior prediction is possible
without a preceding dynamics step).

Usage
-----
>>> from world_model_lens import HookedWorldModel
>>> wm = HookedWorldModel(adapter, cfg, name="cartpole_dreamer")
>>> traj, cache = wm.run_with_cache(obs_seq, action_seq)
>>> cache["z_posterior"].shape       # all T steps stacked
torch.Size([T, 32, 32])
>>> cache["kl"].mean()               # average surprise
tensor(...)
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import (
    Any,
)

import torch
from torch import Tensor

from world_model_lens.backends.base_adapter import WorldModelAdapter
from world_model_lens.core.activation_cache import ActivationCache
from world_model_lens.core.config import WorldModelConfig
from world_model_lens.core.hooks import HookContext, HookPoint, HookRegistry
from world_model_lens.core.latent_state import LatentState
from world_model_lens.core.latent_trajectory import LatentTrajectory
from world_model_lens.utils.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

#: Global mapping from backend name → adapter *class*.
#: Register custom backends with ``BACKEND_REGISTRY["my_backend"] = MyAdapter``.
BACKEND_REGISTRY: dict[str, type[WorldModelAdapter]] = {}


def register_backend(name: str) -> Callable[[type[WorldModelAdapter]], type[WorldModelAdapter]]:
    """Class decorator to register a backend under *name*.

    Examples
    --------
    >>> @register_backend("my_dreamer")
    ... class MyDreamerAdapter(WorldModelAdapter):
    ...     ...
    """

    def decorator(cls: type[WorldModelAdapter]) -> type[WorldModelAdapter]:
        BACKEND_REGISTRY[name] = cls
        return cls

    return decorator


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _kl_categorical(
    post: Tensor,  # [n_cat, n_cls] — probabilities (post-softmax)
    prior: Tensor,  # [n_cat, n_cls] — probabilities (post-softmax)
    eps: float = 1e-8,
) -> Tensor:
    """Scalar KL(posterior ‖ prior) summed over n_cat categoricals."""
    p = post.clamp(min=eps)
    q = prior.clamp(min=eps)
    return (p * (p.log() - q.log())).sum(dim=-1).sum()


def _safe_scalar(t: Tensor) -> float:
    """Squeeze a single-element tensor to a Python float."""
    return t.squeeze().item()


# ---------------------------------------------------------------------------
# HookedWorldModel
# ---------------------------------------------------------------------------


class HookedWorldModel:
    """Interpretability wrapper around any :class:`WorldModelAdapter`.

    Parameters
    ----------
    adapter:
        Backend that implements the model's forward computations.
    cfg:
        Architectural configuration (:class:`~world_model_lens.core.WorldModelConfig`).
    name:
        Human-readable identifier used in logging and checkpoints.

    Examples
    --------
    >>> wm = HookedWorldModel(adapter, cfg, name="dreamer_v3")
    >>> traj, cache = wm.run_with_cache(obs_seq, actions)
    >>> peaks = traj.surprise_peaks(threshold=2.0)
    """

    def __init__(
        self,
        adapter: WorldModelAdapter,
        cfg: WorldModelConfig,
        name: str = "hooked_wm",
    ) -> None:
        self._adapter: WorldModelAdapter = adapter.eval()
        self._cfg: WorldModelConfig = cfg
        self.name: str = name
        self._registry: HookRegistry = HookRegistry()
        self._device: torch.device = self._infer_device()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def cfg(self) -> WorldModelConfig:
        """The model's :class:`~world_model_lens.core.WorldModelConfig`."""
        return self._cfg

    @property
    def adapter(self) -> WorldModelAdapter:
        """The underlying :class:`WorldModelAdapter` backend."""
        return self._adapter

    @property
    def device(self) -> torch.device:
        """The device on which the adapter's parameters reside."""
        return self._device

    def _infer_device(self) -> torch.device:
        """Infer device from the first adapter parameter, or default to CPU."""
        for _, p in self._adapter.named_parameters():
            return p.device
        return torch.device("cpu")

    def to(self, device: str | torch.device) -> HookedWorldModel:
        """Move the adapter and internal state to *device*.  Returns ``self``."""
        dev = torch.device(device)
        self._adapter.to(dev)
        self._device = dev
        return self

    # ------------------------------------------------------------------
    # Core: _apply_and_cache
    # ------------------------------------------------------------------

    def _apply_and_cache(
        self,
        name: str,
        t: int,
        tensor: Tensor,
        ctx: HookContext,
        cache: ActivationCache | None,
        names_filter: set[str] | None,
    ) -> Tensor:
        """Apply all matching registry hooks, then optionally cache the result.

        **This is where the hook-application-order contract is enforced.**

        The tensor returned is the *post-hook* value.  Both the cached copy
        and the value returned to the caller are the same post-hook tensor,
        ensuring that downstream computations receive the (possibly patched)
        activation, never the raw one.

        Parameters
        ----------
        name:
            Activation point name (e.g. ``"z_posterior"``).
        t:
            Current timestep.
        tensor:
            The freshly computed activation (pre-hook).
        ctx:
            :class:`HookContext` for this timestep.
        cache:
            :class:`ActivationCache` accumulator, or ``None`` if caching is
            disabled.
        names_filter:
            Set of names to cache.  ``None`` means cache everything.

        Returns
        -------
        Tensor
            Post-hook activation (may differ from *tensor* if a hook patched it).
        """
        # Step 1 — fire hooks (hooks may patch the tensor)
        tensor = self._registry.apply(name, t, tensor, ctx)

        # Step 2 — cache the post-hook value
        if cache is not None:
            if names_filter is None or name in names_filter:
                cache[name, t] = tensor.detach()

        # Step 3 — return post-hook value for downstream use
        return tensor

    # ------------------------------------------------------------------
    # Core: _run_forward
    # ------------------------------------------------------------------

    def _run_forward(
        self,
        obs_seq: Tensor,
        action_seq: Tensor,
        cache: ActivationCache | None,
        names_filter: set[str] | None,
        no_grad: bool = True,
    ) -> LatentTrajectory:
        """Internal forward pass over an observation–action sequence.

        Parameters
        ----------
        obs_seq:
            Observation tensor of shape ``(T, *obs_shape)``.
        action_seq:
            Action tensor of shape ``(T, *action_shape)`` or ``(T-1, ...)``.
            ``action_seq[t]`` is the action executed *at* step *t* and used
            to transition to step *t+1*.  The last action is unused if
            ``len(action_seq) == T``.
        cache:
            :class:`ActivationCache` to populate (or ``None``).
        names_filter:
            Subset of activation names to cache (``None`` = cache all).
        no_grad:
            If ``True`` (default), wraps the entire pass in
            ``torch.no_grad()``.

        Returns
        -------
        LatentTrajectory
        """
        T = obs_seq.shape[0]
        ctx_mgr = torch.no_grad() if no_grad else contextlib.nullcontext()

        states: list[LatentState] = []
        h: Tensor = self._adapter.initial_state()
        z: Tensor | None = None  # z_{t-1}; populated from t=0 onward

        with ctx_mgr:
            for t in range(T):
                traj_so_far: LatentTrajectory | None = (
                    LatentTrajectory(
                        states=list(states),
                        env_name="",
                        episode_id="partial",
                    )
                    if states
                    else None
                )
                ctx = HookContext(
                    timestep=t,
                    component="forward",
                    trajectory_so_far=traj_so_far,
                )

                # ----------------------------------------------------
                # 1. Encode observation  →  "encoder.out"
                # ----------------------------------------------------
                obs_emb = self._adapter.encode(obs_seq[t])
                obs_emb = self._apply_and_cache("encoder.out", t, obs_emb, ctx, cache, names_filter)

                # ----------------------------------------------------
                # 2. Recurrent state update  →  "rnn.h"
                #    At t=0: h_0 = initial_state (already set above).
                #    At t>0: dynamics_step(h_{t-1}, z_{t-1}, a_{t-1}).
                # ----------------------------------------------------
                if t == 0:
                    h = self._apply_and_cache("rnn.h", 0, h, ctx, cache, names_filter)
                    # Convention: at t=0 the prior is identical to the
                    # posterior (no KL penalty on the first step).
                    z_prior_logits: Tensor | None = None  # resolved after posterior
                else:
                    # Use action from the *previous* timestep.
                    a_prev = action_seq[t - 1]
                    h, z_prior_raw = self._adapter.dynamics_step(h, z, a_prev)  # type: ignore[arg-type]

                    # Hook: rnn.h  (post-GRU, pre-posterior)
                    h = self._apply_and_cache("rnn.h", t, h, ctx, cache, names_filter)

                    # Hook: z_prior.logits
                    z_prior_raw = self._apply_and_cache(
                        "z_prior.logits", t, z_prior_raw, ctx, cache, names_filter
                    )
                    # Hook: z_prior  (post-softmax)
                    z_prior_prob = z_prior_raw.softmax(dim=-1)
                    z_prior_prob = self._apply_and_cache(
                        "z_prior", t, z_prior_prob, ctx, cache, names_filter
                    )
                    z_prior_logits = z_prior_raw  # keep reference for KL

                # ----------------------------------------------------
                # 3. Posterior  →  "z_posterior.logits", "z_posterior"
                # ----------------------------------------------------
                z_post_raw = self._adapter.posterior(h, obs_emb)

                z_post_raw = self._apply_and_cache(
                    "z_posterior.logits", t, z_post_raw, ctx, cache, names_filter
                )

                z_post_prob = z_post_raw.softmax(dim=-1)
                # CRITICAL: hooks on "z_posterior" fire here.  Downstream
                # reward / value / actor / dynamics (next step) all receive
                # the post-hook value.
                z_post_prob = self._apply_and_cache(
                    "z_posterior", t, z_post_prob, ctx, cache, names_filter
                )
                z = z_post_prob  # carry forward for next dynamics step

                # At t=0: prior ≡ posterior (KL₀ = 0 by convention)
                if t == 0:
                    z_prior_logits = z_post_raw
                    z_prior_prob = z_post_prob
                    if cache is not None and (
                        names_filter is None or "z_prior.logits" in names_filter
                    ):
                        cache["z_prior.logits", 0] = z_prior_logits.detach()
                    if cache is not None and (names_filter is None or "z_prior" in names_filter):
                        cache["z_prior", 0] = z_prior_prob.detach()

                # ----------------------------------------------------
                # 4. KL  →  "kl"
                #    Computed from POST-HOOK z_posterior and z_prior.
                # ----------------------------------------------------
                kl = _kl_categorical(z_post_prob, z_prior_prob)  # type: ignore[arg-type]
                if cache is not None and (names_filter is None or "kl" in names_filter):
                    cache["kl", t] = kl.detach()

                # ----------------------------------------------------
                # 5. Optional heads — each fires its own hook point.
                # ----------------------------------------------------
                reward_val: float | None = None
                cont_val: float | None = None
                actor_logits_out: Tensor | None = None
                value_val: float | None = None

                # reward_pred
                try:
                    r = self._adapter.reward_pred(h, z_post_prob)
                    r = self._apply_and_cache("reward_pred", t, r, ctx, cache, names_filter)
                    reward_val = _safe_scalar(r)
                except NotImplementedError:
                    pass

                # cont_pred
                try:
                    c = self._adapter.cont_pred(h, z_post_prob)
                    c = self._apply_and_cache("cont_pred", t, c, ctx, cache, names_filter)
                    cont_val = _safe_scalar(c)
                except NotImplementedError:
                    pass

                # actor.logits
                try:
                    al = self._adapter.actor(h, z_post_prob)
                    al = self._apply_and_cache("actor.logits", t, al, ctx, cache, names_filter)
                    actor_logits_out = al.detach()
                except NotImplementedError:
                    pass

                # value_pred
                try:
                    v = self._adapter.value(h, z_post_prob)
                    v = self._apply_and_cache("value_pred", t, v, ctx, cache, names_filter)
                    value_val = _safe_scalar(v)
                except NotImplementedError:
                    pass

                # ----------------------------------------------------
                # 6. Build LatentState
                # ----------------------------------------------------
                a_t: Tensor | None = action_seq[t].detach() if t < len(action_seq) else None
                state = LatentState(
                    h_t=h.detach(),
                    z_posterior=z_post_prob.detach(),
                    z_prior=z_prior_prob.detach(),  # type: ignore[arg-type]
                    timestep=t,
                    action=a_t,
                    reward_pred=reward_val,
                    cont_pred=cont_val,
                    actor_logits=actor_logits_out,
                    value_pred=value_val,
                )
                states.append(state)

        return LatentTrajectory(
            states=states,
            env_name=self.name,
            episode_id=f"run_{uuid.uuid4().hex[:8]}",
            imagined=False,
        )

    # ------------------------------------------------------------------
    # Public run methods
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run_with_cache(
        self,
        obs_seq: Tensor,
        action_seq: Tensor,
        names_filter: list[str] | set[str] | None = None,
        device: str | torch.device | None = None,
    ) -> tuple[LatentTrajectory, ActivationCache]:
        """Run the full forward pass and return a trajectory + activation cache.

        Parameters
        ----------
        obs_seq:
            Observation sequence, shape ``(T, *obs_shape)``.
        action_seq:
            Action sequence, shape ``(T, *action_shape)`` or ``(T-1, ...)``.
        names_filter:
            If provided, only cache activations whose names are in this
            collection.  ``None`` caches every activation point.
        device:
            Move inputs to this device before running.  Defaults to
            :attr:`self.device`.

        Returns
        -------
        traj : LatentTrajectory
        cache : ActivationCache

        Examples
        --------
        >>> traj, cache = wm.run_with_cache(obs_seq, actions)
        >>> cache["z_posterior"].shape
        torch.Size([T, 32, 32])
        """
        dev = torch.device(device) if device is not None else self._device
        obs_seq = obs_seq.to(dev)
        action_seq = action_seq.to(dev)

        filter_set: set[str] | None = set(names_filter) if names_filter is not None else None
        cache = ActivationCache()
        traj = self._run_forward(
            obs_seq,
            action_seq,
            cache=cache,
            names_filter=filter_set,
            no_grad=True,
        )
        return traj, cache

    def run_with_hooks(
        self,
        obs_seq: Tensor,
        action_seq: Tensor,
        fwd_hooks: list[HookPoint],
        return_cache: bool = False,
    ) -> LatentTrajectory | tuple[LatentTrajectory, ActivationCache]:
        """Register *fwd_hooks* temporarily, run the forward pass, then remove them.

        Hooks in *fwd_hooks* are prepended to the permanent registry for the
        duration of this call and removed atomically afterwards — even if the
        forward pass raises an exception.

        Parameters
        ----------
        obs_seq:
            Observation sequence, shape ``(T, *obs_shape)``.
        action_seq:
            Action sequence, shape ``(T, *action_shape)`` or ``(T-1, ...)``.
        fwd_hooks:
            List of :class:`HookPoint` objects to register temporarily.
            They fire in the order they appear in *fwd_hooks*, before any
            permanent registry hooks on the same component.
        return_cache:
            If ``True``, also return a populated :class:`ActivationCache`.

        Returns
        -------
        LatentTrajectory
            (plus :class:`ActivationCache` when ``return_cache=True``)

        Examples
        --------
        >>> def zero_out(t, ctx): return torch.zeros_like(t)
        >>> hp = HookPoint("z_posterior", "post", zero_out, timestep=5)
        >>> traj = wm.run_with_hooks(obs_seq, actions, [hp])
        """
        obs_seq = obs_seq.to(self._device)
        action_seq = action_seq.to(self._device)

        # Prepend temporary hooks so they fire BEFORE permanent ones.
        for hp in fwd_hooks:
            self._registry._hooks.insert(0, hp)

        cache: ActivationCache | None = ActivationCache() if return_cache else None
        try:
            traj = self._run_forward(
                obs_seq,
                action_seq,
                cache=cache,
                names_filter=None,
                no_grad=True,
            )
        finally:
            # Always clean up — remove exactly the hooks we added.
            temp_ids = {id(hp) for hp in fwd_hooks}
            self._registry._hooks = [h for h in self._registry._hooks if id(h) not in temp_ids]

        if return_cache:
            return traj, cache  # type: ignore[return-value]
        return traj

    # ------------------------------------------------------------------
    # Imagination
    # ------------------------------------------------------------------

    @torch.no_grad()
    def imagine(
        self,
        start_state: LatentState,
        action_sequence: Tensor | None = None,
        policy: Callable[[Tensor, Tensor], Tensor] | None = None,
        horizon: int = 50,
        temperature: float = 1.0,
    ) -> LatentTrajectory:
        """Roll out an imagined trajectory from *start_state* using the world model.

        No observations are consumed — the model predicts entirely in latent
        space.  The stochastic latent at each imagined step is sampled from
        the prior (``z_prior``) returned by the dynamics head.  Because no
        posterior is available, ``z_posterior ≡ z_prior`` in all imagined
        :class:`LatentState` objects (KL = 0 throughout).

        Parameters
        ----------
        start_state:
            Starting :class:`LatentState`.  Its ``h_t`` and ``z_posterior``
            are used as the seed ``(h_0, z_0)`` for the imagined rollout.
        action_sequence:
            Pre-specified actions, shape ``(≤horizon, *action_shape)``.
            If provided, the rollout runs for ``min(horizon, len(action_sequence))``
            steps using these actions.
        policy:
            Callable ``policy(h, z) → action_tensor`` used to select actions
            when *action_sequence* is ``None``.  When both are ``None`` the
            adapter's actor head is used; if that is also absent, a
            :exc:`ValueError` is raised.
        horizon:
            Maximum number of imagined steps.
        temperature:
            Temperature applied to the prior logits before sampling.
            Values < 1 make the distribution sharper (more exploitative);
            values > 1 increase entropy.

        Returns
        -------
        LatentTrajectory
            With ``imagined=True``, ``fork_point=start_state.timestep``.

        Examples
        --------
        >>> forked = wm.imagine(traj[5], horizon=15)
        >>> forked.imagined
        True
        >>> forked.length
        15
        """
        h = start_state.h_t.to(self._device)
        z = start_state.z_posterior.to(self._device)

        n_steps = min(horizon, len(action_sequence)) if action_sequence is not None else horizon

        states: list[LatentState] = []

        for step in range(n_steps):
            # ---- select action ----------------------------------------
            if action_sequence is not None:
                a = action_sequence[step].to(self._device)
            elif policy is not None:
                a = policy(h, z)
            else:
                try:
                    al = self._adapter.actor(h, z)
                    if temperature != 1.0:
                        al = al / temperature
                    a = torch.distributions.Categorical(logits=al).sample()
                except NotImplementedError:
                    raise ValueError(
                        "imagine() requires either action_sequence, policy, "
                        "or an adapter with an actor head."
                    )

            # ---- dynamics (prior only — no observation) ---------------
            h_next, z_prior_raw = self._adapter.dynamics_step(h, z, a)
            ctx = HookContext(timestep=step, component="imagine")

            h_next = self._registry.apply("rnn.h", step, h_next, ctx)
            z_prior_raw = self._registry.apply("z_prior.logits", step, z_prior_raw, ctx)

            if temperature != 1.0:
                z_prior_raw = z_prior_raw / temperature

            z_next = z_prior_raw.softmax(dim=-1)
            z_next = self._registry.apply("z_posterior", step, z_next, ctx)

            # ---- optional heads ---------------------------------------
            reward_val: float | None = None
            cont_val: float | None = None
            actor_logits_out: Tensor | None = None
            value_val: float | None = None

            try:
                r = self._adapter.reward_pred(h_next, z_next)
                r = self._registry.apply("reward_pred", step, r, ctx)
                reward_val = _safe_scalar(r)
            except NotImplementedError:
                pass

            try:
                c = self._adapter.cont_pred(h_next, z_next)
                c = self._registry.apply("cont_pred", step, c, ctx)
                cont_val = _safe_scalar(c)
            except NotImplementedError:
                pass

            try:
                al2 = self._adapter.actor(h_next, z_next)
                al2 = self._registry.apply("actor.logits", step, al2, ctx)
                actor_logits_out = al2.detach()
            except NotImplementedError:
                pass

            try:
                v = self._adapter.value(h_next, z_next)
                v = self._registry.apply("value_pred", step, v, ctx)
                value_val = _safe_scalar(v)
            except NotImplementedError:
                pass

            state = LatentState(
                h_t=h_next.detach(),
                z_posterior=z_next.detach(),  # imagined: posterior ≡ prior
                z_prior=z_next.detach(),
                timestep=step,
                action=a.detach() if isinstance(a, Tensor) else torch.tensor(a),
                reward_pred=reward_val,
                cont_pred=cont_val,
                actor_logits=actor_logits_out,
                value_pred=value_val,
            )
            states.append(state)
            h, z = h_next, z_next

        env = start_state.metadata.get("env_name", self.name) if start_state.metadata else self.name
        return LatentTrajectory(
            states=states,
            env_name=env,
            episode_id=f"imagined_{uuid.uuid4().hex[:8]}",
            imagined=True,
            fork_point=start_state.timestep,
        )

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def add_hook(self, hook: HookPoint) -> None:
        """Permanently register *hook* in the internal :class:`HookRegistry`.

        Parameters
        ----------
        hook:
            :class:`HookPoint` to add.  Use :meth:`remove_hook` or
            :meth:`clear_hooks` to remove it later.

        Examples
        --------
        >>> wm.add_hook(HookPoint("z_posterior", "post", my_fn))
        """
        self._registry.register(hook)

    def remove_hook(self, name: str) -> None:
        """Remove all permanent hooks whose :attr:`HookPoint.name` == *name*.

        Parameters
        ----------
        name:
            Component name to clear (e.g. ``"z_posterior"``).

        Examples
        --------
        >>> wm.remove_hook("z_posterior")
        """
        self._registry.clear(name)

    def clear_hooks(self) -> None:
        """Remove *all* permanent hooks from the registry."""
        self._registry.clear()

    # ------------------------------------------------------------------
    # Weight inspection
    # ------------------------------------------------------------------

    @property
    def named_weights(self) -> dict[str, Tensor]:
        """All weight matrices from the adapter, plus convenient stacked forms.

        Returns every parameter tensor with ``ndim >= 2`` (matrices, not bias
        vectors or scalars) from :meth:`WorldModelAdapter.named_parameters`.

        Additionally, parameters that share a common dot-prefix
        (e.g. ``"rnn.weight_ih_l0"`` and ``"rnn.weight_hh_l0"``) are
        concatenated row-wise and stored under ``"<prefix>._stacked"``.
        This is useful for inspecting the combined input–hidden weight of a
        GRU layer without manually concatenating.

        Returns
        -------
        dict mapping str → Tensor

        Examples
        --------
        >>> for name, W in wm.named_weights.items():
        ...     print(name, W.shape)
        encoder.fc1.weight  torch.Size([256, 128])
        rnn.weight_ih_l0    torch.Size([768, 256])
        rnn.weight_hh_l0    torch.Size([768, 512])
        rnn._stacked        torch.Size([1536, 768])   ← stacked form
        """
        result: dict[str, Tensor] = {}
        prefix_groups: dict[str, list[Tensor]] = {}

        for param_name, param in self._adapter.named_parameters():
            if param.ndim < 2:
                continue
            result[param_name] = param.detach()

            # Group by dot-prefix for stacking
            parts = param_name.split(".")
            prefix = ".".join(parts[:-1]) if len(parts) > 1 else param_name
            prefix_groups.setdefault(prefix, []).append(param.detach())

        # Add stacked forms wherever there are ≥2 matrices with the same prefix
        for prefix, matrices in prefix_groups.items():
            if len(matrices) >= 2:
                try:
                    result[f"{prefix}._stacked"] = torch.cat(matrices, dim=0)
                except RuntimeError:
                    # Shape mismatch — skip stacking for this group
                    pass

        return result

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        **kwargs: Any,
    ) -> HookedWorldModel:
        """Load a pre-trained world model from HuggingFace Hub.

        Parameters
        ----------
        model_name:
            HuggingFace Hub model identifier, e.g.
            ``"bhavith/dreamer-v3-cartpole"``.
        **kwargs:
            Additional keyword arguments forwarded to the hub download.

        Raises
        ------
        NotImplementedError
            Always — this method is a stub pending Hub integration.
        """
        raise NotImplementedError(
            "HookedWorldModel.from_pretrained() is not yet implemented. "
            "Hub integration will be added in a future release.  "
            f"(Requested: {model_name!r})"
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        backend: str,
        cfg: WorldModelConfig | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> HookedWorldModel:
        """Instantiate a :class:`HookedWorldModel` from a saved checkpoint.

        Looks up *backend* in :data:`BACKEND_REGISTRY` to find the correct
        :class:`WorldModelAdapter` subclass, then delegates to that class's
        ``from_checkpoint(path, cfg, **kwargs)`` classmethod.

        Parameters
        ----------
        path:
            Path to the checkpoint file or directory.
        backend:
            Backend key in :data:`BACKEND_REGISTRY` (e.g. ``"dreamer"``).
        cfg:
            Architectural config.  If ``None``, the adapter is expected to
            infer it from the checkpoint.
        name:
            Optional human-readable name for the wrapped model.
        **kwargs:
            Forwarded to the adapter's ``from_checkpoint`` method.

        Returns
        -------
        HookedWorldModel

        Raises
        ------
        KeyError
            If *backend* is not found in :data:`BACKEND_REGISTRY`.
        AttributeError
            If the adapter class does not implement ``from_checkpoint``.

        Examples
        --------
        >>> wm = HookedWorldModel.from_checkpoint(
        ...     "runs/dreamer_run_01/ckpt.pt",
        ...     backend="dreamer",
        ...     cfg=cfg,
        ... )
        """
        if backend not in BACKEND_REGISTRY:
            available = sorted(BACKEND_REGISTRY.keys())
            raise KeyError(
                f"Backend {backend!r} not found in BACKEND_REGISTRY. "
                f"Available: {available}. "
                f"Register a backend with @register_backend('{backend}')."
            )
        adapter_cls = BACKEND_REGISTRY[backend]
        if not hasattr(adapter_cls, "from_checkpoint"):
            raise AttributeError(
                f"Backend {backend!r} ({adapter_cls.__name__}) does not implement "
                f"a 'from_checkpoint' classmethod."
            )
        adapter = adapter_cls.from_checkpoint(path, cfg=cfg, **kwargs)  # type: ignore[attr-defined]
        if cfg is None and hasattr(adapter, "cfg"):
            cfg = adapter.cfg  # type: ignore[assignment]
        if cfg is None:
            raise ValueError(
                f"cfg was not provided and could not be inferred from the "
                f"{backend!r} adapter.  Pass cfg explicitly."
            )
        _name = name or (Path(path).stem if isinstance(path, (str, Path)) else backend)
        return cls(adapter=adapter, cfg=cfg, name=_name)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n_hooks = len(self._registry)
        return (
            f"HookedWorldModel("
            f"name={self.name!r}, "
            f"backend={self._adapter.backend_name!r}, "
            f"d_latent={self._cfg.d_latent}, "
            f"device={self._device}, "
            f"hooks={n_hooks})"
        )
