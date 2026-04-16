"""Central wrapper with hooks and caching for ANY world model.

HookedWorldModel is backend-agnostic and works with ANY world model adapter.
It provides interpretability tools without assuming RL-specific features.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import torch

if TYPE_CHECKING:
    from world_model_lens.core.world_state import WorldState, WorldTrajectory, WorldDynamics
    from world_model_lens.core.world_trajectory import WorldTrajectory
    from world_model_lens.core.config import WorldModelConfig
    from world_model_lens.core.latent_state import LatentState

from world_model_lens.core.hooks import HookContext, HookPoint, HookRegistry
from world_model_lens.core.activation_cache import ActivationCache
from world_model_lens.core.world_state import WorldState
from world_model_lens.core.world_trajectory import WorldTrajectory
from world_model_lens.backends.base_adapter import WorldModelCapabilities
from world_model_lens.core.hook_cache import HookCacheManager
from world_model_lens.core.forward_runner import ForwardRunner
from world_model_lens.core.latent_trajectory import LatentTrajectory


class HookedWorldModel:
    """Unified wrapper for ANY world model with hook + caching layers.

    This class is backend-agnostic. It works with:
    - DreamerV3/V2 (RL)
    - TD-MPC2 (RL)
    - IRIS/transformer-based (RL)
    - Video prediction models (non-RL)
    - Planning models (non-RL)
    - Any custom world model

    Features:
    - run_with_cache(): Full forward pass with activation caching
    - run_with_hooks(): Temporary hooks for patching/interception
    - imagine(): Rollout using dynamics model

    Example:
        >>> from world_model_lens import HookedWorldModel, WorldModelConfig
        >>> from world_model_lens.backends.base_adapter import BaseModelAdapter
        >>>
        >>> # Your world model adapter
        >>> class MyModel(BaseModelAdapter):
        ...     def encode(self, obs, context=None):
        ...         return state, encoding
        ...     def dynamics(self, state, action=None):
        ...         return next_state
        >>>
        >>> # Wrap and use
        >>> wm = HookedWorldModel(adapter=MyModel(config), config=config)
        >>> traj, cache = wm.run_with_cache(observations)
        >>> imagined = wm.imagine(start_state=traj.states[0], horizon=20)
    """

    def __init__(
        self,
        adapter: Any,
        config: Any,
        name: str = "world_model",
    ):
        """Initialize wrapper.

        Args:
            adapter: Any BaseModelAdapter implementation
            config: WorldModelConfig or dict-like config
            name: Optional name for this instance
        """
        self.adapter = adapter
        self.config = config
        self.name = name
        self._hooks = HookRegistry()
        # Central manager that applies hooks and writes to ActivationCache.
        # Attaching it here lets ForwardRunner or other orchestrators use
        # a single canonical implementation instead of reaching into
        # HookRegistry/ActivationCache internals.
        self._hook_cache_manager = HookCacheManager(self._hooks)
        self._device = torch.device("cpu")
        self._uses_capabilities_adapter_api = isinstance(
            getattr(type(self.adapter), "capabilities", None), property
        )

    def _apply_and_cache(
        self,
        name: str,
        t: int,
        tensor: torch.Tensor,
        ctx: HookContext,
        cache: Optional[ActivationCache],
        names_filter: Optional[List[str]],
    ) -> torch.Tensor:
        """Apply hooks via registry, then optionally write to ActivationCache.

        This centralizes hook application + caching semantics used across
        run_with_cache so all writes go through the same pipeline.
        """
        # Prefer the mounted HookCacheManager to centralize behavior; fall
        # back to the local registry if the manager is not present for any
        # reason (backwards-compatible shim).
        manager = getattr(self, "_hook_cache_manager", None)
        if manager is not None:
            return manager.apply_and_cache(name, t, tensor, ctx, cache, names_filter)

        tensor = self._hooks.apply(name, t, tensor, ctx)
        if cache is not None and (names_filter is None or name in names_filter):
            cache[name, t] = tensor.detach()
        return tensor

    def _get_capabilities(self) -> WorldModelCapabilities:
        """Return adapter capabilities with a backward-compatible fallback.

        Some older adapters in the repository implement the generic adapter
        interface and do not expose a ``capabilities`` property. In that case
        we synthesize a conservative descriptor from config flags.
        """
        caps = getattr(self.adapter, "capabilities", None)
        if caps is not None:
            return caps

        config = getattr(self.adapter, "config", self.config)
        return WorldModelCapabilities(
            has_decoder=bool(getattr(config, "has_decoder", False)),
            has_reward_head=bool(getattr(config, "has_reward_head", False)),
            # Older configs often expose "done" semantics, while the newer
            # capability descriptor tracks the corresponding continue head.
            has_continue_head=bool(getattr(config, "has_done_head", False)),
            has_actor=bool(getattr(config, "has_policy_head", False)),
            has_critic=bool(getattr(config, "has_value_head", False)),
            uses_actions=bool(getattr(config, "d_action", 0)),
            is_rl_trained=bool(
                getattr(config, "has_reward_head", False)
                or getattr(config, "has_value_head", False)
                or getattr(config, "has_policy_head", False)
            ),
        )

    def _call_transition(
        self,
        state: torch.Tensor,
        posterior: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Call adapter.transition across the two adapter APIs in the repo."""
        transition = self.adapter.transition
        if self._uses_capabilities_adapter_api:
            return transition(state, posterior, action)

        return transition(state, action, posterior)

    def _call_decode(
        self,
        state: torch.Tensor,
        posterior: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Call adapter.decode across the decode signatures used in the repo."""
        decode = self.adapter.decode
        if self._uses_capabilities_adapter_api:
            return decode(state, posterior)

        return decode(state)

    def _build_state(
        self,
        h: torch.Tensor,
        z_post_prob: torch.Tensor,
        z_prior_prob: torch.Tensor,
        t: int,
        action_seq: Optional[torch.Tensor] = None,
        reward_val: Optional[torch.Tensor] = None,
        cont_val: Optional[torch.Tensor] = None,
        actor_logits_out: Optional[torch.Tensor] = None,
        value_val: Optional[torch.Tensor] = None,
    ) -> Any:
        """Helper to build a LatentState object. Used by ForwardRunner."""
        from world_model_lens.core.latent_state import LatentState

        return LatentState(
            h_t=h,
            z_posterior=z_post_prob,
            z_prior=z_prior_prob,
            timestep=t,
            action=action_seq[t] if action_seq is not None and t < len(action_seq) else None,
            reward_pred=reward_val,
            cont_pred=cont_val,
            value_pred=value_val,
            actor_logits=actor_logits_out,
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        backend: str,
        config: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ) -> "HookedWorldModel":
        """Load a world model from checkpoint.

        Args:
            path: Path to checkpoint file
            backend: Backend name (e.g., 'dreamerv3', 'video')
            config: Optional config
            device: Target device

        Returns:
            HookedWorldModel instance
        """
        from world_model_lens.backends import REGISTRY as registry

        adapter_cls = registry.get(backend)
        adapter = adapter_cls(config) if config else adapter_cls.from_checkpoint(path)

        if device:
            adapter = adapter.to(device)

        return cls(adapter=adapter, config=config or adapter.config, name=path)

    def run_with_cache(
        self,
        observations: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        names_filter: Optional[set] = None,
        device: Optional[torch.device] = None,
        store_logits: bool = True,
    ) -> Tuple[WorldTrajectory, ActivationCache]:
        """Run forward pass with full activation caching.

        This method is backend-agnostic. It works with any world model
        that implements encode() and transition().

        Works with both RL and non-RL world models:
        - RL models: actions, reward prediction, value prediction available
        - Non-RL models: no actions, no reward/value prediction

        Args:
            observations: Observation sequence [T, ...]
            actions: Optional action sequence [T, d_action] (ignored for non-RL models)
            names_filter: Optional list of component names to cache
            device: Target device for outputs
            store_logits: Whether to store logits (for probing)

        Returns:
            Tuple of (WorldTrajectory, ActivationCache)
        """
        # normalize names_filter to a set for efficient membership tests
        # normalize names_filter to a set[str] for efficient membership checks
        names_filter = set(names_filter) if names_filter is not None else None

        T = observations.shape[0]
        cache = ActivationCache()
        states = []

        caps = self._get_capabilities()

        init_result = self.adapter.initial_state(batch_size=1, device=observations.device)
        if isinstance(init_result, tuple):
            h, z = init_result
            state = h
            z_current = z
        else:
            state = init_result
            z_current = None
        if state.dim() > 1 and state.shape[0] == 1:
            state = state.squeeze(0)

        # Use ForwardRunner only when the adapter implements the newer
        # latent-style encode signature (encode(obs) -> embedding) so we
        # avoid breaking legacy adapters that require an h_prev/context arg.
        use_forward_runner = False
        try:
            import inspect

            sig = inspect.signature(self.adapter.encode)
            # parameters includes 'self' so 2 means (self, obs)
            if (
                len(sig.parameters) <= 1
                and hasattr(self.adapter, "dynamics")
                and hasattr(self.adapter, "initial_state")
            ):
                use_forward_runner = True
        except Exception:
            use_forward_runner = False

        # Disable ForwardRunner automatic path for now to preserve backward
        # compatibility with the wide variety of adapter signatures in the
        # repo. The ForwardRunner integration can be re-enabled behind an
        # explicit opt-in once adapters converge on a stable API.
        use_forward_runner = False

        if use_forward_runner:
            # Probe the adapter's encode return type to ensure it returns a
            # single tensor (observation embedding). Some adapters accept a
            # single-argument encode but still return (posterior, obs_enc)
            # tuples — avoid using ForwardRunner in that case.
            probe_ok = False
            try:
                with torch.no_grad():
                    sample = observations[0]
                    out = self.adapter.encode(sample)
                if isinstance(out, torch.Tensor):
                    probe_ok = True
            except Exception:
                probe_ok = False

            if not probe_ok:
                use_forward_runner = False

        if use_forward_runner:
            # Build a thin shim exposing the helpers ForwardRunner expects.
            class _Shim:
                def __init__(self, parent: "HookedWorldModel"):
                    self._parent = parent
                    self.name = parent.name
                    # reuse the shared hook cache manager so hooks and caching
                    # behave identically to the existing run_with_cache pipeline.
                    self._hook_cache_manager = parent._hook_cache_manager

                    # Adapter shim: expose a simplified encode(obs) -> Tensor
                    # API expected by ForwardRunner while delegating to the
                    # real adapter under the hood.
                    class AdapterShim:
                        def __init__(self, real):
                            self._real = real

                        def __getattr__(self, name):
                            return getattr(self._real, name)

                        def initial_state(self):
                            # try no-arg then fall back to common signature
                            try:
                                return self._real.initial_state()
                            except TypeError:
                                try:
                                    return self._real.initial_state(batch_size=1)
                                except Exception:
                                    return self._real.initial_state

                        def encode(self, obs):
                            # Try multiple adapter encode signatures and return
                            # a single tensor. Prefer posterior probabilities if
                            # the adapter returns (logits, probs) pairs.
                            def _first_tensor(x):
                                if isinstance(x, torch.Tensor):
                                    return x
                                if isinstance(x, (tuple, list)):
                                    for el in x:
                                        t = _first_tensor(el)
                                        if t is not None:
                                            return t
                                return None

                            try:
                                out = self._real.encode(obs)
                            except TypeError:
                                try:
                                    out = self._real.encode(obs, None)
                                except Exception:
                                    raise

                            if isinstance(out, torch.Tensor):
                                return out
                            if isinstance(out, (tuple, list)):
                                # common shapes: (posterior, obs_enc) or
                                # ((logits, probs), obs_enc). Prefer probs when
                                # available, otherwise first tensor.
                                first = out[0]
                                if isinstance(first, (tuple, list)) and len(first) > 1:
                                    # try probs at index 1
                                    if isinstance(first[1], torch.Tensor):
                                        return first[1]
                                t = _first_tensor(out)
                                if t is not None:
                                    return t
                            raise TypeError("Adapter.encode returned unsupported type")

                    self._adapter = AdapterShim(parent.adapter)

                def _encode(self, obs, t, ctx, cache, names_filter):
                    post = self._adapter.encode(obs.unsqueeze(0))
                    return post.squeeze(0)

                def _dynamics_and_prior(self, h, z, a_prev, t, ctx, cache, names_filter):
                    prior = self._adapter.dynamics(h.unsqueeze(0) if h.dim() == 1 else h)
                    prior = prior.squeeze(0) if prior.dim() > 1 else prior
                    # return h (unchanged), raw logits (prior), and prior prob
                    return h, prior, prior

                def _posterior(self, h, obs_emb, t, ctx, cache, names_filter):
                    # The adapter.encode call above produced the posterior; here
                    # we interpret obs_emb as the posterior for compatibility.
                    return obs_emb, obs_emb

                def _compute_kl_and_cache(self, z_post, z_prior, t, ctx, cache, names_filter):
                    # Compute KL if shapes match, otherwise None.
                    if z_post.shape == z_prior.shape and z_post.dim() > 1:
                        p = z_post.clamp(min=1e-8)
                        q = z_prior.clamp(min=1e-8)
                        p = p / p.sum(dim=-1, keepdim=True)
                        q = q / q.sum(dim=-1, keepdim=True)
                        kl = (p * (p.log() - q.log())).sum(dim=-1)
                        return kl
                    return None

                def _compute_heads(self, h, z_post, t, ctx, cache, names_filter):
                    # Delegate reward/value/actor predictions to adapter when
                    # available to preserve existing behavior.
                    reward = None
                    value = None
                    actor_logits = None
                    caps = self._parent._get_capabilities()
                    if caps.has_reward_head:
                        try:
                            reward = self._adapter.predict_reward(
                                h.unsqueeze(0), z_post.unsqueeze(0)
                            )
                            if reward is not None:
                                reward = reward.squeeze(0)
                        except NotImplementedError:
                            reward = None
                    if caps.has_critic:
                        try:
                            if hasattr(self._adapter, "critic_forward"):
                                value = self._adapter.critic_forward(
                                    h.unsqueeze(0), z_post.unsqueeze(0)
                                )
                            else:
                                value = self._adapter.predict_value(h.unsqueeze(0), None)
                            if value is not None:
                                value = value.squeeze(0)
                        except NotImplementedError:
                            value = None
                    return reward, None, actor_logits, value

                def _build_state(
                    self,
                    h,
                    z_post_prob,
                    z_prior_prob,
                    t,
                    action_seq,
                    reward_val,
                    cont_val,
                    actor_logits_out,
                    value_val,
                ):
                    return LatentTrajectory(states=[], env_name=self.name, episode_id=None)

                def _apply_and_cache(self, name, t, tensor, ctx, cache, names_filter):
                    return self._parent._apply_and_cache(name, t, tensor, ctx, cache, names_filter)

                def _cache_prior_equivalent_at_t0(self, raw, prob, t, cache, names_filter):
                    if cache is not None:
                        try:
                            cache.set_prior_equivalent(
                                t,
                                raw.detach(),
                                prob.detach(),
                                list(names_filter) if names_filter is not None else None,
                            )
                        except Exception:
                            pass

            shim = _Shim(self)
            runner = ForwardRunner(shim)
            latent_traj = runner.run_forward(
                observations,
                actions if actions is not None else torch.zeros_like(observations),
                cache,
                names_filter,
                no_grad=True,
            )
            # Convert LatentTrajectory -> WorldTrajectory (best-effort)
            states_out = []
            for i, s in enumerate(latent_traj.states):
                # try to extract common fields, fallback to placeholders
                try:
                    h = s.h_t
                except Exception:
                    h = torch.zeros(1)
                try:
                    z_post = s.z_posterior
                except Exception:
                    z_post = torch.zeros(1)
                ws = WorldState(
                    state=h.clone(),
                    timestep=i,
                    action=None,
                    action_source=None,
                    reward_pred=getattr(s, "reward_pred", None),
                    value_pred=getattr(s, "value_pred", None),
                    obs_encoding=getattr(s, "obs_encoding", None),
                )
                states_out.append(ws)
            traj = WorldTrajectory(states=states_out, source="real")
            if device:
                traj = traj.to_device(device)
                cache = cache.to_device(device)
            return traj, cache

        for t in range(T):
            obs = observations[t]
            action = None
            if actions is not None and t < len(actions):
                action = actions[t]

            hook_ctx = HookContext(timestep=t, component="state", trajectory_so_far=states)
            # apply hooks and cache via central helper
            state = self._apply_and_cache("state", t, state, hook_ctx, cache, names_filter)

            posterior, obs_encoding = self.adapter.encode(
                obs.unsqueeze(0), state.unsqueeze(0) if state.dim() == 1 else state
            )
            posterior = posterior.squeeze(0)

            hook_ctx_z = HookContext(timestep=t, component="z_posterior", trajectory_so_far=states)
            posterior = self._apply_and_cache(
                "z_posterior", t, posterior, hook_ctx_z, cache, names_filter
            )

            obs_encoding = obs_encoding.squeeze(0) if obs_encoding is not None else None

            prior = self.adapter.dynamics(
                state.unsqueeze(0) if state.dim() == 1 else state,
            )
            prior = prior.squeeze(0) if prior.dim() > 1 else prior

            # cache common components via helper so hooks are applied uniformly
            # 'h' is essentially the same as state in this wrapper
            self._apply_and_cache("h", t, state, hook_ctx, cache, names_filter)
            # posterior was already passed through hooks above; it has been
            # cached by the central helper when appropriate.
            self._apply_and_cache(
                "z_prior",
                t,
                prior,
                HookContext(timestep=t, component="z_prior", trajectory_so_far=states),
                cache,
                names_filter,
            )
            if obs_encoding is not None:
                self._apply_and_cache(
                    "observation",
                    t,
                    obs_encoding,
                    HookContext(timestep=t, component="observation", trajectory_so_far=states),
                    cache,
                    names_filter,
                )

            # Dedicated hook point for KV-style cache manipulation. Some
            # transformer-based adapters maintain a growing key/value memory
            # that users may want to edit without re-running the whole
            # sequence. We expose a hook component named "kv_cache" which
            # receives the full ActivationCache and can mutate it in-place.
            manager = getattr(self, "_hook_cache_manager", None)
            if manager is not None:
                manager.apply_kv_hooks(
                    cache,
                    t,
                    HookContext(timestep=t, component="kv_cache", trajectory_so_far=states),
                )

            # Check for optional target encoder (e.g. for I-JEPA/JEPA models)
            if hasattr(self.adapter, "target_encode"):
                target_encoding = self.adapter.target_encode(obs.unsqueeze(0))
                if target_encoding is not None:
                    target_encoding = target_encoding.squeeze(0)
                    self._apply_and_cache(
                        "target_encoding",
                        t,
                        target_encoding,
                        HookContext(
                            timestep=t, component="target_encoding", trajectory_so_far=states
                        ),
                        cache,
                        names_filter,
                    )

            reward_pred = None
            if caps.has_reward_head:
                try:
                    reward_pred = self.adapter.predict_reward(
                        state.unsqueeze(0) if state.dim() == 1 else state,
                        posterior.unsqueeze(0) if posterior.dim() == 1 else posterior,
                    )
                    if reward_pred is not None:
                        self._apply_and_cache(
                            "reward",
                            t,
                            reward_pred.squeeze(0),
                            HookContext(timestep=t, component="reward", trajectory_so_far=states),
                            cache,
                            names_filter,
                        )
                except NotImplementedError:
                    pass

            value_pred = None
            if caps.has_critic:
                try:
                    if hasattr(self.adapter, "critic_forward"):
                        # Newer base-adapter implementations expose
                        # critic_forward(h, z), so we pass state/posterior.
                        value_pred = self.adapter.critic_forward(
                            state.unsqueeze(0) if state.dim() == 1 else state,
                            posterior.unsqueeze(0) if posterior.dim() == 1 else posterior,
                        )
                    else:
                        # Legacy generic adapters expose predict_value(state, action),
                        # so this fallback intentionally follows that older calling
                        # convention rather than critic_forward's (h, z) API.
                        value_pred = self.adapter.predict_value(
                            state.unsqueeze(0) if state.dim() == 1 else state,
                            action,
                        )
                    if value_pred is not None:
                        self._apply_and_cache(
                            "value",
                            t,
                            value_pred.squeeze(0),
                            HookContext(timestep=t, component="value", trajectory_so_far=states),
                            cache,
                            names_filter,
                        )
                except NotImplementedError:
                    pass

            kl = None
            if posterior.shape == prior.shape and posterior.dim() > 1:
                p = posterior.clamp(min=1e-8)
                q = prior.clamp(min=1e-8)
                p = p / p.sum(dim=-1, keepdim=True)
                q = q / q.sum(dim=-1, keepdim=True)
                kl = (p * (p.log() - q.log())).sum(dim=-1)
                # allow hooks to observe/modify KL and centralize caching
                self._apply_and_cache(
                    "kl",
                    t,
                    kl,
                    HookContext(timestep=t, component="kl", trajectory_so_far=states),
                    cache,
                    names_filter,
                )

            if caps.has_decoder:
                try:
                    recon = self._call_decode(
                        state.unsqueeze(0) if state.dim() == 1 else state,
                        posterior.unsqueeze(0) if posterior.dim() == 1 else posterior,
                    )
                    if recon is not None:
                        self._apply_and_cache(
                            "reconstruction",
                            t,
                            recon.squeeze(0),
                            HookContext(
                                timestep=t, component="reconstruction", trajectory_so_far=states
                            ),
                            cache,
                            names_filter,
                        )
                except NotImplementedError:
                    pass

            action_for_transition = action if caps.uses_actions else None
            if action_for_transition is not None:
                hook_ctx_a = HookContext(timestep=t, component="action", trajectory_so_far=states)
                action_for_transition = self._hooks.apply(
                    "action", t, action_for_transition, hook_ctx_a
                )
                action = action_for_transition

            next_state = self._call_transition(
                state.unsqueeze(0) if state.dim() == 1 else state,
                posterior.unsqueeze(0) if posterior.dim() == 1 else posterior,
                action_for_transition,
            )
            state = next_state.squeeze(0)

            # Apply transition hook for causality analysis
            transition_ctx = HookContext(
                timestep=t,
                component="transition",
                trajectory_so_far=states,
                metadata={
                    "s_t": state.clone(),  # Current state after transition
                    "s_prev": states[-1].state if states else None,  # Previous state
                    "a_t": action_for_transition,  # Action used
                    "z_t": posterior,  # Latent encoding
                },
            )
            state = self._hooks.apply("transition", t, state, transition_ctx)

            action_for_state = action.clone() if action is not None else None
            action_source = None
            if action_for_state is not None:
                from world_model_lens.core.world_state import ActionSource

                action_source = ActionSource(
                    source_type="externally_provided",
                    temperature=None,
                )

            state_obj = WorldState(
                state=state.clone(),
                timestep=t,
                action=action_for_state,
                action_source=action_source,
                reward_pred=reward_pred.squeeze(0).clone() if reward_pred is not None else None,
                value_pred=value_pred.squeeze(0).clone() if value_pred is not None else None,
                obs_encoding=obs_encoding.clone() if obs_encoding is not None else None,
            )
            states.append(state_obj)

            if z_current is not None:
                z_current = posterior

        traj = WorldTrajectory(
            states=states,
            source="real",
        )

        if device:
            traj = traj.to_device(device)
            cache = cache.to_device(device)

        return traj, cache

    def run_with_hooks(
        self,
        observations: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        fwd_hooks: Optional[Union[List[HookPoint], Tuple[HookPoint, ...]]] = None,
        return_cache: bool = False,
    ) -> Union[WorldTrajectory, Tuple[WorldTrajectory, ActivationCache]]:
        """Run forward pass with temporary hooks.

        Hooks fire after each activation is computed but before it feeds
        into the next computation step.

        Args:
            observations: Observation sequence
            actions: Optional action sequence
            fwd_hooks: List of HookPoints to apply
            return_cache: If True, also return ActivationCache

        Returns:
            WorldTrajectory, or (WorldTrajectory, ActivationCache) if return_cache
        """
        cache: ActivationCache | None = ActivationCache() if return_cache else None

        # Use the registry's context manager so temporary hooks are registered
        # and cleaned up automatically, even if the forward pass raises.
        # coerce names_filter absent -> None handled by run_with_cache
        with self._hooks.temp_hooks(list(fwd_hooks) if fwd_hooks else []):
            traj, cache = self.run_with_cache(observations, actions)

        if return_cache:
            return traj, cache
        return traj

    def imagine(
        self,
        start_state: WorldState,
        actions: Optional[torch.Tensor] = None,
        horizon: int = 50,
        temperature: float = 1.0,
    ) -> WorldTrajectory:
        """Imagine forward from a starting state using dynamics model.

        Works with ANY world model - RL and non-RL. For non-RL models,
        actions are ignored.

        Args:
            start_state: Starting WorldState
            actions: Optional action sequence to execute (ignored for non-RL models)
            horizon: Number of imagination steps
            temperature: Sampling temperature for discrete states

        Returns:
            Imagined WorldTrajectory
        """
        caps = self._get_capabilities()
        state = start_state.state.clone()
        z = start_state.obs_encoding if start_state.obs_encoding is not None else state
        states = [start_state]

        for t in range(horizon):
            action = None
            action_source = None

            if caps.uses_actions:
                if actions is not None and t < len(actions):
                    # Externally provided action
                    action = actions[t]
                    from world_model_lens.core.world_state import ActionSource

                    action_source = ActionSource(
                        source_type="externally_provided",
                        temperature=None,
                    )
                elif caps.has_actor:
                    # Sample action from policy
                    try:
                        policy_logits = self.adapter.actor_forward(
                            state.unsqueeze(0) if state.dim() == 1 else state,
                            z.unsqueeze(0) if z.dim() == 1 else z,
                        )
                        if policy_logits is not None:
                            # Sample action from policy
                            if policy_logits.dim() > 2:
                                policy_logits = policy_logits.squeeze(0)

                            # Handle different action distributions
                            if policy_logits.shape[-1] == 1:
                                # Continuous action
                                action = torch.tanh(policy_logits)  # Assuming tanh squashing
                            else:
                                # Discrete action
                                probs = torch.softmax(policy_logits / temperature, dim=-1)
                                action_idx = torch.multinomial(probs, 1).item()
                                action = torch.tensor([action_idx], dtype=torch.long)

                            from world_model_lens.core.world_state import ActionSource

                            action_source = ActionSource(
                                source_type="policy_sampled",
                                policy_logits=policy_logits.detach(),
                                temperature=temperature,
                            )
                    except NotImplementedError:
                        pass

            prior = self.adapter.dynamics(
                state.unsqueeze(0) if state.dim() == 1 else state,
            )
            if prior.dim() > 1:
                z = self.adapter.sample_z(prior.squeeze(0), temperature=temperature)
            else:
                z = prior.squeeze(0) if prior.dim() == 1 else prior

            prev_state = state.clone()
            state = self._call_transition(state, z, action)

            # Apply transition hook for causality analysis
            transition_ctx = HookContext(
                timestep=t + start_state.timestep + 1,
                component="transition",
                trajectory_so_far=states,
                metadata={
                    "s_t": state.clone(),  # Current state after transition
                    "s_prev": prev_state,  # Previous state
                    "a_t": action,  # Action used
                    "z_t": z,  # Latent encoding
                },
            )
            state = self._hooks.apply(
                "transition", t + start_state.timestep + 1, state, transition_ctx
            )

            reward_pred = None
            if caps.has_reward_head and t > 0:
                try:
                    reward_pred = self.adapter.predict_reward(state, z)
                    reward_pred = reward_pred.squeeze(0) if reward_pred is not None else None
                except NotImplementedError:
                    pass

            state_obj = WorldState(
                state=state.clone(),
                timestep=t + start_state.timestep + 1,
                action=action.clone() if action is not None else None,
                action_source=action_source,
                reward_pred=reward_pred,
            )
            states.append(state_obj)

        return WorldTrajectory(
            states=states[1:],
            source="imagined",
            fork_point=start_state.timestep,
        )

    def add_hook(self, hook: HookPoint) -> None:
        """Add a persistent hook."""
        self._hooks.register(hook)

    def remove_hook(self, hook: HookPoint) -> None:
        """Remove a specific hook."""
        # Delegate to the registry's remove method so only the provided
        # HookPoint is removed (no-op if it isn't registered).
        self._hooks.remove(hook)

    def clear_hooks(self) -> None:
        """Remove all hooks."""
        self._hooks.clear()

    @property
    def named_weights(self) -> Dict[str, torch.Tensor]:
        """All weight matrices from the adapter."""
        return dict(self.adapter.named_parameters())

    @property
    def device(self) -> torch.device:
        """Current device."""
        return self._device

    @property
    def hook_registry(self) -> HookRegistry:
        """Access the hook registry."""
        return self._hooks

    @property
    def capabilities(self) -> WorldModelCapabilities:
        """Access the adapter's capabilities descriptor.

        Returns:
            WorldModelCapabilities indicating which optional features are available.
        """
        return self._get_capabilities()
