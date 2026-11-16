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

from world_model_lens.core.hooks import HookContext, HookPoint, HookRegistry
from world_model_lens.core.activation_cache import ActivationCache
from world_model_lens.core.world_state import ActionSource, WorldState
from world_model_lens.core.world_trajectory import WorldTrajectory
from world_model_lens.backends.base_adapter import WorldModelCapabilities
from world_model_lens.core.hook_cache import HookCacheManager
from world_model_lens.core.forward_runner import ForwardRunner
from world_model_lens.core.latent_state import LatentState
from world_model_lens.core.latent_trajectory import LatentTrajectory
from world_model_lens.core.types import WorldModelFamily


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
        >>> world_traj, latent_traj, cache = wm.run_with_cache(observations)
        >>> imagined_world, imagined_latent = wm.imagine(
        ...     start_state=latent_traj.states[0], horizon=20
        ... )
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

    def _latent_logprobs(self, logits: torch.Tensor) -> torch.Tensor:
        latent_logprobs = getattr(self.adapter, "latent_logprobs", None)
        if callable(latent_logprobs):
            return latent_logprobs(logits)
        return torch.log_softmax(logits.float(), dim=-1)

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

    def _get_world_model_family(self) -> Optional[WorldModelFamily]:
        """Resolve world-model family from wrapper config first, then adapter config."""
        family = getattr(self.config, "world_model_family", None)
        if family is not None:
            return family
        return getattr(getattr(self.adapter, "config", None), "world_model_family", None)

    @staticmethod
    def _normalize_world_state_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Keep JEPA patch states compact when converting to WorldTrajectory."""
        if tensor.dim() > 0 and tensor.shape[0] == 1:
            return tensor.squeeze(0)
        return tensor

    def _latent_traj_to_world_traj(self, latent_traj: LatentTrajectory) -> WorldTrajectory:
        """Convert ForwardRunner output into the public WorldTrajectory API."""
        states_out = []
        family = self._get_world_model_family()
        for i, latent_state in enumerate(latent_traj.states):
            h_t = getattr(latent_state, "h_t", None)
            if not isinstance(h_t, torch.Tensor):
                raise TypeError(
                    "ForwardRunner produced a latent state without a tensor h_t; "
                    f"got {type(h_t).__name__} at index {i}."
                )

            obs_encoding = getattr(latent_state, "obs_encoding", None)
            if isinstance(obs_encoding, torch.Tensor):
                obs_encoding = self._normalize_world_state_tensor(obs_encoding.detach().clone())

            metadata = dict(getattr(latent_state, "metadata", {}) or {})
            z_posterior = getattr(latent_state, "z_posterior", None)
            z_prior = getattr(latent_state, "z_prior", None)
            if isinstance(z_posterior, torch.Tensor):
                metadata.setdefault(
                    "z_posterior",
                    self._normalize_world_state_tensor(z_posterior.detach().clone()),
                )
            if isinstance(z_prior, torch.Tensor):
                metadata.setdefault(
                    "z_prior",
                    self._normalize_world_state_tensor(z_prior.detach().clone()),
                )

            states_out.append(
                WorldState(
                    state=self._normalize_world_state_tensor(h_t.detach().clone()),
                    timestep=getattr(latent_state, "timestep", i),
                    action=getattr(latent_state, "action", None),
                    action_source=None,
                    reward_pred=getattr(latent_state, "reward_pred", None),
                    value_pred=getattr(latent_state, "value_pred", None),
                    obs_encoding=obs_encoding,
                    metadata=metadata,
                )
            )

        return WorldTrajectory(
            states=states_out,
            name=self.name,
            source="imagined" if latent_traj.imagined else "real",
            metadata={
                "forward_runner": True,
                "world_model_family": family.name if hasattr(family, "name") else family,
                "latent_episode_id": latent_traj.episode_id,
            },
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
        sample_latents: bool = False,
    ) -> Tuple[WorldTrajectory, LatentTrajectory, ActivationCache]:
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
            sample_latents: For adapters with categorical latents, sample z from
                the posterior instead of taking its mode.

        Each step t runs ``encode(obs_t, h_t)``, then ``transition(h_t, z_t, a_t)``.
        The states returned for step t hold h_t, the state that observed obs_t.
        For adapters with ``capabilities.categorical_latents``, the cache holds
        posterior and prior logits, ``z`` holds the latent fed to the transition,
        and the latent states hold probabilities.

        Returns:
            Tuple of (WorldTrajectory, LatentTrajectory, ActivationCache)
        """
        # normalize names_filter to a set for efficient membership tests
        # normalize names_filter to a set[str] for efficient membership checks
        names_filter = set(names_filter) if names_filter is not None else None

        T = observations.shape[0]
        cache = ActivationCache()
        world_states = []
        latent_states = []

        caps = self._get_capabilities()

        # Handle both enum and string for world model family
        family = self._get_world_model_family()
        if isinstance(family, str):
            # Handle various string formats: 'ijepa', 'jea', etc.
            family_upper = family.upper()
            if "JEPA" in family_upper:
                family = WorldModelFamily.JEPA
            else:
                try:
                    family = WorldModelFamily(family_upper)
                except ValueError:
                    family = None
        # else: family is already an enum or None

        if family == WorldModelFamily.JEPA:
            runner = ForwardRunner(self)
            action_seq = actions
            if action_seq is None:
                action_seq = torch.zeros((T, 1), device=observations.device)
            latent_traj = runner.run_forward(
                observations,
                action_seq,
                cache,
                names_filter,
                no_grad=True,
            )
            world_traj = self._latent_traj_to_world_traj(latent_traj)
            if device:
                world_traj = world_traj.to_device(device)
                latent_traj = latent_traj.to_device(device)
                cache = cache.to_device(device)
            return world_traj, latent_traj, cache

        init_result = self.adapter.initial_state(batch_size=1, device=observations.device)
        state = init_result[0] if isinstance(init_result, tuple) else init_result
        if state.dim() > 1 and state.shape[0] == 1:
            state = state.squeeze(0)

        categorical = caps.categorical_latents
        last_target_obs = None
        last_target_encoding = None

        def ctx(component: str, **metadata: Any) -> HookContext:
            return HookContext(
                timestep=t, component=component, trajectory_so_far=latent_states, metadata=metadata
            )

        def hook(component: str, tensor: torch.Tensor) -> torch.Tensor:
            return self._apply_and_cache(component, t, tensor, ctx(component), cache, names_filter)

        def batched(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.unsqueeze(0) if tensor.dim() == 1 else tensor

        for t in range(T):
            obs = observations[t]
            action = actions[t] if actions is not None and t < len(actions) else None

            state = hook("state", state)
            state = hook("h", state)

            if hasattr(self.adapter, "hooks"):
                self.adapter.hooks = self._hooks
            if hasattr(self.adapter, "current_timestep"):
                self.adapter.current_timestep = t

            posterior, obs_encoding = self.adapter.encode(obs.unsqueeze(0), batched(state))
            posterior = hook("z_posterior", posterior.squeeze(0))

            prior = self.adapter.dynamics(batched(state))
            prior = hook("z_prior", prior.squeeze(0) if prior.dim() > 1 else prior)

            if obs_encoding is not None:
                obs_encoding = hook("observation", obs_encoding.squeeze(0))

            # kv_cache hooks receive the whole ActivationCache and may edit it in place.
            manager = getattr(self, "_hook_cache_manager", None)
            if manager is not None:
                manager.apply_kv_hooks(cache, t, ctx("kv_cache"))

            if hasattr(self.adapter, "target_encode"):
                if last_target_obs is not None and torch.equal(obs, last_target_obs):
                    target_encoding = last_target_encoding
                else:
                    target_encoding = self.adapter.target_encode(obs.unsqueeze(0))
                if target_encoding is not None:
                    target_encoding = target_encoding.squeeze(0)
                    last_target_obs = obs.detach().clone()
                    last_target_encoding = target_encoding
                    hook("target_encoding", target_encoding)

            if categorical:
                z = hook("z", self.adapter.sample_z(posterior, sample=sample_latents))
            else:
                z = posterior

            reward_pred = None
            if caps.has_reward_head:
                try:
                    reward_pred = self.adapter.predict_reward(batched(state), batched(z))
                except NotImplementedError:
                    pass
                if reward_pred is not None:
                    reward_pred = hook("reward", reward_pred.squeeze(0))

            value_pred = None
            if caps.has_critic:
                try:
                    if hasattr(self.adapter, "critic_forward"):
                        value_pred = self.adapter.critic_forward(batched(state), batched(z))
                    else:
                        # Legacy generic adapters expose predict_value(state, action).
                        value_pred = self.adapter.predict_value(batched(state), action)
                except NotImplementedError:
                    pass
                if value_pred is not None:
                    value_pred = hook("value", value_pred.squeeze(0))

            if categorical:
                posterior_logp = self._latent_logprobs(posterior)
                prior_logp = self._latent_logprobs(prior)
                kl = (posterior_logp.exp() * (posterior_logp - prior_logp)).sum(dim=-1)
                hook("kl", kl)
                posterior_probs, prior_probs = posterior_logp.exp(), prior_logp.exp()
            else:
                if posterior.shape == prior.shape and posterior.dim() > 1:
                    p = posterior.clamp(min=1e-8)
                    q = prior.clamp(min=1e-8)
                    p = p / p.sum(dim=-1, keepdim=True)
                    q = q / q.sum(dim=-1, keepdim=True)
                    hook("kl", (p * (p.log() - q.log())).sum(dim=-1))
                posterior_probs, prior_probs = posterior, prior

            if caps.has_decoder:
                try:
                    recon = self._call_decode(batched(state), batched(z))
                except NotImplementedError:
                    recon = None
                if recon is not None:
                    hook("reconstruction", recon.squeeze(0))

            if action is not None and caps.uses_actions:
                action = self._hooks.apply("action", t, action, ctx("action"))
            action_for_transition = action if caps.uses_actions else None

            next_state = self._call_transition(
                batched(state), batched(z), action_for_transition
            ).squeeze(0)
            next_state = self._hooks.apply(
                "transition",
                t,
                next_state,
                ctx(
                    "transition",
                    s_t=next_state.clone(),
                    s_prev=state.clone(),
                    a_t=action_for_transition,
                    z_t=z,
                ),
            )

            action_for_state = action.clone() if action is not None else None
            action_source = None
            if action_for_state is not None:
                action_source = ActionSource(source_type="externally_provided", temperature=None)

            world_states.append(
                WorldState(
                    state=state.clone(),
                    timestep=t,
                    action=action_for_state,
                    action_source=action_source,
                    reward_pred=reward_pred.clone() if reward_pred is not None else None,
                    value_pred=value_pred.clone() if value_pred is not None else None,
                    obs_encoding=obs_encoding.clone() if obs_encoding is not None else None,
                )
            )
            latent_states.append(
                LatentState(
                    h_t=state.clone(),
                    z_posterior=posterior_probs.clone(),
                    z_prior=prior_probs.clone(),
                    timestep=t,
                    action=action_for_state.clone() if action_for_state is not None else None,
                    reward_pred=reward_pred.clone() if reward_pred is not None else None,
                    value_pred=value_pred.clone() if value_pred is not None else None,
                    obs_encoding=obs_encoding.clone() if obs_encoding is not None else None,
                    z=z.clone() if categorical else None,
                )
            )

            state = next_state

        world_traj = WorldTrajectory(
            states=world_states,
            source="real",
        )
        latent_traj = LatentTrajectory(states=latent_states, env_name=self.name, imagined=False)

        if device:
            world_traj = world_traj.to_device(device)
            latent_traj = latent_traj.to_device(device)
            cache = cache.to_device(device)

        return world_traj, latent_traj, cache

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
            world_traj, _, cache = self.run_with_cache(observations, actions)

        if return_cache:
            return world_traj, cache
        return world_traj

    def imagine(
        self,
        start_state: Union[WorldState, LatentState],
        actions: Optional[torch.Tensor] = None,
        horizon: int = 50,
        temperature: float = 1.0,
    ) -> Tuple[WorldTrajectory, LatentTrajectory]:
        """Imagine forward from a starting state using dynamics model.

        Works with ANY world model - RL and non-RL. For non-RL models,
        actions are ignored.

        Each step runs ``h_{t+1} = transition(h_t, z_t, a_t)`` and then draws
        ``z_{t+1}`` from ``dynamics(h_{t+1})``, starting from the start state's
        latent. Adapters with categorical latents need a LatentState from
        ``run_with_cache`` or ``imagine``, which records the latent in ``z``.

        Args:
            start_state: Starting WorldState or LatentState
            actions: Optional action sequence to execute (ignored for non-RL models)
            horizon: Number of imagination steps
            temperature: Sampling temperature for discrete states

        Returns:
            Tuple of (WorldTrajectory, LatentTrajectory)
        """
        caps = self._get_capabilities()
        categorical = caps.categorical_latents
        start_timestep = start_state.timestep

        if isinstance(start_state, LatentState):
            if categorical and start_state.z is None:
                raise ValueError(
                    "imagine() needs the sampled latent of the start state; "
                    "use a LatentState returned by run_with_cache() or imagine()."
                )
            state = start_state.h_t.clone()
            z = (start_state.z if start_state.z is not None else start_state.z_posterior).clone()
        elif categorical:
            raise ValueError("imagine() needs a LatentState for adapters with categorical latents.")
        else:
            state = start_state.state.clone()
            z = start_state.obs_encoding if start_state.obs_encoding is not None else state

        def batched(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.unsqueeze(0) if tensor.dim() == 1 else tensor

        world_states = []
        latent_states = []

        for t in range(horizon):
            timestep = start_timestep + t + 1
            action = None
            action_source = None

            if caps.uses_actions:
                if actions is not None and t < len(actions):
                    action = actions[t]
                    action_source = ActionSource(source_type="externally_provided", temperature=None)
                elif caps.has_actor:
                    try:
                        policy_logits = self.adapter.actor_forward(batched(state), batched(z))
                    except NotImplementedError:
                        policy_logits = None
                    if policy_logits is not None:
                        if policy_logits.dim() > 2:
                            policy_logits = policy_logits.squeeze(0)
                        if policy_logits.shape[-1] == 1:
                            action = torch.tanh(policy_logits)
                        else:
                            probs = torch.softmax(policy_logits / temperature, dim=-1)
                            action_idx = torch.multinomial(probs, 1).item()
                            action = torch.tensor([action_idx], dtype=torch.long)
                        action_source = ActionSource(
                            source_type="policy_sampled",
                            policy_logits=policy_logits.detach(),
                            temperature=temperature,
                        )

            prev_state = state
            state = self._call_transition(batched(state), batched(z), action).squeeze(0)
            state = self._hooks.apply(
                "transition",
                timestep,
                state,
                HookContext(
                    timestep=timestep,
                    component="transition",
                    trajectory_so_far=latent_states,
                    metadata={"s_t": state.clone(), "s_prev": prev_state, "a_t": action, "z_t": z},
                ),
            )

            prior = self.adapter.dynamics(batched(state))
            prior = prior.squeeze(0) if prior.dim() > 1 else prior
            if categorical:
                z = self.adapter.sample_z(prior, temperature=temperature)
                prior_probs = self._latent_logprobs(prior).exp()
            else:
                z = self.adapter.sample_z(prior, temperature=temperature) if prior.dim() > 1 else prior
                prior_probs = z

            reward_pred = None
            if caps.has_reward_head:
                try:
                    reward_pred = self.adapter.predict_reward(batched(state), batched(z))
                except NotImplementedError:
                    pass
                if reward_pred is not None:
                    reward_pred = reward_pred.squeeze(0)

            world_states.append(
                WorldState(
                    state=state.clone(),
                    timestep=timestep,
                    action=action.clone() if action is not None else None,
                    action_source=action_source,
                    reward_pred=reward_pred,
                )
            )
            latent_states.append(
                LatentState(
                    h_t=state.clone(),
                    z_posterior=prior_probs.clone(),
                    z_prior=prior_probs.clone(),
                    timestep=timestep,
                    action=action.clone() if action is not None else None,
                    reward_pred=reward_pred.clone() if reward_pred is not None else None,
                    z=z.clone() if categorical else None,
                )
            )

        world_traj = WorldTrajectory(
            states=world_states,
            source="imagined",
            fork_point=start_timestep,
        )
        latent_traj = LatentTrajectory(
            states=latent_states,
            env_name=self.name,
            imagined=True,
            fork_point=start_timestep,
        )
        return world_traj, latent_traj

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

    def list_hookable_points(self) -> List[str]:
        """List hook targets exposed by the wrapper and its adapter.

        The wrapper always exposes its timestep-level hook points. If the
        adapter also advertises hook points, they are merged in so callers can
        discover both the public wrapper API and adapter-specific sites.
        """
        points = [
            "state",
            "h",
            "z_posterior",
            "z_prior",
            "z",
            "observation",
            "target_encoding",
            "kl",
            "reconstruction",
            "reward",
            "value",
            "action",
            "transition",
            "kv_cache",
        ]

        adapter = getattr(self, "adapter", None)
        if adapter is None:
            return points

        adapter_points: List[str] = []

        list_hookable_points = getattr(adapter, "list_hookable_points", None)
        if callable(list_hookable_points):
            try:
                adapter_points.extend(list(list_hookable_points()))
            except Exception:
                pass

        hook_point_names = getattr(adapter, "hook_point_names", None)
        if hook_point_names is not None:
            try:
                adapter_points.extend(list(hook_point_names))
            except TypeError:
                pass

        if self._get_world_model_family() == WorldModelFamily.JEPA:
            predictor = getattr(adapter, "predictor", None)
            blocks = getattr(predictor, "blocks", None)
            if blocks is not None:
                adapter_points.extend([f"predictor.layer_{i}" for i in range(len(blocks))])
            adapter_points.extend(
                [
                    "encoder.out",
                    "target_encoder.out",
                    "predictor.final",
                    "predictor_out",
                    "target_encoder_out",
                ]
            )

        return list(dict.fromkeys(points + adapter_points))

    @property
    def capabilities(self) -> WorldModelCapabilities:
        """Access the adapter's capabilities descriptor.

        Returns:
            WorldModelCapabilities indicating which optional features are available.
        """
        return self._get_capabilities()
