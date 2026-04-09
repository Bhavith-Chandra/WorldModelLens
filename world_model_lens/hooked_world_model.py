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
from world_model_lens.core.latent_state import LatentState
from world_model_lens.core.latent_trajectory import LatentTrajectory
from world_model_lens.core.world_state import WorldState
from world_model_lens.core.world_trajectory import WorldTrajectory
from world_model_lens.backends.base_adapter import WorldModelCapabilities


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
        >>> from world_model_lens.backends.generic_adapter import WorldModelAdapter
        >>>
        >>> # Your world model adapter
        >>> class MyModel(WorldModelAdapter):
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
            adapter: Any WorldModelAdapter implementation
            config: WorldModelConfig or dict-like config
            name: Optional name for this instance
        """
        self.adapter = adapter
        self.config = config
        self.name = name
        self._hooks = HookRegistry()
        self._device = torch.device("cpu")
        self._uses_capabilities_adapter_api = isinstance(
            getattr(type(self.adapter), "capabilities", None), property
        )

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
        from world_model_lens.backends import BackendRegistry

        registry = BackendRegistry()
        adapter_cls = registry.get(backend)
        adapter = adapter_cls(config) if config else adapter_cls.from_checkpoint(path)

        if device:
            adapter = adapter.to(device)

        return cls(adapter=adapter, config=config or adapter.config, name=path)

    def run_with_cache(
        self,
        observations: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        names_filter: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
        store_logits: bool = True,
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

        Returns:
            Tuple of (WorldTrajectory, LatentTrajectory, ActivationCache)
        """
        T = observations.shape[0]
        cache = ActivationCache()
        states = []
        latent_states = []

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

        for t in range(T):
            obs = observations[t]
            action = None
            if actions is not None and t < len(actions):
                action = actions[t]

            hook_ctx = HookContext(timestep=t, component="state", trajectory_so_far=states)
            state = self._hooks.apply("state", t, state, hook_ctx)

            posterior, obs_encoding = self.adapter.encode(
                obs.unsqueeze(0), state.unsqueeze(0) if state.dim() == 1 else state
            )
            posterior = posterior.squeeze(0)

            hook_ctx_z = HookContext(timestep=t, component="z_posterior", trajectory_so_far=states)
            posterior = self._hooks.apply("z_posterior", t, posterior, hook_ctx_z)

            obs_encoding = obs_encoding.squeeze(0) if obs_encoding is not None else None

            prior = self.adapter.dynamics(
                state.unsqueeze(0) if state.dim() == 1 else state,
            )
            prior = prior.squeeze(0) if prior.dim() > 1 else prior

            cache["state", t] = state.detach()
            cache["z_posterior", t] = posterior.detach()
            cache["z_prior", t] = prior.detach()
            cache["h", t] = state.detach()
            if obs_encoding is not None:
                cache["observation", t] = obs_encoding.detach()

            reward_pred = None
            if caps.has_reward_head:
                try:
                    reward_pred = self.adapter.predict_reward(
                        state.unsqueeze(0) if state.dim() == 1 else state,
                        posterior.unsqueeze(0) if posterior.dim() == 1 else posterior,
                    )
                    if reward_pred is not None:
                        cache["reward", t] = reward_pred.squeeze(0).detach()
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
                        cache["value", t] = value_pred.squeeze(0).detach()
                except NotImplementedError:
                    pass

            kl = None
            if posterior.shape == prior.shape and posterior.dim() > 1:
                p = posterior.clamp(min=1e-8)
                q = prior.clamp(min=1e-8)
                p = p / p.sum(dim=-1, keepdim=True)
                q = q / q.sum(dim=-1, keepdim=True)
                kl = (p * (p.log() - q.log())).sum(dim=-1)
                cache["kl", t] = kl.detach()

            if caps.has_decoder:
                try:
                    recon = self._call_decode(
                        state.unsqueeze(0) if state.dim() == 1 else state,
                        posterior.unsqueeze(0) if posterior.dim() == 1 else posterior,
                    )
                    if recon is not None:
                        cache["reconstruction", t] = recon.squeeze(0).detach()
                except NotImplementedError:
                    pass

            action_for_transition = action if caps.uses_actions else None
            if action_for_transition is not None:
                hook_ctx_a = HookContext(timestep=t, component="action", trajectory_so_far=states)
                action_for_transition = self._hooks.apply(
                    "action", t, action_for_transition, hook_ctx_a
                )
                action = action_for_transition

            action_for_state = action.clone() if action is not None else None

            latent_state = LatentState(
                h_t=state.detach().clone(),
                z_posterior=posterior.detach().clone(),
                z_prior=prior.detach().clone(),
                timestep=t,
                action=action_for_state.detach().clone() if action_for_state is not None else None,
                reward_pred=reward_pred.squeeze(0).detach().clone()
                if reward_pred is not None
                else None,
                value_pred=value_pred.squeeze(0).detach().clone() if value_pred is not None else None,
                obs_encoding=obs_encoding.detach().clone() if obs_encoding is not None else None,
            )
            latent_states.append(latent_state)

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

        latent_traj = LatentTrajectory(
            states=latent_states,
            env_name=self.name,
            imagined=False,
            metadata={"source": "run_with_cache"},
        )
        traj = WorldTrajectory(
            states=states,
            source="real",
        )

        if device:
            traj = traj.to_device(device)
            cache = cache.to_device(device)

        return traj, latent_traj, cache

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
        if fwd_hooks:
            for hook in fwd_hooks:
                self._hooks.register(hook)

        try:
            traj, _, cache = self.run_with_cache(observations, actions)
        finally:
            if fwd_hooks:
                self._hooks.clear()

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
        self._hooks.clear()

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
