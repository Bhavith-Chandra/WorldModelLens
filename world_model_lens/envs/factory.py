"""Environment factory for creating environment adapters.

Supports auto-detection and prefix-based registry for different
ecosystems (Gymnasium, ProcGen, Isaac Lab, etc.).
"""

from typing import Any, Dict, Optional, Type, Union
import warnings

from world_model_lens.envs.base import EnvironmentAdapter, EnvironmentCapabilities


class AdapterInfo:
    """Metadata about a registered environment adapter."""

    def __init__(
        self,
        cls: Type[EnvironmentAdapter],
        capabilities: EnvironmentCapabilities,
        description: str = "",
    ):
        self.cls = cls
        self.capabilities = capabilities
        self.description = description


class EnvironmentFactory:
    """Factory for creating environment adapters with prefix-based registry.

    Supports auto-detection based on environment ID prefix:
    - "gym:" -> GymnasiumAdapter
    - "procgen:" -> ProcGenAdapter
    - "isaac:" -> IsaacLabAdapter

    Example:
        # Explicit prefix
        env = EnvironmentFactory.create("gym:CartPole-v1")

        # Auto-detect (will try to match known prefixes)
        env = EnvironmentFactory.create("CartPole-v1")

        # Direct class
        env = EnvironmentFactory.create(GymnasiumAdapter, "CartPole-v1")
    """

    def __init__(self):
        self._registry: Dict[str, AdapterInfo] = {}
        self._default_prefix = "gym"

    def register(
        self,
        prefix: str,
        cls: Type[EnvironmentAdapter],
        capabilities: Optional[EnvironmentCapabilities] = None,
        description: str = "",
        set_default: bool = False,
    ) -> None:
        """Register an environment adapter.

        Args:
            prefix: Prefix for auto-detection (e.g., "gym", "procgen", "isaac").
            cls: Adapter class.
            capabilities: Adapter capabilities.
            description: Human-readable description.
            set_default: If True, set this as the default prefix.
        """
        if capabilities is None:
            capabilities = EnvironmentCapabilities()

        self._registry[prefix] = AdapterInfo(
            cls=cls,
            capabilities=capabilities,
            description=description,
        )

        if set_default:
            self._default_prefix = prefix

    def create(
        self,
        env_config: Union[str, Dict[str, Any], Type[EnvironmentAdapter]],
        *args,
        **kwargs,
    ) -> EnvironmentAdapter:
        """Create an environment adapter.

        Args:
            env_config: Environment config (env ID string, dict, or adapter class).
            *args: Additional positional arguments for adapter.
            **kwargs: Additional keyword arguments for adapter.
                     If prefix is in kwargs, it overrides auto-detection.

        Returns:
            EnvironmentAdapter instance.

        Raises:
            ValueError: If adapter cannot be resolved.
        """
        if isinstance(env_config, type) and issubclass(env_config, EnvironmentAdapter):
            return env_config(env_config, *args, **kwargs)

        if isinstance(env_config, dict):
            explicit_prefix = kwargs.pop("prefix", None)
            if explicit_prefix:
                env_id = env_config.get("env_id", env_config.get("id", ""))
                return self._create_from_prefix(explicit_prefix, env_id, *args, **kwargs)
            if "prefix" in env_config:
                prefix = env_config.pop("prefix")
                return self._create_from_prefix(prefix, env_config, *args, **kwargs)
            env_id = env_config.get("env_id", env_config.get("id", ""))
        else:
            env_id = str(env_config)

        prefix, env_id = self._extract_prefix(env_id)
        return self._create_from_prefix(prefix, env_id, *args, **kwargs)

    def _extract_prefix(self, env_id: str) -> tuple:
        """Extract prefix from env_id.

        Args:
            env_id: Environment ID (e.g., "gym:CartPole-v1" or "CartPole-v1").

        Returns:
            Tuple of (prefix, remaining_env_id).
        """
        for prefix in self._registry:
            if env_id.startswith(f"{prefix}:"):
                return prefix, env_id[len(prefix) + 1 :]

        return self._default_prefix, env_id

    def _create_from_prefix(
        self,
        prefix: str,
        env_id: str,
        *args,
        **kwargs,
    ) -> EnvironmentAdapter:
        """Create adapter from prefix.

        Args:
            prefix: Registered prefix.
            env_id: Environment ID.
            *args: Additional args.
            **kwargs: Additional kwargs.

        Returns:
            EnvironmentAdapter instance.
        """
        if prefix not in self._registry:
            available = list(self._registry.keys())
            raise ValueError(
                f"Unknown prefix '{prefix}'. Available: {available}. "
                f"Use create() with explicit class or register a new adapter."
            )

        info = self._registry[prefix]
        return info.cls(env_id, *args, **kwargs)

    def get_info(self, prefix: str) -> AdapterInfo:
        """Get adapter info by prefix.

        Args:
            prefix: Registered prefix.

        Returns:
            AdapterInfo with metadata.
        """
        if prefix not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(f"Prefix '{prefix}' not found. Available: {available}")
        return self._registry[prefix]

    def __contains__(self, prefix: str) -> bool:
        return prefix in self._registry

    def list_prefixes(self) -> list:
        """List all registered prefixes."""
        return list(self._registry.keys())

    def set_default(self, prefix: str) -> None:
        """Set the default prefix for auto-detection.

        Args:
            prefix: Prefix to use as default.
        """
        if prefix not in self._registry:
            raise KeyError(
                f"Prefix '{prefix}' not registered. Available: {list(self._registry.keys())}"
            )
        self._default_prefix = prefix


FACTORY = EnvironmentFactory()


def register(
    prefix: str,
    capabilities: Optional[EnvironmentCapabilities] = None,
    description: str = "",
    set_default: bool = False,
):
    """Decorator to register an environment adapter.

    Usage:
        @register("procgen", EnvironmentCapabilities(...), "ProcGen environments")
        class ProcGenAdapter(EnvironmentAdapter):
            ...
    """

    def decorator(cls: Type[EnvironmentAdapter]) -> Type[EnvironmentAdapter]:
        FACTORY.register(
            prefix=prefix,
            cls=cls,
            capabilities=capabilities,
            description=description,
            set_default=set_default,
        )
        return cls

    return decorator


def create(
    env_config: Union[str, Dict[str, Any], Type[EnvironmentAdapter]], *args, **kwargs
) -> EnvironmentAdapter:
    """Create an environment adapter using the global factory.

    Args:
        env_config: Environment config (env ID string, dict, or adapter class).
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        EnvironmentAdapter instance.
    """
    return FACTORY.create(env_config, *args, **kwargs)
