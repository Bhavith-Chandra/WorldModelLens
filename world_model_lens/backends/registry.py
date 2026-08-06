"""Backend registry for dynamic adapter lookup with family categorization."""

from typing import Dict, Type, Optional, List

from world_model_lens.core.types import WorldModelFamily


class AdapterInfo:
    """Metadata about a registered adapter."""

    def __init__(
        self,
        cls: Type,
        family: WorldModelFamily,
        description: str = "",
        supports_rl: bool = True,
        supports_video: bool = False,
        supports_planning: bool = False,
    ):
        self.cls = cls
        self.family = family
        self.description = description
        self.supports_rl = supports_rl
        self.supports_video = supports_video
        self.supports_planning = supports_planning


class BackendRegistry:
    """Registry for world model adapter classes with family categorization."""

    def __init__(self):
        self._registry: Dict[str, AdapterInfo] = {}

    def register(
        self,
        name: str,
        cls: Type,
        family: WorldModelFamily,
        description: str = "",
        supports_rl: bool = True,
        supports_video: bool = False,
        supports_planning: bool = False,
    ) -> None:
        """Register an adapter class.

        Args:
            name: Backend name (e.g., 'dreamerv3').
            cls: Adapter class.
            family: World model family for categorization.
            description: Human-readable description.
            supports_rl: Whether RL components (reward, value, done) are available.
            supports_video: Whether video prediction is supported.
            supports_planning: Whether planning/imagination is supported.
        """
        self._registry[name] = AdapterInfo(
            cls=cls,
            family=family,
            description=description,
            supports_rl=supports_rl,
            supports_video=supports_video,
            supports_planning=supports_planning,
        )

    def get(self, name: str) -> Type:
        """Get adapter class by name.

        Args:
            name: Backend name.

        Returns:
            Adapter class.

        Raises:
            KeyError: If name not registered.
        """
        if name not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(f"Backend '{name}' not found. Available: {available}")
        return self._registry[name].cls

    def get_info(self, name: str) -> AdapterInfo:
        """Get adapter info by name.

        Args:
            name: Backend name.

        Returns:
            AdapterInfo with metadata.

        Raises:
            KeyError: If name not registered.
        """
        if name not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(f"Backend '{name}' not found. Available: {available}")
        return self._registry[name]

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def keys(self):
        return self._registry.keys()

    def by_family(self, family: WorldModelFamily) -> List[str]:
        """Get all registered backends for a specific family.

        Args:
            family: World model family to filter by.

        Returns:
            List of backend names in that family.
        """
        return [name for name, info in self._registry.items() if info.family == family]

    def by_capability(
        self,
        supports_rl: Optional[bool] = None,
        supports_video: Optional[bool] = None,
        supports_planning: Optional[bool] = None,
    ) -> List[str]:
        """Get backends matching specific capabilities.

        Args:
            supports_rl: Filter by RL support (None = don't filter).
            supports_video: Filter by video support (None = don't filter).
            supports_planning: Filter by planning support (None = don't filter).

        Returns:
            List of backend names matching criteria.
        """
        results = []
        for name, info in self._registry.items():
            if supports_rl is not None and info.supports_rl != supports_rl:
                continue
            if supports_video is not None and info.supports_video != supports_video:
                continue
            if supports_planning is not None and info.supports_planning != supports_planning:
                continue
            results.append(name)
        return results

    def list_all(self) -> Dict[str, AdapterInfo]:
        """List all registered backends with their info.

        Returns:
            Dictionary of name -> AdapterInfo.
        """
        return dict(self._registry)


REGISTRY = BackendRegistry()


def register(
    name: str,
    family: WorldModelFamily,
    description: str = "",
    supports_rl: bool = True,
    supports_video: bool = False,
    supports_planning: bool = False,
):
    """Decorator to register a world model adapter.

    Usage:
        @register("dreamerv3", WorldModelFamily.DREAMER, "DreamerV3 implementation")
        class DreamerV3Adapter(BaseModelAdapter):
            ...
    """

    def decorator(cls: Type) -> Type:
        REGISTRY.register(
            name=name,
            cls=cls,
            family=family,
            description=description,
            supports_rl=supports_rl,
            supports_video=supports_video,
            supports_planning=supports_planning,
        )
        return cls

    return decorator
