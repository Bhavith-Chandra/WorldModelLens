"""Generic world trajectory representation.

Supports both RL trajectories (with rewards, actions, done flags)
and non-RL trajectories (video prediction, planning, etc.).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from world_model_lens.core.world_state import WorldState
import torch


@dataclass
class WorldTrajectory:
    """A sequence of world states forming a trajectory.

    This is the generic trajectory representation. It works with:
    - RL trajectories (with rewards, actions, done flags)
    - Video prediction trajectories (with frame sequences)
    - Planning trajectories (with action sequences)
    - Any future trajectory type

    RL-specific fields are OPTIONAL. Non-RL world models can
    ignore these or use the metadata field for custom data.

    Attributes:
        states: Ordered list of WorldState objects
        name: Optional name for this trajectory
        source: Where this trajectory came from ('real', 'imagined', 'planned')
        episode_id: Optional episode identifier
        fork_point: Optional timestep where trajectory was forked
        start_time: Optional start timestamp
        end_time: Optional end timestamp
        metadata: Arbitrary additional data
    """

    states: List["WorldState"]
    name: str = "trajectory"
    source: str = "real"
    episode_id: Optional[int] = None
    fork_point: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        """Number of timesteps in the trajectory."""
        return len(self.states)

    @property
    def T(self) -> int:
        """Alias for length."""
        return len(self.states)

    @property
    def is_imagined(self) -> bool:
        """Check if this is an imagined trajectory."""
        return self.source == "imagined"

    @property
    def is_real(self) -> bool:
        """Check if this is a real/observed trajectory."""
        return self.source == "real"

    @property
    def is_terminal(self) -> bool:
        """Check if trajectory ends at terminal state."""
        if self.states and self.states[-1].done is not None:
            return bool(self.states[-1].done.item())
        return False

    @property
    def state_sequence(self) -> torch.Tensor:
        """Stacked state tensors [T, d_state]."""
        return torch.stack([s.state for s in self.states], dim=0)

    @property
    def action_sequence(self) -> Optional[torch.Tensor]:
        """Stacked actions [T, d_action] if available."""
        actions = [s.action for s in self.states if s.action is not None]
        if not actions:
            return None
        return torch.stack(actions, dim=0)

    @property
    def reward_sequence(self) -> Optional[torch.Tensor]:
        """Stacked rewards [T] or [T, ...] if available."""
        rewards = []
        for s in self.states:
            r = s.reward if s.reward is not None else s.reward_pred
            if r is not None:
                rewards.append(r)
        if not rewards:
            return None
        return torch.stack(rewards, dim=0)

    @property
    def value_sequence(self) -> Optional[torch.Tensor]:
        """Stacked value estimates [T] if available."""
        values = []
        for s in self.states:
            v = s.value if s.value is not None else s.value_pred
            if v is not None:
                values.append(v)
        if not values:
            return None
        return torch.stack(values, dim=0)

    @property
    def done_sequence(self) -> Optional[torch.Tensor]:
        """Stacked done flags [T] if available."""
        dones = [s.done for s in self.states if s.done is not None]
        if not dones:
            return None
        return torch.stack(dones, dim=0)

    @property
    def total_reward(self) -> Optional[torch.Tensor]:
        """Sum of rewards if available."""
        rewards = self.reward_sequence
        if rewards is None:
            return None
        return rewards.sum()

    @property
    def mean_reward(self) -> Optional[torch.Tensor]:
        """Mean reward if available."""
        rewards = self.reward_sequence
        if rewards is None:
            return None
        return rewards.mean()

    @property
    def timesteps(self) -> List[int]:
        """List of timesteps."""
        return [s.timestep for s in self.states]

    def surprise_peaks(
        self,
        threshold: float,
        metric: str = "kl",
    ) -> List[tuple]:
        """Find timesteps where surprise exceeds threshold.

        Args:
            threshold: Surprise threshold
            metric: 'kl', 'loss', or 'custom'

        Returns:
            List of (timestep, value) tuples above threshold
        """
        peaks = []
        for s in self.states:
            if s.metadata.get("surprise") is not None:
                val = s.metadata["surprise"]
            elif s.metadata.get(f"{metric}_divergence") is not None:
                val = s.metadata[f"{metric}_divergence"]
            else:
                continue
            if val > threshold:
                peaks.append((s.timestep, val))
        return peaks

    def slice(self, start: int, end: int) -> "WorldTrajectory":
        """Slice trajectory between start and end indices."""
        return WorldTrajectory(
            states=self.states[start:end],
            name=self.name,
            source=self.source,
            episode_id=self.episode_id,
            fork_point=self.fork_point,
            metadata=self.metadata.copy(),
        )

    def slice_timesteps(self, start: int, end: int) -> "WorldTrajectory":
        """Slice trajectory by timestep values (not indices)."""
        sliced = [s for s in self.states if start <= s.timestep < end]
        return WorldTrajectory(
            states=sliced,
            name=self.name,
            source=self.source,
            episode_id=self.episode_id,
            fork_point=self.fork_point,
            metadata=self.metadata.copy(),
        )

    def fork_at(self, timestep: int) -> "WorldTrajectory":
        """Create trajectory starting from given timestep."""
        for i, s in enumerate(self.states):
            if s.timestep >= timestep:
                return WorldTrajectory(
                    states=self.states[i:],
                    name=f"{self.name}_fork",
                    source=self.source,
                    episode_id=self.episode_id,
                    fork_point=timestep,
                    metadata=self.metadata.copy(),
                )
        return WorldTrajectory(
            states=[],
            name=f"{self.name}_fork",
            source=self.source,
            episode_id=self.episode_id,
            fork_point=timestep,
            metadata=self.metadata.copy(),
        )

    def to_device(self, device: torch.device) -> "WorldTrajectory":
        """Move all states to device."""
        return WorldTrajectory(
            states=[s.to_device(device) for s in self.states],
            name=self.name,
            source=self.source,
            episode_id=self.episode_id,
            fork_point=self.fork_point,
            metadata=self.metadata.copy(),
        )

    def filter_states(self, predicate: Callable[["WorldState"], bool]) -> "WorldTrajectory":
        """Filter states by predicate."""
        return WorldTrajectory(
            states=[s for s in self.states if predicate(s)],
            name=self.name,
            source=self.source,
            episode_id=self.episode_id,
            fork_point=self.fork_point,
            metadata=self.metadata.copy(),
        )

    def map_states(self, fn: Callable[["WorldState"], "WorldState"]) -> "WorldTrajectory":
        """Apply function to all states."""
        return WorldTrajectory(
            states=[fn(s) for s in self.states],
            name=self.name,
            source=self.source,
            episode_id=self.episode_id,
            fork_point=self.fork_point,
            metadata=self.metadata.copy(),
        )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> "WorldState":
        return self.states[index]

    def __iter__(self):
        return iter(self.states)


@dataclass
class TrajectoryStatistics:
    """Statistics computed from a trajectory or dataset."""

    n_states: int
    n_terminal: int
    mean_reward: Optional[float] = None
    std_reward: Optional[float] = None
    total_reward: Optional[float] = None
    mean_value: Optional[float] = None
    mean_surprise: Optional[float] = None
    max_surprise: Optional[float] = None
    surprise_peaks: Optional[List[tuple]] = None
    unique_episodes: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
