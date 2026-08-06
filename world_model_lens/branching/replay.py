"""Intervention Replay System for world model debugging.

This module provides tools for:
- Replaying trajectories with interventions
- Time-travel debugging (step back/forward)
- Breakpoint debugging
- Recording and replaying execution traces
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from enum import Enum
import torch
import numpy as np


class BreakpointType(Enum):
    """Types of breakpoints for debugging."""

    STATE = "state"
    ACTION = "action"
    REWARD = "reward"
    LATENT = "latent"
    CUSTOM = "custom"


@dataclass
class Breakpoint:
    """A breakpoint in trajectory execution."""

    timestep: int
    breakpoint_type: BreakpointType
    condition: Callable[[Any], bool] | None = None
    action: Callable[[Any], Any] | None = None
    enabled: bool = True

    def should_trigger(self, context: dict[str, Any]) -> bool:
        """Check if breakpoint should trigger."""
        if not self.enabled:
            return False

        if self.condition is None:
            return True

        return self.condition(context)


@dataclass
class Intervention:
    """An intervention to apply during replay."""

    timestep: int
    target: str
    value: Any
    interpolation: str = "hard"

    def apply(self, current_value: torch.Tensor) -> torch.Tensor:
        """Apply intervention to current value."""
        if self.interpolation == "hard":
            return self.value if isinstance(self.value, torch.Tensor) else torch.tensor(self.value)
        elif self.interpolation == "linear":
            t = 0.5
            return (1 - t) * current_value + t * self.value
        return current_value


@dataclass
class ReplayState:
    """Current state of replay session."""

    current_timestep: int
    trajectory: Any
    checkpoints: dict[int, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    breakpoints: list[Breakpoint] = field(default_factory=list)
    interventions: list[Intervention] = field(default_factory=list)


@dataclass
class ReplayResult:
    """Result of a replay operation."""

    trajectory: Any
    checkpoint_timestep: int
    replay_type: str
    modifications: dict[str, Any] = field(default_factory=dict)
    execution_trace: list[dict[str, Any]] = field(default_factory=list)


class InterventionReplaySystem:
    """System for replaying trajectories with interventions and debugging.

    This provides a debugger-like experience for world model trajectories:
    - Set breakpoints at specific timesteps
    - Apply interventions during replay
    - Step forward/backward through execution
    - Record full execution traces
    """

    def __init__(self, wm: Any):
        """Initialize replay system.

        Args:
            wm: HookedWorldModel instance
        """
        self.wm = wm
        self._state: Optional[ReplayState] = None

    def load_trajectory(self, trajectory: Any) -> ReplayState:
        """Load a trajectory for replay.

        Args:
            trajectory: WorldTrajectory to load

        Returns:
            ReplayState for the loaded trajectory
        """
        self._state = ReplayState(
            current_timestep=0,
            trajectory=trajectory,
            checkpoints={0: self._create_checkpoint(trajectory, 0)},
        )
        return self._state

    def add_breakpoint(
        self,
        timestep: int,
        breakpoint_type: BreakpointType = BreakpointType.CUSTOM,
        condition: Callable[[dict], bool] | None = None,
        action: Callable[[dict], Any] | None = None,
    ) -> Breakpoint:
        """Add a breakpoint to the replay session.

        Args:
            timestep: Timestep to break at
            breakpoint_type: Type of breakpoint
            condition: Optional condition for breakpoint
            action: Optional action to take at breakpoint

        Returns:
            Created Breakpoint
        """
        if self._state is None:
            raise RuntimeError("No trajectory loaded")

        bp = Breakpoint(
            timestep=timestep,
            breakpoint_type=breakpoint_type,
            condition=condition,
            action=action,
        )
        self._state.breakpoints.append(bp)
        return bp

    def remove_breakpoint(self, breakpoint: Breakpoint) -> None:
        """Remove a breakpoint."""
        if self._state:
            self._state.breakpoints.remove(breakpoint)

    def add_intervention(
        self,
        timestep: int,
        target: str,
        value: Any,
        interpolation: str = "hard",
    ) -> Intervention:
        """Add an intervention to apply during replay.

        Args:
            timestep: Timestep to apply intervention
            target: Target component (e.g., "state", "action", "h")
            value: Value to inject
            interpolation: How to interpolate ("hard", "linear")

        Returns:
            Created Intervention
        """
        if self._state is None:
            raise RuntimeError("No trajectory loaded")

        intervention = Intervention(
            timestep=timestep,
            target=target,
            value=value,
            interpolation=interpolation,
        )
        self._state.interventions.append(intervention)
        return intervention

    def step_forward(self, num_steps: int = 1) -> Any:
        """Step forward in the trajectory.

        Args:
            num_steps: Number of steps to advance

        Returns:
            Updated trajectory
        """
        if self._state is None:
            raise RuntimeError("No trajectory loaded")

        start = self._state.current_timestep
        end = min(start + num_steps, len(self._state.trajectory.states) - 1)

        for t in range(start, end):
            self._state.history.append(
                {
                    "timestep": t,
                    "action": "step_forward",
                }
            )

        self._state.current_timestep = end

        if end not in self._state.checkpoints:
            self._state.checkpoints[end] = self._create_checkpoint(self._state.trajectory, end)

        return self._build_partial_trajectory(end)

    def step_backward(self, num_steps: int = 1) -> Any:
        """Step backward in the trajectory.

        Args:
            num_steps: Number of steps to go back

        Returns:
            Restored trajectory
        """
        if self._state is None:
            raise RuntimeError("No trajectory loaded")

        target = max(0, self._state.current_timestep - num_steps)

        if target in self._state.checkpoints:
            checkpoint = self._state.checkpoints[target]
            self._restore_checkpoint(checkpoint)

        self._state.current_timestep = target

        return self._build_partial_trajectory(target)

    def goto(self, timestep: int) -> Any:
        """Go to a specific timestep.

        Args:
            timestep: Target timestep

        Returns:
            Trajectory at target timestep
        """
        if self._state is None:
            raise RuntimeError("No trajectory loaded")

        target = max(0, min(timestep, len(self._state.trajectory.states) - 1))

        if target > self._state.current_timestep:
            self.step_forward(target - self._state.current_timestep)
        elif target < self._state.current_timestep:
            self.step_backward(self._state.current_timestep - target)

        return self._build_partial_trajectory(target)

    def run_with_interventions(
        self,
        stop_at_breakpoint: bool = True,
    ) -> ReplayResult:
        """Run full trajectory with all interventions and breakpoints.

        Args:
            stop_at_breakpoint: Whether to stop when breakpoint is hit

        Returns:
            ReplayResult with modified trajectory and trace
        """
        if self._state is None:
            raise RuntimeError("No trajectory loaded")

        execution_trace = []
        modifications = {}

        original_trajectory = self._state.trajectory
        states = list(original_trajectory.states)

        for t in range(len(states)):
            context = {
                "timestep": t,
                "state": states[t].state,
                "action": states[t].action,
                "reward": states[t].reward,
            }

            for bp in self._state.breakpoints:
                if bp.timestep == t and bp.should_trigger(context):
                    execution_trace.append(
                        {
                            "timestep": t,
                            "event": "breakpoint",
                            "type": bp.breakpoint_type.value,
                        }
                    )

                    if bp.action:
                        modified = bp.action(context)
                        modifications[f"breakpoint_{t}"] = modified

                    if stop_at_breakpoint:
                        return ReplayResult(
                            trajectory=self._build_partial_trajectory(t),
                            checkpoint_timestep=t,
                            replay_type="breakpoint",
                            modifications=modifications,
                            execution_trace=execution_trace,
                        )

            for intervention in self._state.interventions:
                if intervention.timestep == t:
                    if intervention.target == "state":
                        old_state = states[t].state
                        new_state = intervention.apply(old_state)
                        states[t] = self._update_state(states[t], new_state)
                        modifications[f"intervention_{t}_state"] = {
                            "old": old_state.cpu().tolist(),
                            "new": new_state.cpu().tolist(),
                        }
                    elif intervention.target == "action":
                        old_action = states[t].action
                        new_action = intervention.apply(old_action)
                        states[t] = self._update_action(states[t], new_action)
                        modifications[f"intervention_{t}_action"] = {
                            "old": old_action.cpu().tolist() if old_action is not None else None,
                            "new": new_action.cpu().tolist() if new_action is not None else None,
                        }

            execution_trace.append(
                {
                    "timestep": t,
                    "event": "step",
                }
            )

        modified_trajectory = self._rebuild_trajectory(states)

        return ReplayResult(
            trajectory=modified_trajectory,
            checkpoint_timestep=len(states) - 1,
            replay_type="full",
            modifications=modifications,
            execution_trace=execution_trace,
        )

    def get_execution_summary(self) -> dict[str, Any]:
        """Get summary of execution.

        Returns:
            Dictionary with execution statistics
        """
        if self._state is None:
            return {}

        return {
            "total_timesteps": len(self._state.trajectory.states),
            "current_timestep": self._state.current_timestep,
            "num_breakpoints": len(self._state.breakpoints),
            "num_interventions": len(self._state.interventions),
            "num_checkpoints": len(self._state.checkpoints),
            "history_length": len(self._state.history),
        }

    def _create_checkpoint(self, trajectory: Any, timestep: int) -> dict[str, Any]:
        """Create a checkpoint at a timestep."""
        return {
            "timestep": timestep,
            "states": [s.state.clone() for s in trajectory.states],
            "actions": [
                s.action.clone() if s.action is not None else None for s in trajectory.states
            ],
            "rewards": [
                s.reward.clone() if s.reward is not None else None for s in trajectory.states
            ],
        }

    def _restore_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Restore from a checkpoint."""
        if self._state is None:
            return

        states = self._state.trajectory.states
        for i, state in enumerate(states):
            if i < len(checkpoint["states"]):
                state.state.copy_(checkpoint["states"][i])

    def _build_partial_trajectory(self, end_timestep: int) -> Any:
        """Build trajectory up to a timestep."""
        from world_model_lens import WorldTrajectory, WorldState

        if self._state is None:
            raise RuntimeError("No trajectory loaded")

        partial_states = self._state.trajectory.states[: end_timestep + 1]
        return WorldTrajectory(states=list(partial_states), source="replay")

    def _rebuild_trajectory(self, states: list[Any]) -> Any:
        """Rebuild trajectory from state list."""
        from world_model_lens import WorldTrajectory

        return WorldTrajectory(states=states, source="replay")

    def _update_state(self, state: Any, new_state: torch.Tensor) -> Any:
        """Update state with new value."""
        from world_model_lens import WorldState

        return WorldState(
            state=new_state,
            timestep=state.timestep,
            action=state.action,
            reward=state.reward,
            done=state.done,
            metadata=state.metadata.copy() if state.metadata else {},
        )

    def _update_action(self, state: Any, new_action: torch.Tensor) -> Any:
        """Update action with new value."""
        from world_model_lens import WorldState

        return WorldState(
            state=state.state,
            timestep=state.timestep,
            action=new_action,
            reward=state.reward,
            done=state.done,
            metadata=state.metadata.copy() if state.metadata else {},
        )


class TimeTravelDebugger:
    """Time-travel debugging for world model trajectories.

    Allows stepping through execution, setting watch points,
    and examining state at any point in execution.
    """

    def __init__(self, wm: Any):
        """Initialize time-travel debugger.

        Args:
            wm: HookedWorldModel instance
        """
        self.wm = wm
        self.replay = InterventionReplaySystem(wm)
        self._watch_points: list[dict[str, Any]] = []

    def watch(
        self,
        expression: str,
        condition: Callable[[Any], bool] | None = None,
    ) -> None:
        """Add a watch point.

        Args:
            expression: Expression to watch (e.g., "state[0].mean()")
            condition: Optional condition for watch
        """
        self._watch_points.append(
            {
                "expression": expression,
                "condition": condition,
                "values": [],
            }
        )

    def watch_state(self, key: str, value: Any) -> None:
        """Watch a specific state value.

        Args:
            key: Key to watch
            value: Current value
        """
        for watch in self._watch_points:
            if watch["expression"] == key:
                watch["values"].append(value)
                break

    def get_watch_values(self) -> dict[str, list[Any]]:
        """Get all watch values.

        Returns:
            Dictionary of watch expressions to values
        """
        return {w["expression"]: w["values"] for w in self._watch_points}

    def clear_watches(self) -> None:
        """Clear all watch points."""
        for watch in self._watch_points:
            watch["values"] = []
