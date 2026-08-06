"""Migration compatibility probe for migrated backend adapters.

This example is intentionally simple and diagnostic (and temporary). It exercises the adapters
through the base adapter contract and then tries a small HookedWorldModel
integration pass, similar in spirit to examples/01_quickstart.py.

It is meant to help validate adapter migration work, not to benchmark or train.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from world_model_lens import HookedWorldModel
from world_model_lens.backends.base_adapter import AdapterConfig
from world_model_lens.backends.decision_transformer import DecisionTransformerAdapter
from world_model_lens.backends.dreamerv1 import DreamerV1Adapter
from world_model_lens.backends.ha_schmidhuber import HaSchmidhuberWorldModelAdapter
from world_model_lens.backends.planet import PlaNetAdapter
from world_model_lens.backends.planning_adapter import PlanningAdapter
from world_model_lens.backends.tdmpc2 import TDMPC2Adapter
from world_model_lens.backends.video_adapter import VideoWorldModelAdapter as SimpleVideoAdapter
from world_model_lens.backends.video_world_model import VideoWorldModelAdapter
from world_model_lens.backends.autonomous_driving import AutonomousDrivingAdapter
from world_model_lens.backends.robotics import RoboticsAdapter
from world_model_lens.backends.contrastive_predictive import ContrastiveAdapter


@dataclass
class AdapterCase:
    name: str
    builder: Callable[
        [], tuple[torch.nn.Module, AdapterConfig, torch.Tensor, Optional[torch.Tensor]]
    ]


def make_config(**overrides) -> AdapterConfig:
    config = AdapterConfig(
        d_h=32,
        d_z=32,
        d_state=32,
        d_action=4,
        d_obs=64,
        d_embed=32,
        n_layers=2,
        n_heads=4,
        imagination_horizon=8,
        has_decoder=False,
        has_reward_head=False,
        has_value_head=False,
        has_policy_head=False,
        has_done_head=False,
        is_discrete=False,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def build_planning() -> tuple[PlanningAdapter, AdapterConfig, torch.Tensor, torch.Tensor]:
    config = make_config(d_obs=64, d_state=32, d_h=32, d_z=32, d_action=4, has_policy_head=True)
    adapter = PlanningAdapter(config)
    obs_seq = torch.randn(4, config.d_obs)
    action_seq = torch.randn(4, config.d_action)
    return adapter, config, obs_seq, action_seq


def build_decision_transformer() -> tuple[
    DecisionTransformerAdapter, AdapterConfig, torch.Tensor, torch.Tensor
]:
    config = make_config(d_obs=64, d_embed=32, d_action=4, has_policy_head=True)
    adapter = DecisionTransformerAdapter(config)
    obs_seq = torch.randn(4, config.d_obs)
    action_seq = torch.randn(4, config.d_action)
    return adapter, config, obs_seq, action_seq


def build_dreamerv1() -> tuple[DreamerV1Adapter, AdapterConfig, torch.Tensor, torch.Tensor]:
    config = make_config(
        d_h=32,
        d_z=32,
        d_obs=128,
        d_action=4,
        has_decoder=True,
        has_reward_head=True,
        has_value_head=True,
        has_policy_head=True,
    )
    adapter = DreamerV1Adapter(config)
    obs_seq = torch.randn(4, 3, 96, 96)
    action_seq = torch.randn(4, config.d_action)
    return adapter, config, obs_seq, action_seq


def build_planet() -> tuple[PlaNetAdapter, AdapterConfig, torch.Tensor, torch.Tensor]:
    config = make_config(
        d_h=32,
        d_z=32,
        d_obs=128,
        d_action=4,
        has_reward_head=True,
    )
    adapter = PlaNetAdapter(config)
    obs_seq = torch.randn(4, 3, 96, 96)
    action_seq = torch.randn(4, config.d_action)
    return adapter, config, obs_seq, action_seq


def build_ha_schmidhuber() -> tuple[
    HaSchmidhuberWorldModelAdapter, AdapterConfig, torch.Tensor, torch.Tensor
]:
    config = make_config(
        d_h=32,
        d_z=16,
        d_obs=64,
        d_action=3,
        has_decoder=True,
        has_policy_head=True,
    )
    adapter = HaSchmidhuberWorldModelAdapter(config)
    obs_seq = torch.randn(4, 3, 96, 96)
    action_seq = torch.randn(4, config.d_action)
    return adapter, config, obs_seq, action_seq


def build_tdmpc2() -> tuple[TDMPC2Adapter, AdapterConfig, torch.Tensor, torch.Tensor]:
    config = make_config(
        d_h=32,
        d_z=32,
        d_obs=64,
        d_action=4,
        has_reward_head=True,
        has_value_head=True,
        has_policy_head=True,
        has_done_head=True,
    )
    adapter = TDMPC2Adapter(config)
    obs_seq = torch.randn(4, config.d_obs)
    action_seq = torch.randn(4, config.d_action)
    return adapter, config, obs_seq, action_seq


def build_video_adapter() -> tuple[SimpleVideoAdapter, AdapterConfig, torch.Tensor, None]:
    frame_shape = (3, 64, 64)
    config = make_config(d_state=32, d_h=32, d_z=32, d_obs=3 * 64 * 64, has_decoder=True)
    adapter = SimpleVideoAdapter(config=config, d_obs=config.d_obs, frame_shape=frame_shape)
    obs_seq = torch.randn(4, *frame_shape)
    return adapter, config, obs_seq, None


def build_video_world_model() -> tuple[VideoWorldModelAdapter, AdapterConfig, torch.Tensor, None]:
    config = make_config(d_embed=32, n_layers=2, n_heads=4, has_decoder=True)
    adapter = VideoWorldModelAdapter(config)
    obs_seq = torch.randn(4, 3, 64, 64)
    return adapter, config, obs_seq, None


def build_autonomous_driving() -> tuple[AutonomousDrivingAdapter, AdapterConfig, torch.Tensor, torch.Tensor]:
    config = make_config(d_h=32, d_z=32, d_obs=64, d_action=3, has_reward_head=True)
    adapter = AutonomousDrivingAdapter(config)
    obs_seq = torch.randn(4, 3, 64, 64)
    action_seq = torch.randn(4, config.d_action)
    return adapter, config, obs_seq, action_seq


def build_robotics() -> tuple[RoboticsAdapter, AdapterConfig, torch.Tensor, torch.Tensor]:
    config = make_config(d_h=32, d_z=32, d_obs=64, d_action=4, has_decoder=True, has_reward_head=True)
    adapter = RoboticsAdapter(config)
    obs_seq = torch.randn(4, 3, 64, 64)
    action_seq = torch.randn(4, config.d_action)
    return adapter, config, obs_seq, action_seq


def build_contrastive() -> tuple[ContrastiveAdapter, AdapterConfig, torch.Tensor, torch.Tensor]:
    config = make_config(d_h=32, d_z=32, d_obs=64, d_action=4)
    adapter = ContrastiveAdapter(config)
    obs_seq = torch.randn(4, config.d_obs)
    action_seq = torch.randn(4, config.d_action)
    return adapter, config, obs_seq, action_seq


CASES = [
    AdapterCase("planning", build_planning),
    AdapterCase("decision_transformer", build_decision_transformer),
    AdapterCase("dreamerv1", build_dreamerv1),
    AdapterCase("planet", build_planet),
    AdapterCase("ha_schmidhuber", build_ha_schmidhuber),
    AdapterCase("tdmpc2", build_tdmpc2),
    AdapterCase("video_adapter", build_video_adapter),
    AdapterCase("video_world_model", build_video_world_model),
    AdapterCase("autonomous_driving", build_autonomous_driving),
    AdapterCase("robotics", build_robotics),
    AdapterCase("contrastive_predictive", build_contrastive),
]


def shape_of(value: object) -> str:
    if value is None:
        return "None"
    if isinstance(value, torch.Tensor):
        return str(tuple(value.shape))
    if isinstance(value, tuple):
        return "(" + ", ".join(shape_of(v) for v in value) + ")"
    return type(value).__name__


def try_call(label: str, fn: Callable[[], object]) -> tuple[bool, str]:
    try:
        value = fn()
        return True, shape_of(value)
    except Exception as exc:  # diagnostic script
        return False, f"{type(exc).__name__}: {exc}"


def probe_case(case: AdapterCase) -> None:
    print("=" * 72)
    print(case.name)
    print("=" * 72)

    adapter, config, obs_seq, action_seq = case.builder()
    wm = HookedWorldModel(adapter=adapter, config=config, name=f"migration_{case.name}")

    print(f"obs_seq: {tuple(obs_seq.shape)}")
    print(f"action_seq: {None if action_seq is None else tuple(action_seq.shape)}")

    ok, msg = try_call("initial_state", lambda: adapter.initial_state(batch_size=1))
    print(f"initial_state: {'PASS' if ok else 'FAIL'} -> {msg}")
    if not ok:
        return

    h0, z0 = adapter.initial_state(batch_size=1)
    obs0 = obs_seq[0]
    action0 = action_seq[0] if action_seq is not None else None

    ok, msg = try_call("encode", lambda: adapter.encode(obs0, h0.squeeze(0)))
    print(f"encode:        {'PASS' if ok else 'FAIL'} -> {msg}")
    if not ok:
        return

    z_post, aux = adapter.encode(obs0, h0.squeeze(0))
    z_for_step = z_post if z_post.dim() > 1 else z_post.unsqueeze(0)

    ok, msg = try_call(
        "transition",
        lambda: adapter.transition(
            h0, z_for_step, action0.unsqueeze(0) if action0 is not None else None
        ),
    )
    print(f"transition:    {'PASS' if ok else 'FAIL'} -> {msg}")

    next_h = None
    if ok:
        next_h = adapter.transition(
            h0, z_for_step, action0.unsqueeze(0) if action0 is not None else None
        )

    if next_h is None:
        next_h = h0

    direct_checks = [
        ("decode", lambda: adapter.decode(next_h, z_for_step)),
        ("predict_reward", lambda: adapter.predict_reward(next_h, z_for_step)),
        ("predict_continue", lambda: adapter.predict_continue(next_h, z_for_step)),
        ("actor_forward", lambda: adapter.actor_forward(next_h, z_for_step)),
        ("critic_forward", lambda: adapter.critic_forward(next_h, z_for_step)),
    ]

    for label, fn in direct_checks:
        ok, msg = try_call(label, fn)
        print(f"{label:<13} {'PASS' if ok else 'FAIL'} -> {msg}")

    ok, msg = try_call("run_with_cache", lambda: wm.run_with_cache(obs_seq, action_seq))
    print(f"run_with_cache: {'PASS' if ok else 'FAIL'} -> {msg}")


def main() -> None:
    torch.manual_seed(0)
    print("Migration Adapter Compatibility Probe")
    print("This example runs simple contract and HookedWorldModel checks.")
    print()

    for case in CASES:
        probe_case(case)
        print()


if __name__ == "__main__":
    main()
