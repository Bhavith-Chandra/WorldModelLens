"""World Model Lens: Backend-agnostic interpretability for ANY world model.

Supports a broad family of world model architectures:

Latent dynamics RL world models:
- Ha & Schmidhuber "World Models" (VAE + MDN-RNN + controller)
- PlaNet (latent RSSM from pixels)
- Dreamer family: V1, V2, V3
- TD-MPC / TD-MPC2 (JEPA-style latent models)
- Contrastive/Predictive latent models (CWM, SPR-style)

Transformer / token-based world models:
- IRIS-style transformer world models over VQVAE tokens
- Decision/Trajectory Transformers

Video and autonomous-driving world models:
- Latent video world models (WorldDreamer-style)
- Autonomous driving world models (camera/LiDAR/radar fusion)

Robotics and embodied world models:
- Embodied/robotic manipulation world models (RSSM, JEPA, etc.)

RL-specific components (reward, value, action, done) are OPTIONAL.
Non-RL world models are first-class citizens.
"""

__version__ = "0.2.0"

from world_model_lens.backends.autonomous_driving import AutonomousDrivingAdapter
from world_model_lens.backends.contrastive_predictive import ContrastiveAdapter
from world_model_lens.backends.decision_transformer import DecisionTransformerAdapter
from world_model_lens.backends.dreamerv1 import DreamerV1Adapter
from world_model_lens.backends.dreamerv2 import DreamerV2Adapter
from world_model_lens.backends.dreamerv3 import DreamerV3Adapter
from world_model_lens.backends.base_adapter import BaseModelAdapter
from world_model_lens.backends.ha_schmidhuber import HaSchmidhuberWorldModelAdapter
from world_model_lens.backends.iris import IRISAdapter
from world_model_lens.backends.planet import PlaNetAdapter
from world_model_lens.backends.registry import REGISTRY, BackendRegistry, register
from world_model_lens.backends.robotics import RoboticsAdapter
from world_model_lens.backends.tdmpc2 import TDMPC2Adapter
from world_model_lens.backends.video_world_model import VideoWorldModelAdapter
from world_model_lens.core import (
    ActivationCache,
    HookContext,
    HookPoint,
    HookRegistry,
    LatentState,
    LatentTrajectory,
    ObservationType,
    TrajectoryStatistics,
    WorldDynamics,
    WorldModelConfig,
    WorldModelOutput,
    WorldState,
    WorldTrajectory,
)
from world_model_lens.core.types import (
    DynamicsType,
    LatentType,
    ModelPurpose,
    ObservationModality,
    WorldModelFamily,
)
from world_model_lens.hooked_world_model import HookedWorldModel
from world_model_lens.probing import LatentProber
from world_model_lens.envs import (
    EnvironmentAdapter,
    EnvironmentFactory,
    FACTORY,
    create,
    GymnasiumAdapter,
    GymnasiumCapabilities,
    EnvSpec,
    EnvStepResult,
)

__all__ = [
    "__version__",
    "WorldState",
    "WorldTrajectory",
    "WorldDynamics",
    "WorldModelOutput",
    "WorldModelConfig",
    "TrajectoryStatistics",
    "LatentState",
    "LatentTrajectory",
    "LatentProber",
    "ObservationType",
    "HookPoint",
    "HookContext",
    "HookRegistry",
    "ActivationCache",
    "LatentType",
    "DynamicsType",
    "ModelPurpose",
    "WorldModelFamily",
    "ObservationModality",
    "BaseModelAdapter",
    "BackendRegistry",
    "REGISTRY",
    "register",
    "HookedWorldModel",
    "DreamerV3Adapter",
    "DreamerV2Adapter",
    "DreamerV1Adapter",
    "PlaNetAdapter",
    "HaSchmidhuberWorldModelAdapter",
    "IRISAdapter",
    "TDMPC2Adapter",
    "DecisionTransformerAdapter",
    "ContrastiveAdapter",
    "VideoWorldModelAdapter",
    "AutonomousDrivingAdapter",
    "RoboticsAdapter",
    # Environment adapters
    "EnvironmentAdapter",
    "EnvironmentFactory",
    "FACTORY",
    "create",
    "GymnasiumAdapter",
    "GymnasiumCapabilities",
    "EnvSpec",
    "EnvStepResult",
]
