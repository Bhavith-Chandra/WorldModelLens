"""Type definitions for world model architectures."""

from enum import Enum, auto
from typing import Optional


class LatentType(Enum):
    """Type of latent representation used by the world model.

    This enum categorizes the different latent space configurations
    that world models can use.
    """

    CONTINUOUS = auto()
    DISCRETE_CATEGORICAL = auto()
    DISCRETE_VQ = auto()
    HYBRID_CONTINUOUS_DISCRETE = auto()
    TRANSFORMER_TOKEN = auto()
    MIXED = auto()


class DynamicsType(Enum):
    """Type of dynamics/transition model used by the world model.

    This categorizes how the world model predicts state transitions.
    """

    RECURRENT = auto()
    TRANSFORMER = auto()
    CONVOLUTIONAL = auto()
    MDN_RNN = auto()
    JEPA = auto()
    NONE = auto()


class ModelPurpose(Enum):
    """Primary purpose/use case of the world model.

    This helps categorize world models by their intended application.
    """

    RL_AGENT = auto()
    VIDEO_PREDICTION = auto()
    PLANNING = auto()
    CONTROL = auto()
    DECISION_MAKING = auto()
    MULTIMODAL = auto()


class WorldModelFamily(Enum):
    """Family/category of world model architecture.

    This enum categorizes world models by their architectural family.
    Used for registry and adapter lookup.
    """

    DREAMER = auto()
    PLA_NET = auto()
    TD_MPC = auto()
    HA_SCHMIDHUBER = auto()
    IRIS = auto()
    DECISION_TRANSFORMER = auto()
    VIDEO_WORLD_MODEL = auto()
    AUTONOMOUS_DRIVING = auto()
    ROBOTICS = auto()
    CONTRASTIVE_PREDICTIVE = auto()
    JEPA = auto()
    CUSTOM = auto()


class ObservationModality(Enum):
    """Type of observation/input modality.

    Categorizes the types of inputs the world model processes.
    """

    PIXEL = auto()
    STATE_VECTOR = auto()
    MULTIMODAL = auto()
    VOXEL = auto()
    POINT_CLOUD = auto()
    SENSOR_FUSION = auto()
