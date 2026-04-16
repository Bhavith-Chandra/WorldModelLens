"""Configuration dataclass for world model architectures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from world_model_lens.core.types import (
    DynamicsType,
    LatentType,
    ModelPurpose,
    ObservationModality,
    WorldModelFamily,
)


@dataclass
class WorldModelConfig:
    """Configuration for world model architectures.

    Attributes:
        d_h: Hidden state dimension (recurrent/continuous latent).
        n_cat: Number of categorical variables in discrete latent (z).
        n_cls: Number of classes per categorical variable.
        d_action: Action space dimension.
        d_obs: Observation dimension (flattened, for vector observations).
        encoder_type: Type of encoder ('cnn', 'mlp', 'vit').
        backend: Backend architecture name.
        n_gru_layers: Number of GRU layers (for recurrent models).
        reward_head: Type of reward head ('twohot', 'gaussian', 'categorical').
        decoder_type: Type of decoder ('cnn', 'mlp').
        activation: Activation function ('silu', 'relu', 'elu').
        n_encoder_channels: Number of channels in first CNN layer.
        encoder_depth: Number of CNN blocks.
        continue_head: Type of continue/persistence prediction head.
        discount: Discount factor for value estimation.
        free_nats: Free nats for KL divergence.
        kl_scale: KL divergence scaling factor.
        actor_entropy_scale: Entropy regularization coefficient.
        imagination_horizon: Default horizon for imagination.
        seed: Random seed.
        latent_type: Type of latent representation (continuous, discrete, VQ, etc.).
        dynamics_type: Type of dynamics model (recurrent, transformer, JEPA, etc.).
        model_purpose: Primary purpose of the world model (RL, video prediction, etc.).
        world_model_family: Family/category of world model architecture.
        observation_modality: Type of observation input (pixels, state, multimodal, etc.).
        has_decoder: Whether the model has an observation decoder.
        has_reward_head: Whether the model predicts rewards.
        has_value_head: Whether the model predicts values.
        has_policy_head: Whether the model has a policy/actor head.
        has_done_head: Whether the model predicts episode termination.
        d_embed: Embedding dimension for transformer models.
        n_layers: Number of layers for transformer models.
        n_heads: Number of attention heads for transformer models.
        vocab_size: Vocabulary size for discrete/token models.
    """

    d_h: int = 512
    n_cat: int = 32
    n_cls: int = 32
    d_action: int = 0
    d_obs: int = 0
    encoder_type: Literal["cnn", "mlp", "vit"] = "cnn"
    backend: Literal[
        "dreamerv3",
        "dreamerv2",
        "iris",
        "tdmpc2",
        "custom",
        "dreamerv1",
        "planet",
        "ha_schmidhuber",
        "decision_transformer",
        "video_world_model",
        "autonomous_driving",
        "robotics",
        "contrastive_predictive",
        "ijepa",
    ] = "dreamerv3"
    n_gru_layers: int = 1
    reward_head: Literal["twohot", "gaussian", "categorical"] = "twohot"
    decoder_type: Literal["cnn", "mlp"] = "cnn"
    activation: Literal["silu", "relu", "elu"] = "silu"
    n_encoder_channels: int = 48
    encoder_depth: int = 4
    continue_head: Literal["logistic", "gaussian"] = "logistic"
    discount: float = 0.99
    free_nats: float = 0.0
    kl_scale: float = 1.0
    actor_entropy_scale: float = 1e-4
    imagination_horizon: int = 50
    seed: int | None = None

    latent_type: LatentType = LatentType.DISCRETE_CATEGORICAL
    dynamics_type: DynamicsType = DynamicsType.RECURRENT
    model_purpose: ModelPurpose = ModelPurpose.RL_AGENT
    world_model_family: WorldModelFamily = WorldModelFamily.DREAMER
    observation_modality: ObservationModality = ObservationModality.PIXEL

    has_decoder: bool = True
    has_reward_head: bool = True
    has_value_head: bool = True
    has_policy_head: bool = True
    has_done_head: bool = True

    d_embed: int = 256
    n_layers: int = 4
    n_heads: int = 4
    vocab_size: int = 512
    
    # I-JEPA specific
    patch_size: int = 16
    num_patches: int = 196
    embed_dim: int = 192
    context_mask_ratio: float = 0.85
    target_mask_scale: float = 0.15
    predictor_embed_dim: int = 384
    predictor_depth: int = 4
    predictor_heads: int = 6

    @property
    def d_z(self) -> int:
        """Total dimension of discrete latent space (n_cat * n_cls)."""
        return self.n_cat * self.n_cls

    @property
    def d_latent(self) -> int:
        """Combined latent dimension (h + z_flat)."""
        return self.d_h + self.d_z

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.d_h <= 0:
            raise ValueError(f"d_h must be positive, got {self.d_h}")
        if self.n_cat <= 0 or self.n_cls <= 0:
            raise ValueError(f"n_cat and n_cls must be positive, got {self.n_cat}, {self.n_cls}")
