"""Shared utilities: logging, device selection, tensor operations."""

from world_model_lens.utils.device import get_device
from world_model_lens.utils.logging import get_logger
from world_model_lens.utils.tensor_ops import (
    flatten_latent,
    safe_kl_divergence,
    stack_trajectories,
)

__all__ = [
    "get_logger",
    "get_device",
    "flatten_latent",
    "stack_trajectories",
    "safe_kl_divergence",
]
