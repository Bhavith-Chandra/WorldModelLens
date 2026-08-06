"""Tensor manipulation utilities."""

from typing import List, Literal, Union
import torch


def flatten_latent(
    z: torch.Tensor,
    n_cat: int,
    n_cls: int,
    mode: Literal["onehot", "indices", "embed"] = "onehot",
) -> torch.Tensor:
    """Flatten a categorical latent tensor.

    Args:
        z: Tensor of shape [..., n_cat, n_cls] (one-hot mode) or [..., n_cat] (indices).
        n_cat: Number of categorical variables.
        n_cls: Number of classes per categorical.
        mode: 'onehot' keeps z as-is, 'indices' converts to argmax,
              'embed' returns flattened float representation.

    Returns:
        Flattened tensor.
    """
    if mode == "onehot":
        return z.flatten(start_dim=-2)
    elif mode == "indices":
        return z.argmax(dim=-1).flatten()
    elif mode == "embed":
        indices = z.argmax(dim=-1)
        one_hot = torch.nn.functional.one_hot(indices, num_classes=n_cls).float()
        return one_hot.flatten(start_dim=-2)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def stack_trajectories(
    trajectories: List[torch.Tensor],
    dim: int = 0,
) -> torch.Tensor:
    """Stack a list of tensors along a new dimension.

    Args:
        trajectories: List of tensors with compatible shapes.
        dim: Dimension along which to stack.

    Returns:
        Stacked tensor.
    """
    return torch.stack(trajectories, dim=dim)


def safe_kl_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    epsilon: float = 1e-8,
    reduction: Literal["none", "sum", "mean"] = "none",
) -> torch.Tensor:
    """Compute numerically stable KL divergence between categorical distributions.

    KL(p || q) = sum(p * log(p / q))

    Args:
        p: First distribution (posterior), shape [..., n_cat, n_cls].
        q: Second distribution (prior), shape [..., n_cat, n_cls].
        epsilon: Small constant for numerical stability.
        reduction: How to reduce the result.

    Returns:
        KL divergence tensor.
    """
    p = p.clamp(min=epsilon)
    q = q.clamp(min=epsilon)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    kl = p * (p.log() - q.log())
    return kl.sum(dim=-1) if reduction == "none" else kl.sum(dim=-1).__getattribute__(reduction)()
