"""Core tensor manipulation helpers used across world_model_lens."""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# flatten_latent
# ---------------------------------------------------------------------------

def flatten_latent(latent: torch.Tensor, start_dim: int = 1) -> torch.Tensor:
    """Flatten a latent representation to a 2-D matrix.

    Collapses all dimensions from *start_dim* onward into a single feature
    dimension, leaving the batch (or time-step) dimension intact.

    Parameters
    ----------
    latent:
        Input tensor of shape ``(B, *spatial_or_channel_dims)``.
    start_dim:
        First dimension to collapse.  Default ``1`` collapses everything
        after the batch dimension.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(B, D)`` where ``D = prod(latent.shape[start_dim:])``.

    Examples
    --------
    >>> x = torch.randn(4, 8, 8, 16)   # batch=4, 8×8 spatial, 16 channels
    >>> flatten_latent(x).shape
    torch.Size([4, 1024])
    """
    # TODO: add optional normalisation pass (L2, layer norm) once stabilised.
    return latent.flatten(start_dim=start_dim)


# ---------------------------------------------------------------------------
# stack_trajectories
# ---------------------------------------------------------------------------

def stack_trajectories(
    trajectories: list[torch.Tensor],
    pad_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack a list of variable-length trajectories into a padded batch tensor.

    Parameters
    ----------
    trajectories:
        List of tensors, each of shape ``(T_i, *feature_dims)``.  Trajectories
        may have different time-lengths ``T_i``.
    pad_value:
        Scalar used to fill shorter sequences.

    Returns
    -------
    padded : torch.Tensor
        Shape ``(N, T_max, *feature_dims)``.
    lengths : torch.Tensor
        1-D long tensor of shape ``(N,)`` containing each trajectory's
        original length.

    Examples
    --------
    >>> t1 = torch.randn(5, 32)
    >>> t2 = torch.randn(3, 32)
    >>> padded, lengths = stack_trajectories([t1, t2])
    >>> padded.shape
    torch.Size([2, 5, 32])
    >>> lengths
    tensor([5, 3])
    """
    if not trajectories:
        raise ValueError("`trajectories` must be a non-empty list.")

    lengths = torch.tensor([t.shape[0] for t in trajectories], dtype=torch.long)
    t_max = int(lengths.max().item())
    feature_dims = trajectories[0].shape[1:]

    padded = torch.full(
        (len(trajectories), t_max, *feature_dims),
        fill_value=pad_value,
        dtype=trajectories[0].dtype,
        device=trajectories[0].device,
    )
    for i, traj in enumerate(trajectories):
        padded[i, : traj.shape[0]] = traj

    return padded, lengths


# ---------------------------------------------------------------------------
# safe_kl_divergence
# ---------------------------------------------------------------------------

def safe_kl_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    eps: float = 1e-8,
    reduction: str = "batchmean",
) -> torch.Tensor:
    """Numerically stable KL divergence  KL(P ‖ Q).

    Applies small-epsilon clamping before taking logarithms so that
    zero-probability bins don't produce ``inf``/``nan``.

    Parameters
    ----------
    p:
        Target distribution tensor.  Can be logits (will be softmaxed) or
        a valid probability distribution.  Shape ``(*, C)``.
    q:
        Approximate distribution tensor.  Same shape as *p*.
    eps:
        Small constant added before log to avoid ``log(0)``.
    reduction:
        Reduction mode passed to :func:`torch.nn.functional.kl_div`.
        Typically ``"batchmean"`` (default) or ``"sum"`` or ``"none"``.

    Returns
    -------
    torch.Tensor
        KL divergence scalar (or per-element tensor if ``reduction="none"``).

    Notes
    -----
    ``torch.nn.functional.kl_div`` expects *log-probabilities* for the input
    and *probabilities* for the target.  This wrapper handles the conversion.

    Examples
    --------
    >>> p = torch.tensor([[0.9, 0.1]])
    >>> q = torch.tensor([[0.6, 0.4]])
    >>> safe_kl_divergence(p, q)
    tensor(0.1719)
    """
    # Normalise to proper probability distributions.
    p_prob = p.softmax(dim=-1) if not _is_prob_dist(p) else p
    q_prob = q.softmax(dim=-1) if not _is_prob_dist(q) else q

    # Clamp to avoid log(0).
    p_prob = p_prob.clamp(min=eps)
    q_prob = q_prob.clamp(min=eps)

    log_q = q_prob.log()
    return F.kl_div(log_q, p_prob, reduction=reduction)


def _is_prob_dist(t: torch.Tensor, atol: float = 1e-3) -> bool:
    """Heuristic: check whether *t* already sums to ~1 along the last dim."""
    return bool(torch.allclose(t.sum(dim=-1), torch.ones_like(t.sum(dim=-1)), atol=atol))
