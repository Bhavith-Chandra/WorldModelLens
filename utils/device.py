"""Device selection utilities."""

from __future__ import annotations

import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Return the best available :class:`torch.device`.

    Priority order (unless *prefer* overrides):

    1. ``cuda``  — if at least one CUDA GPU is visible.
    2. ``mps``   — Apple Silicon GPU (macOS).
    3. ``cpu``   — universal fallback.

    Parameters
    ----------
    prefer:
        Force a specific device string (e.g. ``"cuda:1"``, ``"cpu"``).
        If the requested device is unavailable a ``RuntimeError`` is raised.

    Returns
    -------
    torch.device
        Selected device.

    Examples
    --------
    >>> from world_model_lens.utils.device import get_device
    >>> device = get_device()
    >>> print(device)
    cuda          # or mps / cpu depending on the host machine
    """
    if prefer is not None:
        requested = torch.device(prefer)
        _validate_device(requested)
        return requested

    if torch.cuda.is_available():
        return torch.device("cuda")

    # MPS is available on Apple Silicon with PyTorch ≥ 1.12.
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def _validate_device(device: torch.device) -> None:
    """Raise :class:`RuntimeError` if *device* is not usable on this host."""
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device '{device}' but CUDA is not available on this host."
        )
    if device.type == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError(
                f"Requested device '{device}' but MPS is not available on this host."
            )
