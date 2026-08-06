"""Device selection utilities."""

from typing import Literal

import torch


def get_device(prefer: Literal["cuda", "mps", "cpu"] = None) -> torch.device:
    """Return the best available compute device.

    Priority order (if prefer is None): CUDA > MPS > CPU.

    Args:
        prefer: Override the device preference. If specified, only
                that device type is considered (if available).

    Returns:
        torch.device: The selected device.
    """
    if prefer is not None:
        if prefer == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif prefer == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        elif prefer == "cpu":
            return torch.device("cpu")
        else:
            raise RuntimeError(f"Requested device '{prefer}' is not available.")

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
