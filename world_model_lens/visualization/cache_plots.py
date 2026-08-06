from typing import Any, Dict

import numpy as np
import torch


class CacheSignalPlotter:
    """Plotters for arbitrary signals extracted from ActivationCache.

    Examples:
        plotter = CacheSignalPlotter()
        data = plotter.plot_surprise_timeline(cache)
        # data["timesteps"] and data["kl_values"] for simple plotting
    """

    @staticmethod
    def plot_surprise_timeline(cache: Any) -> Dict[str, np.ndarray]:
        """Extract surprise (KL) over time.

        Args:
            cache: ActivationCache instance.

        Returns:
            Dict containing timesteps and kl_values arrays.
        """
        if hasattr(cache, "surprise"):
            try:
                surprises = cache.surprise()
                timesteps = np.array(sorted(surprises.keys()))
                kl_values = np.array(
                    [
                        surprises[t].item()
                        if isinstance(surprises[t], torch.Tensor)
                        else float(surprises[t])
                        for t in timesteps
                    ]
                )
                return {"timesteps": timesteps, "kl_values": kl_values}
            except (KeyError, AttributeError, ValueError):
                pass

        # Fallback if surprise() fails or not present, but we have "kl" keys
        timesteps = sorted([t for (n, t) in cache.keys() if n == "kl"])
        kl_values = np.array([cache.get("kl", t).item() for t in timesteps])
        return {"timesteps": np.array(timesteps), "kl_values": kl_values}

    @staticmethod
    def plot_reward_timeline(trajectory: Any) -> Dict[str, np.ndarray]:
        """Extract reward predictions over time.

        Args:
            trajectory: LatentTrajectory instance.

        Returns:
            Dict containing timesteps, predicted, and actual.
        """
        timesteps = []
        predicted = []
        actual = []

        for i, state in enumerate(trajectory.states):
            timesteps.append(i)
            r_pred = state.reward_pred
            r_real = getattr(state, "reward_real", None) or getattr(state, "reward", None)

            if r_pred is not None:
                predicted.append(
                    float(r_pred) if not isinstance(r_pred, torch.Tensor) else r_pred.item()
                )
            else:
                predicted.append(0.0)

            if r_real is not None:
                actual.append(
                    float(r_real) if not isinstance(r_real, torch.Tensor) else r_real.item()
                )
            else:
                actual.append(0.0)

        return {
            "timesteps": np.array(timesteps),
            "predicted": np.array(predicted),
            "actual": np.array(actual),
        }

    @staticmethod
    def plot_value_timeline(trajectory: Any) -> Dict[str, np.ndarray]:
        """Extract value predictions over time.

        Args:
            trajectory: LatentTrajectory instance.

        Returns:
            Dict containing timesteps and value_pred.
        """
        timesteps = []
        value_pred = []

        for i, state in enumerate(trajectory.states):
            timesteps.append(i)
            v = state.value_pred

            if v is not None:
                value_pred.append(float(v) if not isinstance(v, torch.Tensor) else v.item())
            else:
                value_pred.append(0.0)

        return {
            "timesteps": np.array(timesteps),
            "value_pred": np.array(value_pred),
        }

    @staticmethod
    def plot_cache_signal(cache: Any, component: str) -> Dict[str, np.ndarray]:
        """Compute the L2 norm of the given activation component over time.

        Args:
            cache: ActivationCache instance.
            component: Name of the component to extract.

        Returns:
            Dict containing timesteps and norms arrays.
        """
        timesteps = sorted([t for (n, t) in cache.keys() if n == component])
        norms = []

        for t in timesteps:
            tensor = cache.get(component, t)
            norms.append(tensor.norm().item())

        return {
            "timesteps": np.array(timesteps),
            "norms": np.array(norms),
        }
