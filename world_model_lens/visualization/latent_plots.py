"""Latent Trajectory Visualization.

PCA/t-SNE projections of latent states over time.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
import torch
import numpy as np


@dataclass
class LatentProjection:
    """2D projection of latent trajectory."""

    x: np.ndarray
    y: np.ndarray
    labels: np.ndarray  # Timestep or other label
    colors: Optional[np.ndarray] = None


class LatentTrajectoryPlotter:
    """Visualize latent trajectories in 2D.

    Example:
        plotter = LatentTrajectoryPlotter(world_model)

        # Generate trajectory
        obs = torch.randn(20, 3, 64, 64)
        traj, cache = world_model.run_with_cache(obs)

        # PCA projection
        pca = plotter.project_pca(traj, n_components=2)

        # t-SNE projection
        tsne = plotter.project_tsne(traj, perplexity=5)

        # Plot with matplotlib
        import matplotlib.pyplot as plt
        plt.scatter(pca.x, pca.y, c=pca.labels)
        plt.colorbar()
        plt.show()
    """

    def __init__(self, world_model: Any):
        """Initialize plotter.

        Args:
            world_model: HookedWorldModel instance
        """
        self.wm = world_model
        self._pca_cache = {}
        self._tsne_cache = {}

    def extract_latents(
        self,
        trajectory: Any,
        component: str = "z_posterior",
    ) -> torch.Tensor:
        """Extract latent states from trajectory.

        Uses the real ``LatentState`` API: ``h_t`` for the recurrent state,
        ``z_posterior`` for the stochastic latent.  Both are flattened to 1-D
        before stacking so the returned tensor is always ``[T, d]``.

        Args:
            trajectory: LatentTrajectory
            component: Which component to extract (``"z_posterior"``,
                ``"z"``, ``"h"``, or ``"hidden"``).

        Returns:
            Tensor of shape [T, d_z]
        """
        latents = []

        for state in trajectory.states:
            if component in ("z_posterior", "z"):
                latents.append(state.z_posterior.flatten())
            elif component in ("h", "hidden"):
                latents.append(state.h_t.flatten())
            elif component == "flat":
                latents.append(state.flat)
            else:
                # Fallback: concatenate h and z
                latents.append(state.flat)

        return torch.stack(latents)

    def project_pca(
        self,
        trajectory: Any,
        n_components: int = 2,
        component: str = "z_posterior",
    ) -> LatentProjection:
        """Project latent trajectory using PCA.

        Args:
            trajectory: WorldTrajectory
            n_components: Number of PCA components
            component: Which latent component to use

        Returns:
            LatentProjection with 2D coordinates
        """
        latents = self.extract_latents(trajectory, component)

        # Flatten
        if latents.dim() > 2:
            latents = latents.flatten(1)

        # Center
        mean = latents.mean(dim=0, keepdim=True)
        latents = latents - mean

        # SVD
        U, S, Vt = torch.pca_lowrank(latents, q=n_components)

        projected = torch.matmul(latents, Vt[:, :n_components])

        projected = projected.detach()

        # Labels: timesteps
        labels = np.arange(len(trajectory.states))

        return LatentProjection(
            x=projected[:, 0].numpy(),
            y=projected[:, 1].numpy(),
            labels=labels,
        )

    def project_tsne(
        self,
        trajectory: Any,
        perplexity: float = 5.0,
        n_iter: int = 1000,
        component: str = "z_posterior",
    ) -> LatentProjection:
        """Project latent trajectory using t-SNE.

        Args:
            trajectory: WorldTrajectory
            perplexity: t-SNE perplexity
            n_iter: Number of iterations
            component: Which latent component to use

        Returns:
            LatentProjection with 2D coordinates
        """
        latents = self.extract_latents(trajectory, component)

        if latents.dim() > 2:
            latents = latents.flatten(1)

        latents_np = latents.numpy()

        from sklearn.manifold import TSNE

        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, len(latents_np) - 1),
            n_iter_without_progress=n_iter,
            random_state=42,
        )

        projected = tsne.fit_transform(latents_np)

        labels = np.arange(len(trajectory.states))

        return LatentProjection(
            x=projected[:, 0],
            y=projected[:, 1],
            labels=labels,
        )

    def project_umap(
        self,
        trajectory: Any,
        n_neighbors: int = 5,
        min_dist: float = 0.1,
        component: str = "z_posterior",
    ) -> LatentProjection:
        """Project latent trajectory using UMAP.

        Args:
            trajectory: WorldTrajectory
            n_neighbors: UMAP n_neighbors
            min_dist: UMAP min_dist
            component: Which latent component to use

        Returns:
            LatentProjection with 2D coordinates
        """
        try:
            import umap
        except ImportError:
            raise ImportError("umap-learn not installed. Install with: pip install umap-learn")

        latents = self.extract_latents(trajectory, component)

        if latents.dim() > 2:
            latents = latents.flatten(1)

        latents_np = latents.numpy()

        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42,
        )

        projected = reducer.fit_transform(latents_np)

        labels = np.arange(len(trajectory.states))

        return LatentProjection(
            x=projected[:, 0],
            y=projected[:, 1],
            labels=labels,
        )

    def color_by_reward(
        self,
        trajectory: Any,
    ) -> np.ndarray:
        """Get colors based on reward values.

        Args:
            trajectory: WorldTrajectory

        Returns:
            Array of reward values
        """
        rewards = []

        for state in trajectory.states:
            # Use the real LatentState fields: reward_pred (predicted) or
            # reward_real (ground-truth from env).  Prefer predicted.
            reward = state.reward_pred if state.reward_pred is not None else state.reward_real
            if reward is not None:
                if isinstance(reward, torch.Tensor):
                    rewards.append(reward.item())
                else:
                    rewards.append(float(reward))
            else:
                rewards.append(0.0)

        return np.array(rewards, dtype=np.float32)

    def color_by_surprise(
        self,
        trajectory: Any,
        cache: Any,
    ) -> np.ndarray:
        """Get colors based on surprise (KL divergence).

        Args:
            trajectory: WorldTrajectory
            cache: ActivationCache

        Returns:
            Array of surprise values
        """
        surprises = []

        for t in range(len(trajectory.states)):
            kl = cache.get("kl", t)
            if kl is not None:
                if isinstance(kl, torch.Tensor):
                    surprises.append(kl.item())
                else:
                    surprises.append(kl)
            else:
                surprises.append(0.0)

        return np.array(surprises)

    def plot_trajectory_gallery(
        self,
        trajectories: List[Any],
        labels: Optional[List[str]] = None,
        method: str = "pca",
    ) -> List[LatentProjection]:
        """Plot multiple trajectories in same space.

        Args:
            trajectories: List of WorldTrajectories
            labels: Labels for each trajectory
            method: Projection method ("pca", "tsne", "umap")

        Returns:
            List of LatentProjections
        """
        if labels is None:
            labels = [f"Traj {i}" for i in range(len(trajectories))]

        projections = []

        for traj, label in zip(trajectories, labels):
            if method == "pca":
                proj = self.project_pca(traj)
            elif method == "tsne":
                proj = self.project_tsne(traj)
            elif method == "umap":
                proj = self.project_umap(traj)
            else:
                raise ValueError(f"Unknown method: {method}")

            projections.append(proj)

        return projections

    def compute_velocity(
        self,
        trajectory: Any,
        component: str = "z_posterior",
    ) -> np.ndarray:
        """Compute velocity (change) between timesteps.

        Args:
            trajectory: WorldTrajectory
            component: Which component

        Returns:
            Array of velocities
        """
        latents = self.extract_latents(trajectory, component)

        if latents.dim() > 2:
            latents = latents.flatten(1)

        velocities = torch.diff(latents, dim=0).norm(dim=1)

        return velocities.numpy()

    def compute_acceleration(
        self,
        trajectory: Any,
        component: str = "z_posterior",
    ) -> np.ndarray:
        """Compute acceleration (change in velocity).

        Args:
            trajectory: WorldTrajectory
            component: Which component

        Returns:
            Array of accelerations
        """
        latents = self.extract_latents(trajectory, component)

        if latents.dim() > 2:
            latents = latents.flatten(1)

        velocities = torch.diff(latents, dim=0)
        accelerations = torch.diff(velocities, dim=0).norm(dim=1)

        return accelerations.numpy()


class SurpriseHeatmap:
    """2-D heatmap of per-dimension KL divergence (surprise) over time.

    Each cell ``[t, d]`` shows how much dimension *d* of the stochastic
    latent contributed to the KL divergence at timestep *t*.  This is the
    most important mechanistic-interpretability visualisation for world
    models: it reveals *when* and *which* latent dimensions were surprised
    by the actual observation.

    The raw KL stored in the :class:`ActivationCache` (key ``"kl"``) is a
    scalar per timestep.  ``SurpriseHeatmap`` instead computes the
    *per-category* KL contribution from the cached ``"z_posterior"`` and
    ``"z_prior"`` entries so that the heatmap has shape ``[T, n_cat]``.

    Example::

        heatmap = SurpriseHeatmap()
        data = heatmap.compute(cache)
        # data["matrix"] has shape [T, n_cat]
        # data["timesteps"]  → [0, 1, …, T-1]
        # data["kl_per_cat"] → same as matrix (alias)
        # data["kl_total"]   → [T] scalar KL per step
    """

    @staticmethod
    def compute(
        cache: Any,
        eps: float = 1e-8,
    ) -> dict:
        """Compute the ``[T, n_cat]`` surprise heatmap from *cache*.

        Reads ``"z_posterior"`` and ``"z_prior"`` from *cache*.  If the
        pre-computed ``"kl"`` scalar is present it is also included in the
        output for convenience.

        Args:
            cache: :class:`ActivationCache` populated by
                ``HookedWorldModel.run_with_cache()``.
            eps: Numerical floor for log-probabilities.

        Returns:
            dict with keys:

            * ``"matrix"``    — ``np.ndarray`` of shape ``[T, n_cat]``
            * ``"timesteps"`` — ``np.ndarray`` of length T
            * ``"kl_per_cat"``— alias for ``"matrix"``
            * ``"kl_total"``  — ``np.ndarray`` of shape ``[T]`` (sum over cats)

        Raises:
            KeyError: If neither ``"z_posterior"`` nor ``"z_prior"`` are
                present in *cache*.
        """
        # Collect timesteps that have both posterior and prior
        post_ts = {t for (n, t) in cache.keys() if n == "z_posterior"}
        prior_ts = {t for (n, t) in cache.keys() if n == "z_prior"}
        shared = sorted(post_ts & prior_ts)

        if not shared:
            raise KeyError(
                "SurpriseHeatmap requires 'z_posterior' and 'z_prior' "
                "entries in the ActivationCache. Run run_with_cache() first."
            )

        rows: List[np.ndarray] = []
        kl_totals: List[float] = []

        for t in shared:
            post = cache.get("z_posterior", t)  # [n_cat, n_cls]
            prior = cache.get("z_prior", t)      # [n_cat, n_cls]

            p = post.softmax(dim=-1).clamp(min=eps)   # [n_cat, n_cls]
            q = prior.softmax(dim=-1).clamp(min=eps)  # [n_cat, n_cls]

            # Per-category KL: [n_cat]  (sum over n_cls)
            kl_cat = (p * (p.log() - q.log())).sum(dim=-1)
            rows.append(kl_cat.detach().cpu().numpy())
            kl_totals.append(kl_cat.sum().item())

        matrix = np.stack(rows, axis=0)  # [T, n_cat]

        return {
            "matrix": matrix,
            "timesteps": np.array(shared),
            "kl_per_cat": matrix,           # alias
            "kl_total": np.array(kl_totals),
        }
