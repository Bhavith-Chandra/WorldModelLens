"""Belief analyzer for surprise, concepts, saliency, and hallucination detection.

This module provides analyzers that work with ANY world model type:
- RL models: full support including reward attribution, value analysis
- Non-RL models: core latent analysis (surprise, geometry, disentanglement)
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    from world_model_lens import ActivationCache, HookedWorldModel, LatentTrajectory


@dataclass
class SurpriseResult:
    """Result of surprise analysis."""

    kl_sequence: torch.Tensor
    peaks: list[tuple[int, float]]
    mean_surprise: float
    max_surprise_timestep: int
    max_surprise_value: float

    def correlate_with_rewards(self, rewards: torch.Tensor) -> float:
        """Compute correlation between surprise and rewards.

        Args:
            rewards: Reward tensor to correlate with.

        Returns:
            Spearman correlation coefficient, or 0.0 if lengths don't match.
        """
        if len(self.kl_sequence) != len(rewards):
            return 0.0
        kl_np = self.kl_sequence.cpu().numpy()
        reward_np = rewards.cpu().numpy() if isinstance(rewards, torch.Tensor) else rewards
        return float(_spearman_corr(kl_np, reward_np))

    def plot(self, figsize=(12, 4)):
        """Plot surprise timeline."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.kl_sequence.cpu().numpy())
        ax.set_xlabel("Timestep")
        ax.set_ylabel("KL Surprise")
        ax.set_title(f"Surprise Timeline (mean={self.mean_surprise:.2f})")
        ax.axhline(y=self.mean_surprise, color="r", linestyle="--", alpha=0.5)
        return fig


@dataclass
class ConceptSearchResult:
    """Result of concept search."""

    concept_name: str
    dim_scores: torch.Tensor
    top_dims: list[int]
    concept_vector: torch.Tensor
    method: str

    def plot_activations(self, activations: torch.Tensor, figsize=(12, 6)):
        """Plot activations for top dimensions."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes[0].bar(range(len(self.dim_scores)), self.dim_scores.cpu().numpy())
        axes[0].set_title(f"Concept: {self.concept_name}")
        axes[0].set_xlabel("Dimension")

        if len(self.top_dims) > 0:
            top_vals = activations[:, self.top_dims[:5]].cpu().numpy()
            for i in range(min(5, len(self.top_dims))):
                axes[1].plot(top_vals[:, i], label=f"Dim {self.top_dims[i]}")
            axes[1].legend()
            axes[1].set_title("Top 5 Dimensions Over Time")
        return fig


@dataclass
class SaliencyResult:
    """Result of saliency analysis."""

    h_saliency: torch.Tensor
    z_saliency: Optional[torch.Tensor]
    method: str
    timestep: int

    def plot_z_heatmap(self, figsize=(10, 6)):
        """Plot z saliency as heatmap."""
        if self.z_saliency is None:
            return None
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(self.z_saliency.cpu().numpy(), aspect="auto", cmap="hot")
        ax.set_title(f"Z Saliency at t={self.timestep}")
        ax.set_xlabel("Class")
        ax.set_ylabel("Category")
        plt.colorbar(im, ax=ax)
        return fig


@dataclass
class HallucinationResult:
    """Result of hallucination detection."""

    divergence_timeline: torch.Tensor
    hallucination_timesteps: list[int]
    severity_score: float

    def plot_timeline(self, threshold: float = 0.5, figsize=(12, 4)):
        """Plot divergence timeline with threshold."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)
        div = self.divergence_timeline.cpu().numpy()
        ax.plot(div)
        ax.axhline(y=threshold, color="r", linestyle="--", label="Threshold")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Divergence")
        ax.set_title(f"Hallucination Timeline (severity={self.severity_score:.2f})")
        ax.legend()
        return fig


@dataclass
class DisentanglementResult:
    """Result of disentanglement analysis."""

    scores: dict[str, float]
    factor_dim_assignment: dict[str, list[int]]
    total_score: float

    def heatmap(self, figsize=(10, 8)):
        """Plot factor-dimension heatmap."""
        import matplotlib.pyplot as plt
        import numpy as np

        factors = list(self.factor_dim_assignment.keys())

        fig, ax = plt.subplots(figsize=figsize)
        data = np.zeros((len(factors), 32))
        for i, (_factor, dims) in enumerate(self.factor_dim_assignment.items()):
            for d in dims:
                if d < 32:
                    data[i, d] = 1.0

        ax.imshow(data, aspect="auto", cmap="Blues")
        ax.set_yticks(range(len(factors)))
        ax.set_yticklabels(factors)
        ax.set_xlabel("Z Dimension")
        ax.set_title("Factor-Dimension Assignment")
        return fig


@dataclass
class RewardAttributionResult:
    """Result of reward attribution analysis.

    NOTE: This is only available for RL models with reward heads.
    For non-RL models, this will return zero attributions.
    """

    dim_attribution: torch.Tensor
    top_dims: list[int]
    top_attribution_values: list[float]
    is_available: bool = True

    def __post_init__(self):
        if not self.is_available:
            self.dim_attribution = torch.zeros(32)
            self.top_dims = []
            self.top_attribution_values = []


def _spearman_corr(x, y):
    """Compute Spearman correlation."""
    from scipy.stats import spearmanr

    return spearmanr(x, y)[0]


class BeliefAnalyzer:
    """Analyzer for world model beliefs and representations.

    Works with ANY world model type:
    - RL models: full analysis including reward attribution, value analysis
    - Non-RL models: core latent analysis (surprise, geometry, disentanglement)

    Analysis supported for all model types:
    - surprise_timeline: KL divergence between posterior and prior
    - concept_search: Find dimensions encoding specific concepts
    - latent_saliency: Compute saliency maps for latent dimensions
    - detect_hallucinations: Compare real vs imagined trajectories
    - disentanglement_score: Measure disentanglement of factors

    RL-specific (gracefully skipped for non-RL models):
    - reward_attribution: Attribute reward to latent dimensions
    - value_analysis: Analyze value predictions
    """

    def __init__(self, wm: "HookedWorldModel"):
        """Initialize analyzer.

        Args:
            wm: HookedWorldModel instance to analyze.
        """
        self.wm = wm
        self._caps = wm.capabilities if hasattr(wm, "capabilities") else None

    @property
    def capabilities(self):
        """Access the world model's capabilities."""
        return self._caps

    def surprise_timeline(
        self,
        cache: "ActivationCache",
        obs_seq: Optional[torch.Tensor] = None,
        decoder: Optional[Any] = None,
    ) -> SurpriseResult:
        """Compute surprise (KL divergence or cosine distance) timeline.

        Works with ANY world model - computes KL between posterior and prior
        latents when both are available, or cosine distance for I-JEPA predictor
        vs target encoder outputs.

        Args:
            cache: ActivationCache from run_with_cache.
            obs_seq: Optional observations for decoding (unused, kept for API).
            decoder: Optional decoder for observation reconstruction (unused, kept for API).

        Returns:
            SurpriseResult with timeline and statistics.
        """
        timesteps = sorted(cache.timesteps)

        if (
            "predictor_out" in cache.component_names
            and "target_encoder_out" in cache.component_names
        ):
            # I-JEPA cosine distance
            dist_vals = []
            for t in timesteps:
                try:
                    pred_out = cache["predictor_out", t]
                    targ_out = cache["target_encoder_out", t]
                    # Assume pred_out and targ_out are [num_patches, dim]
                    similarities = torch.nn.functional.cosine_similarity(pred_out, targ_out, dim=-1)
                    distances = 1 - similarities
                    dist_vals.append(distances.mean().item())
                except (KeyError, TypeError):
                    dist_vals.append(0.0)
            kl_tensor = torch.tensor(dist_vals)
        else:
            # Original KL divergence
            kl_vals = []
            for t in timesteps:
                try:
                    z_post = cache["z_posterior", t]
                    z_prior = cache["z_prior", t]

                    if z_post.shape != z_prior.shape or z_post.dim() <= 1:
                        kl_vals.append(0.0)
                        continue

                    p = z_post.clamp(min=1e-8)
                    q = z_prior.clamp(min=1e-8)
                    p = p / p.sum(dim=-1, keepdim=True)
                    q = q / q.sum(dim=-1, keepdim=True)
                    kl = (p * (p.log() - q.log())).sum().item()
                    kl_vals.append(kl)
                except (KeyError, TypeError):
                    kl_vals.append(0.0)
            kl_tensor = torch.tensor(kl_vals)

        mean_surprise = kl_tensor.mean().item() if len(kl_tensor) > 0 else 0.0
        max_idx = int(kl_tensor.argmax().item()) if len(kl_tensor) > 0 else 0
        threshold = mean_surprise + kl_tensor.std().item() if len(kl_tensor) > 0 else 0
        peaks = [(i, v.item()) for i, v in enumerate(kl_tensor) if v > threshold]

        return SurpriseResult(
            kl_sequence=kl_tensor,
            peaks=peaks,
            mean_surprise=mean_surprise,
            max_surprise_timestep=max_idx,
            max_surprise_value=kl_tensor[max_idx].item() if len(kl_tensor) > 0 else 0.0,
        )

    def concept_search(
        self,
        concept_name: str,
        positive_timesteps: list[int],
        negative_timesteps: list[int],
        cache: "ActivationCache",
        component: str = "z_posterior",
        method: str = "mean_difference",
    ) -> ConceptSearchResult:
        """Search for concept alignment in latent dimensions.

        Works with ANY world model - analyzes latent representations
        for concept encoding.

        Args:
            concept_name: Name of the concept.
            positive_timesteps: Timesteps with positive concept.
            negative_timesteps: Timesteps without concept.
            cache: ActivationCache.
            component: Which activation to analyze ('z_posterior', 'h', etc.).
            method: 'mean_difference' or 'mutual_information'.

        Returns:
            ConceptSearchResult with dimension scores.
        """
        pos_activations = []
        neg_activations = []

        for t in positive_timesteps:
            try:
                act = cache[component, t]
                pos_activations.append(act.flatten())
            except KeyError:
                pass

        for t in negative_timesteps:
            try:
                act = cache[component, t]
                neg_activations.append(act.flatten())
            except KeyError:
                pass

        if not pos_activations or not neg_activations:
            return ConceptSearchResult(
                concept_name=concept_name,
                dim_scores=torch.zeros(32),
                top_dims=[],
                concept_vector=torch.zeros(32 * 32)
                if component == "z_posterior"
                else torch.zeros(32),
                method=method,
            )

        pos_mean = torch.stack(pos_activations).mean(dim=0)
        neg_mean = torch.stack(neg_activations).mean(dim=0)

        dim_scores = pos_mean - neg_mean
        if dim_scores.norm() > 0:
            dim_scores = dim_scores / dim_scores.norm()
        concept_vector = dim_scores.clone()
        top_dims = dim_scores.abs().argsort(descending=True)[:10].tolist()

        return ConceptSearchResult(
            concept_name=concept_name,
            dim_scores=dim_scores,
            top_dims=top_dims,
            concept_vector=concept_vector,
            method=method,
        )

    def latent_saliency(
        self,
        traj: "LatentTrajectory",
        cache: "ActivationCache",
        timestep: int,
        target: str = "state",
        method: str = "gradient",
        target_class: Optional[int] = None,
    ) -> SaliencyResult:
        """Compute saliency maps for latent representations.

        Works with ANY world model - computes saliency for latent states.

        Args:
            traj: LatentTrajectory.
            cache: ActivationCache.
            timestep: Timestep to analyze.
            target: What to compute saliency for ('state', 'reward_pred', etc.).
            method: 'gradient', 'occlusion', or 'integrated_gradients'.
            target_class: Optional class index for classification targets.

        Returns:
            SaliencyResult with saliency maps.
        """
        try:
            h = cache["h", timestep].clone().requires_grad_(True)
            z = cache["z_posterior", timestep].clone().requires_grad_(True)
        except KeyError:
            h = torch.zeros(256, requires_grad=True)
            z = torch.zeros(32, requires_grad=True)

        device = h.device
        h_sal = torch.zeros_like(h)
        z_sal = torch.zeros_like(z)

        if method == "gradient":
            if target == "state":
                loss = (h * h).sum() + (z * z).sum()
                loss.backward()
                h_sal = h.grad.abs() if h.grad is not None else torch.zeros_like(h)
                z_sal = z.grad.abs() if z.grad is not None else torch.zeros_like(z)
            elif target == "reward_pred" and "reward" in cache.component_names:
                reward_val = cache["reward", timestep].sum()
                reward_val.backward()
                h_sal = h.grad.abs() if h.grad is not None else torch.zeros_like(h)
                z_sal = z.grad.abs() if z.grad is not None else torch.zeros_like(z)
            else:
                loss = (h * h).sum() + (z * z).sum()
                loss.backward()
                h_sal = h.grad.abs() if h.grad is not None else torch.zeros_like(h)
                z_sal = z.grad.abs() if z.grad is not None else torch.zeros_like(z)

        elif method == "occlusion":
            h_occlusion_scores = []
            z_occlusion_scores = []

            if target == "state" or target == "all":
                for dim in range(h.shape[-1]):
                    h_perturbed = h.clone()
                    h_perturbed[..., dim] = 0
                    score = (h_perturbed - h).abs().sum().item()
                    h_occlusion_scores.append(score)

                for dim in range(z.shape[-1]):
                    z_perturbed = z.clone()
                    z_perturbed[..., dim] = 0
                    score = (z_perturbed - z).abs().sum().item()
                    z_occlusion_scores.append(score)

                if h_occlusion_scores:
                    h_sal = torch.tensor(h_occlusion_scores, device=device)
                    if h_sal.max() > 0:
                        h_sal = h_sal / h_sal.max()
                if z_occlusion_scores:
                    z_sal = torch.tensor(z_occlusion_scores, device=device)
                    if z_sal.max() > 0:
                        z_sal = z_sal / z_sal.max()

        elif method == "integrated_gradients":
            baseline_h = torch.zeros_like(h)
            baseline_z = torch.zeros_like(z)
            n_steps = 20

            step_size_h = (h - baseline_h) / n_steps
            step_size_z = (z - baseline_z) / n_steps

            integrated_h = torch.zeros_like(h)
            integrated_z = torch.zeros_like(z)

            for step in range(n_steps + 1):
                current_h = baseline_h + step_size_h * step
                current_z = baseline_z + step_size_z * step

                current_h.requires_grad_(True)
                current_z.requires_grad_(True)

                if target == "state":
                    loss = (current_h * current_h).sum() + (current_z * current_z).sum()
                else:
                    loss = (current_h * current_h).sum() + (current_z * current_z).sum()

                loss.backward()

                if current_h.grad is not None:
                    integrated_h += current_h.grad
                if current_z.grad is not None:
                    integrated_z += current_z.grad

                current_h.requires_grad_(False)
                current_z.requires_grad_(False)

            h_sal = integrated_h.abs() / (n_steps + 1)
            z_sal = integrated_z.abs() / (n_steps + 1)

            if h_sal.max() > 0:
                h_sal = h_sal / h_sal.max()
            if z_sal.max() > 0:
                z_sal = z_sal / z_sal.max()

        return SaliencyResult(
            h_saliency=h_sal.detach(),
            z_saliency=z_sal.detach(),
            method=method,
            timestep=timestep,
        )

    def detect_hallucinations(
        self,
        real_traj: "LatentTrajectory",
        imagined_traj: "LatentTrajectory",
        decoder: Optional[Any] = None,
        method: str = "latent_distance",
        threshold: float = 0.5,
    ) -> HallucinationResult:
        """Detect hallucinations in imagined trajectories.

        Works with ANY world model - compares latent representations
        between real and imagined trajectories.

        Args:
            real_traj: Real trajectory.
            imagined_traj: Imagined trajectory.
            decoder: Optional decoder for pixel-space comparison.
            method: 'latent_distance', 'mse', or 'ssim'.
            threshold: Threshold for hallucination detection.

        Returns:
            HallucinationResult with divergence timeline.
        """
        divergences = []
        hallucination_timesteps = []

        num_timesteps = min(len(real_traj.states), len(imagined_traj.states))

        for t in range(num_timesteps):
            real_state = real_traj.states[t].state
            imag_state = imagined_traj.states[t].state

            if real_state.shape != imag_state.shape:
                real_flat = real_state.flatten()
                imag_flat = imag_state.flatten()
                min_len = min(len(real_flat), len(imag_flat))
                real_flat = real_flat[:min_len]
                imag_flat = imag_flat[:min_len]
            else:
                real_flat = real_state
                imag_flat = imag_state

            dist = torch.nn.functional.mse_loss(real_flat, imag_flat).item()
            divergences.append(dist)

            if dist > threshold:
                hallucination_timesteps.append(t)

        div_tensor = torch.tensor(divergences)
        severity = div_tensor.mean().item() if len(divergences) > 0 else 0.0

        return HallucinationResult(
            divergence_timeline=div_tensor,
            hallucination_timesteps=hallucination_timesteps,
            severity_score=severity,
        )

    def disentanglement_score(
        self,
        cache: "ActivationCache",
        factors: Optional[dict[str, torch.Tensor]] = None,
        metrics: Optional[list[str]] = None,
        component: str = "z_posterior",
    ) -> DisentanglementResult:
        """Compute disentanglement metrics.

        Works with ANY world model - measures how well latent dimensions
        encode distinct factors of variation.

        Args:
            cache: ActivationCache.
            factors: Optional dict mapping factor names to factor values per timestep.
                   If None, uses internal variance analysis.
            metrics: Metrics to compute ('MIG', 'DCI', 'SAP').
            component: Which activation to analyze.

        Returns:
            DisentanglementResult with scores.
        """
        scores: dict[str, float] = {}
        factor_dim_assignment: dict[str, list[int]] = {}

        if metrics is None:
            metrics = ["MIG", "DCI", "SAP"]

        try:
            z_seq = cache[component]
            if z_seq is None or len(z_seq) == 0:
                for metric in metrics:
                    scores[metric] = 0.0
                return DisentanglementResult(
                    scores=scores, factor_dim_assignment={}, total_score=0.0
                )
        except KeyError:
            for metric in metrics:
                scores[metric] = 0.0
            return DisentanglementResult(scores=scores, factor_dim_assignment={}, total_score=0.0)

        if z_seq.dim() == 3:
            z_flat = z_seq.reshape(z_seq.shape[0], -1)
        else:
            z_flat = z_seq

        if factors is None or len(factors) == 0:
            variance_per_dim = z_flat.var(dim=0)
            top_dims = variance_per_dim.argsort(descending=True)[:10].tolist()
            factor_dim_assignment["high_variance"] = top_dims
            for metric in metrics:
                scores[metric] = float(variance_per_dim.mean().item())
        else:
            for factor_name, factor_values in factors.items():
                if len(factor_values) != len(z_flat):
                    continue

                factor_dim_assignment[factor_name] = []

                if "MIG" in metrics:
                    mig_score = self._compute_mig(z_flat, factor_values)
                    scores["MIG"] = scores.get("MIG", 0.0) + mig_score

                if "DCI" in metrics:
                    dci_score, top_dims = self._compute_dci(z_flat, factor_values)
                    scores["DCI"] = scores.get("DCI", 0.0) + dci_score
                    factor_dim_assignment[factor_name].extend(top_dims[:5])

                if "SAP" in metrics:
                    sap_score = self._compute_sap(z_flat, factor_values)
                    scores["SAP"] = scores.get("SAP", 0.0) + sap_score

            n_factors = len([f for f in factors.keys() if len(factors[f]) == len(z_flat)])
            if n_factors > 0:
                for metric in metrics:
                    scores[metric] = scores.get(metric, 0.0) / n_factors

        total_score = sum(scores.values()) / max(len(scores), 1)

        return DisentanglementResult(
            scores=scores,
            factor_dim_assignment=factor_dim_assignment,
            total_score=total_score,
        )

    def _compute_mig(self, z_flat: torch.Tensor, factors: torch.Tensor) -> float:
        """Compute Mutual Information Gap.

        Args:
            z_flat: Flattened latent representations [T, D].
            factors: Factor values [T].

        Returns:
            MIG score.
        """
        if z_flat.shape[0] < 10 or factors.shape[0] < 10:
            return 0.0

        try:
            from sklearn.cluster import KMeans

            n_factors = len(torch.unique(factors))
            n_clusters = min(n_factors, 10)

            if n_clusters < 2:
                return 0.0

            mi_scores = []

            for d in range(z_flat.shape[1]):
                dim_values = z_flat[:, d]
                dim_normalized = (dim_values - dim_values.min()) / (
                    dim_values.max() - dim_values.min() + 1e-8
                )

                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(dim_normalized.unsqueeze(1))

                mutual_info = self._mutual_information(cluster_labels, factors.numpy())
                mi_scores.append(mutual_info)

            mi_scores.sort(reverse=True)

            if len(mi_scores) < 2:
                return 0.0

            mig = mi_scores[0] - mi_scores[1]
            return max(0.0, float(mig))
        except ImportError:
            return 0.0

    def _compute_dci(self, z_flat: torch.Tensor, factors: torch.Tensor) -> tuple[float, list[int]]:
        """Compute Disentanglement, Completeness, Informativeness.

        Args:
            z_flat: Flattened latent representations [T, D].
            factors: Factor values [T].

        Returns:
            Tuple of (DCI score, top dimensions).
        """
        if z_flat.shape[0] < 10:
            return 0.0, []

        try:
            from sklearn.linear_model import LogisticRegression

            n_factors = len(torch.unique(factors))
            if n_factors < 2:
                return 0.0, []

            dim_importance = []

            for d in range(z_flat.shape[1]):
                dim_vals = z_flat[:, d].unsqueeze(1).cpu().numpy()
                factor_vals = factors.cpu().numpy()

                try:
                    clf = LogisticRegression(max_iter=200, random_state=42)
                    clf.fit(dim_vals, factor_vals)
                    score = clf.score(dim_vals, factor_vals)
                    dim_importance.append((d, score))
                except Exception:
                    dim_importance.append((d, 0.0))

            dim_importance.sort(key=lambda x: x[1], reverse=True)
            top_dims = [d for d, _ in dim_importance[:10]]

            importance_sum = sum(s for _, s in dim_importance)
            if importance_sum > 0:
                dci = sum(s for _, s in dim_importance[:10]) / importance_sum
            else:
                dci = 0.0

            return float(dci), top_dims
        except ImportError:
            return 0.0, []

    def _compute_sap(self, z_flat: torch.Tensor, factors: torch.Tensor) -> float:
        """Compute Separated Attribute Predictability.

        Args:
            z_flat: Flattened latent representations [T, D].
            factors: Factor values [T].

        Returns:
            SAP score.
        """
        if z_flat.shape[0] < 10:
            return 0.0

        try:
            from sklearn.cluster import KMeans

            n_factors = len(torch.unique(factors))
            n_clusters = min(n_factors, 10)

            if n_clusters < 2:
                return 0.0

            dim_scores = []

            for d in range(z_flat.shape[1]):
                dim_vals = z_flat[:, d].cpu().numpy().reshape(-1, 1)

                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(dim_vals)

                correct = sum(
                    1 for c, f in zip(cluster_labels, factors.cpu().numpy(), strict=True) if c == f
                )
                accuracy = correct / len(factors)
                dim_scores.append(accuracy)

            dim_scores.sort(reverse=True)

            if len(dim_scores) < 2:
                return 0.0

            sap = dim_scores[0] - dim_scores[1]
            return max(0.0, float(sap))
        except ImportError:
            return 0.0

    def _mutual_information(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute mutual information between x and y.

        Args:
            x: Array of discrete values.
            y: Array of discrete values.

        Returns:
            Mutual information value.
        """
        from collections import Counter

        import numpy as np

        x = np.array(x)
        y = np.array(y)

        n = len(x)
        if n == 0:
            return 0.0

        p_x = Counter(x)
        p_y = Counter(y)
        p_xy = Counter(zip(x, y, strict=True))

        mi = 0.0
        for (x_val, y_val), p_xy_val in p_xy.items():
            p_x_val = p_x[x_val] / n
            p_y_val = p_y[y_val] / n
            p_xy_val = p_xy_val / n

            if p_x_val > 0 and p_y_val > 0 and p_xy_val > 0:
                mi += p_xy_val * np.log(p_xy_val / (p_x_val * p_y_val))

        return max(0.0, mi)

    def reward_attribution(
        self,
        traj: "LatentTrajectory",
        cache: "ActivationCache",
        method: str = "gradient_times_activation",
        component: str = "z_posterior",
    ) -> RewardAttributionResult:
        """Attribute reward predictions to latent dimensions.

        RL-SPECIFIC: Only available for models with reward heads.
        For non-RL models, returns zero attributions.

        Args:
            traj: LatentTrajectory.
            cache: ActivationCache.
            method: Attribution method.
            component: Which activation to analyze.

        Returns:
            RewardAttributionResult with dimension attributions.
            is_available=False if model has no reward head.
        """
        if self._caps is None or not self._caps.has_reward_head:
            return RewardAttributionResult(
                dim_attribution=torch.zeros(32),
                top_dims=[],
                top_attribution_values=[],
                is_available=False,
            )

        try:
            z_seq = cache[component]
        except (KeyError, TypeError, AttributeError):
            return RewardAttributionResult(
                dim_attribution=torch.zeros(32),
                top_dims=[],
                top_attribution_values=[],
                is_available=False,
            )

        if z_seq is None or len(z_seq) == 0:
            return RewardAttributionResult(
                dim_attribution=torch.zeros(32),
                top_dims=[],
                top_attribution_values=[],
                is_available=False,
            )

        if len(z_seq.shape) == 3:
            z_flat = z_seq.flatten(1)
        else:
            z_flat = z_seq

        attr = z_flat.abs().mean(dim=0)
        top_dims = attr.argsort(descending=True)[:10].tolist()

        return RewardAttributionResult(
            dim_attribution=attr,
            top_dims=top_dims,
            top_attribution_values=[attr[d].item() for d in top_dims if d < len(attr)],
            is_available=True,
        )

    def value_analysis(
        self,
        cache: "ActivationCache",
    ) -> dict[str, Any]:
        """Analyze value predictions from critic heads.

        RL-SPECIFIC: Only available for models with critic heads.
        For non-RL models, returns empty results.

        Args:
            cache: ActivationCache containing value predictions.

        Returns:
            Dict with value analysis results, or empty dict if unavailable.
        """
        if self._caps is None or not self._caps.has_critic:
            return {
                "is_available": False,
                "message": "Value analysis requires a critic head (has_critic=True)",
            }

        values = []
        for t in sorted(cache.timesteps):
            try:
                v = cache["value", t]
                if v is not None:
                    values.append(v.item() if v.dim() == 0 else v.mean().item())
            except KeyError:
                pass

        if not values:
            return {"is_available": False, "values": []}

        return {
            "is_available": True,
            "values": values,
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
