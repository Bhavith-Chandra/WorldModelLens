"""Multimodal analysis and concept discovery for world models."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score

from world_model_lens.core.activation_cache import ActivationCache

if TYPE_CHECKING:
    pass


@dataclass
class Concept:
    """Discovered concept with interpretability metadata.

    Attributes:
        direction: Concept direction vector in latent space.
        name: Human-readable concept name.
        interpretability_score: How interpretable this concept is.
        channel: Which multimodal channel this belongs to.
    """

    direction: torch.Tensor
    name: str
    interpretability_score: float
    channel: str | None = None

    def similarity(self, other: "Concept") -> float:
        """Compute cosine similarity with another concept."""
        return torch.nn.functional.cosine_similarity(
            self.direction.unsqueeze(0),
            other.direction.unsqueeze(0),
        ).item()


@dataclass
class MultimodalProbeResult:
    """Result of probing a multimodal cache."""

    channel: str
    labels: torch.Tensor
    predictions: torch.Tensor
    accuracy: float
    per_channel_accuracy: dict[str, float]


@dataclass
class ConceptDiscoveryResult:
    """Result of automated concept discovery."""

    concepts: list[Concept]
    cluster_assignments: torch.Tensor
    mutual_info_matrix: torch.Tensor
    n_clusters: int


def probe_multimodal(
    cache: ActivationCache,
    labels: torch.Tensor,
    channel_names: list[str],
    probe_fn: Callable | None = None,
) -> MultimodalProbeResult:
    """Probe each multimodal channel separately.

    Args:
        cache: ActivationCache with multimodal activations.
        labels: Ground truth labels [N].
        channel_names: List of channel names to probe ('vision', 'proprio', etc.).
        probe_fn: Optional custom probe function.

    Returns:
        MultimodalProbeResult with per-channel accuracies.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    predictions = {}
    per_channel_acc = {}

    for channel in channel_names:
        try:
            channel_data = []
            for t in cache.timesteps:
                if (channel, t) in cache:
                    channel_data.append(cache[channel, t])
            if channel_data:
                features = torch.stack(channel_data).flatten(1).cpu().numpy()
                if features.shape[0] >= len(labels):
                    features = features[: len(labels)]
                else:
                    continue

                if probe_fn is not None:
                    preds = probe_fn(features, labels.numpy())
                else:
                    lr = LogisticRegression(max_iter=1000)
                    lr.fit(features, labels.numpy()[: len(features)])
                    preds = lr.predict(features)

                acc = accuracy_score(labels.numpy()[: len(preds)], preds)
                per_channel_acc[channel] = acc
                predictions[channel] = torch.tensor(preds)
        except Exception:
            per_channel_acc[channel] = 0.0

    avg_acc = np.mean(list(per_channel_acc.values())) if per_channel_acc else 0.0

    return MultimodalProbeResult(
        channel="multimodal",
        labels=labels,
        predictions=torch.zeros(len(labels)),
        accuracy=avg_acc,
        per_channel_accuracy=per_channel_acc,
    )


def auto_discover_concepts(
    cache: ActivationCache,
    n_concepts: int = 64,
    component: str = "z_posterior",
) -> list[Concept]:
    """Automatically discover concepts in latents using MI clustering.

    Args:
        cache: ActivationCache containing activations.
        n_concepts: Number of concepts to discover.
        component: Which component to analyze.

    Returns:
        List of discovered Concepts.
    """
    activations = []
    for t in cache.timesteps:
        try:
            act = cache[component, t]
            activations.append(act.flatten())
        except KeyError:
            pass

    if not activations:
        return []

    all_activations = torch.stack(activations).cpu().numpy()

    if all_activations.shape[0] < n_concepts:
        n_concepts = all_activations.shape[0]

    kmeans = KMeans(n_clusters=n_concepts, random_state=42, n_init=10)
    cluster_assignments = kmeans.fit_predict(all_activations)

    concepts = []
    for i in range(n_concepts):
        cluster_mask = cluster_assignments == i
        if cluster_mask.sum() > 0:
            cluster_points = all_activations[cluster_mask]
            centroid = torch.from_numpy(kmeans.cluster_centers_[i]).float()

            cluster_variance = np.var(cluster_points)
            interpretability = 1.0 / (1.0 + cluster_variance)

            concept = Concept(
                direction=centroid,
                name=f"concept_{i}",
                interpretability_score=float(interpretability),
                channel=None,
            )
            concepts.append(concept)

    return concepts


def compute_mutual_information(
    activations: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10,
) -> float:
    """Compute mutual information between activations and labels.

    Args:
        activations: Activation tensor [N, D].
        labels: Label tensor [N].
        n_bins: Number of bins for discretizing activations.

    Returns:
        Mutual information score.
    """
    act_np = activations.cpu().numpy().flatten()
    labels_np = labels.cpu().numpy().flatten()

    act_bins = np.digitize(act_np, np.histogram(act_np, bins=n_bins)[1])

    return mutual_info_score(labels_np, act_bins)


def concept_alignment_score(
    concepts: list[Concept],
    probe_directions: torch.Tensor,
) -> torch.Tensor:
    """Compute alignment between discovered concepts and probe directions.

    Args:
        concepts: List of discovered concepts.
        probe_directions: Probe direction vectors [N, D].

    Returns:
        Alignment scores [len(concepts)].
    """
    concept_dirs = torch.stack([c.direction for c in concepts])

    if concept_dirs.dim() == 1:
        concept_dirs = concept_dirs.unsqueeze(0)
    if probe_directions.dim() == 1:
        probe_directions = probe_directions.unsqueeze(0)

    similarities = torch.nn.functional.cosine_similarity(
        concept_dirs,
        probe_directions,
        dim=-1,
    )

    return similarities.abs()


def find_concept_by_name(
    concepts: list[Concept],
    name: str,
    threshold: float = 0.5,
) -> Concept | None:
    """Find a concept by name with fuzzy matching.

    Args:
        concepts: List of concepts to search.
        name: Name to search for.
        threshold: Similarity threshold for fuzzy matching.

    Returns:
        Matching concept or None.
    """
    name_lower = name.lower().replace("_", " ")

    for concept in concepts:
        concept_name = concept.name.lower().replace("_", " ")
        if name_lower in concept_name or concept_name in name_lower:
            return concept

    return None


def merge_similar_concepts(
    concepts: list[Concept],
    similarity_threshold: float = 0.9,
) -> list[Concept]:
    """Merge highly similar concepts.

    Args:
        concepts: List of concepts.
        similarity_threshold: Threshold above which concepts are merged.

    Returns:
        List of merged concepts.
    """
    if len(concepts) <= 1:
        return concepts

    merged = [concepts[0]]
    remaining = concepts[1:]

    for concept in remaining:
        is_duplicate = False
        for existing in merged:
            if concept.similarity(existing) > similarity_threshold:
                new_direction = (existing.direction + concept.direction) / 2
                existing.direction = new_direction
                existing.interpretability_score = (
                    existing.interpretability_score + concept.interpretability_score
                ) / 2
                is_duplicate = True
                break

        if not is_duplicate:
            merged.append(concept)

    return merged


class MultimodalCache:
    """Extended ActivationCache with multimodal channel support."""

    def __init__(self):
        self.cache = ActivationCache()
        self._channel_metadata: dict[str, dict[str, Any]] = {}

    def add_channel(
        self,
        channel_name: str,
        component: str,
        timestep: int,
        tensor: torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a multimodal channel activation.

        Args:
            channel_name: Name of the channel ('vision', 'proprio', etc.).
            component: Component name.
            timestep: Timestep index.
            tensor: Activation tensor.
            metadata: Optional channel metadata.
        """
        full_name = f"{channel_name}_{component}"
        self.cache[full_name, timestep] = tensor

        if channel_name not in self._channel_metadata:
            self._channel_metadata[channel_name] = {}
        self._channel_metadata[channel_name][component] = metadata or {}

    def get_channel(self, channel_name: str) -> dict[str, list[torch.Tensor]]:
        """Get all activations for a specific channel.

        Args:
            channel_name: Name of channel to retrieve.

        Returns:
            Dict mapping component -> list of tensors.
        """
        result = {}
        for (name, t), _val in self.cache.keys():
            if name.startswith(f"{channel_name}_"):
                component = name[len(f"{channel_name}_") :]
                if component not in result:
                    result[component] = []
                result[component].append(self.cache[name, t])

        return result

    def channel_names(self) -> list[str]:
        """List all available channel names."""
        return list(self._channel_metadata.keys())

    def __getitem__(self, key):
        return self.cache[key]

    def __setitem__(self, key, value):
        self.cache[key] = value

    def __contains__(self, key):
        return key in self.cache

    def keys(self):
        return self.cache.keys()

    @property
    def component_names(self) -> list[str]:
        return self.cache.component_names

    @property
    def timesteps(self) -> list[int]:
        return self.cache.timesteps

    def materialize(self, **kwargs):
        return self.cache.materialize(**kwargs)

    def estimate_memory_gb(self) -> float:
        return self.cache.estimate_memory_gb()
