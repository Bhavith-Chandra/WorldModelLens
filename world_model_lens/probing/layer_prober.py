from __future__ import annotations
"""Probing layer analyzer - analyze which layers encode which concepts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import torch
import numpy as np


@dataclass
class LayerProbingResult:
    """Result of layer-wise probing."""

    layer: str
    concept: str
    accuracy: float
    f1_score: float
    weights: torch.Tensor
    bias: torch.Tensor | None = None


class LayerProber:
    """Probe multiple layers for concept encoding."""

    def __init__(self, wm: Any):
        """Initialize layer prober.

        Args:
            wm: HookedWorldModel instance
        """
        self.wm = wm
        self.results: dict[str, LayerProbingResult] = {}

    def probe_layer(
        self,
        cache: Any,
        layer: str,
        labels: torch.Tensor,
        test_size: float = 0.2,
    ) -> LayerProbingResult:
        """Probe a specific layer for concepts.

        Args:
            cache: ActivationCache
            layer: Layer name
            labels: Concept labels
            test_size: Test split fraction

        Returns:
            LayerProbingResult
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        activations = cache[layer]

        if activations.dim() > 2:
            activations = activations.reshape(activations.shape[0], -1)

        X = activations.cpu().numpy()
        y = labels.cpu().numpy()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)

        result = LayerProbingResult(
            layer=layer,
            concept="unknown",
            accuracy=accuracy,
            f1_score=accuracy,
            weights=torch.from_numpy(model.coef_),
            bias=torch.from_numpy(model.bias_) if hasattr(model, "bias_") else None,
        )

        self.results[layer] = result
        return result

    def probe_all_layers(
        self,
        cache: Any,
        labels: torch.Tensor,
    ) -> dict[str, LayerProbingResult]:
        """Probe all cached layers.

        Args:
            cache: ActivationCache
            labels: Concept labels

        Returns:
            Dictionary of results per layer
        """
        results = {}

        for layer in cache.component_names:
            try:
                result = self.probe_layer(cache, layer, labels)
                results[layer] = result
            except Exception:
                continue

        return results

    def find_best_layer(
        self,
        cache: Any,
        labels: torch.Tensor,
    ) -> tuple[str, LayerProbingResult]:
        """Find the layer with best concept encoding.

        Args:
            cache: ActivationCache
            labels: Concept labels

        Returns:
            Tuple of (layer_name, result)
        """
        results = self.probe_all_layers(cache, labels)

        best_layer = max(results.items(), key=lambda x: x[1].accuracy)
        return best_layer


__all__ = ["LayerProber", "LayerProbingResult"]
