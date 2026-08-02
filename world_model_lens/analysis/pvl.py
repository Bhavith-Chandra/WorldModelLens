"""Physical Variable Localization (PVL) analysis module for WorldModelLens.

Provides tools to discover physical variable direction vectors, fit linear and non-linear probes,
perform causal steering interventions, track emergence through depth, and measure cross-attention consumption.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from world_model_lens import HookedWorldModel
from experiments.pvl.subspace_discovery import (
    discover_subspaces_pca,
    discover_subspaces_ica,
    discover_subspaces_nmf,
)
from experiments.pvl.observables import (
    compute_patch_properties,
    LinearProbe,
    MLPProbe,
    train_probes_for_dataset,
)
from experiments.pvl.intervention import run_steered_inference
from experiments.pvl.analysis import (
    analyze_steering_causality,
    label_subspace,
    localize_layers,
    test_cross_attention_consumption,
)

class PhysicalVariableAnalyzer:
    """High-level analysis suite for Physical Variable Localization (PVL)."""

    def __init__(self, wm: HookedWorldModel):
        self.wm = wm

    def discover_subspaces(
        self,
        activations: torch.Tensor,
        method: str = "pca",
        n_components: int = 5
    ) -> Tuple[torch.Tensor, Any]:
        """Discovers direction vectors for a given activation tensor.
        
        Args:
            activations: Tensor [N, d_embed]
            method: 'pca', 'ica', or 'nmf'
            n_components: Number of directions to extract
        """
        method_lower = method.lower()
        if method_lower == "pca":
            return discover_subspaces_pca(activations, n_components=n_components)
        elif method_lower == "ica":
            return discover_subspaces_ica(activations, n_components=n_components)
        elif method_lower == "nmf":
            return discover_subspaces_nmf(activations, n_components=n_components)
        else:
            raise ValueError(f"Unknown subspace discovery method: {method}. Choose 'pca', 'ica', or 'nmf'.")

    def fit_probes(
        self,
        activations: torch.Tensor,
        metadata: List[Dict[str, Any]],
        image_tensors: List[torch.Tensor],
        probe_type: str = "linear",
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """Fits physical variable probes (linear or MLP) on collected latents."""
        return train_probes_for_dataset(
            activations, metadata, image_tensors, device=device, probe_type=probe_type
        )

    def analyze_steering(
        self,
        img_tensor: torch.Tensor,
        component_name: str,
        U: torch.Tensor,
        probes: Dict[str, Any],
        alphas: List[float] = [-2.0, -1.0, 0.0, 1.0, 2.0]
    ) -> Dict[str, Any]:
        """Measures causal changes in physical variable probes and identity preservation under steering."""
        responses, id_sims = analyze_steering_causality(
            self.wm, img_tensor, component_name, U, probes, alphas
        )
        dominant_var, effect_size, mean_id = label_subspace(responses, id_sims, alphas)
        return {
            "probe_responses": responses,
            "identity_similarities": id_sims,
            "dominant_var": dominant_var,
            "effect_size": effect_size,
            "mean_identity_preservation": mean_id,
        }

    def localize_depth_emergence(
        self,
        images: List[Tuple[str, str]],
        metadata: List[Dict[str, Any]],
        image_tensors: List[torch.Tensor],
        layers_to_check: List[str],
        device: str = "cpu"
    ) -> Dict[str, Dict[str, float]]:
        """Tracks physical variable R^2 probe scores across layer depth."""
        return localize_layers(
            self.wm, images, metadata, image_tensors, layers_to_check, device=device
        )

    def evaluate_cross_attention_consumption(
        self,
        img_tensor: torch.Tensor,
        U: torch.Tensor,
        predictor_layer: str,
        device: str = "cpu"
    ) -> Dict[str, float]:
        """Evaluates KL shift and MSE degradation when projecting out subspace U before cross-attention."""
        return test_cross_attention_consumption(
            self.wm, img_tensor, U, predictor_layer, device=device
        )
