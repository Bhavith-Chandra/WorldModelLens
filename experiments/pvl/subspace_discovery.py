import torch
import numpy as np
from typing import Dict, List, Tuple, Any

def discover_subspaces_pca(
    activations: torch.Tensor,
    n_components: int = 10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Discover candidate direction vectors using Principal Component Analysis (PCA).
    
    Returns:
        components: Tensor of shape [n_components, d_embed] containing PC directions (unit vectors)
        explained_variance: Tensor of shape [n_components]
    """
    # Center activations
    mean = activations.mean(dim=0, keepdim=True)
    centered = activations - mean
    
    # Run SVD-based PCA
    # U: [N, q], S: [q], V: [dim, q]
    q = min(n_components, centered.shape[1])
    U, S, V = torch.pca_lowrank(centered, q=q, center=False)
    
    # Compute explained variance ratio
    total_var = (centered ** 2).sum()
    explained_var = S**2
    explained_var_ratio = explained_var / (total_var + 1e-9)
    
    # V is [dim, n_components] where columns are the basis vectors
    components = V.t() # [n_components, dim]
    
    # Normalize each component to unit length (just to be safe)
    components = components / (torch.norm(components, dim=1, keepdim=True) + 1e-9)
    
    return components, explained_var_ratio

def discover_subspaces_ica(
    activations: torch.Tensor,
    n_components: int = 10
) -> Tuple[torch.Tensor, Any]:
    """Discover candidate direction vectors using Independent Component Analysis (ICA)."""
    try:
        from sklearn.decomposition import FastICA
    except ImportError:
        print("scikit-learn is not installed. FastICA is unavailable. Falling back to PCA.")
        return discover_subspaces_pca(activations, n_components)[0], None
        
    X = activations.cpu().numpy()
    ica = FastICA(n_components=n_components, random_state=42, max_iter=1000)
    # Fit ICA and transform
    ica.fit(X)
    
    # ica.components_ contains independent components of shape [n_components, d_embed]
    components = torch.from_numpy(ica.components_).float()
    
    # Normalize components to unit vectors
    components = components / (torch.norm(components, dim=1, keepdim=True) + 1e-9)
    
    return components, ica

def discover_subspaces_nmf(
    activations: torch.Tensor,
    n_components: int = 10
) -> Tuple[torch.Tensor, Any]:
    """Discover candidate direction vectors using Non-negative Matrix Factorization (NMF).
    
    Since activations can be negative, we apply a ReLU operation before fitting NMF.
    """
    try:
        from sklearn.decomposition import NMF
    except ImportError:
        print("scikit-learn is not installed. NMF is unavailable. Falling back to PCA.")
        return discover_subspaces_pca(activations, n_components)[0], None
        
    # NMF requires non-negative inputs
    X = torch.clamp(activations, min=0.0).cpu().numpy()
    
    nmf = NMF(n_components=n_components, init='random', random_state=42, max_iter=1000)
    nmf.fit(X)
    
    # nmf.components_ contains NMF basis vectors of shape [n_components, d_embed]
    components = torch.from_numpy(nmf.components_).float()
    
    # Normalize components to unit vectors
    components = components / (torch.norm(components, dim=1, keepdim=True) + 1e-9)
    
    return components, nmf
