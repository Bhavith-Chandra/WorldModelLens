import numpy as np
from typing import Dict, List, Tuple

def generate_ijepa_data(
    num_context_patches: int = 16,
    grid_size: Tuple[int, int] = (4, 4),
    seed: int = 42
) -> Dict:
    """
    Generates dummy attribution data for an I-JEPA model.
    
    Returns:
        Dict containing:
            'patch_coords': Dict mapping patch_id to (x, y) coordinates.
            'target_patch': ID of the target patch.
            'attributions': Dict mapping context_patch_id -> weight.
            'prediction_scores': List of MSE improvements as patches are added.
    """
    np.random.seed(seed)
    
    patch_ids = [f"patch_{i}" for i in range(num_context_patches)]
    target_patch = "target_patch"
    
    # Generate random grid coordinates for context patches
    coords = {}
    for i, pid in enumerate(patch_ids):
        coords[pid] = (i % grid_size[0], i // grid_size[1])
    
    # Place target patch slightly separate or in center
    coords[target_patch] = (grid_size[0] // 2, grid_size[1] + 1)
    
    # Random weights (some strong, some weak)
    weights = np.random.exponential(scale=1.0, size=num_context_patches)
    # Normalize a bit
    weights = weights / weights.max()
    
    attributions = {pid: float(w) for pid, w in zip(patch_ids, weights)}
    
    # Dummy prediction quality (MSE starts high, decreases as most important patches are added)
    sorted_weights = sorted(weights, reverse=True)
    base_mse = 1.0
    mse_curve = [base_mse]
    for w in sorted_weights:
        base_mse -= w * 0.05 # Simple linear-ish improvement
        mse_curve.append(max(0.05, float(base_mse)))
        
    return {
        "patch_coords": coords,
        "target_patch": target_patch,
        "attributions": attributions,
        "prediction_scores": mse_curve,
        "context_patches": patch_ids
    }

if __name__ == "__main__":
    data = generate_ijepa_data()
    print(f"Generated data for {len(data['context_patches'])} patches.")
    print(f"Target: {data['target_patch']}")
    print(f"Top attribution: {max(data['attributions'].values()):.4f}")
