import torch
import numpy as np
from typing import Dict, Any, List

try:
    import cv2
except ImportError:
    cv2 = None


class ImagePropertyAnalyzer:
    """Analyzes image properties to correlate with causal/attention failure modes.
    
    Extracts heuristics like texture complexity, contrast, and patch-specific features
    to understand why attention inversion occurs on specific samples.
    """
    
    def __init__(self):
        pass

    def compute_properties(self, img_tensor: torch.Tensor, target_id: int, patch_size: int = 16) -> Dict[str, float]:
        """Compute heuristics for an image tensor.
        
        Args:
            img_tensor: [1, 3, H, W] normalized image tensor.
            target_id: ID of the target patch.
            patch_size: Size of the patch (default 16).
            
        Returns:
            Dictionary of property scores.
        """
        # Convert tensor back to roughly 0-255 numpy array for standard CV operations
        img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_np = ((img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)) * 255
        img_np = img_np.astype(np.uint8)
        
        properties = {}
        
        if cv2 is not None:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
            # 1. Texture Complexity (Laplacian Variance)
            properties["laplacian_variance"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            
            # 2. Overall Contrast (RMS contrast is roughly std dev of pixel intensities)
            properties["rms_contrast"] = float(gray.std())
            
            # 3. Target Patch Context Richness
            h_patches = gray.shape[0] // patch_size
            w_patches = gray.shape[1] // patch_size
            
            # Convert flat target_id back to 2D
            ty = target_id // w_patches
            tx = target_id % w_patches
            
            # Extract target patch
            py_start = ty * patch_size
            px_start = tx * patch_size
            target_patch = gray[py_start:py_start+patch_size, px_start:px_start+patch_size]
            
            properties["target_patch_std"] = float(target_patch.std())
            properties["target_patch_mean"] = float(target_patch.mean())
            
            # Edge density in target patch
            edges = cv2.Canny(target_patch, 100, 200)
            properties["target_edge_density"] = float(np.count_nonzero(edges) / (patch_size * patch_size))
            
        else:
            # Fallback to simple torch-based metrics if cv2 is not available
            gray = img_tensor.mean(dim=1).squeeze(0) # [H, W]
            
            # 1. Texture Complexity (Laplacian Variance via Conv2d)
            laplacian_kernel = torch.tensor([
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]
            ], dtype=torch.float32, device=img_tensor.device).unsqueeze(0).unsqueeze(0)
            
            laplacian = torch.nn.functional.conv2d(
                gray.unsqueeze(0).unsqueeze(0), 
                laplacian_kernel, 
                padding=1
            )
            properties["laplacian_variance"] = float(laplacian.var().item() * (255**2))
            
            # 2. Overall Contrast
            properties["rms_contrast"] = float(gray.std().item() * 255.0)
            
            h_patches = gray.shape[0] // patch_size
            w_patches = gray.shape[1] // patch_size
            ty = target_id // w_patches
            tx = target_id % w_patches
            
            py_start = ty * patch_size
            px_start = tx * patch_size
            target_patch = gray[py_start:py_start+patch_size, px_start:px_start+patch_size]
            
            # 3. Target Patch Context Richness
            properties["target_patch_std"] = float(target_patch.std().item() * 255.0)
            properties["target_patch_mean"] = float(target_patch.mean().item() * 255.0)
            
            # Target edge density via thresholded gradient
            grad_x = target_patch[:, 1:] - target_patch[:, :-1]
            grad_y = target_patch[1:, :] - target_patch[:-1, :]
            grad_mag = torch.zeros_like(target_patch)
            grad_mag[:-1, :-1] = torch.sqrt(grad_x[:-1, :]**2 + grad_y[:, :-1]**2)
            
            edges = (grad_mag > 0.2).float() # Threshold roughly equivalent to Canny
            properties["target_edge_density"] = float(edges.mean().item())

        return properties
