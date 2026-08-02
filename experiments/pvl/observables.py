import torch
import numpy as np
from typing import Dict, List, Any, Tuple
from PIL import Image

def get_patch_pixels(img_tensor: torch.Tensor, patch_idx: int, patch_size: int = 16) -> torch.Tensor:
    """Extracts a 16x16 patch from the preprocessed image tensor.
    
    Args:
        img_tensor: [1, 3, H, W] or [3, H, W]
    """
    if img_tensor.dim() == 4:
        img_tensor = img_tensor.squeeze(0) # [3, H, W]
        
    grid_size = img_tensor.shape[1] // patch_size
    r = patch_idx // grid_size
    c = patch_idx % grid_size
    
    patch = img_tensor[:, r*patch_size : (r+1)*patch_size, c*patch_size : (c+1)*patch_size]
    return patch

def compute_patch_properties(img_tensor: torch.Tensor, patch_idx: int) -> Dict[str, float]:
    """Computes physical properties directly from patch pixels."""
    if img_tensor.dim() == 4:
        img_tensor = img_tensor.squeeze(0)
    patch = get_patch_pixels(img_tensor, patch_idx)
    
    # 1. Brightness: Mean pixel value across channels
    brightness = float(patch.mean().item())
    
    # 2. Contrast: Std dev of pixel values
    contrast = float(patch.std().item())
    
    # 3. Complexity: Laplacian variance proxy (high-freq spatial derivative)
    # Simple laplacian kernel
    lap_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=patch.device).reshape(1, 1, 3, 3)
    # Apply to each channel
    lap_vars = []
    for ch in range(3):
        ch_pixels = patch[ch].unsqueeze(0).unsqueeze(0) # [1, 1, 16, 16]
        lap = torch.nn.functional.conv2d(ch_pixels, lap_kernel, padding=1)
        lap_vars.append(lap.var().item())
    complexity = float(np.mean(lap_vars))
    
    # 4. Position: coordinates
    grid_size = img_tensor.shape[1] // 16
    grid_y = patch_idx // grid_size
    grid_x = patch_idx % grid_size
    
    # 5. Radial distance & spatial aspect proxy
    center = (grid_size - 1) / 2.0
    norm_y = (grid_y - center) / max(center, 1.0)
    norm_x = (grid_x - center) / max(center, 1.0)
    radial_distance = float(np.sqrt(norm_y**2 + norm_x**2))
    aspect_ratio_proxy = float(abs(norm_x) / (abs(norm_y) + 1e-5))
    
    return {
        "brightness": brightness,
        "contrast": contrast,
        "complexity": complexity,
        "grid_y": float(grid_y),
        "grid_x": float(grid_x),
        "radial_distance": radial_distance,
        "aspect_ratio_proxy": aspect_ratio_proxy
    }

class LinearProbe:
    """Simple linear probe to map latents to physical variables."""
    def __init__(self):
        self.w = None
        self.b = None
        
    def fit(self, X: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
        """Fits a Ridge regression probe."""
        # y is [N]
        # X is [N, dim]
        N, dim = X.shape
        X_np = X.cpu().numpy()
        y_np = y.cpu().numpy()
        
        # Add bias column
        X_bias = np.hstack([X_np, np.ones((N, 1))])
        
        # Ridge regression closed-form
        I = np.eye(dim + 1)
        I[-1, -1] = 0.0 # Don't regularize bias
        
        # Solve (X^T X + alpha*I) beta = X^T y
        beta = np.linalg.solve(X_bias.T @ X_bias + alpha * I, X_bias.T @ y_np)
        
        self.w = torch.from_numpy(beta[:-1]).float().to(X.device)
        self.b = float(beta[-1])
        
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict physical variable from latents."""
        w_device = self.w.to(X.device)
        return torch.matmul(X, w_device) + self.b

class MLPProbe(torch.nn.Module):
    """2-layer non-linear MLP probe to map latents to physical variables."""
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net = None
        self.is_fitted = False

    def fit(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 100, lr: float = 1e-3):
        """Fits an MLP regression probe via PyTorch Adam optimizer."""
        N, dim = X.shape
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, 1)
        ).to(X.device)
        
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-4)
        criterion = torch.nn.MSELoss()
        
        y_col = y.unsqueeze(1) if y.dim() == 1 else y
        
        self.net.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            out = self.net(X)
            loss = criterion(out, y_col)
            loss.backward()
            optimizer.step()
            
        self.net.eval()
        self.is_fitted = True

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict physical variable from latents."""
        if not self.is_fitted or self.net is None:
            raise RuntimeError("MLPProbe must be fitted before predict() is called.")
        self.net.eval()
        with torch.no_grad():
            out = self.net(X.to(next(self.net.parameters()).device))
        return out.squeeze(-1)

def train_probes_for_dataset(
    activations: torch.Tensor,
    metadata: List[Dict[str, Any]],
    image_tensors: List[torch.Tensor],
    device: str = "cpu",
    probe_type: str = "linear"
) -> Dict[str, Any]:
    """Train probes (linear or MLP) for each physical variable on the dataset."""
    # Compute ground truth targets for all tokens in metadata
    targets = {
        "brightness": [],
        "contrast": [],
        "complexity": [],
        "grid_y": [],
        "grid_x": [],
        "radial_distance": [],
        "aspect_ratio_proxy": []
    }
    
    # Fill target values based on token metadata and image patches
    for m in metadata:
        img_idx = m["image_idx"]
        patch_idx = m["patch_idx"]
        img_t = image_tensors[img_idx]
        
        props = compute_patch_properties(img_t, patch_idx)
        for k in targets.keys():
            targets[k].append(props[k])
            
    probes = {}
    X = activations.to(device)
    for k, v in targets.items():
        y = torch.tensor(v, dtype=torch.float32, device=device)
        if probe_type == "mlp":
            probe = MLPProbe()
            probe.fit(X, y)
        else:
            probe = LinearProbe()
            probe.fit(X, y)
        probes[k] = probe
        print(f"Trained {probe_type} probe for '{k}'")
        
    return probes

