import torch
import numpy as np
from PIL import Image
import urllib.request
import io
from typing import Tuple, List

def get_sample_image(url: str = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg") -> Image.Image:
    """Downloads a sample image from the web."""
    try:
        with urllib.request.urlopen(url) as response:
            data = response.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        print(f"Failed to download image: {e}. Generating synthetic image.")
        # Fallback: Generate a synthetic image with some patterns
        img_array = np.zeros((224, 224, 3), dtype=np.uint8)
        for i in range(224):
            for j in range(224):
                img_array[i, j] = [i % 255, j % 255, (i+j) % 255]
        return Image.fromarray(img_array)

def preprocess_image(img: Image.Image, size: int = 224) -> torch.Tensor:
    """Resizes and normalizes image to [1, 3, size, size]."""
    img = img.resize((size, size))
    img_array = np.array(img).astype(np.float32) / 255.0
    # Mean/std normalization (ImageNet style) - explicitly float32
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float()
    return tensor

def get_ijepa_masks(
    num_patches: int = 196, 
    num_context: int = 40, 
    num_target: int = 10
) -> Tuple[List[int], List[int]]:
    """
    Randomly samples context and target indices.
    In actual I-JEPA, these would be spatially contiguous blocks.
    """
    indices = np.random.permutation(num_patches)
    context_ids = sorted(indices[:num_context].tolist())
    target_ids = sorted(indices[num_context:num_context+num_target].tolist())
    return context_ids, target_ids

def patchify(img_tensor: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    """Breaks img [1, 3, H, W] into [N, 3, P, P] patches."""
    p = patch_size
    # [1, 3, H, W] -> [H/P, W/P, 3, P, P]
    patches = img_tensor.unfold(2, p, p).unfold(3, p, p)
    patches = patches.permute(2, 3, 1, 4, 5).reshape(-1, 3, p, p)
    return patches
