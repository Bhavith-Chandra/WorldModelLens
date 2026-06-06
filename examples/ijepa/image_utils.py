import os
import torch
import numpy as np
from PIL import Image
import urllib.request
import io
from typing import Tuple, List

def get_sample_image(url_or_path: str = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg") -> Image.Image:
    """Loads a sample image from a URL or local path."""
    # Check if it's a local file first
    if os.path.exists(url_or_path):
        try:
            return Image.open(url_or_path).convert("RGB")
        except Exception as e:
            print(f"Failed to load local image {url_or_path}: {e}")

    # Otherwise, download from URL and cache it
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample_images")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create a safe filename from the URL
    safe_name = "".join(c for c in url_or_path.split("/")[-1] if c.isalnum() or c in "._-")
    if not safe_name:
        safe_name = str(abs(hash(url_or_path))) + ".jpg"
        
    cache_path = os.path.join(cache_dir, safe_name)
    
    # If we already have it, load from cache
    if os.path.exists(cache_path):
        try:
            return Image.open(cache_path).convert("RGB")
        except Exception:
            pass # Try downloading again if cache is corrupt
            
    try:
        import requests
        # Wikipedia and some other sites require a descriptive User-Agent to avoid 403 Forbidden
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        }
        response = requests.get(url_or_path, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Save to cache
        with open(cache_path, "wb") as f:
            f.write(response.content)
            
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"Failed to download image from {url_or_path}: {e}. Generating synthetic image.")
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
