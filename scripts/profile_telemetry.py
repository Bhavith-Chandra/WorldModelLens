"""Telemetry Overhead Benchmarking

Benchmarks the computational and memory overhead of the HookedWorldModel
framework against a bare adapter forward pass.
"""

import os
import sys
import gc
import time
import torch
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))

from world_model_lens import HookedWorldModel
from world_model_lens.backends.ijepa_adapter import IJEPAAdapter
from world_model_lens.core.config import WorldModelConfig
from examples.ijepa.image_utils import get_ijepa_masks

def clear_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

def profile_function(name, func, num_iters=20, warmup=5):
    """Profile a function, returning (ms_per_step, peak_memory_mb)."""
    # Warmup
    for _ in range(warmup):
        func()
        
    clear_cache()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    start_time = time.perf_counter()
    
    for _ in range(num_iters):
        func()
        
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak_mem = 0.0
        
    end_time = time.perf_counter()
    ms_per_step = ((end_time - start_time) / num_iters) * 1000
    
    return ms_per_step, peak_mem

def main():
    parser = argparse.ArgumentParser(description="Profile HookedWorldModel overhead.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for profiling.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--weights", type=str, default="examples/ijepa/ijepa_mini.pth")
    args = parser.parse_args()

    print(f"Profiling telemetry on {args.device} with batch_size {args.batch_size}...")

    # Load Adapter
    config = WorldModelConfig(backend="ijepa", d_embed=192, n_layers=12, n_heads=12)
    
    # Try to load exact weights if available, else random is fine for profiling
    adapter = IJEPAAdapter(config)
    if os.path.exists(args.weights):
        adapter.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True), strict=False)
    adapter.to(device=args.device)
    adapter.eval()
    
    wm = HookedWorldModel(adapter, config)
    
    # Dummy batch
    x = torch.randn(args.batch_size, 3, 224, 224, device=args.device)
    
    # Context/Target ids
    context_ids_list = []
    target_ids_list = []
    for _ in range(args.batch_size):
        c_ids, t_ids = get_ijepa_masks()
        context_ids_list.append(c_ids)
        target_ids_list.append(t_ids)
    
    kwargs = {
        "context_ids": context_ids_list,
        "target_ids": target_ids_list
    }
    
    with torch.no_grad():
        # Baseline
        print("Profiling Baseline...")
        # Make sure no hooks are attached to adapter
        adapter.hooks = None
        adapter.last_context_ids = context_ids_list[0]
        adapter.last_target_ids = target_ids_list[0]
        
        def run_adapter():
            h, _ = adapter.encode(x)
            adapter.dynamics(h)

        base_ms, base_mem = profile_function(
            "Baseline", 
            run_adapter
        )
        
        # Empty Hooks
        print("Profiling Empty Hooks...")
        adapter.hooks = wm._hooks
        empty_ms, empty_mem = profile_function(
            "Empty Hooks", 
            run_adapter
        )
        
        # Heavy Hooks (mocking run_with_cache by attaching many empty hooks)
        print("Profiling Heavy Hooks...")
        heavy_ms, heavy_mem = profile_function(
            "Heavy Hooks", 
            lambda: wm.run_with_cache(x)
        )

    # Calculate overhead
    empty_overhead_ms = ((empty_ms - base_ms) / base_ms) * 100
    heavy_overhead_ms = ((heavy_ms - base_ms) / base_ms) * 100
    
    empty_mem_delta = empty_mem - base_mem
    heavy_mem_delta = heavy_mem - base_mem

    # Format Markdown Table
    print("\n# Telemetry Overhead Profiling Results")
    print(f"*Batch Size: {args.batch_size} | Device: {args.device}*")
    print("\n| Configuration | MS / Step | Overhead % | Peak VRAM (MB) | VRAM Delta (MB) |")
    print("|--------------|-----------|------------|----------------|-----------------|")
    print(f"| **Baseline (Bare Adapter)** | {base_ms:.1f} ms | - | {base_mem:.1f} | - |")
    print(f"| **Empty Hooks (HookedWorldModel)** | {empty_ms:.1f} ms | +{empty_overhead_ms:.1f}% | {empty_mem:.1f} | +{empty_mem_delta:.1f} |")
    print(f"| **Heavy Hooks (run_with_cache)** | {heavy_ms:.1f} ms | +{heavy_overhead_ms:.1f}% | {heavy_mem:.1f} | +{heavy_mem_delta:.1f} |")

if __name__ == "__main__":
    main()
