import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import json
import math
from PIL import Image

# Ensure local library is on path
sys.path.insert(0, os.path.abspath("."))

from world_model_lens import HookedWorldModel
from world_model_lens.backends.ijepa_adapter import IJEPAAdapter
from world_model_lens.core.config import WorldModelConfig
from world_model_lens.hub.model_hub import ModelHub
from examples.ijepa.image_utils import preprocess_image

# Import evaluators
from world_model_lens.patching.positional_patching import PositionalPatchingEvaluator
from world_model_lens.analysis.mlp_ablation import MLPBottleneckAblator
from world_model_lens.analysis.attention_routing_ablation import AttentionRoutingAblator
from world_model_lens.analysis.predictor_attention_ablation import PredictorAttentionAblator

def main():
    print("="*60)
    print("   RUNNING CAUSAL DISSECTION EXPERIMENTS ON I-JEPA ViT-H/14")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load ViT-H/14
    checkpoint_path = "vith14_in1k_ep300.pth.tar"
    if not os.path.exists(checkpoint_path):
        print(f"CRITICAL: Checkpoint {checkpoint_path} not found in root directory!")
        print("Falling back to mini configuration for testing (ijepa_mini.pth)...")
        checkpoint_path = "ijepa_mini.pth"
        if not os.path.exists(checkpoint_path):
            print("CRITICAL: Mini checkpoint not found either. Exiting.")
            sys.exit(1)
        config = WorldModelConfig(backend="ijepa", d_embed=192, n_layers=6, n_heads=3, predictor_embed_dim=384)
        adapter = IJEPAAdapter(config)
        adapter.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True), strict=False)
        adapter = adapter.to(device)
    else:
        print(f"Loading official Meta ViT-H/14 from '{checkpoint_path}'...")
        adapter = ModelHub._load_ijepa(checkpoint_path, device=device)

    adapter.eval()
    wm = HookedWorldModel(adapter, adapter.config)

    # Inferred dimensions
    n_patches = adapter.context_encoder.patch_embed.n_patches
    grid_size = int(math.sqrt(n_patches))
    print(f"Model dimensions: {grid_size}x{grid_size} grid = {n_patches} patches.")
    print(f"Context Encoder Depth: {len(adapter.context_encoder.blocks)} layers.")
    print(f"Predictor Depth: {len(adapter.predictor.blocks)} layers.")

    # Load dataset: 1 image per category from data/eval_dataset
    print("\nLoading dataset images...")
    image_paths = []
    data_dir = "data/eval_dataset"
    if os.path.exists(data_dir):
        for root, dirs, files in os.walk(data_dir):
            cat_files = [os.path.join(root, f) for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
            if cat_files:
                image_paths.append(cat_files[0]) # pick one image per category
    
    if not image_paths:
        print("Warning: No local dataset images found in data/eval_dataset. Using synthetic dummy image.")
        # Create a dummy image
        img_pil = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        dummy_path = "data/dummy_test.png"
        os.makedirs("data", exist_ok=True)
        img_pil.save(dummy_path)
        image_paths = [dummy_path]

    print(f"Loaded {len(image_paths)} images across categories for testing.")
    if device == "cpu":
        print("Running on CPU: capping to 3 images to complete in under 2 minutes.")
        image_paths = image_paths[:3]

    # Define Target vs Context splits dynamically based on grid_size
    # Core target is center 35%
    core_start = int(grid_size * 0.3)
    core_end = int(grid_size * 0.7)
    core_ids = []
    for r in range(core_start, core_end):
        for c in range(core_start, core_end):
            core_ids.append(r * grid_size + c)

    # Background is boundary region (outer 15%)
    border = max(1, int(grid_size * 0.15))
    bg_ids = []
    for r in range(grid_size):
        for c in range(grid_size):
            if r < border or r >= grid_size - border or c < border or c >= grid_size - border:
                bg_ids.append(r * grid_size + c)

    print(f"Dynamic mask definition:")
    print(f"  Core Target Patches: {len(core_ids)}")
    print(f"  Background Patches: {len(bg_ids)}")

    # Setup stages for Context Encoder (32 layers for ViT-H, 6 or 12 layers for mini)
    ce_max_layer = len(adapter.context_encoder.blocks) - 1
    ce_quarter = max(1, (ce_max_layer + 1) // 4)
    ce_stages = {
        "Early Stages": list(range(0, ce_quarter)),
        "Middle Stages": list(range(ce_quarter, 3 * ce_quarter)),
        "Late Stages": list(range(3 * ce_quarter, ce_max_layer + 1)),
        "All Stages": list(range(0, ce_max_layer + 1)),
    }

    # Setup stages for Predictor (typically 4 or 12 layers)
    pred_max_layer = len(adapter.predictor.blocks) - 1
    pred_stages = {
        "Early Stages": list(range(0, max(1, (pred_max_layer + 1) // 2))),
        "Late Stages": list(range(max(1, (pred_max_layer + 1) // 2), pred_max_layer + 1)),
        "All Stages": list(range(0, pred_max_layer + 1)),
    }

    # Instantiate evaluators
    eval_rq1 = PositionalPatchingEvaluator(wm)
    eval_mlp = MLPBottleneckAblator(wm)
    eval_attn_routing = AttentionRoutingAblator(wm)
    eval_pred_attn = PredictorAttentionAblator(wm)

    # Containers for results
    rq1_cross_mses = []
    rq1_orig_mses = []
    rq1_successes = []

    rq5_mlp_results = {stage: [] for stage in ce_stages.keys()}
    rq5_attn_results = {stage: [] for stage in ce_stages.keys()}
    rq5_pred_results = {stage: [] for stage in pred_stages.keys()}

    print("\nProcessing images...")
    for idx, path in enumerate(image_paths):
        print(f"[{idx+1}/{len(image_paths)}] Processing {os.path.basename(path)}...")
        try:
            raw_img = Image.open(path).convert("RGB")
            img_tensor = preprocess_image(raw_img).to(device)

            # Sample context patches dynamically (20% of bg_ids, approx 40 patches)
            np.random.seed(idx)
            context_ids = np.random.choice(bg_ids, size=min(40, len(bg_ids)), replace=False).tolist()
            eval_bg_ids = [idx for idx in bg_ids if idx not in context_ids][:len(core_ids)]

            # ----------------------------------------------------
            # Experiment 1: RQ1 - Positional Counterfactual Patching
            # ----------------------------------------------------
            # Pick two target IDs from core
            if len(core_ids) >= 2:
                tid1, tid2 = core_ids[0], core_ids[-1]
                res_rq1 = eval_rq1.evaluate_swap(img_tensor, context_ids, tid1, tid2)
                rq1_cross_mses.append(res_rq1["cross_mse"])
                rq1_orig_mses.append(res_rq1["orig_mse"])
                rq1_successes.append(float(res_rq1["success"]))

            # ----------------------------------------------------
            # Experiment 2: RQ5 - CE MLP Ablation
            # ----------------------------------------------------
            for stage_name, layers in ce_stages.items():
                if layers:
                    res = eval_mlp.evaluate_ablation(img_tensor, core_ids, eval_bg_ids, context_ids, target_layers=layers)
                    rq5_mlp_results[stage_name].append(res["core_degradation"])

            # ----------------------------------------------------
            # Experiment 3: RQ5 - CE Attention Routing Blockade
            # ----------------------------------------------------
            for stage_name, layers in ce_stages.items():
                if layers:
                    res = eval_attn_routing.evaluate_ablation(img_tensor, core_ids, eval_bg_ids, context_ids, target_layers=layers)
                    rq5_attn_results[stage_name].append(res["core_degradation"])

            # ----------------------------------------------------
            # Experiment 4: RQ5 - Predictor Cross-Attention Blockade
            # ----------------------------------------------------
            for stage_name, layers in pred_stages.items():
                if layers:
                    res = eval_pred_attn.evaluate_ablation(
                        img_tensor, core_ids, eval_bg_ids, context_ids, target_layers=layers, ablation_type="cross_block"
                    )
                    rq5_pred_results[stage_name].append(res["core_degradation"])

        except Exception as e:
            print(f"Error processing image {path}: {e}")
            import traceback
            traceback.print_exc()

    # Calculate statistics
    avg_rq1_cross = np.mean(rq1_cross_mses) if rq1_cross_mses else 0.0
    avg_rq1_orig = np.mean(rq1_orig_mses) if rq1_orig_mses else 0.0
    avg_rq1_success = np.mean(rq1_successes) if rq1_successes else 0.0

    print("\n" + "="*50)
    print("                 NUMERICAL RESULTS")
    print("="*50)
    print("RQ1: Positional Counterfactual Patching")
    print(f"  MSE vs. Swapped Position (target identity swapped): {avg_rq1_cross:.6f}")
    print(f"  MSE vs. Original Position (control):                {avg_rq1_orig:.6f}")
    print(f"  Routing Swap Success Rate:                          {avg_rq1_success * 100:.1f}%")

    print("\nRQ5: Sublayer Ablations (Core MSE Degradation)")
    print(f"{'Stage':<20} | {'CE MLP':<12} | {'CE Attention':<12} | {'Predictor Cross-Attn':<20}")
    print("-"*75)
    
    stages_to_print = ["Early Stages", "Middle Stages", "Late Stages", "All Stages"]
    for s in stages_to_print:
        mlp_val = np.mean(rq5_mlp_results[s]) if rq5_mlp_results[s] else 0.0
        attn_val = np.mean(rq5_attn_results[s]) if rq5_attn_results[s] else 0.0
        
        # Predictor doesn't have "Middle Stages"
        if s in rq5_pred_results:
            pred_val = np.mean(rq5_pred_results[s])
            pred_str = f"{pred_val:+.6f}"
        else:
            pred_str = "N/A"
            
        print(f"{s:<20} | {mlp_val:+.6f} | {attn_val:+.6f} | {pred_str:<20}")

    # Generate Plots
    try:
        import matplotlib.pyplot as plt

        # Plot 1: RQ1 - Positional Patching Bar Chart
        plt.figure(figsize=(6, 5))
        bars = plt.bar(
            ["Control\n(Original Position)", "Counterfactual\n(Position Swapped)"],
            [avg_rq1_orig, avg_rq1_cross],
            color=["#d9534f", "#5cb85c"],
            edgecolor="black",
            width=0.5
        )
        plt.title("RQ1: Target Token Positional Counterfactual Patching", fontsize=12, fontweight="bold", pad=15)
        plt.ylabel("Mean Predictor MSE vs. Target", fontsize=10)
        plt.grid(True, axis="y", alpha=0.3, linestyle="--")
        
        # Add labels on top of bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.0001, f"{yval:.5f}", ha='center', va='bottom', fontweight="bold")

        plt.tight_layout()
        plot1_path = "rq1_positional_patching_vith14.png"
        plt.savefig(plot1_path, dpi=150)
        print(f"\nSaved RQ1 plot to '{plot1_path}'")
        plt.close()

        # Plot 2: Grouped Bar Chart of Sublayer Ablation Degradations
        plt.figure(figsize=(10, 6))
        
        stages_plot = ["Early Stages", "Middle Stages", "Late Stages", "All Stages"]
        x = np.arange(len(stages_plot))
        width = 0.25

        mlp_degs = [np.mean(rq5_mlp_results[s]) for s in stages_plot]
        attn_degs = [np.mean(rq5_attn_results[s]) for s in stages_plot]
        
        # We mapped pred stages: Early -> Early, Late -> Late, All -> All. 
        # For Middle, we put 0 or None since there are only 4 blocks in the predictor.
        pred_degs = []
        for s in stages_plot:
            if s in rq5_pred_results:
                pred_degs.append(np.mean(rq5_pred_results[s]))
            else:
                pred_degs.append(0.0) # 0 for Middle Stages since predictor only has 4 layers (no middle quarter)

        rects1 = plt.bar(x - width, mlp_degs, width, label="Context Encoder MLPs", color="#428bca", edgecolor="black")
        rects2 = plt.bar(x, attn_degs, width, label="Context Encoder Attention", color="#f0ad4e", edgecolor="black")
        rects3 = plt.bar(x + width, pred_degs, width, label="Predictor Cross-Attention", color="#d9534f", edgecolor="black")

        plt.axhline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)
        plt.title("RQ5: Core Target MSE Degradation under Sublayer Ablations", fontsize=14, fontweight="bold", pad=15)
        plt.ylabel("MSE Degradation (Ablated - Clean)", fontsize=11)
        plt.xticks(x, stages_plot, fontsize=10)
        plt.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=10)
        plt.grid(True, axis="y", alpha=0.3, linestyle="--")

        # Add values on top/bottom of bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                if abs(height) < 1e-6:
                    continue
                va_dir = 'bottom' if height >= 0 else 'top'
                offset = 0.001 if height >= 0 else -0.001
                plt.text(rect.get_x() + rect.get_width()/2.0, height + offset,
                         f"{height:+.4f}", ha='center', va=va_dir, fontsize=8, fontweight="bold")

        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)

        plt.tight_layout()
        plot2_path = "rq5_sublayer_ablations_vith14.png"
        plt.savefig(plot2_path, dpi=150)
        print(f"Saved RQ5 plot to '{plot2_path}'")
        plt.close()

    except Exception as e:
        print(f"Error generating plots: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
