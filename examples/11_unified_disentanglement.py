"""Example 11: Unified Disentanglement Evaluation Suite.

This example demonstrates:
1. Using the DisentanglementEvaluationSuite for multi-component evaluation
2. Computing disentanglement metrics across context_encoder, predictor, target_encoder
3. Comparing factor isolation across different model components
"""

import pathlib

import torch
import numpy as np

OUTPUT_DIR = pathlib.Path("assets/examples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from world_model_lens import HookedWorldModel, WorldModelConfig
from world_model_lens.backends.dreamerv3 import DreamerV3Adapter
from world_model_lens.analysis.metrics import DisentanglementEvaluationSuite


def main():
    print("=" * 70)
    print("World Model Lens - Unified Disentanglement Evaluation Suite")
    print("=" * 70)

    # For demonstration, we'll use DreamerV3 (which has similar components)
    # In a real IJEPA setup, you'd use the IJEPA backend
    cfg = WorldModelConfig(d_h=128, n_cat=16, n_cls=16, d_action=4, d_obs=12288)
    adapter = DreamerV3Adapter(cfg)
    wm = HookedWorldModel(adapter=adapter, config=cfg)
    analyzer = DisentanglementEvaluationSuite()

    print("\n[1] Collecting activations from model run...")

    obs_seq = torch.randn(50, 3, 64, 64)
    action_seq = torch.randn(50, cfg.d_action)

    traj, cache = wm.run_with_cache(obs_seq, action_seq)

    print("\n[2] Creating synthetic ground truth factors...")

    # Simulate factors that should be disentangled
    factors = {
        "velocity": torch.tensor([float(i % 10) / 10 for i in range(50)]),  # Cyclic
        "color": torch.tensor([float((i * 7) % 10) / 10 for i in range(50)]),  # Different pattern
        "size": torch.tensor([1.0 if i < 25 else 0.0 for i in range(50)]),  # Binary
        "position_x": torch.tensor([float(i % 5) / 5 for i in range(50)]),  # Grid-like
    }

    print(f"    Factors: {list(factors.keys())}")
    print(f"    Sequence length: {len(obs_seq)}")

    print("\n[3] Evaluating disentanglement across multiple components...")

    # For DreamerV3, we evaluate available components
    # For IJEPA, you'd use: ["context_encoder", "predictor", "target_encoder", "z_posterior"]
    components = ["h", "z_posterior", "z_prior"]
    metrics = ["MIG", "DCI", "SAP"]

    result = analyzer.evaluate_components(
        cache=cache,
        factors=factors,
        components=components,
        metrics=metrics,
    )

    print("\n[4] Per-component results:")
    for component in components:
        comp_scores = result.component_results[component]
        print(f"    {component}:")
        print(f"      MIG: {comp_scores['MIG']:.4f}")
        print(f"      DCI Disentanglement: {comp_scores['DCI_disentanglement']:.4f}")
        print(f"      DCI Completeness: {comp_scores['DCI_completeness']:.4f}")
        print(f"      DCI Informativeness: {comp_scores['DCI_informativeness']:.4f}")
        print(f"      SAP: {comp_scores['SAP']:.4f}")

    print("\n[5] Summary scores (averaged across components):")
    for metric, score in result.summary_scores.items():
        print(f"    {metric}: {score:.4f}")

    print("\n[6] Interpretation:")
    print("    - MIG > 0.5: Good separation between factors")
    print("    - DCI Disentanglement > 0.8: Factors well-isolated in latent space")
    print("    - DCI Completeness > 0.8: All factor information captured")
    print("    - DCI Informativeness > 0.9: Good reconstruction from latents")
    print("    - SAP > 0.5: Strong predictive separation")

    # Save results
    results_file = OUTPUT_DIR / "unified_disentanglement_results.txt"
    with open(results_file, "w") as f:
        f.write("Unified Disentanglement Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Components evaluated: {components}\n")
        f.write(f"Metrics: {metrics}\n")
        f.write(f"Factors: {list(factors.keys())}\n\n")

        f.write("Per-Component Results:\n")
        for component in components:
            f.write(f"\n{component}:\n")
            comp_scores = result.component_results[component]
            for metric, score in comp_scores.items():
                f.write(f"  {metric}: {score:.4f}\n")

        f.write(f"\nSummary Scores:\n")
        for metric, score in result.summary_scores.items():
            f.write(f"  {metric}: {score:.4f}\n")

    print(f"\n    Results saved to: {results_file}")

    print("\n" + "=" * 70)
    print("Unified disentanglement evaluation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
