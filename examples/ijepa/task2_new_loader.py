"""Task 2 companion for the new-loader Task 1/3 evaluation pipeline.

Computes 1,000-resample non-parametric bootstrap CIs, paired t-tests, Wilcoxon tests,
Cohen's d effect sizes, and Benjamini-Hochberg / Bonferroni multiple-comparisons adjustments.

Task 2 consumes Task 1's JSON and therefore requires neither ModelHub nor an
ImageNet loader. Its complete statistical experiment is preserved here so the
new-loader pipeline has three standalone entry points.
"""

import os
import sys
import json
import argparse
import numpy as np

# Ensure local library takes precedence
sys.path.insert(0, os.path.abspath("."))

from world_model_lens.analysis.significance import StatisticalSignificanceSuite


def main():
    parser = argparse.ArgumentParser(description="Task 2: Statistical Significance Package Execution")
    parser.add_argument(
        "--task1_json",
        type=str,
        default="task1_deletion_insertion_results.json",
        help="Path to Task 1 output JSON results file."
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="task2_significance_report.json",
        help="Path to save Task 2 statistical significance report."
    )
    parser.add_argument("--n_bootstraps", type=int, default=1000, help="Number of bootstrap resamples.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance threshold level.")
    args = parser.parse_args()

    print("=" * 70)
    print("TASK 2: Statistical Significance & Multiple-Comparisons Package")
    print("=" * 70)

    if not os.path.exists(args.task1_json):
        print(f"[Error] Task 1 JSON file not found at {args.task1_json}. Run Task 1 first.")
        sys.exit(1)

    print(f"[Loading] Ingesting Task 1 quantitative results from {args.task1_json}...")
    with open(args.task1_json, "r") as f:
        task1_data = json.load(f)

    suite = StatisticalSignificanceSuite(
        n_bootstraps=args.n_bootstraps,
        ci_level=0.95,
        alpha=args.alpha
    )

    print(
        f"\n[Analysis] Computing {args.n_bootstraps:,}-resample Bootstrap CIs "
        f"& Paired Tests (alpha={args.alpha})..."
    )
    report = suite.analyze_task1_results(task1_data)
    if not report:
        raise ValueError("Task 1 JSON contains no sample-level AUC distributions")

    outlier_stability = task1_data.get("summary", {}).get("sample_outlier_summary", {})
    if not outlier_stability:
        samples = task1_data.get("samples", [])
        outlier_stability = {
            "frac_samples_ig_deletion_outlier_zscore": float(np.mean([
                sample.get("ig_random_deletion_zscore", 0.0) >= 2.0 for sample in samples
            ])),
            "frac_samples_ig_deletion_outlier_percentile": float(np.mean([
                sample.get("ig_random_deletion_percentile", 0.0) >= 0.95 for sample in samples
            ])),
            "frac_samples_ig_insertion_outlier_zscore": float(np.mean([
                sample.get("ig_random_insertion_zscore", 0.0) <= -2.0 for sample in samples
            ])),
            "frac_samples_ig_insertion_outlier_percentile": float(np.mean([
                sample.get("ig_random_insertion_percentile", 1.0) <= 0.05 for sample in samples
            ])),
        }

    # Attach metadata
    final_output = {
        "metadata": {
            "task1_source": args.task1_json,
            "weights": task1_data.get("metadata", {}).get("weights", "unknown"),
            "n_samples": report.get("n_samples", 0),
            "n_bootstraps": args.n_bootstraps,
            "alpha": args.alpha
        },
        "statistical_report": report,
        "outlier_stability": outlier_stability,
    }

    with open(args.output_json, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"[Export] Saved statistical significance report to {args.output_json}")

    # Print publication-ready statistical report table
    print("\n" + "=" * 70)
    print("TASK 2 STATISTICAL REPORT SUMMARY")
    print("=" * 70)
    del_rep = report["deletion_auc"]
    ins_rep = report["insertion_auc"]

    print("1. Deletion AUC (Higher is Better):")
    print(f"   IG Mean (95% CI)    : {del_rep['ig_auc']['mean']:.4f} [{del_rep['ig_auc']['ci_95'][0]:.4f}, {del_rep['ig_auc']['ci_95'][1]:.4f}]")
    print(f"   Attn Mean (95% CI)  : {del_rep['attn_auc']['mean']:.4f} [{del_rep['attn_auc']['ci_95'][0]:.4f}, {del_rep['attn_auc']['ci_95'][1]:.4f}]")
    print(f"   Rand Mean (95% CI)  : {del_rep['random_auc']['mean']:.4f} [{del_rep['random_auc']['ci_95'][0]:.4f}, {del_rep['random_auc']['ci_95'][1]:.4f}]")
    print(f"   IG vs Rand Paired t : p_raw = {del_rep['ig_vs_rand']['p_val_ttest']:.4e} | p_FDR = {del_rep['ig_vs_rand']['p_fdr']:.4e} | Cohen's d = {del_rep['ig_vs_rand']['cohens_d']:.4f}")

    print("\n2. Insertion AUC (Lower is Better):")
    print(f"   IG Mean (95% CI)    : {ins_rep['ig_auc']['mean']:.4f} [{ins_rep['ig_auc']['ci_95'][0]:.4f}, {ins_rep['ig_auc']['ci_95'][1]:.4f}]")
    print(f"   Attn Mean (95% CI)  : {ins_rep['attn_auc']['mean']:.4f} [{ins_rep['attn_auc']['ci_95'][0]:.4f}, {ins_rep['attn_auc']['ci_95'][1]:.4f}]")
    print(f"   Rand Mean (95% CI)  : {ins_rep['random_auc']['mean']:.4f} [{ins_rep['random_auc']['ci_95'][0]:.4f}, {ins_rep['random_auc']['ci_95'][1]:.4f}]")
    print(f"   IG vs Rand Paired t : p_raw = {ins_rep['ig_vs_rand']['p_val_ttest']:.4e} | p_FDR = {ins_rep['ig_vs_rand']['p_fdr']:.4e} | Cohen's d = {ins_rep['ig_vs_rand']['cohens_d']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
