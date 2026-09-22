import numpy as np
import scipy.stats as stats
from typing import List, Tuple, Dict, Any, Optional, Union


def compute_bootstrap_ci(
    data: Union[List[float], np.ndarray], 
    n_bootstraps: int = 1000, 
    ci_level: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """Computes mean and non-parametric empirical bootstrap confidence interval (low, high).
    
    Args:
        data: Array or list of numerical metrics across evaluation samples.
        n_bootstraps: Number of bootstrap resamples (default 1000).
        ci_level: Confidence level (default 0.95 for 95% CI).
        seed: Random seed for reproducible resampling.
        
    Returns:
        Tuple[float, float, float]: (mean_value, ci_low, ci_high)
    """
    arr = np.asarray(data, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
        
    mean_val = float(np.mean(arr))
    if len(arr) == 1:
        return mean_val, mean_val, mean_val

    rng = np.random.RandomState(seed)
    boot_means = np.zeros(n_bootstraps, dtype=float)
    n_samples = len(arr)

    for b in range(n_bootstraps):
        sample_indices = rng.choice(n_samples, size=n_samples, replace=True)
        boot_means[b] = np.mean(arr[sample_indices])

    alpha = (1.0 - ci_level) / 2.0
    low = float(np.percentile(boot_means, alpha * 100.0))
    high = float(np.percentile(boot_means, (1.0 - alpha) * 100.0))
    return mean_val, low, high


def compute_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Computes Cohen's d standardized effect size for paired samples."""
    diff = a - b
    std_diff = np.std(diff, ddof=1)
    if std_diff == 0:
        return 0.0
    return float(np.mean(diff) / std_diff)


def compute_paired_tests(
    a: Union[List[float], np.ndarray],
    b: Union[List[float], np.ndarray],
    alternative: str = "two-sided"
) -> Dict[str, Any]:
    """Computes paired t-test, Wilcoxon signed-rank test, and Cohen's d effect size."""
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    
    if len(arr_a) != len(arr_b) or len(arr_a) == 0:
        return {
            "p_val_ttest": 1.0,
            "t_stat": 0.0,
            "p_val_wilcoxon": 1.0,
            "wilcoxon_stat": 0.0,
            "cohens_d": 0.0
        }

    t_stat, p_ttest = stats.ttest_rel(arr_a, arr_b, alternative=alternative)

    try:
        w_res = stats.wilcoxon(arr_a, arr_b, alternative=alternative)
        w_stat, p_wilc = float(w_res.statistic), float(w_res.pvalue)
    except Exception:
        w_stat, p_wilc = 0.0, 1.0

    cohens_d = compute_cohens_d(arr_a, arr_b)

    return {
        "p_val_ttest": float(p_ttest) if not np.isnan(p_ttest) else 1.0,
        "t_stat": float(t_stat) if not np.isnan(t_stat) else 0.0,
        "p_val_wilcoxon": p_wilc,
        "wilcoxon_stat": w_stat,
        "cohens_d": cohens_d
    }


def apply_multiple_comparisons_correction(
    p_values: List[float], 
    alpha: float = 0.05
) -> Dict[str, List[Union[float, bool]]]:
    """Applies Benjamini-Hochberg False Discovery Rate (FDR) and Bonferroni corrections."""
    p_arr = np.asarray(p_values, dtype=float)
    n_tests = len(p_arr)
    
    if n_tests == 0:
        return {
            "p_fdr": [],
            "reject_fdr": [],
            "p_bonferroni": [],
            "reject_bonferroni": []
        }

    p_bonf = np.minimum(1.0, p_arr * n_tests)
    reject_bonf = (p_bonf < alpha).tolist()

    sorted_idx = np.argsort(p_arr)
    sorted_p = p_arr[sorted_idx]
    
    p_fdr_sorted = np.zeros(n_tests, dtype=float)
    cum_min = 1.0
    for i in range(n_tests - 1, -1, -1):
        q_val = sorted_p[i] * n_tests / (i + 1)
        cum_min = min(cum_min, q_val)
        p_fdr_sorted[i] = cum_min

    p_fdr = np.zeros(n_tests, dtype=float)
    p_fdr[sorted_idx] = p_fdr_sorted
    p_fdr = np.minimum(1.0, p_fdr)
    reject_fdr = (p_fdr < alpha).tolist()

    return {
        "p_fdr": p_fdr.tolist(),
        "reject_fdr": reject_fdr,
        "p_bonferroni": p_bonf.tolist(),
        "reject_bonferroni": reject_bonf
    }


class StatisticalSignificanceSuite:
    """Statistical Significance Package for WorldModelLens."""

    def __init__(self, n_bootstraps: int = 1000, ci_level: float = 0.95, alpha: float = 0.05):
        self.n_bootstraps = n_bootstraps
        self.ci_level = ci_level
        self.alpha = alpha

    def analyze_task1_results(self, task1_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyzes Task 1 patch knockout deletion/insertion results data."""
        samples = task1_data.get("samples", [])
        if not samples:
            return {}

        n_samples = len(samples)
        k_values = task1_data.get("metadata", {}).get("k_values", [1, 3, 5, 10, 20])

        ig_del_auc = np.array([s["ig_deletion_auc"] for s in samples])
        attn_del_auc = np.array([s["attn_deletion_auc"] for s in samples])
        rand_del_auc = np.array([np.mean(s["random_deletion_auc"]) for s in samples])

        ig_ins_auc = np.array([s["ig_insertion_auc"] for s in samples])
        attn_ins_auc = np.array([s["attn_insertion_auc"] for s in samples])
        rand_ins_auc = np.array([np.mean(s["random_insertion_auc"]) for s in samples])

        ig_del_ci = compute_bootstrap_ci(ig_del_auc, self.n_bootstraps, self.ci_level)
        attn_del_ci = compute_bootstrap_ci(attn_del_auc, self.n_bootstraps, self.ci_level)
        rand_del_ci = compute_bootstrap_ci(rand_del_auc, self.n_bootstraps, self.ci_level)

        ig_ins_ci = compute_bootstrap_ci(ig_ins_auc, self.n_bootstraps, self.ci_level)
        attn_ins_ci = compute_bootstrap_ci(attn_ins_auc, self.n_bootstraps, self.ci_level)
        rand_ins_ci = compute_bootstrap_ci(rand_ins_auc, self.n_bootstraps, self.ci_level)

        del_ig_vs_attn = compute_paired_tests(ig_del_auc, attn_del_auc)
        del_ig_vs_rand = compute_paired_tests(ig_del_auc, rand_del_auc)

        ins_ig_vs_attn = compute_paired_tests(ig_ins_auc, attn_ins_auc)
        ins_ig_vs_rand = compute_paired_tests(ig_ins_auc, rand_ins_auc)

        raw_p_values = [
            del_ig_vs_attn["p_val_ttest"],
            del_ig_vs_rand["p_val_ttest"],
            ins_ig_vs_attn["p_val_ttest"],
            ins_ig_vs_rand["p_val_ttest"]
        ]
        corrections = apply_multiple_comparisons_correction(raw_p_values, self.alpha)

        return {
            "n_samples": n_samples,
            "k_values": k_values,
            "deletion_auc": {
                "ig_auc": {"mean": ig_del_ci[0], "ci_95": [ig_del_ci[1], ig_del_ci[2]]},
                "attn_auc": {"mean": attn_del_ci[0], "ci_95": [attn_del_ci[1], attn_del_ci[2]]},
                "random_auc": {"mean": rand_del_ci[0], "ci_95": [rand_del_ci[1], rand_del_ci[2]]},
                "ig_vs_attn": {**del_ig_vs_attn, "p_fdr": corrections["p_fdr"][0], "p_bonferroni": corrections["p_bonferroni"][0]},
                "ig_vs_rand": {**del_ig_vs_rand, "p_fdr": corrections["p_fdr"][1], "p_bonferroni": corrections["p_bonferroni"][1]}
            },
            "insertion_auc": {
                "ig_auc": {"mean": ig_ins_ci[0], "ci_95": [ig_ins_ci[1], ig_ins_ci[2]]},
                "attn_auc": {"mean": attn_ins_ci[0], "ci_95": [attn_ins_ci[1], attn_ins_ci[2]]},
                "random_auc": {"mean": rand_ins_ci[0], "ci_95": [rand_ins_ci[1], rand_ins_ci[2]]},
                "ig_vs_attn": {**ins_ig_vs_attn, "p_fdr": corrections["p_fdr"][2], "p_bonferroni": corrections["p_bonferroni"][2]},
                "ig_vs_rand": {**ins_ig_vs_rand, "p_fdr": corrections["p_fdr"][3], "p_bonferroni": corrections["p_bonferroni"][3]}
            }
        }

    def analyze_aaf_results(self, aaf_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyzes AAF multi-layer attribution sweep data (N=448, Predictor Layers 0..3).
        
        Computes 1,000-resample bootstrap 95% CIs and FDR/Bonferroni corrections for:
        - Layer-wise Spearman rank correlation (rho)
        - Layer-wise Jaccard overlap (O_k)
        - Layer-wise Ranking Inversion Rate (rho < 0)
        - Layer-wise Failure Rate (O_k <= 0.3)
        """
        if not aaf_samples:
            return {}

        n_samples = len(aaf_samples)

        # Extract layer-wise metrics across samples
        # Expected structure: sample["layers"][layer_idx] -> {spearman_rho, jaccard_overlap}
        layer_ids = sorted(list(aaf_samples[0]["layers"].keys()))
        
        layer_summary = {}
        layer_p_values = []
        layer_pairs = []

        # Predictor Layer 0 is baseline for paired comparisons
        l0_rhos = np.array([s["layers"][layer_ids[0]]["spearman_rho"] for s in aaf_samples])

        for l_id in layer_ids:
            rhos = np.array([s["layers"][l_id]["spearman_rho"] for s in aaf_samples])
            overlaps = np.array([s["layers"][l_id]["jaccard_overlap"] for s in aaf_samples])

            inversions = (rhos < 0).astype(float)
            failures = (overlaps <= 0.3).astype(float)

            rho_ci = compute_bootstrap_ci(rhos, self.n_bootstraps, self.ci_level)
            overlap_ci = compute_bootstrap_ci(overlaps, self.n_bootstraps, self.ci_level)
            inv_ci = compute_bootstrap_ci(inversions, self.n_bootstraps, self.ci_level)
            fail_ci = compute_bootstrap_ci(failures, self.n_bootstraps, self.ci_level)

            paired_with_l0 = compute_paired_tests(rhos, l0_rhos) if l_id != layer_ids[0] else None
            if paired_with_l0:
                layer_p_values.append(paired_with_l0["p_val_ttest"])
                layer_pairs.append(l_id)

            layer_summary[str(l_id)] = {
                "spearman_rho": {"mean": rho_ci[0], "ci_95": [rho_ci[1], rho_ci[2]]},
                "jaccard_overlap": {"mean": overlap_ci[0], "ci_95": [overlap_ci[1], overlap_ci[2]]},
                "ranking_inversion_rate": {"mean": inv_ci[0], "ci_95": [inv_ci[1], inv_ci[2]]},
                "failure_rate": {"mean": fail_ci[0], "ci_95": [fail_ci[1], fail_ci[2]]},
                "paired_test_vs_layer0": paired_with_l0
            }

        # Apply Benjamini-Hochberg FDR & Bonferroni across layer contrast hypotheses
        corrections = apply_multiple_comparisons_correction(layer_p_values, self.alpha)
        for idx, l_id in enumerate(layer_pairs):
            layer_summary[str(l_id)]["paired_test_vs_layer0"]["p_fdr"] = corrections["p_fdr"][idx]
            layer_summary[str(l_id)]["paired_test_vs_layer0"]["p_bonferroni"] = corrections["p_bonferroni"][idx]

        return {
            "n_samples": n_samples,
            "n_layers": len(layer_ids),
            "layers": layer_summary
        }
