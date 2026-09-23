import pytest
import numpy as np

from world_model_lens.analysis.significance import (
    compute_bootstrap_ci,
    compute_paired_tests,
    compute_cohens_d,
    apply_multiple_comparisons_correction,
    StatisticalSignificanceSuite,
)


def test_compute_bootstrap_ci():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    mean_val, low, high = compute_bootstrap_ci(data, n_bootstraps=500, ci_level=0.95, seed=42)
    
    assert np.isclose(mean_val, 5.5)
    assert low < mean_val < high
    assert 1.0 <= low <= 10.0
    assert 1.0 <= high <= 10.0


def test_compute_cohens_d():
    a = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    b = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
    
    d = compute_cohens_d(a, b)
    assert d > 0.0
    assert isinstance(d, float)


def test_compute_paired_tests():
    a = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    b = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
    
    res = compute_paired_tests(a, b)
    assert "p_val_ttest" in res
    assert "p_val_wilcoxon" in res
    assert "cohens_d" in res
    assert 0.0 <= res["p_val_ttest"] <= 1.0
    assert 0.0 <= res["p_val_wilcoxon"] <= 1.0
    assert res["cohens_d"] > 0.0


def test_apply_multiple_comparisons_correction():
    raw_p = [0.001, 0.01, 0.04, 0.20, 0.50]
    res = apply_multiple_comparisons_correction(raw_p, alpha=0.05)
    
    p_fdr = res["p_fdr"]
    p_bonf = res["p_bonferroni"]
    
    assert len(p_fdr) == 5
    assert len(p_bonf) == 5
    
    for fdr, bonf in zip(p_fdr, p_bonf):
        assert fdr <= bonf + 1e-9


def test_statistical_significance_suite():
    suite = StatisticalSignificanceSuite(n_bootstraps=100)
    
    task1_data = {
        "metadata": {"k_values": [1, 3, 5]},
        "samples": [
            {
                "ig_deletion_auc": 0.3,
                "attn_deletion_auc": 0.2,
                "random_deletion_auc": [0.15] * 5,
                "ig_insertion_auc": 0.2,
                "attn_insertion_auc": 0.3,
                "random_insertion_auc": [0.35] * 5,
            }
            for _ in range(10)
        ]
    }
    
    report = suite.analyze_task1_results(task1_data)
    assert report["n_samples"] == 10
    assert "deletion_auc" in report
    assert "insertion_auc" in report
    assert "p_fdr" in report["deletion_auc"]["ig_vs_rand"]
    assert "p_bonferroni" in report["deletion_auc"]["ig_vs_rand"]
    assert "p_fdr_wilcoxon" in report["deletion_auc"]["ig_vs_rand"]
    assert "p_bonferroni_wilcoxon" in report["insertion_auc"]["ig_vs_attn"]


def test_statistical_significance_suite_aaf():
    suite = StatisticalSignificanceSuite(n_bootstraps=100)
    
    aaf_samples = [
        {
            "sample_idx": i,
            "layers": {
                0: {"spearman_rho": -0.1 if i % 2 == 0 else 0.2, "jaccard_overlap": 0.15},
                1: {"spearman_rho": 0.45, "jaccard_overlap": 0.38},
                2: {"spearman_rho": 0.67, "jaccard_overlap": 0.22},
                3: {"spearman_rho": 0.54, "jaccard_overlap": 0.09},
            }
        }
        for i in range(20)
    ]
    
    report = suite.analyze_aaf_results(aaf_samples)
    assert report["n_samples"] == 20
    assert report["n_layers"] == 4
    assert "0" in report["layers"]
    assert "ranking_inversion_rate" in report["layers"]["0"]
    assert report["layers"]["0"]["ranking_inversion_rate"]["mean"] == 0.5
    assert "p_fdr" in report["layers"]["1"]["paired_test_vs_layer0"]
