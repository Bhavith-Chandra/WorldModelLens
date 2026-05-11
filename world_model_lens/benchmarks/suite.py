"""Rigorous evaluation and benchmarking suite for world model interpretability.

This module provides benchmarks inspired by MI roadmaps:
- Toy Synthetic Benchmarks: Tiny world models with planted circuits
- Feature Benchmark: Synthetic latents with known factorized structure
- Circuit Benchmark: Toy dynamics with known computational graphs
- Universality Benchmark: Compare circuits across adapters
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

if TYPE_CHECKING:
    from world_model_lens.core.activation_cache import ActivationCache
    from world_model_lens.hooked_world_model import HookedWorldModel


@dataclass
class BenchmarkResult:
    """Result from a single benchmark."""

    name: str
    score: float
    details: Dict[str, Any]
    passed: bool
    threshold: float = 0.8

    def __post_init__(self):
        self.passed = self.score >= self.threshold


@dataclass
class BenchmarkSuiteResult:
    """Aggregate results from multiple benchmarks."""

    results: Dict[str, BenchmarkResult]
    timestamp: str
    total_score: float
    pass_rate: float
    report: str

    def to_dict(self) -> Dict:
        return {
            "results": {
                k: {"score": v.score, "passed": v.passed, "details": v.details}
                for k, v in self.results.items()
            },
            "timestamp": self.timestamp,
            "total_score": self.total_score,
            "pass_rate": self.pass_rate,
        }


class ToyWorldModel(nn.Module):
    """Base class for toy world models with planted circuits."""

    def __init__(self, seed: int = 42):
        super().__init__()
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

    @abstractmethod
    def forward(
        self, obs: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass returning all intermediate activations."""
        pass


class PositionTrackingModel(ToyWorldModel):
    """Toy world model that tracks position from observations.

    Planted circuit: position = f(obs) where f encodes x,y from input.
    Probes should recover this linear mapping.
    """

    def __init__(self, d_obs: int = 64, d_latent: int = 32, seed: int = 42):
        super().__init__(seed)
        self.d_obs = d_obs
        self.d_latent = d_latent

        self.encoder = nn.Linear(d_obs, d_latent)
        self.dynamics = nn.Linear(d_latent, d_latent)

        with torch.no_grad():
            self.encoder.weight[:, :2] = torch.eye(d_latent)[:, :2].T[:d_obs]
            self.dynamics.weight = torch.eye(d_latent)

    def forward(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        h = self.encoder(obs)
        if action is not None:
            h = h + self.dynamics(action[:, : self.d_latent])
        else:
            h = self.dynamics(h)
        return {"h": h, "z": h[:, : self.d_latent // 2]}


class RewardGatingModel(ToyWorldModel):
    """Toy world model with reward-gating GRU cell.

    Planted circuit: reward prediction depends on specific GRU unit.
    """

    def __init__(self, d_obs: int = 32, d_action: int = 4, d_hidden: int = 64, seed: int = 42):
        super().__init__(seed)
        self.d_obs = d_obs
        self.d_action = d_action
        self.d_hidden = d_hidden

        self.gru = nn.GRUCell(d_obs + d_action, d_hidden)
        self.reward_head = nn.Linear(d_hidden, 1)

        nn.init.eye_(self.gru.weight_hh)
        with torch.no_grad():
            self.reward_head.weight[0, :1] = 1.0

    def forward(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = obs.shape[0]
        h = torch.zeros(batch_size, self.d_hidden, device=obs.device)

        outputs = []
        for t in range(obs.shape[1] if obs.dim() > 1 else 1):
            o = obs[:, t] if obs.dim() > 1 else obs
            a = (
                action[:, t]
                if action is not None and action.dim() > 1
                else (
                    action
                    if action is not None
                    else torch.zeros(1, self.d_action, device=obs.device)
                )
            )
            x = torch.cat([o, a], dim=-1)
            h = self.gru(x, h)
            r = self.reward_head(h)
            outputs.append(r)

        return {"h": h, "reward": torch.stack(outputs, dim=1) if len(outputs) > 1 else outputs[0]}


class CausalChainModel(ToyWorldModel):
    """Toy model with known causal chain: obs -> position -> action -> reward.

    Tests causal tracing accuracy - patching should recover causal links.
    """

    def __init__(self, d_obs: int = 32, d_action: int = 4, seed: int = 42):
        super().__init__(seed)
        self.d_obs = d_obs
        self.d_action = d_action

        self.pos_encoder = nn.Linear(d_obs, 16)
        self.action_predictor = nn.Linear(16, d_action)
        self.reward_predictor = nn.Linear(d_action, 1)

        with torch.no_grad():
            self.action_predictor.weight[:] = 0.5

    def forward(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if obs.dim() == 3:
            T, B, D = obs.shape
            obs_flat = obs.reshape(T * B, D)
            pos = self.pos_encoder(obs_flat)
            pos = pos.reshape(T, B, -1)
            action_pred = self.action_predictor(pos)
            reward_pred = self.reward_predictor(torch.tanh(action_pred))
            return {
                "pos": pos,
                "action_pred": action_pred,
                "reward_pred:": reward_pred.squeeze(-1),
                "h": pos,
            }
        else:
            pos = self.pos_encoder(obs)
            action_pred = self.action_predictor(pos)
            reward_pred = self.reward_predictor(torch.tanh(action_pred))
            return {
                "pos": pos,
                "action_pred": action_pred,
                "reward_pred": reward_pred,
                "h": pos,
            }


class FactorizedLatentModel(ToyWorldModel):
    """Model with known factorized latent structure.

    Planted: z = [factor_x, factor_y, factor_z] where each factor
    encodes an independent ground-truth variable.
    """

    def __init__(self, n_factors: int = 3, factor_dim: int = 8, d_obs: int = 64, seed: int = 42):
        super().__init__(seed)
        self.n_factors = n_factors
        self.factor_dim = factor_dim

        self.factor_encoders = nn.ModuleList(
            [nn.Linear(d_obs, factor_dim) for _ in range(n_factors)]
        )
        self.decoder = nn.Linear(n_factors * factor_dim, d_obs)

    def forward(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        factors = [enc(obs) for enc in self.factor_encoders]
        z = torch.cat(factors, dim=-1)
        recon = self.decoder(z)
        return {"z": z, "factors": factors, "recon": recon}


class SyntheticBenchmark(ABC):
    """Base class for synthetic benchmarks."""

    @abstractmethod
    def run(self, model: ToyWorldModel, **kwargs) -> BenchmarkResult:
        """Run benchmark on model."""
        pass

    @abstractmethod
    def get_ground_truth(self) -> Dict[str, Any]:
        """Return ground truth for comparison."""
        pass


class PositionTrackingBenchmark(SyntheticBenchmark):
    """Benchmark for position tracking models.

    Tests that probes can recover position from latent.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        threshold: float = 0.9,
    ):
        self.n_samples = n_samples
        self.threshold = threshold

    def get_ground_truth(self) -> Dict[str, Any]:
        return {
            "type": "linear",
            "input_dim": 2,
            "d_latent": 32,
            "expected_accuracy": 0.95,
        }

    def run(self, model: ToyWorldModel, **kwargs) -> BenchmarkResult:
        from world_model_lens.probing import LatentProber

        d_obs = getattr(model, "d_obs", 64)
        d_latent = getattr(model, "d_latent", 32)

        torch.manual_seed(42)
        obs = torch.randn(self.n_samples, d_obs)
        pos_gt = obs[:, :2]

        model.eval()
        with torch.no_grad():
            latents = model(obs)["h"]

        prober = LatentProber(seed=42)
        result = prober.train_probe(
            latents,
            pos_gt.numpy(),
            concept_name="position",
            activation_name="h",
            probe_type="ridge",
        )

        return BenchmarkResult(
            name="position_tracking",
            score=result.r2 or 0.0,
            details={
                "r2": result.r2,
                "probe_weights_shape": list(result.feature_weights.shape),
            },
            threshold=self.threshold,
        )


class CircuitBenchmark(SyntheticBenchmark):
    """Benchmark for causal circuit discovery.

    Tests that patching recovers the correct causal graph.
    """

    def __init__(
        self,
        n_samples: int = 500,
        threshold: float = 0.8,
    ):
        self.n_samples = n_samples
        self.threshold = threshold

    def get_ground_truth(self) -> Dict[str, Any]:
        return {
            "causal_graph": ["obs", "pos", "action", "reward"],
            "edges": [("obs", "pos"), ("pos", "action"), ("action", "reward")],
        }

    def run(self, model: ToyWorldModel, **kwargs) -> BenchmarkResult:
        from world_model_lens.patching import TemporalPatcher
        from world_model_lens import HookedWorldModel

        hooked_wm = kwargs.get("hooked_wm")
        if hooked_wm is None:
            return BenchmarkResult(
                name="circuit",
                score=0.0,
                details={"error": "no hooked world model provided"},
                threshold=self.threshold,
            )

        d_obs = getattr(model, "d_obs", 32)
        d_action = getattr(model, "d_action", 4)

        clean_obs = torch.randn(20, d_obs)
        corrupted_obs = clean_obs + 0.5

        traj_clean, cache_clean = hooked_wm.run_with_cache(clean_obs)
        traj_corrupt, cache_corrupt = hooked_wm.run_with_cache(corrupted_obs)

        patcher = TemporalPatcher(hooked_wm)
        components = ["pos", "action_pred", "reward"]

        results = []
        for comp in components:
            result = patcher.patch_state(
                cache_clean,
                cache_corrupt,
                comp,
                10,
                lambda x: x,
            )
            results.append(result.recovery_rate)

        score = max(results) if results else 0.0

        return BenchmarkResult(
            name="circuit",
            score=score,
            details={"component_recoveries": results},
            threshold=self.threshold,
        )


class FeatureGeometryBenchmark(SyntheticBenchmark):
    """Benchmark for feature geometry and disentanglement.

    Tests that learned latents have factorized structure.
    """

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def get_ground_truth(self) -> Dict[str, Any]:
        return {
            "n_factors": 3,
            "factor_dim": 8,
            "expected_disentanglement": 0.9,
        }

    def run(self, model: ToyWorldModel, **kwargs) -> BenchmarkResult:
        from world_model_lens.probing import geometry

        d_obs = 64
        n_samples = 500

        torch.manual_seed(42)
        obs = torch.randn(n_samples, d_obs)

        model.eval()
        with torch.no_grad():
            output = model(obs)
            latents = output["z"] if "z" in output else output["h"]

        if latents.shape[-1] < 24:
            return BenchmarkResult(
                name="feature_geometry",
                score=0.0,
                details={"error": "latent dim too small"},
                threshold=self.threshold,
            )

        try:
            score = geometry.compute_disentanglement(latents, n_factors=3)
        except Exception:
            score = 0.0

        return BenchmarkResult(
            name="feature_geometry",
            score=score,
            details={"disentanglement_score": score},
            threshold=self.threshold,
        )


class UniversalityBenchmark:
    """Benchmark for testing circuit/feature transfer across models.

    Tests if probes/circuits discovered on one model transfer to others.
    """

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def run(
        self,
        source_model: ToyWorldModel,
        source_cache: "ActivationCache",
        target_models: List[ToyWorldModel],
        target_caches: List["ActivationCache"],
    ) -> BenchmarkResult:
        from world_model_lens.probing import LatentProber

        prober = LatentProber(seed=42)
        d_obs = getattr(source_model, "d_obs", 64)

        torch.manual_seed(42)
        obs = torch.randn(500, d_obs)

        source_model.eval()
        with torch.no_grad():
            latents = source_model(obs)["h"]

        prober.train_probe(latents, np.random.randn(500), concept="dummy", activation_name="h")

        transfer_scores = []
        for target_model, target_cache in zip(target_models, target_caches):
            target_model.eval()
            with torch.no_grad():
                target_latents = target_model(obs)["h"]

            target_result = prober.train_probe(
                target_latents,
                np.random.randn(500),
                concept="dummy",
                activation_name="h",
            )
            transfer_scores.append(target_result.accuracy)

        score = np.mean(transfer_scores) if transfer_scores else 0.0

        return BenchmarkResult(
            name="universality",
            score=score,
            details={
                "n_target_models": len(target_models),
                "transfer_scores": transfer_scores,
            },
            threshold=self.threshold,
        )


class BenchmarkSuite:
    """Complete benchmark suite for mechanistic world model interpretability.

    Example:
        suite = BenchmarkSuite()
        results = suite.run_benchmarks(
            model=toy_model,
            hooked_wm=hooked_wm,
            output_dir="./benchmark_results",
        )
        print(results.report)
    """

    def __init__(self):
        self.benchmarks: Dict[str, SyntheticBenchmark] = {}

    def add_benchmark(self, name: str, benchmark: SyntheticBenchmark) -> None:
        """Add a benchmark to the suite."""
        self.benchmarks[name] = benchmark

    def run_benchmarks(
        self,
        model: Union[ToyWorldModel, "HookedWorldModel"],
        hooked_wm: Optional["HookedWorldModel"] = None,
        output_dir: Optional[str] = None,
    ) -> BenchmarkSuiteResult:
        """Run all benchmarks.

        Args:
            model: Model to benchmark (ToyWorldModel or HookedWorldModel).
            hooked_wm: HookedWorldModel wrapper (for patching benchmarks).
            output_dir: Optional directory to save results.

        Returns:
            BenchmarkSuiteResult with all results.
        """
        from datetime import datetime

        results = {}

        toy_model = model if isinstance(model, ToyWorldModel) else None

        for name, benchmark in tqdm(self.benchmarks.items(), desc="Running benchmarks"):
            try:
                if name == "position_tracking" and toy_model:
                    results[name] = benchmark.run(toy_model)
                elif name == "feature_geometry" and toy_model:
                    results[name] = benchmark.run(toy_model)
                elif name == "circuit" and hooked_wm and toy_model:
                    results[name] = benchmark.run(toy_model, hooked_wm=hooked_wm)
                elif hasattr(benchmark, "run"):
                    results[name] = benchmark.run(model)
            except Exception as e:
                results[name] = BenchmarkResult(
                    name=name,
                    score=0.0,
                    details={"error": str(e)},
                    passed=False,
                )

        total_score = sum(r.score for r in results.values()) / len(results) if results else 0.0
        pass_rate = sum(1 for r in results.values() if r.passed) / len(results) if results else 0.0

        timestamp = datetime.now().isoformat()
        report = self._generate_report(results, total_score, pass_rate)

        suite_result = BenchmarkSuiteResult(
            results=results,
            timestamp=timestamp,
            total_score=total_score,
            pass_rate=pass_rate,
            report=report,
        )

        if output_dir:
            self._save_results(suite_result, output_dir)

        return suite_result

    def _generate_report(
        self,
        results: Dict[str, BenchmarkResult],
        total_score: float,
        pass_rate: float,
    ) -> str:
        lines = [
            "=" * 60,
            "MECHANISTIC WORLD MODEL BENCHMARK REPORT",
            "=" * 60,
            "",
            f"Total Score: {total_score:.3f}",
            f"Pass Rate: {pass_rate:.1%}",
            "",
            "Individual Benchmarks:",
            "-" * 40,
        ]
        for name, result in results.items():
            status = "PASS" if result.passed else "FAIL"
            lines.append(f"  {name}: {result.score:.3f} [{status}]")
            for k, v in result.details.items():
                lines.append(f"    - {k}: {v}")
            lines.append("")

        return "\n".join(lines)

    def _save_results(self, result: BenchmarkSuiteResult, output_dir: str) -> None:
        import os

        os.makedirs(output_dir, exist_ok=True)

        with open(f"{output_dir}/results.json", "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        with open(f"{output_dir}/report.txt", "w") as f:
            f.write(result.report)

    def generate_table(
        self,
        results: BenchmarkSuiteResult,
        format: str = "markdown",
    ) -> str:
        """Generate formatted table from results."""
        if format == "markdown":
            lines = [
                "| Benchmark | Score | Status |",
                "|-----------|-------|--------|",
            ]
            for name, result in results.results.items():
                status = "PASS" if result.passed else "FAIL"
                lines.append(f"| {name} | {result.score:.3f} | {status} |")
            return "\n".join(lines)
        elif format == "html":
            rows = []
            for name, result in results.results.items():
                status = "PASS" if result.passed else "FAIL"
                color = "green" if result.passed else "red"
                rows.append(
                    f"<tr><td>{name}</td><td>{result.score:.3f}</td>"
                    f"<td style='color:{color}'>{status}</td></tr>"
                )
            return f"<table>{''.join(rows)}</table>"
        else:
            raise ValueError(f"Unknown format: {format}")


def create_default_suite() -> BenchmarkSuite:
    """Create the default benchmark suite with all benchmarks."""
    suite = BenchmarkSuite()
    suite.add_benchmark("position_tracking", PositionTrackingBenchmark())
    suite.add_benchmark("feature_geometry", FeatureGeometryBenchmark())
    suite.add_benchmark("circuit", CircuitBenchmark())
    return suite
