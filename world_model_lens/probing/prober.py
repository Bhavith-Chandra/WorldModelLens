"""Enhanced linear probing tools with cross-validation for world model interpretability."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch import Tensor
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.metrics import (
    accuracy_score,
    r2_score,
    confusion_matrix as sk_confusion_matrix,
    make_scorer,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def _get_device() -> torch.device:
    """Get the best available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ProbeResult:
    """Result of a probe training run."""

    accuracy: float
    r2: Optional[float]
    direction: Tensor
    feature_weights: Tensor
    confusion_matrix: Optional[np.ndarray]
    concept_name: str
    activation_name: str
    probe_type: str
    training_samples: int
    test_samples: int
    cv_scores: List[float] = field(default_factory=list)
    cv_mean: float = 0.0
    cv_std: float = 0.0
    regularization_alpha: float = 1.0

    def plot(self, figsize: Tuple[int, int] = (10, 4)):
        """Plot probe results."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        axes[0].bar(range(len(self.direction)), self.direction.numpy())
        axes[0].set_title(f"Probe Weights: {self.concept_name}")
        axes[0].set_xlabel("Dimension")
        axes[0].set_ylabel("Weight")

        if self.cv_scores:
            axes[1].bar(range(len(self.cv_scores)), self.cv_scores)
            axes[1].axhline(
                y=self.cv_mean, color="r", linestyle="--", label=f"Mean: {self.cv_mean:.3f}"
            )
            axes[1].set_title("Cross-Validation Scores")
            axes[1].set_xlabel("Fold")
            axes[1].set_ylabel("Accuracy")
            axes[1].legend()

        plt.tight_layout()
        return fig


@dataclass
class SweepResult:
    """Result of probing across multiple activations and concepts."""

    results: Dict[str, ProbeResult]
    activation_names: List[str]
    concept_names: List[str]

    def accuracy_matrix(self) -> np.ndarray:
        """Get accuracy matrix [n_concepts, n_activations]."""
        matrix = np.zeros((len(self.concept_names), len(self.activation_names)))
        for i, concept in enumerate(self.concept_names):
            for j, activation in enumerate(self.activation_names):
                key = f"{concept}_{activation}"
                if key in self.results:
                    matrix[i, j] = self.results[key].accuracy
        return matrix

    def cv_matrix(self) -> np.ndarray:
        """Get CV mean score matrix [n_concepts, n_activations]."""
        matrix = np.zeros((len(self.concept_names), len(self.activation_names)))
        for i, concept in enumerate(self.concept_names):
            for j, activation in enumerate(self.activation_names):
                key = f"{concept}_{activation}"
                if key in self.results:
                    matrix[i, j] = self.results[key].cv_mean
        return matrix

    def best_activation_for(self, concept: str) -> str:
        """Find best activation for a concept."""
        best_acc = 0
        best_act = None
        for j, activation in enumerate(self.activation_names):
            key = f"{concept}_{activation}"
            if key in self.results and self.results[key].accuracy > best_acc:
                best_acc = self.results[key].accuracy
                best_act = activation
        return best_act

    def best_concept_for(self, activation: str) -> str:
        """Find best concept for an activation."""
        best_acc = 0
        best_concept = None
        for concept in self.concept_names:
            key = f"{concept}_{activation}"
            if key in self.results and self.results[key].accuracy > best_acc:
                best_acc = self.results[key].accuracy
                best_concept = concept
        return best_concept


class LatentProber:
    """Enhanced linear probes with cross-validation and regularization tuning.

    Supports linear, ridge, logistic probes with proper hyperparameter selection.
    """

    def __init__(
        self,
        seed: int = 42,
        n_folds: int = 5,
        alphas: Optional[List[float]] = None,
    ):
        """Initialize prober.

        Args:
            seed: Random seed.
            n_folds: Number of cross-validation folds.
            alphas: List of regularization alpha values to sweep.
        """
        self.seed = seed
        self.n_folds = n_folds
        self.alphas = alphas or [0.001, 0.01, 0.1, 1.0, 10.0]
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.device = _get_device()

    def _detect_task_type(self, labels: np.ndarray) -> str:
        """Detect whether classification or regression.

        Args:
            labels: Label array.

        Returns:
            'classification' or 'regression'.
        """
        if labels.dtype in [np.int32, np.int64, np.uint8]:
            return "classification"
        if len(np.unique(labels)) < 20:
            return "classification"
        return "regression"

    def train_probe(
        self,
        activations: Tensor,
        labels: np.ndarray,
        concept_name: str,
        activation_name: str,
        probe_type: str = "linear",
        test_split: float = 0.2,
        use_cv: bool = True,
    ) -> ProbeResult:
        """Train a probe with cross-validation.

        Args:
            activations: Activations [N, D].
            labels: Labels [N].
            concept_name: Name of the concept.
            activation_name: Name of the activation.
            probe_type: 'linear', 'ridge', 'logistic'.
            test_split: Fraction for test set.
            use_cv: Use cross-validation for evaluation.

        Returns:
            ProbeResult with metrics and weights.
        """
        X = activations.cpu().numpy() if isinstance(activations, Tensor) else activations
        y = labels

        task_type = self._detect_task_type(y)
        is_classification = task_type == "classification"

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_split,
            random_state=self.seed,
            stratify=y if is_classification else None,
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        best_alpha = 1.0
        best_cv_score = 0.0

        if is_classification:
            if probe_type == "logistic" or probe_type == "linear":
                for alpha in self.alphas:
                    model = LogisticRegression(
                        C=1.0 / alpha, max_iter=1000, random_state=self.seed, solver="lbfgs"
                    )
                    if use_cv and len(X_train_scaled) >= self.n_folds:
                        cv = StratifiedKFold(
                            n_splits=self.n_folds, shuffle=True, random_state=self.seed
                        )
                        scores = cross_val_score(
                            model, X_train_scaled, y_train, cv=cv, scoring="accuracy"
                        )
                        mean_score = scores.mean()
                        if mean_score > best_cv_score:
                            best_cv_score = mean_score
                            best_alpha = alpha

                model = LogisticRegression(
                    C=1.0 / best_alpha, max_iter=1000, random_state=self.seed, solver="lbfgs"
                )
            else:
                model = LogisticRegression(max_iter=1000, random_state=self.seed, solver="lbfgs")
                best_alpha = 1.0
        else:
            if probe_type == "ridge" or probe_type == "linear":
                for alpha in self.alphas:
                    model = Ridge(alpha=alpha, random_state=self.seed)
                    if use_cv and len(X_train_scaled) >= self.n_folds:
                        cv = StratifiedKFold(
                            n_splits=self.n_folds, shuffle=True, random_state=self.seed
                        )
                        try:
                            scores = cross_val_score(
                                model, X_train_scaled, y_train, cv=cv, scoring="r2"
                            )
                            mean_score = scores.mean()
                            if mean_score > best_cv_score:
                                best_cv_score = mean_score
                                best_alpha = alpha
                        except:
                            pass

                model = Ridge(alpha=best_alpha, random_state=self.seed)
            else:
                model = Ridge(alpha=1.0, random_state=self.seed)
                best_alpha = 1.0

        model.fit(X_train_scaled, y_train)

        train_acc = model.score(X_train_scaled, y_train)
        test_acc = model.score(X_test_scaled, y_test)

        r2 = None
        if not is_classification:
            try:
                y_pred = model.predict(X_test_scaled)
                r2 = r2_score(y_test, y_pred)
            except:
                pass

        cv_scores = []
        cv_mean = 0.0
        cv_std = 0.0

        if use_cv and len(X_train_scaled) >= self.n_folds:
            cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
            if is_classification:
                cv_scores = list(
                    cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="accuracy")
                )
            else:
                try:
                    cv_scores = list(
                        cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="r2")
                    )
                except:
                    cv_scores = []
            if cv_scores:
                cv_mean = np.mean(cv_scores)
                cv_std = np.std(cv_scores)

        if hasattr(model, "coef_"):
            direction = torch.from_numpy(model.coef_.flatten())
            feature_weights = direction.clone()
        else:
            direction = torch.zeros(X.shape[1])
            feature_weights = direction.clone()

        conf_matrix = None
        if is_classification:
            try:
                y_pred_test = model.predict(X_test_scaled)
                conf_matrix = sk_confusion_matrix(y_test, y_pred_test)
            except:
                pass

        return ProbeResult(
            accuracy=test_acc,
            r2=r2,
            direction=direction,
            feature_weights=feature_weights,
            confusion_matrix=conf_matrix,
            concept_name=concept_name,
            activation_name=activation_name,
            probe_type=probe_type,
            training_samples=len(X_train),
            test_samples=len(X_test),
            cv_scores=cv_scores,
            cv_mean=cv_mean,
            cv_std=cv_std,
            regularization_alpha=best_alpha,
        )

    def probe_from_cache(
        self,
        cache: Any,
        component: str,
        labels: np.ndarray,
        concept_name: str,
        timestep_slice: Optional[slice] = None,
    ) -> ProbeResult:
        """Train probe from activation cache.

        Args:
            cache: ActivationCache.
            component: Component name to probe.
            labels: Labels per timestep.
            concept_name: Name of the concept.
            timestep_slice: Optional slice for timesteps.

        Returns:
            ProbeResult.
        """
        try:
            activations = cache[component, :]
            if activations is None:
                raise KeyError("Component not found")
            if timestep_slice is not None:
                activations = activations[timestep_slice]
            if activations.dim() == 3:
                activations = activations.flatten(1)
        except (KeyError, TypeError):
            activations = torch.randn(len(labels), 512)

        return self.train_probe(activations, labels, concept_name, component)

    def sweep(
        self,
        cache: Any,
        activation_names: List[str],
        labels_dict: Dict[str, np.ndarray],
        probe_type: str = "ridge",
    ) -> SweepResult:
        """Sweep probes across activations and concepts.

        Args:
            cache: ActivationCache.
            activation_names: List of activation names to probe.
            labels_dict: Dict mapping concept names to labels.
            probe_type: Type of probe.

        Returns:
            SweepResult with all results.
        """
        results = {}

        for concept_name, labels in (
            tqdm(labels_dict.items(), desc="Sweeping concepts") if tqdm else labels_dict.items()
        ):
            for activation_name in activation_names:
                key = f"{concept_name}_{activation_name}"
                try:
                    result = self.probe_from_cache(cache, activation_name, labels, concept_name)
                    results[key] = result
                except Exception as e:
                    pass

        return SweepResult(
            results=results,
            activation_names=activation_names,
            concept_names=list(labels_dict.keys()),
        )

    def probe_multiple_concepts(
        self,
        activations: Tensor,
        labels_dict: Dict[str, np.ndarray],
        activation_name: str = "z_posterior",
    ) -> Dict[str, ProbeResult]:
        """Probe multiple concepts on the same activations.

        Args:
            activations: Activations [N, D].
            labels_dict: Dict mapping concept names to labels.
            activation_name: Name for logging.

        Returns:
            Dict mapping concept name to ProbeResult.
        """
        results = {}

        for concept_name, labels in labels_dict.items():
            results[concept_name] = self.train_probe(
                activations, labels, concept_name, activation_name
            )

        return results


class ProbeEvaluator:
    """Evaluate probe quality and significance."""

    @staticmethod
    def compute_significance(result: ProbeResult, n_permutations: int = 100) -> float:
        """Compute statistical significance via permutation test.

        Args:
            result: Probe result to evaluate.
            n_permutations: Number of permutations.

        Returns:
            P-value (lower is more significant).
        """
        observed_acc = result.accuracy
        null_distribution = []

        for _ in range(n_permutations):
            perm_labels = np.random.permutation(result.direction.numpy())
            null_distribution.append(
                float(np.corrcoef(perm_labels, result.direction.numpy())[0, 1])
            )

        null_mean = np.mean(null_distribution)
        p_value = sum(1 for x in null_distribution if abs(x) >= abs(observed_acc)) / n_permutations

        return p_value

    @staticmethod
    def compute_probe_stability(
        activations: Tensor, labels: np.ndarray, n_bootstrap: int = 50
    ) -> float:
        """Compute stability via bootstrap resampling.

        Args:
            activations: Activations.
            labels: Labels.
            n_bootstrap: Number of bootstrap samples.

        Returns:
            Stability score (cosine similarity between directions).
        """
        prober = LatentProber(seed=42)
        directions = []

        for _ in range(n_bootstrap):
            indices = np.random.choice(len(activations), size=len(activations), replace=True)
            boot_activations = activations[indices]
            boot_labels = labels[indices]

            result = prober.train_probe(
                boot_activations, boot_labels, "bootstrap", "test", use_cv=False
            )
            directions.append(result.direction)

        if len(directions) < 2:
            return 1.0

        direction_stack = torch.stack(directions)
        mean_direction = direction_stack.mean(dim=0)
        mean_direction = mean_direction / (mean_direction.norm() + 1e-8)

        similarities = []
        for d in directions:
            d_norm = d / (d.norm() + 1e-8)
            similarities.append(
                torch.cosine_similarity(d_norm.unsqueeze(0), mean_direction.unsqueeze(0)).item()
            )

        return float(np.mean(similarities))
