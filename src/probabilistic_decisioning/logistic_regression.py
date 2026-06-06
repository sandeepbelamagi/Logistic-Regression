"""Sparse Logistic Regression training and experiment helpers."""

from __future__ import annotations

import copy
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from probabilistic_decisioning.constants import (
    BANK_MARKETING_DENSE_FEATURE_NAMES,
    DEFAULT_FEATURE_SET_VERSION,
    DEFAULT_HASH_DIMENSION,
)
from probabilistic_decisioning.metrics import classification_metric_report


@dataclass(frozen=True)
class TrainingExample:
    """Model-ready example loaded from Phase 2 JSONL artifacts."""

    training_row_id: str
    event_id: str
    snapshot_date: str
    event_ts: str
    label_ts: str
    label: int
    sample_weight: float
    dense_features: tuple[float, ...]
    sparse_feature_ids: tuple[int, ...]
    sparse_feature_values: tuple[float, ...]
    feature_set_version: str
    task_context: str


@dataclass(frozen=True)
class LogisticRegressionTrainingConfig:
    """Training hyperparameters for the baseline model."""

    loss: str = "cross_entropy"
    optimizer: str = "adagrad"
    learning_rate: float = 0.05
    max_epochs: int = 12
    l1: float = 0.0
    l2: float = 0.0001
    epsilon: float = 1e-8
    class_weighting: bool = True
    oversampling: bool = False
    shuffle: bool = True
    seed: int = 13
    early_stopping_patience: int = 3
    validation_bins: int = 10

    def validate(self) -> None:
        if self.loss not in {"cross_entropy", "mse"}:
            raise ValueError("loss must be one of: cross_entropy, mse.")
        if self.optimizer not in {"sgd", "adagrad"}:
            raise ValueError("optimizer must be one of: sgd, adagrad.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive.")
        if self.l1 < 0.0 or self.l2 < 0.0:
            raise ValueError("l1 and l2 must be non-negative.")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be non-negative.")
        if self.validation_bins <= 0:
            raise ValueError("validation_bins must be positive.")


@dataclass
class LogisticRegressionModel:
    """Sparse linear probabilistic classifier with hashed features."""

    bias: float
    dense_weights: list[float]
    sparse_weights: dict[int, float]
    dense_feature_names: tuple[str, ...]
    hash_dimension: int
    model_version: str
    feature_set_version: str
    task_context: str
    loss: str
    optimizer: str
    learning_rate: float
    l1: float
    l2: float

    def score(self, example: TrainingExample) -> float:
        """Return the raw logit for one example."""

        return self.score_features(
            example.dense_features,
            example.sparse_feature_ids,
            example.sparse_feature_values,
        )

    def predict_proba(self, example: TrainingExample) -> float:
        """Return sigmoid-transformed probability."""

        return self.predict_proba_features(
            example.dense_features,
            example.sparse_feature_ids,
            example.sparse_feature_values,
        )

    def score_features(
        self,
        dense_features: Sequence[float],
        sparse_feature_ids: Sequence[int],
        sparse_feature_values: Sequence[float],
    ) -> float:
        """Return the raw logit for an arbitrary engineered feature vector."""

        logit = self.bias
        for weight, feature_value in zip(self.dense_weights, dense_features, strict=True):
            logit += weight * feature_value
        for feature_id, feature_value in _compress_sparse_pairs(sparse_feature_ids, sparse_feature_values):
            logit += self.sparse_weights.get(feature_id, 0.0) * feature_value
        return logit

    def predict_proba_features(
        self,
        dense_features: Sequence[float],
        sparse_feature_ids: Sequence[int],
        sparse_feature_values: Sequence[float],
    ) -> float:
        """Return the sigmoid probability for arbitrary engineered features."""

        return _sigmoid(self.score_features(dense_features, sparse_feature_ids, sparse_feature_values))

    def to_dict(self) -> dict[str, object]:
        """Serialize the model to a JSON-compatible dictionary."""

        return {
            "model_version": self.model_version,
            "task_context": self.task_context,
            "feature_set_version": self.feature_set_version,
            "dense_feature_names": list(self.dense_feature_names),
            "hash_dimension": self.hash_dimension,
            "bias": self.bias,
            "dense_weights": self.dense_weights,
            "sparse_weights": {str(feature_id): weight for feature_id, weight in self.sparse_weights.items()},
            "loss": self.loss,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "l1": self.l1,
            "l2": self.l2,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "LogisticRegressionModel":
        """Reconstruct a model from a JSON artifact."""

        sparse_weights = {
            int(feature_id): float(weight)
            for feature_id, weight in dict(payload["sparse_weights"]).items()
        }
        return cls(
            bias=float(payload["bias"]),
            dense_weights=[float(value) for value in payload["dense_weights"]],
            sparse_weights=sparse_weights,
            dense_feature_names=tuple(str(value) for value in payload["dense_feature_names"]),
            hash_dimension=int(payload["hash_dimension"]),
            model_version=str(payload["model_version"]),
            feature_set_version=str(payload["feature_set_version"]),
            task_context=str(payload["task_context"]),
            loss=str(payload["loss"]),
            optimizer=str(payload["optimizer"]),
            learning_rate=float(payload["learning_rate"]),
            l1=float(payload["l1"]),
            l2=float(payload["l2"]),
        )

    def top_dense_coefficients(self, top_n: int = 10) -> list[dict[str, float | str]]:
        """Return the most influential dense coefficients."""

        ranked = sorted(
            zip(self.dense_feature_names, self.dense_weights, strict=True),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:top_n]
        return [
            {"feature": feature_name, "weight": weight, "absolute_weight": abs(weight)}
            for feature_name, weight in ranked
        ]

    def top_sparse_coefficients(self, top_n: int = 10) -> list[dict[str, float | int]]:
        """Return the most influential sparse hashed coefficients."""

        ranked = sorted(self.sparse_weights.items(), key=lambda item: abs(item[1]), reverse=True)[:top_n]
        return [
            {"feature_id": feature_id, "weight": weight, "absolute_weight": abs(weight)}
            for feature_id, weight in ranked
        ]


@dataclass
class LogisticRegressionTrainingState:
    """Mutable optimizer state used during training."""

    bias_accumulator: float
    dense_accumulators: list[float]
    sparse_accumulators: dict[int, float]


@dataclass(frozen=True)
class TrainingRunResult:
    """Artifacts produced by one training run."""

    model: LogisticRegressionModel
    training_config: LogisticRegressionTrainingConfig
    history: list[dict[str, object]]
    class_weights: dict[str, float]
    dataset_summary: dict[str, int]
    train_metrics: dict[str, object]
    validation_metrics: dict[str, object]
    test_metrics: dict[str, object]


def load_training_examples(path: Path) -> list[TrainingExample]:
    """Load one split produced by Phase 2."""

    examples: list[TrainingExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            examples.append(
                TrainingExample(
                    training_row_id=str(payload["training_row_id"]),
                    event_id=str(payload["event_id"]),
                    snapshot_date=str(payload["snapshot_date"]),
                    event_ts=str(payload["event_ts"]),
                    label_ts=str(payload["label_ts"]),
                    label=int(payload["label"]),
                    sample_weight=float(payload.get("sample_weight", 1.0)),
                    dense_features=tuple(float(value) for value in payload.get("dense_features", [])),
                    sparse_feature_ids=tuple(int(value) for value in payload["sparse_feature_ids"]),
                    sparse_feature_values=tuple(float(value) for value in payload["sparse_feature_values"]),
                    feature_set_version=str(payload["feature_set_version"]),
                    task_context=str(payload["task_context"]),
                )
            )
    return examples


def load_training_splits(data_dir: Path) -> dict[str, list[TrainingExample]]:
    """Load train, validation, and test splits from a Phase 2 artifact directory."""

    training_dir = data_dir / "training"
    return {
        "train": load_training_examples(training_dir / "train.jsonl"),
        "validation": load_training_examples(training_dir / "validation.jsonl"),
        "test": load_training_examples(training_dir / "test.jsonl"),
    }


def train_logistic_regression(
    train_examples: Sequence[TrainingExample],
    validation_examples: Sequence[TrainingExample],
    test_examples: Sequence[TrainingExample],
    config: LogisticRegressionTrainingConfig,
    model_version: str,
    feature_set_version: str = DEFAULT_FEATURE_SET_VERSION,
    task_context: str = "bank_marketing",
    hash_dimension: int = DEFAULT_HASH_DIMENSION,
    dense_feature_names: Sequence[str] | None = None,
) -> TrainingRunResult:
    """Fit a sparse Logistic Regression model and evaluate it on each split."""

    config.validate()
    if not train_examples:
        raise ValueError("train_examples cannot be empty.")

    dense_feature_count = len(train_examples[0].dense_features)
    expected_feature_set_version = train_examples[0].feature_set_version
    expected_task_context = train_examples[0].task_context
    for split_name, split_examples in {
        "train": train_examples,
        "validation": validation_examples,
        "test": test_examples,
    }.items():
        for example in split_examples:
            if len(example.dense_features) != dense_feature_count:
                raise ValueError(f"All {split_name} examples must have the same dense feature length.")
            if example.feature_set_version != expected_feature_set_version:
                raise ValueError(f"{split_name} split has mixed feature_set_version values.")
            if example.task_context != expected_task_context:
                raise ValueError(f"{split_name} split has mixed task_context values.")

    if feature_set_version != expected_feature_set_version:
        raise ValueError(
            f"Requested feature_set_version {feature_set_version!r} does not match training data "
            f"feature_set_version {expected_feature_set_version!r}."
        )
    if task_context != expected_task_context:
        raise ValueError(
            f"Requested task_context {task_context!r} does not match training data task_context "
            f"{expected_task_context!r}."
        )

    class_weights = _compute_class_weights(train_examples) if config.class_weighting else {0: 1.0, 1: 1.0}
    working_train_examples = _maybe_oversample(train_examples, config)
    if dense_feature_names is None:
        resolved_dense_feature_names = _resolve_dense_feature_names(dense_feature_count)
    else:
        resolved_dense_feature_names = tuple(str(value) for value in dense_feature_names)
        if len(resolved_dense_feature_names) != dense_feature_count:
            raise ValueError(
                "dense_feature_names length must match the dense feature count in the training examples."
            )
    optimizer_state = _create_optimizer_state(config.optimizer, dense_feature_count)

    model = LogisticRegressionModel(
        bias=0.0,
        dense_weights=[0.0] * dense_feature_count,
        sparse_weights={},
        dense_feature_names=resolved_dense_feature_names,
        hash_dimension=hash_dimension,
        model_version=model_version,
        feature_set_version=feature_set_version,
        task_context=task_context,
        loss=config.loss,
        optimizer=config.optimizer,
        learning_rate=config.learning_rate,
        l1=config.l1,
        l2=config.l2,
    )

    history: list[dict[str, object]] = []
    best_model = copy.deepcopy(model)
    best_validation_loss = math.inf
    patience_counter = 0
    has_validation_examples = len(validation_examples) > 0

    for epoch in range(1, config.max_epochs + 1):
        epoch_examples = list(working_train_examples)
        if config.shuffle:
            random.Random(config.seed + epoch).shuffle(epoch_examples)

        epoch_loss = 0.0
        epoch_weight = 0.0
        for example in epoch_examples:
            example_weight = example.sample_weight * class_weights.get(example.label, 1.0)
            loss_value = _update_model(model, optimizer_state, example, example_weight, config)
            epoch_loss += example_weight * loss_value
            epoch_weight += example_weight

        train_metrics = evaluate_model(model, train_examples, n_bins=config.validation_bins)
        validation_metrics = evaluate_model(model, validation_examples, n_bins=config.validation_bins)
        train_loss = epoch_loss / epoch_weight if epoch_weight else 0.0
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_log_loss": train_metrics["log_loss"],
                "validation_log_loss": validation_metrics["log_loss"],
                "validation_roc_auc": validation_metrics["roc_auc"],
                "validation_pr_auc": validation_metrics["pr_auc"],
                "validation_ece": validation_metrics["ece"],
            }
        )

        if has_validation_examples:
            current_validation_loss = float(validation_metrics["log_loss"])
            if current_validation_loss < best_validation_loss:
                best_validation_loss = current_validation_loss
                best_model = copy.deepcopy(model)
                patience_counter = 0
            else:
                patience_counter += 1
                if config.early_stopping_patience and patience_counter >= config.early_stopping_patience:
                    break
        else:
            best_model = copy.deepcopy(model)

    train_metrics = evaluate_model(best_model, train_examples, n_bins=config.validation_bins)
    validation_metrics = evaluate_model(best_model, validation_examples, n_bins=config.validation_bins)
    test_metrics = evaluate_model(best_model, test_examples, n_bins=config.validation_bins)

    dataset_summary = {
        "train_rows": len(train_examples),
        "validation_rows": len(validation_examples),
        "test_rows": len(test_examples),
        "train_positive_rows": sum(example.label for example in train_examples),
        "validation_positive_rows": sum(example.label for example in validation_examples),
        "test_positive_rows": sum(example.label for example in test_examples),
    }

    return TrainingRunResult(
        model=best_model,
        training_config=config,
        history=history,
        class_weights={"0": class_weights[0], "1": class_weights[1]},
        dataset_summary=dataset_summary,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
    )


def evaluate_model(
    model: LogisticRegressionModel,
    examples: Sequence[TrainingExample],
    n_bins: int = 10,
) -> dict[str, object]:
    """Score a split and compute standard classification metrics."""

    if not examples:
        return {
            "example_count": 0,
            "log_loss": 0.0,
            "roc_auc": 0.0,
            "pr_auc": 0.0,
            "brier_score": 0.0,
            "ece": 0.0,
            "mce": 0.0,
            "calibration_bins": [],
            "positive_rate": 0.0,
            "mean_prediction": 0.0,
        }

    y_true = [example.label for example in examples]
    y_score = [model.predict_proba(example) for example in examples]
    sample_weights = [example.sample_weight for example in examples]
    report = classification_metric_report(y_true, y_score, sample_weights=sample_weights, n_bins=n_bins)
    report["positive_rate"] = sum(y_true) / len(y_true)
    report["mean_prediction"] = sum(y_score) / len(y_score)
    return report


def run_experiment_suite(
    splits: dict[str, list[TrainingExample]],
    base_config: LogisticRegressionTrainingConfig,
    model_version: str,
    feature_set_version: str = DEFAULT_FEATURE_SET_VERSION,
    task_context: str = "bank_marketing",
    hash_dimension: int = DEFAULT_HASH_DIMENSION,
    stability_repeats: int = 0,
) -> dict[str, object]:
    """Run the Phase 3 ablation suite."""

    baseline_result = train_logistic_regression(
        splits["train"],
        splits["validation"],
        splits["test"],
        base_config,
        model_version=f"{model_version}_baseline",
        feature_set_version=feature_set_version,
        task_context=task_context,
        hash_dimension=hash_dimension,
    )

    experiment_runs = {
        "baseline": _result_summary(baseline_result),
        "mse_loss": _result_summary(
            train_logistic_regression(
                splits["train"],
                splits["validation"],
                splits["test"],
                dataclass_replace(base_config, loss="mse"),
                model_version=f"{model_version}_mse",
                feature_set_version=feature_set_version,
                task_context=task_context,
                hash_dimension=hash_dimension,
            )
        ),
        "oversampling": _result_summary(
            train_logistic_regression(
                splits["train"],
                splits["validation"],
                splits["test"],
                dataclass_replace(base_config, class_weighting=False, oversampling=True),
                model_version=f"{model_version}_oversampling",
                feature_set_version=feature_set_version,
                task_context=task_context,
                hash_dimension=hash_dimension,
            )
        ),
        "no_class_weighting": _result_summary(
            train_logistic_regression(
                splits["train"],
                splits["validation"],
                splits["test"],
                dataclass_replace(base_config, class_weighting=False),
                model_version=f"{model_version}_no_class_weight",
                feature_set_version=feature_set_version,
                task_context=task_context,
                hash_dimension=hash_dimension,
            )
        ),
        "no_regularization": _result_summary(
            train_logistic_regression(
                splits["train"],
                splits["validation"],
                splits["test"],
                dataclass_replace(base_config, l2=0.0),
                model_version=f"{model_version}_no_regularization",
                feature_set_version=feature_set_version,
                task_context=task_context,
                hash_dimension=hash_dimension,
            )
        ),
    }

    stability_report = None
    if stability_repeats > 1:
        stability_report = _coefficient_stability_report(
            splits["train"],
            dataclass_replace(base_config, early_stopping_patience=0),
            model_version=model_version,
            feature_set_version=feature_set_version,
            task_context=task_context,
            hash_dimension=hash_dimension,
            repeats=stability_repeats,
        )

    return {
        "model_version": model_version,
        "feature_set_version": feature_set_version,
        "task_context": task_context,
        "baseline": _result_summary(baseline_result),
        "ablation_runs": experiment_runs,
        "stability_report": stability_report,
    }


def save_training_artifacts(
    result: TrainingRunResult,
    output_dir: Path,
    model_artifact_name: str = "model.json",
    metrics_artifact_name: str = "metrics.json",
    history_artifact_name: str = "history.json",
) -> dict[str, Path]:
    """Persist the trained model and evaluation artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / model_artifact_name
    metrics_path = output_dir / metrics_artifact_name
    history_path = output_dir / history_artifact_name

    with model_path.open("w", encoding="utf-8") as handle:
        json.dump(result.model.to_dict(), handle, indent=2, sort_keys=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset_summary": result.dataset_summary,
                "class_weights": result.class_weights,
                "train_metrics": result.train_metrics,
                "validation_metrics": result.validation_metrics,
                "test_metrics": result.test_metrics,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(result.history, handle, indent=2, sort_keys=True)

    return {
        "model_path": model_path,
        "metrics_path": metrics_path,
        "history_path": history_path,
    }


def save_experiment_report(report: dict[str, object], output_dir: Path) -> Path:
    """Persist a suite report."""

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "experiment_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return report_path


def load_model_artifact(path: Path) -> LogisticRegressionModel:
    """Load a trained model from JSON."""

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return LogisticRegressionModel.from_dict(payload)


def dataclass_replace(instance: LogisticRegressionTrainingConfig, **changes: object) -> LogisticRegressionTrainingConfig:
    """Return a modified copy of the config."""

    payload = asdict(instance)
    payload.update(changes)
    return LogisticRegressionTrainingConfig(**payload)


def _update_model(
    model: LogisticRegressionModel,
    optimizer_state: LogisticRegressionTrainingState,
    example: TrainingExample,
    example_weight: float,
    config: LogisticRegressionTrainingConfig,
) -> float:
    """Apply one stochastic gradient step and return the pre-update loss."""

    sparse_pairs = _compress_sparse_pairs(example.sparse_feature_ids, example.sparse_feature_values)
    probability = _sigmoid(model.score(example))
    raw_error = probability - example.label
    if config.loss == "mse":
        raw_error *= probability * (1.0 - probability)

    gradient_scale = example_weight * raw_error
    loss_value = _cross_entropy(probability, example.label) if config.loss == "cross_entropy" else 0.5 * (probability - example.label) ** 2

    _apply_bias_update(model, optimizer_state, gradient_scale, config)

    for index, feature_value in enumerate(example.dense_features):
        weight = model.dense_weights[index]
        gradient = gradient_scale * feature_value + config.l2 * weight + config.l1 * _sign(weight)
        _apply_dense_update(model, optimizer_state, index, gradient, config)

    for feature_id, feature_value in sparse_pairs:
        weight = model.sparse_weights.get(feature_id, 0.0)
        gradient = gradient_scale * feature_value + config.l2 * weight + config.l1 * _sign(weight)
        _apply_sparse_update(model, optimizer_state, feature_id, gradient, config)

    return loss_value


def _apply_bias_update(
    model: LogisticRegressionModel,
    optimizer_state: LogisticRegressionTrainingState,
    gradient: float,
    config: LogisticRegressionTrainingConfig,
) -> None:
    if config.optimizer == "sgd":
        model.bias -= config.learning_rate * gradient
        return

    optimizer_state.bias_accumulator += gradient * gradient
    model.bias -= config.learning_rate * gradient / math.sqrt(optimizer_state.bias_accumulator + config.epsilon)


def _apply_dense_update(
    model: LogisticRegressionModel,
    optimizer_state: LogisticRegressionTrainingState,
    index: int,
    gradient: float,
    config: LogisticRegressionTrainingConfig,
) -> None:
    if config.optimizer == "sgd":
        model.dense_weights[index] -= config.learning_rate * gradient
        return

    optimizer_state.dense_accumulators[index] += gradient * gradient
    model.dense_weights[index] -= config.learning_rate * gradient / math.sqrt(
        optimizer_state.dense_accumulators[index] + config.epsilon
    )


def _apply_sparse_update(
    model: LogisticRegressionModel,
    optimizer_state: LogisticRegressionTrainingState,
    feature_id: int,
    gradient: float,
    config: LogisticRegressionTrainingConfig,
) -> None:
    if config.optimizer == "sgd":
        model.sparse_weights[feature_id] = model.sparse_weights.get(feature_id, 0.0) - config.learning_rate * gradient
        return

    optimizer_state.sparse_accumulators[feature_id] += gradient * gradient
    model.sparse_weights[feature_id] = model.sparse_weights.get(feature_id, 0.0) - config.learning_rate * gradient / math.sqrt(
        optimizer_state.sparse_accumulators[feature_id] + config.epsilon
    )


def _create_optimizer_state(
    optimizer: str,
    dense_feature_count: int,
) -> LogisticRegressionTrainingState:
    if optimizer == "sgd":
        return LogisticRegressionTrainingState(0.0, [0.0] * dense_feature_count, defaultdict(float))
    return LogisticRegressionTrainingState(0.0, [0.0] * dense_feature_count, defaultdict(float))


def _compute_class_weights(examples: Sequence[TrainingExample]) -> dict[int, float]:
    positive_count = sum(example.label for example in examples)
    negative_count = len(examples) - positive_count
    if positive_count == 0 or negative_count == 0:
        return {0: 1.0, 1: 1.0}
    total_count = len(examples)
    return {
        0: total_count / (2.0 * negative_count),
        1: total_count / (2.0 * positive_count),
    }


def _maybe_oversample(
    examples: Sequence[TrainingExample],
    config: LogisticRegressionTrainingConfig,
) -> list[TrainingExample]:
    if not config.oversampling:
        return list(examples)

    positives = [example for example in examples if example.label == 1]
    negatives = [example for example in examples if example.label == 0]
    if not positives or not negatives:
        return list(examples)

    majority_count = max(len(positives), len(negatives))
    minority_examples = positives if len(positives) < len(negatives) else negatives
    rng = random.Random(config.seed)
    additional_examples = [rng.choice(minority_examples) for _ in range(majority_count - len(minority_examples))]
    combined = list(examples) + additional_examples
    if config.shuffle:
        rng.shuffle(combined)
    return combined


def _coefficient_stability_report(
    train_examples: Sequence[TrainingExample],
    config: LogisticRegressionTrainingConfig,
    model_version: str,
    feature_set_version: str,
    task_context: str,
    hash_dimension: int,
    repeats: int,
) -> dict[str, object]:
    dense_weights_by_feature: dict[str, list[float]] = {feature_name: [] for feature_name in BANK_MARKETING_DENSE_FEATURE_NAMES}
    sparse_weights_by_id: dict[int, list[float]] = defaultdict(list)

    stable_config = dataclass_replace(config, early_stopping_patience=0)
    for repeat in range(repeats):
        bootstrap_examples = _bootstrap_sample(train_examples, seed=config.seed + repeat)
        run_result = train_logistic_regression(
            bootstrap_examples,
            [],
            [],
            stable_config,
            model_version=f"{model_version}_stability_{repeat}",
            feature_set_version=feature_set_version,
            task_context=task_context,
            hash_dimension=hash_dimension,
        )
        for feature_name, weight in zip(run_result.model.dense_feature_names, run_result.model.dense_weights, strict=True):
            dense_weights_by_feature[feature_name].append(weight)
        for feature_id, weight in run_result.model.sparse_weights.items():
            sparse_weights_by_id[feature_id].append(weight)

    dense_stability = [
        {
            "feature": feature_name,
            "mean_weight": _mean(weights),
            "stdev_weight": _stdev(weights),
            "absolute_mean_weight": abs(_mean(weights)),
        }
        for feature_name, weights in dense_weights_by_feature.items()
    ]

    sparse_stability = []
    for feature_id, weights in sorted(
        sparse_weights_by_id.items(),
        key=lambda item: abs(_mean(item[1])),
        reverse=True,
    )[:20]:
        sparse_stability.append(
            {
                "feature_id": feature_id,
                "mean_weight": _mean(weights),
                "stdev_weight": _stdev(weights),
                "absolute_mean_weight": abs(_mean(weights)),
            }
        )

    return {
        "repeat_count": repeats,
        "dense_feature_stability": dense_stability,
        "sparse_feature_stability": sparse_stability,
    }


def _bootstrap_sample(examples: Sequence[TrainingExample], seed: int) -> list[TrainingExample]:
    if not examples:
        return []
    rng = random.Random(seed)
    population = list(examples)
    return [rng.choice(population) for _ in range(len(population))]


def _compress_sparse_pairs(
    feature_ids: Sequence[int],
    feature_values: Sequence[float],
) -> list[tuple[int, float]]:
    if len(feature_ids) != len(feature_values):
        raise ValueError("sparse feature ids and values must have the same length.")
    combined: dict[int, float] = {}
    order: list[int] = []
    for feature_id, feature_value in zip(feature_ids, feature_values, strict=True):
        if feature_id not in combined:
            order.append(feature_id)
            combined[feature_id] = float(feature_value)
        else:
            combined[feature_id] += float(feature_value)
    return [(feature_id, combined[feature_id]) for feature_id in order]


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exponent = math.exp(-value)
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _cross_entropy(probability: float, label: int) -> float:
    clipped_probability = min(max(probability, 1e-15), 1.0 - 1e-15)
    return -(label * math.log(clipped_probability) + (1 - label) * math.log(1.0 - clipped_probability))


def _sign(value: float) -> float:
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_value = _mean(values)
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _resolve_dense_feature_names(dense_feature_count: int) -> tuple[str, ...]:
    if dense_feature_count <= len(BANK_MARKETING_DENSE_FEATURE_NAMES):
        return tuple(BANK_MARKETING_DENSE_FEATURE_NAMES[:dense_feature_count])
    extra_names = tuple(f"dense_feature_{index}" for index in range(len(BANK_MARKETING_DENSE_FEATURE_NAMES), dense_feature_count))
    return tuple(BANK_MARKETING_DENSE_FEATURE_NAMES) + extra_names


def _result_summary(result: TrainingRunResult) -> dict[str, object]:
    return {
        "model_version": result.model.model_version,
        "dataset_summary": result.dataset_summary,
        "class_weights": result.class_weights,
        "train_metrics": result.train_metrics,
        "validation_metrics": result.validation_metrics,
        "test_metrics": result.test_metrics,
        "history": result.history,
        "top_dense_coefficients": result.model.top_dense_coefficients(),
        "top_sparse_coefficients": result.model.top_sparse_coefficients(),
    }
