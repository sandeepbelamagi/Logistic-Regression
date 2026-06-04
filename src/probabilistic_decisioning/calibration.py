"""Post-hoc calibration helpers for Phase 4."""

from __future__ import annotations

import json
import math
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from probabilistic_decisioning.logistic_regression import LogisticRegressionModel, TrainingExample
from probabilistic_decisioning.metrics import classification_metric_report


@dataclass(frozen=True)
class PlattScalingCalibrator:
    """One-dimensional sigmoid calibration model."""

    a: float
    b: float
    method: str = "platt_scaling"

    def predict(self, raw_score: float) -> float:
        return _sigmoid(self.a * raw_score + self.b)

    def to_dict(self) -> dict[str, object]:
        return {"method": self.method, "a": self.a, "b": self.b}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "PlattScalingCalibrator":
        return cls(a=float(payload["a"]), b=float(payload["b"]))


@dataclass(frozen=True)
class IsotonicRegressionCalibrator:
    """Monotonic piecewise-constant calibration model."""

    thresholds: list[float]
    values: list[float]
    method: str = "isotonic_regression"

    def predict(self, raw_score: float) -> float:
        if not self.thresholds or not self.values:
            return 0.5
        index = bisect_left(self.thresholds, raw_score)
        if index >= len(self.values):
            index = len(self.values) - 1
        if index < 0:
            index = 0
        return self.values[index]

    def to_dict(self) -> dict[str, object]:
        return {"method": self.method, "thresholds": self.thresholds, "values": self.values}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "IsotonicRegressionCalibrator":
        return cls(
            thresholds=[float(value) for value in payload["thresholds"]],
            values=[float(value) for value in payload["values"]],
        )


@dataclass(frozen=True)
class CalibrationArtifact:
    """Serialized result of selecting a calibration method."""

    calibration_version: str
    model_version: str
    feature_set_version: str
    task_context: str
    selection_metric: str
    selection_metric_key: str
    candidate_methods: list[str]
    selected_method: str
    selected_calibrator: dict[str, object]
    raw_validation_metrics: dict[str, object]
    selected_validation_metrics: dict[str, object]
    method_reports: dict[str, dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "calibration_version": self.calibration_version,
            "model_version": self.model_version,
            "feature_set_version": self.feature_set_version,
            "task_context": self.task_context,
            "selection_metric": self.selection_metric,
            "selection_metric_key": self.selection_metric_key,
            "candidate_methods": self.candidate_methods,
            "selected_method": self.selected_method,
            "selected_calibrator": self.selected_calibrator,
            "raw_validation_metrics": self.raw_validation_metrics,
            "selected_validation_metrics": self.selected_validation_metrics,
            "method_reports": self.method_reports,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "CalibrationArtifact":
        return cls(
            calibration_version=str(payload["calibration_version"]),
            model_version=str(payload["model_version"]),
            feature_set_version=str(payload["feature_set_version"]),
            task_context=str(payload["task_context"]),
            selection_metric=str(payload["selection_metric"]),
            selection_metric_key=str(payload["selection_metric_key"]),
            candidate_methods=[str(value) for value in payload["candidate_methods"]],
            selected_method=str(payload["selected_method"]),
            selected_calibrator=dict(payload["selected_calibrator"]),
            raw_validation_metrics=dict(payload["raw_validation_metrics"]),
            selected_validation_metrics=dict(payload["selected_validation_metrics"]),
            method_reports={key: dict(value) for key, value in dict(payload["method_reports"]).items()},
        )

    def load_selected_calibrator(self) -> PlattScalingCalibrator | IsotonicRegressionCalibrator:
        return _calibrator_from_dict(self.selected_calibrator)


def fit_calibration_artifact(
    model: LogisticRegressionModel,
    validation_examples: Sequence[TrainingExample],
    candidate_methods: Sequence[str] = ("platt_scaling", "isotonic_regression"),
    selection_metric: str = "validation_ece",
    calibration_version: str = "bank_marketing_calibration_v1",
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> CalibrationArtifact:
    """Fit candidate calibrators and select the best one on validation data."""

    if not validation_examples:
        raise ValueError("validation_examples cannot be empty.")
    if not candidate_methods:
        raise ValueError("candidate_methods cannot be empty.")

    raw_scores = [model.score(example) for example in validation_examples]
    labels = [example.label for example in validation_examples]
    sample_weights = [example.sample_weight for example in validation_examples]
    raw_probabilities = [_sigmoid(score) for score in raw_scores]
    raw_validation_metrics = classification_metric_report(
        labels,
        raw_probabilities,
        sample_weights=sample_weights,
    )

    metric_key = _normalize_metric_name(selection_metric)
    metric_direction = _metric_direction(metric_key)
    selected_method: str | None = None
    selected_value: float | None = None
    selected_calibrator: PlattScalingCalibrator | IsotonicRegressionCalibrator | None = None
    selected_validation_metrics: dict[str, object] | None = None
    method_reports: dict[str, dict[str, object]] = {}

    for method_name in candidate_methods:
        calibrator = _fit_calibrator(
            method_name,
            raw_scores,
            labels,
            sample_weights=sample_weights,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        calibrated_probabilities = [calibrator.predict(score) for score in raw_scores]
        calibrated_metrics = classification_metric_report(
            labels,
            calibrated_probabilities,
            sample_weights=sample_weights,
        )
        method_reports[method_name] = {
            "calibrator": calibrator.to_dict(),
            "calibrated_metrics": calibrated_metrics,
        }
        candidate_value = _metric_value(calibrated_metrics, metric_key)
        if selected_method is None or _is_better(candidate_value, selected_value, metric_direction):
            selected_method = method_name
            selected_value = candidate_value
            selected_calibrator = calibrator
            selected_validation_metrics = calibrated_metrics

    if selected_method is None or selected_calibrator is None or selected_validation_metrics is None:
        raise RuntimeError("Failed to select a calibration method.")

    return CalibrationArtifact(
        calibration_version=calibration_version,
        model_version=model.model_version,
        feature_set_version=model.feature_set_version,
        task_context=model.task_context,
        selection_metric=selection_metric,
        selection_metric_key=metric_key,
        candidate_methods=list(candidate_methods),
        selected_method=selected_method,
        selected_calibrator=selected_calibrator.to_dict(),
        raw_validation_metrics=raw_validation_metrics,
        selected_validation_metrics=selected_validation_metrics,
        method_reports=method_reports,
    )


def evaluate_calibrated_model(
    model: LogisticRegressionModel,
    calibrator: PlattScalingCalibrator | IsotonicRegressionCalibrator,
    examples: Sequence[TrainingExample],
) -> dict[str, object]:
    """Evaluate raw and calibrated probabilities on one split."""

    if not examples:
        return {
            "example_count": 0,
            "raw_metrics": classification_metric_report([], []),
            "calibrated_metrics": classification_metric_report([], []),
        }

    labels = [example.label for example in examples]
    raw_scores = [model.score(example) for example in examples]
    sample_weights = [example.sample_weight for example in examples]
    raw_probabilities = [_sigmoid(score) for score in raw_scores]
    calibrated_probabilities = [calibrator.predict(score) for score in raw_scores]

    return {
        "example_count": len(examples),
        "raw_metrics": classification_metric_report(labels, raw_probabilities, sample_weights=sample_weights),
        "calibrated_metrics": classification_metric_report(
            labels,
            calibrated_probabilities,
            sample_weights=sample_weights,
        ),
    }


def save_calibration_artifact(artifact: CalibrationArtifact, output_path: Path) -> Path:
    """Write a calibration artifact to disk."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(artifact.to_dict(), handle, indent=2, sort_keys=True)
    return output_path


def load_calibration_artifact(path: Path) -> CalibrationArtifact:
    """Load a calibration artifact from disk."""

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return CalibrationArtifact.from_dict(payload)


def load_selected_calibrator(artifact: CalibrationArtifact) -> PlattScalingCalibrator | IsotonicRegressionCalibrator:
    return artifact.load_selected_calibrator()


def _fit_calibrator(
    method_name: str,
    raw_scores: Sequence[float],
    labels: Sequence[int],
    sample_weights: Sequence[float],
    max_iterations: int,
    tolerance: float,
) -> PlattScalingCalibrator | IsotonicRegressionCalibrator:
    if method_name == "platt_scaling":
        return fit_platt_scaler(
            raw_scores,
            labels,
            sample_weights=sample_weights,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
    if method_name == "isotonic_regression":
        return fit_isotonic_regressor(raw_scores, labels, sample_weights=sample_weights)
    raise ValueError(f"Unknown calibration method: {method_name!r}")


def fit_platt_scaler(
    raw_scores: Sequence[float],
    labels: Sequence[int],
    sample_weights: Sequence[float] | None = None,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    l2_regularization: float = 1e-6,
) -> PlattScalingCalibrator:
    """Fit a sigmoid calibrator with Newton updates."""

    _validate_inputs(raw_scores, labels, sample_weights)
    weights = _normalize_weights(sample_weights, len(labels))
    positive_weight = sum(weight for label, weight in zip(labels, weights, strict=True) if label == 1)
    total_weight = sum(weights)
    prior = _smoothed_prior(positive_weight, total_weight)

    if positive_weight == 0.0 or positive_weight == total_weight:
        return PlattScalingCalibrator(a=0.0, b=_logit(prior))

    a = 0.0
    b = _logit(prior)
    for _ in range(max_iterations):
        gradient_a = l2_regularization * a
        gradient_b = l2_regularization * b
        hessian_aa = l2_regularization
        hessian_ab = 0.0
        hessian_bb = l2_regularization

        for raw_score, label, weight in zip(raw_scores, labels, weights, strict=True):
            probability = _sigmoid(a * raw_score + b)
            error = probability - label
            curvature = probability * (1.0 - probability) * weight
            gradient_a += weight * error * raw_score
            gradient_b += weight * error
            hessian_aa += curvature * raw_score * raw_score
            hessian_ab += curvature * raw_score
            hessian_bb += curvature

        determinant = hessian_aa * hessian_bb - hessian_ab * hessian_ab
        if abs(determinant) < 1e-12:
            break

        delta_a = (hessian_bb * gradient_a - hessian_ab * gradient_b) / determinant
        delta_b = (-hessian_ab * gradient_a + hessian_aa * gradient_b) / determinant
        a -= delta_a
        b -= delta_b
        if abs(delta_a) + abs(delta_b) < tolerance:
            break

    return PlattScalingCalibrator(a=a, b=b)


def fit_isotonic_regressor(
    raw_scores: Sequence[float],
    labels: Sequence[int],
    sample_weights: Sequence[float] | None = None,
) -> IsotonicRegressionCalibrator:
    """Fit a monotonic calibration model with pooled adjacent violators."""

    _validate_inputs(raw_scores, labels, sample_weights)
    weights = _normalize_weights(sample_weights, len(labels))
    total_weight = sum(weights)
    positive_weight = sum(weight for label, weight in zip(labels, weights, strict=True) if label == 1)
    prior = _smoothed_prior(positive_weight, total_weight)

    if total_weight == 0.0:
        return IsotonicRegressionCalibrator(thresholds=[0.0], values=[0.5])

    grouped = _aggregate_score_groups(raw_scores, labels, weights)
    blocks: list[_IsoBlock] = []
    for score, block_weight, positive_block_weight in grouped:
        blocks.append(
            _IsoBlock(
                lower_score=score,
                upper_score=score,
                weight=block_weight,
                positive_weight=positive_block_weight,
            )
        )
        while len(blocks) >= 2 and blocks[-2].average() > blocks[-1].average():
            merged_block = blocks[-2].merge(blocks[-1])
            blocks = blocks[:-2]
            blocks.append(merged_block)

    if not blocks:
        return IsotonicRegressionCalibrator(thresholds=[0.0], values=[prior])

    thresholds = [block.upper_score for block in blocks]
    values = [block.average() for block in blocks]
    return IsotonicRegressionCalibrator(thresholds=thresholds, values=values)


def score_examples(model: LogisticRegressionModel, examples: Sequence[TrainingExample]) -> list[float]:
    """Return the raw model scores for a split."""

    return [model.score(example) for example in examples]


def calibrate_scores(
    calibrator: PlattScalingCalibrator | IsotonicRegressionCalibrator,
    raw_scores: Sequence[float],
) -> list[float]:
    """Apply a calibrator to a list of raw scores."""

    return [calibrator.predict(score) for score in raw_scores]


def save_calibrated_split(
    model: LogisticRegressionModel,
    calibrator: PlattScalingCalibrator | IsotonicRegressionCalibrator,
    examples: Sequence[TrainingExample],
    output_path: Path,
) -> Path:
    """Write a calibrated split summary to JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_report = evaluate_calibrated_model(model, calibrator, examples)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(raw_report, handle, indent=2, sort_keys=True)
    return output_path


def _aggregate_score_groups(
    raw_scores: Sequence[float],
    labels: Sequence[int],
    weights: Sequence[float],
) -> list[tuple[float, float, float]]:
    grouped: dict[float, tuple[float, float]] = {}
    for raw_score, label, weight in sorted(zip(raw_scores, labels, weights, strict=True), key=lambda item: item[0]):
        if raw_score not in grouped:
            grouped[raw_score] = (0.0, 0.0)
        current_weight, current_positive_weight = grouped[raw_score]
        grouped[raw_score] = (
            current_weight + weight,
            current_positive_weight + weight * label,
        )
    return [
        (score, weight, positive_weight)
        for score, (weight, positive_weight) in sorted(grouped.items(), key=lambda item: item[0])
    ]


@dataclass(frozen=True)
class _IsoBlock:
    lower_score: float
    upper_score: float
    weight: float
    positive_weight: float

    def average(self) -> float:
        if self.weight == 0.0:
            return 0.0
        return self.positive_weight / self.weight

    def merge(self, other: "_IsoBlock") -> "_IsoBlock":
        return _IsoBlock(
            lower_score=self.lower_score,
            upper_score=other.upper_score,
            weight=self.weight + other.weight,
            positive_weight=self.positive_weight + other.positive_weight,
        )


def _normalize_metric_name(metric_name: str) -> str:
    normalized = metric_name.lower().strip()
    if normalized.startswith("validation_"):
        normalized = normalized[len("validation_") :]
    return normalized


def _metric_direction(metric_name: str) -> str:
    if metric_name in {"roc_auc", "pr_auc"}:
        return "max"
    return "min"


def _metric_value(report: dict[str, object], metric_name: str) -> float:
    if metric_name not in report:
        raise KeyError(f"Metric {metric_name!r} is missing from report keys: {sorted(report)}")
    return float(report[metric_name])


def _is_better(candidate: float, incumbent: float | None, direction: str) -> bool:
    if incumbent is None:
        return True
    if direction == "max":
        return candidate > incumbent
    return candidate < incumbent


def _validate_inputs(
    raw_scores: Sequence[float],
    labels: Sequence[int],
    sample_weights: Sequence[float] | None,
) -> None:
    if len(raw_scores) != len(labels):
        raise ValueError("raw_scores and labels must have the same length.")
    if sample_weights is not None and len(sample_weights) != len(labels):
        raise ValueError("sample_weights must match labels length.")
    if not raw_scores:
        raise ValueError("At least one example is required to fit a calibrator.")


def _normalize_weights(sample_weights: Sequence[float] | None, length: int) -> list[float]:
    if sample_weights is None:
        return [1.0] * length
    return [float(weight) for weight in sample_weights]


def _smoothed_prior(positive_weight: float, total_weight: float) -> float:
    return (positive_weight + 1.0) / (total_weight + 2.0)


def _logit(probability: float) -> float:
    clipped = min(max(probability, 1e-15), 1.0 - 1e-15)
    return math.log(clipped / (1.0 - clipped))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exponent = math.exp(-value)
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _calibrator_from_dict(payload: dict[str, object]) -> PlattScalingCalibrator | IsotonicRegressionCalibrator:
    method = str(payload["method"])
    if method == "platt_scaling":
        return PlattScalingCalibrator.from_dict(payload)
    if method == "isotonic_regression":
        return IsotonicRegressionCalibrator.from_dict(payload)
    raise ValueError(f"Unknown calibrator method: {method!r}")
