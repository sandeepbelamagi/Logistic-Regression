"""Binary classification metrics for Phase 3 model evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(frozen=True)
class CalibrationBin:
    """Summary of one reliability bin."""

    bin_index: int
    lower_bound: float
    upper_bound: float
    count: int
    average_prediction: float
    observed_rate: float
    absolute_gap: float


def binary_log_loss(
    y_true: Sequence[int],
    y_score: Sequence[float],
    sample_weights: Sequence[float] | None = None,
    epsilon: float = 1e-15,
) -> float:
    """Compute weighted binary cross-entropy."""

    _validate_lengths(y_true, y_score, sample_weights)
    if not y_true:
        return 0.0

    total_weight = 0.0
    loss = 0.0
    for label, score, weight in _iter_rows(y_true, y_score, sample_weights):
        clipped_score = min(max(score, epsilon), 1.0 - epsilon)
        row_loss = -(label * _safe_log(clipped_score) + (1 - label) * _safe_log(1.0 - clipped_score))
        loss += weight * row_loss
        total_weight += weight
    return loss / total_weight if total_weight else 0.0


def roc_auc_score(
    y_true: Sequence[int],
    y_score: Sequence[float],
    sample_weights: Sequence[float] | None = None,
) -> float:
    """Compute weighted ROC-AUC with tie handling."""

    _validate_lengths(y_true, y_score, sample_weights)
    if not y_true:
        return 0.0

    rows = sorted(_iter_rows(y_true, y_score, sample_weights), key=lambda item: item[1])
    positive_weight = sum(weight for label, _, weight in rows if label == 1)
    negative_weight = sum(weight for label, _, weight in rows if label == 0)
    if positive_weight == 0.0 or negative_weight == 0.0:
        return 0.5

    auc_contribution = 0.0
    cumulative_negative_weight = 0.0
    index = 0
    while index < len(rows):
        score = rows[index][1]
        group_positive_weight = 0.0
        group_negative_weight = 0.0
        while index < len(rows) and rows[index][1] == score:
            label, _, weight = rows[index]
            if label == 1:
                group_positive_weight += weight
            else:
                group_negative_weight += weight
            index += 1

        auc_contribution += group_positive_weight * cumulative_negative_weight
        auc_contribution += 0.5 * group_positive_weight * group_negative_weight
        cumulative_negative_weight += group_negative_weight

    return auc_contribution / (positive_weight * negative_weight)


def average_precision_score(
    y_true: Sequence[int],
    y_score: Sequence[float],
    sample_weights: Sequence[float] | None = None,
) -> float:
    """Compute weighted average precision / PR-AUC."""

    _validate_lengths(y_true, y_score, sample_weights)
    if not y_true:
        return 0.0

    rows = sorted(_iter_rows(y_true, y_score, sample_weights), key=lambda item: item[1], reverse=True)
    positive_weight = sum(weight for label, _, weight in rows if label == 1)
    if positive_weight == 0.0:
        return 0.0

    true_positive_weight = 0.0
    false_positive_weight = 0.0
    previous_recall = 0.0
    area = 0.0

    index = 0
    while index < len(rows):
        score = rows[index][1]
        while index < len(rows) and rows[index][1] == score:
            label, _, weight = rows[index]
            if label == 1:
                true_positive_weight += weight
            else:
                false_positive_weight += weight
            index += 1

        precision = true_positive_weight / max(true_positive_weight + false_positive_weight, 1e-15)
        recall = true_positive_weight / positive_weight
        area += precision * max(recall - previous_recall, 0.0)
        previous_recall = recall

    return area


def brier_score(
    y_true: Sequence[int],
    y_score: Sequence[float],
    sample_weights: Sequence[float] | None = None,
) -> float:
    """Compute weighted Brier score."""

    _validate_lengths(y_true, y_score, sample_weights)
    if not y_true:
        return 0.0

    total_weight = 0.0
    squared_error = 0.0
    for label, score, weight in _iter_rows(y_true, y_score, sample_weights):
        squared_error += weight * (score - label) ** 2
        total_weight += weight
    return squared_error / total_weight if total_weight else 0.0


def calibration_curve(
    y_true: Sequence[int],
    y_score: Sequence[float],
    sample_weights: Sequence[float] | None = None,
    n_bins: int = 10,
) -> tuple[list[CalibrationBin], float, float]:
    """Return reliability bins, ECE, and MCE."""

    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")
    _validate_lengths(y_true, y_score, sample_weights)
    if not y_true:
        return [], 0.0, 0.0

    bin_totals = [0.0 for _ in range(n_bins)]
    bin_predictions = [0.0 for _ in range(n_bins)]
    bin_observed = [0.0 for _ in range(n_bins)]

    total_weight = 0.0
    for label, score, weight in _iter_rows(y_true, y_score, sample_weights):
        index = min(int(score * n_bins), n_bins - 1)
        bin_totals[index] += weight
        bin_predictions[index] += weight * score
        bin_observed[index] += weight * label
        total_weight += weight

    bins: list[CalibrationBin] = []
    expected_calibration_error = 0.0
    maximum_calibration_error = 0.0
    for index in range(n_bins):
        count = bin_totals[index]
        if count == 0.0:
            continue
        lower_bound = index / n_bins
        upper_bound = (index + 1) / n_bins
        average_prediction = bin_predictions[index] / count
        observed_rate = bin_observed[index] / count
        gap = abs(average_prediction - observed_rate)
        expected_calibration_error += gap * (count / total_weight)
        maximum_calibration_error = max(maximum_calibration_error, gap)
        bins.append(
            CalibrationBin(
                bin_index=index,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                count=int(count),
                average_prediction=average_prediction,
                observed_rate=observed_rate,
                absolute_gap=gap,
            )
        )

    return bins, expected_calibration_error, maximum_calibration_error


def classification_metric_report(
    y_true: Sequence[int],
    y_score: Sequence[float],
    sample_weights: Sequence[float] | None = None,
    n_bins: int = 10,
) -> dict[str, object]:
    """Bundle the main model quality metrics into one report."""

    bins, ece, mce = calibration_curve(y_true, y_score, sample_weights=sample_weights, n_bins=n_bins)
    return {
        "example_count": len(y_true),
        "log_loss": binary_log_loss(y_true, y_score, sample_weights=sample_weights),
        "roc_auc": roc_auc_score(y_true, y_score, sample_weights=sample_weights),
        "pr_auc": average_precision_score(y_true, y_score, sample_weights=sample_weights),
        "brier_score": brier_score(y_true, y_score, sample_weights=sample_weights),
        "ece": ece,
        "mce": mce,
        "calibration_bins": [asdict(bin_summary) for bin_summary in bins],
    }


def _iter_rows(
    y_true: Sequence[int],
    y_score: Sequence[float],
    sample_weights: Sequence[float] | None,
) -> list[tuple[int, float, float]]:
    if sample_weights is None:
        return [(int(label), float(score), 1.0) for label, score in zip(y_true, y_score, strict=True)]
    return [
        (int(label), float(score), float(weight))
        for label, score, weight in zip(y_true, y_score, sample_weights, strict=True)
    ]


def _validate_lengths(
    y_true: Sequence[int],
    y_score: Sequence[float],
    sample_weights: Sequence[float] | None,
) -> None:
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length.")
    if sample_weights is not None and len(sample_weights) != len(y_true):
        raise ValueError("sample_weights must match y_true length.")


def _safe_log(value: float) -> float:
    from math import log

    return log(value)
