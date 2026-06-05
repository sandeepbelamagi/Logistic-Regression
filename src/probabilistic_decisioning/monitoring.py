"""Phase 6 monitoring, governance, and rollout controls."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from probabilistic_decisioning.logistic_regression import load_model_artifact, load_training_splits
from probabilistic_decisioning.metrics import classification_metric_report


JSONDict = dict[str, Any]

POSITIVE_ACTIONS_BY_CONTEXT: dict[str, set[str]] = {
    "bank_marketing": {"prioritize_contact"},
    "fraud_policy": {"approve"},
    "loan_policy": {"approve"},
}


@dataclass(frozen=True)
class MonitoringThresholds:
    """Alert thresholds mirrored from configs/monitoring.yaml."""

    ece_critical: float = 0.03
    psi_critical: float = 0.2
    prediction_p95_latency_ms_critical: float = 10.0
    manual_review_utilization_critical: float = 0.85
    subgroup_calibration_gap_critical: float = 0.05
    approval_rate_gap_critical: float = 0.05


@dataclass(frozen=True)
class MonitoringInputs:
    """Paths and knobs required to generate a monitoring report."""

    model_path: Path
    training_data_dir: Path
    prediction_log_path: Path
    decision_log_path: Path
    phase3_metrics_path: Path
    phase4_report_path: Path
    output_dir: Path
    monitoring_config_path: Path | None = None
    group_key: str = "task_context"


@dataclass(frozen=True)
class MonitoringAlert:
    """Single monitoring alert emitted by Phase 6."""

    metric: str
    severity: str
    value: float | None
    threshold: float
    message: str

    def to_dict(self) -> JSONDict:
        return asdict(self)


@dataclass(frozen=True)
class MonitoringRunResult:
    """Files and report produced by one monitoring job."""

    report: JSONDict
    report_path: Path
    alerts_path: Path


def run_monitoring(
    inputs: MonitoringInputs,
    thresholds: MonitoringThresholds | None = None,
) -> MonitoringRunResult:
    """Generate a batch monitoring report from Phase 2 through Phase 5 artifacts."""

    thresholds = thresholds or load_monitoring_thresholds(inputs.monitoring_config_path)

    model = load_model_artifact(inputs.model_path)
    training_splits = load_training_splits(inputs.training_data_dir)
    training_examples = training_splits["train"]
    prediction_records = _load_jsonl(inputs.prediction_log_path)
    decision_records = _load_jsonl(inputs.decision_log_path)
    phase3_metrics = _load_json_file(inputs.phase3_metrics_path)
    phase4_report = _load_json_file(inputs.phase4_report_path)

    _validate_artifact_compatibility(model, phase3_metrics, phase4_report)

    generated_at = datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")
    primary_decisions = [
        decision
        for decision in decision_records
        if str(decision.get("task_context", "")).strip().lower() == model.task_context
    ]

    prediction_summary = _prediction_summary(prediction_records)
    decision_summary = _decision_summary(decision_records)
    live_primary_metrics, live_primary_raw_metrics = _live_model_metrics(primary_decisions)
    feature_drift = _feature_drift_summary(model, training_examples, prediction_records)
    governance = _governance_summary(
        prediction_records,
        decision_records,
        primary_decisions,
        group_key=inputs.group_key,
    )
    calibration_snapshot = _calibration_snapshot(phase4_report, live_primary_metrics, live_primary_raw_metrics)
    system_summary = _system_summary(prediction_summary, decision_summary)

    report: JSONDict = {
        "generated_at": generated_at,
        "monitoring_version": "bank_marketing_monitoring_v1",
        "artifacts": {
            "model_path": str(inputs.model_path),
            "training_data_dir": str(inputs.training_data_dir),
            "prediction_log_path": str(inputs.prediction_log_path),
            "decision_log_path": str(inputs.decision_log_path),
            "phase3_metrics_path": str(inputs.phase3_metrics_path),
            "phase4_report_path": str(inputs.phase4_report_path),
            "monitoring_config_path": str(inputs.monitoring_config_path) if inputs.monitoring_config_path else None,
        },
        "model": {
            "model_version": model.model_version,
            "feature_set_version": model.feature_set_version,
            "task_context": model.task_context,
            "dense_feature_names": list(model.dense_feature_names),
        },
        "summary": system_summary,
        "model_metrics": phase3_metrics,
        "calibration_metrics": calibration_snapshot,
        "feature_drift": feature_drift,
        "governance": governance,
        "alerts": [],
    }

    alerts = _build_alerts(report, thresholds)
    report["alerts"] = [alert.to_dict() for alert in alerts]
    report["rollout_readiness"] = _rollout_readiness(report["alerts"], system_summary)

    output_dir = inputs.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "monitoring_report.json"
    alerts_path = output_dir / "alerts.json"
    _write_json(report_path, report)
    _write_json(alerts_path, [alert.to_dict() for alert in alerts])

    return MonitoringRunResult(report=report, report_path=report_path, alerts_path=alerts_path)


def load_monitoring_thresholds(path: Path | None = None) -> MonitoringThresholds:
    """Load alert thresholds from the YAML config file or fall back to defaults."""

    defaults = asdict(MonitoringThresholds())
    if path is None or not path.exists():
        return MonitoringThresholds(**defaults)

    parsed_alerts = _parse_alert_thresholds(path)
    defaults.update({key: float(value) for key, value in parsed_alerts.items() if key in defaults})
    return MonitoringThresholds(**defaults)


def _validate_artifact_compatibility(model: Any, phase3_metrics: JSONDict, phase4_report: JSONDict) -> None:
    if phase4_report.get("model_version") != model.model_version:
        raise ValueError(
            "Phase 4 report model_version does not match the Phase 3 model artifact. "
            f"Model={model.model_version!r}, report={phase4_report.get('model_version')!r}."
        )
    if phase4_report.get("feature_set_version") != model.feature_set_version:
        raise ValueError(
            "Phase 4 report feature_set_version does not match the Phase 3 model artifact. "
            f"Model={model.feature_set_version!r}, report={phase4_report.get('feature_set_version')!r}."
        )
    if phase4_report.get("task_context") != model.task_context:
        raise ValueError(
            "Phase 4 report task_context does not match the Phase 3 model artifact. "
            f"Model={model.task_context!r}, report={phase4_report.get('task_context')!r}."
        )
    if not phase3_metrics.get("validation_metrics"):
        raise ValueError("Phase 3 metrics artifact must contain validation_metrics.")


def _load_json_file(path: Path) -> JSONDict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> list[JSONDict]:
    records: list[JSONDict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))
    return records


def _prediction_summary(prediction_records: Sequence[JSONDict]) -> JSONDict:
    latencies = [_safe_float(record.get("latency_ms")) for record in prediction_records if record.get("latency_ms") is not None]
    dense_zero_rates = []
    sparse_bucket_counts = [0 for _ in range(32)]
    dense_feature_count = 0
    prediction_ids: set[str] = set()

    for record in prediction_records:
        if record.get("prediction_id") is not None:
            prediction_ids.add(str(record["prediction_id"]))
        feature_vector = record.get("feature_vector") or {}
        dense_features = feature_vector.get("dense_features") or []
        sparse_feature_ids = feature_vector.get("sparse_feature_ids") or []
        dense_feature_count = max(dense_feature_count, len(dense_features))
        if dense_features:
            dense_zero_rates.append(sum(1 for value in dense_features if float(value) == 0.0) / len(dense_features))
        for feature_id in sparse_feature_ids:
            sparse_bucket_counts[int(feature_id) % len(sparse_bucket_counts)] += 1

    return {
        "prediction_count": len(prediction_records),
        "prediction_error_rate": 0.0,
        "prediction_logging_completeness": 1.0 if prediction_records else 0.0,
        "prediction_p50_latency_ms": _percentile(latencies, 50.0),
        "prediction_p95_latency_ms": _percentile(latencies, 95.0),
        "prediction_p99_latency_ms": _percentile(latencies, 99.0),
        "feature_null_rate_proxy": mean(dense_zero_rates) if dense_zero_rates else 0.0,
        "sparse_bucket_distribution": sparse_bucket_counts,
        "feature_freshness_lag_seconds": None,
        "dense_feature_count": dense_feature_count,
        "prediction_ids": prediction_ids,
    }


def _decision_summary(decision_records: Sequence[JSONDict]) -> JSONDict:
    action_counts: dict[str, int] = {}
    manual_review_count = 0
    positive_action_count = 0
    label_delay_days: list[float] = []
    linked_prediction_ids = set()

    for record in decision_records:
        action = str(record.get("action", "unknown"))
        action_counts[action] = action_counts.get(action, 0) + 1
        if bool(record.get("manual_review_required")):
            manual_review_count += 1
        if record.get("prediction_id") is not None:
            linked_prediction_ids.add(str(record["prediction_id"]))
        label_delay = record.get("outcome_delay_days")
        if label_delay is not None:
            label_delay_days.append(float(label_delay))
        if _is_positive_action(record):
            positive_action_count += 1

    decision_count = len(decision_records)
    return {
        "decision_count": decision_count,
        "action_counts": action_counts,
        "manual_review_rate": manual_review_count / decision_count if decision_count else 0.0,
        "positive_action_rate": positive_action_count / decision_count if decision_count else 0.0,
        "label_delay_mean_days": mean(label_delay_days) if label_delay_days else 0.0,
        "label_delay_p95_days": _percentile(label_delay_days, 95.0) if label_delay_days else 0.0,
        "linked_prediction_ids": linked_prediction_ids,
    }


def _live_model_metrics(decision_records: Sequence[JSONDict]) -> tuple[JSONDict, JSONDict]:
    labeled_primary = [
        record
        for record in decision_records
        if record.get("realized_label") is not None
        and record.get("calibrated_probability") is not None
    ]
    if not labeled_primary:
        return {}, {}

    labels = [int(record["realized_label"]) for record in labeled_primary]
    calibrated_probabilities = [float(record["calibrated_probability"]) for record in labeled_primary]
    sample_weights = [1.0 for _ in labeled_primary]
    calibrated_metrics = classification_metric_report(labels, calibrated_probabilities, sample_weights=sample_weights)

    raw_with_labels = [
        record for record in labeled_primary if record.get("raw_probability") is not None
    ]
    raw_metrics: JSONDict = {}
    if raw_with_labels:
        raw_labels = [int(record["realized_label"]) for record in raw_with_labels]
        raw_probabilities = [float(record["raw_probability"]) for record in raw_with_labels]
        raw_weights = [1.0 for _ in raw_with_labels]
        raw_metrics = classification_metric_report(raw_labels, raw_probabilities, sample_weights=raw_weights)

    return calibrated_metrics, raw_metrics


def _feature_drift_summary(
    model: Any,
    training_examples: Sequence[Any],
    prediction_records: Sequence[JSONDict],
) -> JSONDict:
    live_feature_vectors = [
        record.get("feature_vector")
        for record in prediction_records
        if record.get("feature_vector")
    ]
    dense_feature_psi: dict[str, float] = {}
    dense_feature_zero_rate: dict[str, JSONDict] = {}

    for index, feature_name in enumerate(model.dense_feature_names):
        baseline_values = [float(example.dense_features[index]) for example in training_examples if len(example.dense_features) > index]
        live_values = [
            float(feature_vector["dense_features"][index])
            for feature_vector in live_feature_vectors
            if len(feature_vector.get("dense_features", [])) > index
        ]
        dense_feature_psi[feature_name] = population_stability_index(baseline_values, live_values)
        dense_feature_zero_rate[feature_name] = {
            "baseline": _zero_rate(baseline_values),
            "live": _zero_rate(live_values),
        }

    baseline_sparse_buckets = [
        int(feature_id) % 32
        for example in training_examples
        for feature_id in example.sparse_feature_ids
    ]
    live_sparse_buckets = [
        int(feature_id) % 32
        for feature_vector in live_feature_vectors
        for feature_id in feature_vector.get("sparse_feature_ids", [])
    ]

    max_dense_psi = max(dense_feature_psi.values()) if dense_feature_psi else 0.0
    mean_dense_psi = mean(dense_feature_psi.values()) if dense_feature_psi else 0.0
    sparse_bucket_psi = population_stability_index(baseline_sparse_buckets, live_sparse_buckets, n_bins=32)

    return {
        "dense_feature_psi": dense_feature_psi,
        "dense_feature_zero_rate": dense_feature_zero_rate,
        "max_dense_psi": max_dense_psi,
        "mean_dense_psi": mean_dense_psi,
        "sparse_bucket_psi": sparse_bucket_psi,
        "baseline_sparse_bucket_count": len(baseline_sparse_buckets),
        "live_sparse_bucket_count": len(live_sparse_buckets),
    }


def _governance_summary(
    prediction_records: Sequence[JSONDict],
    decision_records: Sequence[JSONDict],
    primary_decisions: Sequence[JSONDict],
    group_key: str = "task_context",
) -> JSONDict:
    prediction_ids = {
        str(record["prediction_id"])
        for record in prediction_records
        if record.get("prediction_id") is not None
    }
    groups: dict[str, list[JSONDict]] = {}
    for record in decision_records:
        group_name = str(record.get(group_key) or "unknown")
        groups.setdefault(group_name, []).append(record)

    group_metrics: dict[str, JSONDict] = {}
    calibration_gaps: list[float] = []
    approval_rates: list[float] = []

    for group_name, group_records in groups.items():
        labeled_records = [
            record
            for record in group_records
            if record.get("realized_label") is not None
            and record.get("calibrated_probability") is not None
        ]
        live_metrics, _ = _live_model_metrics(labeled_records)
        calibrated_ece = _safe_float(live_metrics.get("ece")) if live_metrics else None
        positive_action_rate = _positive_action_rate(group_records)
        manual_review_rate = _manual_review_rate(group_records)
        action_counts = _action_counts(group_records)
        group_metrics[group_name] = {
            "decision_count": len(group_records),
            "action_counts": action_counts,
            "manual_review_rate": manual_review_rate,
            "positive_action_rate": positive_action_rate,
            "mean_calibrated_probability": _mean_probability(group_records, "calibrated_probability"),
            "calibrated_ece": calibrated_ece,
        }
        if calibrated_ece is not None:
            calibration_gaps.append(calibrated_ece)
        approval_rates.append(positive_action_rate)

    subgroup_calibration_gap = _gap(calibration_gaps)
    approval_rate_gap = _gap(approval_rates)
    audit_log_coverage = _audit_log_coverage(primary_decisions, prediction_ids)

    return {
        "group_key": group_key,
        "group_metrics": group_metrics,
        "subgroup_calibration_gap": subgroup_calibration_gap,
        "approval_rate_gap": approval_rate_gap,
        "audit_log_coverage": audit_log_coverage,
        "primary_decision_count": len(primary_decisions),
    }


def _calibration_snapshot(
    phase4_report: JSONDict,
    live_primary_metrics: JSONDict,
    live_primary_raw_metrics: JSONDict,
) -> JSONDict:
    validation_evaluation = phase4_report.get("validation_evaluation", {})
    baseline_calibrated_metrics = validation_evaluation.get("calibrated_metrics", {})
    baseline_raw_metrics = validation_evaluation.get("raw_metrics", {})

    snapshot: JSONDict = {
        "selected_method": phase4_report.get("selected_method"),
        "selection_metric": phase4_report.get("selection_metric"),
        "baseline_validation_calibrated_metrics": baseline_calibrated_metrics,
        "baseline_validation_raw_metrics": baseline_raw_metrics,
        "test_evaluation": phase4_report.get("test_evaluation", {}),
    }

    if live_primary_metrics:
        snapshot["live_primary_calibrated_metrics"] = live_primary_metrics
        snapshot["live_primary_raw_metrics"] = live_primary_raw_metrics
        snapshot["calibration_ece_delta"] = _safe_float(live_primary_metrics.get("ece")) - _safe_float(
            baseline_calibrated_metrics.get("ece")
        )
        snapshot["calibration_log_loss_delta"] = _safe_float(live_primary_metrics.get("log_loss")) - _safe_float(
            baseline_calibrated_metrics.get("log_loss")
        )
    else:
        snapshot["live_primary_calibrated_metrics"] = {}
        snapshot["live_primary_raw_metrics"] = {}
        snapshot["calibration_ece_delta"] = None
        snapshot["calibration_log_loss_delta"] = None

    return snapshot


def _system_summary(prediction_summary: JSONDict, decision_summary: JSONDict) -> JSONDict:
    prediction_count = int(prediction_summary.get("prediction_count", 0))
    decision_count = int(decision_summary.get("decision_count", 0))
    prediction_ids = prediction_summary.get("prediction_ids", set())
    decision_prediction_ids = decision_summary.get("linked_prediction_ids", set())
    linked_prediction_ids = set(prediction_ids).intersection(decision_prediction_ids)
    linked_prediction_count = len(linked_prediction_ids)
    prediction_link_rate = linked_prediction_count / prediction_count if prediction_count else 0.0
    decision_link_rate = linked_prediction_count / decision_count if decision_count else 0.0
    prediction_error_rate = float(prediction_summary.get("prediction_error_rate", 0.0))
    availability = max(0.0, min(1.0, 1.0 - prediction_error_rate))

    return {
        "prediction_count": prediction_count,
        "decision_count": decision_count,
        "linked_prediction_count": linked_prediction_count,
        "unlinked_decision_count": max(0, decision_count - linked_prediction_count),
        "prediction_link_rate": prediction_link_rate,
        "decision_link_rate": decision_link_rate,
        "prediction_error_rate": prediction_error_rate,
        "availability": availability,
        "prediction_logging_completeness": float(prediction_summary.get("prediction_logging_completeness", 0.0)),
        "prediction_p50_latency_ms": float(prediction_summary.get("prediction_p50_latency_ms", 0.0)),
        "prediction_p95_latency_ms": float(prediction_summary.get("prediction_p95_latency_ms", 0.0)),
        "prediction_p99_latency_ms": float(prediction_summary.get("prediction_p99_latency_ms", 0.0)),
        "manual_review_rate": float(decision_summary.get("manual_review_rate", 0.0)),
        "positive_action_rate": float(decision_summary.get("positive_action_rate", 0.0)),
        "label_delay_mean_days": float(decision_summary.get("label_delay_mean_days", 0.0)),
        "label_delay_p95_days": float(decision_summary.get("label_delay_p95_days", 0.0)),
        "feature_null_rate_proxy": float(prediction_summary.get("feature_null_rate_proxy", 0.0)),
        "feature_freshness_lag_seconds": prediction_summary.get("feature_freshness_lag_seconds"),
    }


def _build_alerts(report: JSONDict, thresholds: MonitoringThresholds) -> list[MonitoringAlert]:
    alerts: list[MonitoringAlert] = []
    summary = report["summary"]
    feature_drift = report["feature_drift"]
    calibration_metrics = report["calibration_metrics"]
    governance = report["governance"]

    if summary["prediction_p95_latency_ms"] > thresholds.prediction_p95_latency_ms_critical:
        alerts.append(
            MonitoringAlert(
                metric="prediction_p95_latency_ms",
                severity="critical",
                value=summary["prediction_p95_latency_ms"],
                threshold=thresholds.prediction_p95_latency_ms_critical,
                message=(
                    "Prediction latency exceeds the serving SLO. "
                    "Investigate caching, feature lookup, or model size."
                ),
            )
        )

    if feature_drift["max_dense_psi"] > thresholds.psi_critical:
        alerts.append(
            MonitoringAlert(
                metric="max_dense_psi",
                severity="critical",
                value=feature_drift["max_dense_psi"],
                threshold=thresholds.psi_critical,
                message="Dense feature drift exceeds the PSI drift threshold.",
            )
        )
    if feature_drift["sparse_bucket_psi"] > thresholds.psi_critical:
        alerts.append(
            MonitoringAlert(
                metric="sparse_bucket_psi",
                severity="critical",
                value=feature_drift["sparse_bucket_psi"],
                threshold=thresholds.psi_critical,
                message="Sparse feature drift exceeds the PSI drift threshold.",
            )
        )

    live_calibrated_metrics = calibration_metrics.get("live_primary_calibrated_metrics") or {}
    if live_calibrated_metrics and _safe_float(live_calibrated_metrics.get("ece")) > thresholds.ece_critical:
        alerts.append(
            MonitoringAlert(
                metric="live_primary_calibrated_metrics.ece",
                severity="critical",
                value=_safe_float(live_calibrated_metrics.get("ece")),
                threshold=thresholds.ece_critical,
                message="Live calibration error exceeds the ECE investigation threshold.",
            )
        )

    if summary["manual_review_rate"] > thresholds.manual_review_utilization_critical:
        alerts.append(
            MonitoringAlert(
                metric="manual_review_rate",
                severity="critical",
                value=summary["manual_review_rate"],
                threshold=thresholds.manual_review_utilization_critical,
                message="Manual-review utilization is above the operational limit.",
            )
        )

    if governance["subgroup_calibration_gap"] > thresholds.subgroup_calibration_gap_critical:
        alerts.append(
            MonitoringAlert(
                metric="subgroup_calibration_gap",
                severity="critical",
                value=governance["subgroup_calibration_gap"],
                threshold=thresholds.subgroup_calibration_gap_critical,
                message="Calibration gap across groups is above the fairness threshold.",
            )
        )

    if governance["approval_rate_gap"] > thresholds.approval_rate_gap_critical:
        alerts.append(
            MonitoringAlert(
                metric="approval_rate_gap",
                severity="critical",
                value=governance["approval_rate_gap"],
                threshold=thresholds.approval_rate_gap_critical,
                message="Approval-rate gap across groups is above the fairness threshold.",
            )
        )

    if summary["prediction_link_rate"] < 0.99:
        alerts.append(
            MonitoringAlert(
                metric="audit_log_coverage",
                severity="critical",
                value=summary["prediction_link_rate"],
                threshold=0.99,
                message="Prediction logs are missing linked decision outcomes.",
            )
        )

    return alerts


def _rollout_readiness(alerts: Sequence[JSONDict], summary: JSONDict) -> JSONDict:
    critical_alerts = [alert for alert in alerts if alert.get("severity") == "critical"]
    ready = not critical_alerts and summary["prediction_link_rate"] >= 0.99
    reasons = [str(alert["message"]) for alert in critical_alerts]
    if summary["prediction_link_rate"] < 0.99:
        reasons.append("Prediction logs do not link cleanly to decision outcomes.")
    return {"ready": ready, "reasons": reasons}


def population_stability_index(
    expected_values: Sequence[float],
    actual_values: Sequence[float],
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Compute a simple PSI over equal-width bins."""

    expected = [float(value) for value in expected_values]
    actual = [float(value) for value in actual_values]
    if not expected or not actual:
        return 0.0
    minimum = min(min(expected), min(actual))
    maximum = max(max(expected), max(actual))
    if minimum == maximum:
        return 0.0
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")

    bin_edges = [minimum + (maximum - minimum) * index / n_bins for index in range(1, n_bins)]
    expected_hist = _histogram(expected, bin_edges)
    actual_hist = _histogram(actual, bin_edges)
    total_expected = sum(expected_hist)
    total_actual = sum(actual_hist)
    if total_expected == 0.0 or total_actual == 0.0:
        return 0.0

    psi = 0.0
    for expected_count, actual_count in zip(expected_hist, actual_hist, strict=True):
        expected_rate = max(expected_count / total_expected, epsilon)
        actual_rate = max(actual_count / total_actual, epsilon)
        psi += (actual_rate - expected_rate) * math.log(actual_rate / expected_rate)
    return psi


def load_and_write_monitoring_report(
    inputs: MonitoringInputs,
    thresholds: MonitoringThresholds | None = None,
) -> MonitoringRunResult:
    """Compatibility wrapper for CLI use."""

    return run_monitoring(inputs, thresholds=thresholds)


def _histogram(values: Sequence[float], bin_edges: Sequence[float]) -> list[float]:
    counts = [0.0 for _ in range(len(bin_edges) + 1)]
    for value in values:
        index = 0
        while index < len(bin_edges) and value > bin_edges[index]:
            index += 1
        counts[index] += 1.0
    return counts


def _parse_alert_thresholds(path: Path) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    in_alerts_block = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "alerts:":
            in_alerts_block = True
            continue
        if not line.startswith(" ") and not line.startswith("\t"):
            in_alerts_block = False
        if in_alerts_block and ":" in stripped:
            key, value = stripped.split(":", maxsplit=1)
            key = key.strip()
            value = value.strip()
            if value:
                thresholds[key] = float(value)
    return thresholds


def _safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]

    rank = (percentile / 100.0) * (len(ordered) - 1)
    lower_index = int(math.floor(rank))
    upper_index = int(math.ceil(rank))
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = rank - lower_index
    return ordered[lower_index] * (1.0 - fraction) + ordered[upper_index] * fraction


def _zero_rate(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    zero_count = sum(1 for value in values if float(value) == 0.0)
    return zero_count / len(values)


def _action_counts(records: Sequence[JSONDict]) -> dict[str, int]:
    action_counts: dict[str, int] = {}
    for record in records:
        action = str(record.get("action", "unknown"))
        action_counts[action] = action_counts.get(action, 0) + 1
    return action_counts


def _manual_review_rate(records: Sequence[JSONDict]) -> float:
    if not records:
        return 0.0
    manual_review_count = sum(1 for record in records if bool(record.get("manual_review_required")))
    return manual_review_count / len(records)


def _positive_action_rate(records: Sequence[JSONDict]) -> float:
    if not records:
        return 0.0
    positive_action_count = sum(1 for record in records if _is_positive_action(record))
    return positive_action_count / len(records)


def _mean_probability(records: Sequence[JSONDict], field_name: str) -> float:
    values = [float(record[field_name]) for record in records if record.get(field_name) is not None]
    return mean(values) if values else 0.0


def _is_positive_action(record: JSONDict) -> bool:
    context = str(record.get("task_context", "")).strip().lower()
    action = str(record.get("action", "")).strip().lower()
    positive_actions = POSITIVE_ACTIONS_BY_CONTEXT.get(context, set())
    return action in positive_actions


def _audit_log_coverage(primary_decisions: Sequence[JSONDict], prediction_ids: set[str]) -> float:
    if not primary_decisions:
        return 0.0
    linked_primary = sum(
        1
        for decision in primary_decisions
        if decision.get("prediction_id") is not None and str(decision["prediction_id"]) in prediction_ids
    )
    return linked_primary / len(primary_decisions)


def _gap(values: Sequence[float]) -> float:
    filtered = [float(value) for value in values if value is not None]
    if len(filtered) < 2:
        return 0.0
    return max(filtered) - min(filtered)


def _write_json(path: Path, payload: JSONDict | list[JSONDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
