"""Decision routing policies for Phase 4."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class BankMarketingPolicyConfig:
    """Threshold policy for propensity ranking."""

    policy_version: str = "bank_marketing_policy_v1"
    low_propensity_threshold: float = 0.08
    high_propensity_threshold: float = 0.22
    below_low_action: str = "suppress"
    between_action: str = "nurture"
    above_high_action: str = "prioritize_contact"


@dataclass(frozen=True)
class FraudPolicyConfig:
    """Threshold policy for fraud routing."""

    policy_version: str = "fraud_policy_v1"
    manual_review_threshold: float = 0.15
    auto_block_threshold: float = 0.65
    approve_action: str = "approve"
    review_action: str = "review"
    block_action: str = "block"


@dataclass(frozen=True)
class LoanPolicyConfig:
    """Threshold policy for loan risk routing."""

    policy_version: str = "loan_policy_v1"
    manual_review_threshold: float = 0.12
    decline_threshold: float = 0.18
    approve_action: str = "approve"
    review_action: str = "review"
    decline_action: str = "decline"


@dataclass(frozen=True)
class DecisionOutcome:
    """Single decision record compatible with the project contract."""

    decision_id: str
    decision_date: str
    decision_ts: str
    prediction_id: str
    task_context: str
    action: str
    action_reason: str | None
    threshold_value: float | None
    manual_review_required: bool
    realized_label: int | None = None
    realized_value: float | None = None
    outcome_delay_days: int | None = None
    calibrated_probability: float | None = None
    raw_probability: float | None = None
    decision_policy_version: str | None = None

    def to_contract_dict(self) -> dict[str, object]:
        """Return the compact contract-aligned payload."""

        return {
            "decision_id": self.decision_id,
            "decision_date": self.decision_date,
            "decision_ts": self.decision_ts,
            "prediction_id": self.prediction_id,
            "task_context": self.task_context,
            "action": self.action,
            "action_reason": self.action_reason,
            "threshold_value": self.threshold_value,
            "manual_review_required": self.manual_review_required,
            "realized_label": self.realized_label,
            "realized_value": self.realized_value,
            "outcome_delay_days": self.outcome_delay_days,
        }

    def to_dict(self) -> dict[str, object]:
        """Return the full internal representation."""

        payload = self.to_contract_dict()
        payload.update(
            {
                "calibrated_probability": self.calibrated_probability,
                "raw_probability": self.raw_probability,
                "decision_policy_version": self.decision_policy_version,
            }
        )
        return payload


def route_decision(
    task_context: str,
    calibrated_probability: float,
    event_id: str,
    event_ts: str,
    prediction_id: str | None = None,
    raw_probability: float | None = None,
    realized_label: int | None = None,
    realized_value: float | None = None,
    outcome_delay_days: int | None = 0,
) -> DecisionOutcome:
    """Apply the appropriate routing policy for one example."""

    normalized_context = task_context.strip().lower()
    prediction_identifier = prediction_id or f"prediction_{event_id}"
    decision_identifier = f"decision_{event_id}"
    decision_timestamp = _normalize_timestamp(event_ts)
    decision_date = decision_timestamp.split("T", maxsplit=1)[0]

    if normalized_context == "bank_marketing":
        config = BankMarketingPolicyConfig()
        action, action_reason, threshold_value, manual_review_required = _route_bank_marketing(
            calibrated_probability,
            config,
        )
        policy_version = config.policy_version
    elif normalized_context == "fraud_policy":
        config = FraudPolicyConfig()
        action, action_reason, threshold_value, manual_review_required = _route_fraud(
            calibrated_probability,
            config,
        )
        policy_version = config.policy_version
    elif normalized_context == "loan_policy":
        config = LoanPolicyConfig()
        action, action_reason, threshold_value, manual_review_required = _route_loan(
            calibrated_probability,
            config,
        )
        policy_version = config.policy_version
    else:
        raise ValueError(f"Unknown task_context: {task_context!r}")

    return DecisionOutcome(
        decision_id=decision_identifier,
        decision_date=decision_date,
        decision_ts=decision_timestamp,
        prediction_id=prediction_identifier,
        task_context=normalized_context,
        action=action,
        action_reason=action_reason,
        threshold_value=threshold_value,
        manual_review_required=manual_review_required,
        realized_label=realized_label,
        realized_value=realized_value,
        outcome_delay_days=outcome_delay_days,
        calibrated_probability=calibrated_probability,
        raw_probability=raw_probability,
        decision_policy_version=policy_version,
    )


def evaluate_decision_policy(
    task_context: str,
    calibrated_probabilities: Sequence[float],
    event_ids: Sequence[str],
    event_timestamps: Sequence[str],
    raw_probabilities: Sequence[float] | None = None,
    labels: Sequence[int] | None = None,
    realized_values: Sequence[float] | None = None,
) -> tuple[list[DecisionOutcome], dict[str, object]]:
    """Route a batch of decisions and summarize the actions taken."""

    if len(calibrated_probabilities) != len(event_ids) or len(event_ids) != len(event_timestamps):
        raise ValueError("probabilities, event_ids, and event_timestamps must have the same length.")
    if raw_probabilities is not None and len(raw_probabilities) != len(event_ids):
        raise ValueError("raw_probabilities must match the number of events.")
    if labels is not None and len(labels) != len(event_ids):
        raise ValueError("labels must match the number of events.")
    if realized_values is not None and len(realized_values) != len(event_ids):
        raise ValueError("realized_values must match the number of events.")

    outcomes: list[DecisionOutcome] = []
    for index, (probability, event_id, event_ts) in enumerate(
        zip(calibrated_probabilities, event_ids, event_timestamps, strict=True)
    ):
        outcome = route_decision(
            task_context=task_context,
            calibrated_probability=probability,
            event_id=event_id,
            event_ts=event_ts,
            raw_probability=None if raw_probabilities is None else raw_probabilities[index],
            realized_label=None if labels is None else labels[index],
            realized_value=None if realized_values is None else realized_values[index],
        )
        outcomes.append(outcome)

    summary = summarize_decision_outcomes(outcomes)
    return outcomes, summary


def summarize_decision_outcomes(outcomes: Sequence[DecisionOutcome]) -> dict[str, object]:
    """Aggregate a batch of outcomes into a policy report."""

    if not outcomes:
        return {
            "decision_count": 0,
            "action_counts": {},
            "manual_review_rate": 0.0,
            "positive_rate": 0.0,
            "action_positive_rates": {},
            "action_mean_probability": {},
        }

    action_counts = Counter(outcome.action for outcome in outcomes)
    action_label_totals: dict[str, float] = defaultdict(float)
    action_probability_totals: dict[str, float] = defaultdict(float)
    action_example_counts: dict[str, int] = defaultdict(int)
    total_labels = 0.0
    total_probabilities = 0.0
    manual_review_count = 0

    for outcome in outcomes:
        action_example_counts[outcome.action] += 1
        if outcome.realized_label is not None:
            action_label_totals[outcome.action] += outcome.realized_label
            total_labels += outcome.realized_label
        if outcome.calibrated_probability is not None:
            action_probability_totals[outcome.action] += outcome.calibrated_probability
            total_probabilities += outcome.calibrated_probability
        if outcome.manual_review_required:
            manual_review_count += 1

    action_positive_rates = {
        action: _safe_rate(action_label_totals[action], action_example_counts[action])
        for action in action_counts
    }
    action_mean_probability = {
        action: _safe_rate(action_probability_totals[action], action_example_counts[action])
        for action in action_counts
    }

    return {
        "decision_count": len(outcomes),
        "action_counts": dict(action_counts),
        "manual_review_rate": manual_review_count / len(outcomes),
        "positive_rate": _safe_rate(total_labels, len(outcomes)),
        "mean_calibrated_probability": _safe_rate(total_probabilities, len(outcomes)),
        "action_positive_rates": action_positive_rates,
        "action_mean_probability": action_mean_probability,
    }


def save_decision_outcomes(outcomes: Sequence[DecisionOutcome], output_path: Path) -> Path:
    """Persist decision outcomes as JSONL."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for outcome in outcomes:
            handle.write(json.dumps(outcome.to_contract_dict(), separators=(",", ":")))
            handle.write("\n")
    return output_path


def save_decision_summary(summary: dict[str, object], output_path: Path) -> Path:
    """Persist a policy summary as JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return output_path


def _route_bank_marketing(
    calibrated_probability: float,
    config: BankMarketingPolicyConfig,
) -> tuple[str, str, float | None, bool]:
    if calibrated_probability < config.low_propensity_threshold:
        return config.below_low_action, "below_low_propensity", config.low_propensity_threshold, False
    if calibrated_probability < config.high_propensity_threshold:
        return config.between_action, "between_propensity_thresholds", config.low_propensity_threshold, False
    return config.above_high_action, "above_high_propensity", config.high_propensity_threshold, False


def _route_fraud(
    calibrated_probability: float,
    config: FraudPolicyConfig,
) -> tuple[str, str, float | None, bool]:
    if calibrated_probability < config.manual_review_threshold:
        return config.approve_action, "below_manual_review_threshold", config.manual_review_threshold, False
    if calibrated_probability < config.auto_block_threshold:
        return config.review_action, "between_manual_review_and_auto_block", config.manual_review_threshold, True
    return config.block_action, "above_auto_block_threshold", config.auto_block_threshold, True


def _route_loan(
    calibrated_probability: float,
    config: LoanPolicyConfig,
) -> tuple[str, str, float | None, bool]:
    if calibrated_probability < config.manual_review_threshold:
        return config.approve_action, "below_manual_review_threshold", config.manual_review_threshold, False
    if calibrated_probability < config.decline_threshold:
        return config.review_action, "between_manual_review_and_decline", config.manual_review_threshold, True
    return config.decline_action, "above_decline_threshold", config.decline_threshold, True


def _normalize_timestamp(value: str) -> str:
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _safe_rate(numerator: float, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
