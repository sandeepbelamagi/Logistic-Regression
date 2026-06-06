"""Phase 7 hybrid reranking and exploration helpers."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from probabilistic_decisioning.bank_marketing import BankMarketingRecord
from probabilistic_decisioning.calibration import evaluate_calibrated_model, load_calibration_artifact, load_selected_calibrator
from probabilistic_decisioning.constants import BANK_MARKETING_DENSE_FEATURE_NAMES
from probabilistic_decisioning.logistic_regression import (
    LogisticRegressionModel,
    LogisticRegressionTrainingConfig,
    TrainingExample,
    load_model_artifact,
    load_training_splits,
    save_training_artifacts,
    train_logistic_regression,
)
from probabilistic_decisioning.metrics import classification_metric_report


HYBRID_RERANKER_DENSE_FEATURE_NAMES: tuple[str, ...] = (
    "stage1_raw_score",
    "stage1_probability",
    "stage1_calibrated_probability",
    "stage1_uncertainty",
    "age",
    "balance",
    "day",
    "campaign",
    "pdays",
    "previous",
    "prior_contact_flag",
    "age_x_balance",
    "balance_x_campaign",
    "campaign_x_previous",
    "pdays_x_prior_contact",
    "balance_x_prior_contact",
    "raw_score_x_balance",
    "calibrated_probability_x_campaign",
    "uncertainty_x_previous",
)


@dataclass(frozen=True)
class HybridRankingInputs:
    """Paths required to train and evaluate the hybrid reranker."""

    training_data_dir: Path
    model_path: Path
    calibration_path: Path
    output_dir: Path


@dataclass(frozen=True)
class HybridRankingConfig:
    """Training and policy configuration for the hybrid ranking system."""

    top_k: int = 25
    exploration_rate: float = 0.15
    reranker_model_version: str = "bank_marketing_hybrid_reranker_v1"
    reranker_feature_set_version: str = "bank_marketing_hybrid_v1"
    reranker_task_context: str = "hybrid_ranking"
    reranker_hash_dimension: int = 262_144
    reranker_learning_rate: float = 0.05
    reranker_max_epochs: int = 10
    reranker_l2: float = 0.0005
    reranker_seed: int = 31
    reranker_early_stopping_patience: int = 3

    def validate(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive.")
        if not 0.0 <= self.exploration_rate <= 1.0:
            raise ValueError("exploration_rate must be between 0 and 1.")
        if self.reranker_learning_rate <= 0.0:
            raise ValueError("reranker_learning_rate must be positive.")
        if self.reranker_max_epochs <= 0:
            raise ValueError("reranker_max_epochs must be positive.")
        if self.reranker_l2 < 0.0:
            raise ValueError("reranker_l2 must be non-negative.")
        if self.reranker_early_stopping_patience < 0:
            raise ValueError("reranker_early_stopping_patience must be non-negative.")


@dataclass
class HybridCandidateRecord:
    """One candidate scored by the stage-1 and stage-2 rankers."""

    candidate_id: str
    split_name: str
    training_row_id: str
    event_id: str
    label: int
    stage1_raw_score: float
    stage1_probability: float
    stage1_calibrated_probability: float
    reranker_probability: float
    uncertainty_score: float
    hybrid_feature_values: tuple[float, ...]
    exploration_bonus: float = 0.0
    final_score: float = 0.0
    stage1_rank: int = 0
    reranker_rank: int = 0
    final_rank: int = 0
    was_explored: bool = False

    def to_preview_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "split_name": self.split_name,
            "training_row_id": self.training_row_id,
            "event_id": self.event_id,
            "label": self.label,
            "stage1_raw_score": self.stage1_raw_score,
            "stage1_probability": self.stage1_probability,
            "stage1_calibrated_probability": self.stage1_calibrated_probability,
            "reranker_probability": self.reranker_probability,
            "uncertainty_score": self.uncertainty_score,
            "exploration_bonus": self.exploration_bonus,
            "final_score": self.final_score,
            "stage1_rank": self.stage1_rank,
            "reranker_rank": self.reranker_rank,
            "final_rank": self.final_rank,
            "was_explored": self.was_explored,
            "hybrid_features": {
                feature_name: feature_value
                for feature_name, feature_value in zip(
                    HYBRID_RERANKER_DENSE_FEATURE_NAMES,
                    self.hybrid_feature_values,
                    strict=True,
                )
            },
        }


@dataclass(frozen=True)
class HybridRankingRunResult:
    """Artifacts produced by the hybrid ranking pipeline."""

    report: dict[str, object]
    report_path: Path
    reranker_artifacts: dict[str, Path]
    ranked_candidate_paths: dict[str, Path]


def run_hybrid_ranking(
    inputs: HybridRankingInputs,
    config: HybridRankingConfig | None = None,
) -> HybridRankingRunResult:
    """Train the reranker and evaluate hybrid reranking plus exploration."""

    config = config or HybridRankingConfig()
    config.validate()

    model = load_model_artifact(inputs.model_path)
    calibration_artifact = load_calibration_artifact(inputs.calibration_path)
    if calibration_artifact.model_version != model.model_version:
        raise ValueError(
            "Calibration artifact model_version does not match the Phase 3 model artifact. "
            f"Model={model.model_version!r}, calibration={calibration_artifact.model_version!r}."
        )
    if calibration_artifact.feature_set_version != model.feature_set_version:
        raise ValueError(
            "Calibration artifact feature_set_version does not match the Phase 3 model artifact. "
            f"Model={model.feature_set_version!r}, calibration={calibration_artifact.feature_set_version!r}."
        )
    if calibration_artifact.task_context != model.task_context:
        raise ValueError(
            "Calibration artifact task_context does not match the Phase 3 model artifact. "
            f"Model={model.task_context!r}, calibration={calibration_artifact.task_context!r}."
        )
    calibrator = load_selected_calibrator(calibration_artifact)
    splits = load_training_splits(inputs.training_data_dir)

    hybrid_train, hybrid_validation, hybrid_test = _build_hybrid_splits(
        splits,
        base_model=model,
        calibrator=calibrator,
        feature_set_version=config.reranker_feature_set_version,
        task_context=config.reranker_task_context,
    )

    reranker_config = LogisticRegressionTrainingConfig(
        loss="cross_entropy",
        optimizer="adagrad",
        learning_rate=config.reranker_learning_rate,
        max_epochs=config.reranker_max_epochs,
        l1=0.0,
        l2=config.reranker_l2,
        class_weighting=True,
        oversampling=False,
        seed=config.reranker_seed,
        early_stopping_patience=config.reranker_early_stopping_patience,
    )
    reranker_result = train_logistic_regression(
        hybrid_train,
        hybrid_validation,
        hybrid_test,
        reranker_config,
        model_version=config.reranker_model_version,
        feature_set_version=config.reranker_feature_set_version,
        task_context=config.reranker_task_context,
        hash_dimension=config.reranker_hash_dimension,
        dense_feature_names=HYBRID_RERANKER_DENSE_FEATURE_NAMES,
    )
    reranker_dir = inputs.output_dir / "reranker"
    reranker_artifacts = save_training_artifacts(reranker_result, reranker_dir)

    ranking_dir = inputs.output_dir / "rankings"
    ranking_dir.mkdir(parents=True, exist_ok=True)

    split_reports: dict[str, dict[str, object]] = {}
    ranked_candidate_paths: dict[str, Path] = {}
    for split_name in ("train", "validation", "test"):
        split_report, ranked_candidates = _rank_split(
            split_name=split_name,
            examples=splits[split_name],
            hybrid_examples={
                "train": hybrid_train,
                "validation": hybrid_validation,
                "test": hybrid_test,
            }[split_name],
            base_model=model,
            calibrator=calibrator,
            reranker_model=reranker_result.model,
            top_k=config.top_k,
            exploration_rate=config.exploration_rate,
        )
        split_reports[split_name] = split_report
        ranked_candidate_path = ranking_dir / f"{split_name}_top_k.jsonl"
        _write_jsonl(ranked_candidate_path, [candidate.to_preview_dict() for candidate in ranked_candidates])
        ranked_candidate_paths[split_name] = ranked_candidate_path
        split_report["ranked_candidate_path"] = str(ranked_candidate_path)

    overall_report = _overall_summary(split_reports)
    rollout_readiness = _rollout_readiness(split_reports)
    generated_at = datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")

    report: dict[str, object] = {
        "generated_at": generated_at,
        "hybrid_version": "bank_marketing_hybrid_ranking_v1",
        "artifacts": {
            "model_path": str(inputs.model_path),
            "calibration_path": str(inputs.calibration_path),
            "training_data_dir": str(inputs.training_data_dir),
            "reranker_model_path": str(reranker_artifacts["model_path"]),
            "reranker_metrics_path": str(reranker_artifacts["metrics_path"]),
            "reranker_history_path": str(reranker_artifacts["history_path"]),
        },
        "base_model": {
            "model_version": model.model_version,
            "feature_set_version": model.feature_set_version,
            "task_context": model.task_context,
            "calibration_version": calibration_artifact.calibration_version,
        },
        "reranker": {
            "model_version": reranker_result.model.model_version,
            "feature_set_version": reranker_result.model.feature_set_version,
            "task_context": reranker_result.model.task_context,
            "dense_feature_names": list(reranker_result.model.dense_feature_names),
            "training_config": asdict(reranker_result.training_config),
            "train_metrics": reranker_result.train_metrics,
            "validation_metrics": reranker_result.validation_metrics,
            "test_metrics": reranker_result.test_metrics,
            "dataset_summary": reranker_result.dataset_summary,
        },
        "configuration": asdict(config),
        "splits": split_reports,
        "overall": overall_report,
        "rollout_readiness": rollout_readiness,
    }

    inputs.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = inputs.output_dir / "hybrid_ranking_report.json"
    _write_json(report_path, report)

    return HybridRankingRunResult(
        report=report,
        report_path=report_path,
        reranker_artifacts=reranker_artifacts,
        ranked_candidate_paths=ranked_candidate_paths,
    )


def _build_hybrid_splits(
    splits: dict[str, list[TrainingExample]],
    base_model: LogisticRegressionModel,
    calibrator: Any,
    feature_set_version: str,
    task_context: str,
) -> tuple[list[TrainingExample], list[TrainingExample], list[TrainingExample]]:
    return tuple(
        _build_hybrid_examples(examples, base_model, calibrator, feature_set_version, task_context)
        for examples in (splits["train"], splits["validation"], splits["test"])
    )  # type: ignore[return-value]


def _build_hybrid_examples(
    examples: Sequence[TrainingExample],
    base_model: LogisticRegressionModel,
    calibrator: Any,
    feature_set_version: str,
    task_context: str,
) -> list[TrainingExample]:
    hybrid_examples: list[TrainingExample] = []
    for example in examples:
        hybrid_feature_values = _build_hybrid_feature_values(example, base_model, calibrator)
        hybrid_examples.append(
            TrainingExample(
                training_row_id=example.training_row_id,
                event_id=example.event_id,
                snapshot_date=example.snapshot_date,
                event_ts=example.event_ts,
                label_ts=example.label_ts,
                label=example.label,
                sample_weight=example.sample_weight,
                dense_features=hybrid_feature_values,
                sparse_feature_ids=(),
                sparse_feature_values=(),
                feature_set_version=feature_set_version,
                task_context=task_context,
            )
        )
    return hybrid_examples


def _build_hybrid_feature_values(
    example: TrainingExample,
    base_model: LogisticRegressionModel,
    calibrator: Any,
) -> tuple[float, ...]:
    raw_score = base_model.score(example)
    stage1_probability = base_model.predict_proba(example)
    stage1_calibrated_probability = calibrator.predict(raw_score)
    stage1_uncertainty = 1.0 - abs(2.0 * stage1_calibrated_probability - 1.0)
    dense_values = {
        feature_name: feature_value
        for feature_name, feature_value in zip(
            BANK_MARKETING_DENSE_FEATURE_NAMES,
            example.dense_features,
            strict=True,
        )
    }

    return (
        raw_score,
        stage1_probability,
        stage1_calibrated_probability,
        stage1_uncertainty,
        dense_values["age"],
        dense_values["balance"],
        dense_values["day"],
        dense_values["campaign"],
        dense_values["pdays"],
        dense_values["previous"],
        dense_values["prior_contact_flag"],
        _compress_interaction(dense_values["age"] * dense_values["balance"]),
        _compress_interaction(dense_values["balance"] * dense_values["campaign"]),
        _compress_interaction(dense_values["campaign"] * dense_values["previous"]),
        _compress_interaction(dense_values["pdays"] * dense_values["prior_contact_flag"]),
        _compress_interaction(dense_values["balance"] * dense_values["prior_contact_flag"]),
        _compress_interaction(raw_score * dense_values["balance"]),
        _compress_interaction(stage1_calibrated_probability * dense_values["campaign"]),
        _compress_interaction(stage1_uncertainty * dense_values["previous"]),
    )


def _rank_split(
    *,
    split_name: str,
    examples: Sequence[TrainingExample],
    hybrid_examples: Sequence[TrainingExample],
    base_model: LogisticRegressionModel,
    calibrator: Any,
    reranker_model: LogisticRegressionModel,
    top_k: int,
    exploration_rate: float,
) -> tuple[dict[str, object], list[HybridCandidateRecord]]:
    stage1_evaluation = evaluate_calibrated_model(base_model, calibrator, examples)
    reranker_evaluation = {
        "example_count": len(hybrid_examples),
        "metrics": classification_metric_report(
            [example.label for example in hybrid_examples],
            [reranker_model.predict_proba(example) for example in hybrid_examples],
            sample_weights=[example.sample_weight for example in hybrid_examples],
        ),
    }

    candidates = _build_candidate_records(split_name, examples, hybrid_examples, base_model, calibrator, reranker_model)
    stage1_sorted = sorted(
        candidates,
        key=lambda candidate: (
            -candidate.stage1_calibrated_probability,
            -candidate.stage1_raw_score,
            candidate.candidate_id,
        ),
    )
    reranker_sorted = sorted(
        candidates,
        key=lambda candidate: (
            -candidate.reranker_probability,
            -candidate.stage1_calibrated_probability,
            candidate.candidate_id,
        ),
    )

    stage1_rank_map = {candidate.candidate_id: index + 1 for index, candidate in enumerate(stage1_sorted)}
    reranker_rank_map = {candidate.candidate_id: index + 1 for index, candidate in enumerate(reranker_sorted)}
    for candidate in candidates:
        candidate.stage1_rank = stage1_rank_map[candidate.candidate_id]
        candidate.reranker_rank = reranker_rank_map[candidate.candidate_id]

    final_candidates = _select_final_candidates(candidates, top_k, exploration_rate)
    final_rank_map = {candidate.candidate_id: index + 1 for index, candidate in enumerate(final_candidates)}
    for candidate in final_candidates:
        candidate.final_rank = final_rank_map[candidate.candidate_id]

    top_k_effective = min(top_k, len(candidates))
    stage1_top_k = stage1_sorted[:top_k_effective]
    reranker_top_k = reranker_sorted[:top_k_effective]
    final_top_k = final_candidates[:top_k_effective]

    ranking_metrics = _ranking_metrics(
        candidates=candidates,
        stage1_top_k=stage1_top_k,
        reranker_top_k=reranker_top_k,
        final_top_k=final_top_k,
        top_k=top_k_effective,
        exploration_rate=exploration_rate,
    )

    split_report = {
        "split_name": split_name,
        "candidate_count": len(candidates),
        "top_k": top_k_effective,
        "stage1_evaluation": stage1_evaluation,
        "reranker_evaluation": reranker_evaluation,
        "ranking_metrics": ranking_metrics,
        "selected_candidate_count": len(final_top_k),
    }
    return split_report, final_top_k


def _build_candidate_records(
    split_name: str,
    examples: Sequence[TrainingExample],
    hybrid_examples: Sequence[TrainingExample],
    base_model: LogisticRegressionModel,
    calibrator: Any,
    reranker_model: LogisticRegressionModel,
) -> list[HybridCandidateRecord]:
    candidates: list[HybridCandidateRecord] = []
    for example, hybrid_example in zip(examples, hybrid_examples, strict=True):
        raw_score = base_model.score(example)
        stage1_probability = base_model.predict_proba(example)
        stage1_calibrated_probability = calibrator.predict(raw_score)
        reranker_probability = reranker_model.predict_proba(hybrid_example)
        uncertainty_score = 1.0 - abs(2.0 * stage1_calibrated_probability - 1.0)
        candidates.append(
            HybridCandidateRecord(
                candidate_id=example.training_row_id,
                split_name=split_name,
                training_row_id=example.training_row_id,
                event_id=example.event_id,
                label=example.label,
                stage1_raw_score=raw_score,
                stage1_probability=stage1_probability,
                stage1_calibrated_probability=stage1_calibrated_probability,
                reranker_probability=reranker_probability,
                uncertainty_score=uncertainty_score,
                hybrid_feature_values=hybrid_example.dense_features,
            )
        )
    return candidates


def _select_final_candidates(
    candidates: Sequence[HybridCandidateRecord],
    top_k: int,
    exploration_rate: float,
) -> list[HybridCandidateRecord]:
    if not candidates:
        return []

    ranked_by_reranker = sorted(
        candidates,
        key=lambda candidate: (
            -candidate.reranker_probability,
            -candidate.stage1_calibrated_probability,
            candidate.candidate_id,
        ),
    )

    top_k_effective = min(top_k, len(candidates))
    exploration_slots = min(top_k_effective, int(round(top_k_effective * exploration_rate)))
    exploit_slots = max(0, top_k_effective - exploration_slots)

    exploit_candidates = ranked_by_reranker[:exploit_slots]
    exploit_ids = {candidate.candidate_id for candidate in exploit_candidates}
    exploration_pool = [candidate for candidate in ranked_by_reranker if candidate.candidate_id not in exploit_ids]
    exploration_pool = sorted(
        exploration_pool,
        key=lambda candidate: (
            -candidate.uncertainty_score,
            -candidate.reranker_probability,
            candidate.candidate_id,
        ),
    )
    exploration_candidates = exploration_pool[:exploration_slots]
    exploration_ids = {candidate.candidate_id for candidate in exploration_candidates}

    selected_by_id = {
        candidate.candidate_id: candidate
        for candidate in exploit_candidates + exploration_candidates
    }
    for candidate in selected_by_id.values():
        candidate.was_explored = candidate.candidate_id in exploration_ids
        candidate.exploration_bonus = candidate.uncertainty_score * exploration_rate if candidate.was_explored else 0.0
        candidate.final_score = candidate.reranker_probability + candidate.exploration_bonus

    return sorted(
        selected_by_id.values(),
        key=lambda candidate: (
            -candidate.final_score,
            -candidate.reranker_probability,
            candidate.candidate_id,
        ),
    )


def _ranking_metrics(
    *,
    candidates: Sequence[HybridCandidateRecord],
    stage1_top_k: Sequence[HybridCandidateRecord],
    reranker_top_k: Sequence[HybridCandidateRecord],
    final_top_k: Sequence[HybridCandidateRecord],
    top_k: int,
    exploration_rate: float,
) -> dict[str, object]:
    stage1_top_ids = {candidate.candidate_id for candidate in stage1_top_k}
    reranker_top_ids = {candidate.candidate_id for candidate in reranker_top_k}
    final_top_ids = {candidate.candidate_id for candidate in final_top_k}
    top_k_effective = len(final_top_k)

    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    stage1_rank_map = {candidate.candidate_id: candidate.stage1_rank for candidate in candidates}
    reranker_rank_map = {candidate.candidate_id: candidate.reranker_rank for candidate in candidates}

    return {
        "top_k": top_k,
        "exploration_rate_configured": exploration_rate,
        "exploration_slots": sum(1 for candidate in final_top_k if candidate.was_explored),
        "stage1_top_k_positive_rate": _positive_rate(stage1_top_k),
        "reranker_top_k_positive_rate": _positive_rate(reranker_top_k),
        "final_top_k_positive_rate": _positive_rate(final_top_k),
        "stage1_ndcg_at_k": _ndcg_at_k(stage1_top_k),
        "reranker_ndcg_at_k": _ndcg_at_k(reranker_top_k),
        "final_ndcg_at_k": _ndcg_at_k(final_top_k),
        "stage1_expected_positive_count": sum(candidate.stage1_calibrated_probability for candidate in stage1_top_k),
        "reranker_expected_positive_count": sum(candidate.reranker_probability for candidate in reranker_top_k),
        "final_mean_score": mean(candidate.final_score for candidate in final_top_k) if final_top_k else 0.0,
        "stage1_overlap_with_final": _overlap_rate(stage1_top_ids, final_top_ids, top_k_effective),
        "reranker_overlap_with_final": _overlap_rate(reranker_top_ids, final_top_ids, top_k_effective),
        "mean_rank_shift_vs_stage1": _mean_rank_shift(stage1_rank_map, final_top_k),
        "mean_rank_shift_vs_reranker": _mean_rank_shift(reranker_rank_map, final_top_k),
        "exploration_positive_rate": _positive_rate([candidate for candidate in final_top_k if candidate.was_explored]),
        "exploit_positive_rate": _positive_rate([candidate for candidate in final_top_k if not candidate.was_explored]),
        "lift_vs_stage1": _positive_rate(final_top_k) - _positive_rate(stage1_top_k),
        "lift_vs_reranker": _positive_rate(final_top_k) - _positive_rate(reranker_top_k),
        "selected_candidate_ids": [candidate.candidate_id for candidate in final_top_k],
    }


def _overall_summary(split_reports: dict[str, dict[str, object]]) -> dict[str, object]:
    weights = [float(report.get("candidate_count", 0)) or 1.0 for report in split_reports.values()]

    def weighted_average(key_path: Sequence[str]) -> float:
        values: list[float] = []
        local_weights: list[float] = []
        for report, weight in zip(split_reports.values(), weights, strict=True):
            value = report
            for part in key_path:
                value = value[part]  # type: ignore[index]
            values.append(float(value))
            local_weights.append(weight)
        return _weighted_mean(values, local_weights)

    validation_report = split_reports.get("validation", {})
    test_report = split_reports.get("test", {})
    exploration_rates = [
        float(report.get("ranking_metrics", {}).get("exploration_slots", 0.0))
        / float(report.get("ranking_metrics", {}).get("top_k", 1.0) or 1.0)
        for report in split_reports.values()
    ]

    return {
        "average_stage1_positive_rate": weighted_average(("ranking_metrics", "stage1_top_k_positive_rate")),
        "average_reranker_positive_rate": weighted_average(("ranking_metrics", "reranker_top_k_positive_rate")),
        "average_final_positive_rate": weighted_average(("ranking_metrics", "final_top_k_positive_rate")),
        "average_stage1_ndcg_at_k": weighted_average(("ranking_metrics", "stage1_ndcg_at_k")),
        "average_reranker_ndcg_at_k": weighted_average(("ranking_metrics", "reranker_ndcg_at_k")),
        "average_final_ndcg_at_k": weighted_average(("ranking_metrics", "final_ndcg_at_k")),
        "average_lift_vs_stage1": weighted_average(("ranking_metrics", "lift_vs_stage1")),
        "average_lift_vs_reranker": weighted_average(("ranking_metrics", "lift_vs_reranker")),
        "average_exploration_rate": _weighted_mean(exploration_rates, weights),
        "validation_lift_vs_stage1": float(validation_report.get("ranking_metrics", {}).get("lift_vs_stage1", 0.0)),
        "validation_lift_vs_reranker": float(validation_report.get("ranking_metrics", {}).get("lift_vs_reranker", 0.0)),
        "test_lift_vs_stage1": float(test_report.get("ranking_metrics", {}).get("lift_vs_stage1", 0.0)),
        "test_lift_vs_reranker": float(test_report.get("ranking_metrics", {}).get("lift_vs_reranker", 0.0)),
    }


def _rollout_readiness(split_reports: dict[str, dict[str, object]]) -> dict[str, object]:
    validation_report = split_reports.get("validation", {})
    test_report = split_reports.get("test", {})
    validation_reranker_lift = float(validation_report.get("ranking_metrics", {}).get("lift_vs_stage1", 0.0))
    test_reranker_lift = float(test_report.get("ranking_metrics", {}).get("lift_vs_stage1", 0.0))
    ready = validation_reranker_lift >= 0.0 and test_reranker_lift >= -0.02
    reasons: list[str] = []
    if validation_reranker_lift < 0.0:
        reasons.append("Validation reranker lift is below the stage-1 baseline.")
    if test_reranker_lift < -0.02:
        reasons.append("Test reranker lift is materially below the stage-1 baseline.")
    return {
        "ready": ready,
        "reasons": reasons,
        "validation_reranker_lift_vs_stage1": validation_reranker_lift,
        "test_reranker_lift_vs_stage1": test_reranker_lift,
    }


def _positive_rate(candidates: Sequence[HybridCandidateRecord]) -> float:
    if not candidates:
        return 0.0
    return sum(candidate.label for candidate in candidates) / len(candidates)


def _ndcg_at_k(candidates: Sequence[HybridCandidateRecord]) -> float:
    if not candidates:
        return 0.0
    gains = [float(candidate.label) for candidate in candidates]
    dcg = sum((2.0**gain - 1.0) / math.log2(index + 2.0) for index, gain in enumerate(gains))
    ideal_gains = sorted(gains, reverse=True)
    ideal_dcg = sum((2.0**gain - 1.0) / math.log2(index + 2.0) for index, gain in enumerate(ideal_gains))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def _overlap_rate(left_ids: set[str], right_ids: set[str], top_k: int) -> float:
    if top_k <= 0:
        return 0.0
    return len(left_ids.intersection(right_ids)) / top_k


def _mean_rank_shift(rank_map: dict[str, int], candidates: Sequence[HybridCandidateRecord]) -> float:
    shifts = [float(rank_map[candidate.candidate_id] - candidate.final_rank) for candidate in candidates if candidate.final_rank]
    return mean(shifts) if shifts else 0.0


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    if not values:
        return 0.0
    total_weight = sum(weights)
    if total_weight == 0.0:
        return 0.0
    return sum(value * weight for value, weight in zip(values, weights, strict=True)) / total_weight


def _compress_interaction(value: float) -> float:
    if value == 0.0:
        return 0.0
    return math.copysign(math.log1p(abs(value)), value)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")))
            handle.write("\n")
