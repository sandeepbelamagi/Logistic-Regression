# Phase 6 Implementation Notes

## Purpose

Phase 6 turns the project into a monitored system.
It aggregates model quality, data drift, system reliability, business behavior, and governance checks into one batch report.

This phase is the bridge between an offline ML project and a production-minded control plane.

## Inputs

Phase 6 consumes artifacts from the earlier phases:

- Phase 2 training splits
- Phase 3 `metrics.json`
- Phase 4 `calibration_and_policy_report.json`
- Phase 5 `prediction_log.jsonl`
- Phase 5 `decision_log.jsonl`

The prediction log now includes the engineered feature snapshot used at scoring time so drift can be compared against the training distribution.

## What Phase 6 Measures

### System Health

- prediction latency p50, p95, and p99
- logging completeness
- prediction-to-decision linkage
- availability proxy

### Model Health

- live calibrated log loss
- live ECE
- live raw-vs-calibrated comparison
- calibration drift against the Phase 4 baseline

### Data Drift

- dense-feature PSI against the Phase 2 training distribution
- sparse hashed-bucket PSI
- feature zero-rate changes

### Governance

- subgroup calibration gap
- approval-rate gap
- audit-log coverage
- group-level action summaries

## Alert Thresholds

Thresholds are configured in `configs/monitoring.yaml`.

The default critical thresholds are:

- `ece_critical: 0.03`
- `psi_critical: 0.2`
- `prediction_p95_latency_ms_critical: 10`
- `manual_review_utilization_critical: 0.85`
- `subgroup_calibration_gap_critical: 0.05`
- `approval_rate_gap_critical: 0.05`

## Outputs

The monitoring job writes:

- `monitoring_report.json`
- `alerts.json`

The report contains:

- artifact provenance
- model identity
- system summary
- model metrics
- calibration snapshot
- feature drift summary
- governance summary
- rollout readiness

## File Layout

- `src/probabilistic_decisioning/monitoring.py` implements the monitoring engine
- `pipelines/run_monitoring.py` exposes the CLI
- `scripts/phase6_runner.py` provides sample/full convenience runs

## Run Commands

### Sample Run

```bash
python scripts/phase6_runner.py --mode sample
```

### Full Run

```bash
python scripts/phase6_runner.py --mode full
```

### Explicit CLI

```bash
python pipelines/run_monitoring.py --model-path artifacts/bank_marketing_lr_cv/model.json --training-data-dir artifacts/bank_marketing_smoke_cv --prediction-log artifacts/bank_marketing_phase5_logs_sample/prediction_log.jsonl --decision-log artifacts/bank_marketing_phase5_logs_sample/decision_log.jsonl --phase3-metrics artifacts/bank_marketing_lr_cv/metrics.json --phase4-report artifacts/bank_marketing_phase4_cv/policies/calibration_and_policy_report.json --output-dir artifacts/bank_marketing_phase6_sample
```

## Validation

Phase 6 is healthy when:

- all artifact paths load successfully
- the report contains system, model, drift, and governance sections
- prediction-link coverage is close to 1.0 for scored requests
- rollout readiness is true when no critical alerts are present

