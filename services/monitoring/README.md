# Monitoring Service

Phase 6 implements batch monitoring and rollout controls.

## Responsibilities

- aggregate model, data, system, business, and governance metrics
- compare live behavior against Phase 3 and Phase 4 baselines
- compute feature drift, calibration drift, and subgroup gaps
- write alert summaries and rollout-readiness signals

## Inputs

- Phase 2 training splits
- Phase 3 `metrics.json`
- Phase 4 `calibration_and_policy_report.json`
- Phase 5 `prediction_log.jsonl`
- Phase 5 `decision_log.jsonl`

## Outputs

- `monitoring_report.json`
- `alerts.json`

## Implementation

- engine: `src/probabilistic_decisioning/monitoring.py`
- CLI: `pipelines/run_monitoring.py`
- convenience runner: `scripts/phase6_runner.py`

## Run Command

```bash
python scripts/phase6_runner.py --mode sample
```

Use `--mode full` for the full-artifact path set.

