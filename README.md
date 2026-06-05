# Probabilistic Decisioning Platform

Phase 1 scaffolds a FAANG-level project around Logistic Regression as a calibrated probability engine for:

- Bank Marketing term-deposit propensity prediction
- fraud-style thresholding and action routing
- loan-style delayed feedback, fairness, and governance
- hybrid decisioning with linear retrieval and deep reranking

## Phase Status

- Phase 1: architecture, contracts, KPI framework, and repo scaffold
- Phase 2: offline data pipeline and feature engineering
- Phase 3: Logistic Regression training and evaluation
- Phase 4: calibration and decision policy engine
- Phase 5: online serving and low-latency inference
- Phase 6: monitoring, governance, and rollout controls
- Phase 7: hybrid reranking and exploration

## Primary Dataset

- Primary: UCI Bank Marketing `bank-full.csv`
- Secondary simulators: fraud-cost policy and delayed-default policy built on top of calibrated probabilities

## Key Artifacts

- `docs/architecture.md`
- `docs/problem_statement.md`
- `docs/kpi_framework.md`
- `docs/interview_coverage_matrix.md`
- `docs/phase1_implementation_notes.md`
- `docs/phase2_implementation_notes.md`
- `docs/phase3_implementation_notes.md`
- `docs/phase4_implementation_notes.md`
- `docs/phase5_implementation_notes.md`
- `docs/phase6_implementation_notes.md`
- `data_contracts/`
- `configs/`

## Phase 2 Implementation

Phase 2 adds offline data ingestion and feature engineering for the Bank Marketing dataset:

- streaming parser for semicolon-delimited CSV
- raw contact-event and subscription-label artifact generation
- numeric feature transformation using signed log and `log1p`
- sparse categorical feature hashing
- train, validation, and test dataset materialization
- metadata summary generation

### Build Command

```bash
python pipelines/build_training_dataset.py --input samples/bank_full_smoke.csv --output-dir artifacts/bank_marketing_smoke
```

### Train Command

```bash
python pipelines/train_logistic_regression.py --data-dir artifacts/bank_marketing_smoke --output-dir artifacts/bank_marketing_lr
```

## Phase 4 Implementation

Phase 4 adds calibration and decision routing on top of the trained model:

- Platt scaling and isotonic regression calibration
- validation-driven calibrator selection
- bank marketing, fraud, and loan policy routing
- policy summaries and decision outcome exports

### Calibrate and Route

```bash
python pipelines/calibrate_and_route.py --data-dir artifacts/bank_marketing_smoke --model-path artifacts/bank_marketing_lr/model.json --output-dir artifacts/bank_marketing_phase4
```

If you use the 3-row smoke file, the default hash split leaves validation empty. For a smoke-safe Phase 4 run, rebuild Phase 2 with contiguous splitting:

```bash
python pipelines/build_training_dataset.py --input samples/bank_full_smoke.csv --output-dir artifacts/bank_marketing_smoke_cv --split-strategy contiguous --train-ratio 0.34 --validation-ratio 0.33 --test-ratio 0.33
python pipelines/train_logistic_regression.py --data-dir artifacts/bank_marketing_smoke_cv --output-dir artifacts/bank_marketing_lr_cv
python pipelines/calibrate_and_route.py --data-dir artifacts/bank_marketing_smoke_cv --model-path artifacts/bank_marketing_lr_cv/model.json --output-dir artifacts/bank_marketing_phase4_cv
```

## Phase 5 Implementation

Phase 5 adds a local serving runtime for feature lookup, prediction, and decision routing:

- `/health` readiness check
- `/v1/features/lookup` engineered feature preview
- `/v1/predict` raw score, calibrated score, and routed decision
- `/v1/decision` decision-only routing from calibrated probabilities
- prediction and decision JSONL logs for replay and audit

### Serve Locally

```bash
python pipelines/serve_prediction_api.py --model-path artifacts/bank_marketing_lr_cv/model.json --calibration-path artifacts/bank_marketing_phase4_cv/calibration/calibration.json --log-dir artifacts/bank_marketing_phase5_logs
```

### Example Call

```bash
curl -X POST http://127.0.0.1:8000/v1/features/lookup -H "Content-Type: application/json" -d "{\"request_id\":\"req_1\",\"event_id\":\"evt_1\",\"event_ts\":\"2026-01-01T00:00:00Z\",\"task_context\":\"bank_marketing\",\"features\":{\"age\":\"42\",\"job\":\"admin.\",\"marital\":\"married\",\"education\":\"secondary\",\"default\":\"no\",\"balance\":\"1200\",\"housing\":\"yes\",\"loan\":\"no\",\"contact\":\"cellular\",\"day\":\"15\",\"month\":\"oct\",\"duration\":\"80\",\"campaign\":\"2\",\"pdays\":\"-1\",\"previous\":\"0\",\"poutcome\":\"unknown\"}}"
```

### Validation

- `python -B -m unittest tests.test_serving -v`
- the HTTP server loads the Phase 3 and Phase 4 artifacts
- prediction logs match `data_contracts/prediction_log.yaml`

## Phase 6 Implementation

Phase 6 adds monitoring, governance, and rollout controls on top of the Phase 2 to Phase 5 artifacts:

- model, data, system, and policy aggregation
- calibration drift and feature drift analysis
- subgroup calibration and approval-gap checks
- audit-log coverage and rollout readiness

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

### Validation

- `python -B -m unittest tests.test_monitoring -v`
- the monitoring report contains system, model, drift, governance, and alert summaries
- `scripts/phase6_runner.py` works for both sample and full artifact sets

## Planned Repository Layout

- `docs/` system design and requirements
- `data_contracts/` event and model I/O schemas
- `configs/` environment and pipeline configuration
- `pipelines/` ingestion, feature, train, and scoring jobs
- `feature_store/` offline and online feature definitions
- `models/` scoring and calibration modules
- `services/` runtime APIs and monitoring services
- `experiments/` ablation studies and analysis notebooks
- `tests/` unit, integration, and ML validation suites
- `samples/bank_full_smoke.csv` runnable smoke fixture
