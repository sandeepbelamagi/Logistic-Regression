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
