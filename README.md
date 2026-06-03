# Probabilistic Decisioning Platform

Phase 1 scaffolds a FAANG-level project around Logistic Regression as a calibrated probability engine for:

- CTR prediction and auction-safe ranking
- fraud-style thresholding and action routing
- loan-style delayed feedback, fairness, and governance
- hybrid ranking with linear retrieval and deep reranking

## Phase Status

- Phase 1: architecture, contracts, KPI framework, and repo scaffold
- Phase 2: offline data pipeline and feature engineering
- Phase 3: Logistic Regression training and evaluation
- Phase 4: calibration and decision policy engine
- Phase 5: online serving and low-latency inference
- Phase 6: monitoring, governance, and rollout controls
- Phase 7: hybrid reranking and exploration

## Primary Dataset

- Primary: Criteo CTR `train.txt`
- Secondary simulators: fraud-cost policy and delayed-default policy built on top of calibrated probabilities

## Key Artifacts

- `docs/architecture.md`
- `docs/problem_statement.md`
- `docs/kpi_framework.md`
- `docs/interview_coverage_matrix.md`
- `data_contracts/`
- `configs/`

## Phase 2 Implementation

Phase 2 adds offline data ingestion and feature engineering for the Criteo CTR dataset:

- streaming parser for `train.txt`
- raw impression and click-label artifact generation
- dense feature transformation using `log1p`
- sparse categorical feature hashing
- train, validation, and test dataset materialization
- metadata summary generation

### Build Command

```bash
python pipelines/build_training_dataset.py --input path/to/train.txt --output-dir artifacts/criteo
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
