# Architecture

## Objective

Build a production-grade probabilistic decisioning platform where calibrated Logistic Regression probabilities are the core contract between modeling and business policy.

The platform must cover:

- Logistic Regression fundamentals and cross-entropy optimization
- severe class imbalance and thresholding
- calibration drift and recalibration
- delayed labels and feedback loops
- large-scale sparse feature training and low-latency serving
- hybrid decisioning with linear retrieval and deep reranking

## Architecture Principles

1. Event log is the source of truth.
2. Feature definitions are versioned and point-in-time correct.
3. Scoring and decisioning are separate services.
4. Calibration is a first-class layer, not an afterthought.
5. Model quality and business policy are monitored independently.
6. Delayed labels and feedback loops are handled explicitly.
7. The baseline Logistic Regression system must be production worthy before deep models are added.

## Non-Functional Requirements

- `p95` online scoring latency below `10ms` for the linear scorer
- `99.9%` availability for prediction and decision APIs
- reproducible training with versioned datasets, features, and configs
- full auditability for scoring, thresholding, and overrides
- safe rollout via champion-challenger and canary deployment

## System Context

The same probability platform supports three decision contexts:

- **Bank Marketing propensity ranking**: maximize expected subscription value with strict calibration
- **fraud-style routing**: approve, review, or block based on expected cost
- **loan-style risk policy**: optimize long-term value under delayed outcomes and fairness constraints

## High-Level Components

### 1. Data Plane

- event ingestion for contact records, subscription labels, contextual metadata, and downstream outcomes
- immutable bronze layer for raw events
- cleaned silver layer for validated events
- gold training tables with point-in-time features and label windows

### 2. Feature Platform

- offline feature store for backfills and training
- online feature store for low-latency lookups
- shared feature definitions to prevent training-serving skew
- freshness and completeness checks for online features

### 3. Training Plane

- sparse Logistic Regression baseline with sigmoid and cross-entropy
- feature hashing for large categorical spaces
- regularization with `L1` and `L2`
- optimizers such as `Adagrad` and `FTRL-Proximal`
- experiment framework for class weighting, oversampling, and calibration studies

### 4. Calibration Plane

- post-hoc calibration using Platt scaling and isotonic regression
- calibration refresh jobs on rolling windows
- Expected Calibration Error, reliability curves, and bucketed outcome monitoring

### 5. Decision Plane

- business threshold service that consumes calibrated probability
- routing policies for approve, review, block
- context-specific cost matrices for bank marketing, fraud, and loan workflows
- human-in-the-loop escalation support

### 6. Serving Plane

- feature lookup API
- prediction API for raw and calibrated probabilities
- decision API for action selection
- prediction logging for offline analysis and replay

### 7. Monitoring and Governance Plane

- drift monitoring, calibration monitoring, and system SLOs
- fairness and subgroup performance analysis
- model registry, audit logs, rollout policies, and rollback controls

## Logical Flow

```text
Client/Event Source
  -> Ingestion
  -> Raw Event Store
  -> Validation + Standardization
  -> Offline Feature Store + Online Feature Store
  -> Training Dataset Builder
  -> Logistic Regression Trainer
  -> Calibration Trainer
  -> Model Registry
  -> Prediction API
  -> Decision API
  -> Prediction Logs + Outcome Logs
  -> Monitoring + Retraining Triggers
```

## Training Architecture

### Dataset Strategy

- primary training corpus: UCI Bank Marketing `bank-full.csv`
- deterministic train, validation, and holdout splits
- time-aware validation windows to simulate production drift
- delayed-label simulation for loan-style and conversion-style use cases

### Training Stages

1. ingest and validate raw data
2. transform categorical and dense features
3. build hashed sparse representation
4. train baseline Logistic Regression
5. run ablations:
   - cross-entropy vs MSE study
   - class weighting vs oversampling
   - regularization and multicollinearity stability
6. fit calibration layer
7. publish model, calibration artifact, and evaluation report

## Serving Architecture

### Online Request Path

1. request arrives with context and entity identifiers
2. feature API retrieves online features and validates freshness
3. prediction API computes raw score and calibrated probability
4. decision API applies policy thresholds and optional manual-review routing
5. full request and response metadata are logged for replay and audit

### Latency Design

- prediction complexity is `O(non-zero features)`
- only active sparse features participate in scoring
- model weights are memory-mapped or cached in-process
- calibration layer remains lightweight and deterministic

## Monitoring Architecture

### Model Quality

- `ROC-AUC`
- `PR-AUC`
- log loss
- `ECE`
- reliability curves

### Data and Feature Health

- feature null rate
- feature freshness lag
- Population Stability Index
- training-serving skew rate

### Business and Policy Health

- revenue proxy uplift
- fraud capture rate
- false positive cost
- manual-review queue size
- risk-adjusted profit

### Governance

- subgroup fairness metrics
- approval-rate parity
- explanation completeness
- audit log coverage

## Storage and Interfaces

### Storage

- raw event store for immutable ingestion records
- warehouse tables for cleaned and joined features
- model registry for weights, configs, calibration artifacts, and metrics
- online store for low-latency feature retrieval

### Interfaces

- ingestion contract for raw events
- feature contract for offline and online feature parity
- prediction contract for model outputs
- decision contract for policy actions and outcomes

## Deployment Strategy

- local development for data-contract and pipeline iteration
- staging with replay traffic and synthetic drift tests
- production with canary release and champion-challenger evaluation

## Phase 1 Boundaries

Phase 1 creates:

- architecture and planning documents
- KPI and governance framework
- schema contracts
- configuration stubs
- repository scaffold

Phase 1 does not create:

- executable training code
- serving code
- pipeline implementations
- dashboards or deployment manifests

## Repository Skeleton

```text
docs/
data_contracts/
configs/
pipelines/
feature_store/
models/
services/
  feature-api/
  prediction-api/
  decision-api/
  monitoring/
experiments/
tests/
```
