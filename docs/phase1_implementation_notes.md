# Phase 1 Implementation Notes

## Purpose

Phase 1 defines the architecture, scope, contracts, and success criteria for the project.

This phase exists to make the rest of the work coherent. It answers:

- what the platform is
- which use cases it supports
- how the system is partitioned
- what “good” means for the model, policy, and serving layers

## Deliverables

Phase 1 creates the planning and contract layer of the repository:

- `docs/architecture.md`
- `docs/problem_statement.md`
- `docs/kpi_framework.md`
- `docs/interview_coverage_matrix.md`
- `data_contracts/`
- `configs/`
- repository scaffold directories for future pipeline, model, service, and experiment code

## What Phase 1 Covers

### Architecture

The architecture document defines the platform as a probabilistic decisioning system with:

- a data plane
- a feature platform
- a training plane
- a calibration plane
- a decision plane
- a serving plane
- a monitoring and governance plane

The architecture is intentionally broader than the Bank Marketing dataset.
The dataset is the implementation backbone, but the system must also explain fraud and loan-style decisioning.

### Problem Framing

The problem statement ties the project to the interview notes.
It establishes that Logistic Regression is being used as a production probability engine, not just a classroom classifier.

It also explains why the project uses Bank Marketing as the primary dataset:

- it is small enough to run locally
- it still supports sparse tabular modeling
- it naturally fits probability ranking and thresholding
- it can be extended into fraud and loan policy simulations

### KPI Framework

The KPI framework separates the system into four measurement layers:

1. model discrimination
2. model calibration
3. policy and business outcomes
4. system reliability and governance

That separation is important because the project must explain why good ranking does not automatically mean good calibration or good business policy.

### Interview Coverage Matrix

The coverage matrix maps each concept from the interview notes to a concrete phase or artifact.

It is the traceability layer for the whole project:

- cross-entropy vs MSE
- fraud thresholding
- class weighting vs oversampling
- calibration drift
- delayed labels
- multicollinearity
- internet-scale sparse Logistic Regression
- calibration vs discrimination
- hybrid reranking

## Key Contracts

Phase 1 defines the core data and policy schemas used later by the pipeline:

- `raw_contact_event`
- `raw_subscription_label`
- `training_row`
- `prediction_log`
- `decision_outcome`

These contracts establish the fields, ownership, and quality rules for later phases.

## Configuration Layer

Phase 1 also defines the initial configuration files under `configs/`:

- training defaults
- model defaults
- calibration defaults
- threshold policy defaults
- monitoring defaults
- environment defaults
- serving defaults

These are the control knobs for later implementation.
They let the project evolve without hardcoding phase behavior into the code.

## Repository Scaffold

Phase 1 creates the directory skeleton for future implementation work:

- `pipelines/`
- `feature_store/`
- `models/`
- `services/`
- `experiments/`
- `tests/`
- `samples/`

The scaffold matters because it makes the project look and behave like a real platform, not a single notebook.

## What Phase 1 Does Not Do

Phase 1 does not:

- train any model
- build any feature pipeline
- expose any runtime API
- generate any production artifacts
- calibrate probabilities
- implement threshold routing

## Why Phase 1 Matters

Phase 1 is the governance and design contract for the whole repository.

Without it, the later code would be a collection of unrelated scripts.
With it, the later phases can be explained as pieces of one system.

