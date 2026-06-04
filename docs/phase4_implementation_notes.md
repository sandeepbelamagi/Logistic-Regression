# Phase 4 Implementation Notes

## Purpose

Phase 4 turns the trained Logistic Regression model into a calibrated probability engine and a threshold-based decision system.

This phase exists to show that the model output is not just a score. It is a business signal that can drive routing, review, and prioritization.

## What Phase 4 Adds

Phase 4 introduces two new layers:

1. post-hoc calibration
2. decision policy routing

The implementation lives in:

- `src/probabilistic_decisioning/calibration.py`
- `src/probabilistic_decisioning/decision_policy.py`
- `pipelines/calibrate_and_route.py`

## Inputs

Phase 4 consumes:

- the Phase 3 `model.json`
- the Phase 2 JSONL splits

The calibration step uses the validation split as the fitting set for the calibrator.
The policy step then applies the selected calibrated probabilities to validation and test data.

## Outputs

The default CLI produces:

- `calibration/calibration.json`
- `policies/calibration_and_policy_report.json`
- `decision_outcomes/*.jsonl`

These artifacts let the project explain both calibration quality and policy behavior.

## Calibration Design

The calibration layer supports two candidate methods:

- **Platt scaling**
- **Isotonic regression**

The CLI fits both methods on the validation split and selects the winner using the configured selection metric.
The default selection metric is `validation_ece`.

### Platt Scaling

Platt scaling fits a one-dimensional sigmoid on top of the raw model score.

It is useful when the base model already ranks well but the probabilities need to be adjusted.

### Isotonic Regression

Isotonic regression fits a monotonic piecewise-constant mapping.

It is useful when calibration needs more flexibility than a sigmoid can provide.

## Why Calibration Matters

The model can rank well and still be poorly calibrated.
That matters because downstream business logic uses the predicted probability directly.

This is especially important for:

- campaign prioritization
- fraud review thresholds
- loan approval risk bands

## Decision Policy Design

The policy layer converts calibrated probabilities into actions.

The implementation supports three contexts:

- `bank_marketing`
- `fraud_policy`
- `loan_policy`

### Bank Marketing

The bank marketing policy maps probabilities into:

- `suppress`
- `nurture`
- `prioritize_contact`

This supports campaign ranking and contact prioritization.

### Fraud Policy

The fraud policy maps probabilities into:

- `approve`
- `review`
- `block`

This demonstrates analyst review queues and cost-sensitive thresholding.

### Loan Policy

The loan policy maps probabilities into:

- `approve`
- `review`
- `decline`

This demonstrates delayed-risk routing and governance.

## Policy Reporting

For each policy context, Phase 4 reports:

- action counts
- manual-review rate
- positive rate
- mean calibrated probability
- positive rate by action
- mean calibrated probability by action

This is the operational evidence that connects model scores to business behavior.

## CLI Usage

### Calibrate and Route

```bash
python pipelines/calibrate_and_route.py --data-dir artifacts/bank_marketing_lr --model-path artifacts/bank_marketing_lr/model.json --output-dir artifacts/bank_marketing_phase4
```

The command expects a Phase 3 model and Phase 2 splits.
It can route all three policy contexts in one run.

## Validation

Phase 4 is healthy when:

- the calibration artifact is written successfully
- the selected calibrator can be loaded back
- calibration metrics are produced for validation and test splits
- policy summaries are produced for each context
- decision outcome JSONL files are written

## Why Phase 4 Matters

Phase 4 is where the project moves from “good model” to “actionable system.”

It gives the project the ability to explain:

- why calibration is different from ranking
- how thresholds change behavior
- why one probability can drive multiple policy contexts

