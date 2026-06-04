# Phase 2 Implementation Notes

## Purpose

Phase 2 turns the project from design into a runnable offline data pipeline.

This phase exists to:

- ingest the Bank Marketing dataset
- normalize raw rows into contracts
- engineer model-ready features
- generate training, validation, and test splits
- produce reproducible artifacts for Phase 3

## Inputs

Phase 2 consumes a local copy of the UCI Bank Marketing file:

- `bank-full.csv`

The pipeline supports the classic semicolon-delimited format and ignores header rows when present.

## Outputs

The offline builder creates three artifact groups:

- raw event logs
- training JSONL splits
- dataset metadata

### Raw Artifacts

- `raw/raw_contact_events.jsonl`
- `raw/raw_subscription_labels.jsonl`

### Training Artifacts

- `training/train.jsonl`
- `training/validation.jsonl`
- `training/test.jsonl`

### Metadata

- `metadata/dataset_summary.json`

## Phase 2 Flow

The offline builder in `src/probabilistic_decisioning/dataset_builder.py` performs the following steps:

1. parse each Bank Marketing row
2. build the raw contact event contract
3. build the subscription label contract
4. transform numeric features
5. hash categorical features into sparse IDs
6. build the training row contract
7. assign the row to train, validation, or test
8. write JSONL artifacts
9. emit a summary file with row counts and label rates

## Parsing Layer

The parser in `src/probabilistic_decisioning/bank_marketing.py` reads the standard Bank Marketing columns:

- 16 input fields
- 1 binary label field

It also records the `duration` field as leakage-prone data.
That field is preserved in the raw contract, but excluded from the default model features.

## Feature Engineering

Phase 2 creates two feature groups:

### Dense Features

Numeric inputs are transformed into a stable representation:

- `age`
- `balance`
- `day`
- `campaign`
- `pdays`
- `previous`
- `prior_contact_flag`

The feature layer uses log-style transforms to keep large values bounded and stable.

### Sparse Features

Categorical features are mapped into a hashed sparse space:

- `job`
- `marital`
- `education`
- `default`
- `housing`
- `loan`
- `contact`
- `month`
- `poutcome`

This gives the project a scalable sparse representation without having to build a dictionary vocabulary by hand.

## Why the Feature Design Matters

The feature design is deliberate:

- it keeps the pipeline reproducible
- it supports Logistic Regression well
- it mirrors internet-scale sparse modeling patterns
- it avoids leakage by default
- it prepares the project for the calibration and serving phases

## Splitting Strategy

Phase 2 supports two split modes:

- `hash`
- `contiguous`

### Hash Split

The hash split is deterministic and useful for a stable offline split when you do not want to depend on row order.

### Contiguous Split

The contiguous split is useful for time-like validation and deterministic smoke tests.

## CLI Usage

### Smoke Run

```bash
python pipelines/build_training_dataset.py --input samples/bank_full_smoke.csv --output-dir artifacts/bank_marketing_smoke --max-rows 3
```

### Full Run

```bash
python pipelines/build_training_dataset.py --input data/raw/bank_marketing/bank-full.csv --output-dir artifacts/bank_marketing_full
```

## Validation

Phase 2 is considered healthy when:

- the parser test passes
- the feature engineering test passes
- the dataset builder test passes
- the CLI can build artifacts from the smoke file
- the dataset summary reports the expected split counts

## What Phase 2 Enables

Phase 2 produces the stable offline inputs required by Phase 3:

- reproducible train/validation/test rows
- labels and weights
- model-ready dense and sparse vectors
- metadata for debugging and audit

Without Phase 2, Phase 3 would have no dependable training data contract.

