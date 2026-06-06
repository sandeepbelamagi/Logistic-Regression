# Phase 7 Implementation Notes

## Purpose

Phase 7 adds the hybrid ranking extension on top of the calibrated decision platform.
It uses the Phase 3 linear scorer as the retrieval layer, trains a second-stage reranker on richer interaction features, and then applies an explicit exploration policy to the final shortlist.

## What Phase 7 Adds

Phase 7 introduces three layers:

1. stage-1 linear retrieval with the calibrated Logistic Regression baseline
2. stage-2 reranking with interaction features derived from the stage-1 output and Bank Marketing context
3. stage-3 exploration that reserves a small fraction of shortlist slots for uncertain candidates

The implementation lives in:

- `src/probabilistic_decisioning/hybrid_ranking.py`
- `pipelines/run_hybrid_ranking.py`
- `scripts/phase7_runner.py`

## Inputs

Phase 7 consumes:

- the Phase 2 training splits
- the Phase 3 `model.json`
- the Phase 4 `calibration.json`

The reranker is trained on derived features built from the calibrated base model and the original Bank Marketing feature vectors.

## Feature Design

The reranker feature vector includes:

- raw stage-1 logit
- stage-1 probability
- calibrated probability
- uncertainty score
- dense Bank Marketing features
- interaction terms such as `age × balance`, `balance × campaign`, and `campaign × previous`

This keeps the extension interpretable while still allowing the second stage to learn non-linear preference patterns.

## Exploration Policy

Phase 7 uses a deterministic uncertainty-top-up policy:

- reserve `top_k × exploration_rate` slots
- fill the remaining slots with the highest reranker scores
- fill the exploration slots with the most uncertain candidates that were not already selected

This gives the system a controlled exploration path instead of blindly taking the top reranker scores.

## Outputs

The Phase 7 job writes:

- `reranker/model.json`
- `reranker/metrics.json`
- `reranker/history.json`
- `rankings/train_top_k.jsonl`
- `rankings/validation_top_k.jsonl`
- `rankings/test_top_k.jsonl`
- `hybrid_ranking_report.json`

## Validation

Phase 7 is healthy when:

- the reranker artifacts are written successfully
- ranking previews are produced for train, validation, and test
- validation reranker lift is not worse than the stage-1 baseline
- rollout readiness is `true`

## Run Commands

### Sample Run

```bash
python scripts/phase7_runner.py --mode sample
```

### Full Run

```bash
python scripts/phase7_runner.py --mode full
```

### Explicit CLI

```bash
python pipelines/run_hybrid_ranking.py --data-dir artifacts/bank_marketing_smoke_cv --model-path artifacts/bank_marketing_lr_cv/model.json --calibration-path artifacts/bank_marketing_phase4_cv/calibration/calibration.json --output-dir artifacts/bank_marketing_phase7_sample
```

