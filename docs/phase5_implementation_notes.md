# Phase 5 Implementation Notes

## Objective

Phase 5 turns the trained and calibrated Bank Marketing model into a local online serving runtime.
It exposes the feature, prediction, and decision layers as HTTP endpoints and writes replayable logs.

## Runtime Boundary

The runtime is built from three Phase 3 and Phase 4 artifacts:

- `model.json` from Phase 3
- `calibration.json` from Phase 4
- feature-engineering logic shared with Phase 2

The serving bundle validates that the model and calibration artifacts agree on:

- `model_version`
- `feature_set_version`
- `task_context`

## API Surface

### `GET /health`

Returns readiness metadata:

- model version
- feature-set version
- task context
- calibration version
- whether prediction logging is enabled

### `POST /v1/features/lookup`

Accepts a Bank Marketing request payload and returns engineered features:

- dense feature vector
- sparse hashed feature ids and values
- feature-set metadata

### `POST /v1/predict`

Accepts a Bank Marketing request payload and returns:

- raw score
- raw probability
- calibrated probability
- routed decision
- feature lookup timing
- scoring timing
- calibration timing
- decision timing

### `POST /v1/decision`

Accepts either:

- a calibrated probability and decision context, or
- a full raw-feature request

If raw features are supplied, the runtime can score and route in one call.

## Logging

Phase 5 writes two JSONL logs when log paths are supplied:

- `prediction_log.jsonl` aligned with `data_contracts/prediction_log.yaml`
- `decision_log.jsonl` aligned with the decision-outcome contract used in Phase 4

These logs support replay, offline analysis, and later monitoring work.

## File Layout

- `src/probabilistic_decisioning/serving.py` implements the runtime and HTTP server
- `pipelines/serve_prediction_api.py` starts the local server

## Run Command

```bash
python pipelines/serve_prediction_api.py --model-path artifacts/bank_marketing_lr_cv/model.json --calibration-path artifacts/bank_marketing_phase4_cv/calibration/calibration.json --log-dir artifacts/bank_marketing_phase5_logs
```

## Example Requests

### Feature Lookup

```bash
curl -X POST http://127.0.0.1:8000/v1/features/lookup -H "Content-Type: application/json" -d "{\"request_id\":\"req_1\",\"event_id\":\"evt_1\",\"event_ts\":\"2026-01-01T00:00:00Z\",\"task_context\":\"bank_marketing\",\"features\":{\"age\":\"42\",\"job\":\"admin.\",\"marital\":\"married\",\"education\":\"secondary\",\"default\":\"no\",\"balance\":\"1200\",\"housing\":\"yes\",\"loan\":\"no\",\"contact\":\"cellular\",\"day\":\"15\",\"month\":\"oct\",\"duration\":\"80\",\"campaign\":\"2\",\"pdays\":\"-1\",\"previous\":\"0\",\"poutcome\":\"unknown\"}}"
```

### Decision-Only Routing

```bash
curl -X POST http://127.0.0.1:8000/v1/decision -H "Content-Type: application/json" -d "{\"request_id\":\"req_2\",\"event_id\":\"evt_2\",\"event_ts\":\"2026-01-01T00:00:00Z\",\"task_context\":\"fraud_policy\",\"calibrated_probability\":0.8}"
```

## Validation

- `python -B -m unittest tests.test_serving -v`
- the full test suite passes with `python -B -m unittest discover -s tests -v`
- prediction logs contain the expected contract fields
- decision routing works with and without raw features
