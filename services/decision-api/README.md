# Decision API

Implemented responsibility:

- apply policy thresholds to calibrated probabilities
- route records to approve, review, block, or rank
- emit decision logs for downstream evaluation

Phase 4 adds the offline calibration and policy-routing implementation that feeds this service.

The local runtime is available through:

```bash
python pipelines/serve_prediction_api.py --model-path artifacts/bank_marketing_lr_cv/model.json --calibration-path artifacts/bank_marketing_phase4_cv/calibration/calibration.json --log-dir artifacts/bank_marketing_phase5_logs
```
