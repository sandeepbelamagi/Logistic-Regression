# Feature API

Implemented responsibility:

- fetch online features
- enforce freshness checks
- return feature-set version metadata

In the current phase, feature lookup is served by:

```bash
python pipelines/serve_prediction_api.py --model-path artifacts/bank_marketing_lr_cv/model.json --calibration-path artifacts/bank_marketing_phase4_cv/calibration/calibration.json --log-dir artifacts/bank_marketing_phase5_logs
```
