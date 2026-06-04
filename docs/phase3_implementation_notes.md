# Phase 3 Implementation Notes

## Purpose

Phase 3 turns the Phase 2 Bank Marketing artifacts into a trained Logistic Regression baseline and a repeatable evaluation workflow.

This phase exists to prove four things:

1. the model can learn from the engineered features
2. the model can output usable probabilities, not just class labels
3. the pipeline can compare training choices like cross-entropy vs MSE
4. the trained artifact can be saved and replayed deterministically

## Inputs

Phase 3 consumes the JSONL artifacts written by Phase 2:

- `training/train.jsonl`
- `training/validation.jsonl`
- `training/test.jsonl`
- `metadata/dataset_summary.json`

Each training row already contains:

- `label`
- `dense_features`
- `sparse_feature_ids`
- `sparse_feature_values`
- `feature_set_version`
- `task_context`

The trainer refuses to run if the feature version or task context does not match the expected Bank Marketing configuration.

## Outputs

The default training command writes three artifacts:

- `model.json`
- `metrics.json`
- `history.json`

The optional experiment-suite mode writes:

- `experiment_report.json`

## Training Flow

The main training path in `src/probabilistic_decisioning/logistic_regression.py` does the following:

1. load train, validation, and test examples from JSONL
2. validate schema consistency across all splits
3. compute class weights from the training split
4. optionally oversample the minority class for the ablation runs
5. initialize a sparse Logistic Regression model
6. train with stochastic updates over multiple epochs
7. evaluate after each epoch on train and validation splits
8. stop early when validation log loss stops improving
9. restore the best validation checkpoint
10. evaluate the final model on train, validation, and test splits
11. save the trained model and evaluation artifacts

## Model Design

The model is a sparse linear classifier with:

- one bias term
- dense weights for Bank Marketing numeric features
- hashed sparse weights for categorical features

The training code keeps the model simple on purpose.
That makes the model:

- fast to train
- easy to debug
- easy to explain
- suitable for later calibration in Phase 4

## Optimization Choices

The trainer supports two losses:

- **cross-entropy**: the default and the correct probabilistic loss
- **MSE**: included as a comparison baseline from the interview notes

The trainer supports two optimizers:

- **SGD**
- **Adagrad**

The default configuration uses:

- `cross_entropy`
- `Adagrad`
- light `L2` regularization
- class weighting enabled

## Why Cross-Entropy Is the Default

Cross-entropy is the right default because it directly optimizes probabilistic prediction.

It is preferred over MSE because it:

- penalizes confident wrong predictions more strongly
- produces cleaner gradients for classification
- converges better for logistic models
- matches the Bernoulli likelihood assumption

The Phase 3 suite still keeps MSE as an experiment so the project can explain the tradeoff clearly.

## Class Weighting and Oversampling

The training code includes both imbalance strategies because the notes require a direct comparison.

### Class Weighting

Class weighting changes the loss contribution of each class without changing the dataset.

This is the default because it:

- scales well
- keeps the original data distribution intact
- usually preserves probability calibration better than oversampling

### Oversampling

Oversampling duplicates minority examples to make the optimizer see them more often.

It is included as an ablation because it is useful to demonstrate:

- optimization behavior
- overfitting risk
- calibration side effects

## Regularization

The model supports `L1` and `L2` regularization.

- `L1` encourages sparsity
- `L2` stabilizes coefficients

Regularization matters here because the project also needs to explain multicollinearity and coefficient stability.

## Metrics

The trainer reports the following metrics on train, validation, and test splits:

- `log_loss`
- `ROC-AUC`
- `PR-AUC`
- `Brier score`
- `ECE`
- `MCE`
- reliability bins
- positive rate
- mean predicted probability

### Why These Metrics

- **log loss** checks probability quality directly
- **ROC-AUC** checks ranking quality
- **PR-AUC** is useful when positives are rare
- **Brier score** measures probabilistic error
- **ECE/MCE** measure calibration quality

The project keeps discrimination and calibration separate because they answer different questions.

## Experiment Suite

The Phase 3 experiment suite runs one baseline plus four comparison variants on the same dataset split:

- baseline cross-entropy model
- MSE loss variant
- oversampling variant
- no-class-weighting variant
- no-regularization variant

It can also run repeated bootstrap training to estimate coefficient stability.

That gives the project concrete evidence for:

- cross-entropy vs MSE
- class weighting vs oversampling
- regularization impact on coefficient stability

## Artifact Layout

### Trained Model

`model.json` stores:

- model version
- feature-set version
- task context
- bias
- dense weights
- hashed sparse weights
- training hyperparameters

### Metrics

`metrics.json` stores:

- class weights
- dataset summary
- train metrics
- validation metrics
- test metrics

### History

`history.json` stores the per-epoch log, including:

- train loss
- validation log loss
- validation ROC-AUC
- validation PR-AUC
- validation ECE

## CLI Usage

### Train the Baseline

```bash
python pipelines/train_logistic_regression.py --data-dir artifacts/bank_marketing_smoke --output-dir artifacts/bank_marketing_lr
```

### Run the Experiment Suite

```bash
python pipelines/train_logistic_regression.py --data-dir artifacts/bank_marketing_smoke --output-dir artifacts/bank_marketing_suite --run-suite --stability-repeats 2
```

## Validation

Phase 3 is considered healthy when:

- the unit tests pass
- the CLI can train on the Phase 2 smoke dataset
- the output model artifact can be loaded back
- the metrics file contains train, validation, and test summaries

## Why This Phase Matters

Phase 3 is the point where the project stops being just a data pipeline and becomes a real machine learning system.

It creates the core baseline that later phases will extend with:

- calibration
- threshold policies
- online serving
- monitoring
- governance
