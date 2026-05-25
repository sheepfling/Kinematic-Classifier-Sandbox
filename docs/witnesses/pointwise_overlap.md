# Witness: pointwise_overlap

Purpose: Show the baseline likelihood and prior machinery on overlapping local evidence.

Class set: current common 1D classes from `artifacts/pointwise_baseline/dataset_manifest.json`.

Feature set: instantaneous pointwise observations.

Classifier/filter family: pointwise likelihood classifier.

Prior regime: explicit prior sweep in `artifacts/prior_sensitivity_pointwise_v1/`.

Corpus objective: simple overlapping class evidence with enough ambiguity to expose posterior flips.

What it proves: pointwise likelihoods, priors, posterior normalization, and flip thresholds are visible and auditable.

What it does not prove: history, dynamics, robust window features, or 3D separability.

Key equations: `log p(y_t | c)` and normalized posterior update.

Key plots:
- `artifacts/pointwise_baseline/pointwise_baseline_diagnostics.png`
- `artifacts/prior_sensitivity_pointwise_v1/posterior_vs_prior.png`

Key tables:
- `artifacts/pointwise_baseline/confusion_final.csv`
- `artifacts/prior_sensitivity_pointwise_v1/prior_flip_thresholds.csv`

Key artifacts:
- `artifacts/pointwise_baseline/posterior_history.csv`
- `artifacts/generic_inference_contract/posterior_history_schema.json`

Promotion status: promote as the baseline evidence witness.

Next extension toward 3D: replace scalar observations with vector-valued 3D observation features while preserving the posterior contract.
