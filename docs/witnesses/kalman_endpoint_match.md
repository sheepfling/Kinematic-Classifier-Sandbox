# Witness: kalman_endpoint_match

Purpose: Show that model-based dynamics evidence can help when endpoint or shape evidence is ambiguous.

Class set: Kalman model classes in `artifacts/kalman_filter_bank/kalman_model_definitions.json`.

Feature set: innovation and residual histories.

Classifier/filter family: Kalman filter bank.

Prior regime: uniform or configured class priors from `artifacts/kalman_filter_bank/kalman_bank_config.yaml`.

Corpus objective: matched-endpoint trajectories where simple position summaries can be ambiguous but dynamics residuals differ.

What it proves: innovation likelihoods can act as classifier evidence under the same posterior history contract.

What it does not prove: Kalman banks are generally superior or sufficient for nonlinear 3D motion.

Key equations: class-conditioned innovation likelihood and posterior normalization.

Key plots:
- `artifacts/kalman_filter_bank/kalman_bank_diagnostics.png`
- `artifacts/kalman_variant_comparison_v1/kalman_variant_heatmap.png`

Key tables:
- `artifacts/kalman_filter_bank/confusion_final.csv`
- `artifacts/kalman_variant_comparison_v1/kalman_variant_summary.csv`

Key artifacts:
- `artifacts/kalman_filter_bank/innovation_history.csv`
- `artifacts/kalman_filter_bank/posterior_history.csv`

Promotion status: promote for matched-endpoint dynamics evidence.

Next extension toward 3D: add 3D state-space models and compare innovation likelihoods across maneuver families.
