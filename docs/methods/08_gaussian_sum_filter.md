# Gaussian Sum Filter

Gaussian Sum Filters approximate a posterior with a small Gaussian mixture
before escalating to particle methods.

## Role In The Classifier

Use GSF when a single Gaussian collapses multimodal posterior structure but the
number of relevant modes is still small enough to track with a manageable
mixture.

The current oracle-backed witness is:

- `gsf_abs_range_multimodal_oracle_v1`

## Contract Hook

The GSF witness emits:

- `artifacts/gsf_abs_range_multimodal_oracle_v1/truth_trajectory.csv`
- `artifacts/gsf_abs_range_multimodal_oracle_v1/measurements.csv`
- `artifacts/gsf_abs_range_multimodal_oracle_v1/grid_oracle_posterior_history.csv`
- `artifacts/gsf_abs_range_multimodal_oracle_v1/method_posterior_history.csv`
- `artifacts/gsf_abs_range_multimodal_oracle_v1/gaussian_baseline_posterior_history.csv`
- `artifacts/gsf_abs_range_multimodal_oracle_v1/component_history.csv`
- `artifacts/gsf_abs_range_multimodal_oracle_v1/state_estimate_history.csv`
- `artifacts/gsf_abs_range_multimodal_oracle_v1/metrics_against_oracle.csv`
- `artifacts/gsf_abs_range_multimodal_oracle_v1/decision_card.md`

## Current Read

The current GSF witness is strong enough to mark the method
`witness_supported` on the abs-range multimodal family. The repo now also
emits a direct GSF-vs-PF frontier under
`artifacts/advanced_filter_comparison_v1/gsf_vs_pf_frontier_summary.csv`.
The current crossover is `metric_split`: GSF is dramatically cheaper and has
lower oracle KL, while PF still holds the better sign-mass error.

GSF is still not `study_justified` because the repo lacks:

- comparison against future UKF and robust-Kalman blockers.

## Research Note

This lane exists to prevent a weak argument of the form:

"A single Gaussian failed, therefore particles are necessary."

If a small Gaussian mixture captures the oracle posterior adequately, PF should
not be promoted automatically.
