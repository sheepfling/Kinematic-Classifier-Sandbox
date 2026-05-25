# Witness: windowed_outlier_extrema

Purpose: Show that raw extrema can be fragile under outlier stress and robust extrema can improve posterior stability.

Class set: current common 1D classes from `artifacts/windowed_baseline/dataset_manifest.json`.

Feature set: raw extrema, robust extrema, and windowed shape features.

Classifier/filter family: windowed feature likelihood classifier.

Prior regime: raw and robust window prior sweeps in `artifacts/prior_sensitivity_windowed_raw_v1/` and `artifacts/prior_sensitivity_windowed_robust_v1/`.

Corpus objective: outlier and extrema stress that differentiates raw feature behavior from robust feature behavior.

What it proves: feature design changes posterior stability and confusion.

What it does not prove: windowed cumulative features are independent Bayesian evidence.

Key equations: `log p(phi_t | c)` with a warning that overlapping windows can double-count if treated as independent fresh evidence.

Key plots:
- `artifacts/windowed_baseline/windowed_baseline_diagnostics.png`
- `artifacts/feature_analysis_robust_extrema_v1/pairwise_overlap_heatmap.png`

Key tables:
- `artifacts/windowed_baseline/confusion_raw.csv`
- `artifacts/windowed_baseline/confusion_robust.csv`

Key artifacts:
- `artifacts/feature_analysis_raw_extrema_v1/feature_separation_scores.csv`
- `artifacts/feature_analysis_robust_extrema_v1/feature_separation_scores.csv`

Promotion status: revise/promote by feature set and class pair.

Next extension toward 3D: define robust vector-valued extrema and stress them under 3D trajectory outliers.
