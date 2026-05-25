# Witness: sequential_history

Purpose: Show that evidence accumulated over time can resolve cases where pointwise evidence is weak or noisy.

Class set: common 1D classes from the accumulator dataset manifest.

Feature set: pointwise evidence accumulated through time.

Classifier/filter family: sequential Bayes accumulator.

Prior regime: accumulator priors and sensitivity summary in `artifacts/bayes_accumulator/prior_sensitivity.csv`.

Corpus objective: time-series examples where local evidence varies and history should improve confidence.

What it proves: posterior histories, confidence crossings, and time-to-correct behavior are meaningful outputs.

What it does not prove: dynamics-model residuals or switching-mode transitions.

Key equations: recursive log-evidence accumulation and log-sum-exp posterior normalization.

Key plots:
- `artifacts/bayes_accumulator/bayes_accumulator_diagnostics.png`
- `artifacts/monte_carlo_accumulator/accuracy_vs_time.png`

Key tables:
- `artifacts/bayes_accumulator/confidence_crossings.csv`
- `artifacts/monte_carlo_accumulator/metrics_by_time.csv`

Key artifacts:
- `artifacts/bayes_accumulator/posterior_history.csv`
- `artifacts/monte_carlo_accumulator/calibration_bins.csv`

Promotion status: promote as the sequential posterior witness.

Next extension toward 3D: accumulate evidence from 3D feature families and compare time-to-confidence by trajectory regime.
