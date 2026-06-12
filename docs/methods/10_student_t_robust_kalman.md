# Student-t / Robust Kalman

Student-t and related robust Kalman updates cover the heavy-tail and outlier
regime before the ladder escalates to particle methods.

## Role In The Classifier

Use this rung when the system dynamics remain linear enough for a Gaussian
state summary, but the measurement noise is heavy-tailed enough that a
Gaussian likelihood becomes overconfident and brittle.

The current oracle-backed witness is:

- `student_t_heavy_tail_oracle_v1`

## Contract Hook

The witness emits:

- `artifacts/student_t_heavy_tail_oracle_v1/truth_trajectory.csv`
- `artifacts/student_t_heavy_tail_oracle_v1/measurements.csv`
- `artifacts/student_t_heavy_tail_oracle_v1/grid_oracle_posterior_history.csv`
- `artifacts/student_t_heavy_tail_oracle_v1/robust_posterior_history.csv`
- `artifacts/student_t_heavy_tail_oracle_v1/gaussian_baseline_posterior_history.csv`
- `artifacts/student_t_heavy_tail_oracle_v1/state_estimate_history.csv`
- `artifacts/student_t_heavy_tail_oracle_v1/summary.csv`
- `artifacts/student_t_heavy_tail_oracle_v1/decision_card.md`

## Current Read

The current witness is strong enough to mark Student-t / robust Kalman as
`witness_supported` on the `heavy_tail_outlier_tracking` family. The result
says:

- the robust update stays much closer to the oracle posterior than the
  Gaussian Kalman proxy,
- the robust update restores coverage under outliers rather than only chasing
  point estimates,
- this failure mode does not by itself justify PF.

The claim boundary is deliberate: this fills the heavy-tail blocker rung
before PF, but it does not yet generalize beyond the named witness family.
