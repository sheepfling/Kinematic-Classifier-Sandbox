# UKF / EKF

Unscented and extended Kalman filters cover the nonlinear-but-still-unimodal
regime before the ladder escalates to Gaussian mixtures or particles.

## Role In The Classifier

Use UKF / EKF when the measurement or dynamics are nonlinear but the posterior
is still well summarized by a single Gaussian. This is the blocker rung that
prevents a weak escalation story of the form:

"The linear Kalman model failed, therefore particles are necessary."

The current oracle-backed witness is:

- `ukf_nonlinear_unimodal_oracle_v1`

## Contract Hook

The witness emits:

- `artifacts/ukf_nonlinear_unimodal_oracle_v1/truth_trajectory.csv`
- `artifacts/ukf_nonlinear_unimodal_oracle_v1/measurements.csv`
- `artifacts/ukf_nonlinear_unimodal_oracle_v1/grid_oracle_posterior_history.csv`
- `artifacts/ukf_nonlinear_unimodal_oracle_v1/method_posterior_history.csv`
- `artifacts/ukf_nonlinear_unimodal_oracle_v1/kalman_baseline_posterior_history.csv`
- `artifacts/ukf_nonlinear_unimodal_oracle_v1/state_estimate_history.csv`
- `artifacts/ukf_nonlinear_unimodal_oracle_v1/summary.csv`
- `artifacts/ukf_nonlinear_unimodal_oracle_v1/decision_card.md`

## Current Read

The current witness and follow-on audit are now strong enough to mark UKF as
`study_justified` on the `nonlinear_unimodal_sensor` family. The result says:

- UKF stays much closer to the oracle posterior than the linear Kalman proxy.
- The witness is still unimodal, so the gain does not by itself justify GSF or PF.
- The bounded promotion packet
  `artifacts/ukf_nonlinear_promotion_audit_v1/ukf_nonlinear_promotion_audit_report.md`
  shows the gain survives a small seed-and-geometry sweep.
- The repo still lacks EKF comparison and 2D nonlinear geometry.

That is the intended claim boundary: UKF now occupies the nonlinear Gaussian
blocker rung, but it is not yet generalized beyond the named witness family.
