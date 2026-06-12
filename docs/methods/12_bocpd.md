# BOCPD

Bayesian online changepoint detection covers unknown maneuver onset before the
ladder escalates to richer latent-state models.

## Role In The Classifier

Use BOCPD when the main failure mode is onset uncertainty rather than explicit
dwell-time structure. This is the blocker rung for cases where HMM or HSMM
posteriors may smooth regime labels, but do not localize the maneuver start
cleanly enough.

The current witness is:

- `bocpd_unknown_maneuver_onset_v1`

## Contract Hook

The witness emits:

- `artifacts/bocpd_unknown_maneuver_onset_v1/truth_history.csv`
- `artifacts/bocpd_unknown_maneuver_onset_v1/posterior_history.csv`
- `artifacts/bocpd_unknown_maneuver_onset_v1/onset_posterior.csv`
- `artifacts/bocpd_unknown_maneuver_onset_v1/state_estimate_history.csv`
- `artifacts/bocpd_unknown_maneuver_onset_v1/summary.csv`
- `artifacts/bocpd_unknown_maneuver_onset_v1/decision_card.md`

## Current Read

The current witness is strong enough to mark BOCPD as `witness_supported` on
the `unknown_maneuver_onset` family. The result says:

- the changepoint posterior localizes onset more explicitly than HMM/HSMM
  maneuver posteriors,
- BOCPD improves onset-facing posterior quality without relying on fixed
  dwell-time assumptions,
- this fills the unknown-onset blocker rung after HSMM.

The claim boundary is narrow: this is a named single-onset witness, not yet a
general statement about all changepoint or segmentation problems.
