# HSMM

Hidden semi-Markov models cover explicit dwell-time and duration structure
before the ladder escalates to changepoint machinery or richer latent-state
models.

## Role In The Classifier

Use HSMM when regime duration is part of the failure mode. This is the blocker
rung for cases where a geometric-duration HMM can smooth labels, but still
gets maneuver onset or exit timing wrong because it has no explicit dwell-time
model.

The current witness is:

- `hsmm_duration_limited_maneuver_v1`

## Contract Hook

The witness emits:

- `artifacts/hsmm_duration_limited_maneuver_v1/truth_history.csv`
- `artifacts/hsmm_duration_limited_maneuver_v1/posterior_history.csv`
- `artifacts/hsmm_duration_limited_maneuver_v1/duration_chain_posterior.csv`
- `artifacts/hsmm_duration_limited_maneuver_v1/state_estimate_history.csv`
- `artifacts/hsmm_duration_limited_maneuver_v1/metrics.csv`
- `artifacts/hsmm_duration_limited_maneuver_v1/decision_card.md`

## Current Read

The current witness is strong enough to mark HSMM as `witness_supported` on
the `duration_limited_maneuver` family. The result says:

- explicit duration improves mode timing relative to an ordinary HMM,
- the main gain is on maneuver exit and dwell consistency rather than generic
  smoothing,
- this fills the duration blocker rung before BOCPD.

The claim boundary is still narrow: this is a named fixed-duration maneuver
witness, not yet a generalized statement about all duration-sensitive regime
problems.
