# Kalman Bank

The Kalman-bank rung turns model-specific innovations into class evidence. Each
candidate dynamics model has its own Kalman filter; innovation likelihoods are
compared as evidence for the corresponding class or mode.

## Role In The Classifier

Use this rung when feature or endpoint evidence is ambiguous but dynamics are
separable. The witness is `kalman_endpoint_match`.

## Contract

- Inputs: measurements, class-conditioned linear-Gaussian model specs, priors.
- Outputs: innovation history, posterior history, and benchmark summaries.
- Gate: the Kalman bank should improve dynamics-separable cases before
  switching-aware methods are promoted.

The intermediate packet for this rung now includes:

- `artifacts/kalman_filter_bank/traces/filter_step_trace.csv`
- `artifacts/kalman_filter_bank/plots/intermediate/measurement_prediction_timeline.png`
- `artifacts/kalman_filter_bank/plots/intermediate/innovation_likelihood_strip.png`
- `artifacts/kalman_filter_bank/plots/intermediate/uncertainty_diagnostics.png`
- `artifacts/kalman_filter_bank/step_cards/t_mid.md`

## Claim Boundary

The Kalman bank is the model-based baseline for IMM. If this rung is sufficient,
IMM should not be promoted for the same study.
