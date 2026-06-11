# TCN / InceptionTime

The repo now has a sequence-style proxy frontier for the neural classifier lane:

- study id: `neural_sequence_vs_physics_frontier_v1`
- artifacts: `artifacts/neural_sequence_vs_physics_frontier_v1/`

## What It Proves

This packet keeps the neural sequence lane concrete without pretending that
full deep-model training already exists in the repo. It compares:

- `tcn_proxy`
- `inception_proxy`
- `windowed_robust`
- `rocket_proxy`
- `kalman_bank`

on the shared binary dynamics corpus with an explicit train/test split.

The current packet is enough to justify:

- `tcn` moving from `researched` to `implemented`
- `inceptiontime` moving from `researched` to `implemented`

## Claim Boundary

This is not yet a claim that the repo has trained or validated real TCN or
InceptionTime models.

What remains open:

- real model training and split discipline for neural baselines
- calibration and seed-stability checks for trained models
- broader comparison against stronger physics-aware methods on named witnesses

The registry therefore keeps both methods at `implemented`, not
`witness_supported`.
