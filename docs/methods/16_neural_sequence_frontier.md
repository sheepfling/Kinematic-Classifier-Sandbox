# TCN / InceptionTime

The repo now has a real local training frontier for the neural classifier lane:

- study id: `neural_sequence_vs_physics_frontier_v1`
- artifacts: `artifacts/neural_sequence_vs_physics_frontier_v1/`

The lane now also has a bounded robustness companion:

- study id: `neural_sequence_robustness_frontier_v1`
- purpose: bounded multi-seed robustness read for `tcn` and `inceptiontime`

## What It Proves

This packet now trains local `torch` models with a held-out calibration split.
It compares:

- `tcn`
- `inceptiontime`
- `windowed_robust`
- `rocket_proxy`
- `kalman_bank`

on the shared binary dynamics corpus with an explicit train/calibration/test split.

The current packet is enough to justify:

- `tcn` moving to `witness_supported`
- `inceptiontime` moving to `witness_supported`

## Claim Boundary

This is a claim that the repo trains real local neural sequence models and
applies held-out temperature scaling. It is not yet a claim of broad
robustness or external benchmark completeness.

What remains open:

- broader seed-stability and robustness checks beyond the current bounded
  multi-seed packet
- broader comparison against stronger physics-aware methods on named witnesses

The registry therefore now keeps both methods at `witness_supported` with
conservative claim boundaries.
