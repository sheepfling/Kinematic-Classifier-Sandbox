# Learned Model Mismatch

The repo tracks the learned-filter lane, but it does not yet have a trained
learned-filter witness.

## What It Proves

The current gate is intentionally still open:

- `kalmannet` remains `researched`
- `differentiable_pf` remains `researched`

The coverage matrix already tracks `kalmannet_family` and the broader
learned-filter lane, but that is not the same thing as a witness that proves a
learned correction is better than the current classical mismatch ladder.

The witness that still needs to be built is:

- `learned_model_mismatch`

## Claim Boundary

What remains open before this lane can be promoted:

- a trained KalmanNet-style witness on a partial-model-knowledge problem
- a differentiable PF witness that does not collapse into a generic PF proxy
- matched-budget comparisons against UKF, Student-t Kalman, and particle methods
- robustness sweeps showing the gain survives seed and noise variation
