# Advanced State Inference: 1D IMM Proof To 3D PVA Lift

The advanced-state-inference rung is the switching-aware extension of the
classifier/filter ladder. In this repo it is currently proved with a 1D
interacting multiple model (IMM) witness, but the contract is intentionally
dimension-agnostic so the same evaluation surface can lift to 3D PVA without
changing the downstream harness.

The decision to escalate into this rung is governed by the rung sufficiency
evaluator in `PLN-026`. That layer decides whether the current rung is
sufficient, near its practical limit, or legitimately replaced by a stronger
state-inference method. In practice, the IMM proof is only meaningful when the
corpus passes preconditions, the switching witness is learnable, and the
measured improvement over the transition-matrix rung is real.

## What stays invariant

- The backend must emit state summaries, evidence summaries, diagnostics, and
  posterior-compatible rows.
- The evaluation surface must continue to consume shared posterior histories
  rather than method-specific internals.
- The comparison target remains the same ladder contract:
  `pointwise -> windowed -> sequential Bayes -> Kalman bank -> transition
  matrix -> advanced state inference`.

## What changes in 3D PVA

- The state vector grows from scalar kinematics to a block PVA state.
- The measurement matrix changes from position-only observation to the
  relevant 3D observation layout.
- Mode models become axis-blocked instead of scalar-only.
- The same mode posterior, innovation likelihood, and diagnostic rows remain
  valid.

## Why the 1D proof matters

The 1D IMM witness proves the integration claim that matters most:

1. switching trajectories can be evaluated with a shared evidence contract,
2. mode posteriors can be compared against the transition-matrix rung,
3. state estimates and diagnostics can be written into the same artifact
   pipeline as the rest of the study framework.

That means the repo is not committing to PF or RBPF prematurely. It is proving
the switching-aware rung first, then keeping the contract stable for the later
3D lift.

## Expected 3D artifact differences

- `state_mean` becomes a 3D PVA block vector instead of a 1D PVA vector.
- `state_covariance` becomes a larger block covariance matrix.
- Diagnostic summaries may add axis-specific residuals or per-axis innovation
  norms.
- The artifact names, posterior history, and evaluation checks stay the same.

## Practical rule

If a future 3D backend can be compared through the same `FilterEvidence`,
`FilterStateSummary`, and shared posterior history rows, it belongs in the same
advanced-state-inference rung. If it needs a different evaluation surface, the
contract is too narrow and should be widened before the lift.
