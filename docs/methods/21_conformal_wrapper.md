# Conformal / Sequential Conformal

The repo now has a dedicated coverage-control witness for the uncertainty lane:

- study id: `coverage_control_under_shift_v1`
- artifacts: `artifacts/coverage_control_under_shift_v1/`

## What It Proves

This witness starts from the temperature-scaled posterior surface and wraps it
with split-conformal prediction sets on a shifted evaluation slice. In the
current narrow binary witness, the hard `short_noisy` slice can legitimately
expand to the full two-label set.

The current witness is enough to justify:

- `conformal_wrapper` moving from `researched` to `witness_supported`

## Claim Boundary

This is not yet a full abstention or sequential-conformal workbench.

What remains open:

- explicit abstain/defer policies tied to set size or utility
- sequential conformal variants
- broader multi-class and multi-regime coverage studies
