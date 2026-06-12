# Physics Family Promotion Audit

The repo now has a bounded family-level audit for the physics-aware lane:

- study id: `physics_family_promotion_audit_v1`
- artifacts: `artifacts/physics_family_promotion_audit_v1/`

## What It Proves

This packet does not introduce a new witness. It summarizes why the
advanced-filter blocker set is or is not still holding back the physics-aware
family at the Epic 2 level.

The audit focuses on the methods that currently matter most for family closure:

- `imm`
- `ukf`
- `gaussian_sum_filter`
- `rbpf`

It combines their current witness decisions with the repo's current
method-validation status and the advanced-filter comparison surface.

`switching_kalman_slds` is intentionally outside this closure packet. The repo
still tracks it in the registry, but the current Epic 2 family audit treats it
as a deferred extension rather than a core 1D closure requirement.

## Current Read

The current bounded read is now positive for the advanced-filter blocker set:

- the advanced-filter core blockers are cleared on the audited witness set
- `imm` now has a narrow switching promotion audit and is the current
  study-justified switching state-mixing blocker on the audited witness family
- `ukf` now has a narrow nonlinear promotion audit and is the current
  study-justified nonlinear Gaussian blocker before mixture or particle
  escalation on the audited witness family
- `gaussian_sum_filter` now has a narrow multimodal promotion audit and is the
  current study-justified least-complex blocker before PF on the audited
  witness family
- `rbpf` now clears the bounded compute-normalized frontier on the latent
  witness while the smooth witness remains a negative-control `metric_split`
  rather than a universal RBPF win

## Claim Boundary

This is an advanced-filter family audit, not a new promotion packet.

It is enough to show that the advanced-filter core blockers are cleared on the
current bounded witness set. The broader physics-aware family gate now depends
on how conservative we want to be about wider robustness and study-justified
coverage, not on a missing witness path for HMM / transition or the Kalman
bank.
