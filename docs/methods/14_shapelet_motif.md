# Shapelet / Motif

The repo now has a dedicated localized-motif witness for the `shapelet` lane:

- study id: `shapelet_maneuver_motif_v1`
- artifacts: `artifacts/shapelet_maneuver_motif_v1/`

## What It Proves

This witness targets the case where two trajectories share the same broad
track-level summaries but differ in a short maneuver signature. The simpler
windowed baseline uses global summary features and is intentionally unable to
separate the classes reliably. The shapelet lane scans short velocity-residual
subsequences and scores them against class-specific motif templates.

The current witness is enough to justify:

- a localized motif baseline in the method ladder
- `shapelet` moving from `researched` to `witness_supported`

## Claim Boundary

This is not yet a claim that the repo has a full general-purpose shapelet
library or that motif methods are broadly superior across the common corpus.

What remains open:

- larger corpus coverage beyond the dedicated motif witness
- comparison against the broader modern TSC family on the same localized motif
  studies
- robustness sweeps over motif duration, amplitude, and timing jitter
