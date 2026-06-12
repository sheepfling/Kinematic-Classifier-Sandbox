# DrCIF / Interval Forests

The repo now tracks the interval-forest lane through the shared modern-TSC
execution packet:

- study id: `tsc_archive_baseline_frontier_v1`
- artifacts: `artifacts/tsc_archive_baseline_frontier_v1/`
- family method id in that packet: `drcif_interval_forests`

## What It Proves

This lane is now explicit in the method-validation operating system and has a
concrete execution surface plus backend provenance on the shared binary
dynamics corpus.

The current packet is enough to justify:

- `drcif_interval_forests` moving to `trace_validated`

The lane is now also covered by:

- `archive_vs_physics_witness_v1`, where DrCIF now matches the best current
  baseline overall but does not exceed it
- `archive_feature_headroom_witness_v1`, where the bounded external DrCIF row
  now matches the engineered timing-order champion
- `drcif_interval_promotion_audit_v1`, which turns the remaining blocker into a
  narrow method-level decision instead of a broad archive-family diagnosis

## Claim Boundary

This is not a claim that the repo contains a finished or parity-backed DrCIF
implementation.

What remains open:

- stronger than tie-level witness performance on at least one named archive
  packet
- a positive method-level promotion audit rather than only parity on the named
  witnesses
- broader comparison against named physics-aware witnesses

The optional backend layer now attempts a real `sktime` `DrCIF` path before
falling back. The current shared frontier uses a compact bounded configuration
so the method really executes inside the Epic 2 packet instead of timing out
indefinitely.

The method now has repeatable non-fallback benchmark evidence, bounded
seed/calibration support, and a narrow promotion audit. It therefore reaches
`trace_validated`, but the current bounded archive witnesses only bring it to
parity rather than a positive witness win.

The registry therefore keeps `drcif_interval_forests` at `trace_validated`, not
`witness_supported`.
