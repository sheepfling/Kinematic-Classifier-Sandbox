# Archive vs Physics Witness

The repo now has a named comparison packet for the generic archive lane against
the current interpretable and physics-aware baselines:

- study id: `archive_vs_physics_witness_v1`
- artifacts: `artifacts/archive_vs_physics_witness_v1/`

## What It Proves

This packet compares:

- `minirocket_family`
- `drcif_interval_forests`
- `dictionary_tde_family`
- `hive_cote`

against:

- `windowed_robust`
- `kalman_bank`

on the shared binary dynamics corpus.

The packet is deliberately narrow. Its job is to make the archive lane compete
against real Epic 2 baselines under a named witness instead of remaining only a
wrapper/provenance surface.

It records:

- per-method test accuracy, NLL, ECE, and scenario slices
- archive-minus-best-baseline deltas on `test`, `short_noisy`,
  `endpoint_match`, and `outlier`
- scenario winner rows
- inherited archive backend provenance, seed-stability, and calibration reads

## Claim Boundary

This is not enough to promote the archive lane by itself.

If any archive family row is still backed by `local_proxy`, the promotion gate
stays closed even if that row beats the current baselines on this witness.

Promotion requires:

- non-fallback external execution for the archive families
- bounded robustness and calibration
- broader named witness coverage beyond this single shared-corpus packet

The current real external run is no longer just a negative result. All archive
families execute externally, the bounded robustness/calibration read passes,
and `minirocket_family` is now the shared-corpus witness champion over the
current `windowed_robust` and `kalman_bank` baselines.

That still does not close the whole archive family by itself. It only proves
that the generic-TSC lane now has a real positive witness path instead of a
wrapper-only story.
