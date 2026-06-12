# BOSS / WEASEL / TDE Dictionary Methods

The repo now tracks the dictionary-method lane through the shared modern-TSC
execution packet:

- study id: `tsc_archive_baseline_frontier_v1`
- artifacts: `artifacts/tsc_archive_baseline_frontier_v1/`
- family method id in that packet: `dictionary_tde_family`

## What It Proves

This lane is now explicit in the method-validation operating system and has a
concrete execution surface plus backend provenance on the shared binary
dynamics corpus.

The current packet is enough to justify:

- `dictionary_tde_family` moving to `witness_supported`

The lane is now also covered by:

- `archive_vs_physics_witness_v1`, where the current dictionary-family row
  beats the existing interpretable and physics-aware baselines overall
- `archive_feature_headroom_witness_v1`, where the bounded external
  dictionary-family row matches the engineered timing-order champion
- `archive_backend_diagnosis_v1`, which now acts as a bounded tuning surface
  rather than a pure failure packet

## Claim Boundary

This is not a claim that the repo contains finished or parity-backed BOSS,
WEASEL, or TDE implementations.

What remains open:

- faithful external-method wrapping or implementation
- broader calibration and seed-stability coverage beyond the bounded packet
- broader comparison against named physics-aware witnesses

The current environment now executes a real external `sktime`
`WEASEL` backend in the shared archive frontier. That is enough to prove the
lane is more than pure fallback scaffolding and enough to support bounded
witness status, but it is still not enough to claim finished BOSS / WEASEL /
TDE parity or broad promotion.

The registry therefore now keeps `dictionary_tde_family` at
`witness_supported` on the current bounded archive packets.
