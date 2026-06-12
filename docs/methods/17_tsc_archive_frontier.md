# HIVE-COTE / Modern TSC Frontier

The repo now has a dedicated packet for the broader modern TSC lane:

- study id: `tsc_archive_baseline_frontier_v1`
- artifacts: `artifacts/tsc_archive_baseline_frontier_v1/`

## What It Proves

This packet keeps the modern TSC lane explicit while recording backend
provenance per family. It compares:

- `minirocket_family`
- `drcif_interval_forests`
- `dictionary_tde_family`
- `hive_cote`
- `windowed_robust`
- `kalman_bank`

on the shared binary dynamics corpus.

The current packet is enough to justify:

- family-level archive method rows in the frontier artifact
- optional external wrapper routing when `aeon` or `sktime` is installed
- explicit local fallback reporting when those packages are absent
- an explicit integration read of `wrapper_stage_only`,
  `mixed_external_and_fallback`, or `all_external`
- explicit per-family external `available / attempted / succeeded / failed`
  evidence instead of collapsing all failures into `local_proxy`
- a narrow seed-sweep summary and bounded binary calibration read inside the
  same shared packet

With the current `.venv`, all four archive families now execute through their
real bounded external backends in this packet:

- `MiniRocketClassifier`
- `DrCIF`
- `WEASEL`
- `HIVECOTEV2`

## Claim Boundary

This is an execution and provenance frontier, not a witness-backed promotion
surface for the archive families.

What remains open:

- non-fallback, family-appropriate witness support for each archive method
- broader calibration and seed-stability coverage beyond the narrow bounded
  packet
- broader comparison against named physics-aware witnesses

The current external run is no longer a blanket negative. The archive rows now
pass the execution check, bounded calibration read, and bounded seed-stability
read. Several archive-family rows also beat the current shared-corpus
baselines.

The repo now also has a bounded diagnosis companion:

- study id: `archive_backend_diagnosis_v1`
- artifacts: `artifacts/archive_backend_diagnosis_v1/`

That packet tests panel variants, channel sets, resample lengths, and warning
load. It now acts as a bounded tuning surface rather than only a failure
explanation packet.

The packet now attempts real family-appropriate wrapper paths for:

- `MiniRocket` / `MultiRocket` / `HYDRA`
- `DrCIF`
- `WEASEL`
- `HIVECOTEV2`

The optional backend loader now also assigns a local `NUMBA_CACHE_DIR` under the
machine temp/cache area before importing `aeon`, so the MiniRocket-family path
does not fail just because the virtualenv lives behind the repo's symlinked
cache tree.

For heavier archive families, the current wrapper layer uses compact bounded
configs rather than default large-budget settings. That is deliberate. The
claim here is "real external execution inside the shared Epic 2 packet," not
"full-scale parity with unconstrained archive benchmark defaults."

but it still keeps the gate closed whenever the run falls back on any local
proxy row or only partially executes the external family.

`RocketClassifier` does not count as a faithful `minirocket_family` backend.
If the current environment only exposes generic ROCKET wrappers while the exact
MiniRocket-family classes are missing or broken, this packet records
`minirocket_family` as fallback rather than inflating the external count.

For environment diagnosis, the repo now also has a tiny companion probe:

- study id: `tsc_archive_backend_smoke_v1`
- artifacts: `artifacts/tsc_archive_backend_smoke_v1/`

That smoke packet is separate from the shared archive frontier. Its job is to
show whether each archive family is unavailable, attempted and failed,
attempted and timed out, or succeeded on a minimal external-wrapper smoke
surface. The smoke surface is intentionally minimal: one trajectory per class
and scenario family, not the full shared benchmark split.

The registry now keeps `minirocket_family`, `dictionary_tde_family`, and
`hive_cote` at `witness_supported` on the current bounded archive packets.
`drcif_interval_forests` remains the lone archive-family holdout because the
current bounded packets bring it to parity rather than a clean positive witness
win.
