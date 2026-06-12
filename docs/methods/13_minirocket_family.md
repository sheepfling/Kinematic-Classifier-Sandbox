# MiniRocket / MultiRocket / HYDRA

The repo now has a ROCKET-style benchmark scaffold for the modern time-series
classifier lane.

## Current Scope

This is not yet a claim that exact MiniRocket, MultiRocket, or HYDRA has been
executed in the current environment. The lane now has two surfaces:

- a deterministic local `rocket_proxy` benchmark in the shared comparison
- an optional external archive-wrapper path in the archive frontier

## Contract Hook

The current scaffold is visible through:

- `artifacts/common_dataset_comparison_v1/method_summary.csv`
- `artifacts/common_dataset_comparison_v1/common_dataset_comparison_report.md`
- `artifacts/tsc_archive_baseline_frontier_v1/metric_summary.csv`

The method name in that surface is:

- `rocket_proxy`

## Claim Boundary

This moves the `minirocket_family` lane to `implemented`, because the repo now
has an explicit execution surface and backend-provenance packet for this
family.

The current external path is now a real `aeon` `MiniRocketClassifier` run when
the optional backend environment is healthy. The repo sets a local
`NUMBA_CACHE_DIR` before importing `aeon`, which avoids the cache-locator
failure that previously made the MiniRocket-family path look unavailable.

The lane is now also covered by:

- `archive_vs_physics_witness_v1`, which still shows the archive family losing
  to `windowed_robust` and `kalman_bank`
- `archive_feature_headroom_witness_v1`, which still shows the archive family
  losing decisively to the engineered timing-order baseline
- `archive_backend_diagnosis_v1`, which tests bounded panel/resampling variants
  and still keeps the gate closed

It is still not promoted as finished because:

- the current family result can still rely on fallback behavior,
- there is no exact MiniRocket / MultiRocket / HYDRA parity claim yet,
- generic `RocketClassifier` execution does not count as MiniRocket-family parity,
- the broader archive family still has timeout-heavy `DrCIF` and `HIVE-COTE` rows,
- the lane has not passed seed robustness or calibration checks.
