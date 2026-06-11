# MiniRocket / MultiRocket / HYDRA

The repo now has a ROCKET-style benchmark scaffold for the modern time-series
classifier lane.

## Current Scope

This is not yet a claim that exact MiniRocket, MultiRocket, or HYDRA has been
faithfully implemented. The current lane is an honest proxy:

- a deterministic random-convolution style transform
- shared-corpus classification on the existing 1D dynamics benchmark
- output through the existing common-dataset comparison surface

## Contract Hook

The current scaffold is visible through:

- `artifacts/common_dataset_comparison_v1/method_summary.csv`
- `artifacts/common_dataset_comparison_v1/common_dataset_comparison_report.md`

The method name in that surface is:

- `rocket_proxy`

## Claim Boundary

This moves the `minirocket_family` lane from pure research tracking to
`implemented`, because the repo now has a non-physics TSC benchmark scaffold in
the right artifact surface.

It is still not `witness_supported` because:

- the current `tsc_archive_baseline_frontier` packet is only a proxy frontier,
- there is no exact MiniRocket / MultiRocket / HYDRA fidelity claim yet,
- the lane has not passed seed robustness or calibration checks.
