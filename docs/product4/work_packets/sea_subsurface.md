# SEA-SUB work packet

## Objective

Add an independent subsurface artifact and define a legitimate multi-episode task without flattening
asynchronous navigation, measured depth, or deployment identity into a classifier label.

## Current boundary

- Lane: `sea_subsurface`
- The IOOS/UAF unit-191 selected profile passes the channel-aware common front.
- It is one profile, has no defensible classifier target cohort, and remains classifier-blocked.
- Platform type is metadata, not an automatic target label.

## Inputs needed

- One exact Sentry/PPL artifact with source hash and rights/terms evidence.
- Documentation for time basis, datum, depth convention, processing, and navigation resets.
- Multiple independent profiles, deployments, or platforms sufficient for a target task.

## Agent tasks

1. Inspect Sentry channel timing and compare it to the IOOS asynchronous-channel contract.
2. Preserve measured, derived, dead-reckoned, GPS, pressure, and source-depth semantics separately.
3. Define grouping namespaces before proposing a target label or split.
4. Prepare only a task with a defensible label namespace and independent holdout.

## Acceptance gates

- Duplicate timestamps are represented as channel events, not silently deduplicated states.
- Time, depth, datum, and navigation-phase findings are explicit in quality summaries.
- Deployment/platform/recording groups prevent leakage across splits.
- The target label is evidence-backed and not merely `platform_type`.
- Classifier projection remains absent until the multi-episode task is approved.

## Validation

```bash
PYTHONPATH=src python3 -m pytest -q -m product4_sea_subsurface
PYTHONPATH=src python3 scripts/audit/evaluate_product4_lane_matrix.py \
  --snapshot /external/product4-sea-subsurface/snapshot.json \
  --assignments /external/product4-sea-subsurface/assignments.json \
  --expected-lane sea_subsurface
```

## Do not claim

Do not claim subsurface platform classification, source-shift performance, or navigation truth from
the single IOOS profile.
