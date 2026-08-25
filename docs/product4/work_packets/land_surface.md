# LAND work packet

## Objective

Turn the bounded LAND/TGSIM work into a defensible source-backed cohort without confusing the
three-track fixture with the separate prepared vehicle-class task.

## Current boundary

- Lane: `land_surface`
- Existing prepared task: TGSIM speed-profile `passenger_car` versus `truck`.
- Aggregate six-lane fixture: common-front validation only; it has no classifier view.
- The classifier target must remain `platform_class`; physical-platform grouping is mandatory.

## Inputs needed

- One complete-track, hash-pinned source artifact with actual rows, labels, timestamps, and split groups.
- The existing external prepared cohort and its train/validation/test assignments.
- An independent road-vehicle provider or geography for source-shift validation.

## Agent tasks

1. Inspect the complete source and record schema, coordinate frame, sampling, labels, and rights.
2. Extend the common front only where the source semantics require it; keep identifiers out of features.
3. Build an external prepared snapshot with deterministic grouped splits and train-only empirical hooks.
4. Run the Product 2 bridge using `platform_class`; report within-source and source-shift results separately.

## Acceptance gates

- Source and artifact hashes are recorded; raw rows remain external.
- Physical-platform, recording, and geography grouping audits pass.
- Classifier assets contain only the declared numeric projection and timestamps.
- Both vehicle classes occur in train, validation, and test under the declared study policy.
- Independent-source performance is reported as a separate gate, not pooled into the original task.

## Validation

```bash
PYTHONPATH=src python3 scripts/run/run_product4_tests.py --workers 4
PYTHONPATH=src python3 scripts/audit/build_analysis_product_manifest.py \
  --snapshot /external/product4-land-prepared/snapshot.json \
  --product classifier_ladder \
  --target-label-namespace platform_class \
  --output /external/product4-land-prepared/classifier-ladder.json
```

## Do not claim

Do not claim population-wide road performance, provider generalization, or six-lane readiness from
the bounded TGSIM task alone.
