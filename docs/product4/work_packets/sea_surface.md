# SEA-SURF work packet

## Objective

Extend the bounded CMRE/Brest route-motion task toward independent maritime validation while
preserving the distinction between route association and vessel-family classification.

## Current boundary

- Lane: `sea_surface`
- Prepared task: `R_06` versus `R_14` route-motion classification.
- Target namespace: `route`; route is not a vessel-family label.
- The current source/key and raw rows remain external and rights-restricted.

## Inputs needed

- An independent AIS provider or source, such as Kystverket or NOAA, with terms reviewed.
- A repeat-observation artifact with stable source and recording identifiers.
- The existing CMRE external snapshot, identity key, nomenclature, and assignments.

## Agent tasks

1. Acquire and checksum the independent artifact without copying restricted bytes into Git.
2. Map native fields into the common contract, retaining irregular sampling and missing vertical state.
3. Define a source-shift cohort whose target namespace is explicitly declared.
4. Re-run grouped leakage and Product 2 evaluation independently from the CMRE result.

## Acceptance gates

- Provider, recording, mission-event, and physical-platform grouping pass.
- Route labels remain out-of-band and are never renamed as vessel classes.
- Rights and redistribution status are explicit for raw and derived assets.
- The independent source has a distinct snapshot ID and source hash.
- Results report CMRE and independent-provider performance separately.

## Validation

```bash
PYTHONPATH=src python3 scripts/run/run_product4_tests.py --workers 4
PYTHONPATH=src python3 scripts/audit/evaluate_product4_lane_matrix.py \
  --snapshot /external/product4-sea-surface-independent/snapshot.json \
  --assignments /external/product4-sea-surface-independent/assignments.json \
  --expected-lane sea_surface
```

## Do not claim

Do not report the route task as vessel-family, maritime-population, release, or cross-domain
performance.
