# AIR work packet

## Objective

Replace the documented readsb parser witness with a rights-reviewed historical flight corpus and
an independent provider holdout.

## Current boundary

- Lane: `air_atmospheric`
- Current fixture: seven-sample documented A320 trace used to test readsb parsing and vertical basis.
- Evidence state: `access_verified`; it is not a historical release trace.
- Classifier view remains intentionally blocked.

## Inputs needed

- A genuine historical ADS-B/readsb trace with immutable artifact hash.
- Rights/terms decision for raw and derived redistribution.
- A bounded independent-provider holdout, such as OpenSky, with rights resolved.

## Agent tasks

1. Confirm trace schema, leg segmentation, stale-position handling, and time semantics.
2. Preserve barometric versus geometric altitude and vertical-rate basis per sample.
3. Build one historical flight-leg common-contract episode, then a provider holdout.
4. Define physical-aircraft, recording, mission-event, and temporal grouping before any classifier task.

## Acceptance gates

- Historical artifact and source rights are recorded in the registry.
- Flight-leg segmentation is reproducible and quality findings are retained.
- Missing vertical values remain null with validity flags.
- Provider/source shift is evaluated separately from within-source performance.
- Prepared promotion occurs only after identity-free classifier projection and grouped splits pass.

## Validation

```bash
PYTHONPATH=src python3 -m pytest -q -m product4_air_atmospheric
PYTHONPATH=src python3 scripts/audit/evaluate_product4_lane_matrix.py \
  --snapshot /external/product4-air-historical/snapshot.json \
  --assignments /external/product4-air-historical/assignments.json \
  --expected-lane air_atmospheric
```

## Do not claim

Do not treat the documented A320 fixture as historical flight evidence, authoritative aircraft
identity, or classifier performance.
