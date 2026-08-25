# SPACE-ORB work packet

## Objective

Build a multi-object orbital cohort with verified provenance while preserving frame, epoch, and
propagation lineage.

## Current boundary

- Lane: `space_orbital`
- NASA ISS CCSDS OEM fixture passes common-front contract construction but is validation-only.
- Source velocity remains distinct from finite-difference velocity.
- Classifier projection is intentionally blocked; IGS is the preferred independent anchor candidate.

## Inputs needed

- Exact official IGS SP3 artifact, checksum, access terms, and source-equivalence evidence.
- Multiple physical orbital objects and epoch-valid identity mappings.
- A declared cohort/split policy that holds out objects, recordings, or source providers as appropriate.

## Agent tasks

1. Acquire and checksum the official artifact without committing restricted bytes.
2. Validate SP3/OEM parsing, frame, epoch, central-body, and propagated-versus-observed semantics.
3. Preserve source velocity and derived kinematics as separate channels and lineage steps.
4. Build a multi-object validation snapshot before proposing any classifier view.

## Acceptance gates

- Official artifact provenance and rights are resolved.
- Object identity is epoch-valid and used only for grouping/audit.
- Frame and time conversions are explicit and reproducible.
- Object/provider/recording grouping passes leakage audits.
- A classifier task has a defensible target label unrelated to NORAD/spacecraft identity.

## Validation

```bash
PYTHONPATH=src python3 -m pytest -q -m product4_space_orbital
PYTHONPATH=src python3 scripts/audit/evaluate_product4_lane_matrix.py \
  --snapshot /external/product4-space-orbital/snapshot.json \
  --assignments /external/product4-space-orbital/assignments.json \
  --expected-lane space_orbital
```

## Do not claim

Do not claim orbital population performance, measurement truth, or classifier readiness from the
single NASA ISS OEM validation arc.
