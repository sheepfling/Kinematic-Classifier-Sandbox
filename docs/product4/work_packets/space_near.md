# SPACE-NEAR work packet

## Objective

Resolve the authoritative semantic boundary for the six near-space fixtures before creating a
prepared classifier cohort.

## Current boundary

- Lane: `space_near`
- Six bounded missions pass common-front contract construction: 206 samples and 16 label assertions.
- Authoritative semantic sign-off remains pending.
- Classifier projection is intentionally absent.

## Inputs needed

- Source documentation or exact artifacts resolving vertical datum and source-frame meaning.
- Mission-specific launch epoch, phase, powered/coast, apogee, descent, and terminal-event evidence.
- Independent-provider interpretation for the NASA Endurance reference solution.

## Agent tasks

1. Review each mission's source-native versus reference/analysis state semantics.
2. Resolve WGS84/height assumptions, Cartesian frame semantics, launch epochs, and phase intervals.
3. Keep mission, platform, recording, and temporal identities as audit-only grouping values.
4. Propose a prepared task only after a defensible target label and multi-mission split policy exist.

## Acceptance gates

- Each datum/frame/epoch assumption is sourced or explicitly marked unresolved.
- Reference solution is not relabeled as observation truth.
- Mission identity never enters classifier numeric channels.
- Grouped train/validation/test assignments pass for the proposed task.
- Rights and source-artifact boundaries are recorded before prepared promotion.

## Validation

```bash
PYTHONPATH=src python3 -m pytest -q -m product4_space_near
PYTHONPATH=src python3 scripts/run/build_space_near_validation_snapshot.py \
  --snapshot-root /external/product4-space-near \
  --snapshot-id product4-space-near-validation-v1
```

## Do not claim

Do not claim near-space classifier performance, propulsion-state classification, or authoritative
measurement truth from the current six-fixture tranche.
