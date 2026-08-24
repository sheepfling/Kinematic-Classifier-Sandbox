# BLK-002 — pNEUMA Official License Text Conflict

## Evidence

Official pNEUMA pages currently disagree:

- the download page displays `CC BY-NC 4.0`;
- the FAQ on that page says `CC BY 4.0`;
- the terms page states free distribution/use with attribution but does not clearly reconcile the
  license version.

## Decision required

COMMON-FRONT/coordinator should establish the governing rights statement before any pNEUMA raw-row
fixture is redistributed in the shared corpus.

## Affected checks

- source hard gate: rights / redistribution;
- fixture portability;
- release packaging.

## Safest temporary disposition

Treat access as verified but rights as `conditional`; do not redistribute pNEUMA source rows yet.
