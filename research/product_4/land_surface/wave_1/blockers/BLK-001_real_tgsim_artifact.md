# BLK-001 — Bounded Real TGSIM Trajectory Rows

## Evidence

The official JSON distribution is reachable, but the complete response exceeds the current
web-response size limit. The local execution environment has package-only network access and cannot
resolve `data.transportation.gov`, so the same endpoint cannot currently be downloaded and sliced
locally.

## Decision / action required

Run a bounded complete-track query in an environment with direct external network access, or supply
a retained bounded TGSIM CSV/JSON artifact.

## Affected gates

- `artifact_acquired`
- actual-row `schema_inspected`
- `mapping_complete`
- `fixture_validated`

## Safest temporary disposition

`access_verified`
