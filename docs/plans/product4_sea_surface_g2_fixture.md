# Product 4 SEA-Surface G2 Fixture Tranche

## Decision

Land this tranche as a stacked draft PR on `feature/product-4-real-world-corpus`.
It establishes a real `sea_surface` G2 expansion/regression witness while keeping the preferred
NOAA anchor and Kystverket independent holdout at their honest evidence states.

## Delivered surface

- a generic `TrajectoryEpisodeManifest` / `TrajectoryStateViewManifest` metadata contract;
- a source-specific CMRE/Brest route-tracklet adapter;
- source-native AIS channels and a named local-ENU analysis view;
- explicit per-component validity, including unavailable vertical state as invalid and `NaN`;
- reported SOG/COG-to-horizontal-velocity derivation with lineage;
- source duplicate-time preservation plus named classifier-view deduplication;
- route labels outside classifier assets;
- opaque physical-platform grouping across repeated source tracklets;
- machine-readable source portfolio, bounded-extract specification, and validation receipt;
- network-free tests using a synthetic shape fixture.

## Evidence disposition

| Source | Role | Current evidence state | This tranche |
| --- | --- | --- | --- |
| CMRE/Brest route tracklets | expansion/regression | `fixture_validated` | integrated as the real G2 witness |
| NOAA 2024 AIS | preferred anchor | `schema_inspected` | documented, not promoted |
| Kystverket historical AIS | independent holdout | `access_verified` | documented, not promoted |

## Claim boundary

This tranche proves that real AIS-derived surface-maritime tracklets can be represented through the
Product 4 common front with truthful frame, time, state-role, label, quality, grouping, and
classifier-access semantics.

It does not establish:

- vessel-family classifier performance;
- complete-voyage segmentation;
- a population-representative surface-maritime corpus;
- a completed NOAA anchor pilot;
- independent-provider validation;
- underwater or military-submarine semantics.

## Redistribution decision

Do not commit the real CMRE source rows or generated coordinate assets in this PR. The available
evidence supports a conservative `accept_with_restrictions` disposition because the accompanying
article identifies CC BY-NC-SA 4.0 terms while available catalog metadata was not fully consistent.
The repository stores the upstream commit/blob identity, selected tracklet IDs, external fixture
packet checksum, adapter, and tests instead.

## Merge gates

1. Contract and adapter tests pass without network access.
2. Ruff, Pyright, import-simplicity audit, and repository check lane pass in CI.
3. PR body preserves source roles and evidence states exactly.
4. No raw MMSI, route target, provider identifier, or source filename enters classifier assets.
5. No vertical value is represented as observed zero.
6. Reviewers accept the conservative non-redistribution treatment.

## Follow-up gates

- Acquire and hash an official NOAA 2024 repeated-observation artifact and validate the preferred
  anchor fixture.
- Acquire a bounded Kystverket artifact and validate the independent-provider holdout.
- Reconcile authoritative CMRE rights metadata before any real-row or derived-asset redistribution.
- Only then advance SEA-SURF toward `pilot_prepared`, classifier study execution, and independent
  validation.
