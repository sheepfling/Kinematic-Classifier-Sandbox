# Product 4 SPACE-NEAR fixtures

This package contains the bridge loader for the first Product 4 `space_near`
research tranche. It loads bounded, real NASA and JAXA mission fixtures into the
existing `CorpusTrajectory` / `NormalizedTrack` interface while preserving the
richer episode evidence as sidecar manifests.

The loader does **not** introduce a competing persistent root schema. The
per-mission compressed JSON fixture records are evidence for the planned Product 4
`TrajectoryEpisodeManifest + TrajectoryStateView[]` common front. Until that
common front lands, the adapter projects exactly one named ECEF analysis state
view into the current corpus interface.

## Included missions

- JAXA S-310-40 — anchor apogee fixture
- NASA Endurance 47.001 — independent-provider reference-solution fixture
- JAXA S-310-44 — same-family replication
- JAXA S-520-26 — early-ascent expansion
- JAXA S-520-27 — apogee expansion
- JAXA S-520-29 — terminal-descent/splashdown expansion

## Invariants

- source-native fixture rows remain alongside the analysis state in each compressed record;
- raw source workbooks and full mission files are not committed;
- source artifact URLs and SHA-256 values remain machine-readable;
- ECEF position is a named derived view and never relabeled as observation truth;
- velocity and acceleration in `NormalizedTrack` are derived for compatibility,
  while source velocity remains absent unless actually ingested;
- mission, object, provider, dataset, and recording identity remain metadata and
  split-group material, not numeric channels;
- all label dependencies and processing steps remain auditable in the episode
  manifest;
- authoritative Product 4 G2 validation remains pending COMMON-FRONT.

## Claim boundary

This tranche demonstrates reproducible ingestion and semantic preservation for
six bounded near-space missions. It does not establish classifier performance,
production adapter readiness for full raw archives, high-energy reentry
coverage, native Endurance GPS velocity ingestion, or authoritative G2
acceptance.
