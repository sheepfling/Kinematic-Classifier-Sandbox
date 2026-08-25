# Product 4 SPACE-NEAR research tranche

## Scope

This tranche contributes six bounded real-world sounding-rocket mission
fixtures from two independent providers. It is stacked on the Product 4
architecture branch and is intended to merge after the Product 4 declaration.

## Portfolio

| Mission | Provider | Role | Samples | Regime |
| --- | --- | --- | ---: | --- |
| S-310-40 | JAXA DARTS | anchor | 31 | apogee transition |
| Endurance 47.001 | NASA SPDF/CDAWeb | independent validation | 61 | apogee transition |
| S-310-44 | JAXA DARTS | same-family replication | 31 | apogee transition |
| S-520-26 | JAXA DARTS | expansion | 31 | early ascent |
| S-520-27 | JAXA DARTS | expansion | 21 | apogee transition |
| S-520-29 | JAXA DARTS | expansion | 31 | terminal descent |

The repository fixtures total 206 samples. The Endurance repository fixture is a
61-sample contiguous apogee-centered subset of the larger research return. Their
purpose is contract and semantic validation, not sample-volume accumulation.

## Representation

Every fixture preserves:

- a source-native state asset;
- a separately derived ECEF analysis asset;
- a hash-pinned episode manifest;
- processing lineage;
- evidence-bearing label assertions;
- identity-only grouping keys;
- quality findings and claim limitations.

The bridge adapter projects the ECEF analysis state into the current
`CorpusTrajectory` interface. It computes compatibility-layer derived velocity
and acceleration because the current `NormalizedTrack` contract requires them.
Those arrays are not represented as source observations.

## Merge and validation status

- lane-local fixture integrity and semantic checks are executable in repository
  tests;
- source bytes are intentionally not redistributed;
- the Product 4 common-front episode loader remains the authority for final G2
  acceptance;
- no classifier-performance claim is made.
