# Validation

## Local ReV-StED witness

```text
local structural validator: PASS
source records inspected locally: 6
state views: 2
duration: 0.05 s
state interval: 0.01 s
grouping keys: 3
quality findings: 5
repository fixture bytes committed: no
authoritative COMMON-FRONT validation: not run
```

The local validator checked exact retained-asset hashes, increasing state time, separation of
state and transport clocks, invalid vertical components, source-native/analysis lineage, opaque
grouping values, classifier-view isolation, and quality/rights findings.

Position-derived versus source-frame horizontal velocity RMSE for the bounded local slice was
`0.433662 m/s`. This is an informational consistency result for the tiny witness, not a general
sensor-accuracy claim.

## Repository intake

`land-wave1-repository-intake-v0.2` passes against the rights-safe repository tranche. It checks
that YAML and JSON parse, source-card and scorecard identifiers are complete and unique, every
registry entry is a mapping with a non-empty unique source identifier, and the three portfolio
sets agree exactly. It also enforces that both raw and derived highD redistribution remain
disabled.

Negative mutation checks were run locally and correctly rejected:

- a duplicate registry source identifier;
- a malformed registry entry without `source_dataset_id`;
- a permissive highD derived-data redistribution flag.

The validator additionally checks that Product 4 G2 remains false, no restricted fixture assets
are committed, no raw data containers are present, and the tranche remains within a small
text-only research boundary.

## Evidence state

```text
TGSIM: access_verified
pNEUMA: access_verified / rights conditional
highD: access_verified / raw and derived redistribution disabled
ReV-StED: schema_inspected / local validation only
Amazon Precision GNSS: access_verified
LAND overall: access_verified
Product 4 LAND G2: not closed
```
