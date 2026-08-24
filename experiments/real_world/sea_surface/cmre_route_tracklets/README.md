# Product 4 SEA-Surface — CMRE Route-Tracklet Fixture

This experiment declaration records the first validated `sea_surface` fixture for Product 4.
It uses real AIS-derived five-contact route tracklets from the CMRE/Brest route-association
corpus to exercise the common episode/state-view front.

The source is an **expansion/regression witness**, not the preferred platform-family anchor.
It supports route-label mapping, missing-vertical semantics, repeated-platform grouping,
source-quality findings, and classifier-view leakage checks. It does not support vessel-family
classification, complete-voyage reconstruction, or population-representative maritime claims.

## Redistribution boundary

The repository does not include the real source rows or derived state assets. The accompanying
data article identifies CC BY-NC-SA 4.0 terms, while available catalog metadata was not fully
consistent during the research pass. Until the authoritative record is reconciled, treat the
source as attribution-required, non-commercial, and share-alike.

`bounded_extract_spec.yaml` pins the upstream commit/blob and the twelve source tracklet IDs used
for the validated external fixture. `fixture_receipt.json` records the external packet checksum,
validation counts, and claim boundary without redistributing source coordinates or MMSI values.

## Build from a locally acquired artifact

Create a private identity-grouping key outside the repository and retain it with the corpus build
configuration. The adapter requires at least 16 bytes and never writes the key into prepared
assets or manifests.

```bash
python - <<'PY'
from pathlib import Path
import secrets

Path("/secure/path/cmre-identity.key").write_bytes(secrets.token_bytes(32))
PY

PYTHONPATH=src python3 scripts/run/build_cmre_route_tracklet_fixture.py \
  /path/to/tracklets.csv \
  /path/to/nomen.csv \
  artifacts/real_world/sea_surface/cmre_route_tracklets \
  --source-artifact-id cmre-tracklets-local \
  --corpus-snapshot-id sea-surface-cmre-local \
  --identity-key-file /secure/path/cmre-identity.key \
  --tracklet-id 1 \
  --tracklet-id 2
```

The builder preserves source-native timestamps and reported AIS channels, writes a named local-ENU
analysis view, masks the unavailable vertical component, derives horizontal velocity from reported
SOG/COG with explicit lineage, and creates a classifier-facing view that stable-sorts timestamps,
retains the first source occurrence at duplicate times, carries channel-validity masks, and excludes
raw identity, grouping values, and route targets. Physical-platform groups use keyed HMAC-SHA256;
a public hash of the low-entropy MMSI space is not used.

## Evidence state

- CMRE/Brest route tracklets: `fixture_validated` as an expansion/regression source.
- NOAA 2024 AIS: `schema_inspected`; preferred anchor still needs an official repeated-observation
  artifact and fixture.
- Kystverket historical AIS: `access_verified`; preferred independent-provider holdout still needs
  a bounded acquisition and fixture.
