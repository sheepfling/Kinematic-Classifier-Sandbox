# Acquisition Plan — LAND Wave 1

## 1. TGSIM anchor: first priority

The official Foggy Bottom JSON and CSV distributions are verified. The complete JSON response is
too large for the current web-response surface, so the next acquisition must be a **bounded
complete-track query**, not a whole-response fetch and not arbitrary first-N rows.

Required retained evidence:

1. exact query URL/body;
2. retrieval timestamp;
3. Data.gov modified/version metadata;
4. exact bytes;
5. SHA-256 and byte size;
6. source schema metadata;
7. complete row set for each selected native track;
8. reference image and boundaries needed to interpret region and frame semantics.

Selection rule: identify candidate track IDs first, then retrieve every row for those IDs. Do not
construct a fixture from a truncated trajectory.

## 2. pNEUMA: independent road validation

The Zenodo archive is decomposed enough to choose a bounded file intentionally. A preferred
starting member is a small per-drone half-hour file such as `20181024_d1_0830_0900.csv`, reported
as 85.6 MB in the official archive preview.

Before retaining or redistributing a fixture, resolve the official license-text discrepancy: the
download page displays CC BY-NC 4.0 while its FAQ says CC BY 4.0.

After rights disposition:

1. retain one complete CSV member;
2. verify it against the Zenodo record/version and local SHA-256;
3. inspect semicolon/repeated-column structure from the actual file;
4. preserve file/date/drone/session identity for grouping;
5. create a tiny complete-track fixture from selected source rows.

## 3. highD: secondary validation

Do not spend Wave 1 acquisition effort on highD unless TGSIM and pNEUMA are blocked. Manual
approval and redistribution restrictions make it less suitable for the portable common fixture.

## 4. ReV-StED: local-only contract witness

The immutable ADMA sample has already been inspected locally. Source rows and derived coordinate
arrays remain outside the repository until a concrete redistribution license is established.
No further acquisition is required for the current local structural-validation claim.

## 5. Amazon Precision GNSS: state-quality expansion

Retrieve and hash `trajectories.zip`, inspect all paired low-cost/reference CSV streams, and select
one complete drive for estimate/reference state-view validation. This source must not be promoted
as vehicle-class evidence.
