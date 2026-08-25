# AIR Atmospheric Source Groundwork

This tranche establishes the first AIR source parser and mapping evidence for Product 4 —
Real-World Corpus & Validation. It is deliberately narrower than a prepared AIR pilot.

## Scope

The implementation parses the documented `readsb` historical trace format used by the proposed
ADSB.lol anchor source. It preserves source ordering and AIR-specific semantics needed by the
common real-world corpus front:

- Unix root timestamp plus per-sample offsets;
- latitude and longitude;
- primary altitude with explicit barometric/geometric basis;
- separate geometric altitude;
- ground speed, track, and vertical rate;
- primary vertical-rate basis;
- stale-position and source-derived new-leg flags;
- per-sample source type;
- optional registration, type code, description, and database flags.

The parser also exposes source-derived flight-leg candidates without converting those candidates
into verified mission truth.

## Source portfolio

| Role | Source | Current evidence state | Claim boundary |
| --- | --- | --- | --- |
| Anchor | ADSB.lol historical `readsb` traces | `access_verified` | Network surveillance observations; not physical truth |
| Independent validation | OpenSky scientific state vectors | access verified, rights conditional | Separate network/provider validation only after rights disposition |
| Enrichment | FAA releasable aircraft/reference data | `access_verified` | Technical metadata for defensible identity-linked matches; not trajectory state |

The machine-readable portfolio is in `source_portfolio.yaml`.

## Native-to-common mapping

The mapping in `adsblol_readsb_mapping.yaml` keeps barometric and geometric vertical semantics
separate. It does not apply a first-non-null altitude fallback. The intended analysis view may
convert geodetic state to ECEF or local ENU only through an explicit, versioned transformation.
Acceleration is intentionally outside this tranche.

## Grouping and leakage boundary

AIR requires at least these grouping concepts:

- `physical_platform`: restricted/pseudonymous aircraft identity;
- `source_recording`: aircraft by source collection day;
- `mission_event`: source-derived flight leg;
- `source_dataset`: ADSB.lol/readsb source identity;
- `temporal_collection`: archive date or collection window;
- `route`: optional, only where defensibly established.

ICAO identifiers, registration, callsign, provider identity, source filenames, routes, and type
metadata are never ordinary kinematic classifier features. Classifier-visible episode identifiers
must be opaque and must not encode aircraft identity or type.

## Included fixture

`tests/corpus/real_world/fixtures/readsb_documented_a320_trace.json` is the seven-sample example
published in the authoritative `readsb` JSON-format documentation. It validates parser mechanics
and native semantics. It is **not** an ADSB.lol historical-corpus fixture and does not close AIR G2.

## Validation in this PR

The focused tests cover:

- documented multi-sample trace parsing;
- absolute sample-time construction;
- barometric/geometric altitude separation;
- barometric/geometric vertical-rate separation;
- ground sentinel handling;
- stale-position and new-leg flags;
- source-derived leg extraction;
- duplicate and out-of-order timestamps as findings;
- plain and gzip JSON loading;
- rejection of invalid coordinates without silent clamping.

## Evidence state and remaining G2 work

This PR supports `schema_inspected` and parser/mapping groundwork, but it does not claim
`fixture_validated`.

Remaining work:

- [ ] Download and checksum one pinned ADSB.lol historical release asset.
- [ ] Extract a genuine `trace_full_<icao>.json.gz` source artifact.
- [ ] Compare its observed schema against this parser.
- [ ] Select one sufficiently populated source-derived flight leg.
- [ ] Materialize source-native and analysis state views under `trajectory-corpus-v0.1`.
- [ ] Validate grouping, label evidence, quality findings, and classifier-view restrictions.
- [ ] Resolve and run an independent validation cohort.

Until those steps pass, AIR G2 remains open.

## Sources

- ADSB.lol historical data: <https://www.adsb.lol/docs/open-data/historical/>
- ADSB.lol release repository: <https://github.com/adsblol/globe_history_2026>
- `readsb` JSON format: <https://github.com/wiedehopf/readsb/blob/dev/README-json.md>
- OpenSky scientific data: <https://opensky-network.org/data/scientific>
- FAA releasable aircraft data: <https://www.faa.gov/licenses_certificates/aircraft_certification/aircraft_registry/releasable_aircraft_download>
