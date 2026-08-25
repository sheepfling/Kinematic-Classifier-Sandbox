# Product 4 SPACE-ORB Wave 1

This directory records the first `space_orbital` source-research tranche for Product 4 —
Real-World Corpus & Validation.

## Status and claim boundary

The tranche is complete through `fixture_validated`. It is not a production pilot and does not
claim cross-domain classifier performance, orbit-regime classification, SGP4 validation, or a
population-level orbital corpus.

The admitted real fixture is a bounded NASA/JSC/FOD/TOPO CCSDS OEM arc for the International Space
Station. The source declares `EME2000`, `UTC`, Earth center, source position, and source velocity.
The adapter preserves those declarations, applies only SI unit conversion, and keeps source velocity
separate from finite-difference velocity.

The bounded fixture is explicitly marked `validation_fixture_only` and
`classifier_eligible=false`. It proves ingestion, provenance, frame/time semantics, physical-object
grouping, and the Product 4 contract. It is not admitted into a classifier study merely because it
passes the schema.

The preferred anchor remains the IGS Final SP3 product family. This PR adds a source-faithful SP3-c
position parser and machine-readable source evidence, but it intentionally does not commit the
third-party mirror bytes used during lane-side validation. Promotion of an IGS release fixture
remains blocked on acquisition and comparison of the exact official compressed artifact.

## Source portfolio

| Role | Source | Physical object | State role | Repository disposition |
| --- | --- | --- | --- | --- |
| Anchor candidate | IGS Final SP3 | epoch-resolved GNSS objects | reference solution | parser and metadata only |
| Independent validation | NASA ISS OEM | `1998-067A` | operational estimate | bounded validation-only fixture |
| Propagation stress | CelesTrak OMM | `1998-067A` for the inspected record | propagated estimate after SGP4 | deferred |
| Historical regression | CODE Repro1 SP3 | `1992-079A` for the inspected G01 epoch | fitted reference solution | internal only |

The NASA and CelesTrak ISS products are not independent because they represent the same lifetime
physical object. Provider and product differences do not create a new split group.

## Source-asset provenance

The manifest separates two different artifacts rather than pairing one artifact's hash with another
artifact's URL:

1. The required source asset is the exact committed 13-state fixture, identified by a repository URI
   and its SHA-256.
2. The optional source asset is the immutable full OEM product from which the bounded fixture was
   extracted. Its URL is retained as upstream provenance and is not assigned the bounded fixture's
   checksum.

This distinction makes a future downloader or verifier unambiguous about which bytes each checksum
covers.

## Parser scope

The OEM parser intentionally accepts one state-only CCSDS OEM 2.0 KVN metadata segment with
`TIME_SYSTEM=UTC`. It rejects multiple metadata segments, non-UTC time systems, invalid metadata
time ordering, non-monotonic state epochs, and states outside the declared useable interval. Support
for additional segments, covariance blocks, or other time systems requires an explicit extension
rather than silent reinterpretation.

The SP3 parser intentionally accepts complete regular-cadence SP3-c position products. It validates
the declared epoch count, first epoch, cadence, satellite-record uniqueness, position sentinels, and
header semantics. Conversion to GPS seconds is permitted only when the SP3 header explicitly
declares `GPS`; other time systems remain unconverted.

## Common-contract mapping

The NASA fixture maps to the existing Product 4 contract as follows:

- `PhysicalDomain.SPACE`
- `TimeBasis.UNIX_UTC_SECONDS`
- `CoordinateFrameKind.ECI` with frame ID `EME2000`
- source position in meters
- source velocity in meters per second
- independently derived velocity and acceleration retained separately
- split group `physical_object:1998-067A`
- `persistent_orbit` as a lane-normalized scope label with `derived` evidence
- source identity, provider, product creation time, and product family retained as grouping/audit
  metadata rather than numeric classifier channels

The OEM product remains an `estimate` with `operational_prediction` value basis. It is not promoted
to measurement truth.

## Identity and leakage policy

SPACE-ORB splitting begins with lifetime physical identity, not source product rows, orbital epochs,
provider names, or PRNs. The required hierarchy is:

```text
physical object
  -> mission or launch
    -> catalog identity
      -> source product family
        -> product or epoch family
          -> propagation configuration
            -> episode
              -> segment
                -> window
```

For GNSS SP3 products, a PRN must be resolved through epoch-valid satellite metadata before a split
group is assigned. For example, the inspected IGS Final G01 arc on 2024-003 resolves to G063 /
`2011-036A`, while the inspected historical CODE G01 arc resolves to a different physical object,
`1992-079A`.

## Included implementation

- `space_orbital_oem_parsing.py`: bounded single-segment UTC CCSDS OEM parsing primitives.
- `space_orbital_oem.py`: exact NASA ISS fixture manifest and Product 4 adapter.
- `space_orbital_sp3.py`: strict SP3-c position parser with explicit frame/time/clock semantics and
  no invented source velocity.
- `nasa_iss_oem_20220427_excerpt.kvn`: exact 13-state bounded public-domain fixture.
- focused tests for source roles, provenance alignment, parsing, SI conversion, Product 4 contract
  construction, source/derived kinematics separation, hash enforcement, physical-object grouping,
  time-system guards, interval guards, cadence guards, and error handling.

## Remaining restrictions

- Acquire and hash the exact official IGS Final compressed artifact.
- Compare the official decompressed content with the inspected immutable mirror before admitting a
  release fixture.
- Keep CelesTrak OMM and historical mirror material out of the release corpus until rights and
  propagation-lineage reviews are complete.
- Require COMMON-FRONT countersignature before advancing this lane from `fixture_validated` to
  `pilot_prepared`.

## Evidence file

`evidence.yaml` contains the NASA OEM and IGS SP3 source cards, the NASA adapter mapping, and the
independent-validation matrix used for coordinator review.
