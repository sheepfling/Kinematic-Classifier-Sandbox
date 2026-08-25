# Common Real-World Corpus Contract

The real-world corpus is organized around one trajectory contract for the four major physical
domains: land, sea, air, and space. Domain adapters normalize source data into the same numeric
trajectory representation and attach metadata that preserves what the source actually measured.

## Common boundary

Every corpus item has three layers:

1. `CorpusDatasetMetadata` describes the source dataset, declared physical domains, observation
   modalities, time basis, native coordinate description, canonical coordinate frame, source type,
   citation, license, and adapter version.
2. `CorpusTrajectoryMetadata` describes one trajectory's physical domain, observation modalities,
   platform/subject identity when known, coordinate frame, source metadata, and domain-specific
   extension metadata.
3. `NormalizedTrack` contains the numeric trajectory: monotonically increasing timestamps,
   canonical Cartesian position, derived velocity and acceleration, optional source velocity and
   acceleration, arbitrary named numeric/categorical channels, labels, provenance, and quality.

`CorpusTrajectory` binds those layers and validates that dataset identity, domain, time basis,
coordinate frame, and observation modalities agree.

## Canonical numeric rule

The cross-domain classifier boundary uses metric Cartesian trajectories. An adapter may ingest
latitude/longitude/altitude, AIS positions, ADS-B states, orbital ephemerides, radar tracks, image
tracks, GNSS, odometry, or another native representation, but it must either convert position to a
declared Cartesian frame or explicitly use a source-defined Cartesian frame.

The original representation is not discarded. Native coordinate-system information belongs in
dataset/source metadata, and source-specific quantities remain named channels or extensions.

Examples of canonical frames include:

- land: local Cartesian, ENU, NED, or ECEF
- sea: ENU/ECEF for surface vessels; NED/ECEF for subsurface vehicles
- air: ENU/NED/ECEF for local or global flight tracks
- space: ECI for inertial orbital analysis or ECEF for Earth-fixed applications

The contract does not force every dataset into one global frame. It requires the frame to be
explicit and internally consistent so a study can deliberately transform or compare trajectories.

## Time rule

Timestamps are always numeric seconds, with a declared `TimeBasis`:

- `relative_seconds`
- `unix_utc_seconds`
- `gps_seconds`
- `source_native_seconds`

Adapters must not silently reinterpret epochs. If a source time convention cannot be safely
converted, preserve it as source-native seconds and record the convention in metadata.

## Observation modalities

Observation modality is metadata, not a classifier label. Initial common values include GNSS,
ADS-B, AIS, radar, optical tracking, inertial, odometry, telemetry, and multi-sensor fusion.
Keeping modality explicit lets evaluation distinguish genuine kinematic generalization from a model
that has learned source/sensor artifacts.

## Domain-specific extensions

The common contract intentionally does not place every possible field in the core schema.
Domain-specific values live in named channels or `domain_extensions` while the common kinematic
boundary remains stable.

Typical extensions include:

| Domain | Example extensions |
| --- | --- |
| Land | steering mode, wheel RPM, terrain/surface, grade, suspension, tracked/skid-steer marker |
| Sea | course over ground, speed over ground, draft, depth, current/wind context, vessel class |
| Air | pressure/geometric altitude, climb rate, bank/pitch, flight phase, squawk/source quality |
| Space | reference epoch, central body, state source, orbital regime, covariance, osculating elements |

Orbital elements are useful space metadata/features but do not replace the canonical Cartesian
state. This keeps space compatible with the same derivative, windowing, provenance, and evidence
machinery used by the other domains.

## Adapter interface

New domain sources should implement `RealWorldCorpusAdapter`:

```python
class RealWorldCorpusAdapter(Protocol):
    adapter_id: str
    adapter_version: str

    @property
    def corpus_metadata(self) -> CorpusDatasetMetadata: ...

    def load_corpus(self, path: str | Path) -> tuple[CorpusTrajectory, ...]: ...
```

Adapters may expose richer source-specific load results in addition to this interface. The common
interface is the interoperability boundary consumed by corpus inventory, windowing, splitting,
feature extraction, and cross-domain studies.

## TGSIM compatibility

The existing TGSIM parser remains available as `TgsimFoggyBottomAdapter` for source-specific
inspection. `TgsimFoggyBottomCorpusAdapter` wraps the same normalized tracks in the common corpus
interface and declares:

- domain: `land`
- modality: `optical_tracking`
- time basis: `relative_seconds`
- canonical frame: source-defined local Cartesian meters
- source type: overhead-video trajectory extraction

No duplicate trajectory representation is introduced.

## Cross-domain invariants

A sea, land, air, or space adapter is acceptable only when it preserves these invariants:

- source dataset, asset, recording/run, track, and split-group provenance are retained;
- labels state their evidence strength and proxy status;
- coordinate frame and time basis are explicit;
- source-derived and independently derived kinematics remain distinguishable;
- source/sensor/context metadata are not silently inserted into kinematics-only features;
- parent track groups remain intact through train/validation/test splitting;
- domain-specific information can be retained without changing the common trajectory schema;
- license, citation, version, and adapter identity remain machine-readable.

This makes land the first populated corpus domain, not a special-case architecture.

## Portfolio, snapshots, and evaluation

The machine-readable portfolio lives at
`docs/product4/real_world_source_registry.yaml`. It is the authority for source identity,
adapter version, evidence lifecycle, artifact/query provenance, grouping namespaces, claim
boundaries, and promotion blockers across the six Product 4 lanes. Source cards and lane-local
research files remain detailed evidence; the portfolio is the common cross-domain index.

The portfolio API in `portfolio.py` supplies four governed operations:

- `load_source_registry()` and `evaluate_source_registry()` report lane coverage and the best
  evidence state without treating fixture validation as classifier readiness;
- `write_snapshot_manifest()`, `load_snapshot_manifest()`, and `load_snapshot_episodes()` define
  an immutable snapshot boundary and verify every referenced episode asset hash on load;
- `select_snapshot_episodes()` applies lane, domain, source, and state-role policy before study
  projection;
- `evaluate_snapshot()` reports source, state-role, modality, quality, label-evidence, grouping,
  proxy-label, and classifier-view counts while rejecting reference/episode mismatches;
- `audit_split_assignments()` rejects physical-platform, source-recording, and mission-event
  groups that cross train/validation/test partitions.

The promotion boundary is intentionally two-dimensional:

```text
source evidence lifecycle: lead -> access -> schema -> mapping -> fixture -> prepared -> released
study eligibility:         metadata -> contract fixture -> snapshot -> leakage audit -> classifier view
```

An entry at `fixture_validated` proves ingestion and semantics only. A source becomes eligible for
a classifier-facing snapshot only after its episode manifests, asset hashes, grouping assignments,
quality findings, and leakage-safe classifier view pass the common snapshot gates. A prepared
snapshot is immutable: add new data under a new snapshot ID instead of mutating a released one.
