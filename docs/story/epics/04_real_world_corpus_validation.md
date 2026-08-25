# Real-World Corpus & Validation

Core question: do the methodology and classifier claims survive independently collected physical observations across the six active real-world corpus lanes?

This epic turns heterogeneous trajectory sources into governed, reproducible evidence for the existing study and classifier machinery. It is the observational grounding product for the sandbox, not a replacement for synthetic corpus search.

## Product boundary

Product 4 owns the path from source artifact to governed classifier-facing study view:

```text
source registry
    -> acquisition / immutable source reference
    -> source adapter
    -> source-native state
    -> normalized state view
    -> trajectory episode manifest
    -> quality / labels / grouping / lineage
    -> immutable corpus snapshot
    -> study selection policy
    -> ClassifierTrajectoryView
    -> existing Product 1 / Product 2 evaluation
```

Product 3 continues to own generated/search corpora and hard-case discovery. Product 2 continues to own evidence-builder comparison and classifier-complexity escalation. Product 4 supplies real observations and independent validation cohorts to both.

## Active corpus lanes

| Lane | Initial reference direction | Main semantic stress |
| --- | --- | --- |
| `land_surface` | TGSIM road trajectories | physical-object and recording identity, local Cartesian coordinates |
| `sea_surface` | AIS/GNSS vessel trajectories | irregular sampling, MMSI leakage, absent vertical observation |
| `sea_subsurface` | glider/AUV trajectory products | GPS resets, dead reckoning, measured depth, deployment identity |
| `air_atmospheric` | ADS-B / flight trajectories | flight segmentation, altitude semantics, source/provider fingerprints |
| `space_near` | mission reference/reconstructed trajectories | mission phase, reference-vs-observed state, event identity |
| `space_orbital` | ephemeris/elements/propagated states | frame, epoch, central body, propagation lineage, persistent object grouping |

The named sources are implementation directions, not blanket access or redistribution claims. Each source must carry verified citation, version/access identity, license or terms status, and immutable artifact/query provenance before promotion.

## Common persistent model

The root persisted object is an episode manifest, not a classifier window and not a bare normalized track.

```text
TrajectoryEpisodeManifest
    episode_id
    physical_domain
    source_dataset_id
    physical_object_id?
    mission_or_deployment_id?
    recording_or_run_id?
    time_basis
    state_views[]
    segments[]
    labels[]
    grouping_keys[]
    quality_findings[]
    processing_lineage[]
    domain_extension
```

Each `TrajectoryStateView` declares the semantics of the state it contains, for example:

- `source_native`
- `observed`
- `estimated`
- `reconstructed`
- `reference`
- `propagated`
- `simulated_truth`
- `normalized`

A state view must declare time basis, coordinate frame, units, dimensionality, vertical semantics, observation modality, and derivation lineage. A transformed or propagated state never silently becomes an observation.

`NormalizedTrack` remains a useful numeric trajectory representation inside a state view. It is not the cross-domain persistence root.

## Classifier-facing boundary

Studies construct a separate `ClassifierTrajectoryView` from one or more approved episode state views. The projection policy must explicitly select which channels are classifier-visible.

Product 4 exposes three named consumer profiles so deep source or kinematic analysis does not
become an accidental Product 2 dependency: `source_audit` may inspect all state views,
`kinematic_analysis` selects normalized analysis state only, and `classifier_ladder` selects
only identity-free classifier assets from prepared sources with labels stored out of band. The
target-label namespace must be declared by the study; a generic “some label exists” check is not
enough. The profile is recorded in a snapshot-hash-bound `AnalysisProductManifest` and has its own isolated
`product4_analysis` test surface.

The baseline classifier view must not silently include:

- source, provider, recording, mission, object, or track identifiers
- absolute coordinates when they encode geography or source identity without explicit study justification
- lane/intersection/region identifiers
- vehicle dimensions or other audit-only labels
- callsigns, MMSI, ICAO identifiers, NORAD identifiers, or equivalents
- adapter identity
- collection-platform identity

Audit and context fields remain available in metadata for leakage analysis.

## Cross-domain invariants

Every promoted adapter and prepared corpus must satisfy these invariants:

1. source dataset, artifact/query, object/mission/deployment/recording, trajectory, and split-group provenance are retained;
2. time basis and epoch conventions are explicit;
3. coordinate frame and transform lineage are explicit;
4. vertical semantics are explicit and missing vertical observation is not fabricated;
5. source-derived and independently derived kinematics remain distinguishable;
6. labels carry evidence strength, source, and proxy status;
7. physical grouping survives windowing and train/validation/test splitting;
8. domain-specific metadata can be retained without mutating the common numeric classifier contract;
9. source/sensor/context metadata are not silently inserted into kinematics-only studies;
10. citation, access date, source version, adapter version, and hashes remain machine-readable;
11. prepared snapshots are immutable and reproducible from declared inputs where redistribution permits;
12. propagated/reconstructed/reference states remain semantically distinct from observations.

## Corpus storage policy

The Git repository stores corpus definitions, contracts, adapters, tiny lawful fixtures, study manifests, snapshot manifests, and tests. Large source bytes and prepared corpora may live outside Git.

Recommended external/local shape:

```text
data/
├── raw/
│   ├── land/
│   ├── sea_surface/
│   ├── sea_subsurface/
│   ├── air/
│   ├── space_near/
│   └── space_orbital/
├── prepared/
│   └── trajectory-corpus-v0.1/
└── snapshots/
    └── <snapshot_id>/
        ├── corpus_snapshot.json
        ├── source_catalog.parquet
        ├── episode_catalog.parquet
        ├── label_catalog.parquet
        ├── quality_catalog.parquet
        └── SHA256SUMS
```

Raw redistribution is source-specific. A snapshot may therefore contain immutable acquisition instructions and hashes without redistributing source bytes.

## Validation surfaces

Product 4 must support progressively stronger evaluation surfaces:

1. within-source held-out physical objects;
2. held-out recording/run/deployment/mission;
3. held-out geography where meaningful;
4. held-out provider or collection modality;
5. independent source validation within a domain;
6. cross-domain kinematics-only studies where the class hypothesis is physically meaningful.

Raw row count is not a primary adequacy measure. Coverage reporting should prioritize independent physical objects, missions, deployments, recordings, source duration, and sampling regimes.

## Real-world evidence packet

A real-world validation run should emit enough information to audit the claim without reconstructing hidden decisions. The target packet contains:

- corpus snapshot identity and hashes;
- source/version/adapter inventory;
- lane/domain coverage;
- physical-object, mission, deployment, and recording counts;
- duration and sampling summaries;
- label-evidence and proxy summaries;
- state-role and observation-modality summaries;
- quality findings and rejected-episode counts;
- grouping policy and split assignments;
- source-fingerprint/leakage diagnostics;
- study manifest and classifier projection policy;
- downstream Product 1 / Product 2 result references;
- claim boundary and promotion decision.

Decision language:

- `real_world_evidence_supported`
- `supported_with_limits`
- `independent_validation_required`
- `revise_source_portfolio`
- `revise_grouping_policy`
- `revise_label_claim`
- `insufficient_real_world_evidence`
- `reject_invalid_source`

## Relationship to current LAND implementation

The `corpus.real_world` LAND/TGSIM work is Product 4 milestone zero and the reference vertical slice. Preserve its tested capabilities:

- immutable normalized tracks;
- source metadata and provenance;
- quality findings;
- gap-safe windowing;
- leakage-safe split grouping;
- explicit classifier projections;
- study manifests and artifact I/O;
- tiny repository fixtures and regression tests.

Refactor only where needed to make LAND conform to the episode/state-view common front. Do not duplicate the TGSIM trajectory representation or replace working source-specific parsing with a generic parser.

## Implementation backlog

### P4-001 — Product declaration and claim boundary

Status: **in progress in this branch**.

- add Product 4 to the canonical product map;
- define ownership relative to Products 1-3;
- add this epic and reading-order references;
- preserve presentation/showcase as export profiles, not another product.

Acceptance: a new reader can identify the four products, their questions, and the Product 3/Product 4 boundary from canonical docs.

### P4-002 — Episode/state-view common contract

- introduce `TrajectoryEpisodeManifest`;
- introduce `TrajectoryStateView` and state-role enum;
- attach existing `NormalizedTrack` to a state-view role instead of treating it as the corpus root;
- model label assertions, grouping keys, quality findings, and processing lineage as explicit collections;
- preserve current TGSIM compatibility.

Acceptance: the existing TGSIM fixture can be represented without data loss through the new aggregate and all current LAND tests remain valid or are intentionally migrated.

### P4-003 — Corpus catalog and source registry

- define machine-readable source cards;
- record access URL/query identity, citation, license/terms, accessed date, source version, native schema, frame/time conventions, adapter version, and hashes;
- distinguish `lead_only`, `access_verified`, `artifact_acquired`, `schema_inspected`, `mapping_complete`, `fixture_validated`, `prepared`, and `released` lifecycle states.

Acceptance: catalog queries can enumerate source readiness and provenance without opening source-specific code.

### P4-004 — Immutable prepared snapshots

- define `CorpusSnapshotManifest`;
- emit source, episode, label, grouping, quality, and lineage flat catalogs;
- hash all prepared assets and manifest inputs;
- prohibit in-place mutation of released snapshot IDs;
- record reproducibility limitations when raw redistribution is not permitted.

Acceptance: two builds from the same pinned inputs and adapter versions produce equivalent manifest identities and content hashes where deterministic processing applies.

### P4-005 — Common loader and study selection API

- load episodes from one snapshot through one API;
- select by domain/lane/source/state role/label evidence/quality policy;
- construct `ClassifierTrajectoryView` under an explicit projection policy;
- reuse existing windowing/splitting machinery where valid;
- keep physical-object grouping intact across all generated windows.

Acceptance: a study configuration can switch between approved real-world sources without classifier code importing a source adapter.

### P4-006 — Cross-domain leakage and source-fingerprint audit

- audit split-group overlap;
- audit overlapping temporal windows across partitions;
- audit object/mission/deployment identity reuse;
- measure whether source/provider/modality/geography is trivially predictable from intended classifier features;
- flag label dependencies on fields that are unavailable at inference time.

Acceptance: a deliberately contaminated fixture is rejected and a clean multi-source fixture produces an inspectable leakage report.

### P4-007 — LAND reference migration

- wrap TGSIM output as episode manifests/state views;
- preserve the existing passenger-car versus truck study;
- retain source-specific diagnostic metadata outside baseline classifier features;
- emit Product 4 snapshot/catalog artifacts for the bounded fixture and full local run.

Acceptance: the current TGSIM workflow still runs, now through the Product 4 common loader path.

### P4-008 — SEA surface adapter

- implement one bounded AIS/GNSS source adapter;
- preserve vessel/recording identity and observation gaps;
- make absent vertical observation explicit;
- prevent MMSI/source identifiers from entering baseline classifier features;
- produce one real common-contract fixture.

Acceptance: fixture passes P4-002 through P4-006 and can produce a classifier view through the common API.

### P4-009 — AIR atmospheric adapter

- implement one bounded ADS-B/flight source adapter;
- define geometric/barometric/source-native altitude semantics;
- define flight-leg grouping and discontinuity handling;
- preserve provider/source-quality metadata for audit only by default.

Acceptance: same as P4-008, with explicit altitude and flight-group tests.

### P4-010 — SEA subsurface adapter

- implement one glider/AUV deployment source;
- preserve GPS reset, dead-reckoned, fused/estimated, and measured-depth semantics;
- group by physical deployment rather than arbitrary file/window identity;
- retain navigation-mode transitions in lineage/context.

Acceptance: no inferred underwater state is mislabeled as direct observation and deployment-safe splits are demonstrated.

### P4-011 — SPACE orbital adapter

- implement one ephemeris/elements source;
- require central body, reference frame, epoch/time scale, and state role;
- preserve propagation configuration and element-to-state derivation lineage;
- group repeated epochs/windows by persistent physical object.

Acceptance: propagated Cartesian samples remain `propagated`, never `observed`, and object-safe splitting is enforced.

### P4-012 — SPACE near adapter

- implement one bounded reference/reconstructed mission trajectory;
- model mission/event identity and phase-aware segments;
- preserve observed/reference/reconstructed distinctions;
- define any frame transformations explicitly.

Acceptance: mission phase and state role survive normalization and classifier projection cannot silently consume mission identity.

### P4-013 — Six-lane fixture convergence gate

- maintain at least one tiny real or legally redistributable fixture per lane, or an immutable fixture-generation query where bytes cannot be distributed;
- run all fixtures through the same common loader and contract validation;
- generate a combined six-lane catalog and coverage report.

Acceptance: 6/6 lane fixtures compile and validate through one Product 4 path.

### P4-014 — Independent validation protocol

- define anchor versus independent-validation source roles;
- require held-out physical-object evaluation for every promoted study;
- add provider/geography/source-shift validation where available;
- distinguish exploratory results from independent validation claims.

Acceptance: real-world evidence packets cannot emit `real_world_evidence_supported` from a single non-independent train/test source slice unless the claim is explicitly scoped to that source.

### P4-015 — Real-world baseline release

- freeze a versioned six-lane snapshot or portfolio manifest;
- report source/lane balance by independent objects/missions/deployments/recordings and duration;
- run common leakage/quality gates;
- publish reproducibility and claim-boundary documentation;
- expose Product 4 snapshot selection to Product 1 and Product 2 study configs.

Acceptance: a single declared snapshot/portfolio can recreate the study inputs for representative lanes without source-specific classifier code.

## Delivery gates

| Gate | Meaning | Minimum evidence |
| --- | --- | --- |
| G0 | contract agreed | P4-001/P4-002 docs and tests |
| G1 | LAND common-front migration | TGSIM through episode/state-view path |
| G2 | multi-source convergence | at least LAND + one independent physical-domain adapter |
| G3 | six-lane fixture convergence | 6/6 fixtures through common loader |
| G4 | study readiness | snapshot + selection + leakage + classifier projection |
| G5 | independent validation | at least one independent source/provider validation study |
| G6 | real-world baseline release | versioned governed portfolio with explicit claim boundary |

## Deferred work

Do not create separate UAS, military-submarine, tracked-vehicle, cislunar, environmental-context, sensor-observation, synthetic-data, or video-digitization products yet. Those may become new sources or domain extensions when the common front proves insufficient.

A dedicated Corpus QA/Leakage agent becomes justified after at least two independent real source fixtures have passed the common contract and there is enough multi-source material to audit source fingerprints meaningfully.

## Claim boundary

Product 4 proves reproducible ingestion, governed trajectory semantics, leakage-safe grouping, explicit classifier projection, and increasingly independent real-world evaluation. It does not by itself prove that any class pair is physically identifiable, that a classifier generalizes to deployment, or that six heterogeneous domains are interchangeable. Those stronger claims require explicit Product 1/Product 2 studies and independent evidence packets.
