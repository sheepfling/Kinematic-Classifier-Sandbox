# Real-World Trajectory Ingestion

This package preserves source trajectories, labels, provenance, quality findings, and leakage-safe
study slices before a study projects them into the existing classifier contracts. It is additive:
synthetic witnesses and one-dimensional experiment adapters remain unchanged.

## TGSIM Foggy Bottom

`TgsimFoggyBottomAdapter` reads the published Foggy Bottom trajectory CSV and produces immutable
`NormalizedTrack` objects. The adapter expects the verified source fields for identity, time,
position, lane/region, velocity, acceleration, dimensions, and road-user type. Header punctuation
is normalized, so the report-style hyphenated names and equivalent underscore names are accepted.

```python
from datetime import date

from kinematic_classifier_sandbox.corpus.real_world.adapters.tgsim import (
    TgsimFoggyBottomAdapter,
)
from kinematic_classifier_sandbox.corpus.real_world.adapters.tgsim_contracts import (
    TgsimFoggyBottomAdapterConfig,
)

adapter = TgsimFoggyBottomAdapter(
    config=TgsimFoggyBottomAdapterConfig(accessed_on=date(2026, 8, 22))
)
result = adapter.load_file("TGSIM-Foggy Bottom-Data.csv")

for track in result.tracks:
    print(track.provenance.split_group_id, track.labels.normalized_class)
```

The repository does not redistribute the raw dataset. The manifest records the official landing
page, download endpoint, citation, DOI, access date, coordinate-frame description, and license.

## AIR atmospheric source groundwork

`adapters.adsblol.readsb_trace` parses plain or gzip-compressed historical `readsb` trace JSON.
It preserves source sample order, explicit time offsets, ground-state sentinels, stale-position and
source-derived new-leg flags, source type, and separate barometric/geometric altitude and vertical-
rate semantics.

This is parser and mapping groundwork for the proposed ADSB.lol Product 4 AIR anchor. It does not
yet claim a prepared AIR pilot or a validated historical flight-leg fixture. The source portfolio,
native mapping, grouping boundary, and remaining G2 work are documented in
`docs/product4/air_atmospheric/`.

## Leakage-safe windowing and splits

`windowing.py` breaks tracks at timing discontinuities before creating deterministic half-open
windows. A window never crosses a detected gap and always retains the parent `split_group_id`.

`splits.py` assigns whole split groups to train, validation, or test partitions. Assignment is
deterministic from a declared seed and is stratified by normalized class by default. When a class
has enough groups to populate every nonzero partition, the allocator preserves at least one group
per partition.

The default road-vehicle study evaluates 10-, 20-, and 30-second windows with 50 percent overlap.
Every duration uses the same group assignment, so duration sensitivity cannot reshuffle vehicles
between partitions.

## Existing-contract projections

`projection.py` supports two explicit one-dimensional projections into `ExecutableTrajectory`:

- `speed_profile`: derived speed over the track's declared 2D or 3D speed axes
- `cumulative_path_length`: distance traveled from the beginning of the window in meters

Projection metadata remains separate and carries dataset, recording, run, source track, window,
split-group, native-label, normalized-class, and partition identity. No provenance identifier is
inserted into the classifier measurement sequence.

## First road-vehicle study

`build_road_vehicle_study()` prepares the initial `passenger_car` versus `truck` study and emits
both projection families for every accepted window. Its partition summaries include class-level
track counts, source duration, accepted windows, and rejected-window counts.

The full TGSIM workflow can be run with:

```bash
PYTHONPATH=src python3 scripts/run/run_tgsim_road_vehicle_study.py \
  "TGSIM-Foggy Bottom-Data.csv" \
  artifacts/real_world/tgsim_car_vs_truck
```

The generated evidence packet contains:

- `study_manifest.json`
- `tracks.csv`
- `report.md`
- one directory per window duration
- `split_assignments.csv`
- `partition_summary.csv`
- `windows.csv`
- `projection_metadata.csv`

## Leakage boundary

The baseline motion lane may use position-derived and source-provided kinematic channels. The
following fields are retained for diagnostics but are not baseline classifier features:

- smoothed vehicle length and width (`audit_only`)
- lane or intersection-region identifier (`context`)
- absolute source coordinates
- source, run, and track identifiers

Every source track receives exactly one split-group assignment. Window and projection metadata
retain that assignment, but classifier measurements do not.

## Common snapshot front

`snapshot_builder.py` is the common handoff from domain adapters to Product 4 evaluation. It reads
episode manifests from an external snapshot root, verifies that each episode belongs to the
declared registry source and lane, pins the manifest file hash, records source artifact IDs and
adapter versions, and optionally requires every selected source to be `prepared`. It does not
promote `fixture_validated` evidence or copy restricted source bytes into Git.

The validation-only common-front builders are intentionally separated by lane:

- `land.common_front`: TGSIM source/analysis views with no classifier asset;
- `sea_subsurface.common_front`: IOOS measured, dead-reckoned, pressure, and depth channels with
  asynchronous events preserved;
- `air.common_front`: readsb source and normalized altitude/vertical-rate views with per-sample
  barometric/geometric basis retained;
- `space_near.common_front`: bounded mission fixtures with reference/analysis lineage;
- `space_orbital.common_front`: NASA OEM EME2000 source velocity kept distinct from derived
  velocity; and
- the existing SEA-SURF adapter: route-tracklet source/analysis/classifier-candidate assets with
  keyed platform grouping.

`portfolio.assign_grouped_snapshot_splits()` unions shared physical-platform, source-recording,
and mission-event keys before proposing deterministic partitions. It is a split proposal only;
`audit_split_assignments()` remains the release gate.

`portfolio_matrix.py` evaluates the same gates lane by lane against a single immutable snapshot.
It is the prioritization/reporting surface: lane blockers stay visible, while `all_lanes_pass` is
reserved for the full cross-domain decision.

## Analysis-product boundaries

`analysis_products.py` makes the consumer boundary explicit instead of relying on callers to
remember which assets are appropriate:

- `source_audit` may inspect every state view and provenance/quality context, but never selects a
  classifier asset;
- `kinematic_analysis` selects only normalized `analysis` state assets for domain-aware motion
  analysis, without source-native audit assets; and
- `classifier_ladder` selects only identity-free classifier assets, requires prepared sources, and
  keeps an explicitly declared target-label namespace outside the feature asset.

The resulting `AnalysisProductManifest` is snapshot-hash-bound and records selected asset
references without copying raw data. Product 2 bridge work must consume the third profile; a
source or kinematic analysis result is not evidence of classifier readiness.

The prepared-cohort builder and Product 2 bridge keep this distinction operational. A task-scoped
cohort writes labels and grouping metadata to episode manifests while classifier assets contain only
timestamps and projected measurements. The bridge requires an explicit grouping policy, caps
correlated windows per physical platform, and builds references and measurement scale from the
training split only. This supports a bounded task claim; it does not promote the six-lane portfolio
or establish source-shift generalization.

## Current claim boundary

This package now validates and normalizes real observations, creates gap-safe windows, assigns
whole-track partitions, and prepares road-vehicle projections and evidence manifests. It does not
yet establish that passenger cars and trucks are kinematically identifiable or report classifier
accuracy on the complete 350 MB Foggy Bottom dataset. Those claims require a full-data run and the
existing classifier evaluation ladder.
