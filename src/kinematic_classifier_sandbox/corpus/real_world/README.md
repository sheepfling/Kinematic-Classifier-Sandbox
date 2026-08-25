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

- `speed_profile`: derived planar speed in meters per second
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

## Current claim boundary

This package now validates and normalizes real observations, creates gap-safe windows, assigns
whole-track partitions, and prepares road-vehicle projections and evidence manifests. It does not
yet establish that passenger cars and trucks are kinematically identifiable or report classifier
accuracy on the complete 350 MB Foggy Bottom dataset. Those claims require a full-data run and the
existing classifier evaluation ladder.
