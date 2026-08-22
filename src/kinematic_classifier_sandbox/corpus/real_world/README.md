# Real-World Trajectory Ingestion

This package preserves source trajectories, labels, provenance, and quality findings before a
study projects them into the existing classifier contracts. It is additive: synthetic witnesses
and one-dimensional experiment adapters remain unchanged.

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
    config=TgsimFoggyBottomAdapterConfig(accessed_on=date(2026, 8, 21))
)
result = adapter.load_file("TGSIM-Foggy Bottom-Data.csv")

for track in result.tracks:
    print(track.provenance.split_group_id, track.labels.normalized_class)
```

The repository does not redistribute the raw dataset. The manifest records the official landing
page, download endpoint, citation, DOI, access date, coordinate-frame description, and license.

## Leakage boundary

The baseline motion lane may use position-derived and source-provided kinematic channels. The
following fields are retained for diagnostics but are not baseline classifier features:

- smoothed vehicle length and width (`audit_only`)
- lane or intersection-region identifier (`context`)
- absolute source coordinates
- source, run, and track identifiers

All windows derived from one source track must retain its `split_group_id`; a later splitter must
never place one parent track in multiple partitions.

## Current claim boundary

This package validates and normalizes real observations. It does not yet window tracks, assign
train/validation/test partitions, project tracks into `ExecutableTrajectory`, or establish that
vehicle classes are kinematically identifiable. Those are separate evidence-producing steps.
