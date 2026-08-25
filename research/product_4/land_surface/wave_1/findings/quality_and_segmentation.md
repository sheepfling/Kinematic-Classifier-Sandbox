# Quality and Segmentation

## Common LAND rules

- preserve the full recording or track as the split parent;
- create maneuver and quality intervals as segments, not independent objects;
- break analysis windows at real timing discontinuities without creating new physical groups;
- preserve source-native state and named analysis views separately;
- treat unavailable vertical state as invalid rather than zero;
- retain reported, derived, reference, and transport channels as distinct evidence roles.

## Source-specific findings

TGSIM requires actual-row audits for duplicate timestamps, out-of-order rows, sample gaps,
position jumps, source-versus-derived derivative disagreement, label inconsistency, and possible
track fragmentation.

pNEUMA requires file/session/drone-aware grouping, mixed cadence handling, drone-outage
missingness, and georegistration findings.

ReV-StED produced five local findings:

- `LAND_TRANSPORT_TIME_NOT_STATE_TIME`
- `LAND_LOCAL_FRAME_SEMANTICS_PARTIAL`
- `LAND_POSITION_VELOCITY_CONSISTENCY`
- `LAND_FIXTURE_TOO_SHORT_FOR_STUDY`
- `LAND_SOURCE_RIGHTS_UNRESOLVED`

The bounded local ReV-StED state timestamps were strictly increasing at 0.01 s. Irregular
middleware timestamps confirmed that message-arrival time must not define vehicle kinematics.

Amazon Precision GNSS must preserve each low-cost/reference pair as two state views of one drive,
with tunnel outages and GNSS quality regimes annotated rather than silently repaired.
