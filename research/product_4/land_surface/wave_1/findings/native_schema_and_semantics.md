# Native Schema and Semantics

## FHWA TGSIM Foggy Bottom

TGSIM remains the class-labeled anchor. Its documented native representation provides scoped
track identity, source-relative time, local horizontal position, horizontal velocity and
acceleration estimates, road-region context, dimension proxies, and road-user type codes.
Actual bounded complete-track rows are not retained in this tranche, so identity scope,
sentinels, duplicate behavior, and run boundaries remain open evidence questions.

## EPFL pNEUMA

pNEUMA is the preferred independent road source. Its documented layout represents one vehicle
trajectory per row with repeated time-sample fields, WGS-84 horizontal coordinates, road-user
type, speed, and vehicle-frame acceleration. The source documents mixed frame-rate behavior,
drone outages, and at least one georegistration issue. Actual trajectory member bytes are not
retained here.

## RWTH highD

highD is retained as a secondary validation direction because its provider documentation and
access terms are strong but raw and modified redistribution is restricted. It should not be the
portable common fixture.

## THI ReV-StED

The inspected immutable CSV is a wide synchronized vehicle-state record containing GNSS, INS,
local position, velocity, uncertainty, vehicle dynamics, status, and middleware transport
fields.

ADMA INS week/millisecond fields formed a regular 10 ms state clock in the bounded local slice.
`rosbagTimestamp` behaved as an irregular transport timestamp and therefore remains source
context rather than physical state time.

Horizontal geodetic position maps from `ins_lat_abs` and `ins_long_abs`. `ins_height` stays
separate because its vertical datum is unresolved. The analysis-view prototype uses
`ins_pos_rel_x`, `ins_pos_rel_y`, and horizontal `ins_vel_frame` components without inventing an
origin or axis rotation.

## Amazon Precision GNSS

The publisher documents four approximately two-hour, 10 Hz drives with paired low-cost and
reference GNSS streams over urban, tunnel, suburban, and highway regimes. It is suitable for
estimate/reference state-quality validation. Its CSV schema remains uninspected in this
tranche.
