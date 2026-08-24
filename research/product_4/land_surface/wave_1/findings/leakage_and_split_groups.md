# Leakage and Split Groups

LAND split planning must preserve physical and collection dependence before window creation.
Recommended opaque grouping namespaces include:

- `physical_platform`
- `source_recording`
- `source_track_or_trip`
- `route_or_road_segment`
- `geography`
- `temporal_collection`
- `source_dataset`

TGSIM windows from one scoped native track remain in one partition. pNEUMA rows from the same
recording/drone/session context must not be split as independent source cohorts. ReV-StED longer
extracts and maneuver segments inherit the same physical-platform and source-recording groups.

For Amazon Precision GNSS, each drive is one recording parent. Its low-cost and reference streams
are two views of that same episode—not independent train/test objects.

Raw IDs, file names, recording identities, provider identity, route identity, lane/region,
dimension proxies, and labels remain grouping, context, or audit metadata unless a study
explicitly authorizes a non-kinematic feature set.
