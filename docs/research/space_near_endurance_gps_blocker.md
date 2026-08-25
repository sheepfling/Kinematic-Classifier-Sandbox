# Endurance GPS ECEF velocity blocker

NASA metadata identifies `ENDURANCE_EPHEMERIS_GPS` as containing onboard-GPS
ECEF velocity components in metres per second. The numerical CDF payload was not
persisted through the available acquisition path for this tranche.

Therefore:

- the included Endurance fixture remains a postflight reference position and
  attitude solution;
- `source_velocity_mps` remains absent;
- finite-difference velocity is used only as the compatibility-layer
  `derived_velocity_mps` required by the current `NormalizedTrack` contract;
- no derived velocity is relabeled as measured GPS velocity;
- native GPS velocity ingestion remains follow-up work.
