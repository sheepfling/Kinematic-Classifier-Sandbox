# HIVE-COTE / Modern TSC Frontier

The repo now has a dedicated proxy packet for the broader modern TSC lane:

- study id: `tsc_archive_baseline_frontier_v1`
- artifacts: `artifacts/tsc_archive_baseline_frontier_v1/`

## What It Proves

This packet keeps the modern TSC lane explicit without claiming faithful
external implementations. It compares:

- `rocket_proxy`
- `drcif_proxy`
- `dictionary_proxy`
- `hive_cote_proxy`
- `windowed_robust`
- `kalman_bank`

on the shared binary dynamics corpus.

The current packet is enough to justify:

- `drcif_interval_forests` moving from tracked-only status to `implemented`
- `dictionary_tde_family` moving from tracked-only status to `implemented`
- `hive_cote` moving from `researched` to `implemented`
- the shared `tsc_archive_baseline_frontier` witness moving from missing to a
  proxy-stage packet

## Claim Boundary

This is not yet a claim that the repo has wrapped or trained faithful
DrCIF, BOSS/WEASEL/TDE, HIVE-COTE, or related archive baselines.

What remains open:

- faithful external-method implementations or wrappers
- calibration and seed-stability checks on those methods
- broader comparison against named physics-aware witnesses

The registry therefore keeps `minirocket_family`,
`drcif_interval_forests`, `dictionary_tde_family`, and `hive_cote` at
`implemented`, not `witness_supported`.
