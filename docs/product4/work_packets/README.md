# Product 4 lane work packets

These packets are the handoff surface for the six real-world corpus research agents. Each agent
owns one lane, keeps raw/restricted data outside Git, and returns only provenance, contracts,
tests, manifests, and documentation that are appropriate for the lane's evidence state.

| Lane | Packet | Current state | Immediate promotion target |
| --- | --- | --- | --- |
| LAND | [land_surface.md](land_surface.md) | bounded prepared task exists; aggregate fixture is not classifier-ready | complete-track source plus independent road holdout |
| SEA-SURF | [sea_surface.md](sea_surface.md) | bounded R_06-versus-R_14 route task is classifier-ready | independent AIS provider/source shift |
| SEA-SUB | [sea_subsurface.md](sea_subsurface.md) | canonical IOOS fixture validated; classifier blocked | independent Sentry artifact and legitimate multi-episode task |
| AIR | [air_atmospheric.md](air_atmospheric.md) | documented parser fixture only; access verified | genuine historical trace and provider holdout |
| SPACE-NEAR | [space_near.md](space_near.md) | six fixture contracts pass; authoritative semantic gate pending | resolve datum/frame/mission semantics before preparation |
| SPACE-ORB | [space_orbital.md](space_orbital.md) | NASA ISS validation fixture; classifier blocked | multi-object cohort with verified IGS provenance |

## Shared rules

- Do not commit raw restricted bytes, identity keys, generated snapshots, PDFs, `.tex`, or `.ltx` files.
- Keep generated snapshots, reports, and analysis manifests under an external path such as `/private/tmp`.
- Preserve source hashes, access dates, license/terms status, adapter versions, and claim boundaries.
- Keep physical identity, mission, provider, geography, and recording values in audit/grouping metadata.
- A classifier view must be identity-free, label-out-of-band, grouped, and backed by a prepared source.
- Do not advance `evidence_state` merely because a parser or common-front test passes.

Every packet should return:

1. a short evidence/status note;
2. changed registry and adapter paths;
3. external input paths and SHA-256 values;
4. focused tests and their output;
5. the exact promotion decision and remaining blockers.
