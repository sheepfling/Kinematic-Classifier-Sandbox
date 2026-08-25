# Product 4 Real-World Corpus Portfolio

This page is the operating view for the six-domain corpus. The registry at
`docs/product4/real_world_source_registry.yaml` is the machine-readable source of truth; this
page explains the promotion order and the current claim boundary.

## Current maturity

The portfolio is coherent enough to integrate and evaluate, but it is not yet a prepared
classifier corpus:

| Lane | Best current evidence | What that means now | Next promotion gate |
| --- | --- | --- | --- |
| `land_surface` | `access_verified` | TGSIM is the reference adapter direction; no real bounded fixture is admitted | Acquire and validate a bounded complete-track artifact |
| `sea_surface` | `fixture_validated` | CMRE/Brest is a real expansion/regression witness; raw rows remain external | Build the common snapshot and add an independent AIS provider |
| `sea_subsurface` | `mapping_complete` | The selected IOOS profile has rich state semantics; common-front fixture is still open | Preserve asynchronous measured/dead-reckoned channels, then validate Sentry independently |
| `air_atmospheric` | `access_verified` | `readsb` parser semantics are covered by a documented fixture, not a historical flight corpus | Acquire a genuine historical trace and construct one flight-leg episode |
| `space_near` | `fixture_validated` | Six bounded mission fixtures exercise reference/analysis lineage and mission grouping | Convert fixtures through the common snapshot loader; keep mission identity audit-only |
| `space_orbital` | `fixture_validated` | NASA ISS proves OEM semantics as validation-only; IGS remains the preferred anchor candidate | Build a multi-object prepared cohort and resolve official IGS provenance |

Three lanes have fixture-level evidence, but zero lanes are marked `prepared` or `released` in
the registry. That is the correct result: the current work proves contract coherence and source
semantics, not cross-domain classifier performance.

## Priority order

Prioritize by readiness and information gain, not by the apparent importance of a physical domain.

1. Finish the common snapshot path and promote the existing fixture-backed witnesses: SEA-SURF,
   SPACE-NEAR, and SPACE-ORB. This exercises repeated platform, mission identity, frame/time,
   missing-state, and reference-versus-observation boundaries with the least new acquisition risk.
2. Close the two anchor gaps in parallel: bounded LAND/TGSIM acquisition for the reference road
   study, and the SEA-SUB selected IOOS profile with channel-aware timestamp handling.
3. Add independent cohorts before making within-domain generalization claims: Kystverket or NOAA
   for SEA-SURF, Sentry for SEA-SUB, Endurance/OpenSky for AIR, and IGS or a separate orbital
   object family for SPACE-ORB.
4. Promote only lanes that pass grouped splits, quality/label review, classifier-view leakage
   checks, and a declared study policy. Cross-domain evaluation comes after those per-lane gates,
   and only for hypotheses whose labels and state roles have physical meaning across the selected
   domains.

## Update rules

- Add a new source as a new registry entry with a unique dataset ID and artifact IDs.
- Advance `evidence_state` only when the corresponding evidence and hash/rights boundary are
  committed alongside the entry.
- Keep raw or restricted bytes outside Git; retain a pinned query, source hash, and regeneration
  instructions where redistribution is unavailable.
- Never reuse a snapshot ID after changing episodes, source artifacts, adapter versions, or split
  assignments.
- Keep source identity, provider, geography, mission, and grouping values in metadata; classifier
  assets must remain explicitly projected and identity-free.
- Treat `fixture_validated` as a semantic regression gate, not as evidence of classifier readiness.

The evaluation report is intended to become the front door for every future corpus update: first
the registry must cover the required lanes, then the snapshot must account for every episode and
hash, and finally the split/leakage and downstream classifier gates must pass.
