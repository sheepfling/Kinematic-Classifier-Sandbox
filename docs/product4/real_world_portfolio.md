# Product 4 Real-World Corpus Portfolio

This page is the operating view for the six-domain corpus. The registry at
`docs/product4/real_world_source_registry.yaml` is the machine-readable source of truth; this
page explains the promotion order and the current claim boundary.

## Current maturity

The portfolio is coherent enough to integrate and evaluate, but it is not yet a prepared
six-lane classifier corpus. LAND now has one explicitly bounded, task-scoped prepared cohort;
that does not promote the full portfolio or establish source-shift generalization:

| Lane | Best current evidence | What that means now | Next promotion gate |
| --- | --- | --- | --- |
| `land_surface` | `prepared` | A hash-pinned official 12-track cohort has an external task-scoped speed-profile classifier snapshot; grouped 10/20/30-second studies contain both vehicle classes in train/validation/test; raw rows remain external | Add an independent road-vehicle source shift |
| `sea_surface` | `fixture_validated` | CMRE/Brest is a real expansion/regression witness; raw rows remain external | Build the common snapshot and add an independent AIS provider |
| `sea_subsurface` | `mapping_complete` | The selected IOOS profile now has a channel-aware validation-only common front; G2 remains open | Preserve asynchronous measured/dead-reckoned channels, then validate Sentry independently |
| `air_atmospheric` | `access_verified` | The documented readsb fixture now has a validation-only common front, not a historical flight corpus | Acquire a genuine historical trace and construct one flight-leg episode |
| `space_near` | `fixture_validated` | Six bounded mission fixtures exercise reference/analysis lineage and mission grouping | Convert fixtures through the common snapshot loader; keep mission identity audit-only |
| `space_orbital` | `fixture_validated` | NASA ISS now has a validation-only common front; IGS remains the preferred anchor candidate | Build a multi-object prepared cohort and resolve official IGS provenance |

Three lanes have fixture-level evidence and LAND has bounded task-scoped prepared evidence; zero
lanes are `released`. That is the correct result: the current work proves contract coherence and
one explicit within-source classifier task, not cross-domain or source-shift performance.

## Priority order

Prioritize by readiness and information gain, not by the apparent importance of a physical domain.

1. Run the external six-lane validation builder using the already pinned or committed bounded
   inputs. This exercises repeated platform, mission identity, frame/time, missing-state, and
   reference-versus-observation boundaries without promoting the remaining sources to `prepared`.
2. Treat the bounded LAND/TGSIM Product 2 bridge as the current task-level evidence boundary: it
   uses one deterministic held-out representative per physical platform and train-only empirical
   references. Close the SEA-SUB selected IOOS profile with channel-aware timestamp handling; the
   LAND source-shift gate remains open.
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

Run the composed report without creating a repository artifact:

```bash
PYTHONPATH=src python3 scripts/audit/evaluate_product4_gates.py
```

Build a manifest only after domain adapters have written episode JSON files into an external
snapshot root:

```bash
PYTHONPATH=src python3 scripts/run/build_product4_snapshot.py \
  --snapshot-root /path/to/external/product4-snapshot \
  --snapshot-id product4-snapshot-v0.1
```

Use `--require-prepared-sources` for a classifier promotion attempt. Without that flag, the
builder can create a validation snapshot from fixture-backed episodes, but the composed gate will
continue to report the missing evidence, rights, quality, coverage, or classifier-view reasons.

The current SPACE-NEAR tranche has a reproducible external builder:

```bash
PYTHONPATH=src python3 scripts/run/build_space_near_validation_snapshot.py \
  --snapshot-root /path/to/external/product4-space-near \
  --snapshot-id product4-space-near-validation-v0.1
```

It materializes six common-front episode manifests and hashed state assets, but keeps the
classifier view intentionally absent while the source portfolio remains `fixture_validated`.

The full validation-only tranche can be assembled when the pinned SEA-SURF files and private
grouping key are available locally. All generated assets, raw rows, and the key stay outside Git:

```bash
PYTHONPATH=src python3 scripts/run/build_product4_validation_snapshot.py \
  --snapshot-root /path/to/external/product4-six-lane \
  --snapshot-id product4-six-lane-validation-v0.1 \
  --cmre-tracklets /path/to/tracklets.csv \
  --cmre-nomenclature /path/to/nomen.csv \
  --cmre-identity-key /secure/path/cmre-identity.key

PYTHONPATH=src python3 scripts/audit/evaluate_product4_gates.py \
  --snapshot /path/to/external/product4-six-lane/snapshot.json \
  --assignments /path/to/external/product4-six-lane/assignments.json

PYTHONPATH=src python3 scripts/audit/build_analysis_product_manifest.py \
  --snapshot /path/to/external/product4-six-lane/snapshot.json \
  --product kinematic_analysis \
  --output /path/to/external/product4-six-lane/kinematic-analysis.json
```

The validation snapshot is expected to pass immutable snapshot integrity, six-lane coverage,
quality, and grouped leakage gates. It must continue to fail classifier projection, prepared-source,
or release-rights gates for the remaining selected sources until they have independently passed
those promotions.

The `source_audit` and `kinematic_analysis` profiles can be materialized from this validation
snapshot. The `classifier_ladder` profile is intentionally stricter: it selects only episodes with
classifier views and fails unless their registry sources are `prepared`, so a deep analysis product
cannot be mistaken for a Product 2 held-out evaluation input. A classifier request must also name
the target-label namespace explicitly; route or provenance labels are not silently treated as
platform classes.

The checked-in registry currently reports `revise_source_portfolio`: all six lanes are represented,
LAND is prepared only for the bounded task-scoped cohort, and the full classifier bridge remains
blocked until the remaining lanes are prepared. A validation snapshot can be evaluated from
outside the repository with `--snapshot PATH --assignments PATH`;
add `--require-pass` when the caller wants a blocked promotion to fail the command. Keep restricted
source bytes, snapshot assets, and generated report files outside Git unless their rights and
repository role are explicitly established.
