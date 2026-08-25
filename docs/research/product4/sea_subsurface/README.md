# SEA-SUB source evidence and mixed-provenance fixtures

This directory is the Wave 1 Product 4 contribution for `sea_subsurface`. It preserves the source
portfolio, a retained selected-anchor artifact, provenance mappings, two restricted contract
fixtures, leakage-safe grouping, and the vertical-semantics clarification exposed by underwater
data.

It is not a completed production adapter or classifier study.

## Portfolio and gate status

| Role | Source | Evidence state | Current use |
| --- | --- | --- | --- |
| Anchor | IOOS/UAF Slocum `unit_191-20240309T1200` | `fixture_validated` | Retained 99-row mixed-provenance profile passes the canonical channel-aware common front; classifier view intentionally blocked |
| Independent validation | MGDS/WHOI Sentry AT26-09, DOI `10.60521/331959` | `access_verified` | Independent AUV navigation lineage |
| Contract regression | Official IOOS Murphy profile artifact | `fixture_validated`, restricted | Pressure versus derived depth; conservative interpolated-position semantics |
| Contract bridge | OOI/OOICI `unit_364` parser resource | `fixture_validated`, restricted | Separate GPS, dead-reckoned horizontal state, depth, and pressure |

`G1_source_portfolio` and `G2_selected_anchor_fixture` are complete. The selected anchor is
acquired, hashed, schema-inspected, mapped, and validated through the canonical multi-state-view
common front. Independent validation remains open because no Sentry PPL file has been retained;
classifier readiness is also blocked because one profile is not a target-bearing cohort.

## Selected-anchor evidence

The bounded profile `1709942882` is retained byte-for-byte at
`fixtures/ioos_uaf_unit_191_profile_1709942882.csv` with SHA-256
`29114562885c844dec7440148a4dbe8bfbc21efcc4f793190bdd0c5ff2a6d13a`.

Its 99 data rows demonstrate that the native table is a sparse asynchronous event stream rather
than a rectangular joint-state trajectory:

- one onboard GPS event and two dead-reckoned horizontal events, with no coincident GPS/DR row;
- 63 measured-pressure events, 24 source-depth events, and 63 standardized-depth events;
- five duplicate timestamps whose rows carry different populated channels;
- provider GPS units metadata declaring degrees-minutes while the retained values fit decimal-degree
  ranges and align with the profile neighborhood;
- constant provider-standardized latitude/longitude acting as profile context rather than per-sample
  observed trajectory.

Therefore, same-time rows must not be dropped, DDMM conversion is prohibited for this pinned
artifact, and GPS presence alone does not establish a surface-phase label.

## Contract findings

The evidence requires the common front to preserve:

- onboard GPS and underwater dead reckoning as distinct state roles;
- measured pressure separately from calculated or otherwise derived depth;
- sparse channel-event lineage and same-time rows until an explicit channel-aware coalescing policy;
- missing components through explicit null/validity semantics rather than zero filling;
- source-native frames and unresolved datums until a defensible transform exists;
- platform, deployment, profile/dive, recording, provider, and source identity for split planning;
- identity and provider fields outside ordinary kinematic classifier features;
- explicit source, processing, and release restrictions for every fixture.

The selected anchor and both restricted fixtures intentionally block classifier-view construction.
Their purpose is to prove representation and guardrails, not to supply a study-ready trajectory.

## Layout

- `source_cards/` and `scorecards/`: selected anchor and independent holdout.
- `mappings/`: source-to-common state-role mappings.
- `registry_updates/`: four-source Product 4 registry patch.
- `acquisition/`: immutable query, selected-anchor inspection, and Sentry event inventory.
- `fixtures/`: retained lawful source bytes and compact common-contract manifests.
- `change_requests/`: `SCR-SEA-SUB-001`, an acceptance clarification rather than a schema fork.
- `agent_status.yaml` and `coordination_sync.yaml`: comparable six-lane status reports.
- `MERGE_HANDOFF.md`: stack order, validation, and merge-agent claim boundary.

## Validation

```bash
python -m pytest -q -p no:cacheprovider \
  tests/corpus/real_world/test_sea_subsurface_research_fixtures.py \
  tests/corpus/real_world/test_sea_subsurface_selected_anchor.py
```

The repository checks validate source hashes and sizes, scorecard arithmetic, selected-anchor
channel counts and lifecycle state, asynchronous duplicate handling, the GPS units/value mismatch,
IOOS pressure/depth semantics, OOI DDMM conversion, separate GPS/dead-reckoned views, missing-state
handling, identity-only grouping, classifier-view blocking, and completed G2 common-front validation.

## Claim boundary

This tranche supports source selection, selected-anchor acquisition, canonical fixture validation,
restricted fixture validation, and a concrete common-contract clarification. It does not claim a
completed `P4-010` production adapter, Sentry artifact validation, a prepared pilot, classifier
performance, or military-submarine truth.
