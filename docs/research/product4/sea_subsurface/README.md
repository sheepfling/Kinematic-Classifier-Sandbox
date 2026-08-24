# SEA-SUB source evidence and mixed-provenance fixtures

This directory is the Wave 1 Product 4 contribution for `sea_subsurface`. It preserves the
source portfolio, provenance mappings, two restricted real-data fixtures, leakage-safe grouping,
and the vertical-semantics clarification exposed by underwater data.

It is not a completed production adapter or classifier study.

## Portfolio and gate status

| Role | Source | Evidence state | Current use |
| --- | --- | --- | --- |
| Anchor | IOOS/UAF Slocum `unit_191-20240309T1200` | `access_verified` | Selected mixed GPS/dead-reckoning pilot |
| Independent validation | MGDS/WHOI Sentry AT26-09, DOI `10.60521/331959` | `access_verified` | Independent AUV navigation lineage |
| Contract regression | Official IOOS Murphy profile artifact | `fixture_validated`, restricted | Pressure versus derived depth; conservative interpolated-position semantics |
| Contract bridge | OOI/OOICI `unit_364` parser resource | `fixture_validated`, restricted | Separate surface GPS, dead-reckoned horizontal state, depth, and pressure |

`G1_source_portfolio` is complete. `G2_selected_anchor_fixture` remains open because the selected
IOOS/UAF response has not been retained and hashed. Independent validation remains open because no
Sentry PPL file has been retained.

## Contract findings

The evidence requires the common front to preserve:

- surface GPS and underwater dead reckoning as distinct state roles;
- measured pressure separately from calculated or otherwise derived depth;
- missing components through explicit null/validity semantics rather than zero filling;
- source-native frames and unresolved datums until a defensible transform exists;
- platform, deployment, profile/dive, recording, provider, and source identity for split planning;
- identity and provider fields outside ordinary kinematic classifier features;
- explicit source, processing, and release restrictions for every fixture.

Both retained fixtures intentionally block classifier-view construction. Their purpose is to prove
representation and guardrails, not to supply a study-ready trajectory.

## Layout

- `source_cards/` and `scorecards/`: selected anchor and independent holdout.
- `mappings/`: source-to-common state-role mappings.
- `registry_updates/`: four-source Product 4 registry patch.
- `acquisition/`: immutable query and Sentry event inventory.
- `fixtures/`: retained lawful source bytes and compact common-contract manifests.
- `change_requests/`: `SCR-SEA-SUB-001`, an acceptance clarification rather than a schema fork.
- `agent_status.yaml` and `coordination_sync.yaml`: comparable six-lane status reports.

## Validation

```bash
python -m pytest -q tests/corpus/real_world/test_sea_subsurface_research_fixtures.py
```

The repository tests validate source hashes, scorecard arithmetic, IOOS pressure/depth semantics,
OOI raw-row parsing and DDMM conversion, separate GPS/dead-reckoned views, missing-state handling,
identity-only grouping, registry status, classifier-view blocking, and the open G2 boundary.

## Claim boundary

This tranche supports source selection, provenance mapping, restricted fixture validation, and a
concrete common-contract clarification. It does not claim a completed `P4-010` adapter, selected
anchor acceptance, Sentry artifact validation, a prepared pilot, classifier performance, or
military-submarine truth.
