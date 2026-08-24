# Product 4 LAND Surface — Wave 1 Source Evidence

This research tranche records the first governed source portfolio for the Product 4
`land_surface` lane. It is stacked on the Product 4 architecture work and intentionally stays
outside the runtime package.

## Portfolio

| Role | Source | Evidence state |
|---|---|---|
| Anchor | FHWA TGSIM Foggy Bottom | `access_verified` |
| Independent class validation | EPFL pNEUMA | `access_verified`, rights conditional |
| Secondary class validation | RWTH highD | `access_verified`, redistribution restricted |
| Auxiliary contract witness | THI ReV-StED | `schema_inspected`, local validation only |
| State-quality expansion | Amazon Precision GNSS | `access_verified` |

TGSIM remains the required class-labeled anchor. ReV-StED exercises time, frame, validity,
lineage, grouping, and classifier-view isolation, but its single instrumented platform cannot
support road-user class discrimination. Amazon Precision GNSS is a paired estimate/reference
quality source, not a vehicle-class corpus.

## What this tranche establishes

- source cards and provisional 100-point scorecards for five serious candidates;
- one anchor, one independent class-validation source, one secondary source, and two typed
  auxiliary/expansion roles;
- TGSIM and ReV-StED adapter mappings against `trajectory-corpus-v0.1`;
- explicit time, frame, vertical, state-role, label, grouping, leakage, and claim boundaries;
- a proposed source-registry patch and artifact inventory;
- a reproducible repository-intake validator;
- blockers that prevent promotion beyond the recorded evidence states.

## Rights-safe repository boundary

Six consecutive ReV-StED records were inspected and used to construct a local two-view fixture.
The local structural validator passed. The concrete ReV-StED redistribution license remains
unresolved, so this public tranche does **not** commit those records, coordinate arrays, or
derivative fixture assets. It commits only the source identity, mapping, aggregate validation
results, non-reconstructive hashes, and claim boundary.

pNEUMA source rows are likewise withheld until the conflict between official license statements
is resolved. highD raw and modified data remain outside the repository under provider terms.

## Current gate status

```text
source portfolio accepted for review: yes
local ReV-StED structural validation: pass
authoritative COMMON-FRONT validation: not run
TGSIM complete bounded tracks retained: no
Product 4 LAND G2 fixture_validated: no
overall LAND evidence state: access_verified
```

This tranche does not claim classifier accuracy, car-versus-truck identifiability, a prepared
pilot, cross-source validation, tracked-vehicle coverage, armored semantics, or tank data.
