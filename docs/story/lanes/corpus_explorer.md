# Corpus Explorer Lane

## Purpose

Corpus Explorer generates, selects, searches, and audits corpora after a study candidate clears static admissibility.

Static audit findings become corpus objectives:

- hard class pair -> boundary-search objective
- prior pathology -> prior-sweep objective
- coverage gap -> feature-space coverage objective
- synergy candidate -> ablation objective

## Outputs

- `artifacts/corpus_adequacy_audit_v1/`
- `artifacts/generic_corpus_exploration/`
- `artifacts/trajectory_exploration_rl/`
- `artifacts/packets/static_admissibility_mvp/decision_card.md`

## Claim Boundary

CEM and PPO are search backends, not magic data generators. They are promoted only when baseline comparison, adequacy/leakage checks, and downstream diagnostic yield support the claim.

## Next Work

- Add more seed/objective stability for CEM/PPO.
- Wire static-audit warnings directly into generated objective specs.
- Keep scalar corpus scores paired with Pareto and adequacy surfaces.

