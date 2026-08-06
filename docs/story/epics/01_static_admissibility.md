# Static Admissibility

Core question: before corpus generation or classifier/filter work, can the proposed feature, class, and prior setup support a meaningful study?

Epic 1 is the front-door gate for that question. It treats a study candidate as a declared feature/class/prior bundle and asks whether the study is identifiable, covered enough to be worth pursuing, prior-robust enough to move, and clean enough to avoid obvious leakage or unsupported evidence.

## Inputs

Epic 1 accepts a portable file-backed bundle:

- `static_audit_bundle.yaml`
- `samples.csv`
- `feature_schema.csv`
- `class_schema.csv`
- optional `class_feature_signature.csv` for planned classes without current samples

The bundle may declare any feature-vector dimension through descriptive
`dimension` metadata. Set `allow_unobserved_classes: true` when the declared
future class set intentionally includes classes not present in the current
sample table. Those classes remain unverified until labeled witness samples
arrive.

For notional 3D-inspired studies, the bundle may declare
`study_dimension` metadata with `source_type: normalized_feature_bundle` and
`static_audit_only: true`. That is a claim boundary, not a workaround.

## Outputs

Primary packet outputs:

- `decision_card.md`
- `validation_report.md`
- `claim_boundary.md`
- `hero_chart_manifest.csv`
- `lane_proof_matrix.md`
- class-pair collision, class-feature observability, alias, prior, redundancy,
  leakage, and coverage source tables
- run-backed hero charts

Primary Epic 1 packets:

- `artifacts/validation_packets/01_static_admissibility/`
- `artifacts/validation_packets/01_static_admissibility_multi_domain_3d/`

## Decision Routes

The static audit does not just dump metrics. It routes the study:

- `promote_to_corpus_explorer`
- `promote_with_warnings`
- `revise_feature_set`
- `revise_class_set`
- `revise_prior`
- `reject`

Wide uncertainty or thin cells can still route to Corpus Explorer rather than forcing premature class or feature revision.

## What Epic 1 Checks

- class confusability and separability
- feature relevance
- redundancy, aliasing, threshold subsumption, and decision redundancy
- candidate synergy
- prior pathology and prior evidence budgets
- leakage and provenance blockers
- observability gaps and unsupported classes
- exact shared-vector and near-signature collision candidates
- declared future classes with unverified expected signatures
- estimator reliability and static bounds

## Exemplar Suite

The top-level Epic 1 validation packet is the exemplar atlas:

- `experiments/static_admissibility/epic1_exemplar_suite.yaml`
- `docs/story/epics/01_static_admissibility_exemplars.md`
- `artifacts/validation_packets/01_static_admissibility/`

It proves that portable study bundles can be ingested, routed, validated, and explained without relying on internal repo history.

## MD3D Exemplar

The strongest teaching exemplar is the notional multi-domain 3D packet:

- `artifacts/validation_packets/01_static_admissibility_multi_domain_3d/`

This is a static 3D-inspired feature/class/prior audit over normalized synthetic feature values. It is not a raw 3D tracklet pipeline, not a 3D simulator, and not an operational Army/Navy/Space Force feature library.

## Claim Boundary

A static pass means the study is worth routing forward. It does not prove downstream dynamic classifier performance or operational 3D tracking performance.

Important guardrails:

- candidate synergy remains candidate until ablation-backed
- leakage blockers cannot promote
- unsupported features are not admissible evidence
- near-threshold retained features stay candidate-level when the threshold gap is below declared uncertainty
- static bounds are admissibility diagnostics, not operational performance guarantees

## How This Feeds Epic 2 And Epic 3

Epic 1 should narrow the search space before heavier work:

- Corpus Explorer receives thin-cell and coverage objectives.
- Later classifier/filter ladders receive only study candidates that survived static admissibility.
- Candidate synergy and retained threshold aliases become explicit ablation follow-up tasks.

## Regeneration

Build the top-level Epic 1 packet:

```bash
PYTHONPATH=src python3 -m kinematic_classifier_sandbox run-static-audit-suite \
  experiments/static_admissibility/epic1_exemplar_suite.yaml \
  --output-dir artifacts/validation_packets/01_static_admissibility
```

Build the MD3D teaching exemplar:

```bash
PYTHONPATH=src python3 -m kinematic_classifier_sandbox run-static-audit-multi-domain-3d \
  --output-dir artifacts/validation_packets/01_static_admissibility_multi_domain_3d
```
