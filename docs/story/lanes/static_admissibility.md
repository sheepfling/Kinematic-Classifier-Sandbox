# Static Admissibility Lane

## Purpose

Static admissibility decides whether a proposed feature set, class set, and prior regime is meaningful enough to send to Corpus Explorer or classifier/filter evaluation.

It is the first gate in the methodology pipeline:

```text
feature set + class set + prior regime
  -> static admissibility audit
  -> promote/revise/block decision
  -> Corpus Explorer or study revision
```

## Inputs

- Declared classes.
- Declared features.
- Prior probabilities.
- Feature provenance and online/offline availability.
- Controlled feature rows from the current witness corpus.

## Outputs

- `artifacts/packets/static_admissibility_mvp/decision_card.md`
- `artifacts/packets/static_admissibility_mvp/static_audit_report.md`
- `artifacts/packets/static_admissibility_mvp/class_confusability_matrix.csv`
- `artifacts/packets/static_admissibility_mvp/feature_relevance_table.csv`
- `artifacts/packets/static_admissibility_mvp/feature_redundancy_matrix.csv`
- `artifacts/packets/static_admissibility_mvp/feature_synergy_candidates.csv`
- `artifacts/packets/static_admissibility_mvp/prior_pathology_report.csv`
- `artifacts/packets/static_admissibility_mvp/static_leakage_provenance_audit.csv`

## Claim Boundary

The lane is an early admissibility screen. It does not prove final classifier performance, feature independence, or 3D deployment readiness. Synergy rows are labeled as candidates until downstream ablation confirms them.

## Hero Artifacts

- `02b_static_audit_decision_card.png`
- `02c_class_pair_confusability_matrix.png`
- `02g_prior_pathology_surface.png`
- `02e_feature_redundancy_graph.png`

## Tests And Validators

- `tests/static_admissibility/test_static_admissibility_packet.py`
- `tests/analysis/test_static_feature_class_prior_audit.py`
- `python -m kinematic_classifier_sandbox validate-packet artifacts/packets/static_admissibility_mvp`

## Next Work

- Route hard class pairs into Corpus Explorer boundary objectives.
- Route prior pathology into explicit prior sweeps.
- Validate synergy candidates by downstream ablation.
- Add a 3D PVA static admissibility dry run after vector feature schemas land.

