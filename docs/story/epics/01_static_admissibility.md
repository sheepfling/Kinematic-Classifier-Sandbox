# Static Admissibility

Core question: before corpus generation or classifier/filter work, can the proposed feature, class, and prior setup support a meaningful study?

This epic screens for class confusability, feature relevance, redundancy, candidate synergy, prior pathology, coverage feasibility, and leakage/provenance risk. Its output is a static decision, not just a pile of diagnostics.

Primary artifacts:

- `artifacts/packets/static_admissibility_mvp/README.md`
- `artifacts/packets/static_admissibility_mvp/decision_card.md`
- `artifacts/static_feature_class_prior_audit_v1/static_audit_report.md`
- `artifacts/static_feature_class_prior_audit_v1/class_confusability_matrix.csv`
- `artifacts/static_feature_class_prior_audit_v1/prior_pathology_report.csv`

Main chart set:

- `02b_static_audit_decision_card`
- `02c_class_pair_confusability_matrix`
- `02g_prior_pathology_surface`

Claim boundary: a static pass means the study is worth routing forward. It does not prove downstream dynamic classifier performance. Feature synergy remains candidate-level until confirmed by ablation.

Decision language:

- `promote_to_corpus_explorer`
- `promote_with_warnings`
- `revise_feature_set`
- `revise_class_set`
- `revise_prior_regime`
- `block_due_to_leakage`
- `reject_as_not_decisionable`

