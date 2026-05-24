# PLN-011 Generic Feature Taxonomy

Title: M13 Generic Feature Taxonomy And Feature-Set Proof
Plan ID: PLN-011
Status: done
Owner: @rick
Priority: P1
Objective: Prove that the repo's feature machinery is based on generic feature principles rather than one-off 1D feature hacks by attaching role, sensitivity, history, and dimensional-transfer metadata to every feature.
Scope:
- Define a feature taxonomy schema with history behavior, geometry, dependency, and sensitivity tags.
- Require all declared features to carry metadata needed for transfer and audit.
- Make feature-set selection possible by tags as well as by explicit name lists.
Out of Scope:
- Full redesign of every feature extractor.
- Full 3D feature implementation.
- Learned representation features.
Implementation Steps:
1. Define the feature taxonomy fields and tag vocabulary.
2. Annotate every feature with role, history behavior, sensitivity, and transferability.
3. Emit taxonomy, sensitivity, dependency, and transfer reports.
4. Add tests that enforce taxonomy completeness and cumulative/history labels.
Validation:
- Every feature has metadata.
- Every feature declares history behavior.
- Every cumulative feature is labeled cumulative.
- Every feature declares whether it is scalar-only or vector-compatible.
- Feature-set runner can select features by tags.
Artifacts / Config:
- `artifacts/feature_taxonomy/feature_taxonomy.json`
- `artifacts/feature_taxonomy/feature_sets.json`
- `artifacts/feature_taxonomy/feature_sensitivity_matrix.csv`
- `artifacts/feature_taxonomy/feature_dependency_matrix.csv`
- `artifacts/feature_taxonomy/feature_transfer_report.md`
Dependencies:
- `PLN-009`
- `PLN-010`
- existing feature registry and feature-set manifest machinery
Last Updated: 2026-05-24
