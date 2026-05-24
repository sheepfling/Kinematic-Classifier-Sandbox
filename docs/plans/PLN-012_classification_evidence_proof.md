# PLN-012 Classification Evidence Proof

Title: M14 Generic Classification And Evidence-Combination Proof
Plan ID: PLN-012
Status: done
Owner: @rick
Priority: P1
Objective: Prove that the repo's classifiers are instances of a common evidence-provider plus posterior-updater pattern rather than separate algorithm silos.
Scope:
- Define a generic `EvidenceProvider` interpretation for current classifier families.
- Keep posterior updates generic across evidence sources.
- Add equivalence tests proving identical likelihood streams imply identical posterior histories.
Out of Scope:
- New advanced classifiers.
- Full model replacement of the current ladder.
- IMM/PF/RBPF implementation.
Implementation Steps:
1. Define evidence-provider categories for pointwise, windowed, empirical-feature, residual, and Kalman-innovation methods.
2. Separate evidence production from posterior updating in the contract layer.
3. Add equivalence tests and method manifests.
4. Emit a classification-principles report.
Validation:
- Two evidence providers with identical log-likelihood streams produce identical posterior histories.
- Different evidence providers can still emit artifacts with the same shape.
- Metrics and plots can compare those outputs directly.
Artifacts / Config:
- `artifacts/classification_evidence_proof/evidence_provider_manifest.json`
- `artifacts/classification_evidence_proof/method_equivalence_tests.json`
- `artifacts/classification_evidence_proof/classification_principles_report.md`
Dependencies:
- `PLN-010`
- current pointwise, windowed, accumulator, and Kalman implementations
Last Updated: 2026-05-24
