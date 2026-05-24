# PLN-015 Corpus Coverage Framework

Title: M17 Corpus Adequacy And Coverage Framework
Plan ID: PLN-015
Status: proposed
Owner: @rick
Priority: P1
Objective: Make corpus quality measurable independently of classifier choice by auditing balance, excitation, leakage, duration, scenario coverage, and boundary coverage through a stable methodology layer.
Scope:
- Generalize corpus adequacy from the current 1D audit into an explicit methodology milestone.
- Require corpus manifests, adequacy reports, and coverage artifacts that can be reused by future corpora.
- Treat corpus quality as a first-class gating layer before classifier comparisons are trusted.
Out of Scope:
- Full replacement of the current generator.
- External data ingestion.
- Production sensor data QA.
Implementation Steps:
1. Define the generalized corpus manifest and adequacy expectations.
2. Audit class balance, scenario balance, duration/sample-count coverage, feature excitation, class-pair coverage, covariate leakage, and sensor regime coverage.
3. Ensure intentionally injected leakage and missing-boundary conditions are detected automatically.
4. Emit a stable artifact family that future corpora can reuse.
Validation:
- Injected leakage is detected.
- Missing class-pair boundary coverage is detected.
- Coverage artifacts can be consumed independently of classifier choice.
Artifacts / Config:
- `artifacts/corpus_manifest.json`
- `artifacts/corpus_adequacy_report.md`
- `artifacts/class_balance.csv`
- `artifacts/scenario_balance.csv`
- `artifacts/covariate_leakage_audit.csv`
- `artifacts/feature_excitation_matrix.csv`
- `artifacts/class_pair_coverage.csv`
Dependencies:
- `PLN-003`
- `PLN-009`
- `PLN-010`
- `PLN-014`
Last Updated: 2026-05-24
