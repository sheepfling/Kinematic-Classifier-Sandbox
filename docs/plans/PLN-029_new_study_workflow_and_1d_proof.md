# PLN-029 New Study Workflow And 1D Proof

Status: in_progress
Owner: @codex
Priority: P1
Last Updated: 2026-05-25

## Objective

Package the repo's existing study-candidate, feature-analysis, corpus-generation, corpus-audit, ladder-evaluation, and decision surfaces into a canonical new-user workflow that is explicit, rerunnable, and proven end-to-end on a 1D witness-style study.

## Scope

- Create a user-facing workflow guide for declaring, analyzing, generating, auditing, evaluating, deciding, and packaging a study.
- Add declaration templates for study candidates, class manifests, feature manifests, prior manifests, and corpus objectives.
- Add workflow scripts for:
  - feature/class geometry analysis
  - corpus generation and exploration
  - selected-corpus audit
  - classifier-ladder execution
  - final evaluation and packaging
- Produce a canonical `artifacts/new_study_workflow_demo/` bundle for the 1D `constant_velocity` vs `constant_acceleration` study.

## Out of Scope

- Replacing the existing study-candidate and witness-suite subsystems.
- Claiming the workflow is already generic across arbitrary 3D studies.
- Rewriting classifier implementations or trajectory generators.
- Introducing a full top-level `kc-study` CLI before the workflow contract is stable.

## Implementation Steps

1. Add a new-user workflow guide and checklist under `docs/workflows/`.
2. Add declaration templates under `templates/`.
3. Add a canonical workflow demo study config under `experiments/new_study_workflow_demo/`.
4. Add workflow scripts under `scripts/workflows/` that wrap existing artifact writers and package outputs into phase directories.
5. Create the phase outputs:
   - `00_study_declaration`
   - `01_feature_class_analysis`
   - `02_corpus_generation`
   - `03_corpus_audit`
   - `04_ladder_evaluation`
   - `05_report`
6. Add a final packaging step that writes `study_report.md`, `decision_card.md`, and `visual_gallery.md`.
7. Add focused regression coverage for the workflow demo.

## Validation

- The workflow guide explains the full path without requiring source-code reading.
- The demo study config is valid YAML and references existing class pairs, feature sets, and classifier IDs.
- The end-to-end workflow script produces all phase directories under `artifacts/new_study_workflow_demo/`.
- The final bundle includes:
  - `pairwise_auc.csv`
  - `candidate_scores.csv`
  - `class_validity_scores.csv`
  - `posterior_history_by_method.csv`
  - `study_report.md`
- Focused workflow regression passes.

## Artifacts / Config

- `docs/workflows/new_study_user_guide.md`
- `docs/workflows/new_study_checklist.md`
- `templates/study_candidate.yaml`
- `templates/class_manifest.csv`
- `templates/feature_manifest.csv`
- `templates/prior_manifest.csv`
- `templates/corpus_objective.yaml`
- `experiments/new_study_workflow_demo/new_study_workflow_demo.yaml`
- `scripts/workflows/analyze_feature_class_geometry.py`
- `scripts/workflows/generate_and_explore_corpus.py`
- `scripts/workflows/audit_selected_corpus.py`
- `scripts/workflows/run_classifier_ladder.py`
- `scripts/workflows/evaluate_and_package.py`
- `artifacts/new_study_workflow_demo/`

## Dependencies

- `study_candidate_protocol.py`
- `study_candidate_generation.py`
- `feature_analysis.py`
- `candidate_generation.py`
- `corpus_autodevelopment.py`
- `selected_generated_corpus.py`
- `corpus_adequacy_audit.py`
- `common_experiment_harness.py`
- `rung_sufficiency/analysis.py`

