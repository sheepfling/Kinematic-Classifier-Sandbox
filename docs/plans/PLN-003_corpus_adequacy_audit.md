# PLN-003 Corpus Adequacy Audit

Plan ID: PLN-003
Status: done
Owner: @codex
Priority: P1
Objective: Add a single enforceable corpus adequacy audit that gates feature coverage, declared hard class pairs, scenario balance, and covariate leakage for the common 1D synthetic corpus.
Scope:
- Add a dedicated audit module and artifact writer for corpus adequacy.
- Reuse the existing trajectory generator, feature analysis, feature-set manifest, and class-pair manifest.
- Emit pass/warn/fail outputs plus explicit missing-coverage recommendations.
Out of Scope:
- Changing the generator physics or class definitions themselves.
- Replacing feature analysis or PCA artifacts.
- Adding a training-time classifier gate beyond corpus adequacy.
Implementation Steps:
1. Build `corpus_adequacy_audit.py` on top of `analyze_feature_datasets`.
2. Score feature coverage by manifest feature set and active feature excitation thresholds.
3. Score manifest class pairs against required tiers plus separability/overlap gates.
4. Audit class balance and covariate leakage and emit recommendations.
5. Export markdown, JSON, CSV, and PNG artifacts and wire them into package exports.
6. Add tests for analysis and artifact generation.
Validation:
- Unit tests for result structure, expected gates, and artifact emission.
- Manual run through the artifact export script to confirm outputs land under `artifacts/corpus_adequacy_audit_v1`.
Artifacts / Config:
- `src/kinematic_classifier_sandbox/corpus_adequacy_audit.py`
- `tests/test_corpus_adequacy_audit.py`
- `artifacts/corpus_adequacy_audit_v1/`
- `experiments/common_1d_classifier_study/feature_sets.json`
- `experiments/common_1d_classifier_study/class_pair_manifest.json`
Dependencies:
- Existing feature analysis outputs and trajectory generator metadata.
- Matplotlib for rendered audit plots.
Last Updated: 2026-05-23

Completion Notes:
- Implemented `corpus_adequacy_audit.py` with feature coverage, class-pair boundary coverage, class balance, and covariate leakage gates.
- Added `coverage_report.py` and `scripts/coverage_report.py` so the audit can be run directly and summarized across feature and classifier space.
- Added artifact export wiring, package exports, and tests for adequacy and coverage reporting.
- Verified current repo state with `python3 -m pytest -q`, which now passes end-to-end.
