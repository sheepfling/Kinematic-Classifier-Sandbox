# PLN-020 LaTeX Documentation Coverage Audit And Expansion

Title: LaTeX Documentation Coverage Audit And Expansion
Plan ID: PLN-020
Status: in_progress
Owner: @rick
Priority: P1
Objective: Expand the survey and methodology documentation so every substantive methodology module under `src/kinematic_classifier_sandbox/` is mapped to an explicit documentation home, build path, and artifact output, while keeping recursive inference, evaluation methodology, classifier ladder, corpus generation, and dimensional-lift material separated into maintainable LaTeX and Markdown companions.
Scope:
- Add a checked-in coverage manifest for substantive methodology modules.
- Expand the existing posterior and evaluation survey documents.
- Add new deep-dive documents for classifier ladder/contracts, corpus generation/search, and dimensional-lift/advanced-filter gates.
- Add a survey index and generated coverage artifacts.
- Add build scripts and tests for the methodology documentation stack.
- Refresh the top-level methodology paper source so it acts as a concise synthesis document that links out to the deeper survey notes.
Out of Scope:
- Rewriting the classifier, corpus, or filtering implementations themselves.
- Replacing the team packet or showcase system.
- Treating package glue or export plumbing as standalone math papers.

Implementation Steps:
1. Add a source-of-truth coverage manifest in `docs/surveys/`.
   - Record one row per module with:
     - `module_path`
     - `coverage_kind`
     - `primary_doc`
     - `primary_section`
     - `artifact_outputs`
     - `status`
   - Include infrastructure-only modules and map them to the survey index or appendix layer.
2. Add a renderer that turns the checked-in manifest into generated artifacts.
   - Emit `artifacts/latex/methodology_doc_coverage.json`.
   - Emit `artifacts/latex/methodology_doc_coverage.md`.
3. Expand `posterior_update_math.{tex,md}`.
   - Keep it focused on recursive inference mechanics and posterior artifact generation.
   - Cover `toy_1d.py`, `identity_1d.py`, `posterior_explainer.py`, `identity_posterior_explainer.py`, `bayesian_walkthroughs.py`, and posterior-oriented artifact writers.
4. Expand `methodology_evaluation_framework.{tex,md}`.
   - Keep it focused on priors, AUC/overlap/confusion, PCA, corpus adequacy, coverage, and feature taxonomy internals.
   - Cover `prior_sensitivity_analysis.py`, `feature_analysis.py`, `pca_analysis.py`, `short_horizon_identifiability.py`, `corpus_adequacy_audit.py`, `coverage_report.py`, `inspection_bundle.py`, and `generic_feature_taxonomy.py`.
5. Add `classifier_ladder_and_contracts.{tex,md}`.
   - Document the pointwise/windowed/accumulator/Kalman/transition ladder.
   - Document shared schemas, common harness flow, backend adapter logic, and contract proofs.
6. Add `corpus_generation_and_search.{tex,md}`.
   - Document trajectory generation, adaptive corpus logic, corpus search, autodevelopment, candidate generation, and promotion logic.
7. Add `dimensional_lift_and_advanced_filter_gates.{tex,md}`.
   - Document dimensional-lift audit logic and advanced-filter decision gates.
8. Add `methodology_doc_index.md`.
   - Explain which survey to open first and which module families each document covers.
9. Add build scripts for each deep-dive document and an aggregate builder.
10. Add tests that validate:
   - manifest completeness
   - script smoke builds
   - generated artifact existence
   - expected subsystem anchors in generated docs

Validation:
- Every `.py` module in `src/kinematic_classifier_sandbox/` is present in the coverage manifest.
- Every manifest row has `primary_doc`, `primary_section`, and `coverage_kind`.
- The coverage renderer emits nonempty JSON and Markdown artifacts.
- All survey build scripts complete successfully.
- The generated docs mention the intended subsystem anchors for posterior, evaluation, ladder/contracts, corpus/search, and dimensional-lift/advanced-filter topics.
- All generated PDFs and Markdown mirrors exist and are nonempty.

Artifacts / Config:
- `docs/surveys/methodology_doc_coverage.yaml`
- `docs/surveys/methodology_doc_index.md`
- `docs/surveys/classifier_ladder_and_contracts.tex`
- `docs/surveys/classifier_ladder_and_contracts.md`
- `docs/surveys/corpus_generation_and_search.tex`
- `docs/surveys/corpus_generation_and_search.md`
- `docs/surveys/dimensional_lift_and_advanced_filter_gates.tex`
- `docs/surveys/dimensional_lift_and_advanced_filter_gates.md`
- `artifacts/latex/methodology_doc_coverage.json`
- `artifacts/latex/methodology_doc_coverage.md`
- `artifacts/classifier_ladder_and_contracts.pdf`
- `artifacts/classifier_ladder_and_contracts.md`
- `artifacts/corpus_generation_and_search.pdf`
- `artifacts/corpus_generation_and_search.md`
- `artifacts/dimensional_lift_and_advanced_filter_gates.pdf`
- `artifacts/dimensional_lift_and_advanced_filter_gates.md`
- `scripts/build/build_classifier_ladder_and_contracts.sh`
- `scripts/build/build_corpus_generation_and_search.sh`
- `scripts/build/build_dimensional_lift_and_advanced_filter_gates.sh`
- `scripts/build/build_methodology_docs.sh`
- `scripts/render/render_methodology_doc_coverage.py`

Dependencies:
- `PLN-017` automated methodology proof and LaTeX exposition
- Existing posterior and evaluation survey docs
- Existing artifact bundles under `artifacts/`
- Existing methodology paper source in `src/kinematic_classifier_sandbox/methodology_latex.py`

Last Updated: 2026-05-24
