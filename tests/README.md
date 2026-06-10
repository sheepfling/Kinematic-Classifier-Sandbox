# Test Suite Map

The test suite is intentionally broad and mostly mirrors the code and artifact
surface one module at a time.

Read next:

- [Repo story](../docs/story/00_repo_story.md)
- [Package map](../src/kinematic_classifier_sandbox/README.md)
- [Scripts layout](../scripts/README.md)

## How the tests are organized

- Most files follow `tests/test_<module_or_artifact>.py`
- Source modules usually have a matching test module
- Major artifact families and study bundles also have direct regression tests

## Main test groupings

- Inference and classifier ladder:
  - `test_pointwise_baseline.py`
  - `test_windowed_baseline.py`
  - `test_sequential_bayes_accumulator.py`
  - `test_kalman_filter_bank.py`
  - `test_transition_matrix_accumulator.py`
  - `test_toy_1d.py`
  - `test_identity_1d.py`
- Corpus and study-candidate tooling:
  - `test_trajectory_generator.py`
  - `test_corpus_adequacy_audit.py`
  - `test_coverage_report.py`
  - `test_corpus_*`
  - `test_study_candidate_*`
- Analysis and abstract inspection:
  - `test_feature_analysis.py`
  - `test_pca_analysis.py`
  - `test_inspection_bundle.py`
  - `test_dimensional_lift_audit.py`
- Generic methodology proof artifacts:
  - `test_generic_inference_contract.py`
  - `test_generic_feature_taxonomy.py`
  - `test_generic_filtering_contract.py`
  - `test_generic_classification_evidence_proof.py`
- Showcase, story, and documentation layers:
  - `test_showcase_builder.py`
  - `test_repo_story.py`
  - `test_methodology_doc_coverage.py`
  - `test_methodology_latex.py`
  - `test_methodology_compendium.py`
- Advanced filters:
  - `test_imm_filter.py`
  - `test_particle_filter.py`
  - `test_particle_filter_bank.py`
  - `test_rbpf.py`
  - `test_advanced_filter_*`

## Useful targeted runs

- Fast environment and artifact-facing checks:
  - `python3 -m pytest tests/test_artifacts.py tests/test_showcase_builder.py`
- Feature and inspection stack:
  - `python3 -m pytest tests/test_feature_analysis.py tests/test_pca_analysis.py tests/test_inspection_bundle.py`
- Core ladder methods:
  - `python3 -m pytest tests/test_pointwise_baseline.py tests/test_windowed_baseline.py tests/test_sequential_bayes_accumulator.py tests/test_kalman_filter_bank.py`

## Rule of thumb

- Add a new test file when a new module or artifact family becomes first-class.
- Prefer one clear responsibility per test module over giant mixed suites.
- Keep generated-state assumptions explicit so failures tell you whether the
  issue is code, config, or missing artifact preconditions.
