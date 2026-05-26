# PLN-032 Module Split Map

Status: proposed
Owner: @rick
Priority: P2
Last Updated: 2026-05-26

## Metadata

- Plan ID: PLN-032
- Title: Module Split Map
- Objective: Split the remaining mixed modules into scenario-specific, shared helper, and reporting/plotting layers so each file has one clear responsibility and the public import surface stays stable.

## Scope

- Split mixed analysis and corpus modules where scenario-specific logic is currently interleaved with reusable helper code and plotting/output code.
- Keep pure contracts, dataclasses, and named result types separate from orchestration.
- Preserve current artifact names, CSV schemas, and report text unless a file split requires a thin compatibility wrapper.
- Use the existing witness/surface pattern as the default boundary for scenario-specific study code.

## Out of Scope

- Changing numerical behavior or benchmark semantics.
- Renaming user-facing artifact directories unless a migration is explicitly required by a split.
- Rewriting already clean contract-only modules.
- Addressing unrelated worktree edits that belong to other refactor lines.

## Current File Map

- [`src/kinematic_classifier_sandbox/corpus/adaptive_stress.py`](/Users/rick/Library/Mobile%20Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/src/kinematic_classifier_sandbox/corpus/adaptive_stress.py)
  - Scenario-specific orchestration: `analyze_adaptive_stress_corpus`, `write_adaptive_stress_corpus_artifacts`.
  - Keep orchestration here; move only reusable helpers out if they become shared.
- [`src/kinematic_classifier_sandbox/corpus/adaptive_stress_utils.py`](/Users/rick/Library/Mobile%20Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/src/kinematic_classifier_sandbox/corpus/adaptive_stress_utils.py)
  - Mixed file.
  - Generic helpers: `_local_window_features`, `_observable_pair_posterior`, `_infer_shared_scenario_name`, `_to_shared_trajectory`, `_reference_window_stats`, `_classify_window_row`, `_prediction_bundle`, `_static_candidate_row`.
  - Scenario-specific helpers: `_stress_targets`, `_random_action`, `_guided_action`, `_transition_delay_candidates`, and the active score functions.
  - This is the best candidate for a split into helpers, scenario generators, and score/evaluation modules.
- [`src/kinematic_classifier_sandbox/corpus/exploration/backend_adapter_proof.py`](/Users/rick/Library/Mobile%20Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/src/kinematic_classifier_sandbox/corpus/exploration/backend_adapter_proof.py)
  - Mixed file.
  - Generic helpers: `_stable_input_hash`, `_times_for_candidate`, `_deterministic_noise`, `_run_row`, `_equivalence_rows`, `_render_telemetry_comparison_png`, `_render_failure_taxonomy_png`.
  - Scenario-specific truth/candidate construction: `_parameter_truth`, `_switching_truth`, `_shared_boundary_candidate`, `_switching_candidate`, `_environment_candidate`, `_failing_candidates`.
  - Adapters and top-level proof orchestration should be kept separate from the plotting/report code.
- [`src/kinematic_classifier_sandbox/trajectory_generator_rendering.py`](/Users/rick/Library/Mobile%20Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/src/kinematic_classifier_sandbox/trajectory_generator_rendering.py)
  - Mostly report/output code.
  - Generic row and manifest builders: `_trajectory_rows`, `_true_state_rows`, `_dataset_manifest`, `_trajectory_manifest`.
  - Plot/report helpers: `_render_dataset_plot`, `_render_trajectory_generator_report`, `_render_figure_svg`, `_render_figure_png`.
  - Likely does not need a large split unless the row builders become shared by another consumer.
- [`src/kinematic_classifier_sandbox/validation/technique_comparison_runner.py`](/Users/rick/Library/Mobile%20Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/src/kinematic_classifier_sandbox/validation/technique_comparison_runner.py)
  - Method-comparison orchestration with reusable row builders.
  - Generic helper: `_safe_mean`.
  - Method-specific row builders: `_pointwise_row`, `_windowed_rows`, `_accumulator_row`, `_kalman_row`, `_kalman_velocity_aided_row`.
  - If split, move row builders to a companion module and keep `analyze_technique_comparison` here.
- [`src/kinematic_classifier_sandbox/witnesses/advanced_state_inference_witnesses.py`](/Users/rick/Library/Mobile%20Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/src/kinematic_classifier_sandbox/witnesses/advanced_state_inference_witnesses.py)
  - Scenario-specific witness generator.
  - Keep as-is unless a downstream consumer needs a shared scenario contract.
- [`src/kinematic_classifier_sandbox/witnesses/benchmarks/transition_matrix_runner.py`](/Users/rick/Library/Mobile%20Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/src/kinematic_classifier_sandbox/witnesses/benchmarks/transition_matrix_runner.py)
  - Benchmark runner with mixed scenario setup and math helpers.
  - Scenario helpers: `default_switching_mode_specs`, `default_transition_matrix`, `_true_mode_series`, `generate_transition_switching_scenarios`.
  - Generic math helpers: `_speed_and_accel`, `_emission_log_scores`, `_emission_term_breakdown`, `_kalman_update_scalar`.
  - Benchmark execution: `_run_mode_accumulator`, `_SwitchingKalmanModeBank`, `_run_kalman_mode_bank`, `run_transition_benchmark`.
  - If split, keep the benchmark runner thin and move only reused math into a companion helper module.
- [`src/kinematic_classifier_sandbox/corpus/quality_diversity_utils.py`](/Users/rick/Library/Mobile%20Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/src/kinematic_classifier_sandbox/corpus/quality_diversity_utils.py)
  - Mixed archive/row-builder module.
  - Generic helpers: `_bucket`, `_archive_cell_id`, `_episode_row`, `_metric_row`.
  - Scenario/experiment logic: `build_quality_diversity_corpus`.
  - Good candidate for moving row-shaping helpers into a shared corpus archive utility module.
- [`src/kinematic_classifier_sandbox/corpus/selected_generated_corpus.py`](/Users/rick/Library/Mobile%20Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/src/kinematic_classifier_sandbox/corpus/selected_generated_corpus.py)
  - Mixed selection/orchestration module.
  - Generic helpers: `_objective_lookup`, `_canonical_pair`, `_canonical_scenario_id`, `_record_to_executable`.
  - Top-level orchestration: `analyze_selected_generated_corpus`.
  - Good candidate for a selection helper module plus a thin orchestration wrapper.
- [`src/kinematic_classifier_sandbox/corpus/synthesis_comparison.py`](/Users/rick/Library/Mobile%20Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/src/kinematic_classifier_sandbox/corpus/synthesis_comparison.py)
  - Mixed comparison module.
  - Generic helpers: `_bucket`, `_archive_cell_id`, `_manual_generator_rows`, `_rows_from_search`, `_rows_from_qd`, `_rows_from_stress`, `_metric_row`.
  - Scenario/analysis orchestration: `analyze_corpus_synthesis_comparison`.
  - Can stay as the orchestrator once the row helpers are extracted.
- [`src/kinematic_classifier_sandbox/__init__.py`](/Users/rick/Library/Mobile%20Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/src/kinematic_classifier_sandbox/__init__.py)
  - Generic package entrypoint wrapper only.
  - No split needed.
- [`scripts/export_artifacts.py`](/Users/rick/Library/Mobile%20Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/scripts/export_artifacts.py)
  - Top-level export orchestration only.
  - No semantic split; clean up imports after module boundaries settle.

## Implementation Steps

1. Split `corpus/adaptive_stress_utils.py` into:
   - `adaptive_stress_helpers.py` for shared helpers and shared row builders.
   - `adaptive_stress_scenarios.py` for target and action generation.
   - `adaptive_stress_scoring.py` for the score functions.
   - keep `adaptive_stress.py` as orchestration plus artifact writing.
2. Split `corpus/exploration/backend_adapter_proof.py` into:
   - `backend_adapter_proof_scenarios.py` for candidate construction and truth synthesis.
   - `backend_adapter_proof_adapters.py` for adapter classes.
   - `backend_adapter_proof_reporting.py` for rows, plots, and report text.
   - keep `backend_adapter_proof.py` as the top-level entrypoint if compatibility is needed.
3. Extract row/manifest helpers from `trajectory_generator_rendering.py` only if another module reuses them; otherwise leave it as the rendering boundary.
4. Split `validation/technique_comparison_runner.py` only if the method row builders are reused elsewhere; otherwise keep it as the method-comparison orchestrator.
5. Split `witnesses/benchmarks/transition_matrix_runner.py` only if the math helpers become shared with reporting or other benchmark surfaces; otherwise leave the benchmark family together.
6. Split `corpus/quality_diversity_utils.py`, `corpus/selected_generated_corpus.py`, and `corpus/synthesis_comparison.py` into row-helper modules plus thin orchestration modules.
7. Leave `witnesses/advanced_state_inference_witnesses.py`, `inference/transition_matrix/contracts.py`, and `src/kinematic_classifier_sandbox/__init__.py` as-is unless a specific consumer forces a boundary change.

## Validation

- Run `python3 -m py_compile` on any files touched by a split.
- Run focused tests for the affected modules after each extraction.
- Confirm no public artifact path or CSV schema changes unless explicitly intended.
- Confirm import cycles are not introduced when moving helpers across module boundaries.

## Artifacts / Config

- `docs/plans/PLN-032_module_split_map.md`
- Potential new modules:
  - `corpus/adaptive_stress_helpers.py`
  - `corpus/adaptive_stress_scenarios.py`
  - `corpus/adaptive_stress_scoring.py`
  - `corpus/exploration/backend_adapter_proof_helpers.py`
  - `corpus/exploration/backend_adapter_proof_scenarios.py`
  - `corpus/exploration/backend_adapter_proof_reporting.py`
  - `corpus/quality_diversity_rows.py`
  - `corpus/selected_generated_corpus_rows.py`
  - `corpus/synthesis_comparison_rows.py`

## Dependencies

- `PLN-030_row_schema_dataclass_inventory_and_conversion.md`
- `PLN-031_tuple_return_type_normalization.md`
- Existing witness surfaces in `src/kinematic_classifier_sandbox/witnesses/surface.py`
- Existing shared utilities in `src/kinematic_classifier_sandbox/utils/`

