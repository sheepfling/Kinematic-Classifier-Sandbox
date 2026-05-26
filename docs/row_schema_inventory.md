# Row Schema Inventory

This inventory was assembled from a repo-wide scan of row-shaped payloads on 2026-05-25.
It groups the repeated record shapes by inferred dataclass name so the codebase can move away from raw `dict` rows in a controlled way.

## Calibration and prediction rows

### `BinaryPredictionRow`

Used by binary calibration and posterior-quality metrics.

- `src/kinematic_classifier_sandbox/rung_sufficiency/analysis.py`
  - `_ece(rows, bins=10)`
  - `_binary_prediction_metrics(rows)`
  - `_posterior_quality_row(prediction_rows=...)`

Representative fields:

- `class_a`
- `class_b`
- `true_class`
- `predicted_class`
- `confidence`
- `posterior_class_a`
- `posterior_class_b`

### `CalibrationRow`

Used by calibration summaries in benchmark and inference surfaces.

- `src/kinematic_classifier_sandbox/inference/monte_carlo_benchmark.py`
- `src/kinematic_classifier_sandbox/advanced_filters/evaluation.py`

Representative fields:

- `confidence`
- `predicted_class`
- `true_class`
- `posterior`
- `correct`

## Rung sufficiency and decision rows

### `CorpusPreconditionRow`

Rows that describe whether a corpus can support a given evaluation.

- `src/kinematic_classifier_sandbox/rung_sufficiency/analysis.py`

Representative fields:

- `study_id`
- `corpus_id`
- `class_pair_id`
- `feature_set_id`
- `classifier_id`
- `corpus_status`
- `feature_excitation_status`
- `class_validity_status`
- `leakage_status`
- `boundary_coverage_status`
- `identifiability_status`

### `OracleGapRow`

Rows that compare achievable and current accuracy.

- `src/kinematic_classifier_sandbox/rung_sufficiency/analysis.py`

Representative fields:

- `oracle_accuracy`
- `best_oracle_accuracy_for_pair`
- `current_accuracy`
- `oracle_gap`
- `learnability_status`
- `learnable`

### `LearnabilitySurfaceRow`

Rows that combine corpus, oracle, and posterior signals into a promotion surface.

- `src/kinematic_classifier_sandbox/rung_sufficiency/analysis.py`

Representative fields:

- `mean_posterior_margin`
- `pairwise_auc`
- `overlap_estimate`
- `confusability_score`
- `oracle_threshold`
- `pairwise_auc_threshold`
- `posterior_margin_threshold`
- `overlap_threshold`

### `FailureModeRow`

Rows that record why a promotion or rejection happened.

- `src/kinematic_classifier_sandbox/rung_sufficiency/analysis.py`

Representative fields:

- `failure_mode`
- `failure_rationale`
- `current_rung_id`
- `candidate_next_rung_id`

### `PromotionDecisionRow`

Rows that encode the final promote/defer decision.

- `src/kinematic_classifier_sandbox/rung_sufficiency/analysis.py`

Representative fields:

- `decision`
- `rationale`
- `measured_next_accuracy`
- `measured_improvement`
- `runtime_cost_ratio`

## Common experiment rows

### `FeatureEvaluationRow`

Feature-level row outputs from the common experiment runner.

- `src/kinematic_classifier_sandbox/common_experiment/runner.py`
- `src/kinematic_classifier_sandbox/common_experiment/scoring.py`
- `src/kinematic_classifier_sandbox/common_experiment/plot_pack.py`

Representative fields:

- `study_id`
- `feature_set_id`
- `classifier_id`
- `feature_name`
- `score`
- `supporting_statistics`

### `PredictionRow`

Per-example classifier predictions and posteriors.

- `src/kinematic_classifier_sandbox/common_experiment/runner.py`
- `src/kinematic_classifier_sandbox/inference/pointwise_baseline.py`
- `src/kinematic_classifier_sandbox/inference/sequential_bayes_accumulator.py`

Representative fields:

- `trajectory_id`
- `time`
- `true_class`
- `predicted_class`
- `confidence`
- `posterior_by_class`

### `PosteriorHistoryRow`

History rows for posterior traces.

- `src/kinematic_classifier_sandbox/common_experiment/runner.py`
- `src/kinematic_classifier_sandbox/inference/windowed_baseline.py`
- `src/kinematic_classifier_sandbox/inference/transition_matrix_accumulator.py`

Representative fields:

- `posterior`
- `log_posterior`
- `confidence`
- `time`

### `LikelihoodHistoryRow`

Likelihood traces aligned to a study or trajectory.

- `src/kinematic_classifier_sandbox/common_experiment/runner.py`
- `src/kinematic_classifier_sandbox/inference/monte_carlo_benchmark.py`

Representative fields:

- `log_likelihood`
- `time`
- `classifier_id`
- `feature_set_id`

## Advanced filter and state-inference rows

### `ModeProbabilityRow`

Rows that track mode posteriors over time.

- `src/kinematic_classifier_sandbox/advanced_filters/runner.py`
- `src/kinematic_classifier_sandbox/inference/advanced_state_inference.py`

Representative fields:

- `trajectory_id`
- `time`
- `mode`
- `probability`

### `MixingRow`

Rows that describe the mixing weights between modes.

- `src/kinematic_classifier_sandbox/advanced_filters/runner.py`
- `src/kinematic_classifier_sandbox/inference/advanced_state_inference.py`

Representative fields:

- `source_mode`
- `destination_mode`
- `mixing_probability`

### `StateEstimateRow`

Rows that capture inferred state means and covariances.

- `src/kinematic_classifier_sandbox/advanced_filters/runner.py`
- `src/kinematic_classifier_sandbox/inference/advanced_state_inference.py`

Representative fields:

- `state_mean`
- `state_covariance`
- `position`
- `velocity`
- `acceleration`

### `DiagnosticRow`

Rows that carry extra per-step metrics and status flags.

- `src/kinematic_classifier_sandbox/advanced_filters/evaluation.py`
- `src/kinematic_classifier_sandbox/inference/advanced_state_inference.py`

Representative fields:

- `ess`
- `resampled`
- `entropy`
- `rmse`

## Corpus and sweep rows

### `PolicySweepRow`

Rows emitted by corpus-policy tuning and ablation sweeps.

- `src/kinematic_classifier_sandbox/corpus/policy_sweep.py`

Representative fields:

- `policy_id`
- `policy_score`
- `selected_jaccard_vs_default`
- `rank_spearman_vs_default`
- `rank_kendall_vs_default`

### `AblationRow`

Rows that compare a baseline against a single changed factor.

- `src/kinematic_classifier_sandbox/corpus/policy_sweep.py`

Representative fields:

- `factor_name`
- `baseline_score`
- `variant_score`
- `score_delta`

### `AdequacyRow`

Rows used by corpus adequacy audits and boundary checks.

- `src/kinematic_classifier_sandbox/corpus/adequacy_audit.py`
- `src/kinematic_classifier_sandbox/corpus/coverage_report.py`

Representative fields:

- `class_pair_id`
- `trajectory_id`
- `adequacy_score`
- `boundary_score`
- `coverage_score`

## Presentation and documentation rows

### `RegistryEntryRow`

Rows rendered in math and methodology registries.

- `scripts/render/render_math_metadata.py`
- `src/kinematic_classifier_sandbox/formal_math_visual_registry.py`

Representative fields:

- `symbol`
- `meaning`
- `status`
- `source`

### `MethodologySummaryRow`

Rows used for charts, report tables, and showcase summaries.

- `src/kinematic_classifier_sandbox/methodology_latex.py`
- `src/kinematic_classifier_sandbox/methodology_compendium.py`
- `src/kinematic_classifier_sandbox/analysis/inspection_bundle.py`

Representative fields:

- `title`
- `status`
- `category`
- `value`

## Notes

- The inventory intentionally groups several nearby row shapes together where the fields are already close enough to share a base dataclass plus small specializations.
- The highest-value next conversion targets are the binary calibration rows, the common-experiment prediction rows, and the advanced-state per-step records.
