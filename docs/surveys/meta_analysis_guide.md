# Meta Analysis Guide

This guide maps the repo's main analysis questions to the current artifact bundles, reports, CSVs, and plots that answer them.

Use this when you want to know:

- which features matter
- which classes are confusable
- whether errors are due to priors, weak evidence, or corpus problems
- which classifier family performs best on which scenario
- whether PCA is showing structure or overlap
- whether the corpus itself is trustworthy
- whether advanced filters are justified

## First Stops

If you only open a few files first, open these:

- [PLN-002_kinematic_classification_roadmap.md](docs/plans/PLN-002_kinematic_classification_roadmap.md)
- [common_experiment_report.md](artifacts/common_1d_classifier_study/common_experiment_report.md)
- [technique_comparison_report.md](artifacts/technique_comparison_v1/technique_comparison_report.md)
- [common_dataset_comparison_report.md](artifacts/common_dataset_comparison_v1/common_dataset_comparison_report.md)
- [corpus_adequacy_report.md](artifacts/corpus_adequacy_audit_v1/corpus_adequacy_report.md)

## Feature Utility

Question:
- Which engineered features help, which are redundant, and which are noise-sensitive?

Primary artifacts:
- [feature_analysis_report.md](artifacts/feature_analysis_v1/feature_analysis_report.md)
- [feature_transfer_report.md](artifacts/feature_taxonomy/feature_transfer_report.md)
- [feature_workflow.md](docs/surveys/feature_workflow.md)

Key machine-readable surfaces:
- [feature_matrix.csv](artifacts/feature_analysis_v1/feature_matrix.csv)
- [identifiability_matrix.csv](artifacts/feature_analysis_v1/identifiability_matrix.csv)
- [feature_taxonomy.json](artifacts/feature_taxonomy/feature_taxonomy.json)
- [feature_sensitivity_matrix.csv](artifacts/feature_taxonomy/feature_sensitivity_matrix.csv)
- [feature_dependency_matrix.csv](artifacts/feature_taxonomy/feature_dependency_matrix.csv)

Use this when:
- deciding whether to add or remove a feature
- checking whether a feature bundle is mostly duration/noise leakage
- checking history behavior and dimensional transferability

## Class Separability And Confusability

Question:
- Which class pairs are well separated, and which are fundamentally confusable?

Primary artifacts:
- [feature_analysis_report.md](artifacts/feature_analysis_v1/feature_analysis_report.md)
- [common_experiment_report.md](artifacts/common_1d_classifier_study/common_experiment_report.md)

Key machine-readable surfaces:
- [identifiability_matrix.csv](artifacts/feature_analysis_v1/identifiability_matrix.csv)
- [pairwise_overlap_matrix.csv](artifacts/feature_analysis_v1/pairwise_overlap_matrix.csv)
- [pairwise_auc_matrix.csv](artifacts/feature_analysis_v1/pairwise_auc_matrix.csv)
- [metrics_by_class_pair.csv](artifacts/common_1d_classifier_study/metrics_by_class_pair.csv)
- [identifiability_matrix.csv](artifacts/common_1d_classifier_study/identifiability_matrix.csv)

Use this when:
- asking whether a failure is a classifier problem or a separability problem
- choosing which class pair to harden next

## Confusion And Error Structure

Question:
- Where do classifiers confuse classes, and under which scenario families?

Primary artifacts:
- [common_experiment_report.md](artifacts/common_1d_classifier_study/common_experiment_report.md)
- [technique_comparison_report.md](artifacts/technique_comparison_v1/technique_comparison_report.md)

Key machine-readable surfaces:
- [class_pair_scenario_study.csv](artifacts/common_1d_classifier_study/class_pair_scenario_study.csv)
- [class_pair_duration_study.csv](artifacts/common_1d_classifier_study/class_pair_duration_study.csv)
- benchmark-specific `confusion_final.csv` files under:
  - `artifacts/pointwise_baseline/`
  - `artifacts/windowed_baseline/`
  - `artifacts/kalman_filter_bank/`

Use this when:
- asking whether the error is early-horizon, noisy-horizon, outlier-driven, or boundary-driven

## Prior Sensitivity And Decision Fragility

Question:
- Are decisions evidence-driven or prior-driven?

Primary artifacts:
- [prior_sensitivity_report.md](artifacts/prior_sensitivity_v1/prior_sensitivity_report.md)
- [cross_method_prior_comparison_report.md](artifacts/prior_sensitivity_cross_method_v1/cross_method_prior_comparison_report.md)

Method-specific variants:
- [prior_sensitivity_pointwise_v1](artifacts/prior_sensitivity_pointwise_v1)
- [prior_sensitivity_windowed_raw_v1](artifacts/prior_sensitivity_windowed_raw_v1)
- [prior_sensitivity_windowed_robust_v1](artifacts/prior_sensitivity_windowed_robust_v1)

Key machine-readable surfaces:
- [prior_sensitivity.csv](artifacts/prior_sensitivity_v1/prior_sensitivity.csv)
- [prior_flip_thresholds.csv](artifacts/prior_sensitivity_v1/prior_flip_thresholds.csv)
- [prior_dominance_metrics.json](artifacts/prior_sensitivity_v1/prior_dominance_metrics.json)
- [prior_sensitivity_by_class_pair.csv](artifacts/common_1d_classifier_study/prior_sensitivity_by_class_pair.csv)

Use this when:
- checking whether a classifier is brittle on ambiguous tracks
- comparing fragility across method families

## Cross-Method Classifier Performance

Question:
- Which classifier family performs best overall, and on which failure mode?

Primary artifacts:
- [technique_comparison_report.md](artifacts/technique_comparison_v1/technique_comparison_report.md)
- [common_dataset_comparison_report.md](artifacts/common_dataset_comparison_v1/common_dataset_comparison_report.md)
- [common_experiment_report.md](artifacts/common_1d_classifier_study/common_experiment_report.md)

Key machine-readable surfaces:
- [technique_summary.csv](artifacts/technique_comparison_v1/technique_summary.csv)
- [method_summary.csv](artifacts/common_dataset_comparison_v1/method_summary.csv)
- [metrics_by_classifier.csv](artifacts/common_1d_classifier_study/metrics_by_classifier.csv)
- [metrics_by_sensor_regime.csv](artifacts/common_1d_classifier_study/metrics_by_sensor_regime.csv)

Use this when:
- comparing pointwise, windowed, Bayesian, and Kalman methods
- separating same-sensor comparisons from richer-sensor comparisons

Important interpretation rule:
- `technique_summary.csv` and `method_summary.csv` are now capability-aware surfaces.
- Read `applicability_status`, `primary_evaluation_family`, and `witness_artifact` before treating a row as part of the same leaderboard.
- `witness_only` means the method is in the shared family and reporting vocabulary, but its primary evidence currently comes from a targeted witness rather than the shared binary corpus.

## Feature-Set Performance

Question:
- Which feature bundle is strongest on a given pair or difficulty type?

Primary artifacts:
- [common_experiment_report.md](artifacts/common_1d_classifier_study/common_experiment_report.md)

Key machine-readable surfaces:
- [feature_set_comparison.csv](artifacts/common_1d_classifier_study/feature_set_comparison.csv)
- [metrics_by_classifier_and_feature_set.csv](artifacts/common_1d_classifier_study/metrics_by_classifier_and_feature_set.csv)
- [oracle_classifier_results.csv](artifacts/common_1d_classifier_study/oracle_classifier_results.csv)

Use this when:
- deciding whether to improve features or change classifier family
- checking whether classifier underperformance is below feature-only oracle separability

## Corpus Coverage, Leakage, And Validity

Question:
- Is the corpus good enough to trust the classifier comparisons?

Primary artifacts:
- [corpus_adequacy_report.md](artifacts/corpus_adequacy_audit_v1/corpus_adequacy_report.md)
- [coverage_report.md](artifacts/coverage_report_v1/coverage_report.md)

Key machine-readable surfaces:
- [corpus_adequacy_summary.json](artifacts/corpus_adequacy_audit_v1/corpus_adequacy_summary.json)
- [class_balance.csv](artifacts/corpus_adequacy_audit_v1/class_balance.csv)
- [class_pair_coverage.csv](artifacts/corpus_adequacy_audit_v1/class_pair_coverage.csv)
- [covariate_leakage_audit.csv](artifacts/corpus_adequacy_audit_v1/covariate_leakage_audit.csv)
- [feature_set_coverage.csv](artifacts/corpus_adequacy_audit_v1/feature_set_coverage.csv)

Use this when:
- deciding whether a benchmark is too easy
- checking class-linked duration, cadence, or irregular-sampling leakage
- checking whether declared hard pairs are actually hard

## PCA And Dimensionality Diagnostics

Question:
- Do the engineered features separate classes in low-dimensional diagnostic space?

Primary artifacts:
- [pca_report.md](artifacts/pca_analysis_v1/pca_report.md)
- [feature_workflow.md](docs/surveys/feature_workflow.md)

Additional PCA bundles:
- `artifacts/pca_analysis_instantaneous_v1/`
- `artifacts/pca_analysis_raw_extrema_v1/`
- `artifacts/pca_analysis_robust_extrema_v1/`
- `artifacts/pca_analysis_shape_window_v1/`
- `artifacts/pca_analysis_model_residuals_v1/`

Key machine-readable surfaces:
- [pca_coordinates.csv](artifacts/pca_analysis_v1/pca_coordinates.csv)
- [pca_loadings.csv](artifacts/pca_analysis_v1/pca_loadings.csv)
- [pca_explained_variance.csv](artifacts/pca_analysis_v1/pca_explained_variance.csv)

Important limitation:
- PCA is documented here as a diagnostic, not as a production classifier.
- There is currently no first-class PCA-classifier study bundle in the repo.

## Posterior And Evidence Mechanics

Question:
- How are posterior updates structured, and are evidence providers interchangeable?

Primary artifacts:
- [contract_report.md](artifacts/generic_inference_contract/contract_report.md)
- [classification_principles_report.md](artifacts/classification_evidence_proof/classification_principles_report.md)
- [posterior_update_math.md](docs/surveys/posterior_update_math.md)

Use this when:
- checking contract compatibility across classifier families
- checking whether two methods differ in evidence production or posterior recursion

## Filtering And Kalman-Family Diagnostics

Question:
- What does the filter backend emit, and how strong is the current Kalman ladder?

Primary artifacts:
- [filtering_principles_report.md](artifacts/filtering_contract/filtering_principles_report.md)
- [kalman_bank_report.md](artifacts/kalman_filter_bank/kalman_bank_report.md)
- [kalman_variant_comparison_report.md](artifacts/kalman_variant_comparison_v1/kalman_variant_comparison_report.md)
- [kalman_observable_comparison_report.md](artifacts/kalman_observable_comparison_v1/kalman_observable_comparison_report.md)
- [velocity_aided_kalman_comparison_report.md](artifacts/velocity_aided_kalman_comparison_v1/velocity_aided_kalman_comparison_report.md)

Use this when:
- checking whether a failure is due to weak observability or backend choice
- comparing position-only vs stronger-sensor Kalman regimes

## Switching And Transition Models

Question:
- Does transition-aware sequential inference help on switching trajectories?

Primary artifacts:
- [transition_matrix_accumulator_report.md](artifacts/transition_matrix_accumulator_v1/transition_matrix_accumulator_report.md)

Use this when:
- deciding whether switching scenarios justify more advanced mode models
- comparing static accumulation, transition accumulation, and switching Kalman behavior

## Advanced Filter Decision Gates

Question:
- Are IMM, particle filtering, or RBPF justified yet?

Primary artifacts:
- [advanced_filter_decision_report.md](artifacts/advanced_filter_decision_v1/advanced_filter_decision_report.md)
- [particle_filter_decision_report.md](artifacts/filtering_contract/particle_filter_decision_report.md)
- [rbpf_decision_report.md](artifacts/filtering_contract/rbpf_decision_report.md)

Use this when:
- deciding whether to add a new advanced backend
- documenting why the current answer is still “defer”

## 3D Readiness And Dimensional Lift

Question:
- Which parts of the repo are already dimension-agnostic, and which are still 1D-specific?

Primary artifacts:
- [dimensional_lift_audit.md](artifacts/dimensional_lift_audit/dimensional_lift_audit.md)

Key machine-readable surfaces:
- [module_dimension_status.csv](artifacts/dimensional_lift_audit/module_dimension_status.csv)
- [scalar_assumption_inventory.csv](artifacts/dimensional_lift_audit/scalar_assumption_inventory.csv)
- [required_3d_adapters.md](artifacts/dimensional_lift_audit/required_3d_adapters.md)

Use this when:
- scoping the next 3D-capable adapter work
- deciding whether a new study is blocked by scalar assumptions

## Suggested Reading Orders

If you are diagnosing a bad classifier result:
1. [common_experiment_report.md](artifacts/common_1d_classifier_study/common_experiment_report.md)
2. [common_dataset_comparison_report.md](artifacts/common_dataset_comparison_v1/common_dataset_comparison_report.md)
3. [prior_sensitivity_report.md](artifacts/prior_sensitivity_v1/prior_sensitivity_report.md)
4. [feature_analysis_report.md](artifacts/feature_analysis_v1/feature_analysis_report.md)
5. [corpus_adequacy_report.md](artifacts/corpus_adequacy_audit_v1/corpus_adequacy_report.md)

If you are designing new features:
1. [feature_workflow.md](docs/surveys/feature_workflow.md)
2. [feature_transfer_report.md](artifacts/feature_taxonomy/feature_transfer_report.md)
3. [feature_analysis_report.md](artifacts/feature_analysis_v1/feature_analysis_report.md)
4. [pca_report.md](artifacts/pca_analysis_v1/pca_report.md)
5. [corpus_adequacy_report.md](artifacts/corpus_adequacy_audit_v1/corpus_adequacy_report.md)

If you are deciding whether a more advanced filter is warranted:
1. [common_dataset_comparison_report.md](artifacts/common_dataset_comparison_v1/common_dataset_comparison_report.md)
2. [kalman_variant_comparison_report.md](artifacts/kalman_variant_comparison_v1/kalman_variant_comparison_report.md)
3. [transition_matrix_accumulator_report.md](artifacts/transition_matrix_accumulator_v1/transition_matrix_accumulator_report.md)
4. [advanced_filter_decision_report.md](artifacts/advanced_filter_decision_v1/advanced_filter_decision_report.md)
