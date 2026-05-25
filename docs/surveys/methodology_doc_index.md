# Methodology Documentation Index

This index explains how the methodology and algorithm documentation is split
across the current survey documents.

## Single Cohesive Entry Point

If you want one document that pulls the survey stack into a single narrative,
start with:

- [kinematic_classifier_methodology.pdf](artifacts/latex/kinematic_classifier_methodology.pdf)

If you want one document that literally combines the current survey notes into a
single long-form reference, open:

- [methodology_compendium.md](artifacts/methodology_compendium.md)

That paper is the synthesis layer. The survey notes below remain the
implementation-linked deep dives.

## Open These First

If you want the shortest path through the documentation stack:

1. [kinematic_classifier_methodology.pdf](artifacts/latex/kinematic_classifier_methodology.pdf)
2. [methodology_compendium.md](artifacts/methodology_compendium.md)
3. [posterior_update_math.md](docs/surveys/posterior_update_math.md)
4. [methodology_evaluation_framework.md](docs/surveys/methodology_evaluation_framework.md)
5. [classifier_ladder_and_contracts.md](docs/surveys/classifier_ladder_and_contracts.md)

Then open the corpus and dimensional-lift notes as needed.

## Document Roles

### `posterior_update_math`

Use this when you want:

- recursive Bayesian update math
- toy 1D filter-bank mechanics
- identity 1D direct-evidence mechanics
- posterior walkthrough and evidence-term interpretation

Primary module families:

- `toy_1d.py`
- `identity_1d.py`
- `posterior_explainer.py`
- `identity_posterior_explainer.py`
- `bayesian_walkthroughs.py`
- posterior-oriented artifact writers in `artifacts.py`

Primary artifacts:

- `artifacts/posterior_update_math.pdf`
- `artifacts/posterior_numeric_walkthrough.png`
- `artifacts/toy_1d_posterior_*`
- `artifacts/identity_1d_posterior_*`

### `methodology_evaluation_framework`

Use this when you want:

- prior sensitivity methodology
- pairwise AUC and overlap interpretation
- confusion interpretation
- corpus adequacy and leakage evaluation
- feature registry and inspection-bundle logic

Primary module families:

- `prior_sensitivity_analysis.py`
- `feature_analysis.py`
- `pca_analysis.py`
- `short_horizon_identifiability.py`
- `corpus_adequacy_audit.py`
- `coverage_report.py`
- `inspection_bundle.py`
- `generic_feature_taxonomy.py`
- `class_validity.py`
- `generated_corpus_features.py`
- `corpus_classifier_scoring.py`

Primary artifacts:

- `artifacts/methodology_evaluation_framework.pdf`
- `artifacts/prior_sensitivity_v1/`
- `artifacts/feature_analysis_v1/`
- `artifacts/abstract_inspection_v1/`
- `artifacts/corpus_adequacy_audit_v1/`

### `classifier_ladder_and_contracts`

Use this when you want:

- pointwise, windowed, accumulator, Kalman, and transition-aware methods
- the classifier ladder in implementation terms
- common inference and filtering contracts
- shared comparison harness and backend-adapter logic

Primary module families:

- `pointwise_baseline.py`
- `windowed_baseline.py`
- `sequential_bayes_accumulator.py`
- `kalman_filter_bank.py`
- `transition_matrix_accumulator.py`
- `common_dataset_comparison.py`
- `common_experiment_harness.py`
- `generic_inference_contract.py`
- `generic_filtering_contract.py`
- `backend_adapter_proof.py`

Primary artifacts:

- `artifacts/classifier_ladder_and_contracts.pdf`
- `artifacts/common_dataset_comparison_v1/`
- `artifacts/generic_inference_contract/`
- `artifacts/filtering_contract/`
- `artifacts/transition_matrix_accumulator_v1/`

### `corpus_generation_and_search`

Use this when you want:

- trajectory parameterization
- corpus candidate generation
- corpus adequacy objective scoring
- corpus search and Pareto reasoning
- study candidate generation and promotion logic

Primary module families:

- `trajectory_generator.py`
- `corpus_objectives.py`
- `adaptive_stress_corpus.py`
- `environment_aware_corpus.py`
- `quality_diversity_corpus.py`
- `corpus_autodevelopment.py`
- `corpus_gym.py`
- `corpus_search_baseline.py`
- `corpus_synthesis_comparison.py`
- `generic_corpus_exploration.py`
- `corpus_policy.py`
- `corpus_policy_sweep.py`
- `candidate_generation.py`
- `study_candidate_generation.py`
- `study_candidate_protocol.py`
- `capability_aware_search.py`

Primary artifacts:

- `artifacts/corpus_generation_and_search.pdf`
- `artifacts/corpus_autodevelopment_v1/`
- `artifacts/corpus_synthesis_comparison/`
- `artifacts/study_candidate_generation/`

### `dimensional_lift_and_advanced_filter_gates`

Use this when you want:

- dimensional-lift readiness
- scalar-assumption inventory logic
- advanced-filter go/no-go reasoning
- IMM / PF / RBPF decision-gate interpretation

Primary module families:

- `dimensional_lift_audit.py`
- `pca_dimensionality_audit.py`
- `advanced_filter_decision.py`
- `advanced_state_inference.py`
- `rl_backend_decision.py`
- advanced-filter portions of `generic_filtering_contract.py`

Primary artifacts:

- `artifacts/dimensional_lift_and_advanced_filter_gates.pdf`
- `artifacts/dimensional_lift_audit/`
- `artifacts/advanced_filter_decision_v1/`
- `artifacts/filtering_contract/`

## Worked Numeric Example Audit

The posterior documents already have worked numeric examples because the update
math is local and compositional. Not every newer methodology module deserves the
same treatment.

Good candidates for a worked numeric example:

- `sequential_bayes_accumulator.py`
  - Worth it because the forgetting-factor and abstention threshold can be shown
    on one short trajectory with explicit posterior values.
- `transition_matrix_accumulator.py`
  - Worth it because one mode-switch trajectory can show prior propagation,
    transition mixing, and emission terms numerically.
- `corpus_autodevelopment.py`
  - Worth it because one candidate corpus score can be decomposed into balance,
    coverage, excitation, leakage, triviality, and degeneracy terms.
- `advanced_filter_decision.py`
  - Worth it because one decision row can be shown from concrete gains like
    transition post-switch improvement and velocity-aided short-noisy gain.

Borderline candidates:

- `feature_analysis.py`
  - A worked AUC/overlap toy pair could help, but this is less urgent because
    the outputs are already heavily artifact-driven and not one recursive
    pipeline.
- `class_validity.py`
  - A single telemetry-derived relabel example could be useful if class schema
    work becomes more central.

Low-value candidates right now:

- `generic_inference_contract.py`
  - Better documented as schema and equivalence proof than as a numeric example.
- `dimensional_lift_audit.py`
  - Better documented as contract smoke-test structure than as arithmetic.
- `candidate_generation.py`
  - Better documented through sampler logic and promotion flow than through one
    numeric row.

Implemented numeric-example additions:

1. transition-matrix mode update walkthrough
2. corpus-autodevelopment score decomposition

Implemented numeric-example additions:

1. transition-matrix mode update walkthrough
2. corpus-autodevelopment score decomposition
3. advanced-filter decision gate walkthrough
4. corpus-gym reward walkthrough
5. generic corpus-explorer utility walkthrough

## Infrastructure Appendix

These modules are intentionally documented only as infrastructure or appendix
surfaces rather than standalone math papers:

- `__init__.py`
- `__main__.py`
- `catalog.py`
- `formal_math_registry.py`
- `formal_math_visual_registry.py`
- `milestones.py`
- `functional_surface_catalog.py`
- `repo_story.py`
- `showcase_builder.py`
- `methodology_latex.py`
- `runtime_paths.py`
- `strict_equation_audit.py`

Their mappings live in:

- `docs/surveys/methodology_doc_coverage.yaml`
- `artifacts/latex/methodology_doc_coverage.md`
