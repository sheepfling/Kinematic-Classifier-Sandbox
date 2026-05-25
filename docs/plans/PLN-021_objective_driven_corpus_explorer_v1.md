# PLN-021 Objective-Driven Corpus Explorer v1

Title: Objective-Driven Corpus Explorer v1
Plan ID: PLN-021
Status: done
Owner: @rick
Priority: P1
Last Updated: 2026-05-24

Objective:
Upgrade the multi-backend exploration proof from PLN-019 into an objective-driven automated corpus synthesis engine. The new system must generate candidates from declared objectives, validate class labels independently from requested targets, extract real features, score trajectories with actual classifier outputs, maintain an iterative quality-diversity archive, and materialize a selected corpus that can be consumed directly by the common study harness.

Scope:
- Add an explicit corpus-objective contract for target-driven generation.
- Replace the hard-coded candidate pool with pluggable samplers driven by objectives and backend capabilities.
- Add class-validity scoring so generated trajectories are not trusted solely because they were requested under a target class.
- Route generated trajectories through the real feature pipeline rather than only simple proxy features.
- Add classifier-in-the-loop scoring using actual classifier outputs instead of heuristic stress labels.
- Upgrade the archive from a best-of-fixed-pool summary into an iterative quality-diversity process.
- Materialize a selected corpus as a real data product with normalized trajectory data, feature matrices, validity scores, and classifier outputs.
- Keep the design backend-aware and dimension-aware so the same machinery can later support 3D and external simulators.

Out of Scope:
- Deep RL policy training. That remains downstream of stronger objective-driven search and archive proof.
- Real external high-fidelity backend integration in this plan.
- Claiming operational realism from generated corpora without separate domain validation.
- Replacing the common study harness. This plan must integrate with it.

Implementation Steps:
1. Add a corpus objective schema and example objective set.
   - Objectives must support:
     - target class
     - target class pair
     - target feature excitation
     - target difficulty
     - target posterior entropy
     - target environment regime
     - leakage constraints
     - backend constraints
     - runtime budget
2. Implement a candidate sampler layer.
   - Provide:
     - random sampling
     - grid sampling
     - Latin hypercube or Sobol
     - boundary mutation
     - archive mutation
     - stress mutation
   - Candidate generation must take objective files as input rather than rely on a fixed candidate list.
3. Add class-validity scoring and relabel flow.
   - A trajectory must be scored as:
     - `valid_target_class`
     - `ambiguous`
     - `invalid`
     - `relabel_candidate`
   - Class validity must be determined from normalized telemetry, events, and features.
4. Integrate the real feature pipeline.
   - Generated corpora should produce:
     - feature matrix
     - feature manifest
     - feature excitation scores
     - feature-set membership metadata where relevant
5. Add classifier-in-the-loop scoring.
   - Run actual scorers where applicable:
     - pointwise
     - windowed
     - sequential Bayes accumulator
     - Kalman bank
   - Compute:
     - posterior entropy
     - top-two margin
     - classifier correctness
     - confidence
     - confident error
     - prior flip threshold
     - time to confidence
     - method disagreement
     - oracle gap when available
6. Implement iterative quality-diversity archive filling.
   - Track:
     - archive coverage by iteration
     - cell elites
     - mutation lineage
     - successful vs failed archive cells
7. Materialize the selected corpus.
   - The output corpus must be directly consumable by the common study harness.
   - Preserve normalized telemetry, features, validity scores, classifier scores, and posterior histories.

Validation:
- Objective schema validates example objectives.
- Candidate samplers generate non-hard-coded candidates from objective files.
- Class-validity scoring rejects at least one invalid trajectory and flags at least one ambiguous trajectory.
- Feature matrices are generated for selected trajectories through the real feature path.
- Classifier-in-the-loop scores differ materially from the old heuristic stress scores.
- Archive coverage increases over iterations.
- Failed runs do not count toward successful archive coverage.
- Selected corpus can be consumed by the common study harness.
- The full targeted regression for this plan passes.

Artifacts / Config:
- `docs/plans/PLN-021_objective_driven_corpus_explorer_v1.md`
- `artifacts/corpus_objectives/corpus_objective_schema.json`
- `artifacts/corpus_objectives/example_objectives.yaml`
- `artifacts/corpus_objectives/objective_validation_report.md`
- `artifacts/candidate_generation/sampler_manifest.json`
- `artifacts/candidate_generation/generated_candidates.csv`
- `artifacts/candidate_generation/candidate_generation_report.md`
- `artifacts/class_validity/class_definition_schema.json`
- `artifacts/class_validity/class_validity_scores.csv`
- `artifacts/class_validity/class_validity_report.md`
- `artifacts/generated_corpus_features/feature_matrix.csv`
- `artifacts/generated_corpus_features/feature_manifest.json`
- `artifacts/generated_corpus_features/feature_excitation_scores.csv`
- `artifacts/generated_corpus_features/feature_generation_report.md`
- `artifacts/corpus_classifier_scoring/classifier_candidate_scores.csv`
- `artifacts/corpus_classifier_scoring/posterior_history.csv`
- `artifacts/corpus_classifier_scoring/prior_sensitivity_scores.csv`
- `artifacts/corpus_classifier_scoring/method_disagreement_scores.csv`
- `artifacts/corpus_classifier_scoring/classifier_scoring_report.md`
- `artifacts/quality_diversity_corpus_v1/archive_cells.csv`
- `artifacts/quality_diversity_corpus_v1/archive_elites.csv`
- `artifacts/quality_diversity_corpus_v1/archive_coverage_by_iteration.csv`
- `artifacts/quality_diversity_corpus_v1/archive_lineage.csv`
- `artifacts/quality_diversity_corpus_v1/qd_report.md`
- `artifacts/selected_generated_corpus/corpus_manifest.json`
- `artifacts/selected_generated_corpus/trajectories.csv`
- `artifacts/selected_generated_corpus/observations.csv`
- `artifacts/selected_generated_corpus/truth_states.csv`
- `artifacts/selected_generated_corpus/events.csv`
- `artifacts/selected_generated_corpus/environment_traces.csv`
- `artifacts/selected_generated_corpus/feature_matrix.csv`
- `artifacts/selected_generated_corpus/class_validity_scores.csv`
- `artifacts/selected_generated_corpus/classifier_scores.csv`
- `artifacts/selected_generated_corpus/posterior_history.csv`

Dependencies:
- `PLN-018` agentic corpus synthesis
- `PLN-019` multi-backend trajectory exploration
- `PLN-015` corpus coverage framework
- `PLN-011` generic feature taxonomy
- `PLN-010` generic inference contract
- `PLN-013` generic filtering contract
- existing modules:
  - `trajectory_backend_contract.py`
  - `backend_adapter_proof.py`
  - `environment_aware_corpus.py`
  - `capability_aware_search.py`
  - `generic_corpus_exploration.py`
  - `feature_analysis.py`
  - `common_experiment_harness.py`
  - `quality_diversity_corpus.py`
  - `adaptive_stress_corpus.py`

Milestones:
- `M36`: Corpus objective schema
  - Goal:
    - Define a formal objective contract for corpus generation.
  - Deliverables:
    - `artifacts/corpus_objectives/corpus_objective_schema.json`
    - `artifacts/corpus_objectives/example_objectives.yaml`
    - `artifacts/corpus_objectives/objective_validation_report.md`
  - Visualizations:
    - objective-field relationship diagram
    - example objective coverage map
  - Exit criterion:
    - Example objectives validate and cover class, feature, difficulty, environment, leakage, and runtime dimensions.
  - Progress:
    - complete

- `M37`: Candidate sampler layer
  - Goal:
    - Replace the fixed candidate pool with sampler-generated candidates driven by objectives.
  - Deliverables:
    - `artifacts/candidate_generation/sampler_manifest.json`
    - `artifacts/candidate_generation/generated_candidates.csv`
    - `artifacts/candidate_generation/candidate_generation_report.md`
  - Visualizations:
    - sampler family comparison chart
    - generated-candidate coverage plot
    - mutation lineage preview
  - Exit criterion:
    - The system generates candidates from objective files rather than relying on a hard-coded `_candidate_pool()`.
  - Progress:
    - complete

- `M38`: Class-validity scoring
  - Goal:
    - Add explicit class validation and relabel flow for generated trajectories.
  - Deliverables:
    - `artifacts/class_validity/class_definition_schema.json`
    - `artifacts/class_validity/class_validity_scores.csv`
    - `artifacts/class_validity/class_validity_report.md`
  - Visualizations:
    - class-validity confusion chart
    - ambiguous vs invalid vs valid distribution plot
    - alternate-class similarity heatmap
  - Exit criterion:
    - At least one trajectory is rejected as invalid and at least one is flagged ambiguous under the scoring logic.
  - Progress:
    - complete

- `M39`: Feature extraction integration
  - Goal:
    - Route generated corpora through the real feature pipeline.
  - Deliverables:
    - `artifacts/generated_corpus_features/feature_matrix.csv`
    - `artifacts/generated_corpus_features/feature_manifest.json`
    - `artifacts/generated_corpus_features/feature_excitation_scores.csv`
    - `artifacts/generated_corpus_features/feature_generation_report.md`
  - Visualizations:
    - feature excitation heatmap
    - feature-space coverage plot
    - selected-feature gallery
  - Exit criterion:
    - The selected generated corpus produces a real feature matrix, not only simple proxy score columns.
  - Progress:
    - complete

- `M40`: Classifier-in-the-loop scoring
  - Goal:
    - Score generated candidates with actual classifier outputs and posterior histories.
  - Deliverables:
    - `artifacts/corpus_classifier_scoring/classifier_candidate_scores.csv`
    - `artifacts/corpus_classifier_scoring/posterior_history.csv`
    - `artifacts/corpus_classifier_scoring/prior_sensitivity_scores.csv`
    - `artifacts/corpus_classifier_scoring/method_disagreement_scores.csv`
    - `artifacts/corpus_classifier_scoring/classifier_scoring_report.md`
  - Visualizations:
    - posterior entropy distribution
    - method disagreement heatmap
    - prior flip examples
    - time-to-confidence curves
  - Exit criterion:
    - Heuristic stress labels are replaced by measured classifier stress metrics.
  - Progress:
    - complete

- `M41`: Quality-diversity archive v1
  - Goal:
    - Implement an iterative archive that grows coverage over time.
  - Deliverables:
    - `artifacts/quality_diversity_corpus_v1/archive_cells.csv`
    - `artifacts/quality_diversity_corpus_v1/archive_elites.csv`
    - `artifacts/quality_diversity_corpus_v1/archive_coverage_by_iteration.csv`
    - `artifacts/quality_diversity_corpus_v1/archive_lineage.csv`
    - `artifacts/quality_diversity_corpus_v1/qd_report.md`
  - Visualizations:
    - archive coverage by iteration
    - elite-score distribution by cell
    - mutation lineage graph
  - Exit criterion:
    - Archive coverage increases over iterations and failed runs are separated from successful coverage.
  - Progress:
    - complete

- `M42`: Materialized selected corpus
  - Goal:
    - Emit a study-ready selected corpus rather than only a selection report.
  - Deliverables:
    - `artifacts/selected_generated_corpus/corpus_manifest.json`
    - `artifacts/selected_generated_corpus/trajectories.csv`
    - `artifacts/selected_generated_corpus/observations.csv`
    - `artifacts/selected_generated_corpus/truth_states.csv`
    - `artifacts/selected_generated_corpus/events.csv`
    - `artifacts/selected_generated_corpus/environment_traces.csv`
    - `artifacts/selected_generated_corpus/feature_matrix.csv`
    - `artifacts/selected_generated_corpus/class_validity_scores.csv`
    - `artifacts/selected_generated_corpus/classifier_scores.csv`
    - `artifacts/selected_generated_corpus/posterior_history.csv`
  - Visualizations:
    - selected corpus summary dashboard
    - class-validity breakdown plot
    - feature and classifier score gallery
  - Exit criterion:
    - The selected corpus is directly consumable by the common study harness.
  - Progress:
    - complete

Recommended Execution Order:
1. `M36` objective schema
2. `M37` sampler layer
3. `M38` class-validity scoring
4. `M39` feature extraction integration
5. `M40` classifier-in-the-loop scoring
6. `M41` quality-diversity archive v1
7. `M42` materialized selected corpus

Design Notes:
- PLN-019 should be treated as the architecture proof and adapter foundation, not the finished explorer.
- Candidate generation must become objective-driven rather than manually seeded.
- Feature generation must reuse the real feature stack already present in the repo.
- Classifier stress must become a measured quantity from classifier outputs, not a scenario-family heuristic.
- Successful and failed archive coverage must be separated explicitly.
- The selected corpus should become a first-class data product, not just a manifest of top rows.

Success Criteria:
- The corpus explorer can start from declared objectives and automatically generate backend-compatible candidates.
- Generated trajectories carry validated or relabeled class status rather than assumed truth.
- Selected corpora include real feature matrices and classifier-derived scoring surfaces.
- Archive coverage grows over iterations rather than only summarizing a fixed pool.
- The selected generated corpus can be passed directly into the common study harness without custom glue code.
