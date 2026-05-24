# PLN-017 Automated Methodology Proof And LaTeX Exposition

Title: Automated Methodology Proof, Corpus Synthesis, And LaTeX Exposition
Plan ID: PLN-017
Status: done
Owner: @rick
Priority: P1
Objective: Extend the current kinematic-classifier methodology stack so it can automatically define, score, generate, validate, explain, and promote Corpus + Feature Set + Class Set + Classifier/Filter study candidates, while producing proof-oriented showcase artifacts and a LaTeX methodology document.
Scope:
- Add a machine-readable study-candidate layer above the current corpus, feature, classifier, and showcase stack.
- Add an explicit analysis protocol for evaluating Feature + Class + Classifier proposals.
- Add automatic corpus development against adequacy objectives rather than only manual corpus tweaking.
- Add automatic study proposal generation, static screening, Monte Carlo scoring, and promotion decisions.
- Add a validation ladder that unifies static compatibility, corpus adequacy, oracle separability, classifier performance, prior sensitivity, robustness, and dimensional-transfer assessment.
- Add Bayesian evidence walkthrough artifacts that expose priors, likelihoods, Bayes factors, posterior histories, and flip thresholds.
- Add a LaTeX methodology document that explains the meta-process, toy witness problems, Bayesian equations, algorithm ladder, and 1D-to-3D transition.
- Refresh the showcase around proof claims rather than only artifact categories.
Out of Scope:
- Implementing IMM, PF, or RBPF beyond current decision reports unless the new gates explicitly justify them.
- Claiming full 3D readiness without vector corpus, vector feature, and vector filter implementations.
- Replacing the current showcase packet with raw CSVs or code-only documentation.
- Treating the current 1D corpus as final rather than as a witness/problem-framing surface.

Implementation Steps:
1. Define the generic study-candidate object and decision protocol.
   - Represent a study candidate as:
     - `CorpusSpec`
     - `FeatureSetSpec`
     - `ClassSetSpec`
     - `ClassifierSpec`
     - `PriorSpec`
     - optional `FilterSpec`
     - optional `VisualizationSpec`
   - Add an explicit promotion vocabulary:
     - `promote`
     - `revise`
     - `reject`
     - `defer`
2. Add the analysis protocol.
   - Create one team-readable, stepwise protocol for evaluating any new Feature + Class + Classifier proposal.
   - Make the protocol checklist-driven rather than dependent on ad hoc judgment.
3. Add automatic corpus development.
   - Define corpus adequacy objectives in config.
   - Generate candidate corpora, score them, compare them, and select better candidates.
   - Preserve rejected candidates and Pareto-front evidence rather than only the winner.
4. Add automatic study proposal generation.
   - Generate study candidates from current feature sets, class pairs, classifier families, priors, corpus tiers, and optional filter families.
   - Score candidates statically before Monte Carlo.
5. Add the static/statistical validation ladder.
   - For each candidate, run:
     1. static compatibility
     2. corpus adequacy
     3. feature separability
     4. oracle separability
     5. classifier performance
     6. posterior/calibration quality
     7. prior sensitivity
     8. adversarial/stress robustness
     9. dimensional transfer assessment
     10. promotion decision
6. Add Bayesian evidence walkthroughs.
   - Expose priors, likelihoods, Bayes factors, posterior updates, log odds, feature contribution, and flip thresholds for representative studies.
   - Treat cumulative and correlated features explicitly as Bayesian warnings, not just as numerical inputs.
7. Add the LaTeX methodology document.
   - Build a paper-style artifact that explains:
     - problem formulation
     - generic study object
     - corpus synthesis loop
     - study candidate screening loop
     - Bayesian update equations
     - feature taxonomy
     - classifier ladder
     - filtering taxonomy and advanced-method gates
     - toy witness problems
     - 3D transition path
     - limitations and next work
8. Refresh the showcase around proof claims.
   - Add a proof gallery organized by explicit claims and the visuals that support them.
   - Treat the current 1D problems as witness problems proving distinct methodology layers.

Validation:
- Full regression still passes after all new automation, packet, and LaTeX tooling changes.
- A machine-readable study candidate schema exists and generated candidates conform to it.
- The corpus-candidate score table is nonempty.
- The selected corpus has a higher adequacy score than at least one rejected candidate.
- The static candidate score table is nonempty.
- The validation ladder emits `promote`, `revise`, `reject`, or `defer` decisions.
- Bayesian walkthrough plots reference existing trajectories, classes, and study candidates.
- Prior-to-posterior calculations match hand-checkable examples.
- The LaTeX document builds successfully.
- The proof gallery references existing artifacts only.
- Every feature listed in the packet retains taxonomy metadata.
- Every class pair in the active study manifests has at least one identifiability row in the generated packet.
- Advanced-method sections explicitly state go/no-go status and justification.
- The 3D transition section explicitly separates:
  - `dimension_agnostic`
  - `adapter_compatible`
  - `rewrite_required`

Artifacts / Config:
- `docs/protocols/feature_class_classifier_analysis_protocol.md`
- `experiments/corpus_objectives/common_1d_corpus_objectives.yaml`
- `artifacts/study_candidate_generation/study_candidate_schema.json`
- `artifacts/study_candidate_generation/generated_study_candidates.json`
- `artifacts/study_candidate_generation/static_candidate_scores.csv`
- `artifacts/study_candidate_generation/promoted_candidates.csv`
- `artifacts/study_candidate_generation/rejected_candidates.csv`
- `artifacts/study_candidate_generation/monte_carlo_candidate_scores.csv`
- `artifacts/study_candidate_generation/candidate_decision_report.md`
- `artifacts/corpus_autodevelopment_v1/corpus_objectives.yaml`
- `artifacts/corpus_autodevelopment_v1/candidate_corpus_manifest.csv`
- `artifacts/corpus_autodevelopment_v1/corpus_candidate_scores.csv`
- `artifacts/corpus_autodevelopment_v1/selected_corpus_manifest.json`
- `artifacts/corpus_autodevelopment_v1/rejected_corpus_manifest.csv`
- `artifacts/corpus_autodevelopment_v1/corpus_pareto_front.csv`
- `artifacts/corpus_autodevelopment_v1/corpus_adequacy_comparison.csv`
- `artifacts/corpus_autodevelopment_v1/feature_excitation_comparison.csv`
- `artifacts/corpus_autodevelopment_v1/leakage_comparison.csv`
- `artifacts/corpus_autodevelopment_v1/corpus_autodevelopment_report.md`
- `artifacts/validation_ladder/validation_ladder_schema.json`
- `artifacts/validation_ladder/validation_ladder_scores.csv`
- `artifacts/validation_ladder/validation_ladder_decisions.csv`
- `artifacts/validation_ladder/validation_ladder_report.md`
- `artifacts/bayesian_walkthroughs/bayesian_evidence_walkthrough_report.md`
- `artifacts/bayesian_walkthroughs/bayesian_step_tables.csv`
- `artifacts/bayesian_walkthroughs/prior_sweep_examples.csv`
- `artifacts/bayesian_walkthroughs/feature_contribution_examples.csv`
- `artifacts/bayesian_walkthroughs/posterior_flip_thresholds.csv`
- `docs/latex/kinematic_classifier_methodology.tex`
- `docs/latex/figures/`
- `docs/latex/tables/`
- `artifacts/latex/kinematic_classifier_methodology.pdf`
- `artifacts/latex/kinematic_classifier_methodology.tex`
- `artifacts/latex/algorithm_ladder_table.tex`
- `artifacts/latex/bayesian_update_walkthrough_table.tex`
- `artifacts/latex/toy_problem_summary_table.tex`
- `artifacts/latex/study_candidate_generation_algorithm.tex`
- `artifacts/showcase/proof_gallery.md`
- refreshed `artifacts/showcase/index.md`
- refreshed `artifacts/team_packet/`
- helper scripts expected to appear, expand, or be refreshed:
  - `scripts/build_showcase.py`
  - `scripts/build_gallery.py`
  - `scripts/export_team_packet.py`
  - `scripts/validate_artifacts.py`
  - `scripts/audit_corpus.py`
  - `scripts/audit_dimensions.py`
  - new study/corpus generation helpers as needed

Dependencies:
- `PLN-007` common experiment harness
- `PLN-009` feature-set and class-pair comparison
- `PLN-010` generic inference contract
- `PLN-011` generic feature taxonomy
- `PLN-012` classification evidence proof
- `PLN-013` generic filtering contract
- `PLN-014` dimensional lift audit
- `PLN-015` corpus coverage framework
- `PLN-016` team-facing methodology showcase
- existing abstract inspection bundle, corpus adequacy artifacts, coverage reports, and advanced-filter decision surfaces

Recommended Milestone Order:
- `M18`: Analysis protocol and candidate schema
- `M19`: Automatic corpus development
- `M20`: Automatic Feature + Class + Classifier study generation
- `M21`: Static/statistical validation ladder
- `M22`: Bayesian evidence walkthrough suite
- `M23`: LaTeX algorithm methodology document
- `M24`: Proof gallery and showcase refresh

Milestone Deliverables:
- `M18`: Analysis protocol and candidate schema
  - `docs/protocols/feature_class_classifier_analysis_protocol.md`
  - `artifacts/study_candidate_generation/study_candidate_schema.json`
  - `artifacts/validation_ladder/validation_ladder_schema.json`
  - Exit criterion: a new proposal can be represented as a machine-readable study candidate and evaluated through a checklist protocol.
- `M19`: Automatic corpus development
  - corpus objectives config
  - candidate corpus tables
  - selected-corpus and Pareto-front artifacts
  - Exit criterion: the repo can generate multiple corpus candidates and select a better one by adequacy objectives.
- `M20`: Automatic study proposal generation
  - generated candidates, static scores, promoted/rejected tables, and candidate decision report
  - Exit criterion: the repo can propose and score Feature + Class + Classifier combinations automatically.
- `M21`: Static/statistical validation ladder
  - ladder scores, ladder decisions, and ladder report
  - Exit criterion: every major study gets a `promote/revise/reject/defer` decision.
- `M22`: Bayesian evidence walkthrough suite
  - Bayesian walkthrough report, example tables, prior sweeps, feature contribution examples, flip-threshold tables, and plots
  - Exit criterion: a team member can inspect one representative trajectory and understand why the classifier moved from prior to posterior.
- `M23`: LaTeX algorithm methodology document
  - `docs/latex/kinematic_classifier_methodology.tex`
  - `artifacts/latex/kinematic_classifier_methodology.pdf`
  - Exit criterion: the document explains the full methodology, toy-problem evidence, algorithm ladder, Bayesian update process, advanced-filter gates, and 3D transition.
- `M24`: Proof gallery and showcase refresh
  - `artifacts/showcase/proof_gallery.md`
  - refreshed packet index and team packet
  - Exit criterion: the showcase is organized by explicit claims and evidence rather than only by artifact folders.

Claim-Oriented Proof Gallery Requirements:
- Claim 1: Bayesian update machinery works
  - prior-to-posterior single-step
  - likelihood curves
  - posterior timeline
  - log-odds timeline
  - prior sensitivity sweep
- Claim 2: History helps
  - pointwise vs accumulator posterior timelines
  - accuracy vs prefix length
  - true-class posterior quantiles
- Claim 3: Features matter
  - feature ablation chart
  - feature separation ranking
  - feature distribution by class
  - feature correlation heatmap
  - feature contribution to posterior
- Claim 4: Class pairs have different difficulty
  - pairwise confusion heatmap
  - pairwise overlap heatmap
  - pairwise oracle accuracy
  - duration sensitivity by class pair
- Claim 5: Corpus quality matters
  - corpus adequacy scorecard
  - feature excitation matrix
  - leakage audit
  - candidate corpus comparison
- Claim 6: Filtering helps when dynamics matter
  - Kalman innovation likelihood timeline
  - Kalman vs windowed comparison
  - state estimate vs truth
  - model posterior over time
- Claim 7: Advanced filters require evidence
  - advanced filter decision matrix
  - switching benchmark results
  - PF/RBPF go-no-go table
- Claim 8: 3D transition is planned
  - dimension-lift audit chart
  - feature transfer matrix
  - generic-vs-1D-specific layer diagram

Important Tables To Produce:
- `toy_problem_summary.csv`
  - `toy_problem_id`
  - `purpose`
  - `classes`
  - `features`
  - `classifiers`
  - `priors`
  - `what_it_proves`
  - `key_artifacts`
  - `known_limitations`
- `algorithm_ladder_proof.csv`
  - `level`
  - `algorithm`
  - `new_capability`
  - `assumption_added`
  - `failure_mode_addressed`
  - `toy_problem_evidence`
  - `promotion_status`
- `feature_evidence_table.csv`
  - `feature_name`
  - `feature_group`
  - `history_behavior`
  - `evidence_role`
  - `double_counting_risk`
  - `noise_sensitivity`
  - `duration_sensitivity`
  - `sample_count_sensitivity`
  - `3d_transfer_status`
  - `best_class_pairs`
  - `worst_class_pairs`
- `prior_sensitivity_explanation_table.csv`
  - `study_id`
  - `class_pair`
  - `feature_set`
  - `classifier`
  - `baseline_prior`
  - `flip_fraction`
  - `median_log_prior_shift_to_flip`
  - `most_prior_sensitive_scenario`
  - `interpretation`
- `corpus_candidate_scores.csv`
  - `candidate_id`
  - `balance_score`
  - `boundary_score`
  - `feature_excitation_score`
  - `leakage_penalty`
  - `triviality_penalty`
  - `overall_score`
  - `decision`

Success Criteria:
- A study proposal can be defined, screened, validated, and decided through a repeatable machine-readable process.
- The repo can generate and compare multiple corpus candidates rather than only auditing the current one.
- The repo can generate and score Feature + Class + Classifier candidates statically before running Monte Carlo.
- Every major study in the active stack receives a promotion-style decision.
- A team member can inspect Bayesian intermediate steps for representative trajectories and understand why posteriors moved.
- A LaTeX document exists that explains the meta-process, witness problems, Bayesian update ladder, filtering ladder, and 3D transition path.
- The showcase is organized by claims and evidence, not only by file groups.
- The current 1D problems are documented as witness problems proving distinct layers of the methodology.

Work Order Summary:
- Focus first on proving the meta-process, not on adding new classifier families.
- Prioritize:
  - protocol
  - corpus autodevelopment
  - study candidate generation
  - validation ladder
  - Bayesian walkthroughs
  - LaTeX exposition
  - proof-gallery refresh
- Treat the current 1D stack as the witness surface used to prove methodology layers rather than as the final target domain.

Last Updated: 2026-05-24
