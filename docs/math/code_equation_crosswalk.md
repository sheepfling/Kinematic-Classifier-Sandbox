# Code-to-Equation Crosswalk

| Equation / concept | What it computes | Code path | Test path | Artifact path |
| --- | --- | --- | --- | --- |
| Bayes recursive update | Posterior history under sequential evidence | `src/kinematic_classifier_sandbox/sequential_bayes_accumulator.py` | `tests/test_corpus_classifier_scoring.py` | `artifacts/corpus_classifier_scoring/posterior_history.csv` |
| Two-class log odds / prior flip | Prior sensitivity and flip threshold | `src/kinematic_classifier_sandbox/bayesian_walkthroughs.py` | `tests/test_bayesian_walkthroughs.py` | `artifacts/bayesian_walkthroughs/posterior_flip_thresholds.csv` |
| Transition-matrix update | Switching-aware posterior propagation | `src/kinematic_classifier_sandbox/transition_matrix_accumulator.py` | `tests/test_transition_matrix_accumulator.py` | `artifacts/transition_matrix_accumulator_v1/transition_matrix_accumulator_report.md` |
| Kalman innovation likelihood | Model-based evidence increment per class | `src/kinematic_classifier_sandbox/kalman_filter_bank.py` | `tests/test_corpus_classifier_scoring.py` | `artifacts/corpus_classifier_scoring/classifier_candidate_scores.csv` |
| Corpus autodevelopment scalar score | Corpus adequacy ranking | `src/kinematic_classifier_sandbox/corpus_autodevelopment.py` | `tests/test_corpus_autodevelopment.py` | `artifacts/corpus_autodevelopment_v1/candidate_scores.csv` |
| Pareto dominance | Non-dominated corpus alternatives | `src/kinematic_classifier_sandbox/corpus_autodevelopment.py` | `tests/test_corpus_autodevelopment.py` | `artifacts/corpus_autodevelopment_v1/pareto_front.csv` |
| CorpusGym reward | Targeted trajectory search utility | `src/kinematic_classifier_sandbox/corpus_gym.py` | `tests/test_corpus_gym.py` | `artifacts/corpus_gym/corpus_gym_report.md` |
| QD archive utility and cell mapping | Elite selection over archive cells | `src/kinematic_classifier_sandbox/objective_driven_qd_archive.py` | `tests/test_objective_driven_qd_archive.py` | `artifacts/quality_diversity_corpus_v1/archive_elites.csv` |
| Candidate sampler mixture | Objective-driven candidate proposal family | `src/kinematic_classifier_sandbox/candidate_generation.py` | `tests/test_candidate_generation.py` | `artifacts/candidate_generation/generated_candidates.csv` |
| Class-validity status logic | Valid / ambiguous / relabel / invalid partitioning | `src/kinematic_classifier_sandbox/class_validity.py` | `tests/test_class_validity.py` | `artifacts/class_validity/class_validity_scores.csv` |
