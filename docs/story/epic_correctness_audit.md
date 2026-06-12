# Epic Correctness Audit

This page answers one narrow question: are the Epic 1, Epic 2, and Epic 3 algorithm surfaces present, tested, and claim-bounded?

The short answer is yes for coverage, with narrower claim boundaries for promotion.

## Epic 1: Static Admissibility

Status: covered for correctness.

What is present:

- static feature/class/prior admissibility audit
- class confusability, feature relevance, redundancy, synergy, prior pathology, leakage
- packet and claim-boundary validation

Code and tests:

- `src/kinematic_classifier_sandbox/static_admissibility/*`
- `src/kinematic_classifier_sandbox/analysis/static_feature_class_prior_audit.py`
- `tests/static_admissibility/test_static_admissibility_packet.py`
- `tests/analysis/test_static_feature_class_prior_audit.py`
- `tests/validation/test_correctness.py`

Artifacts:

- `artifacts/packets/static_admissibility_mvp/decision_card.md`
- `artifacts/packets/static_admissibility_mvp/static_audit_report.md`
- `artifacts/packets/static_admissibility_mvp/class_confusability_matrix.csv`
- `artifacts/packets/static_admissibility_mvp/feature_redundancy_matrix.csv`
- `artifacts/packets/static_admissibility_mvp/feature_synergy_candidates.csv`
- `artifacts/packets/static_admissibility_mvp/prior_pathology_report.csv`
- `artifacts/packets/static_admissibility_mvp/static_leakage_provenance_audit.csv`

Claim boundary:

- static admissibility is a gate, not a classifier benchmark
- synergy remains candidate-level until downstream ablation confirms it

## Epic 2: Classifier / Filter Ladder

Status: covered for method-surface correctness, but not complete for
family-level closure.

What is present:

- pointwise, windowed, sequential Bayes, Kalman bank, transition matrix
- IMM, PF, and RBPF
- shapelet, modern-TSC archive, neural sequence, and TS2Vec witness surfaces
- shared posterior/evidence contract
- witness-supported advanced-filter gate logic
- family-level maturity and closure surfaces

Code and tests:

- `src/kinematic_classifier_sandbox/inference/*`
- `src/kinematic_classifier_sandbox/advanced_filters/*`
- `tests/inference/test_pointwise_baseline.py`
- `tests/inference/test_windowed_baseline.py`
- `tests/inference/test_sequential_bayes_accumulator.py`
- `tests/inference/test_kalman_filter_bank.py`
- `tests/inference/test_transition_matrix_accumulator.py`
- `tests/advanced_filters/test_advanced_filter_contract.py`
- `tests/advanced_filters/test_oracle_1d.py`
- `tests/advanced_filters/test_oracle_pf_1d.py`
- `tests/advanced_filters/test_imm_filter.py`
- `tests/advanced_filters/test_particle_filter.py`
- `tests/advanced_filters/test_rbpf.py`
- `tests/analysis/test_shapelet_motif_witness.py`
- `tests/analysis/test_tsc_archive_frontier.py`
- `tests/analysis/test_archive_backend_diagnosis.py`
- `tests/analysis/test_archive_family_promotion_audit.py`
- `tests/analysis/test_neural_sequence_frontier.py`
- `tests/analysis/test_neural_sequence_robustness.py`
- `tests/analysis/test_embedding_baseline_frontier.py`
- `tests/analysis/test_physics_family_promotion_audit.py`
- `tests/validation/test_correctness.py`

Artifacts:

- `artifacts/generic_inference_contract/evidence_provider_schema.json`
- `artifacts/generic_inference_contract/posterior_history_schema.json`
- `artifacts/classification_evidence_proof/evidence_provider_manifest.json`
- `artifacts/method_validation_os_v1/algorithm_promotion_status_matrix.csv`
- `artifacts/method_validation_os_v1/epic2_family_maturity_matrix.csv`
- `artifacts/advanced_filter_comparison_v1/advanced_method_gate_matrix.csv`
- `artifacts/shapelet_maneuver_motif_v1/*`
- `artifacts/tsc_archive_baseline_frontier_v1/*`
- `artifacts/archive_backend_diagnosis_v1/*`
- `artifacts/archive_family_promotion_audit_v1/*`
- `artifacts/neural_sequence_vs_physics_frontier_v1/*`
- `artifacts/neural_sequence_robustness_frontier_v1/*`
- `artifacts/embedding_baseline_frontier_v1/*`
- `artifacts/ts2vec_backend_parity_v1/*`
- `artifacts/physics_family_promotion_audit_v1/*`

Claim boundary:

- the shared ladder and companion classifier surfaces are covered and tested
- Epic 2 is now complete on the bounded 1D classifier-family surface, while
  broader robustness and generalization remain follow-on work
- the interpretable kinematic family is proven on the current 1D witness set
- the physics-aware family is now witness-backed on the current 1D core ladder,
  while the advanced-filter audit remains a bounded blocker-clearance packet
- the generic-TSC family is now witness-backed on the current 1D archive
  surface, while DrCIF remains an explicit partial method-level holdout
- the learned family now has bounded neural robustness and TS2Vec parity
  packets, but broader benchmark breadth remains open
- IMM is witness-supported, not a global default
- PF is study-justified only on the current abs-range multimodal oracle route
- RBPF is now study-justified on the bounded latent witness, not broadly
  promoted outside that claim boundary

## Epic 3: Corpus Explorer

Status: covered for correctness.

What is present:

- corpus policy
- corpus autodevelopment
- CorpusGym
- search/selection/adequacy/leakage
- backend registry and trajectory exploration
- CEM and PPO search backends

Code and tests:

- `src/kinematic_classifier_sandbox/corpus/*`
- `tests/corpus/test_corpus_policy.py`
- `tests/corpus/test_corpus_policy_sweep.py`
- `tests/corpus/test_corpus_gym.py`
- `tests/corpus/test_corpus_autodevelopment.py`
- `tests/corpus/test_corpus_adequacy_audit.py`
- `tests/corpus/test_control_surface_backends.py`
- `tests/corpus/test_trajectory_backend_contract.py`
- `tests/corpus/test_rl_backend_decision.py`
- `tests/corpus/exploration/test_generic_corpus_exploration.py`
- `tests/corpus/exploration/test_generic_corpus_exploration_weight_sweep.py`
- `tests/corpus/trajectory_exploration/test_trajectory_exploration_cli.py`
- `tests/validation/test_correctness.py`

Artifacts:

- `artifacts/generic_corpus_exploration/*`
- `artifacts/corpus_adequacy_audit_v1/*`
- `artifacts/trajectory_exploration_backend_registry_v1/*`
- `artifacts/trajectory_exploration_rl/*`
- `artifacts/packets/corpus_exploration_mvp/*`

Claim boundary:

- Corpus Explorer is a governance and selection layer, not a magic generator
- CEM and PPO are search backends with explicit claim gates

## Shared Packet And Claim Correctness

Status: covered for correctness.

What is present:

- packet validators
- claim/evidence matrices
- presentation packet validators
- correctness ladder command

Code and tests:

- `src/kinematic_classifier_sandbox/validation/correctness.py`
- `src/kinematic_classifier_sandbox/static_admissibility/validation.py`
- `scripts/audit/validate_presentation_hero_packet.py`
- `tests/validation/test_correctness.py`

Artifacts:

- `artifacts/presentation_hero_charts_v5/*`
- `artifacts/repo_story/claim_evidence_matrix.csv`
- `artifacts/repo_story/artifact_manifest.json`

## Bottom Line

All three epic surfaces are present and correctness-covered. The remaining distinctions are not about coverage but about claim boundaries: which methods are witness-supported, which are candidate-diagnostic, and which are only roadmap or architecture.
