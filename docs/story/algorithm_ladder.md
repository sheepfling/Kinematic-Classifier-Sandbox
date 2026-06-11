# Algorithm Ladder

The classifier/filter ladder organizes evidence providers by the capability they add.

| Rung | Algorithm | Evidence source | Adds | Failure addressed | 1D witness | Status |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Pointwise | `log p(y_t | c)` | local likelihood | no baseline | pointwise_overlap | promote |
| 1 | Windowed | `log p(phi_t | c)` | local history | outliers and noise | windowed_outlier_extrema | revise/promote by case |
| 2 | Sequential Bayes | recursive evidence | memory | pointwise ignores history | sequential_history | promote |
| 3 | Kalman bank | innovation likelihood | dynamics | endpoint ambiguity | kalman_endpoint_match | promote |
| 4 | Transition matrix | `T_ij` mode transition | switching | static class assumption | transition_switching | pass |
| 5 | IMM | switching-aware state inference | switching trajectories with shared evidence contract | demonstrated switching failures | imm_switching_v1 | witness_supported / study_justified by case |
| 6 | PF | nonlinear particle evidence | nonlinear/non-Gaussian state evidence | single-Gaussian projection collapses a multimodal posterior, or mean-reverting stochastic dynamics require sampled state evidence | pf_abs_range_multimodal_oracle_v1, ornstein_uhlenbeck_mean_reversion_1d | witness_supported / study_justified by case |
| 7 | RBPF | sampled mode path + conditional Kalman state | mixed discrete/continuous inference | latent maneuver onset where PF-vs-RBPF frontier shows structure-aware efficiency or accuracy advantage | latent_maneuver_onset_1d, pf_vs_rbpf_frontier | witness_supported / study_justified by case |

The ladder rule is simple: each rung must be justified by a failure mode that the previous rung cannot explain or solve. IMM, PF, and RBPF are now implemented as advanced-filter evidence providers and promoted on their targeted witness failures. The OU-style witness lives inside the PF branch as a concrete mean-reverting stochastic case, not as a separate rung. Those promotions are witness-specific; they do not claim universal dominance over simpler rungs.

The comparison rule is now hybrid rather than naive: PF, RBPF, and the OU witness are first-class members of the shared classifier family at the registry and method-manifest level, but broad scorecards are capability-aware. Shared binary-corpus tables score only methods whose manifests declare `shared_binary_dynamics` support; advanced methods stay visible with explicit `witness_only` or `not_applicable` status plus their supporting witness artifact.

The advanced-method architecture and research notes are maintained under `docs/methods/`. The generated status surface is `artifacts/advanced_filter_comparison_v1/advanced_method_gate_matrix.csv`, which separates `implemented`, `witness_supported`, `justified_for_study`, and `generalized` so a witness promotion cannot silently become a universal claim.

The governance layer for deciding whether a rung is sufficient, near its practical limit, or ready to escalate is `PLN-026 Rung Sufficiency And Escalation Evaluator`. That evaluator is the bridge between the ladder and the advanced-filter gates: it checks corpus adequacy, feature learnability, oracle gap, posterior quality, failure mode, and capability match before it permits escalation.

The shared contract is that each rung provides class evidence that can be accumulated into comparable posterior histories and evaluated under the same prior, calibration, confusion, and promotion rules.

This ladder is intentionally narrower than the repo's broader algorithm map. The wider lane registry lives in [algorithm_map.md](/Users/rick/Library/Mobile%20Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/docs/story/algorithm_map.md) and its generated artifact bundle under `artifacts/algorithm_coverage_matrix_v1/`. That surface tracks benchmark time-series classifiers, supervised and unsupervised learning evidence providers, neural sequence baselines, learned filters, calibration wrappers, optimizer/generator methods, and future multi-target tracking extensions without collapsing them into the proof ladder prematurely.
