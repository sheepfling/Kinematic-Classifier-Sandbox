# Corpus Evaluation and Advanced Exploration

Core question: can the workbench select, audit, or generate hard but valid cases that stress the study and reveal useful failure modes?

This epic owns corpus adequacy, leakage, class validity, feature excitation, backend capability, quality-diversity search, CEM, PPO, and downstream diagnostic yield. Hard examples are only valuable when they remain valid, non-leaky, adequately covered, and useful for study decisions.

Primary artifacts:

- `artifacts/packets/corpus_exploration_mvp/README.md`
- `artifacts/packets/corpus_exploration_mvp/decision_card.md`
- `artifacts/generic_corpus_exploration/candidate_scores.csv`
- `artifacts/trajectory_exploration_rl/ppo_vs_cem_boundary_control/aggregate_metrics_by_backend.csv`
- `artifacts/rl_corpus_agent/rl_backend_decision_summary.json`

Main chart set:

- `03_corpus_candidate_frontier`
- `21_search_backend_comparison_frontier`
- `27_novelty_to_filter_escalation_bridge`

Claim boundary: CEM and PPO are evaluated as corpus-search backends, not magic generators. PPO remains experimental unless baseline comparison, ablation, seed stability, and downstream diagnostic yield all clear.

Decision language:

- `selected_corpus_supported`
- `revise_corpus_policy`
- `route_hard_pair_to_ladder`
- `trigger_advanced_filter_candidate`
- `reject_invalid_hard_case`

