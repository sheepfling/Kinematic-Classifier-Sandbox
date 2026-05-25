# PLN-027 Corpus Policy Hyperparameter Tuning

Status: in_progress
Owner: @codex
Priority: P1
Last Updated: 2026-05-25

## Objective

Turn corpus scoring constants into configurable, auditable policy hyperparameters, then evaluate whether the default corpus policy is stable, useful, and robust under perturbation, ablation, sampler-budget, gate-threshold, and dev/holdout studies.

## Scope

- Define a canonical `CorpusPolicySpec`.
- Add `default_corpus_policy_v1.yaml`.
- Generate `artifacts/corpus_hyperparameter_tuning_v1/`.
- Evaluate policy sensitivity, ablations, selected-set stability, sampler budgets, gate thresholds, dev/holdout behavior, Pareto tradeoffs, and a recommended policy.

## Out of Scope

- Claiming any tuned policy is universally optimal.
- Tuning corpus policies to make a single classifier win.
- Rewriting trajectory generation or classifier algorithms.
- Replacing all historical reports in one pass.

## Implementation Steps

1. Add policy spec, validation, normalization, and schema export.
2. Add default policy config covering corpus autodevelopment, generic explorer, CorpusGym, archive, sampler budgets, and gates.
3. Add policy sweep runner over a fixed candidate/objective surface.
4. Add ablation and perturbation studies with Jaccard and rank stability metrics.
5. Add sampler-budget and gate-threshold sweeps.
6. Add dev/holdout validation.
7. Generate recommended policy and report.
8. Wire tests and run regression.

## Validation

- Default policy config reproduces the current generic explorer baseline scores.
- Invalid policy weights fail validation.
- Sweep, ablation, stability, sampler-budget, gate-threshold, and dev/holdout outputs are nonempty.
- Recommended policy references an evaluated policy ID.
- Full regression passes.

## Artifacts / Config

- `experiments/corpus_policies/default_corpus_policy_v1.yaml`
- `artifacts/corpus_hyperparameter_tuning_v1/weight_spec_schema.json`
- `artifacts/corpus_hyperparameter_tuning_v1/default_weight_spec.yaml`
- `artifacts/corpus_hyperparameter_tuning_v1/sweep_design.csv`
- `artifacts/corpus_hyperparameter_tuning_v1/sweep_results.csv`
- `artifacts/corpus_hyperparameter_tuning_v1/ablation_results.csv`
- `artifacts/corpus_hyperparameter_tuning_v1/local_perturbation_results.csv`
- `artifacts/corpus_hyperparameter_tuning_v1/selected_set_jaccard.csv`
- `artifacts/corpus_hyperparameter_tuning_v1/rank_stability.csv`
- `artifacts/corpus_hyperparameter_tuning_v1/sampler_budget_sweep.csv`
- `artifacts/corpus_hyperparameter_tuning_v1/gate_threshold_sweep.csv`
- `artifacts/corpus_hyperparameter_tuning_v1/dev_holdout_results.csv`
- `artifacts/corpus_hyperparameter_tuning_v1/pareto_front.csv`
- `artifacts/corpus_hyperparameter_tuning_v1/recommended_policy.yaml`
- `artifacts/corpus_hyperparameter_tuning_v1/corpus_hyperparameter_tuning_report.md`

## Dependencies

- Existing generic corpus exploration candidate pool.
- Existing corpus autodevelopment, CorpusGym, QD archive, and study-candidate scoring semantics.

