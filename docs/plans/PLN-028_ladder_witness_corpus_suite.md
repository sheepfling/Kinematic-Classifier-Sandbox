# PLN-028 Ladder Witness Corpus Suite

Title: Ladder Witness Corpus Suite
Plan ID: PLN-028
Status: proposed
Owner: @codex
Priority: P1
Last Updated: 2026-05-25

## Objective

Build a suite of controlled 1D witness corpora that proves, visually and quantitatively, when each rung of the classifier ladder is sufficient and when it is insufficient. The suite should stop describing these as toy examples and instead frame them as 1D ladder witness corpora: controlled data sets whose job is to prove a specific methodological claim.

## Scope

- Define a formal witness schema, manifest, and claim matrix for the classifier ladder.
- Create paired sufficiency and insufficiency witnesses for each rung under test.
- Generate corpus, feature, posterior, and method-comparison artifacts for every witness.
- Produce visual proof bundles that make sufficiency, failure, and escalation obvious.
- Tie each witness to the evidence stack already present in the repo: corpus adequacy, feature separability, posterior quality, prior sensitivity, confusion, and advanced-filter justification.
- Express the ladder as an evidence sequence, not a leaderboard.

## Out of Scope

- Claiming any rung is universally optimal.
- Treating witness corpora as generic benchmarks detached from a methodological claim.
- Replacing the existing ladder or corpus evaluation stack.
- Adding 3D trajectory engines as part of this plan.
- Implementing new advanced filters before a lower-rung failure justifies them.

## Implementation Steps

1. Define the witness schema and manifest surface.
   - Add `witness_schema.json`.
   - Add `witness_manifest.json`.
   - Add `rung_claim_matrix.csv`.
   - Encode claim type, rung under test, methods, priors, objective, expected result, and success criteria.
2. Implement sufficiency/insufficiency gates.
   - Require declared corpus adequacy and class-validity preconditions.
   - Define gates for accuracy, prior fragility, switch delay, state RMSE, NLL, Brier, ECE, ESS, and leakage where relevant.
   - Separate “insufficient because the corpus is weak” from “insufficient because the rung is wrong.”
3. Build the witness corpora and per-witness artifact directories.
   - Add the required witness families `W00` through `W12`.
   - Emit the required CSV, JSON, markdown, and plot artifacts for each witness.
   - Preserve a consistent directory structure so every witness is easy to rerun and inspect.
4. Produce the ladder comparison and sufficiency/failure matrices.
   - Add `rung_sufficiency_matrix.csv`.
   - Add `rung_failure_matrix.csv`.
   - Add `ladder_method_comparison.csv`.
   - Summarize the promotion or escalation logic per rung.
5. Build the visual proof gallery.
   - Add `ladder_visual_index.md`.
   - Add overview plots and per-witness plots.
   - Include trajectory examples, feature traces, posterior timelines, log-evidence timelines, confusion matrices, method comparisons, and interpretation cards.
6. Add reports and docs.
   - Add `witness_suite_report.md`.
   - Add `rung_by_rung_report.md`.
   - Add `sufficiency_and_failure_report.md`.
   - Add `advanced_filter_justification_report.md`.
7. Wire tests, runners, and exports.
   - Add config validation and artifact-existence tests.
   - Export the new surfaces through the package namespace and artifact runner.
   - Keep the suite rerunnable from YAML-configured witnesses rather than hardcoded one-off code paths.

## Validation

- Every witness config validates against the witness schema.
- Every witness declares at least one sufficiency or insufficiency claim.
- Every witness writes a corpus manifest, feature matrix, posterior history, method metrics, and at least one corpus-quality diagnostic.
- The sufficiency and insufficiency gates run automatically and produce reproducible decisions.
- The ladder comparison uses shared trajectory IDs and shared output contracts across methods.
- Every plot referenced in the visual index exists.
- The full regression suite remains green after wiring the witness suite into the artifact pipeline.

## Artifacts / Config

- `artifacts/ladder_witness_suite_v1/`
- `artifacts/ladder_witness_suite_v1/index.md`
- `artifacts/ladder_witness_suite_v1/witness_manifest.json`
- `artifacts/ladder_witness_suite_v1/witness_schema.json`
- `artifacts/ladder_witness_suite_v1/rung_claim_matrix.csv`
- `artifacts/ladder_witness_suite_v1/rung_sufficiency_matrix.csv`
- `artifacts/ladder_witness_suite_v1/rung_failure_matrix.csv`
- `artifacts/ladder_witness_suite_v1/ladder_method_comparison.csv`
- `artifacts/ladder_witness_suite_v1/ladder_visual_index.md`
- `artifacts/ladder_witness_suite_v1/reports/witness_suite_report.md`
- `artifacts/ladder_witness_suite_v1/reports/rung_by_rung_report.md`
- `artifacts/ladder_witness_suite_v1/reports/sufficiency_and_failure_report.md`
- `artifacts/ladder_witness_suite_v1/reports/advanced_filter_justification_report.md`
- `artifacts/ladder_witness_suite_v1/datasets/`
- `artifacts/ladder_witness_suite_v1/plots/`
- `experiments/ladder_witness_suite/`

## Dependencies

- `corpus_adequacy_audit.py`
- `coverage_report.py`
- `feature_analysis.py`
- `pca_analysis.py`
- `prior_sensitivity_analysis.py`
- `inspection_bundle.py`
- `validation_ladder.py`
- `common_experiment_harness.py`
- `transition_matrix_accumulator.py`
- `kalman_filter_bank.py`
- `advanced_state_inference.py`
- `advanced_filters/`
- `corpus_policy.py`
- `corpus_policy_sweep.py`
- `generic_corpus_exploration.py`
- `repo_story.py`
- `showcase_builder.py`

## Milestones

### M69: Witness suite schema and manifest

- Deliver `witness_schema.json`, `witness_manifest.json`, and `rung_claim_matrix.csv`.
- Exit criterion: every rung has at least one sufficient and one insufficient witness declared.

### M70: Pointwise and windowed witnesses

- Deliver `W00_pointwise_separable`, `W01_pointwise_insufficient_temporal`, `W02_windowed_extrema_sufficient`, and `W03_windowed_insufficient_dynamics`.
- Exit criterion: pointwise and windowed failure/sufficiency claims are backed by plots and metrics.

### M71: Sequential Bayes and transition witnesses

- Deliver `W04_sequential_bayes_sufficient`, `W05_sequential_bayes_insufficient_switching`, and `W08_transition_sufficient_finite_switching`.
- Exit criterion: posterior accumulation, prior sensitivity, and switching lag are visually explained.

### M72: Kalman witnesses

- Deliver `W06_kalman_sufficient_linear_dynamics` and `W07_kalman_insufficient_nonlinear_or_switching`.
- Exit criterion: Kalman innovation evidence is shown sufficient for linear dynamics and insufficient for at least one nonlinear, non-Gaussian, or switching case.

### M73: Advanced filter witnesses

- Deliver `W10_imm_sufficient_switching_linear_modes`, `W11_pf_sufficient_nonlinear_nongaussian`, and `W12_rbpf_sufficient_latent_mode`.
- Exit criterion: advanced filters are justified by lower-rung failures, not simply added as complexity.

### M74: Ladder comparison harness

- Deliver `ladder_method_comparison.csv`, `posterior_history_by_method.csv`, `confusion_by_method.csv`, `rung_sufficiency_matrix.csv`, and `rung_failure_matrix.csv`.
- Exit criterion: all rungs run through the same output and evaluation contract.

### M75: Visual proof gallery

- Deliver `ladder_visual_index.md` and a claim-oriented proof gallery.
- Exit criterion: each witness has trajectory, feature, posterior, evidence, confusion, comparison, and corpus-quality visuals.

### M76: Witness report and LaTeX appendix

- Deliver `witness_suite_report.md`, `docs/latex/ladder_witness_suite.tex`, and `artifacts/ladder_witness_suite_v1/ladder_witness_suite.pdf`.
- Exit criterion: a reader can understand when each rung is sufficient and insufficient without reading code.
