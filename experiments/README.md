# Experiment Front Door

The `experiments/` tree is the configuration surface for reproducible studies.

Each directory defines one study family or one reusable input bundle. The code in
`src/kinematic_classifier_sandbox/` should be able to consume these configs
without needing custom one-off wiring in notebooks or ad hoc shell sessions.

## Main study areas

- `common_1d_classifier_study/`
  - canonical 1D cross-method study
  - contains `common_experiment_config.yaml`, `feature_sets.json`,
    `classifier_manifest.json`, and `class_pair_manifest.json`
- `common_1d_boundary_study/`
  - boundary-focused variant of the common 1D study
  - emphasizes hard class-pair comparisons
- `advanced_filters/`
  - focused configs for IMM / PF / RBPF witness or decision studies
- `ladder_witness_suite/`
  - witness suite config for classifier/filter ladder proofs
- `generic_corpus_exploration_weight_sweep/`
  - weight-sweep config for corpus exploration and selection tradeoffs
- `corpus_objectives/`
  - objective definitions for corpus generation and selection
- `corpus_policies/`
  - reusable corpus policy inputs
- `new_study_workflow_demo/`
  - small example config showing the new-study workflow shape

## Typical entrypoints

- Run a full study:
  - `python3 scripts/run_study.py experiments/common_1d_classifier_study/common_experiment_config.yaml --output-dir artifacts`
- Run the abstract inspection bundle:
  - `python3 scripts/run_abstract_inspection.py`
- Render corpus exploration sweep:
  - `python3 scripts/render/render_generic_corpus_exploration_weight_sweep.py --output-dir artifacts --config experiments/generic_corpus_exploration_weight_sweep/generic_corpus_exploration_weight_sweep.yaml`
- Render the ladder witness suite:
  - `python3 scripts/render/render_ladder_witness_suite.py --output-dir artifacts --config experiments/ladder_witness_suite/ladder_witness_suite.yaml`

## Design rule

Add new study definitions here when they are:

- config-driven
- rerunnable
- meant to produce comparable artifacts

Do not put generated outputs in `experiments/`; they belong under `artifacts/`.
