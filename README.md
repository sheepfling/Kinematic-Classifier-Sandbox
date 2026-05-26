# kinematic-classifier-sandbox

Methodology workbench for kinematic classification studies.

This repository is a reusable kinematic-classification framework, not a single toy benchmark. It defines studies through contracts and manifests, generates or selects corpora that exercise intended boundaries, runs multiple classifier and filter families on the same evidence surface, and audits feature coverage, confusability, leakage, adequacy, and promotion decisions. The current 1D problems are witness problems, not final deployment corpora; they live under `kinematic_classifier_sandbox.witnesses`.

For the longer narrative, start with [docs/story/00_repo_story.md](docs/story/00_repo_story.md).

Start here:

- [Documentation front door](docs/README.md)
- [Experiment front door](experiments/README.md)
- [Package map](src/kinematic_classifier_sandbox/README.md)
- [Test suite map](tests/README.md)
- [Canonical repo story](docs/story/00_repo_story.md)
- [Canonical reading order](docs/story/02_reading_order.md)
- [Claim evidence matrix](docs/story/claim_evidence_matrix.md)

## Repo map

- `docs/`: narrative front doors, plans, surveys, witness notes, and showcase/story material
- `experiments/`: config-driven study definitions, feature-set manifests, class-pair manifests, and corpus-policy inputs
- `src/kinematic_classifier_sandbox/`: package code, grouped into inference, corpus, analysis, validation, and advanced-filter layers
- `scripts/`: runnable entrypoints and grouped helpers under `audit/`, `build/`, `render/`, `run/`, and `workflows/`
- `tests/`: regression suite, mostly one test module per source module or artifact family
- `templates/`: starter manifests and YAML templates for new studies and corpus/class/feature/prior definitions
- `artifacts/`: generated outputs that sync via the filesystem but are ignored by git

If you are new to the repo, read these in order:

1. [docs/README.md](docs/README.md)
2. [experiments/README.md](experiments/README.md)
3. [src/kinematic_classifier_sandbox/README.md](src/kinematic_classifier_sandbox/README.md)
4. [tests/README.md](tests/README.md)

## Initial scope

The first phase is intentionally narrow:

- map traditional feature-engineered classifiers
- map model-based kinematic and multiple-model methods
- map modern deep sequence and self-supervised methods
- provide a reusable method catalog that can drive later benchmark and dataset work

The repo does not yet implement:

- dataset ingestion pipelines
- benchmark training loops
- model fitting code
- experiment tracking

## Quick start

```bash
python3 scripts/check_env.py
PYTHONPYCACHEPREFIX=/Users/rick/LocalStorage/GIT_LOCAL/active/CACHE/kinematic-classifier-sandbox/.pycache python3 scripts/all.py
PYTHONPYCACHEPREFIX=/Users/rick/LocalStorage/GIT_LOCAL/active/CACHE/kinematic-classifier-sandbox/.pycache python3 scripts/export_artifacts.py
```

## Tooling

Use these commands from the repo root:

```bash
python3 -m pip install -e '.[dev]'
python3 scripts/check.py
python3 scripts/check.py --fix
python3 scripts/lint.py
python3 scripts/lint.py --fix
python3 scripts/format.py
```

The `--fix` flag applies available Ruff fixes before rerunning checks. Pyright itself is still read-only at the CLI level.

## Cache policy

The intended cache root is:

`/Users/rick/LocalStorage/GIT_LOCAL/active/CACHE/kinematic-classifier-sandbox`

When the repo is promoted into its canonical CloudDocs location, keep caches and
virtual environments outside the sync tree and prefer `PYTHONPYCACHEPREFIX`
pointing at that cache root.

## Survey entrypoints

- Source survey: [kinematic_method_landscape.md](docs/surveys/kinematic_method_landscape.md)
- Generated artifact: [method_survey_summary.md](artifacts/method_survey_summary.md)
- Feature workflow: [feature_workflow.md](docs/surveys/feature_workflow.md)

## Feature workflow

The generic feature-analysis stack is now:

- registry-backed
- feature-set aware
- PCA-aware
- corpus-adequacy-aware

The context model is also split intentionally:

- `BaseFeatureComputationContext` for generic trajectory/time-series structure
- `OneDimensionalFeatureComputationContext` for current 1D-derived signals

That keeps the current 1D feature library intact while making the extension point for future 3D feature families explicit.

Main files:

- [feature_analysis.py](src/kinematic_classifier_sandbox/feature_analysis.py)
- [pca_analysis.py](src/kinematic_classifier_sandbox/pca_analysis.py)
- [feature_sets.json](experiments/common_1d_classifier_study/feature_sets.json)
- [class_pair_manifest.json](experiments/common_1d_classifier_study/class_pair_manifest.json)

Quick examples:

```python
from kinematic_classifier_sandbox import analyze_feature_datasets, analyze_feature_pca

feature_result = analyze_feature_datasets(feature_set="shape_window")
pca_result = analyze_feature_pca(feature_set="model_residuals", n_components=2)
```

```python
from kinematic_classifier_sandbox import analyze_feature_datasets

feature_result = analyze_feature_datasets(
    feature_names=("position_range", "speed_range", "linear_fit_residual"),
)
```

Use [feature_workflow.md](docs/surveys/feature_workflow.md) for the full add-feature and rerun process.

The main abstract-inspection landing artifact is:

- [abstract_inspection_index.md](artifacts/abstract_inspection_v1/abstract_inspection_index.md)
- [abstract_inspection_summary.json](artifacts/abstract_inspection_v1/abstract_inspection_summary.json)

## Workflow scripts

- `python3 scripts/check_env.py`
- `python3 scripts/all.py`
- `python3 scripts/export_artifacts.py`
- `python3 scripts/run_abstract_inspection.py`
- `python3 scripts/inspect_abstract_recommendations.py`
- `python3 scripts/run_milestone.py list`
- `python3 scripts/run_milestone.py m1-m9 --output-dir artifacts`
- `python3 scripts/run_study.py experiments/common_1d_classifier_study/common_experiment_config.yaml --output-dir artifacts`
- `python3 scripts/render/render_generic_corpus_exploration_weight_sweep.py --output-dir artifacts --config experiments/generic_corpus_exploration_weight_sweep/generic_corpus_exploration_weight_sweep.yaml`
- `python3 scripts/render/render_formal_math_visual_registry.py --output-dir artifacts`
- `python3 scripts/render/render_ladder_witness_suite.py --output-dir artifacts --config experiments/ladder_witness_suite/ladder_witness_suite.yaml`
- `python3 scripts/dev.py`

## Milestone reruns

For milestone-oriented reruns and documentation refreshes, use:

```bash
python3 scripts/run_milestone.py list
python3 scripts/run_milestone.py m6 --output-dir artifacts
python3 scripts/run_milestone.py m1-m9 --output-dir artifacts
```

See [milestones.md](docs/milestones.md) for the per-milestone mapping from command to artifact directory.

## Study reruns

For the `M10` common experiment harness, run a study directly from its config:

```bash
python3 scripts/run_study.py experiments/common_1d_classifier_study/common_experiment_config.yaml --output-dir artifacts
```

That writes the unified study run directory declared by the config and makes the study adapter choice explicit through `study_adapter`.
