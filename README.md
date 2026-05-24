# kinematic-classifier-sandbox

Survey-first sandbox for kinematic classification methods across maneuver, trajectory, and inertial-motion domains.

## What this repo contains

- `docs/surveys/`: method surveys and framing notes
- `src/kinematic_classifier_sandbox/`: small method catalog and artifact helpers
- `tests/`: standard-library tests for the catalog and artifact exporter
- `scripts/`: repeatable validation and export entrypoints
- `artifacts/`: generated survey outputs that sync via the filesystem but are ignored by git

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

## Cache policy

The intended cache root is:

`/Users/rick/LocalStorage/GIT_LOCAL/active/CACHE/kinematic-classifier-sandbox`

When the repo is promoted into its canonical CloudDocs location, keep caches and
virtual environments outside the sync tree and prefer `PYTHONPYCACHEPREFIX`
pointing at that cache root.

## Survey entrypoints

- Source survey: [kinematic_method_landscape.md](/Users/rick/LocalStorage/GIT_LOCAL/active/kinematic-classifier-sandbox/docs/surveys/kinematic_method_landscape.md)
- Generated artifact: [method_survey_summary.md](/Users/rick/LocalStorage/GIT_LOCAL/active/kinematic-classifier-sandbox/artifacts/method_survey_summary.md)
- Feature workflow: [feature_workflow.md](/Users/rick/Library/Mobile%20Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/docs/surveys/feature_workflow.md)

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

- [feature_analysis.py](/Users/rick/Library/Mobile%20Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/src/kinematic_classifier_sandbox/feature_analysis.py)
- [pca_analysis.py](/Users/rick/Library/Mobile%20Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/src/kinematic_classifier_sandbox/pca_analysis.py)
- [feature_sets.json](/Users/rick/Library/Mobile%20Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/experiments/common_1d_classifier_study/feature_sets.json)
- [class_pair_manifest.json](/Users/rick/Library/Mobile%20Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/experiments/common_1d_classifier_study/class_pair_manifest.json)

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

Use [feature_workflow.md](/Users/rick/Library/Mobile%20Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/docs/surveys/feature_workflow.md) for the full add-feature and rerun process.

The main abstract-inspection landing artifact is:

- [abstract_inspection_index.md](/Users/rick/Library/Mobile%20Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/artifacts/abstract_inspection_v1/abstract_inspection_index.md)
- [abstract_inspection_summary.json](/Users/rick/Library/Mobile%20Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/artifacts/abstract_inspection_v1/abstract_inspection_summary.json)

## Workflow scripts

- `python3 scripts/check_env.py`
- `python3 scripts/all.py`
- `python3 scripts/export_artifacts.py`
- `python3 scripts/run_abstract_inspection.py`
- `python3 scripts/inspect_abstract_recommendations.py`
- `python3 scripts/run_milestone.py list`
- `python3 scripts/run_milestone.py m1-m9 --output-dir artifacts`
- `python3 scripts/run_study.py experiments/common_1d_classifier_study/common_experiment_config.yaml --output-dir artifacts`
- `python3 scripts/dev.py`

## Milestone reruns

For milestone-oriented reruns and documentation refreshes, use:

```bash
python3 scripts/run_milestone.py list
python3 scripts/run_milestone.py m6 --output-dir artifacts
python3 scripts/run_milestone.py m1-m9 --output-dir artifacts
```

See [milestones.md](/Users/rick/Library/Mobile%20Documents/com~apple~CloudDocs/GIT/kinematic-classifier-sandbox/docs/milestones.md) for the per-milestone mapping from command to artifact directory.

## Study reruns

For the `M10` common experiment harness, run a study directly from its config:

```bash
python3 scripts/run_study.py experiments/common_1d_classifier_study/common_experiment_config.yaml --output-dir artifacts
```

That writes the unified study run directory declared by the config and makes the study adapter choice explicit through `study_adapter`.
