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

## Workflow scripts

- `python3 scripts/check_env.py`
- `python3 scripts/all.py`
- `python3 scripts/export_artifacts.py`
- `python3 scripts/dev.py`
