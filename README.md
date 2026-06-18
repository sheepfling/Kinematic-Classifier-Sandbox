# Kinematic Classifier Sandbox

This repository is a config-driven kinematic-classification workbench. It first audits whether a feature/class/prior setup is statically admissible, then searches or selects corpora, compares classifier/filter evidence providers on a shared posterior contract, and exports decision packets for technical review.

The current 1D problems are witness problems, not final deployment corpora. They prove methodology layers before the framework is lifted to 3D PVA architectures.

For the longer narrative, start with [docs/story/00_repo_story.md](docs/story/00_repo_story.md).

Start here:

- [Documentation front door](docs/README.md)
- [Experiment front door](experiments/README.md)
- [Package map](src/kinematic_classifier_sandbox/README.md)
- [Test suite map](tests/README.md)
- [Canonical repo story](docs/story/00_repo_story.md)
- [Canonical reading order](docs/story/02_reading_order.md)
- [Claim evidence matrix](docs/story/claim_evidence_matrix.md)

## One pipeline, five lanes

| Lane | Purpose | First presentable artifact |
| --- | --- | --- |
| Static admissibility | Analyze feature/class/prior sufficiency before corpus or classifier work. | `artifacts/packets/static_admissibility_mvp/` |
| Advanced classifier/filter survey | Organize methods by capability, failure mode, and evidence contract. | `docs/story/algorithm_ladder.md` |
| Corpus evaluation/exploration | Generate, select, audit, and search corpora for valid hard cases. | `artifacts/presentation_hero_charts_v4/` |
| Presentation blend | Export the integrated story for a specific audience. | `main_deck.pptx` + `appendix_deck.pptx` |
| Package utility | Keep the workbench installable, runnable, validated, and reusable. | CLI/API docs + tests |

The C2/tracking presentation is an export profile over this pipeline, not a separate code path.

## Quick Start

```bash
python3 -m pip install -e '.[dev]'
python3 -m pip install -e '.[classifiers]'
python3 scripts/check.py
PYTHONPATH=src python3 -m kinematic_classifier_sandbox run-static-audit experiments/static_admissibility/common_1d_static_audit.yaml --output-dir artifacts/packets/static_admissibility_mvp
PYTHONPATH=src python3 -m kinematic_classifier_sandbox validate-packet artifacts/packets/static_admissibility_mvp
```

## Workbench Quickstart

Regenerate the Epic 1 evidence set and presentation showcase from declared inputs:

```bash
PYTHONPATH=src python3 -m kinematic_classifier_sandbox build-epic1-showcase \
  --output-dir artifacts/epic1_showcase \
  --presentation-output-dir artifacts/presentation_hero_charts_v4
```

The command emits a regenerated workbench run, workbench packet, governed CEM/PPO corpus-search lane, static-admissibility packets, V4 presentation packet, manifest, validation summary, and artifact index. See [docs/workflows/epic1_showcase_regeneration.md](docs/workflows/epic1_showcase_regeneration.md) for the full workflow and a fast smoke command.

Run a declared study into the standard workbench artifact shape:

```bash
PYTHONPATH=src python3 -m kinematic_classifier_sandbox validate-study \
  experiments/common_1d_classifier_study/common_experiment_config.yaml

PYTHONPATH=src python3 -m kinematic_classifier_sandbox run-study \
  experiments/common_1d_classifier_study/common_experiment_config.yaml \
  --output-dir artifacts/runs/interview_demo

PYTHONPATH=src python3 -m kinematic_classifier_sandbox analyze-run \
  --run-dir artifacts/runs/interview_demo

PYTHONPATH=src python3 -m kinematic_classifier_sandbox validate-packet \
  --profile workbench \
  --packet-dir artifacts/runs/interview_demo
```

Export profiles are separate from the run itself:

```bash
PYTHONPATH=src python3 -m kinematic_classifier_sandbox export-packet \
  --profile workbench \
  --run-dir artifacts/runs/interview_demo \
  --output-dir artifacts/workbench_reports/interview_demo

PYTHONPATH=src python3 -m kinematic_classifier_sandbox export-packet \
  --profile presentation \
  --run-dir artifacts/runs/interview_demo \
  --output-dir artifacts/presentation_hero_charts_v4
```

The workbench run is the product. The presentation packet is a public-safe export profile over governed artifacts and evidence tiers.

## Static Admissibility Quickstart

Run the built-in multi-domain 3D static audit demo:

```bash
PYTHONPATH=src python3 -m kinematic_classifier_sandbox run-static-audit-multi-domain-3d \
  --output-dir artifacts/validation_packets/01_static_admissibility_multi_domain_3d
```

Validate the generated packet:

```bash
PYTHONPATH=src python3 -m kinematic_classifier_sandbox validate-packet \
  artifacts/validation_packets/01_static_admissibility_multi_domain_3d
```

The packet includes a decision card, source bundles, source artifacts, hero charts, and a claim boundary. This demo is a notional static feature/class/prior audit, not a full 3D tracking implementation.

## Generated Packets

- Static admissibility MVP: `artifacts/packets/static_admissibility_mvp/`
- Presentation V4 packet: `artifacts/presentation_hero_charts_v4/`
- Repo story packet: `artifacts/repo_story/`

## Repo Map

- `docs/`: narrative front doors, plans, surveys, witness notes, and showcase/story material
- `experiments/`: config-driven study definitions, feature-set manifests, class-pair manifests, and corpus-policy inputs
- `src/kinematic_classifier_sandbox/`: package code, grouped into static admissibility, corpus, inference, validation, advanced filters, registry, methodology, and witnesses
- `scripts/`: runnable entrypoints and grouped helpers under `audit/`, `build/`, `render/`, `run/`, and `workflows/`
- `tests/`: regression suite, mostly one test module per source module or artifact family
- `templates/`: starter manifests and YAML templates for new studies and corpus/class/feature/prior definitions
- `artifacts/`: generated outputs that sync via the filesystem but are ignored by git

If you are new to the repo, read these in order:

1. [docs/README.md](docs/README.md)
2. [experiments/README.md](experiments/README.md)
3. [src/kinematic_classifier_sandbox/README.md](src/kinematic_classifier_sandbox/README.md)
4. [tests/README.md](tests/README.md)

## Current Claim Boundaries

- Static admissibility is an early screen, not a final classifier guarantee.
- Corpus search backends are evaluated by valid hard-case discovery and downstream diagnostic yield.
- Advanced filters are promoted only for named witness regimes, not as global defaults.
- 3D transition is a controlled backend/feature/dynamics lift, not a completed deployment claim.

## Tooling

Use these commands from the repo root:

```bash
python3 -m pip install -e '.[dev]'
python3 -m pip install -e '.[classifiers]'
python3 -m pip install -e '.[rl]'
python3 scripts/check.py
python3 scripts/check.py --fix
python3 scripts/lint.py
python3 scripts/lint.py --fix
python3 scripts/format.py
```

The `--fix` flag applies available Ruff fixes before rerunning checks. Pyright itself is still read-only at the CLI level.

## Corpus Search And RL Witnesses

CEM/PPO novelty-search work lives in the Corpus Explorer lane. PPO is still treated as an experimental sequential-control witness unless baseline comparison, adequacy/leakage checks, and downstream diagnostic yield support a stronger claim.

Use `python3 -m kinematic_classifier_sandbox --help` for runnable corpus-search commands and keep detailed rerun recipes in experiment or developer docs rather than the root front door.

## Cache Policy

Keep caches, virtual environments, and bytecode outside the synced repo tree. When running Python directly, prefer setting `PYTHONPYCACHEPREFIX` to a local cache directory.

The repo root `.venv` is expected to be a symlink into a local cache path, not
an iCloud-hosted virtualenv directory. High file count and large binary wheels
make a synced virtualenv a bad fit for this repo.

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
