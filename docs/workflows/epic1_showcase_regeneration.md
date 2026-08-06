# Epic 1 Showcase Regeneration

This workflow regenerates the Epic 1 evidence set and presentation showcase from declared inputs. It is the easiest way to confirm that Epic 1 is still a usable methodology workbench, not just a static deck export.

## Full Regeneration

Run the standard Epic 1 showcase:

```bash
PYTHONPYCACHEPREFIX=/Users/rick/LocalStorage/GIT_LOCAL/active/CACHE/kinematic-classifier-sandbox/.pycache \
PYTHONPATH=src \
python3 -m kinematic_classifier_sandbox build-epic1-showcase \
  --output-dir artifacts/epic1_showcase \
  --presentation-output-dir artifacts/presentation_hero_charts_v4
```

The command writes:

- `artifacts/epic1_showcase/workbench_run/`
- `artifacts/epic1_showcase/workbench_packet/`
- `artifacts/epic1_showcase/corpus_search/`
- `artifacts/epic1_showcase/static_admissibility/`
- `artifacts/epic1_showcase/static_admissibility_multi_domain_3d/`
- `artifacts/epic1_showcase/regeneration_summary.md`
- `artifacts/epic1_showcase/epic1_showcase_manifest.json`
- `artifacts/epic1_showcase/validation_summary.csv`
- `artifacts/epic1_showcase/artifact_index.csv`
- `artifacts/presentation_hero_charts_v4/`

For the Product 1 pitch, open
`artifacts/epic1_showcase/static_admissibility/executive_brief.md` after the
static suite has regenerated. It is generated from the seven source runs and
summarizes feature-space, class-space, prior-space, and corpus-search-space
findings with their recommended routes.

## Fast Smoke Regeneration

Use this when you only need to check the CLI and workbench path:

```bash
PYTHONPYCACHEPREFIX=/Users/rick/LocalStorage/GIT_LOCAL/active/CACHE/kinematic-classifier-sandbox/.pycache \
PYTHONPATH=src \
python3 -m kinematic_classifier_sandbox build-epic1-showcase \
  --output-dir /private/tmp/kcs_epic1_showcase_smoke \
  --skip-static \
  --skip-presentation \
  --trajectories-per-case 2
```

The smoke command still validates the study spec, regenerates a workbench run, exports a workbench packet, and emits the governed CEM/PPO corpus-search lane.

## Validate Outputs

Validate the workbench packet:

```bash
PYTHONPYCACHEPREFIX=/Users/rick/LocalStorage/GIT_LOCAL/active/CACHE/kinematic-classifier-sandbox/.pycache \
PYTHONPATH=src \
python3 -m kinematic_classifier_sandbox validate-packet \
  --profile workbench \
  --packet-dir artifacts/epic1_showcase/workbench_packet
```

Validate the public presentation packet:

```bash
PYTHONPYCACHEPREFIX=/Users/rick/LocalStorage/GIT_LOCAL/active/CACHE/kinematic-classifier-sandbox/.pycache \
PYTHONPATH=src \
python3 -m kinematic_classifier_sandbox validate-packet \
  --profile presentation \
  --packet-dir artifacts/presentation_hero_charts_v4
```

## Example Inputs

Start from these examples:

- `experiments/common_1d_classifier_study/common_experiment_config.yaml`: default workbench run.
- `experiments/templates/basic_classifier_study.yaml`: reusable classifier-study template.
- `experiments/templates/corpus_search_study.yaml`: governed random/CEM/PPO corpus-search lane.
- `experiments/templates/advanced_filter_witness.yaml`: advanced-filter witness template.
- `experiments/templates/private_work_study.yaml`: private adapter placeholder that stays out of public packets.
- `experiments/static_admissibility/epic1_exemplar_suite.yaml`: static-admissibility exemplar family suite.

## Claim Boundary

The showcase regeneration confirms a repeatable Epic 1 workbench and presentation export profile. It does not claim operational performance, full 3D tracking, or broad promotion of CEM/PPO, PF, or RBPF. Those lanes remain evidence-tiered and validator-governed.
