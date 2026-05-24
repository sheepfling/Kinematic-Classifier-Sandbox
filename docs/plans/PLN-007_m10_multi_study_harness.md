# PLN-007 M10 Multi-Study Harness

Title: Build a Real Multi-Study Common Experiment Harness
Plan ID: PLN-007
Status: done
Owner: @codex
Priority: P1
Objective: Replace the current common-1D-specialized harness path with a real multi-study experiment harness that runs from manifest/config inputs, emits one shared artifact contract, and serves as the execution backbone for the generic methodology proof phase rather than only the current scalar toy problem.
Scope:
- Generalize the current `M10` harness so studies are selected by config rather than hardcoded executable pair logic.
- Make the harness consume dataset, class-pair, feature-set, and classifier manifests directly.
- Preserve the current artifact contract while making the execution path reusable across multiple study definitions.
- Prepare the repo for the generic-inference, feature-taxonomy, filtering-contract, and dimensional-lift work in `M12` through `M16`.
- Prepare the repo for a second study domain and future 3D work without requiring harness rewrites.
Out of Scope:
- Implementing the first full 3D study itself.
- Replacing all current milestone-specific benchmark modules immediately.
- IMM, transition-matrix accumulation, or advanced-filter decision gates.
- Redesigning evidence or posterior schemas beyond what is needed to support the follow-on contract plans.
Implementation Steps:
1. Define a study registry and study configuration boundary.
   - Introduce a study-level abstraction so `common_1d_classifier_study` becomes one entry rather than the implicit default.
   - Require each study to declare dataset source, class registry, feature-set manifest, class-pair manifest, classifier manifest, and output directory.
2. Remove hardcoded executable pair assumptions from the common harness.
   - Refactor `common_experiment_harness.py` so it no longer hardcodes binary class dynamics or special pair specs in the core execution path.
   - Move any current scalar pair-specific logic behind a study adapter layer.
3. Make manifests executable.
   - Wire `classifier_manifest.json`, `feature_sets.json`, and `class_pair_manifest.json` into the actual run graph.
   - Ensure classifier execution, feature-set selection, and pairwise studies are driven by manifest/config data rather than duplicated code logic.
4. Generalize dataset ingestion for the harness.
   - Accept trajectory datasets from generated study outputs rather than rebuilding separate shared-dynamics toy trajectories in the harness.
   - Standardize how trajectory metadata, scenario metadata, and sensor-regime metadata enter the harness.
5. Unify shared outputs.
   - Emit one consistent set of predictions, posterior history, likelihood history, feature matrices, class-pair metrics, feature-set metrics, and study summary outputs across all runnable studies.
   - Keep the same logical columns across studies wherever the contract permits.
6. Make pairwise studies first-class outputs.
   - Add explicit per-pair runners and metrics generation driven by the class-pair manifest.
   - Ensure duration, noise, prior, and identifiability slices can be emitted by pair without special-case code.
7. Expose the harness as a junior-friendly entrypoint.
   - Add a script or CLI entrypoint that takes a study config path and writes the unified run directory.
   - Document the command, expected outputs, and rerun flow.
8. Prove the harness on the current 1D study.
   - Migrate `common_1d_classifier_study` onto the new multi-study path without loss of current artifacts.
   - Leave the current specialized helpers in place only as compatibility shims where necessary.
9. Make the harness a stable substrate for the next proof phase.
   - Ensure `M12` generic inference contract work can validate pointwise, windowed, Bayesian accumulator, and Kalman outputs through one harness surface.
   - Ensure `M16` dimensional lift work can add a fake vector-valued corpus adapter without rewriting run orchestration.
Validation:
- Unit tests for study-config parsing, manifest loading, and run-graph construction.
- Contract tests verifying that all harness outputs include required metadata columns and consistent classifier, feature-set, and sensor-regime identifiers.
- Regression tests proving the new harness can reproduce the current `common_1d_classifier_study` artifact families.
- One end-to-end harness test that runs from config only and produces the unified output directory.
- Full-suite regression run with `python3 -m pytest -q`.
Artifacts / Config:
- `src/kinematic_classifier_sandbox/common_experiment_harness.py`
- `src/kinematic_classifier_sandbox/shared_evaluation.py`
- `src/kinematic_classifier_sandbox/contracts.py`
- `experiments/common_1d_classifier_study/common_experiment_config.yaml`
- future study configs under `experiments/`
- junior-facing runner script under `scripts/`
- documentation for the harness rerun surface
Dependencies:
- Current contracts and artifact schema in `contracts.py`
- Current manifests under `experiments/common_1d_classifier_study/`
- Current classifier adapters and shared evaluation helpers
- Current feature analysis, coverage, and generator outputs
Last Updated: 2026-05-24

## Progress Notes

- Added a public `scripts/run_study.py` entrypoint for config-driven study reruns.
- Refactored the common harness so study resolution now uses an explicit `study_adapter` id instead of assuming the experiment name is the execution path.
- Promoted more of the YAML scaffold into executable config, including dataset metadata, declared pairs, and output filenames.
- Removed the fixed executable-pair tuple from the harness; the pair run graph now follows the config-declared class pairs.
- Extended the executable subset to cover all five currently declared manifest pairs, including `constant_acceleration_vs_maneuver` and `maneuver_vs_bounded_acceleration`.

## File-by-File Change Plan

### `src/kinematic_classifier_sandbox/common_experiment_harness.py`

- Split the module into:
  - config loading
  - study adapter resolution
  - classifier execution
  - pairwise metric generation
  - artifact writing
- Delete or isolate the hardcoded executable pair specs from the core path.
- Replace direct scalar shared-dynamics assumptions with study-provided trajectory sets and study-provided class registries.

### `src/kinematic_classifier_sandbox/shared_evaluation.py`

- Expand the shared classifier adapter surface so harness execution does not depend on bespoke per-study wiring.
- Standardize how sensor regime, measurement dimension, coordinate frame, classifier id, and feature-set id are attached to shared runs.

### `src/kinematic_classifier_sandbox/contracts.py`

- Review the current contracts for any fields that are still too 1D-implicit.
- Ensure the common harness outputs can carry study id, dataset id, feature-set id, classifier id, sensor-regime id, and pair id without ad hoc columns.

### `experiments/common_1d_classifier_study/common_experiment_config.yaml`

- Make this the first real consumer of the new multi-study harness.
- Ensure every currently emitted output path maps cleanly to the new shared output surface.

### `experiments/common_1d_classifier_study/classifier_manifest.json`

- Use this as executable input, not just descriptive documentation.
- Add any missing fields needed to instantiate runs unambiguously from config.

### `experiments/common_1d_classifier_study/feature_sets.json`

- Treat the manifest as the authoritative feature-set registry for the harness.
- Ensure feature-set studies are generated from this manifest rather than duplicated selection logic.

### `experiments/common_1d_classifier_study/class_pair_manifest.json`

- Drive pairwise runner creation from this manifest.
- Ensure expected difficulty, required separators, and pair identity appear in emitted metrics.

### `scripts/`

- Add a junior-facing harness runner, for example `scripts/run_study.py`.
- Keep the invocation simple: one config path in, one run directory out.

### `docs/`

- Add documentation showing:
  - how to run one study
  - how to inspect outputs
  - how to add a second study without editing core harness logic

## Graduation Criteria

`M10` should be considered complete only when:

- the harness can run `common_1d_classifier_study` from config alone
- no core execution path depends on hardcoded binary pair specs
- classifier, feature-set, and class-pair manifests all drive real execution
- one unified artifact contract is emitted for the full run
- the code structure clearly supports adding a second study domain without rewriting the harness

## Immediate Follow-On

Once `M10` is complete, the next proof is architectural rather than algorithmic:

- `M11` should finish feature-set and class-pair studies as first-class dimensions.
- `M12` should prove the generic inference contract.
- `M13` through `M16` should prove that features, evidence, filters, and dimensional lift can be expressed through stable contracts.

Only after that proof phase should the repo decide whether a second study domain, IMM, PF, or RBPF is the next highest-value move.
