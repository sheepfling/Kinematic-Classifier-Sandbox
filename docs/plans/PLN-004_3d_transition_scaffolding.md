# PLN-004 3D Transition Scaffolding

Title: 3D Transition Scaffolding
Plan ID: PLN-004
Status: proposed
Owner: @rick
Priority: P1
Objective: Make the current 1D-first sandbox explicitly dimension-ready so future 3D trajectory generators, feature extractors, and classifiers can plug into stable contracts without forcing a schema rewrite, and so the later dimensional-lift audit has a concrete adapter target rather than only abstract metadata fields.
Scope:
- Add dimension-aware trajectory and classifier metadata to the shared artifact contracts.
- Add explicit coordinate-frame and measurement-axis metadata so scalar-line, ENU, NED, ECEF, or body-frame studies can be distinguished cleanly.
- Keep the current 1D baselines and artifacts working without changing their inference behavior.
- Define the minimum interfaces a future 3D shared corpus, 3D feature pipeline, and 3D Kalman bank must satisfy.
- Add fairness-bucket reporting that continues to separate sensor-regime gains from inference gains after 3D work begins.
- Prepare a fake vector-valued corpus path so the generic harness can be exercised before full 3D physics exists.
Out of Scope:
- Full 3D trajectory generation.
- Full 3D Kalman, IMM, particle-filter, or aerodynamic parameter estimation.
- Coordinate transforms, geodesy, or terrain models beyond naming the contract fields they will require.
- Replacing the existing 1D baselines as the primary regression suite.

Implementation Steps:
1. Harden the current contracts.
   - Add `measurement_dim`, `measurement_axes`, `state_dim`, `state_axes`, and `coordinate_frame` to trajectory-level artifacts.
   - Add optional truth-component containers so future `x`, `y`, `z`, `vx`, `vy`, `vz`, `ax`, `ay`, `az` series can be recorded without changing the artifact shape again.
   - Add classifier-level metadata fields such as `classifier_id`, `sensor_regime_id`, and `run_id` where missing.
2. Propagate dimension metadata through the shared evaluator layer.
   - Ensure shared classifier runs carry `measurement_dim` and `coordinate_frame`.
   - Ensure shared comparison summaries and CSVs expose those fields.
3. Define 3D-ready dataset expectations.
   - Require future shared datasets to declare coordinate frame, measurement axes, and state axes explicitly.
   - Require a trajectory manifest to distinguish scalar, planar, and volumetric studies.
4. Define the 3D-ready feature contract.
   - Reserve feature groups for speed magnitude, climb rate, turn rate, curvature, lateral acceleration, vertical acceleration, and altitude-envelope features.
   - Keep the current feature-manifest format, but require future feature sets to label axis assumptions and frame assumptions.
5. Define the 3D-ready model-bank contract.
   - Require future per-class filter specs to declare state layout, measurement layout, process model family, and frame assumptions.
   - Keep the existing 1D Kalman bank as the scalar reference implementation.
6. Add migration notes and examples.
   - Document how a new 3D classifier should register with the shared evaluator.
   - Document how a 3D shared corpus should fit into the common comparison harness without breaking 1D studies.
7. Leave a direct handoff to the later dimensional-lift audit.
   - Make explicit which contract fields are enough for a fake vector-valued corpus proof.
   - Record which remaining modules still need adapter work before a true 3D study is realistic.

Validation:
- Existing 1D tests continue to pass unchanged in behavior.
- Contract tests validate dimension metadata and reject axis/dimension mismatches.
- Shared evaluation tests confirm `measurement_dim` and `coordinate_frame` flow through registered classifiers.
- Shared comparison artifacts emit dimension and frame metadata in run summaries and sensor-regime summaries.
- Roadmap references are updated so future 3D work targets the new contract fields explicitly.

Artifacts / Config:
- `docs/plans/PLN-004_3d_transition_scaffolding.md`
- `trajectory_path` and `method_run_summary.csv` entries with dimension and frame fields
- `sensor_regimes.json`
- `metrics_by_sensor_regime.csv`
- future `sensor_axes_manifest.json` or `coordinate_frame_manifest.json` if the 3D corpus requires them

Dependencies:
- `PLN-002_kinematic_classification_roadmap.md`
- later `PLN-014` dimensional lift audit
- shared contract layer in `src/kinematic_classifier_sandbox/contracts.py`
- shared evaluator layer in `src/kinematic_classifier_sandbox/validation/shared_evaluation.py`
- common comparison harness in `src/kinematic_classifier_sandbox/common_dataset_comparison.py`

Last Updated: 2026-05-24
