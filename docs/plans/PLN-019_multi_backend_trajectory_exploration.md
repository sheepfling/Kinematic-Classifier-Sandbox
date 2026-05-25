# PLN-019 Multi-Backend Trajectory Exploration

Title: Generic Multi-Backend Trajectory Exploration Architecture
Plan ID: PLN-019
Status: done
Owner: @rick
Priority: P1
Last Updated: 2026-05-24

Objective:
Design and prove a generic exploratory corpus-generation mechanism that can operate across diverse trajectory engines, from simple 1D toy models to future 3D, 3DOF, 6DOF, and external file-based simulators. The search layer must remain backend-agnostic and interact through capability descriptors, scenario specs, design variables, control policies, environment specs, normalized trajectory telemetry, and decomposed score components.

Architecture Principle:
- Treat every simulator backend as a typed black box.
- The exploration engine must never reason directly in backend-native terms such as throttle, fuel, angle of attack, pitch-over, or aerodynamic table formats.
- Backend adapters own simulator-specific translation and execution.
- The common exploration stack only sees:
  - `ScenarioSpec`
  - `DesignVariableSpec`
  - `ControlPolicySpec`
  - `EnvironmentSpec`
  - `TrajectoryRun`
  - normalized features, labels, classifier outputs, and scores

Scope:
- Define a backend contract that treats each simulator as a typed black box with declared capabilities.
- Separate:
  - backend-specific input/output mapping
  - normalized telemetry and provenance
  - feature extraction and class validity checks
  - search, archive, corpus, and stress machinery
- Prove the architecture first with multiple 1D backend families:
  - parameter-only 1D backend
  - controlled 1D backend
  - environment-aware 1D backend
  - mock file-in/file-out backend
- Add capability-aware search planning so backend runtime/fidelity/control structure influences the allowed search methods.
- Add environment-aware corpus and leakage auditing surfaces.
- Add a generic exploration dashboard that can compare selected trajectories across backend types.
- Keep the design explicitly 3D-ready without requiring immediate integration of high-fidelity external simulators.
- Cover future backend families conceptually, even if only 1D proofs are implemented now:
  - 1D toy generator
  - 3+3 translational backend
  - 3DOF point-mass engine
  - 6DOF rigid-body engine
  - file-in/file-out external simulator
  - replay-from-data backend
  - surrogate-model backend

Out of Scope:
- Direct integration with real external high-fidelity tools such as TAOS, TGx, FLITES, or mission-specific simulators in this milestone.
- Replacing the current 1D generator stack; this plan adds a multi-backend exploration layer around it.
- Training a real RL policy here. RL remains covered by the separate decision gate from PLN-018.
- Operational targeting or mission optimization objectives.
- Full 3D or 6DOF physics implementation before the backend abstraction is proven in 1D.
- Direct coupling of the search engine to simulator-specific APIs or parameter names.

Backend Black-Box Contract:
- Every backend should behave like a typed service with:
  - declared accepted inputs
  - declared controllable variables
  - declared environment dependencies
  - declared state outputs
  - declared event outputs
  - declared runtime and fidelity estimate
  - declared determinism/reproducibility behavior
- Every backend adapter should provide equivalent phases:
  - `prepare(input_bundle)`
  - `run()`
  - `parse_outputs()`
  - `validate_run()`
  - `normalize_output() -> TrajectoryRun`

Common Object Model:
- `ScenarioSpec`
  - `scenario_id`
  - `backend_id`
  - `vehicle_or_model_id`
  - `target_class`
  - `target_class_pair`
  - `scenario_family`
  - `difficulty_tier`
  - `environment_id`
  - `sensor_regime_id`
  - `validity_constraints`
- `DesignVariableSpec`
  - `name`
  - `type`
  - `units`
  - `bounds`
  - `sampling_distribution`
  - `is_class_defining`
  - `is_environmental`
  - `is_control_related`
  - `is_sensitive_for_leakage`
- `ControlPolicySpec`
  - supports fixed, piecewise, spline, scheduled, event-triggered, closed-loop, replay, and future RL policies
  - control channels should declare:
    - `control_name`
    - `units`
    - `bounds`
    - `rate_limits`
    - `event_constraints`
    - `backend_mapping`
- `EnvironmentSpec`
  - `atmosphere_model_id`
  - `gravity_model_id`
  - `wind_model_id`
  - `temperature_profile_id`
  - `density_profile_id`
  - `turbulence_profile_id`
  - `terrain_or_reference_surface_id`
  - `coordinate_frame`
- `TrajectoryRun`
  - `run_id`
  - `backend_id`
  - `scenario_id`
  - `seed`
  - `success`
  - `failure_reason`
  - `times`
  - `truth_state`
  - `observations`
  - `controls`
  - `environment_trace`
  - `events`
  - `metadata`

Implementation Steps:
1. Define the common backend-facing object model.
   - `ScenarioSpec`
   - `DesignVariableSpec`
   - `ControlPolicySpec`
   - `EnvironmentSpec`
   - `TrajectoryRun`
   - `TrajectoryBackendCapabilities`
2. Implement a typed backend adapter contract.
   - Backend adapters own engine-specific input mapping, execution, parsing, validation, and normalization.
   - The exploration engine only sees normalized candidates, telemetry, metadata, and score components.
3. Build backend capability descriptors.
   - Each backend must declare:
     - dimensionality
     - fidelity
     - supported input modes
     - environment support
     - sequential-control support
     - event support
     - stochastic support
     - runtime class
     - determinism behavior
     - valid search methods
4. Prove the adapter architecture with 1D backends.
   - Parameter-only 1D backend
   - Controlled 1D backend
   - Environment-aware 1D backend
   - Mock file-based backend
5. Add environment-aware corpus generation and auditing.
   - Record environment traces and environment provenance.
   - Extend leakage auditing so class-linked environment variables are visible.
6. Add capability-aware search planning.
   - Cheap parameter-only backends get broad search.
   - Expensive or file-based backends get budgeted planning, caching, and surrogate-friendly settings.
   - Sequential-control backends are the only ones that enable future adaptive-control or RL-style exploration.
7. Add a generic corpus exploration dashboard.
   - Compare backend families on coverage, feature excitation, corpus quality, stress discovery, and provenance completeness.
8. Keep class validity explicit and independent from requested target labels.
   - A generated run should not automatically inherit the requested class.
   - Class validity should be re-evaluated from normalized telemetry, events, and feature signatures.
9. Preserve search and artifact provenance throughout the stack.
   - Every generated candidate, cached run, selected corpus item, and report row should remain traceable to backend, adapter, search policy, and input hash.

Adapter Layering:
- Backend adapter
  - maps generic candidate specs into engine-native execution inputs
  - generates input decks, API payloads, or config files
  - handles execution and raw-output collection
- Telemetry adapter
  - normalizes raw outputs into `TrajectoryRun`
  - standardizes times, truth state, observations, controls, events, and environment traces
- Feature adapter
  - converts `TrajectoryRun` into feature tables
  - declares required state fields, required environment fields, and supported dimensionality
- Class-label adapter
  - evaluates class validity
  - computes hard/soft class fit signals
  - can expose alternate-class similarity for ambiguity analysis

Exploratory Objective Hierarchy:
- Corpus coverage
  - class balance
  - scenario balance
  - duration coverage
  - feature excitation
  - state-space coverage
  - environment coverage
  - sensor-regime coverage
- Class-pair boundary discovery
  - posterior entropy
  - top-two posterior margin
  - oracle margin
  - feature overlap
  - ambiguity score
- Feature excitation
  - speed range
  - acceleration range
  - curvature
  - residual structure
  - sign changes
  - monotonicity
  - innovation magnitude
  - posterior entropy
- Classifier stress
  - wrong classification
  - confident wrong classification
  - poor calibration
  - slow time-to-confidence
  - high prior sensitivity
  - method disagreement
- Method differentiation
  - cases where one family beats another in a diagnostically useful way

Score Decomposition:
- `validity_score`
- `class_fit_score`
- `feature_excitation_score`
- `coverage_novelty_score`
- `boundary_score`
- `classifier_stress_score`
- `prior_sensitivity_score`
- `method_differentiation_score`
- `leakage_penalty`
- `physical_invalidity_penalty`
- `runtime_cost_penalty`

Quality-Diversity Guidance:
- QD should be a first-class exploration strategy, not an afterthought.
- The early archive should prefer interpretable axes over abstract latent axes.
- Suggested 1D archive axes:
  - `target_class`
  - `class_pair`
  - `duration_bucket`
  - `sample_count_bucket`
  - `speed_range_bucket`
  - `acceleration_range_bucket`
  - `monotonicity_bucket`
  - `linear_residual_bucket`
  - `posterior_entropy_bucket`
  - `prior_sensitivity_bucket`
  - `classifier_error_type`
- Suggested 3D archive axes later:
  - `target_class`
  - `class_pair`
  - `duration_bucket`
  - `speed_norm_range_bucket`
  - `acceleration_norm_range_bucket`
  - `altitude_band_bucket`
  - `curvature_bucket`
  - `turn_rate_bucket`
  - `energy_change_bucket`
  - `environment_regime_bucket`
  - `sensor_regime_bucket`
  - `posterior_entropy_bucket`
  - `classifier_error_type`

Validation:
- Every backend declares a valid capability descriptor.
- Every backend can produce a normalized `TrajectoryRun`.
- A compatible scenario can run through at least two backends and normalize into the same telemetry contract.
- Failed backend runs are captured structurally rather than appearing as crashes or missing rows.
- Environment variables are carried into telemetry metadata and leakage audits.
- The search planner disables sequential-control methods for parameter-only backends.
- The search planner avoids broad expensive search for high-cost backends.
- The generic dashboard can compare selected trajectories from at least two backend families.
- All selected trajectories retain backend, environment, search, and provenance metadata.
- Run cache keys remain stable from normalized input hashes.
- Failed backend runs are preserved as structured artifacts rather than silent omissions.
- Environment-linked leakage can be surfaced independently of class-target success.

Artifacts / Config:
- `docs/plans/PLN-019_multi_backend_trajectory_exploration.md`
- `artifacts/trajectory_backend_contract/backend_contract.json`
- `artifacts/trajectory_backend_contract/backend_capability_schema.json`
- `artifacts/trajectory_backend_contract/scenario_spec_schema.json`
- `artifacts/trajectory_backend_contract/design_variable_schema.json`
- `artifacts/trajectory_backend_contract/control_policy_schema.json`
- `artifacts/trajectory_backend_contract/environment_spec_schema.json`
- `artifacts/trajectory_backend_contract/trajectory_run_schema.json`
- `artifacts/trajectory_backend_contract/backend_contract_report.md`
- `artifacts/backend_adapter_proof/backend_manifest.json`
- `artifacts/backend_adapter_proof/backend_run_examples.csv`
- `artifacts/backend_adapter_proof/backend_output_equivalence_report.md`
- `artifacts/backend_adapter_proof/adapter_failure_cases.csv`
- `artifacts/environment_aware_corpus/environment_manifest.json`
- `artifacts/environment_aware_corpus/environment_coverage.csv`
- `artifacts/environment_aware_corpus/environment_leakage_audit.csv`
- `artifacts/environment_aware_corpus/atmosphere_like_1d_report.md`
- `artifacts/capability_aware_search/search_planner_rules.json`
- `artifacts/capability_aware_search/search_method_selection_matrix.csv`
- `artifacts/capability_aware_search/backend_search_plan.csv`
- `artifacts/capability_aware_search/search_method_selection_report.md`
- `artifacts/generic_corpus_exploration/exploration_manifest.json`
- `artifacts/generic_corpus_exploration/candidate_scores.csv`
- `artifacts/generic_corpus_exploration/archive_cells.csv`
- `artifacts/generic_corpus_exploration/selected_corpus_manifest.json`
- `artifacts/generic_corpus_exploration/backend_comparison.csv`
- `artifacts/generic_corpus_exploration/corpus_exploration_report.md`
- `runs/external_backend/{run_id}/input/...`
- `runs/external_backend/{run_id}/execution/...`
- `runs/external_backend/{run_id}/raw_output/...`
- `runs/external_backend/{run_id}/normalized/...`
- `runs/external_backend/{run_id}/validation/...`

Dependencies:
- `PLN-004` 3D transition scaffolding
- `PLN-010` generic inference contract
- `PLN-011` generic feature taxonomy
- `PLN-013` generic filtering contract
- `PLN-014` dimensional lift audit
- `PLN-015` corpus coverage framework
- `PLN-018` agentic corpus synthesis
- existing modules:
  - `corpus_gym.py`
  - `corpus_search_baseline.py`
  - `quality_diversity_corpus.py`
  - `adaptive_stress_corpus.py`
  - `trajectory_generator.py`
  - `feature_analysis.py`
  - `corpus_adequacy_audit.py`

Milestones:
- `M31`: Multi-backend trajectory exploration contract
  - Goal:
    - Define the common schemas and capability descriptors that all trajectory engines must satisfy.
  - Deliverables:
    - `artifacts/trajectory_backend_contract/backend_contract.json`
    - `artifacts/trajectory_backend_contract/backend_capability_schema.json`
    - `artifacts/trajectory_backend_contract/scenario_spec_schema.json`
    - `artifacts/trajectory_backend_contract/design_variable_schema.json`
    - `artifacts/trajectory_backend_contract/control_policy_schema.json`
    - `artifacts/trajectory_backend_contract/environment_spec_schema.json`
    - `artifacts/trajectory_backend_contract/trajectory_run_schema.json`
    - `artifacts/trajectory_backend_contract/backend_contract_report.md`
  - Visualizations:
    - backend contract relationship diagram
    - capability matrix heatmap by backend family
  - Exit criterion:
    - At least four 1D backend families declare valid capabilities and can target the common `TrajectoryRun` schema.

- `M32`: Backend adapter proof with 1D engines
  - Goal:
    - Prove that multiple 1D backends can execute through the same adapter pipeline and normalize into shared telemetry.
  - Deliverables:
    - `artifacts/backend_adapter_proof/backend_manifest.json`
    - `artifacts/backend_adapter_proof/backend_run_examples.csv`
    - `artifacts/backend_adapter_proof/backend_output_equivalence_report.md`
    - `artifacts/backend_adapter_proof/adapter_failure_cases.csv`
  - Visualizations:
    - backend input-to-output adapter flow diagram
    - normalized telemetry comparison plots across backends
    - adapter failure taxonomy chart
  - Exit criterion:
    - The same compatible scenario runs through at least two backends and both normalize into equivalent telemetry/artifact structures.

- `M33`: Environment-aware corpus generation
  - Goal:
    - Add environment metadata and environment-sensitive leakage analysis without requiring 3D physics yet.
  - Deliverables:
    - `artifacts/environment_aware_corpus/environment_manifest.json`
    - `artifacts/environment_aware_corpus/environment_coverage.csv`
    - `artifacts/environment_aware_corpus/environment_leakage_audit.csv`
    - `artifacts/environment_aware_corpus/atmosphere_like_1d_report.md`
  - Visualizations:
    - environment regime coverage heatmap
    - environment-variable leakage plot by class
    - environment-conditioned trajectory gallery
  - Exit criterion:
    - Environment variables are recorded, searchable, and auditable, and the leakage audit can flag class-linked environment correlations.

- `M34`: Capability-aware search planner
  - Goal:
    - Make the exploration engine choose search methods based on backend runtime, fidelity, and control structure.
  - Deliverables:
    - `artifacts/capability_aware_search/search_planner_rules.json`
    - `artifacts/capability_aware_search/search_method_selection_matrix.csv`
    - `artifacts/capability_aware_search/backend_search_plan.csv`
    - `artifacts/capability_aware_search/search_method_selection_report.md`
  - Visualizations:
    - search-method selection matrix
    - backend-versus-search-strategy decision tree
    - projected cost-versus-coverage frontier plot
  - Exit criterion:
    - Parameter-only, sequential-control, cheap, and expensive backend types all receive different search plans for explicit reasons.

- `M35`: Generic corpus exploration dashboard
  - Goal:
    - Run a small multi-backend exploration and compare selected corpora across backend types.
  - Deliverables:
    - `artifacts/generic_corpus_exploration/exploration_manifest.json`
    - `artifacts/generic_corpus_exploration/candidate_scores.csv`
    - `artifacts/generic_corpus_exploration/archive_cells.csv`
    - `artifacts/generic_corpus_exploration/selected_corpus_manifest.json`
    - `artifacts/generic_corpus_exploration/backend_comparison.csv`
    - `artifacts/generic_corpus_exploration/corpus_exploration_report.md`
  - Visualizations:
    - backend coverage comparison chart
    - archive coverage heatmap
    - score-component parallel-coordinates plot
    - selected trajectory gallery
    - provenance completeness dashboard
  - Exit criterion:
    - The selected corpus contains trajectories from at least two backend types, improves coverage over a random baseline, and preserves backend/environment/search provenance.

Recommended Execution Order:
1. `M31` backend and schema contract
2. `M32` adapter proof with 1D backend families
3. `M33` environment-aware corpus extension
4. `M34` capability-aware search planner
5. `M35` generic corpus exploration dashboard

1D Proof Backends:
- Backend 1: parameter-only 1D engine
  - purpose: prove DOE, random search, static scoring, and corpus adequacy
  - inputs:
    - class type
    - initial position
    - initial velocity
    - acceleration
    - duration
    - noise level
    - sample count
- Backend 2: controlled 1D engine
  - purpose: prove scheduled-control search and switching/stress cases
  - inputs:
    - piecewise acceleration schedule
    - jerk limits
    - mode switch time
    - noise schedule
    - sampling schedule
- Backend 3: atmosphere-like 1D engine
  - purpose: prove `EnvironmentSpec`, environment-aware metadata, and leakage auditing before 3D
  - inputs:
    - drag coefficient
    - density profile
    - wind-like disturbance
    - altitude-like coordinate
- Backend 4: file-in/file-out mock backend
  - purpose: prove external adapter shape before real TAOS/TGx/FLITES/custom integration
  - inputs and surfaces:
    - input deck template
    - run command
    - output CSV
    - parser
    - structured failure cases
    - stable cache key

Search Method Selection Matrix:
- `parameter_only_1d`
  - recommended: random, LHS, Sobol, QD
- `controlled_1d`
  - recommended: QD, adaptive stress, cross-entropy method, RL later
- `environment_aware_1d`
  - recommended: DOE, QD, leakage-aware search
- `mock_file_backend`
  - recommended: budgeted DOE, caching, QD
- future `3+3` backend
  - recommended: Sobol, QD, stress search
- future `6DOF` backend
  - recommended: small DOE, surrogate, active learning
- future external high-fidelity backend
  - recommended: adapter-driven batch search, surrogate-assisted planning

External Backend Run Layout:
- `runs/external_backend/{run_id}/input/`
  - `scenario.yaml`
  - `design_variables.json`
  - `control_policy.json`
  - `environment.json`
  - `backend_input_deck.*`
- `runs/external_backend/{run_id}/execution/`
  - `command.txt`
  - `stdout.log`
  - `stderr.log`
  - `return_code.txt`
  - `runtime.json`
- `runs/external_backend/{run_id}/raw_output/`
- `runs/external_backend/{run_id}/normalized/`
  - `trajectory_run.json`
  - `telemetry.csv`
  - `events.csv`
  - `observations.csv`
- `runs/external_backend/{run_id}/validation/`
  - `validation_report.json`
  - `failure_reason.txt`

Artifact Provenance Requirements:
- `trajectory_id`
- `backend_id`
- `backend_version`
- `adapter_version`
- `scenario_id`
- `search_method`
- `search_iteration`
- `candidate_id`
- `seed`
- `input_hash`
- `environment_id`
- `class_target`
- `class_validity_score`
- `feature_cell_id`
- `score_components`
- `run_status`

Future 3D Transition:
- Phase 3D-A: `3+3` translational backend
  - prove vector telemetry, vector features, and 3D-ready schemas
- Phase 3D-B: atmosphere-aware `3DOF` backend
  - prove environment coupling, drag-like effects, wind-like effects, and fuel/mass metadata if needed
- Phase 3D-C: external backend mocks
  - `mock_TAOS_adapter`
  - `mock_TGx_adapter`
  - `mock_FLITES_adapter`
  - prove deck creation, output parsing, execution capture, caching, and failure handling
- Phase 3D-D: real backend integration
  - connect an actual high-fidelity backend only after the adapter proof path is stable

Design Notes:
- The exploration engine must not call simulator-specific functions directly.
- Backend adapters own:
  - design-variable mapping
  - control-policy mapping
  - environment mapping
  - execution
  - parsing
  - normalization
- The generic search/corpus layer owns:
  - candidate generation
  - archive logic
  - corpus selection
  - score decomposition
  - stress discovery
  - provenance tracking
- The search layer must not embed backend conditionals such as:
  - `if backend == "taos": ...`
  - `elif backend == "flites": ...`
- Prefer adapter methods such as:
  - `map_design_variables(candidate)`
  - `map_control_policy(candidate)`
  - `map_environment(candidate)`
  - `run()`
  - `normalize_output()`
- Class labels must be checked from telemetry and features rather than blindly copied from requested targets.
- Future 3D readiness should be proven by showing that the exploration machinery is backend-agnostic, not by claiming that 1D physics is sufficient.

Success Criteria:
- The repo can explore trajectories across multiple backend families through one normalized contract.
- The corpus machinery no longer depends on one simulator’s input shape.
- Environment-aware metadata and leakage auditing work before 3D integration.
- Search-method choice becomes capability-aware instead of hard-coded.
- The system is demonstrably ready for a future 3D backend adapter without redesigning the exploration layer.
- The plan proves that the corpus/search/classifier machinery does not care which compatible backend produced the trajectory.

Follow-On:
- PLN-019 proves the backend-agnostic exploration architecture.
- The next maturity layer is [PLN-021_objective_driven_corpus_explorer_v1.md](docs/plans/PLN-021_objective_driven_corpus_explorer_v1.md), which covers:
  - objective-driven candidate generation
  - class-validity scoring
  - real feature extraction integration
  - classifier-in-the-loop scoring
  - iterative quality-diversity archive growth
  - materialized selected corpus output
