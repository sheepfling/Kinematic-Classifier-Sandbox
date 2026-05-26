from __future__ import annotations

from .trajectory_backend_contract_types import (
    BackendContractDefinition,
    ControlChannelSpec,
    ControlPolicySpec,
    DesignVariableSpec,
    EnvironmentSpec,
    ScenarioSpec,
    TrajectoryBackendCapabilities,
    TrajectoryBackendContractResult,
    TrajectoryRun,
)
from .trajectory_backend_contract_utils import (
    _backend_capability_schema,
    _capability_matrix_rows,
    _control_policy_schema,
    _design_variable_schema,
    _environment_spec_schema,
    _example_times,
    _scenario_spec_schema,
    _trajectory_run_schema,
    validate_backend_contract_definition,
)

def _parameter_backend_definition() -> BackendContractDefinition:
    backend_id = "parameter_only_1d"
    times = _example_times()
    return BackendContractDefinition(
        capabilities=TrajectoryBackendCapabilities(
            backend_id=backend_id,
            display_name="Parameter-Only 1D Backend",
            family="parameter_only_1d",
            dimensionality="1d",
            fidelity="toy",
            input_modes=("design_variables",),
            supports_environment=False,
            supports_sequential_control=False,
            supports_events=True,
            supports_stochastic_runs=True,
            runtime_class="cheap",
            determinism="seeded",
            state_outputs=("position", "velocity", "acceleration"),
            observation_outputs=("position",),
            event_outputs=("termination",),
            valid_search_methods=("random", "grid", "lhs", "sobol", "quality_diversity", "adaptive_stress"),
        ),
        scenario_spec=ScenarioSpec(
            scenario_id="parameter_cv_boundary",
            backend_id=backend_id,
            vehicle_or_model_id="toy_point_mass",
            target_class="constant_velocity",
            target_class_pair=("constant_velocity", "constant_acceleration"),
            scenario_family="boundary_case",
            difficulty_tier="boundary_v1",
            environment_id="vacuum_1d",
            sensor_regime_id="position_only",
            validity_constraints={"max_acceleration_rms": 0.12},
        ),
        design_variables=(
            DesignVariableSpec(
                name="initial_position",
                variable_type="float",
                units="m",
                bounds=(-2.0, 2.0),
                sampling_distribution="uniform",
                is_class_defining=False,
                is_environmental=False,
                is_control_related=False,
                is_sensitive_for_leakage=False,
            ),
            DesignVariableSpec(
                name="initial_velocity",
                variable_type="float",
                units="m/s",
                bounds=(0.1, 2.5),
                sampling_distribution="uniform",
                is_class_defining=True,
                is_environmental=False,
                is_control_related=False,
                is_sensitive_for_leakage=False,
            ),
            DesignVariableSpec(
                name="duration",
                variable_type="float",
                units="s",
                bounds=(1.0, 4.0),
                sampling_distribution="uniform",
                is_class_defining=False,
                is_environmental=False,
                is_control_related=False,
                is_sensitive_for_leakage=True,
            ),
            DesignVariableSpec(
                name="measurement_std",
                variable_type="float",
                units="m",
                bounds=(0.01, 0.35),
                sampling_distribution="uniform",
                is_class_defining=False,
                is_environmental=False,
                is_control_related=False,
                is_sensitive_for_leakage=True,
            ),
        ),
        control_policy=ControlPolicySpec(
            policy_id="none",
            policy_type="static_parameters",
            sequential=False,
            channels=(),
        ),
        environment_spec=EnvironmentSpec(
            environment_id="vacuum_1d",
            atmosphere_model_id=None,
            gravity_model_id=None,
            wind_model_id=None,
            temperature_profile_id=None,
            density_profile_id=None,
            turbulence_profile_id=None,
            terrain_or_reference_surface_id=None,
            coordinate_frame="scalar_line",
        ),
        example_run=TrajectoryRun(
            run_id="run_parameter_only_1d_example",
            backend_id=backend_id,
            scenario_id="parameter_cv_boundary",
            seed=7,
            success=True,
            failure_reason=None,
            times=times,
            truth_state={
                "position": (0.0, 0.55, 1.09, 1.64, 2.21),
                "velocity": (1.10, 1.09, 1.10, 1.12, 1.13),
                "acceleration": (0.0, -0.02, 0.02, 0.03, 0.02),
            },
            observations={"position": (0.01, 0.57, 1.08, 1.67, 2.18)},
            events=({"time": 2.0, "event_type": "termination", "event_value": "nominal_end"},),
            metadata={"measurement_dim": 1, "coordinate_frame": "scalar_line", "provenance_mode": "in_process"},
        ),
    )


def _controlled_backend_definition() -> BackendContractDefinition:
    backend_id = "controlled_1d"
    times = _example_times()
    return BackendContractDefinition(
        capabilities=TrajectoryBackendCapabilities(
            backend_id=backend_id,
            display_name="Controlled 1D Backend",
            family="controlled_1d",
            dimensionality="1d",
            fidelity="toy",
            input_modes=("design_variables", "piecewise_controls"),
            supports_environment=False,
            supports_sequential_control=True,
            supports_events=True,
            supports_stochastic_runs=True,
            runtime_class="cheap",
            determinism="seeded",
            state_outputs=("position", "velocity", "acceleration"),
            observation_outputs=("position",),
            event_outputs=("mode_switch", "termination"),
            valid_search_methods=("random", "lhs", "quality_diversity", "adaptive_stress", "cross_entropy"),
        ),
        scenario_spec=ScenarioSpec(
            scenario_id="controlled_switch_case",
            backend_id=backend_id,
            vehicle_or_model_id="toy_piecewise_accel",
            target_class="braking",
            target_class_pair=("constant_velocity", "braking"),
            scenario_family="switching_case",
            difficulty_tier="stress_v1",
            environment_id="vacuum_1d",
            sensor_regime_id="position_only",
            validity_constraints={"required_events": ["mode_switch"]},
        ),
        design_variables=(
            DesignVariableSpec(
                name="initial_position",
                variable_type="float",
                units="m",
                bounds=(-1.0, 1.0),
                sampling_distribution="uniform",
                is_class_defining=False,
                is_environmental=False,
                is_control_related=False,
                is_sensitive_for_leakage=False,
            ),
            DesignVariableSpec(
                name="initial_velocity",
                variable_type="float",
                units="m/s",
                bounds=(0.5, 2.5),
                sampling_distribution="uniform",
                is_class_defining=True,
                is_environmental=False,
                is_control_related=False,
                is_sensitive_for_leakage=False,
            ),
            DesignVariableSpec(
                name="switch_time",
                variable_type="float",
                units="s",
                bounds=(0.4, 1.4),
                sampling_distribution="uniform",
                is_class_defining=True,
                is_environmental=False,
                is_control_related=True,
                is_sensitive_for_leakage=False,
            ),
        ),
        control_policy=ControlPolicySpec(
            policy_id="piecewise_acceleration_schedule",
            policy_type="piecewise_constant_control",
            sequential=True,
            channels=(
                ControlChannelSpec(
                    control_name="acceleration_command",
                    units="m/s^2",
                    bounds=(-2.5, 2.5),
                    rate_limits=(-5.0, 5.0),
                    event_constraints=("switch_time_defined",),
                    backend_mapping="backend.accel_schedule",
                ),
            ),
        ),
        environment_spec=EnvironmentSpec(
            environment_id="vacuum_1d",
            atmosphere_model_id=None,
            gravity_model_id=None,
            wind_model_id=None,
            temperature_profile_id=None,
            density_profile_id=None,
            turbulence_profile_id=None,
            terrain_or_reference_surface_id=None,
            coordinate_frame="scalar_line",
        ),
        example_run=TrajectoryRun(
            run_id="run_controlled_1d_example",
            backend_id=backend_id,
            scenario_id="controlled_switch_case",
            seed=11,
            success=True,
            failure_reason=None,
            times=times,
            truth_state={
                "position": (0.0, 0.75, 1.43, 1.95, 2.32),
                "velocity": (1.50, 1.45, 1.25, 0.90, 0.55),
                "acceleration": (0.0, -0.20, -0.40, -0.70, -0.70),
            },
            observations={"position": (-0.01, 0.76, 1.42, 1.99, 2.31)},
            controls={"acceleration_command": (0.0, -0.20, -0.40, -0.70, -0.70)},
            events=(
                {"time": 0.5, "event_type": "mode_switch", "event_value": "enter_braking"},
                {"time": 2.0, "event_type": "termination", "event_value": "nominal_end"},
            ),
            metadata={"measurement_dim": 1, "coordinate_frame": "scalar_line", "provenance_mode": "in_process"},
        ),
    )


def _environment_backend_definition() -> BackendContractDefinition:
    backend_id = "environment_aware_1d"
    times = _example_times()
    return BackendContractDefinition(
        capabilities=TrajectoryBackendCapabilities(
            backend_id=backend_id,
            display_name="Environment-Aware 1D Backend",
            family="environment_aware_1d",
            dimensionality="1d",
            fidelity="toy_plus_environment",
            input_modes=("design_variables",),
            supports_environment=True,
            supports_sequential_control=False,
            supports_events=True,
            supports_stochastic_runs=True,
            runtime_class="cheap",
            determinism="seeded",
            state_outputs=("position", "velocity", "acceleration"),
            observation_outputs=("position",),
            event_outputs=("threshold_crossing", "termination"),
            valid_search_methods=("random", "lhs", "sobol", "quality_diversity", "leakage_aware_search"),
        ),
        scenario_spec=ScenarioSpec(
            scenario_id="environment_density_case",
            backend_id=backend_id,
            vehicle_or_model_id="toy_drag_1d",
            target_class="constant_acceleration",
            target_class_pair=("constant_velocity", "constant_acceleration"),
            scenario_family="environment_regime_case",
            difficulty_tier="realistic_v1",
            environment_id="standard_density_gradient_1d",
            sensor_regime_id="position_only",
            validity_constraints={"max_density_leakage_ratio": 1.1},
        ),
        design_variables=(
            DesignVariableSpec(
                name="initial_position",
                variable_type="float",
                units="m",
                bounds=(0.0, 1.0),
                sampling_distribution="uniform",
                is_class_defining=False,
                is_environmental=False,
                is_control_related=False,
                is_sensitive_for_leakage=False,
            ),
            DesignVariableSpec(
                name="initial_velocity",
                variable_type="float",
                units="m/s",
                bounds=(0.3, 2.0),
                sampling_distribution="uniform",
                is_class_defining=True,
                is_environmental=False,
                is_control_related=False,
                is_sensitive_for_leakage=False,
            ),
            DesignVariableSpec(
                name="drag_coefficient",
                variable_type="float",
                units="1",
                bounds=(0.0, 0.8),
                sampling_distribution="uniform",
                is_class_defining=False,
                is_environmental=True,
                is_control_related=False,
                is_sensitive_for_leakage=True,
            ),
        ),
        control_policy=ControlPolicySpec(
            policy_id="environment_passive",
            policy_type="static_parameters",
            sequential=False,
            channels=(),
        ),
        environment_spec=EnvironmentSpec(
            environment_id="standard_density_gradient_1d",
            atmosphere_model_id="atmosphere_like_1d",
            gravity_model_id="constant_g",
            wind_model_id="wind_gust_1d",
            temperature_profile_id="flat_temperature",
            density_profile_id="exp_decay_density",
            turbulence_profile_id="mild_turbulence",
            terrain_or_reference_surface_id="flat_reference_line",
            coordinate_frame="scalar_line",
        ),
        example_run=TrajectoryRun(
            run_id="run_environment_aware_1d_example",
            backend_id=backend_id,
            scenario_id="environment_density_case",
            seed=13,
            success=True,
            failure_reason=None,
            times=times,
            truth_state={
                "position": (0.0, 0.41, 0.90, 1.46, 2.08),
                "velocity": (0.70, 0.95, 1.03, 1.15, 1.26),
                "acceleration": (0.55, 0.43, 0.26, 0.24, 0.22),
            },
            observations={"position": (0.03, 0.39, 0.94, 1.44, 2.09)},
            environment_trace={
                "density_scale": (1.0, 0.94, 0.88, 0.82, 0.78),
                "wind_bias": (0.0, 0.03, 0.05, 0.02, 0.01),
            },
            events=(
                {"time": 1.0, "event_type": "threshold_crossing", "event_value": "density_scale_below_0.9"},
                {"time": 2.0, "event_type": "termination", "event_value": "nominal_end"},
            ),
            metadata={"measurement_dim": 1, "coordinate_frame": "scalar_line", "provenance_mode": "in_process"},
        ),
    )


def _mock_file_backend_definition() -> BackendContractDefinition:
    backend_id = "mock_file_backend_1d"
    times = _example_times()
    return BackendContractDefinition(
        capabilities=TrajectoryBackendCapabilities(
            backend_id=backend_id,
            display_name="Mock File-In/File-Out 1D Backend",
            family="mock_file_backend_1d",
            dimensionality="1d",
            fidelity="adapter_mock",
            input_modes=("input_deck", "design_variables", "control_schedule"),
            supports_environment=True,
            supports_sequential_control=True,
            supports_events=True,
            supports_stochastic_runs=False,
            runtime_class="medium",
            determinism="deterministic",
            state_outputs=("position", "velocity", "acceleration"),
            observation_outputs=("position", "velocity"),
            event_outputs=("phase_change", "termination", "constraint_violation"),
            valid_search_methods=("lhs", "sobol", "quality_diversity", "budgeted_doe", "surrogate_assisted"),
        ),
        scenario_spec=ScenarioSpec(
            scenario_id="mock_file_nominal_case",
            backend_id=backend_id,
            vehicle_or_model_id="mock_external_engine",
            target_class="maneuver",
            target_class_pair=("maneuver", "bounded_acceleration"),
            scenario_family="file_backend_case",
            difficulty_tier="adversarial_v1",
            environment_id="mock_file_environment",
            sensor_regime_id="position_plus_velocity",
            validity_constraints={"required_artifacts": ["input_deck", "stdout", "telemetry_csv"]},
        ),
        design_variables=(
            DesignVariableSpec(
                name="input_deck_hash",
                variable_type="string_token",
                units="sha256",
                bounds=None,
                sampling_distribution="fixed",
                is_class_defining=False,
                is_environmental=False,
                is_control_related=False,
                is_sensitive_for_leakage=False,
            ),
            DesignVariableSpec(
                name="maneuver_magnitude",
                variable_type="float",
                units="m/s^2",
                bounds=(0.2, 1.2),
                sampling_distribution="lhs",
                is_class_defining=True,
                is_environmental=False,
                is_control_related=True,
                is_sensitive_for_leakage=False,
            ),
            DesignVariableSpec(
                name="sample_period",
                variable_type="float",
                units="s",
                bounds=(0.25, 0.75),
                sampling_distribution="lhs",
                is_class_defining=False,
                is_environmental=False,
                is_control_related=False,
                is_sensitive_for_leakage=True,
            ),
        ),
        control_policy=ControlPolicySpec(
            policy_id="scheduled_external_controls",
            policy_type="scheduled_control",
            sequential=True,
            channels=(
                ControlChannelSpec(
                    control_name="longitudinal_command",
                    units="normalized",
                    bounds=(-1.0, 1.0),
                    rate_limits=(-2.0, 2.0),
                    event_constraints=("phase_change_allowed",),
                    backend_mapping="input_deck.controls.longitudinal",
                ),
            ),
        ),
        environment_spec=EnvironmentSpec(
            environment_id="mock_file_environment",
            atmosphere_model_id="tabular_atmosphere_mock",
            gravity_model_id="constant_g",
            wind_model_id="calm_air",
            temperature_profile_id="isa_mock",
            density_profile_id="isa_density_mock",
            turbulence_profile_id=None,
            terrain_or_reference_surface_id="flat_reference_line",
            coordinate_frame="scalar_line",
        ),
        example_run=TrajectoryRun(
            run_id="run_mock_file_backend_1d_example",
            backend_id=backend_id,
            scenario_id="mock_file_nominal_case",
            seed=0,
            success=True,
            failure_reason=None,
            times=times,
            truth_state={
                "position": (0.0, 0.44, 1.02, 1.52, 1.96),
                "velocity": (0.80, 1.02, 1.21, 0.99, 0.77),
                "acceleration": (0.45, 0.38, -0.08, -0.45, -0.44),
            },
            observations={
                "position": (0.01, 0.45, 1.00, 1.54, 1.97),
                "velocity": (0.78, 1.01, 1.19, 0.98, 0.78),
            },
            controls={"longitudinal_command": (0.4, 0.5, 0.1, -0.4, -0.4)},
            environment_trace={"density_scale": (1.0, 0.99, 0.98, 0.97, 0.96)},
            events=(
                {"time": 1.0, "event_type": "phase_change", "event_value": "maneuver_peak"},
                {"time": 2.0, "event_type": "termination", "event_value": "nominal_end"},
            ),
            metadata={
                "measurement_dim": 1,
                "coordinate_frame": "scalar_line",
                "provenance_mode": "file_mock",
                "input_deck_template": "mock_backend_v1",
            },
        ),
    )


def default_backend_contract_definitions() -> tuple[BackendContractDefinition, ...]:
    return (
        _parameter_backend_definition(),
        _controlled_backend_definition(),
        _environment_backend_definition(),
        _mock_file_backend_definition(),
    )


def _capability_matrix_rows(definitions: tuple[BackendContractDefinition, ...]) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for definition in definitions:
        capabilities = definition.capabilities
        rows.append(
            {
                "backend_id": capabilities.backend_id,
                "family": capabilities.family,
                "runtime_class": capabilities.runtime_class,
                "supports_environment": int(capabilities.supports_environment),
                "supports_sequential_control": int(capabilities.supports_sequential_control),
                "supports_events": int(capabilities.supports_events),
                "supports_stochastic_runs": int(capabilities.supports_stochastic_runs),
                "state_output_count": len(capabilities.state_outputs),
                "observation_output_count": len(capabilities.observation_outputs),
                "event_output_count": len(capabilities.event_outputs),
                "search_method_count": len(capabilities.valid_search_methods),
            }
        )
    return tuple(rows)



def analyze_trajectory_backend_contract() -> TrajectoryBackendContractResult:
    definitions = default_backend_contract_definitions()
    validation_rows: list[dict[str, Any]] = []
    for definition in definitions:
        errors = validate_backend_contract_definition(definition)
        validation_rows.append(
            {
                "backend_id": definition.capabilities.backend_id,
                "family": definition.capabilities.family,
                "valid": not errors,
                "error_count": len(errors),
                "errors": errors,
            }
        )

    backend_contract = {
        "contract_version": "m31_v1",
        "goal": "Typed black-box backend contract for backend-agnostic corpus exploration.",
        "object_model": [
            "ScenarioSpec",
            "DesignVariableSpec",
            "ControlPolicySpec",
            "EnvironmentSpec",
            "TrajectoryRun",
            "TrajectoryBackendCapabilities",
        ],
        "backend_families": [
            {
                "backend_id": definition.capabilities.backend_id,
                "family": definition.capabilities.family,
                "capabilities": definition.capabilities.model_dump(),
                "scenario_spec": definition.scenario_spec.model_dump(),
                "design_variables": [variable.model_dump() for variable in definition.design_variables],
                "control_policy": definition.control_policy.model_dump(),
                "environment_spec": definition.environment_spec.model_dump(),
                "example_run": definition.example_run.model_dump(),
            }
            for definition in definitions
        ],
        "validation": validation_rows,
    }
    capability_rows = _capability_matrix_rows(definitions)
    valid_count = sum(1 for row in validation_rows if row["valid"])
    sequential_count = sum(1 for definition in definitions if definition.capabilities.supports_sequential_control)
    environment_count = sum(1 for definition in definitions if definition.capabilities.supports_environment)

    from .trajectory_backend_contract_rendering import _render_contract_markdown

    report_markdown = _render_contract_markdown(
        definitions,
        valid_count=valid_count,
        sequential_count=sequential_count,
        environment_count=environment_count,
    )

    return TrajectoryBackendContractResult(
        backend_contract=backend_contract,
        backend_capability_schema=_backend_capability_schema(),
        scenario_spec_schema=_scenario_spec_schema(),
        design_variable_schema=_design_variable_schema(),
        control_policy_schema=_control_policy_schema(),
        environment_spec_schema=_environment_spec_schema(),
        trajectory_run_schema=_trajectory_run_schema(),
        capability_matrix_rows=capability_rows,
        report_markdown=report_markdown,
    )
