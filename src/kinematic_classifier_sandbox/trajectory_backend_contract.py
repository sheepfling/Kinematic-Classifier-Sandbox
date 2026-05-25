from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    scenario_id: str
    backend_id: str
    vehicle_or_model_id: str
    target_class: str | None
    target_class_pair: tuple[str, str] | None
    scenario_family: str
    difficulty_tier: str
    environment_id: str
    sensor_regime_id: str
    validity_constraints: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DesignVariableSpec:
    name: str
    variable_type: str
    units: str
    bounds: tuple[float, float] | None
    sampling_distribution: str
    is_class_defining: bool
    is_environmental: bool
    is_control_related: bool
    is_sensitive_for_leakage: bool


@dataclass(frozen=True, slots=True)
class ControlChannelSpec:
    control_name: str
    units: str
    bounds: tuple[float, float] | None
    rate_limits: tuple[float, float] | None
    event_constraints: tuple[str, ...] = ()
    backend_mapping: str | None = None


@dataclass(frozen=True, slots=True)
class ControlPolicySpec:
    policy_id: str
    policy_type: str
    sequential: bool
    channels: tuple[ControlChannelSpec, ...] = ()


@dataclass(frozen=True, slots=True)
class EnvironmentSpec:
    environment_id: str
    atmosphere_model_id: str | None
    gravity_model_id: str | None
    wind_model_id: str | None
    temperature_profile_id: str | None
    density_profile_id: str | None
    turbulence_profile_id: str | None
    terrain_or_reference_surface_id: str | None
    coordinate_frame: str


@dataclass(frozen=True, slots=True)
class TrajectoryRun:
    run_id: str
    backend_id: str
    scenario_id: str
    seed: int
    success: bool
    failure_reason: str | None
    times: tuple[float, ...]
    truth_state: dict[str, tuple[float, ...]]
    observations: dict[str, tuple[float, ...]]
    controls: dict[str, tuple[float, ...]] = field(default_factory=dict)
    environment_trace: dict[str, tuple[float, ...]] = field(default_factory=dict)
    events: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrajectoryBackendCapabilities:
    backend_id: str
    display_name: str
    family: str
    dimensionality: str
    fidelity: str
    input_modes: tuple[str, ...]
    supports_environment: bool
    supports_sequential_control: bool
    supports_events: bool
    supports_stochastic_runs: bool
    runtime_class: str
    determinism: str
    state_outputs: tuple[str, ...]
    observation_outputs: tuple[str, ...]
    event_outputs: tuple[str, ...]
    valid_search_methods: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BackendContractDefinition:
    capabilities: TrajectoryBackendCapabilities
    scenario_spec: ScenarioSpec
    design_variables: tuple[DesignVariableSpec, ...]
    control_policy: ControlPolicySpec
    environment_spec: EnvironmentSpec
    example_run: TrajectoryRun


@dataclass(frozen=True, slots=True)
class TrajectoryBackendContractResult:
    backend_contract: dict[str, Any]
    backend_capability_schema: dict[str, Any]
    scenario_spec_schema: dict[str, Any]
    design_variable_schema: dict[str, Any]
    control_policy_schema: dict[str, Any]
    environment_spec_schema: dict[str, Any]
    trajectory_run_schema: dict[str, Any]
    capability_matrix_rows: tuple[dict[str, Any], ...]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class TrajectoryBackendContractArtifacts:
    run_dir: Path
    backend_contract_path: Path
    backend_capability_schema_path: Path
    scenario_spec_schema_path: Path
    design_variable_schema_path: Path
    control_policy_schema_path: Path
    environment_spec_schema_path: Path
    trajectory_run_schema_path: Path
    capability_matrix_csv_path: Path
    capability_matrix_png_path: Path
    report_path: Path


def _validate_series_length(name: str, series: dict[str, tuple[float, ...]], expected_length: int) -> list[str]:
    errors: list[str] = []
    for field_name, values in sorted(series.items()):
        if len(values) != expected_length:
            errors.append(f"{name}.{field_name} must match times length")
    return errors


def validate_trajectory_run(run: TrajectoryRun) -> list[str]:
    errors: list[str] = []
    if not run.run_id:
        errors.append("run_id is required")
    if not run.backend_id:
        errors.append("backend_id is required")
    if not run.scenario_id:
        errors.append("scenario_id is required")
    if len(run.times) == 0:
        errors.append("times must not be empty")
    for index, value in enumerate(run.times):
        if not isinstance(value, (int, float)):
            errors.append(f"time[{index}] must be numeric")
        if index > 0 and run.times[index] <= run.times[index - 1]:
            errors.append("times must be strictly increasing")
            break
    errors.extend(_validate_series_length("truth_state", run.truth_state, len(run.times)))
    errors.extend(_validate_series_length("observations", run.observations, len(run.times)))
    errors.extend(_validate_series_length("controls", run.controls, len(run.times)))
    errors.extend(_validate_series_length("environment_trace", run.environment_trace, len(run.times)))
    return errors


def validate_backend_capabilities(capabilities: TrajectoryBackendCapabilities) -> list[str]:
    errors: list[str] = []
    if not capabilities.backend_id:
        errors.append("backend_id is required")
    if capabilities.runtime_class not in {"cheap", "medium", "expensive"}:
        errors.append("runtime_class must be cheap, medium, or expensive")
    if capabilities.determinism not in {"seeded", "deterministic", "nondeterministic"}:
        errors.append("determinism must be seeded, deterministic, or nondeterministic")
    if not capabilities.input_modes:
        errors.append("input_modes must not be empty")
    if not capabilities.state_outputs:
        errors.append("state_outputs must not be empty")
    if not capabilities.observation_outputs:
        errors.append("observation_outputs must not be empty")
    if capabilities.supports_sequential_control and "control_schedule" not in capabilities.input_modes and "piecewise_controls" not in capabilities.input_modes:
        errors.append("sequential-control backends must declare a control-capable input mode")
    return errors


def validate_backend_contract_definition(definition: BackendContractDefinition) -> list[str]:
    errors = validate_backend_capabilities(definition.capabilities)
    if definition.scenario_spec.backend_id != definition.capabilities.backend_id:
        errors.append("scenario_spec.backend_id must match capabilities.backend_id")
    if definition.example_run.backend_id != definition.capabilities.backend_id:
        errors.append("example_run.backend_id must match capabilities.backend_id")
    if definition.example_run.scenario_id != definition.scenario_spec.scenario_id:
        errors.append("example_run.scenario_id must match scenario_spec.scenario_id")
    if definition.capabilities.supports_environment != bool(definition.environment_spec.atmosphere_model_id or definition.environment_spec.wind_model_id or definition.environment_spec.density_profile_id):
        if definition.capabilities.supports_environment:
            errors.append("environment-aware backends must declare nontrivial environment spec")
    if definition.capabilities.supports_sequential_control != definition.control_policy.sequential:
        errors.append("control_policy.sequential must match supports_sequential_control")
    errors.extend(validate_trajectory_run(definition.example_run))
    return errors


def _json_schema(title: str, required: tuple[str, ...], properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": title,
        "type": "object",
        "required": list(required),
        "properties": properties,
        "additionalProperties": True,
    }


def _backend_capability_schema() -> dict[str, Any]:
    return _json_schema(
        "TrajectoryBackendCapabilities",
        (
            "backend_id",
            "display_name",
            "family",
            "dimensionality",
            "fidelity",
            "input_modes",
            "supports_environment",
            "supports_sequential_control",
            "supports_events",
            "supports_stochastic_runs",
            "runtime_class",
            "determinism",
            "state_outputs",
            "observation_outputs",
            "event_outputs",
            "valid_search_methods",
        ),
        {
            "backend_id": {"type": "string"},
            "display_name": {"type": "string"},
            "family": {"type": "string"},
            "dimensionality": {"type": "string"},
            "fidelity": {"type": "string"},
            "input_modes": {"type": "array", "items": {"type": "string"}},
            "supports_environment": {"type": "boolean"},
            "supports_sequential_control": {"type": "boolean"},
            "supports_events": {"type": "boolean"},
            "supports_stochastic_runs": {"type": "boolean"},
            "runtime_class": {"type": "string"},
            "determinism": {"type": "string"},
            "state_outputs": {"type": "array", "items": {"type": "string"}},
            "observation_outputs": {"type": "array", "items": {"type": "string"}},
            "event_outputs": {"type": "array", "items": {"type": "string"}},
            "valid_search_methods": {"type": "array", "items": {"type": "string"}},
        },
    )


def _scenario_spec_schema() -> dict[str, Any]:
    return _json_schema(
        "ScenarioSpec",
        (
            "scenario_id",
            "backend_id",
            "vehicle_or_model_id",
            "scenario_family",
            "difficulty_tier",
            "environment_id",
            "sensor_regime_id",
        ),
        {
            "scenario_id": {"type": "string"},
            "backend_id": {"type": "string"},
            "vehicle_or_model_id": {"type": "string"},
            "target_class": {"type": ["string", "null"]},
            "target_class_pair": {"type": ["array", "null"], "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
            "scenario_family": {"type": "string"},
            "difficulty_tier": {"type": "string"},
            "environment_id": {"type": "string"},
            "sensor_regime_id": {"type": "string"},
            "validity_constraints": {"type": "object"},
        },
    )


def _design_variable_schema() -> dict[str, Any]:
    return _json_schema(
        "DesignVariableSpec",
        (
            "name",
            "variable_type",
            "units",
            "sampling_distribution",
            "is_class_defining",
            "is_environmental",
            "is_control_related",
            "is_sensitive_for_leakage",
        ),
        {
            "name": {"type": "string"},
            "variable_type": {"type": "string"},
            "units": {"type": "string"},
            "bounds": {"type": ["array", "null"], "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
            "sampling_distribution": {"type": "string"},
            "is_class_defining": {"type": "boolean"},
            "is_environmental": {"type": "boolean"},
            "is_control_related": {"type": "boolean"},
            "is_sensitive_for_leakage": {"type": "boolean"},
        },
    )


def _control_policy_schema() -> dict[str, Any]:
    return _json_schema(
        "ControlPolicySpec",
        ("policy_id", "policy_type", "sequential", "channels"),
        {
            "policy_id": {"type": "string"},
            "policy_type": {"type": "string"},
            "sequential": {"type": "boolean"},
            "channels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["control_name", "units", "event_constraints"],
                    "properties": {
                        "control_name": {"type": "string"},
                        "units": {"type": "string"},
                        "bounds": {"type": ["array", "null"], "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                        "rate_limits": {"type": ["array", "null"], "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                        "event_constraints": {"type": "array", "items": {"type": "string"}},
                        "backend_mapping": {"type": ["string", "null"]},
                    },
                },
            },
        },
    )


def _environment_spec_schema() -> dict[str, Any]:
    return _json_schema(
        "EnvironmentSpec",
        ("environment_id", "coordinate_frame"),
        {
            "environment_id": {"type": "string"},
            "atmosphere_model_id": {"type": ["string", "null"]},
            "gravity_model_id": {"type": ["string", "null"]},
            "wind_model_id": {"type": ["string", "null"]},
            "temperature_profile_id": {"type": ["string", "null"]},
            "density_profile_id": {"type": ["string", "null"]},
            "turbulence_profile_id": {"type": ["string", "null"]},
            "terrain_or_reference_surface_id": {"type": ["string", "null"]},
            "coordinate_frame": {"type": "string"},
        },
    )


def _trajectory_run_schema() -> dict[str, Any]:
    return _json_schema(
        "TrajectoryRun",
        (
            "run_id",
            "backend_id",
            "scenario_id",
            "seed",
            "success",
            "times",
            "truth_state",
            "observations",
            "events",
            "metadata",
        ),
        {
            "run_id": {"type": "string"},
            "backend_id": {"type": "string"},
            "scenario_id": {"type": "string"},
            "seed": {"type": "integer"},
            "success": {"type": "boolean"},
            "failure_reason": {"type": ["string", "null"]},
            "times": {"type": "array", "items": {"type": "number"}},
            "truth_state": {"type": "object"},
            "observations": {"type": "object"},
            "controls": {"type": "object"},
            "environment_trace": {"type": "object"},
            "events": {"type": "array", "items": {"type": "object"}},
            "metadata": {"type": "object"},
        },
    )


def _example_times() -> tuple[float, ...]:
    return (0.0, 0.5, 1.0, 1.5, 2.0)


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
            DesignVariableSpec("initial_position", "float", "m", (-2.0, 2.0), "uniform", False, False, False, False),
            DesignVariableSpec("initial_velocity", "float", "m/s", (0.1, 2.5), "uniform", True, False, False, False),
            DesignVariableSpec("duration", "float", "s", (1.0, 4.0), "uniform", False, False, False, True),
            DesignVariableSpec("measurement_std", "float", "m", (0.01, 0.35), "uniform", False, False, False, True),
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
            DesignVariableSpec("initial_position", "float", "m", (-1.0, 1.0), "uniform", False, False, False, False),
            DesignVariableSpec("initial_velocity", "float", "m/s", (0.5, 2.5), "uniform", True, False, False, False),
            DesignVariableSpec("switch_time", "float", "s", (0.4, 1.4), "uniform", True, False, True, False),
        ),
        control_policy=ControlPolicySpec(
            policy_id="piecewise_acceleration_schedule",
            policy_type="piecewise_constant_control",
            sequential=True,
            channels=(
                ControlChannelSpec("acceleration_command", "m/s^2", (-2.5, 2.5), (-5.0, 5.0), ("switch_time_defined",), "backend.accel_schedule"),
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
            DesignVariableSpec("initial_position", "float", "m", (0.0, 1.0), "uniform", False, False, False, False),
            DesignVariableSpec("initial_velocity", "float", "m/s", (0.3, 2.0), "uniform", True, False, False, False),
            DesignVariableSpec("drag_coefficient", "float", "1", (0.0, 0.8), "uniform", False, True, False, True),
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
            DesignVariableSpec("input_deck_hash", "string_token", "sha256", None, "fixed", False, False, False, False),
            DesignVariableSpec("maneuver_magnitude", "float", "m/s^2", (0.2, 1.2), "lhs", True, False, True, False),
            DesignVariableSpec("sample_period", "float", "s", (0.25, 0.75), "lhs", False, False, False, True),
        ),
        control_policy=ControlPolicySpec(
            policy_id="scheduled_external_controls",
            policy_type="scheduled_control",
            sequential=True,
            channels=(
                ControlChannelSpec("longitudinal_command", "normalized", (-1.0, 1.0), (-2.0, 2.0), ("phase_change_allowed",), "input_deck.controls.longitudinal"),
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


def _render_capability_matrix_png(rows: tuple[dict[str, Any], ...]) -> bytes:
    metric_names = (
        "supports_environment",
        "supports_sequential_control",
        "supports_events",
        "supports_stochastic_runs",
        "state_output_count",
        "observation_output_count",
        "event_output_count",
        "search_method_count",
    )
    data = [[float(row[name]) for name in metric_names] for row in rows]
    backend_labels = [str(row["family"]) for row in rows]

    fig, ax = plt.subplots(figsize=(10, 3.6))
    image = ax.imshow(data, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(metric_names)), labels=[name.replace("_", "\n") for name in metric_names], fontsize=8)
    ax.set_yticks(range(len(backend_labels)), labels=backend_labels, fontsize=9)
    ax.set_title("Backend Capability Matrix")
    for row_index, row_values in enumerate(data):
        for column_index, value in enumerate(row_values):
            ax.text(column_index, row_index, f"{value:.0f}", ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.04)
    fig.tight_layout()

    from io import BytesIO

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


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
                "capabilities": asdict(definition.capabilities),
                "scenario_spec": asdict(definition.scenario_spec),
                "design_variables": [asdict(variable) for variable in definition.design_variables],
                "control_policy": asdict(definition.control_policy),
                "environment_spec": asdict(definition.environment_spec),
                "example_run": asdict(definition.example_run),
            }
            for definition in definitions
        ],
        "validation": validation_rows,
    }
    capability_rows = _capability_matrix_rows(definitions)
    valid_count = sum(1 for row in validation_rows if row["valid"])
    sequential_count = sum(1 for definition in definitions if definition.capabilities.supports_sequential_control)
    environment_count = sum(1 for definition in definitions if definition.capabilities.supports_environment)

    report_markdown = "\n".join(
        [
            "# Trajectory Backend Contract",
            "",
            "## Summary",
            f"- backend families declared: `{len(definitions)}`",
            f"- fully valid contract declarations: `{valid_count}`",
            f"- sequential-control backends: `{sequential_count}`",
            f"- environment-aware backends: `{environment_count}`",
            "",
            "## Backend Families",
            "| Backend | Family | Runtime | Sequential | Environment | Search Methods |",
            "| --- | --- | --- | --- | --- | --- |",
            *[
                f"| `{definition.capabilities.display_name}` | `{definition.capabilities.family}` | `{definition.capabilities.runtime_class}` | "
                f"`{definition.capabilities.supports_sequential_control}` | `{definition.capabilities.supports_environment}` | "
                f"`{', '.join(definition.capabilities.valid_search_methods)}` |"
                for definition in definitions
            ],
            "",
            "## Relationship Diagram",
            "```mermaid",
            'graph TD',
            '    A["ScenarioSpec"] --> F["TrajectoryBackend Adapter"]',
            '    B["DesignVariableSpec"] --> F',
            '    C["ControlPolicySpec"] --> F',
            '    D["EnvironmentSpec"] --> F',
            '    E["TrajectoryBackendCapabilities"] --> F',
            '    F --> G["TrajectoryRun"]',
            '    G --> H["Features / Labels / Classifiers"]',
            '    H --> I["Search / Archive / Corpus Selection"]',
            "```",
            "",
            "## Notes",
            "- This milestone defines the typed contract only. It does not yet prove full adapter execution across multiple engines.",
            "- The common search layer should only inspect capability descriptors and normalized `TrajectoryRun` objects, never simulator-specific control names directly.",
            "- The mock file backend exists to prove future file-in/file-out simulator integration patterns without binding to a real external tool yet.",
        ]
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


def write_trajectory_backend_contract_artifacts(
    base_dir: str | Path,
    *,
    result: TrajectoryBackendContractResult | None = None,
) -> TrajectoryBackendContractArtifacts:
    run_dir = Path(base_dir) / "trajectory_backend_contract"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = result or analyze_trajectory_backend_contract()

    backend_contract_path = run_dir / "backend_contract.json"
    backend_capability_schema_path = run_dir / "backend_capability_schema.json"
    scenario_spec_schema_path = run_dir / "scenario_spec_schema.json"
    design_variable_schema_path = run_dir / "design_variable_schema.json"
    control_policy_schema_path = run_dir / "control_policy_schema.json"
    environment_spec_schema_path = run_dir / "environment_spec_schema.json"
    trajectory_run_schema_path = run_dir / "trajectory_run_schema.json"
    capability_matrix_csv_path = run_dir / "capability_matrix.csv"
    capability_matrix_png_path = run_dir / "capability_matrix.png"
    report_path = run_dir / "backend_contract_report.md"

    backend_contract_path.write_text(json.dumps(payload.backend_contract, indent=2), encoding="utf-8")
    backend_capability_schema_path.write_text(json.dumps(payload.backend_capability_schema, indent=2), encoding="utf-8")
    scenario_spec_schema_path.write_text(json.dumps(payload.scenario_spec_schema, indent=2), encoding="utf-8")
    design_variable_schema_path.write_text(json.dumps(payload.design_variable_schema, indent=2), encoding="utf-8")
    control_policy_schema_path.write_text(json.dumps(payload.control_policy_schema, indent=2), encoding="utf-8")
    environment_spec_schema_path.write_text(json.dumps(payload.environment_spec_schema, indent=2), encoding="utf-8")
    trajectory_run_schema_path.write_text(json.dumps(payload.trajectory_run_schema, indent=2), encoding="utf-8")
    report_path.write_text(payload.report_markdown, encoding="utf-8")

    fieldnames = list(payload.capability_matrix_rows[0].keys()) if payload.capability_matrix_rows else []
    with capability_matrix_csv_path.open("w", encoding="utf-8", newline="") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload.capability_matrix_rows:
            writer.writerow(row)

    capability_matrix_png_path.write_bytes(_render_capability_matrix_png(payload.capability_matrix_rows))

    return TrajectoryBackendContractArtifacts(
        run_dir=run_dir,
        backend_contract_path=backend_contract_path,
        backend_capability_schema_path=backend_capability_schema_path,
        scenario_spec_schema_path=scenario_spec_schema_path,
        design_variable_schema_path=design_variable_schema_path,
        control_policy_schema_path=control_policy_schema_path,
        environment_spec_schema_path=environment_spec_schema_path,
        trajectory_run_schema_path=trajectory_run_schema_path,
        capability_matrix_csv_path=capability_matrix_csv_path,
        capability_matrix_png_path=capability_matrix_png_path,
        report_path=report_path,
    )
