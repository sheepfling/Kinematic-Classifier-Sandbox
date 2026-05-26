from __future__ import annotations

from typing import Any

from .trajectory_backend_contract_types import (
    BackendContractDefinition,
    ControlPolicySpec,
    DesignVariableSpec,
    EnvironmentSpec,
    ScenarioSpec,
    TrajectoryBackendCapabilities,
    TrajectoryRun,
)


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


def _backend_capability_schema() -> dict[str, Any]:
    return TrajectoryBackendCapabilities.model_json_schema()


def _scenario_spec_schema() -> dict[str, Any]:
    return ScenarioSpec.model_json_schema()


def _design_variable_schema() -> dict[str, Any]:
    return DesignVariableSpec.model_json_schema()


def _control_policy_schema() -> dict[str, Any]:
    return ControlPolicySpec.model_json_schema()


def _environment_spec_schema() -> dict[str, Any]:
    return EnvironmentSpec.model_json_schema()


def _trajectory_run_schema() -> dict[str, Any]:
    return TrajectoryRun.model_json_schema()


def _example_times() -> tuple[float, ...]:
    return (0.0, 0.5, 1.0, 1.5, 2.0)


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
