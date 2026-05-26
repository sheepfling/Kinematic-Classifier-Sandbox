from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ScenarioSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_id: str
    backend_id: str
    vehicle_or_model_id: str
    target_class: str | None
    target_class_pair: tuple[str, str] | None
    scenario_family: str
    difficulty_tier: str
    environment_id: str
    sensor_regime_id: str
    validity_constraints: dict[str, Any] = Field(default_factory=dict)


class DesignVariableSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    variable_type: str
    units: str
    bounds: tuple[float, float] | None
    sampling_distribution: str
    is_class_defining: bool
    is_environmental: bool
    is_control_related: bool
    is_sensitive_for_leakage: bool


class ControlChannelSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    control_name: str
    units: str
    bounds: tuple[float, float] | None
    rate_limits: tuple[float, float] | None
    event_constraints: tuple[str, ...] = Field(default_factory=tuple)
    backend_mapping: str | None = None


class ControlPolicySpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    policy_id: str
    policy_type: str
    sequential: bool
    channels: tuple[ControlChannelSpec, ...] = Field(default_factory=tuple)


class EnvironmentSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    environment_id: str
    atmosphere_model_id: str | None
    gravity_model_id: str | None
    wind_model_id: str | None
    temperature_profile_id: str | None
    density_profile_id: str | None
    turbulence_profile_id: str | None
    terrain_or_reference_surface_id: str | None
    coordinate_frame: str


class TrajectoryRun(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    backend_id: str
    scenario_id: str
    seed: int
    success: bool
    failure_reason: str | None
    times: tuple[float, ...]
    truth_state: dict[str, tuple[float, ...]]
    observations: dict[str, tuple[float, ...]]
    controls: dict[str, tuple[float, ...]] = Field(default_factory=dict)
    environment_trace: dict[str, tuple[float, ...]] = Field(default_factory=dict)
    events: tuple[dict[str, Any], ...] = Field(default_factory=tuple)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrajectoryBackendCapabilities(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

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


class BackendContractDefinition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    capabilities: TrajectoryBackendCapabilities
    scenario_spec: ScenarioSpec
    design_variables: tuple[DesignVariableSpec, ...]
    control_policy: ControlPolicySpec
    environment_spec: EnvironmentSpec
    example_run: TrajectoryRun


class TrajectoryBackendContractResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    backend_contract: dict[str, Any]
    backend_capability_schema: dict[str, Any]
    scenario_spec_schema: dict[str, Any]
    design_variable_schema: dict[str, Any]
    control_policy_schema: dict[str, Any]
    environment_spec_schema: dict[str, Any]
    trajectory_run_schema: dict[str, Any]
    capability_matrix_rows: tuple[dict[str, Any], ...]
    report_markdown: str
