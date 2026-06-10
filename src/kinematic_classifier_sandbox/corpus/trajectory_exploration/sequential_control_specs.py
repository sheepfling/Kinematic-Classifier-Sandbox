from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True, slots=True)
class StateChannelSpec:
    name: str
    units: str
    semantic_role: str


@dataclass(frozen=True, slots=True)
class ObservationChannelSpec:
    name: str
    units: str
    source: str


@dataclass(frozen=True, slots=True)
class ControlChannelSpec:
    name: str
    units: str
    lower_bound: float
    upper_bound: float
    semantic_role: str


@dataclass(frozen=True, slots=True)
class SequentialControlProblemSpec:
    problem_id: str
    vehicle_family: str
    implementation_status: str
    geometry: str
    dynamics_family: str
    measurement_family: str
    state_channels: tuple[StateChannelSpec, ...]
    observation_channels: tuple[ObservationChannelSpec, ...]
    control_channels: tuple[ControlChannelSpec, ...]
    transition_path: tuple[str, ...]
    adapter_requirements: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def as_payload(self) -> dict[str, object]:
        return asdict(self)


def default_one_dimensional_acceleration_problem_spec() -> SequentialControlProblemSpec:
    return SequentialControlProblemSpec(
        problem_id="point_mass_1d_acceleration_control",
        vehicle_family="point_mass",
        implementation_status="implemented",
        geometry="1d_scalar",
        dynamics_family="constant_dt_point_mass",
        measurement_family="position_only",
        state_channels=(
            StateChannelSpec("position", "m", "translational_position"),
            StateChannelSpec("velocity", "m/s", "translational_velocity"),
            StateChannelSpec("acceleration", "m/s^2", "translational_acceleration"),
        ),
        observation_channels=(
            ObservationChannelSpec("position_measurement", "m", "direct_position_sensor"),
        ),
        control_channels=(
            ControlChannelSpec("acceleration_command", "normalized", -1.0, 1.0, "longitudinal_acceleration"),
        ),
        transition_path=(
            "replace scalar position/velocity/acceleration with vector-valued translational state",
            "add vector-compatible feature extractors and geometry-aware boundary objectives",
            "reuse posterior, artifact, checkpoint, and objective-generation machinery",
        ),
        adapter_requirements=(
            "vector state adapter",
            "vector feature context adapter",
            "vehicle-specific dynamics backend",
        ),
        notes=(
            "This is the minimal witness proving sequential control, resumable PPO artifacts, and generated objective targeting.",
        ),
    )


def default_three_dimensional_point_mass_problem_spec() -> SequentialControlProblemSpec:
    return SequentialControlProblemSpec(
        problem_id="point_mass_3d_acceleration_control",
        vehicle_family="point_mass",
        implementation_status="planned_adapter",
        geometry="3d_vector",
        dynamics_family="constant_dt_point_mass",
        measurement_family="position_or_position_velocity",
        state_channels=(
            StateChannelSpec("position_x", "m", "translational_position"),
            StateChannelSpec("position_y", "m", "translational_position"),
            StateChannelSpec("position_z", "m", "translational_position"),
            StateChannelSpec("velocity_x", "m/s", "translational_velocity"),
            StateChannelSpec("velocity_y", "m/s", "translational_velocity"),
            StateChannelSpec("velocity_z", "m/s", "translational_velocity"),
            StateChannelSpec("acceleration_x", "m/s^2", "translational_acceleration"),
            StateChannelSpec("acceleration_y", "m/s^2", "translational_acceleration"),
            StateChannelSpec("acceleration_z", "m/s^2", "translational_acceleration"),
        ),
        observation_channels=(
            ObservationChannelSpec("position_x_measurement", "m", "direct_position_sensor"),
            ObservationChannelSpec("position_y_measurement", "m", "direct_position_sensor"),
            ObservationChannelSpec("position_z_measurement", "m", "direct_position_sensor"),
        ),
        control_channels=(
            ControlChannelSpec("acceleration_x_command", "normalized", -1.0, 1.0, "body_or_world_axis_acceleration"),
            ControlChannelSpec("acceleration_y_command", "normalized", -1.0, 1.0, "body_or_world_axis_acceleration"),
            ControlChannelSpec("acceleration_z_command", "normalized", -1.0, 1.0, "body_or_world_axis_acceleration"),
        ),
        transition_path=(
            "swap the 1D state backend for a 3D vector backend",
            "lift objectives from scalar feature cells to vector-aware feature cells and class regions",
            "reuse checkpoint/resume, artifact capture, and generated objective suites",
        ),
        adapter_requirements=(
            "3D state representation adapter",
            "3D measurement adapter",
            "vector feature extractor registry",
        ),
        notes=(
            "This is the direct dimensional lift target for the current point-mass witness.",
        ),
    )


def default_air_vehicle_control_problem_spec() -> SequentialControlProblemSpec:
    return SequentialControlProblemSpec(
        problem_id="air_vehicle_surface_control_placeholder",
        vehicle_family="aerodynamic_vehicle",
        implementation_status="planned_adapter",
        geometry="3d_vector_with_attitude",
        dynamics_family="vehicle_specific_longitudinal_or_six_dof",
        measurement_family="navigation_plus_vehicle_state",
        state_channels=(
            StateChannelSpec("position_x", "m", "translational_position"),
            StateChannelSpec("position_y", "m", "translational_position"),
            StateChannelSpec("position_z", "m", "translational_position"),
            StateChannelSpec("velocity", "m/s", "airspeed_or_groundspeed"),
            StateChannelSpec("flight_path_angle", "rad", "attitude_or_trajectory_angle"),
            StateChannelSpec("heading", "rad", "attitude_or_trajectory_angle"),
        ),
        observation_channels=(
            ObservationChannelSpec("position", "m", "navigation_solution"),
            ObservationChannelSpec("velocity", "m/s", "navigation_solution"),
            ObservationChannelSpec("attitude_proxy", "rad", "vehicle_state_estimate"),
        ),
        control_channels=(
            ControlChannelSpec("angle_of_attack_rate_command", "normalized", -1.0, 1.0, "lift_or_pitch_surface_command"),
            ControlChannelSpec("bank_rate_command", "normalized", -1.0, 1.0, "roll_surface_command"),
            ControlChannelSpec("throttle_command", "normalized", 0.0, 1.0, "propulsion_command"),
        ),
        transition_path=(
            "replace point-mass integrator with vehicle-specific dynamics adapter",
            "bind generated objective targets to aerodynamic or mission feature cells",
            "preserve PPO/CEM runners, artifact schema, and checkpoint-resume behavior",
        ),
        adapter_requirements=(
            "air-vehicle dynamics adapter",
            "surface-command normalization adapter",
            "vehicle-state feature extractor registry",
        ),
        notes=(
            "This is a pathway contract, not a claim that aerodynamic dynamics are implemented today.",
        ),
    )


def default_sequential_control_problem_catalog() -> tuple[SequentialControlProblemSpec, ...]:
    return (
        default_one_dimensional_acceleration_problem_spec(),
        default_three_dimensional_point_mass_problem_spec(),
        default_air_vehicle_control_problem_spec(),
    )
