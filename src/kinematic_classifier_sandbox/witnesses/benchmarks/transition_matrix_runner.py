from __future__ import annotations

from dataclasses import dataclass
from math import log, pi

from ...trajectory_generator import generate_switching_scenarios
from ...utils.math import (
    _gaussian_logpdf as _sigma_to_variance_gaussian_logpdf,
)
from ...utils.math import (
    _matmul,
    _matvec,
    _mean,
    _normalize_log_scores,
    _transpose,
)
from .transition_matrix_contracts import (
    SwitchingModeSpec,
    SwitchingScenario,
    TransitionBenchmarkResult,
    TransitionBenchmarkSummary,
    TransitionPosteriorStep,
    TransitionRun,
)


def default_switching_mode_specs() -> tuple[SwitchingModeSpec, ...]:
    return (
        SwitchingModeSpec("stationary", mean_speed=0.0, sigma_speed=0.18, mean_accel=0.0, sigma_accel=0.18, mean_abs_accel=0.0, sigma_abs_accel=0.15, prior_weight=0.25),
        SwitchingModeSpec("constant_velocity", mean_speed=1.10, sigma_speed=0.35, mean_accel=0.0, sigma_accel=0.20, mean_abs_accel=0.05, sigma_abs_accel=0.12, prior_weight=0.25),
        SwitchingModeSpec("braking", mean_speed=1.00, sigma_speed=0.45, mean_accel=-0.80, sigma_accel=0.28, mean_abs_accel=0.80, sigma_abs_accel=0.25, prior_weight=0.25),
        SwitchingModeSpec("maneuver", mean_speed=1.00, sigma_speed=0.45, mean_accel=0.0, sigma_accel=0.85, mean_abs_accel=0.70, sigma_abs_accel=0.30, prior_weight=0.25),
    )


def default_transition_matrix() -> dict[str, dict[str, float]]:
    return {
        "stationary": {"stationary": 0.90, "constant_velocity": 0.10, "braking": 0.0, "maneuver": 0.0},
        "constant_velocity": {"stationary": 0.02, "constant_velocity": 0.84, "braking": 0.08, "maneuver": 0.06},
        "braking": {"stationary": 0.08, "constant_velocity": 0.10, "braking": 0.80, "maneuver": 0.02},
        "maneuver": {"stationary": 0.02, "constant_velocity": 0.12, "braking": 0.06, "maneuver": 0.80},
    }


def _true_mode_series(times: tuple[float, ...], *, segment_modes: list[str], switch_time: float) -> tuple[str, ...]:
    mode_a, mode_b = segment_modes
    return tuple(mode_a if time < switch_time else mode_b for time in times)


def generate_transition_switching_scenarios(*, seed: int = 7, replicas: int = 8) -> tuple[SwitchingScenario, ...]:
    scenarios: list[SwitchingScenario] = []
    for replica in range(replicas):
        for artifact in generate_switching_scenarios(seed=seed + replica * 31):
            params = artifact.generator_parameters
            segment_modes = list(params["segment_modes"])
            switch_time = float(params["switch_time"])
            scenarios.append(
                SwitchingScenario(
                    trajectory_id=f"{artifact.trajectory_id}_{replica}",
                    scenario_name=artifact.scenario_id,
                    seed=artifact.seed,
                    times=artifact.times,
                    measurements=artifact.measurements,
                    true_mode_by_step=_true_mode_series(artifact.times, segment_modes=segment_modes, switch_time=switch_time),
                )
            )
    return tuple(scenarios)


def _speed_and_accel(measurements: tuple[float, ...], times: tuple[float, ...], index: int) -> tuple[float, float]:
    if index == 0:
        return 0.0, 0.0
    dt = max(times[index] - times[index - 1], 1e-9)
    speed = (measurements[index] - measurements[index - 1]) / dt
    if index < 2:
        return speed, 0.0
    prev_dt = max(times[index - 1] - times[index - 2], 1e-9)
    prev_speed = (measurements[index - 1] - measurements[index - 2]) / prev_dt
    accel = (speed - prev_speed) / max(times[index] - times[index - 1], 1e-9)
    return speed, accel


def _emission_log_scores(
    specs: tuple[SwitchingModeSpec, ...],
    *,
    speed: float,
    accel: float,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    abs_accel = abs(accel)
    for spec in specs:
        scores[spec.name] = (
            _sigma_to_variance_gaussian_logpdf(speed, spec.mean_speed, spec.sigma_speed * spec.sigma_speed)
            + _sigma_to_variance_gaussian_logpdf(accel, spec.mean_accel, spec.sigma_accel * spec.sigma_accel)
            + _sigma_to_variance_gaussian_logpdf(abs_accel, spec.mean_abs_accel, spec.sigma_abs_accel * spec.sigma_abs_accel)
        )
    return scores


def _emission_term_breakdown(spec: SwitchingModeSpec, *, speed: float, accel: float) -> dict[str, float]:
    abs_accel = abs(accel)
    speed_term = _sigma_to_variance_gaussian_logpdf(speed, spec.mean_speed, spec.sigma_speed * spec.sigma_speed)
    accel_term = _sigma_to_variance_gaussian_logpdf(accel, spec.mean_accel, spec.sigma_accel * spec.sigma_accel)
    abs_accel_term = _sigma_to_variance_gaussian_logpdf(abs_accel, spec.mean_abs_accel, spec.sigma_abs_accel * spec.sigma_abs_accel)
    return {
        "speed_term": speed_term,
        "accel_term": accel_term,
        "abs_accel_term": abs_accel_term,
        "emission_total": speed_term + accel_term + abs_accel_term,
    }


def _run_mode_accumulator(
    scenario: SwitchingScenario,
    specs: tuple[SwitchingModeSpec, ...],
    *,
    mode: str,
    transition_matrix: dict[str, dict[str, float]] | None = None,
) -> TransitionRun:
    total_prior = sum(spec.prior_weight for spec in specs)
    posterior = {spec.name: spec.prior_weight / total_prior for spec in specs}
    steps: list[TransitionPosteriorStep] = []
    for index, (time, measurement, true_mode) in enumerate(zip(scenario.times, scenario.measurements, scenario.true_mode_by_step)):
        speed, accel = _speed_and_accel(scenario.measurements, scenario.times, index)
        emission = _emission_log_scores(specs, speed=speed, accel=accel)
        prior = dict(posterior)
        if transition_matrix is not None:
            transitioned: dict[str, float] = {}
            for target in prior:
                transitioned[target] = sum(prior[source] * transition_matrix[source][target] for source in prior)
            prior = transitioned
        log_scores = {name: log(max(prior[name], 1e-12)) + emission[name] for name in emission}
        posterior = _normalize_log_scores(log_scores)
        predicted = max(posterior, key=posterior.get)
        steps.append(
            TransitionPosteriorStep(
                step=index,
                time=time,
                measurement=measurement,
                estimated_speed=speed,
                estimated_accel=accel,
                prior_weights=prior,
                posterior_weights=posterior,
                predicted_mode=predicted,
                true_mode=true_mode,
                confidence=posterior[predicted],
            )
        )
    switch_index = next((idx for idx, step in enumerate(steps) if step.true_mode != steps[0].true_mode), len(steps))
    post_switch_steps = steps[switch_index:] if switch_index < len(steps) else steps[-1:]
    return TransitionRun(
        trajectory_id=scenario.trajectory_id,
        scenario_name=scenario.scenario_name,
        mode=mode,
        steps=tuple(steps),
        final_weights=posterior,
        final_predicted_mode=max(posterior, key=posterior.get),
        accuracy=_mean([1.0 if step.predicted_mode == step.true_mode else 0.0 for step in steps]),
        post_switch_accuracy=_mean([1.0 if step.predicted_mode == step.true_mode else 0.0 for step in post_switch_steps]),
    )


@dataclass
class _KalmanModeState:
    x: list[float]
    p: list[list[float]]


def _kalman_update_scalar(
    state: _KalmanModeState,
    *,
    f: list[list[float]],
    q: list[list[float]],
    h: list[float],
    measurement: float,
    r: float,
    control: list[float] | None = None,
) -> tuple[_KalmanModeState, float]:
    x_pred = _matvec(f, state.x)
    if control is not None:
        x_pred = [value + control[index] for index, value in enumerate(x_pred)]
    fp = _matmul(f, state.p)
    p_pred = _matmul(fp, _transpose(f))
    for row_index in range(len(p_pred)):
        for col_index in range(len(p_pred[row_index])):
            p_pred[row_index][col_index] += q[row_index][col_index]
    innovation = measurement - sum(h[index] * x_pred[index] for index in range(len(h)))
    innovation_cov = sum(h[row] * sum(p_pred[row][col] * h[col] for col in range(len(h))) for row in range(len(h))) + r
    kalman_gain = [
        sum(p_pred[row][col] * h[col] for col in range(len(h))) / max(innovation_cov, 1e-9)
        for row in range(len(h))
    ]
    x_upd = [x_pred[index] + kalman_gain[index] * innovation for index in range(len(x_pred))]
    kh = [[kalman_gain[row] * h[col] for col in range(len(h))] for row in range(len(kalman_gain))]
    identity = [[1.0 if row == col else 0.0 for col in range(len(h))] for row in range(len(h))]
    i_minus_kh = [[identity[row][col] - kh[row][col] for col in range(len(h))] for row in range(len(h))]
    p_upd = _matmul(i_minus_kh, p_pred)
    log_likelihood = -0.5 * (log(2.0 * pi * max(innovation_cov, 1e-9)) + (innovation * innovation) / max(innovation_cov, 1e-9))
    return _KalmanModeState(x=x_upd, p=p_upd), log_likelihood


class _SwitchingKalmanModeBank:
    def __init__(self) -> None:
        self._states: dict[str, _KalmanModeState] | None = None
        self._posterior = {"stationary": 0.25, "constant_velocity": 0.25, "braking": 0.25, "maneuver": 0.25}

    def _init_states(self, measurement: float) -> None:
        self._states = {
            "stationary": _KalmanModeState(x=[measurement], p=[[0.5]]),
            "constant_velocity": _KalmanModeState(x=[measurement, 0.0], p=[[0.5, 0.0], [0.0, 1.0]]),
            "braking": _KalmanModeState(x=[measurement, 1.2], p=[[0.5, 0.0], [0.0, 1.0]]),
            "maneuver": _KalmanModeState(x=[measurement, 0.0, 0.0], p=[[0.5, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        }

    def update(self, *, dt: float, measurement: float) -> dict[str, float]:
        if self._states is None:
            self._init_states(measurement)
            return dict(self._posterior)
        assert self._states is not None
        log_scores: dict[str, float] = {}
        updated_states: dict[str, _KalmanModeState] = {}
        updated_states["stationary"], ll = _kalman_update_scalar(self._states["stationary"], f=[[1.0]], q=[[0.02]], h=[1.0], measurement=measurement, r=0.06)
        log_scores["stationary"] = log(max(self._posterior["stationary"], 1e-12)) + ll
        updated_states["constant_velocity"], ll = _kalman_update_scalar(self._states["constant_velocity"], f=[[1.0, dt], [0.0, 1.0]], q=[[0.04, 0.0], [0.0, 0.05]], h=[1.0, 0.0], measurement=measurement, r=0.06)
        log_scores["constant_velocity"] = log(max(self._posterior["constant_velocity"], 1e-12)) + ll
        updated_states["braking"], ll = _kalman_update_scalar(self._states["braking"], f=[[1.0, dt], [0.0, 1.0]], q=[[0.05, 0.0], [0.0, 0.06]], h=[1.0, 0.0], measurement=measurement, r=0.06, control=[0.5 * -0.85 * dt * dt, -0.85 * dt])
        log_scores["braking"] = log(max(self._posterior["braking"], 1e-12)) + ll
        updated_states["maneuver"], ll = _kalman_update_scalar(self._states["maneuver"], f=[[1.0, dt, 0.5 * dt * dt], [0.0, 1.0, dt], [0.0, 0.0, 1.0]], q=[[0.08, 0.0, 0.0], [0.0, 0.08, 0.0], [0.0, 0.0, 0.20]], h=[1.0, 0.0, 0.0], measurement=measurement, r=0.07)
        log_scores["maneuver"] = log(max(self._posterior["maneuver"], 1e-12)) + ll
        self._posterior = _normalize_log_scores(log_scores)
        self._states = updated_states
        return dict(self._posterior)


def _run_kalman_mode_bank(scenario: SwitchingScenario) -> TransitionRun:
    bank = _SwitchingKalmanModeBank()
    steps: list[TransitionPosteriorStep] = []
    previous_time = scenario.times[0]
    posterior = {"stationary": 0.25, "constant_velocity": 0.25, "braking": 0.25, "maneuver": 0.25}
    for index, (time, measurement, true_mode) in enumerate(zip(scenario.times, scenario.measurements, scenario.true_mode_by_step)):
        speed, accel = _speed_and_accel(scenario.measurements, scenario.times, index)
        dt = 0.0 if index == 0 else time - previous_time
        previous_time = time
        prior = dict(posterior)
        posterior = bank.update(dt=dt, measurement=measurement)
        predicted = max(posterior, key=posterior.get)
        steps.append(
            TransitionPosteriorStep(
                step=index,
                time=time,
                measurement=measurement,
                estimated_speed=speed,
                estimated_accel=accel,
                prior_weights=prior,
                posterior_weights=posterior,
                predicted_mode=predicted,
                true_mode=true_mode,
                confidence=posterior[predicted],
            )
        )
    switch_index = next((idx for idx, step in enumerate(steps) if step.true_mode != steps[0].true_mode), len(steps))
    post_switch_steps = steps[switch_index:] if switch_index < len(steps) else steps[-1:]
    return TransitionRun(
        trajectory_id=scenario.trajectory_id,
        scenario_name=scenario.scenario_name,
        mode="kalman_mode_bank",
        steps=tuple(steps),
        final_weights=posterior,
        final_predicted_mode=max(posterior, key=posterior.get),
        accuracy=_mean([1.0 if step.predicted_mode == step.true_mode else 0.0 for step in steps]),
        post_switch_accuracy=_mean([1.0 if step.predicted_mode == step.true_mode else 0.0 for step in post_switch_steps]),
    )


def run_transition_benchmark(
    *,
    seed: int = 7,
    replicas: int = 8,
    specs: tuple[SwitchingModeSpec, ...] | None = None,
) -> TransitionBenchmarkResult:
    selected_specs = specs or default_switching_mode_specs()
    scenarios = generate_transition_switching_scenarios(seed=seed, replicas=replicas)
    transition_matrix = default_transition_matrix()
    static_runs = tuple(_run_mode_accumulator(scenario, selected_specs, mode="static") for scenario in scenarios)
    transition_runs = tuple(
        _run_mode_accumulator(scenario, selected_specs, mode="transition_matrix", transition_matrix=transition_matrix)
        for scenario in scenarios
    )
    kalman_runs = tuple(_run_kalman_mode_bank(scenario) for scenario in scenarios)
    improved = sum(1 for static_run, transition_run in zip(static_runs, transition_runs) if transition_run.post_switch_accuracy > static_run.post_switch_accuracy)
    summary = TransitionBenchmarkSummary(
        num_scenarios=len(scenarios),
        static_accuracy=_mean([run.accuracy for run in static_runs]),
        transition_accuracy=_mean([run.accuracy for run in transition_runs]),
        kalman_accuracy=_mean([run.accuracy for run in kalman_runs]),
        static_post_switch_accuracy=_mean([run.post_switch_accuracy for run in static_runs]),
        transition_post_switch_accuracy=_mean([run.post_switch_accuracy for run in transition_runs]),
        kalman_post_switch_accuracy=_mean([run.post_switch_accuracy for run in kalman_runs]),
        improved_scenarios=improved,
    )
    return TransitionBenchmarkResult(scenarios=scenarios, static_runs=static_runs, transition_runs=transition_runs, kalman_runs=kalman_runs, summary=summary)
