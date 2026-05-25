from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import io
from math import exp, log, pi
import os
from pathlib import Path

from .trajectory_generator import generate_switching_scenarios


def _prepare_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp/kinematic-classifier-sandbox-mpl")))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _gaussian_logpdf(value: float, mean: float, sigma: float) -> float:
    variance = max(sigma * sigma, 1e-9)
    return -0.5 * (log(2.0 * pi * variance) + ((value - mean) ** 2) / variance)


def _normalize_log_scores(log_scores: dict[str, float]) -> dict[str, float]:
    pivot = max(log_scores.values())
    normalizer = pivot + log(sum(exp(value - pivot) for value in log_scores.values()))
    return {name: exp(value - normalizer) for name, value in log_scores.items()}


@dataclass(frozen=True, slots=True)
class SwitchingModeSpec:
    name: str
    mean_speed: float
    sigma_speed: float
    mean_accel: float
    sigma_accel: float
    mean_abs_accel: float
    sigma_abs_accel: float
    prior_weight: float


@dataclass(frozen=True, slots=True)
class SwitchingScenario:
    trajectory_id: str
    scenario_name: str
    seed: int
    times: tuple[float, ...]
    measurements: tuple[float, ...]
    true_mode_by_step: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TransitionPosteriorStep:
    step: int
    time: float
    measurement: float
    estimated_speed: float
    estimated_accel: float
    prior_weights: dict[str, float]
    posterior_weights: dict[str, float]
    predicted_mode: str
    true_mode: str
    confidence: float


@dataclass(frozen=True, slots=True)
class TransitionRun:
    trajectory_id: str
    scenario_name: str
    mode: str
    steps: tuple[TransitionPosteriorStep, ...]
    final_weights: dict[str, float]
    final_predicted_mode: str
    accuracy: float
    post_switch_accuracy: float


@dataclass(frozen=True, slots=True)
class TransitionBenchmarkSummary:
    num_scenarios: int
    static_accuracy: float
    transition_accuracy: float
    kalman_accuracy: float
    static_post_switch_accuracy: float
    transition_post_switch_accuracy: float
    kalman_post_switch_accuracy: float
    improved_scenarios: int


@dataclass(frozen=True, slots=True)
class TransitionBenchmarkResult:
    scenarios: tuple[SwitchingScenario, ...]
    static_runs: tuple[TransitionRun, ...]
    transition_runs: tuple[TransitionRun, ...]
    kalman_runs: tuple[TransitionRun, ...]
    summary: TransitionBenchmarkSummary


@dataclass(frozen=True, slots=True)
class TransitionBenchmarkArtifacts:
    run_dir: Path
    report_path: Path
    numeric_walkthrough_path: Path
    posterior_history_path: Path
    scenario_summary_path: Path
    config_path: Path
    dataset_manifest_path: Path
    plot_png_path: Path


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
            _gaussian_logpdf(speed, spec.mean_speed, spec.sigma_speed)
            + _gaussian_logpdf(accel, spec.mean_accel, spec.sigma_accel)
            + _gaussian_logpdf(abs_accel, spec.mean_abs_accel, spec.sigma_abs_accel)
        )
    return scores


def _emission_term_breakdown(
    spec: SwitchingModeSpec,
    *,
    speed: float,
    accel: float,
) -> dict[str, float]:
    abs_accel = abs(accel)
    speed_term = _gaussian_logpdf(speed, spec.mean_speed, spec.sigma_speed)
    accel_term = _gaussian_logpdf(accel, spec.mean_accel, spec.sigma_accel)
    abs_accel_term = _gaussian_logpdf(abs_accel, spec.mean_abs_accel, spec.sigma_abs_accel)
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


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    rows = len(a)
    cols = len(b[0])
    inner = len(b)
    return [[sum(a[r][k] * b[k][c] for k in range(inner)) for c in range(cols)] for r in range(rows)]


def _matvec(a: list[list[float]], x: list[float]) -> list[float]:
    return [sum(row[i] * x[i] for i in range(len(x))) for row in a]


def _transpose(a: list[list[float]]) -> list[list[float]]:
    return [list(row) for row in zip(*a)]


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
        self._posterior = {
            "stationary": 0.25,
            "constant_velocity": 0.25,
            "braking": 0.25,
            "maneuver": 0.25,
        }

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
        updated_states["stationary"], ll = _kalman_update_scalar(
            self._states["stationary"],
            f=[[1.0]],
            q=[[0.02]],
            h=[1.0],
            measurement=measurement,
            r=0.06,
        )
        log_scores["stationary"] = log(max(self._posterior["stationary"], 1e-12)) + ll
        updated_states["constant_velocity"], ll = _kalman_update_scalar(
            self._states["constant_velocity"],
            f=[[1.0, dt], [0.0, 1.0]],
            q=[[0.04, 0.0], [0.0, 0.05]],
            h=[1.0, 0.0],
            measurement=measurement,
            r=0.06,
        )
        log_scores["constant_velocity"] = log(max(self._posterior["constant_velocity"], 1e-12)) + ll
        updated_states["braking"], ll = _kalman_update_scalar(
            self._states["braking"],
            f=[[1.0, dt], [0.0, 1.0]],
            q=[[0.05, 0.0], [0.0, 0.06]],
            h=[1.0, 0.0],
            measurement=measurement,
            r=0.06,
            control=[0.5 * -0.85 * dt * dt, -0.85 * dt],
        )
        log_scores["braking"] = log(max(self._posterior["braking"], 1e-12)) + ll
        updated_states["maneuver"], ll = _kalman_update_scalar(
            self._states["maneuver"],
            f=[[1.0, dt, 0.5 * dt * dt], [0.0, 1.0, dt], [0.0, 0.0, 1.0]],
            q=[[0.08, 0.0, 0.0], [0.0, 0.08, 0.0], [0.0, 0.0, 0.20]],
            h=[1.0, 0.0, 0.0],
            measurement=measurement,
            r=0.07,
        )
        log_scores["maneuver"] = log(max(self._posterior["maneuver"], 1e-12)) + ll
        self._posterior = _normalize_log_scores(log_scores)
        self._states = updated_states
        return dict(self._posterior)


def _run_kalman_mode_bank(scenario: SwitchingScenario) -> TransitionRun:
    bank = _SwitchingKalmanModeBank()
    steps: list[TransitionPosteriorStep] = []
    previous_time = scenario.times[0]
    posterior = {
        "stationary": 0.25,
        "constant_velocity": 0.25,
        "braking": 0.25,
        "maneuver": 0.25,
    }
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
    improved = sum(
        1
        for static_run, transition_run in zip(static_runs, transition_runs)
        if transition_run.post_switch_accuracy > static_run.post_switch_accuracy
    )
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
    return TransitionBenchmarkResult(
        scenarios=scenarios,
        static_runs=static_runs,
        transition_runs=transition_runs,
        kalman_runs=kalman_runs,
        summary=summary,
    )


def render_transition_benchmark_report(result: TransitionBenchmarkResult) -> str:
    scenario_lines = []
    for static_run, transition_run, kalman_run in zip(result.static_runs[:9], result.transition_runs[:9], result.kalman_runs[:9]):
        scenario_lines.append(
            f"| {static_run.scenario_name} | {static_run.accuracy:.3f} | {transition_run.accuracy:.3f} | {kalman_run.accuracy:.3f} | {static_run.post_switch_accuracy:.3f} | {transition_run.post_switch_accuracy:.3f} | {kalman_run.post_switch_accuracy:.3f} |"
        )
    return "\n".join(
        [
            "# Transition-Matrix Accumulator",
            "",
            "Milestone 16 comparison between a static mode accumulator, a transition-matrix accumulator, and a Kalman mode bank on explicit switching scenarios.",
            "",
            "## Summary",
            "",
            f"- Scenarios: {result.summary.num_scenarios}",
            f"- Static accuracy: {result.summary.static_accuracy:.3f}",
            f"- Transition accuracy: {result.summary.transition_accuracy:.3f}",
            f"- Kalman accuracy: {result.summary.kalman_accuracy:.3f}",
            f"- Static post-switch accuracy: {result.summary.static_post_switch_accuracy:.3f}",
            f"- Transition post-switch accuracy: {result.summary.transition_post_switch_accuracy:.3f}",
            f"- Kalman post-switch accuracy: {result.summary.kalman_post_switch_accuracy:.3f}",
            f"- Scenarios improved post-switch: {result.summary.improved_scenarios}",
            "",
            "## Scenario Comparison",
            "",
            "| scenario_name | static_accuracy | transition_accuracy | kalman_accuracy | static_post_switch | transition_post_switch | kalman_post_switch |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            *scenario_lines,
            "",
            "## Notes",
            "",
            "- Both methods use the same local emission model over derived speed and acceleration.",
            "- The only difference is whether prior mass can move through an explicit transition matrix.",
            "- The Kalman mode bank adds innovation-likelihood behavior from per-mode kinematic filters without transition mixing.",
            "- This isolates the value of transition dynamics before moving to IMM.",
        ]
    )


def _select_transition_walkthrough(
    result: TransitionBenchmarkResult,
) -> tuple[SwitchingScenario, TransitionRun, TransitionRun]:
    preferred_names = (
        "constant_velocity_then_braking",
        "constant_velocity_then_maneuver",
        "stationary_then_moving",
    )
    for preferred_name in preferred_names:
        for scenario, static_run, transition_run in zip(result.scenarios, result.static_runs, result.transition_runs):
            if scenario.scenario_name == preferred_name:
                return scenario, static_run, transition_run
    return result.scenarios[0], result.static_runs[0], result.transition_runs[0]


def render_transition_numeric_walkthrough_markdown(
    result: TransitionBenchmarkResult,
    *,
    specs: tuple[SwitchingModeSpec, ...] | None = None,
    transition_matrix: dict[str, dict[str, float]] | None = None,
) -> str:
    selected_specs = specs or default_switching_mode_specs()
    transition = transition_matrix or default_transition_matrix()
    scenario, static_run, transition_run = _select_transition_walkthrough(result)
    switch_index = next(
        (index for index, step in enumerate(transition_run.steps) if step.true_mode != transition_run.steps[0].true_mode),
        len(transition_run.steps) - 1,
    )
    start_index = max(0, switch_index - 1)
    end_index = min(len(transition_run.steps), switch_index + 2)
    selected_steps = transition_run.steps[start_index:end_index]

    switch_step = transition_run.steps[switch_index]
    previous_posterior = (
        transition_run.steps[switch_index - 1].posterior_weights
        if switch_index > 0
        else transition_run.steps[switch_index].prior_weights
    )
    switched_mode = switch_step.true_mode
    contribution_rows = []
    total_prior = 0.0
    for source_mode, source_weight in previous_posterior.items():
        contribution = source_weight * transition[source_mode][switched_mode]
        total_prior += contribution
        contribution_rows.append(
            f"| `{source_mode}` | {source_weight:.3f} | {transition[source_mode][switched_mode]:.3f} | {contribution:.3f} |"
        )

    lines = [
        "# Transition-Matrix Numeric Walkthrough",
        "",
        "This worked example uses a real benchmark run from `transition_matrix_accumulator.py` and shows the full transition-aware recursion on a short switching trajectory.",
        "",
        "## Selected trajectory",
        "",
        f"- Scenario: `{scenario.scenario_name}`",
        f"- Trajectory: `{scenario.trajectory_id}`",
        f"- True mode sequence starts as `{transition_run.steps[0].true_mode}` and switches to `{switched_mode}` at step `{switch_index}`",
        f"- Static post-switch accuracy: `{static_run.post_switch_accuracy:.3f}`",
        f"- Transition-matrix post-switch accuracy: `{transition_run.post_switch_accuracy:.3f}`",
        "",
        "## Transition propagation at the first switched step",
        "",
        f"For the switched target mode `{switched_mode}`, the propagated prior is",
        "",
        "```tex",
        rf"\bar{{p}}_t({switched_mode}) = \sum_s p_{{t-1}}(s)\,T_{{s,{switched_mode}}}",
        "```",
        "",
        "| source mode | previous posterior | transition probability | contribution |",
        "| --- | ---: | ---: | ---: |",
        *contribution_rows,
        f"| **total** |  |  | **{total_prior:.3f}** |",
        "",
    ]

    for step in selected_steps:
        lines.extend(
            [
                f"## Step `{step.step}` at time `{step.time:.3f}`",
                "",
                f"- Measurement: `{step.measurement:.3f}`",
                f"- Estimated speed: `{step.estimated_speed:.3f}`",
                f"- Estimated acceleration: `{step.estimated_accel:.3f}`",
                f"- True mode: `{step.true_mode}`",
                f"- Predicted mode: `{step.predicted_mode}` with confidence `{step.confidence:.3f}`",
                "",
                "| mode | propagated prior | log prior | speed term | accel term | abs-accel term | emission total | log numerator | posterior |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for spec in selected_specs:
            terms = _emission_term_breakdown(spec, speed=step.estimated_speed, accel=step.estimated_accel)
            prior = step.prior_weights[spec.name]
            log_prior = log(max(prior, 1e-12))
            log_numerator = log_prior + terms["emission_total"]
            posterior = step.posterior_weights[spec.name]
            lines.append(
                f"| `{spec.name}` | {prior:.3f} | {log_prior:.3f} | {terms['speed_term']:.3f} | {terms['accel_term']:.3f} | {terms['abs_accel_term']:.3f} | {terms['emission_total']:.3f} | {log_numerator:.3f} | {posterior:.3f} |"
            )
        lines.extend(
            [
                "",
                "The transition-aware update at this step is",
                "",
                "```tex",
                r"\log \tilde{p}_t(s) = \log \bar{p}_t(s) + \log E_t(s),",
                r"\qquad",
                r"p_t(s) = \frac{\exp(\log \tilde{p}_t(s))}{\sum_j \exp(\log \tilde{p}_t(j))}.",
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation",
            "",
            "- The static accumulator and transition-matrix accumulator use the same emission model; the difference is only the prior propagation step.",
            f"- On this trajectory, the switched target mode `{switched_mode}` gets nontrivial prior mass before the emission term is applied because the transition matrix allows probability to move from the pre-switch mode family.",
            "- That is the concrete mechanism by which the transition-aware accumulator improves post-switch behavior before the repo needs a full IMM implementation.",
        ]
    )
    return "\n".join(lines)


def _build_figure(result: TransitionBenchmarkResult):
    plt = _prepare_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.0))
    selected_names = ("stationary_then_moving", "constant_velocity_then_braking", "constant_velocity_then_maneuver")
    colors = {
        "stationary": "#6b7280",
        "constant_velocity": "#2563eb",
        "braking": "#dc2626",
        "maneuver": "#7c3aed",
    }
    for axis, scenario_name in zip(axes.flat[:3], selected_names):
        transition_run = next(run for run in result.transition_runs if run.scenario_name == scenario_name)
        kalman_run = next(run for run in result.kalman_runs if run.scenario_name == scenario_name)
        for mode in colors:
            axis.plot(
                [step.time for step in transition_run.steps],
                [step.posterior_weights[mode] for step in transition_run.steps],
                color=colors[mode],
                linewidth=2.2,
                label=f"{mode} transition",
            )
            axis.plot(
                [step.time for step in kalman_run.steps],
                [step.posterior_weights[mode] for step in kalman_run.steps],
                color=colors[mode],
                linewidth=1.2,
                alpha=0.35,
                linestyle="--",
            )
        axis.set_title(scenario_name, loc="left", fontsize=12, fontweight="bold")
        axis.set_ylim(0.0, 1.0)
        axis.grid(True, alpha=0.25)
        axis.set_xlabel("time")
        axis.set_ylabel("posterior")
    summary_ax = axes.flat[3]
    labels = ["static", "transition", "kalman"]
    values = [
        result.summary.static_post_switch_accuracy,
        result.summary.transition_post_switch_accuracy,
        result.summary.kalman_post_switch_accuracy,
    ]
    summary_ax.bar(labels, values, color=["#9ca3af", "#059669", "#2563eb"], width=0.55)
    summary_ax.set_ylim(0.0, 1.0)
    summary_ax.set_title("Post-switch accuracy", loc="left", fontsize=12, fontweight="bold")
    summary_ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def write_transition_benchmark_artifacts(
    output_dir: str | Path,
    *,
    result: TransitionBenchmarkResult | None = None,
) -> TransitionBenchmarkArtifacts:
    analysis = result or run_transition_benchmark()
    run_dir = Path(output_dir) / "transition_matrix_accumulator_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "transition_matrix_accumulator_report.md"
    numeric_walkthrough_path = run_dir / "transition_matrix_numeric_walkthrough.md"
    posterior_history_path = run_dir / "transition_matrix_posterior_history.csv"
    scenario_summary_path = run_dir / "transition_matrix_scenario_summary.csv"
    config_path = run_dir / "transition_matrix_config.yaml"
    dataset_manifest_path = run_dir / "transition_matrix_dataset_manifest.json"
    plot_png_path = run_dir / "transition_matrix_diagnostics.png"

    report_path.write_text(render_transition_benchmark_report(analysis), encoding="utf-8")
    numeric_walkthrough_path.write_text(
        render_transition_numeric_walkthrough_markdown(analysis),
        encoding="utf-8",
    )
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  name: transition_matrix_accumulator",
                "dataset:",
                "  source: switching_scenarios_v1",
                "classifier:",
                "  baseline: static_mode_accumulator",
                "  candidate: transition_matrix_accumulator",
                "",
            ]
        ),
        encoding="utf-8",
    )
    dataset_manifest_path.write_text(
        __import__("json").dumps(
            {
                "scenario_count": analysis.summary.num_scenarios,
                "scenario_names": sorted({scenario.scenario_name for scenario in analysis.scenarios}),
                "replicas": len(analysis.scenarios) // 3 if analysis.scenarios else 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    posterior_rows: list[dict[str, object]] = []
    scenario_rows: list[dict[str, object]] = []
    for static_run, transition_run, kalman_run in zip(analysis.static_runs, analysis.transition_runs, analysis.kalman_runs):
        for run in (static_run, transition_run, kalman_run):
            for step in run.steps:
                posterior_rows.append(
                    {
                        "trajectory_id": run.trajectory_id,
                        "scenario_name": run.scenario_name,
                        "mode": run.mode,
                        "step": step.step,
                        "time": step.time,
                        "measurement": step.measurement,
                        "estimated_speed": step.estimated_speed,
                        "estimated_accel": step.estimated_accel,
                        "true_mode": step.true_mode,
                        "predicted_mode": step.predicted_mode,
                        "confidence": step.confidence,
                        **{f"posterior_{name}": value for name, value in step.posterior_weights.items()},
                    }
                )
        scenario_rows.append(
            {
                "scenario_name": static_run.scenario_name,
                "trajectory_id": static_run.trajectory_id,
                "static_accuracy": static_run.accuracy,
                "transition_accuracy": transition_run.accuracy,
                "kalman_accuracy": kalman_run.accuracy,
                "static_post_switch_accuracy": static_run.post_switch_accuracy,
                "transition_post_switch_accuracy": transition_run.post_switch_accuracy,
                "kalman_post_switch_accuracy": kalman_run.post_switch_accuracy,
            }
        )

    mode_names = list(analysis.static_runs[0].steps[0].posterior_weights) if analysis.static_runs else []
    _write_csv(
        posterior_history_path,
        posterior_rows,
        [
            "trajectory_id",
            "scenario_name",
            "mode",
            "step",
            "time",
            "measurement",
            "estimated_speed",
            "estimated_accel",
            "true_mode",
            "predicted_mode",
            "confidence",
            *[f"posterior_{name}" for name in mode_names],
        ],
    )
    _write_csv(
        scenario_summary_path,
        scenario_rows,
        [
            "scenario_name",
            "trajectory_id",
            "static_accuracy",
            "transition_accuracy",
            "kalman_accuracy",
            "static_post_switch_accuracy",
            "transition_post_switch_accuracy",
            "kalman_post_switch_accuracy",
        ],
    )
    plt = _prepare_matplotlib()
    fig = _build_figure(analysis)
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        plot_png_path.write_bytes(buffer.getvalue())
    finally:
        plt.close(fig)
    return TransitionBenchmarkArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        numeric_walkthrough_path=numeric_walkthrough_path,
        posterior_history_path=posterior_history_path,
        scenario_summary_path=scenario_summary_path,
        config_path=config_path,
        dataset_manifest_path=dataset_manifest_path,
        plot_png_path=plot_png_path,
    )
