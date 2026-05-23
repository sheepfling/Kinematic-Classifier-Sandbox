from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, pi
import csv
import io
import os
from pathlib import Path
import random

from .kalman_filter_bank import KalmanModelSpec, run_kalman_filter_bank


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


def _gaussian_logpdf(value: float, mean: float, variance: float) -> float:
    safe_variance = max(variance, 1e-9)
    return -0.5 * (log(2.0 * pi * safe_variance) + ((value - mean) ** 2) / safe_variance)


def _logsumexp(values: list[float]) -> float:
    pivot = max(values)
    return pivot + log(sum(exp(value - pivot) for value in values))


def _normalize_log_scores(log_scores: dict[str, float]) -> dict[str, float]:
    log_norm = _logsumexp(list(log_scores.values()))
    return {name: exp(value - log_norm) for name, value in log_scores.items()}


def _normalize_prior(prior: dict[str, float] | None) -> dict[str, float]:
    if prior is None:
        return {class_name: 1.0 / len(CLASS_NAMES) for class_name in CLASS_NAMES}
    total = sum(max(prior.get(class_name, 0.0), 0.0) for class_name in CLASS_NAMES)
    if total <= 1e-12:
        return {class_name: 1.0 / len(CLASS_NAMES) for class_name in CLASS_NAMES}
    return {class_name: max(prior.get(class_name, 0.0), 0.0) / total for class_name in CLASS_NAMES}


@dataclass(frozen=True, slots=True)
class SharedDynamicsTrajectory:
    trajectory_id: str
    true_class: str
    scenario_name: str
    seed: int
    times: tuple[float, ...]
    measurements: tuple[float, ...]
    true_position: tuple[float, ...]
    true_velocity: tuple[float, ...]
    true_acceleration: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class CommonMethodRun:
    method_name: str
    trajectory_id: str
    true_class: str
    scenario_name: str
    final_predicted_class: str
    final_confidence: float
    final_weights: dict[str, float]


@dataclass(frozen=True, slots=True)
class CommonComparisonRow:
    method_name: str
    overall_accuracy: float
    easy_accuracy: float
    irregular_accuracy: float
    noisy_accuracy: float
    prior_flip_fraction: float


@dataclass(frozen=True, slots=True)
class CommonComparisonResult:
    trajectories: tuple[SharedDynamicsTrajectory, ...]
    runs: tuple[CommonMethodRun, ...]
    rows: tuple[CommonComparisonRow, ...]


@dataclass(frozen=True, slots=True)
class CommonComparisonArtifacts:
    run_dir: Path
    report_path: Path
    trajectory_path: Path
    run_summary_path: Path
    method_summary_path: Path
    heatmap_svg_path: Path
    heatmap_png_path: Path
    confusion_svg_path: Path
    confusion_png_path: Path


CLASS_NAMES = ("constant_velocity", "constant_acceleration")


def _shared_times(scenario_name: str) -> tuple[float, ...]:
    if scenario_name == "irregular":
        return (0.0, 0.7, 1.6, 2.8, 4.1, 5.0, 6.6, 7.4, 8.9, 10.0)
    return tuple(float(step) for step in range(10))


def _make_shared_trajectory(
    *,
    trajectory_id: str,
    true_class: str,
    scenario_name: str,
    seed: int,
    measurement_sigma: float,
) -> SharedDynamicsTrajectory:
    rng = random.Random(seed)
    times = _shared_times(scenario_name)
    position0 = 0.0
    velocity0 = 0.8
    acceleration = 0.0 if true_class == "constant_velocity" else 0.28
    true_position = tuple(position0 + velocity0 * time + 0.5 * acceleration * time * time for time in times)
    true_velocity = tuple(velocity0 + acceleration * time for time in times)
    true_acceleration = tuple(acceleration for _ in times)
    measurements = tuple(value + rng.gauss(0.0, measurement_sigma) for value in true_position)
    return SharedDynamicsTrajectory(
        trajectory_id=trajectory_id,
        true_class=true_class,
        scenario_name=scenario_name,
        seed=seed,
        times=times,
        measurements=measurements,
        true_position=true_position,
        true_velocity=true_velocity,
        true_acceleration=true_acceleration,
    )


def generate_shared_dynamics_dataset(*, seed: int = 7, trajectories_per_case: int = 8) -> tuple[SharedDynamicsTrajectory, ...]:
    trajectories: list[SharedDynamicsTrajectory] = []
    scenario_defs = (
        ("easy", 0.10),
        ("irregular", 0.10),
        ("noisy", 0.35),
    )
    for scenario_index, (scenario_name, measurement_sigma) in enumerate(scenario_defs):
        for class_index, class_name in enumerate(CLASS_NAMES):
            for example_index in range(trajectories_per_case):
                trajectories.append(
                    _make_shared_trajectory(
                        trajectory_id=f"{scenario_name}_{class_name}_{example_index}",
                        true_class=class_name,
                        scenario_name=scenario_name,
                        seed=seed + scenario_index * 100 + class_index * 20 + example_index,
                        measurement_sigma=measurement_sigma,
                    )
                )
    return tuple(trajectories)


def _class_expected_position(class_name: str, time: float) -> float:
    velocity0 = 0.8
    acceleration = 0.0 if class_name == "constant_velocity" else 0.28
    return velocity0 * time + 0.5 * acceleration * time * time


def _pointwise_predict(
    trajectory: SharedDynamicsTrajectory,
    *,
    prior: dict[str, float] | None = None,
) -> CommonMethodRun:
    last_measurement = trajectory.measurements[-1]
    last_time = trajectory.times[-1]
    prior_weights = _normalize_prior(prior)
    log_scores = {
        class_name: log(max(prior_weights[class_name], 1e-12))
        + _gaussian_logpdf(last_measurement, _class_expected_position(class_name, last_time), 0.35 ** 2)
        for class_name in CLASS_NAMES
    }
    weights = _normalize_log_scores(log_scores)
    predicted = max(weights, key=weights.get)
    return CommonMethodRun(
        method_name="pointwise",
        trajectory_id=trajectory.trajectory_id,
        true_class=trajectory.true_class,
        scenario_name=trajectory.scenario_name,
        final_predicted_class=predicted,
        final_confidence=weights[predicted],
        final_weights=weights,
    )


def _accumulator_predict(
    trajectory: SharedDynamicsTrajectory,
    *,
    prior: dict[str, float] | None = None,
) -> CommonMethodRun:
    posterior = _normalize_prior(prior)
    for time, measurement in zip(trajectory.times, trajectory.measurements):
        log_scores = {
            class_name: log(max(posterior[class_name], 1e-12))
            + _gaussian_logpdf(measurement, _class_expected_position(class_name, time), 0.25 ** 2)
            for class_name in CLASS_NAMES
        }
        posterior = _normalize_log_scores(log_scores)
    predicted = max(posterior, key=posterior.get)
    return CommonMethodRun(
        method_name="accumulator",
        trajectory_id=trajectory.trajectory_id,
        true_class=trajectory.true_class,
        scenario_name=trajectory.scenario_name,
        final_predicted_class=predicted,
        final_confidence=posterior[predicted],
        final_weights=posterior,
    )


def _window_features(trajectory: SharedDynamicsTrajectory) -> dict[str, float]:
    times = list(trajectory.times)
    values = list(trajectory.measurements)
    n = len(values)
    mean_t = sum(times) / n
    mean_y = sum(values) / n
    denominator = sum((time - mean_t) ** 2 for time in times)
    slope = 0.0 if denominator <= 1e-9 else sum((time - mean_t) * (value - mean_y) for time, value in zip(times, values)) / denominator
    quadratic_proxy = 0.0
    if len(times) >= 3:
        midpoint = len(times) // 2
        quadratic_proxy = values[-1] - 2.0 * values[midpoint] + values[0]
    return {"slope": slope, "quadratic_proxy": quadratic_proxy}


def _windowed_predict(
    trajectory: SharedDynamicsTrajectory,
    *,
    robust: bool,
    prior: dict[str, float] | None = None,
) -> CommonMethodRun:
    features = _window_features(trajectory)
    prior_weights = _normalize_prior(prior)
    if robust:
        specs = {
            "constant_velocity": {"slope": 0.8, "quadratic_proxy": 0.0, "sigma_slope": 0.20, "sigma_quad": 0.80},
            "constant_acceleration": {"slope": 2.2, "quadratic_proxy": 6.5, "sigma_slope": 0.35, "sigma_quad": 1.50},
        }
        method_name = "windowed_robust"
    else:
        specs = {
            "constant_velocity": {"slope": 0.8, "quadratic_proxy": 0.0, "sigma_slope": 0.22, "sigma_quad": 1.20},
            "constant_acceleration": {"slope": 2.2, "quadratic_proxy": 6.5, "sigma_slope": 0.40, "sigma_quad": 2.40},
        }
        method_name = "windowed_raw"
    log_scores = {}
    for class_name in CLASS_NAMES:
        spec = specs[class_name]
        log_scores[class_name] = (
            log(max(prior_weights[class_name], 1e-12))
            +
            _gaussian_logpdf(features["slope"], spec["slope"], spec["sigma_slope"] ** 2)
            + _gaussian_logpdf(features["quadratic_proxy"], spec["quadratic_proxy"], spec["sigma_quad"] ** 2)
        )
    weights = _normalize_log_scores(log_scores)
    predicted = max(weights, key=weights.get)
    return CommonMethodRun(
        method_name=method_name,
        trajectory_id=trajectory.trajectory_id,
        true_class=trajectory.true_class,
        scenario_name=trajectory.scenario_name,
        final_predicted_class=predicted,
        final_confidence=weights[predicted],
        final_weights=weights,
    )


def _kalman_predict(
    trajectory: SharedDynamicsTrajectory,
    *,
    prior: dict[str, float] | None = None,
) -> CommonMethodRun:
    prior_weights = _normalize_prior(prior)
    model_specs = (
        KalmanModelSpec(
            name="constant_velocity",
            state_dim=2,
            process_sigma=0.14,
            measurement_sigma=0.20,
            initial_covariance_scale=5.0,
            prior_weight=prior_weights["constant_velocity"],
        ),
        KalmanModelSpec(
            name="constant_acceleration",
            state_dim=3,
            process_sigma=0.24,
            measurement_sigma=0.20,
            initial_covariance_scale=6.0,
            prior_weight=prior_weights["constant_acceleration"],
        ),
    )
    from .kalman_filter_bank import KalmanTrajectory as SharedKalmanTrajectory

    kalman_trajectory = SharedKalmanTrajectory(
        trajectory_id=trajectory.trajectory_id,
        true_class=trajectory.true_class,
        scenario_name=trajectory.scenario_name,
        seed=trajectory.seed,
        times=trajectory.times,
        measurements=trajectory.measurements,
        true_position=trajectory.true_position,
        true_velocity=trajectory.true_velocity,
        true_acceleration=trajectory.true_acceleration,
    )
    run = run_kalman_filter_bank(kalman_trajectory, model_specs)
    return CommonMethodRun(
        method_name="kalman_bank",
        trajectory_id=trajectory.trajectory_id,
        true_class=trajectory.true_class,
        scenario_name=trajectory.scenario_name,
        final_predicted_class=run.final_predicted_class,
        final_confidence=run.final_confidence,
        final_weights=run.final_weights,
    )


def _predict_with_method(
    method_name: str,
    trajectory: SharedDynamicsTrajectory,
    *,
    prior: dict[str, float] | None = None,
) -> CommonMethodRun:
    if method_name == "pointwise":
        return _pointwise_predict(trajectory, prior=prior)
    if method_name == "windowed_raw":
        return _windowed_predict(trajectory, robust=False, prior=prior)
    if method_name == "windowed_robust":
        return _windowed_predict(trajectory, robust=True, prior=prior)
    if method_name == "accumulator":
        return _accumulator_predict(trajectory, prior=prior)
    if method_name == "kalman_bank":
        return _kalman_predict(trajectory, prior=prior)
    raise ValueError(f"Unsupported method: {method_name}")


def _shared_prior_flip_fraction(method_name: str, trajectories: tuple[SharedDynamicsTrajectory, ...]) -> float:
    base_prior = {"constant_velocity": 0.5, "constant_acceleration": 0.5}
    priors_to_compare = (
        {"constant_velocity": 0.25, "constant_acceleration": 0.75},
        {"constant_velocity": 0.75, "constant_acceleration": 0.25},
    )
    flip_count = 0
    for trajectory in trajectories:
        baseline = _predict_with_method(method_name, trajectory, prior=base_prior).final_predicted_class
        alternatives = [
            _predict_with_method(method_name, trajectory, prior=prior).final_predicted_class
            for prior in priors_to_compare
        ]
        if any(predicted != baseline for predicted in alternatives):
            flip_count += 1
    return flip_count / len(trajectories)


def analyze_common_dataset_comparison(*, seed: int = 7, trajectories_per_case: int = 8) -> CommonComparisonResult:
    trajectories = generate_shared_dynamics_dataset(seed=seed, trajectories_per_case=trajectories_per_case)
    method_names = ("pointwise", "windowed_raw", "windowed_robust", "accumulator", "kalman_bank")
    runs: list[CommonMethodRun] = []
    for trajectory in trajectories:
        for method_name in method_names:
            runs.append(_predict_with_method(method_name, trajectory))

    rows: list[CommonComparisonRow] = []
    for method_name in method_names:
        method_runs = [run for run in runs if run.method_name == method_name]
        overall_accuracy = sum(1.0 if run.final_predicted_class == run.true_class else 0.0 for run in method_runs) / len(method_runs)

        def _scenario_accuracy(scenario_name: str) -> float:
            selected = [run for run in method_runs if run.scenario_name == scenario_name]
            return sum(1.0 if run.final_predicted_class == run.true_class else 0.0 for run in selected) / len(selected)

        rows.append(
            CommonComparisonRow(
                method_name=method_name,
                overall_accuracy=overall_accuracy,
                easy_accuracy=_scenario_accuracy("easy"),
                irregular_accuracy=_scenario_accuracy("irregular"),
                noisy_accuracy=_scenario_accuracy("noisy"),
                prior_flip_fraction=_shared_prior_flip_fraction(method_name, trajectories),
            )
        )
    return CommonComparisonResult(trajectories=trajectories, runs=tuple(runs), rows=tuple(rows))


def render_common_dataset_comparison_report(result: CommonComparisonResult) -> str:
    lines = [
        "# Common-Dataset Technique Comparison",
        "",
        "This artifact evaluates the current technique families on the same shared binary dynamics corpus: constant velocity versus constant acceleration with easy, irregular-`dt`, and noisy scenarios.",
        "",
        "## Method Metrics",
        "",
        "| method | overall | easy | irregular | noisy | prior_flip_fraction |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result.rows:
        lines.append(
            f"| {row.method_name} | {row.overall_accuracy:.3f} | {row.easy_accuracy:.3f} | {row.irregular_accuracy:.3f} | {row.noisy_accuracy:.3f} | {row.prior_flip_fraction:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is the first apples-to-apples technique comparison on one shared corpus.",
            "- Pointwise should act as the weak lower bound because it only uses the last measurement.",
            "- Windowed methods should improve once curvature becomes informative.",
            "- The Kalman bank is expected to do best on irregular-`dt` and acceleration-sensitive cases because the motion model matches the data-generating process.",
        ]
    )
    return "\n".join(lines)


def _render_common_metric_heatmap(result: CommonComparisonResult):
    plt = _prepare_matplotlib()
    fields = ("overall_accuracy", "easy_accuracy", "irregular_accuracy", "noisy_accuracy", "prior_flip_fraction")
    matrix = [[getattr(row, field) for field in fields] for row in result.rows]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    image = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_title("Common-Dataset Method Metrics", loc="left", fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(fields)))
    ax.set_xticklabels(["overall", "easy", "irregular", "noisy", "prior_flip"], rotation=25, ha="right")
    ax.set_yticks(range(len(result.rows)))
    ax.set_yticklabels([row.method_name for row in result.rows])
    for row_index, row in enumerate(result.rows):
        for col_index, field in enumerate(fields):
            ax.text(col_index, row_index, f"{getattr(row, field):.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="metric value")
    fig.tight_layout()
    return fig


def _render_common_confusion_bars(result: CommonComparisonResult):
    plt = _prepare_matplotlib()
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    method_names = [row.method_name for row in result.rows]
    overall = [row.overall_accuracy for row in result.rows]
    irregular = [row.irregular_accuracy for row in result.rows]
    noisy = [row.noisy_accuracy for row in result.rows]
    x = list(range(len(method_names)))
    ax.bar([value - 0.25 for value in x], overall, width=0.24, label="overall", color="#2563eb")
    ax.bar(x, irregular, width=0.24, label="irregular", color="#16a34a")
    ax.bar([value + 0.25 for value in x], noisy, width=0.24, label="noisy", color="#d97706")
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title("Shared-Corpus Accuracy by Method", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _figure_to_svg(fig) -> str:
    plt = _prepare_matplotlib()
    buffer = io.StringIO()
    try:
        fig.savefig(buffer, format="svg", bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def _figure_to_png(fig) -> bytes:
    plt = _prepare_matplotlib()
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def write_common_dataset_comparison_artifacts(
    output_dir: str | Path,
    *,
    seed: int = 7,
    result: CommonComparisonResult | None = None,
) -> CommonComparisonArtifacts:
    comparison = result or analyze_common_dataset_comparison(seed=seed)
    output_root = Path(output_dir)
    run_dir = output_root / "common_dataset_comparison_v1"
    run_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / "common_dataset_comparison_report.md"
    trajectory_path = run_dir / "shared_trajectories.csv"
    run_summary_path = run_dir / "method_run_summary.csv"
    method_summary_path = run_dir / "method_summary.csv"
    heatmap_svg_path = run_dir / "common_dataset_metric_heatmap.svg"
    heatmap_png_path = run_dir / "common_dataset_metric_heatmap.png"
    confusion_svg_path = run_dir / "shared_accuracy_bars.svg"
    confusion_png_path = run_dir / "shared_accuracy_bars.png"

    report_path.write_text(render_common_dataset_comparison_report(comparison), encoding="utf-8")
    _write_csv(
        trajectory_path,
        [
            {
                "trajectory_id": trajectory.trajectory_id,
                "true_class": trajectory.true_class,
                "scenario_name": trajectory.scenario_name,
                "seed": trajectory.seed,
                "times": " ".join(f"{value:.3f}" for value in trajectory.times),
                "measurements": " ".join(f"{value:.3f}" for value in trajectory.measurements),
            }
            for trajectory in comparison.trajectories
        ],
        ["trajectory_id", "true_class", "scenario_name", "seed", "times", "measurements"],
    )
    _write_csv(
        run_summary_path,
        [
            {
                "method_name": run.method_name,
                "trajectory_id": run.trajectory_id,
                "true_class": run.true_class,
                "scenario_name": run.scenario_name,
                "final_predicted_class": run.final_predicted_class,
                "final_confidence": run.final_confidence,
                **{f"posterior_{class_name}": run.final_weights[class_name] for class_name in CLASS_NAMES},
            }
            for run in comparison.runs
        ],
        ["method_name", "trajectory_id", "true_class", "scenario_name", "final_predicted_class", "final_confidence", "posterior_constant_velocity", "posterior_constant_acceleration"],
    )
    _write_csv(
        method_summary_path,
        [
            {
                "method_name": row.method_name,
                "overall_accuracy": row.overall_accuracy,
                "easy_accuracy": row.easy_accuracy,
                "irregular_accuracy": row.irregular_accuracy,
                "noisy_accuracy": row.noisy_accuracy,
                "prior_flip_fraction": row.prior_flip_fraction,
            }
            for row in comparison.rows
        ],
        ["method_name", "overall_accuracy", "easy_accuracy", "irregular_accuracy", "noisy_accuracy", "prior_flip_fraction"],
    )
    heatmap_svg_path.write_text(_figure_to_svg(_render_common_metric_heatmap(comparison)), encoding="utf-8")
    heatmap_png_path.write_bytes(_figure_to_png(_render_common_metric_heatmap(comparison)))
    confusion_svg_path.write_text(_figure_to_svg(_render_common_confusion_bars(comparison)), encoding="utf-8")
    confusion_png_path.write_bytes(_figure_to_png(_render_common_confusion_bars(comparison)))
    return CommonComparisonArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        trajectory_path=trajectory_path,
        run_summary_path=run_summary_path,
        method_summary_path=method_summary_path,
        heatmap_svg_path=heatmap_svg_path,
        heatmap_png_path=heatmap_png_path,
        confusion_svg_path=confusion_svg_path,
        confusion_png_path=confusion_png_path,
    )
