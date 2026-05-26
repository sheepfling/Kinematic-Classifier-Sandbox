from __future__ import annotations

import io
import json
import random
from dataclasses import asdict, dataclass
from math import log, sqrt
from pathlib import Path

from ...utils.io import write_csv

from ...markdown_builder import MarkdownDocument
from ...utils.math import (
    _gaussian_logpdf,
    _least_squares_slope,
    _monotonicity_score,
    _normalize_log_scores,
    _prefix_running_max,
    _prefix_running_min,
    _sign_change_count,
    _trimmed_quantile,
)
from ...utils.plotting import plt


@dataclass(frozen=True, slots=True)
class WindowedClassSpec:
    name: str
    prior_weight: float
    feature_means: dict[str, float]
    feature_sigmas: dict[str, float]


@dataclass(frozen=True, slots=True)
class WindowedTrajectory:
    trajectory_id: str
    true_class: str
    scenario_name: str
    seed: int
    times: tuple[float, ...]
    measurements: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class WindowedFeatureRow:
    trajectory_id: str
    time: float
    window_start: float
    window_end: float
    sample_count: int
    duration: float
    running_min: float
    running_max: float
    window_min: float
    window_max: float
    robust_min: float
    robust_max: float
    running_range: float
    window_range: float
    trimmed_range: float
    window_mean: float
    window_std: float
    slope: float
    monotonicity: float
    sign_changes: int


@dataclass(frozen=True, slots=True)
class WindowedPosteriorStep:
    time: float
    feature_row: WindowedFeatureRow
    posterior_weights: dict[str, float]
    log_likelihood_terms: dict[str, float]
    predicted_class: str
    confidence: float


@dataclass(frozen=True, slots=True)
class WindowedClassificationRun:
    trajectory_id: str
    true_class: str
    scenario_name: str
    feature_mode: str
    steps: tuple[WindowedPosteriorStep, ...]
    final_weights: dict[str, float]
    final_predicted_class: str


@dataclass(frozen=True, slots=True)
class WindowedBenchmarkSummary:
    total_trajectories: int
    raw_final_accuracy: float
    robust_final_accuracy: float
    raw_spike_accuracy: float
    robust_spike_accuracy: float
    per_class_accuracy_raw: dict[str, float]
    per_class_accuracy_robust: dict[str, float]
    confusion_raw: dict[str, dict[str, int]]
    confusion_robust: dict[str, dict[str, int]]


@dataclass(frozen=True, slots=True)
class WindowedBenchmarkResult:
    class_specs: tuple[WindowedClassSpec, ...]
    trajectories: tuple[WindowedTrajectory, ...]
    raw_runs: tuple[WindowedClassificationRun, ...]
    robust_runs: tuple[WindowedClassificationRun, ...]
    feature_rows: tuple[WindowedFeatureRow, ...]
    summary: WindowedBenchmarkSummary


@dataclass(frozen=True, slots=True)
class WindowedBenchmarkArtifacts:
    run_dir: Path
    report_path: Path
    feature_matrix_path: Path
    posterior_history_path: Path
    confusion_raw_path: Path
    confusion_robust_path: Path
    config_path: Path
    dataset_manifest_path: Path
    feature_manifest_path: Path
    plot_png_path: Path


class WindowedFeatureClassifier:
    def __init__(
        self,
        class_specs: tuple[WindowedClassSpec, ...],
        *,
        feature_mode: str,
        prior: dict[str, float] | None = None,
    ) -> None:
        self._class_specs = class_specs
        self._feature_mode = feature_mode
        total_prior = sum(spec.prior_weight for spec in class_specs)
        self._prior = prior or {spec.name: spec.prior_weight / total_prior for spec in class_specs}
        self._posterior = dict(self._prior)
        self._history: list[WindowedPosteriorStep] = []

    def reset(self, prior: dict[str, float] | None = None) -> None:
        self._posterior = dict(prior or self._prior)
        self._history.clear()

    def update(self, feature_row: WindowedFeatureRow) -> WindowedPosteriorStep:
        selected_features = _selected_features(self._feature_mode)
        log_scores = {}
        for spec in self._class_specs:
            log_score = log(max(self._posterior[spec.name], 1e-12))
            for feature_name in selected_features:
                log_score += _gaussian_logpdf(
                    getattr(feature_row, feature_name),
                    spec.feature_means[feature_name],
                    spec.feature_sigmas[feature_name] ** 2,
                )
            log_scores[spec.name] = log_score
        posterior = _normalize_log_scores(log_scores)
        predicted_class = max(posterior, key=posterior.get)
        step = WindowedPosteriorStep(
            time=feature_row.time,
            feature_row=feature_row,
            posterior_weights=posterior,
            log_likelihood_terms=log_scores,
            predicted_class=predicted_class,
            confidence=posterior[predicted_class],
        )
        self._posterior = posterior
        self._history.append(step)
        return step

    def posterior(self) -> dict[str, float]:
        return dict(self._posterior)

    def predict(self) -> str:
        return max(self._posterior, key=self._posterior.get)

    def history(self) -> tuple[WindowedPosteriorStep, ...]:
        return tuple(self._history)


def _selected_features(feature_mode: str) -> tuple[str, ...]:
    if feature_mode == "raw":
        return ("running_min", "running_max", "running_range", "slope")
    if feature_mode == "robust":
        return ("robust_min", "robust_max", "trimmed_range", "slope")
    raise ValueError(f"Unknown feature mode: {feature_mode}")


def default_windowed_class_specs() -> tuple[WindowedClassSpec, ...]:
    return (
        WindowedClassSpec(
            "low",
            prior_weight=0.5,
                feature_means={
                "running_min": -0.9,
                "running_max": -0.1,
                "running_range": 0.35,
                "robust_min": -0.85,
                "robust_max": -0.15,
                "trimmed_range": 0.30,
                "slope": 0.0,
            },
            feature_sigmas={
                "running_min": 0.18,
                "running_max": 0.22,
                "running_range": 0.16,
                "robust_min": 0.16,
                "robust_max": 0.16,
                "trimmed_range": 0.12,
                "slope": 0.18,
            },
        ),
        WindowedClassSpec(
            "high",
            prior_weight=0.5,
            feature_means={
                "running_min": 0.1,
                "running_max": 0.9,
                "running_range": 0.35,
                "robust_min": 0.15,
                "robust_max": 0.85,
                "trimmed_range": 0.30,
                "slope": 0.0,
            },
            feature_sigmas={
                "running_min": 0.18,
                "running_max": 0.22,
                "running_range": 0.16,
                "robust_min": 0.16,
                "robust_max": 0.16,
                "trimmed_range": 0.12,
                "slope": 0.18,
            },
        ),
    )


def generate_windowed_trajectories(
    *,
    seed: int = 7,
    steps: int = 18,
    dt: float = 1.0,
    obs_sigma: float = 0.18,
) -> tuple[WindowedTrajectory, ...]:
    rng = random.Random(seed)
    trajectories: list[WindowedTrajectory] = []
    scenario_defs = (
        ("low_clean", "low", -0.35, 0.03, None),
        ("high_clean", "high", 0.35, 0.03, None),
        ("low_spike", "low", -0.35, 0.03, 1.40),
        ("high_dip", "high", 0.35, 0.03, -1.40),
        ("low_long", "low", -0.35, 0.03, None),
        ("high_long", "high", 0.35, 0.03, None),
    )
    lengths = {
        "low_long": steps + 10,
        "high_long": steps + 10,
    }
    for index, (scenario_name, class_name, center, slope, spike) in enumerate(scenario_defs):
        scenario_rng = random.Random(rng.randrange(1 << 30) + index)
        count = lengths.get(scenario_name, steps)
        times = tuple(float(step) * dt for step in range(count))
        measurements = []
        for step in range(count):
            value = center + slope * (step / max(count - 1, 1) - 0.5) * 2.0
            value += scenario_rng.gauss(0.0, obs_sigma)
            if spike is not None and step == count // 2:
                value += spike
            measurements.append(value)
        trajectories.append(
            WindowedTrajectory(
                trajectory_id=f"{scenario_name}_{index}",
                true_class=class_name,
                scenario_name=scenario_name,
                seed=seed + index,
                times=times,
                measurements=tuple(measurements),
            )
        )
    return tuple(trajectories)


def extract_windowed_feature_rows(
    trajectory: WindowedTrajectory,
    *,
    window_size: int = 5,
    trim_fraction: float = 0.2,
) -> tuple[WindowedFeatureRow, ...]:
    rows: list[WindowedFeatureRow] = []
    running_min = _prefix_running_min(list(trajectory.measurements))
    running_max = _prefix_running_max(list(trajectory.measurements))
    for index, time in enumerate(trajectory.times):
        start = max(0, index - window_size + 1)
        window_times = list(trajectory.times[start : index + 1])
        window_values = list(trajectory.measurements[start : index + 1])
        duration = window_times[-1] - window_times[0] if len(window_times) > 1 else 0.0
        window_min = min(window_values)
        window_max = max(window_values)
        robust_min = _trimmed_quantile(window_values, 0.10, trim_fraction)
        robust_max = _trimmed_quantile(window_values, 0.90, trim_fraction)
        mean = sum(window_values) / len(window_values)
        variance = sum((value - mean) ** 2 for value in window_values) / max(len(window_values), 1)
        slope = _least_squares_slope(window_times, window_values)
        rows.append(
            WindowedFeatureRow(
                trajectory_id=trajectory.trajectory_id,
                time=time,
                window_start=window_times[0],
                window_end=window_times[-1],
                sample_count=len(window_values),
                duration=duration,
                running_min=running_min[index],
                running_max=running_max[index],
                window_min=window_min,
                window_max=window_max,
                robust_min=robust_min,
                robust_max=robust_max,
                running_range=running_max[index] - running_min[index],
                window_range=window_max - window_min,
                trimmed_range=robust_max - robust_min,
                window_mean=mean,
                window_std=sqrt(max(variance, 0.0)),
                slope=slope,
                monotonicity=_monotonicity_score(window_values),
                sign_changes=_sign_change_count(window_values),
            )
        )
    return tuple(rows)


def _run_classifier(
    trajectory: WindowedTrajectory,
    class_specs: tuple[WindowedClassSpec, ...],
    *,
    feature_mode: str,
    window_size: int,
    trim_fraction: float,
) -> WindowedClassificationRun:
    classifier = WindowedFeatureClassifier(class_specs, feature_mode=feature_mode)
    classifier.reset()
    feature_rows = extract_windowed_feature_rows(trajectory, window_size=window_size, trim_fraction=trim_fraction)
    for row in feature_rows:
        classifier.update(row)
    return WindowedClassificationRun(
        trajectory_id=trajectory.trajectory_id,
        true_class=trajectory.true_class,
        scenario_name=trajectory.scenario_name,
        feature_mode=feature_mode,
        steps=classifier.history(),
        final_weights=classifier.posterior(),
        final_predicted_class=classifier.predict(),
    )


def run_windowed_benchmark(
    *,
    seed: int = 7,
    steps: int = 18,
    window_size: int = 5,
    trim_fraction: float = 0.2,
    class_specs: tuple[WindowedClassSpec, ...] | None = None,
) -> WindowedBenchmarkResult:
    specs = class_specs or default_windowed_class_specs()
    trajectories = generate_windowed_trajectories(seed=seed, steps=steps)
    raw_runs = tuple(_run_classifier(traj, specs, feature_mode="raw", window_size=window_size, trim_fraction=trim_fraction) for traj in trajectories)
    robust_runs = tuple(_run_classifier(traj, specs, feature_mode="robust", window_size=window_size, trim_fraction=trim_fraction) for traj in trajectories)
    feature_rows = tuple(
        row
        for trajectory in trajectories
        for row in extract_windowed_feature_rows(trajectory, window_size=window_size, trim_fraction=trim_fraction)
    )
    summary = summarize_windowed_runs(raw_runs, robust_runs)
    return WindowedBenchmarkResult(
        class_specs=specs,
        trajectories=trajectories,
        raw_runs=raw_runs,
        robust_runs=robust_runs,
        feature_rows=feature_rows,
        summary=summary,
    )


def summarize_windowed_runs(
    raw_runs: tuple[WindowedClassificationRun, ...],
    robust_runs: tuple[WindowedClassificationRun, ...],
) -> WindowedBenchmarkSummary:
    class_names = sorted({run.true_class for run in raw_runs} | {run.true_class for run in robust_runs})
    confusion_raw = {name: {predicted: 0 for predicted in class_names} for name in class_names}
    confusion_robust = {name: {predicted: 0 for predicted in class_names} for name in class_names}
    per_class_correct_raw = {name: 0 for name in class_names}
    per_class_total = {name: 0 for name in class_names}
    per_class_correct_robust = {name: 0 for name in class_names}
    raw_spike_correct = 0
    robust_spike_correct = 0
    spike_cases = 0
    for raw_run, robust_run in zip(raw_runs, robust_runs):
        per_class_total[raw_run.true_class] += 1
        confusion_raw[raw_run.true_class][raw_run.final_predicted_class] += 1
        confusion_robust[robust_run.true_class][robust_run.final_predicted_class] += 1
        if raw_run.final_predicted_class == raw_run.true_class:
            per_class_correct_raw[raw_run.true_class] += 1
        if robust_run.final_predicted_class == robust_run.true_class:
            per_class_correct_robust[robust_run.true_class] += 1
        if "spike" in raw_run.scenario_name or "dip" in raw_run.scenario_name:
            spike_cases += 1
            if raw_run.final_predicted_class == raw_run.true_class:
                raw_spike_correct += 1
            if robust_run.final_predicted_class == robust_run.true_class:
                robust_spike_correct += 1
    return WindowedBenchmarkSummary(
        total_trajectories=len(raw_runs),
        raw_final_accuracy=sum(per_class_correct_raw.values()) / max(len(raw_runs), 1),
        robust_final_accuracy=sum(per_class_correct_robust.values()) / max(len(robust_runs), 1),
        raw_spike_accuracy=raw_spike_correct / max(spike_cases, 1),
        robust_spike_accuracy=robust_spike_correct / max(spike_cases, 1),
        per_class_accuracy_raw={name: per_class_correct_raw[name] / per_class_total[name] if per_class_total[name] else 0.0 for name in class_names},
        per_class_accuracy_robust={name: per_class_correct_robust[name] / per_class_total[name] if per_class_total[name] else 0.0 for name in class_names},
        confusion_raw=confusion_raw,
        confusion_robust=confusion_robust,
    )


def _render_report(result: WindowedBenchmarkResult) -> str:
    summary = result.summary
    report = MarkdownDocument("Windowed Feature Baseline")
    report.paragraph(
        "This benchmark compares a raw-extrema classifier against a robust-extrema classifier "
        "built on sliding-window features."
    )
    report.heading("Summary", level=2)
    report.bullet_list(
        [
            f"Trajectories: {summary.total_trajectories}",
            f"Raw final accuracy: {summary.raw_final_accuracy:.3f}",
            f"Robust final accuracy: {summary.robust_final_accuracy:.3f}",
            f"Raw spike accuracy: {summary.raw_spike_accuracy:.3f}",
            f"Robust spike accuracy: {summary.robust_spike_accuracy:.3f}",
        ]
    )
    report.heading("Per-Class Accuracy", level=2)
    report.bullet_list(
        [
            (
                f"`{class_name}` raw={summary.per_class_accuracy_raw[class_name]:.3f} "
                f"robust={summary.per_class_accuracy_robust[class_name]:.3f}"
            )
            for class_name in summary.per_class_accuracy_raw
        ]
    )
    report.heading("Acceptance Notes", level=2)
    report.bullet_list(
        [
            "Running minima and maxima are tracked explicitly.",
            "Sliding-window features use elapsed time, not just sample count.",
            "Robust extrema should be less sensitive to spikes and longer histories than raw extrema.",
        ]
    )
    return report.text()


def _build_figure(result: WindowedBenchmarkResult):
    spike_run = next(run for run in result.raw_runs if "spike" in run.scenario_name)
    raw_run = spike_run
    robust_run = next(run for run in result.robust_runs if run.scenario_name == spike_run.scenario_name)
    class_names = [spec.name for spec in result.class_specs]
    colors = {name: color for name, color in zip(class_names, ("#2563eb", "#7c3aed"))}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=False)
    measurement_ax, raw_ax = axes[0]
    _, robust_ax = axes[1]
    feature_ax = axes[1][0]

    times = [row.time for row in extract_windowed_feature_rows(next(traj for traj in result.trajectories if traj.scenario_name == spike_run.scenario_name))]
    measurements = list(next(traj for traj in result.trajectories if traj.scenario_name == spike_run.scenario_name).measurements)
    measurement_ax.plot(times, measurements, color="#111827", linewidth=2.0)
    measurement_ax.set_title("Representative spiky trajectory", loc="left", fontsize=12, fontweight="bold")
    measurement_ax.grid(True, alpha=0.25)
    measurement_ax.set_ylabel("measurement")

    for spec in result.class_specs:
        raw_ax.plot([step.time for step in raw_run.steps], [step.posterior_weights[spec.name] for step in raw_run.steps], color=colors[spec.name], label=spec.name)
        robust_ax.plot([step.time for step in robust_run.steps], [step.posterior_weights[spec.name] for step in robust_run.steps], color=colors[spec.name], label=spec.name)
    raw_ax.set_title("Raw extrema posterior", loc="left", fontsize=12, fontweight="bold")
    robust_ax.set_title("Robust extrema posterior", loc="left", fontsize=12, fontweight="bold")
    raw_ax.set_ylim(0.0, 1.0)
    robust_ax.set_ylim(0.0, 1.0)
    raw_ax.grid(True, alpha=0.25)
    robust_ax.grid(True, alpha=0.25)
    raw_ax.legend(frameon=False)
    robust_ax.legend(frameon=False)

    feature_rows = extract_windowed_feature_rows(next(traj for traj in result.trajectories if traj.scenario_name == spike_run.scenario_name))
    feature_ax.plot([row.time for row in feature_rows], [row.running_max for row in feature_rows], color="#b45309", label="raw running max")
    feature_ax.plot([row.time for row in feature_rows], [row.robust_max for row in feature_rows], color="#059669", label="robust max")
    feature_ax.plot([row.time for row in feature_rows], [row.running_min for row in feature_rows], color="#dc2626", label="raw running min")
    feature_ax.plot([row.time for row in feature_rows], [row.robust_min for row in feature_rows], color="#7c3aed", label="robust min")
    feature_ax.set_title("Extrema comparison", loc="left", fontsize=12, fontweight="bold")
    feature_ax.grid(True, alpha=0.25)
    feature_ax.legend(frameon=False)

    feature_ax.set_xlabel("time")
    raw_ax.set_xlabel("time")
    robust_ax.set_xlabel("time")
    feature_ax.set_ylabel("feature value")
    fig.suptitle("Windowed Feature Baseline Diagnostics", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def render_windowed_benchmark_report(result: WindowedBenchmarkResult) -> str:
    return _render_report(result)


def render_windowed_benchmark_svg(result: WindowedBenchmarkResult) -> str:
    fig = _build_figure(result)
    buffer = io.StringIO()
    try:
        fig.savefig(buffer, format="svg", bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def render_windowed_benchmark_png_bytes(result: WindowedBenchmarkResult) -> bytes:
    fig = _build_figure(result)
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def write_windowed_benchmark_artifacts(
    output_dir: str | Path,
    *,
    seed: int = 7,
    result: WindowedBenchmarkResult | None = None,
) -> WindowedBenchmarkArtifacts:
    benchmark_result = result or run_windowed_benchmark(seed=seed)
    output_root = Path(output_dir)
    run_dir = output_root / "windowed_baseline"
    run_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / "windowed_baseline_report.md"
    feature_matrix_path = run_dir / "feature_matrix.csv"
    posterior_history_path = run_dir / "posterior_history.csv"
    confusion_raw_path = run_dir / "confusion_raw.csv"
    confusion_robust_path = run_dir / "confusion_robust.csv"
    config_path = run_dir / "windowed_classifier_config.yaml"
    dataset_manifest_path = run_dir / "dataset_manifest.json"
    feature_manifest_path = run_dir / "feature_manifest.json"
    plot_png_path = run_dir / "windowed_baseline_diagnostics.png"

    report_path.write_text(render_windowed_benchmark_report(benchmark_result), encoding="utf-8")
    plot_png_path.write_bytes(render_windowed_benchmark_png_bytes(benchmark_result))

    feature_rows_dicts = [asdict(row) for row in benchmark_result.feature_rows]
    write_csv(
        feature_matrix_path,
        feature_rows_dicts,
        [
            "trajectory_id",
            "time",
            "window_start",
            "window_end",
            "sample_count",
            "duration",
            "running_min",
            "running_max",
            "window_min",
            "window_max",
            "robust_min",
            "robust_max",
            "running_range",
            "window_range",
            "trimmed_range",
            "window_mean",
            "window_std",
            "slope",
            "monotonicity",
            "sign_changes",
        ],
    )

    posterior_rows: list[dict[str, object]] = []
    for run in (*benchmark_result.raw_runs, *benchmark_result.robust_runs):
        for step_index, step in enumerate(run.steps):
            posterior_rows.append(
                {
                    "trajectory_id": run.trajectory_id,
                    "scenario_name": run.scenario_name,
                    "feature_mode": run.feature_mode,
                    "step": step_index,
                    "time": step.time,
                    "true_class": run.true_class,
                    "predicted_class": step.predicted_class,
                    "confidence": step.confidence,
                    **{f"posterior_{name}": step.posterior_weights[name] for name in benchmark_result.summary.confusion_raw},
                    **{f"log_likelihood_{name}": step.log_likelihood_terms[name] for name in benchmark_result.summary.confusion_raw},
                }
            )
    posterior_fieldnames = [
        "trajectory_id",
        "scenario_name",
        "feature_mode",
        "step",
        "time",
        "true_class",
        "predicted_class",
        "confidence",
        *[f"posterior_{name}" for name in benchmark_result.summary.confusion_raw],
        *[f"log_likelihood_{name}" for name in benchmark_result.summary.confusion_raw],
    ]
    write_csv(posterior_history_path, posterior_rows, posterior_fieldnames)
    write_csv(confusion_raw_path, [{"true_class": true, **preds} for true, preds in benchmark_result.summary.confusion_raw.items()], ["true_class", *benchmark_result.summary.confusion_raw.keys()])
    write_csv(confusion_robust_path, [{"true_class": true, **preds} for true, preds in benchmark_result.summary.confusion_robust.items()], ["true_class", *benchmark_result.summary.confusion_robust.keys()])

    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  name: windowed_baseline",
                f"  seed: {seed}",
                "classifier:",
                "  types: [raw, robust]",
                "dataset:",
                "  scenarios: [low_clean, high_clean, low_spike, high_dip, low_long, high_long]",
                "  window_size: 5",
                "  trim_fraction: 0.2",
                "",
            ]
        ),
        encoding="utf-8",
    )
    dataset_manifest_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "scenario_names": [trajectory.scenario_name for trajectory in benchmark_result.trajectories],
                "trajectory_count": benchmark_result.summary.total_trajectories,
                "spike_cases": [trajectory.scenario_name for trajectory in benchmark_result.trajectories if "spike" in trajectory.scenario_name or "dip" in trajectory.scenario_name],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    feature_manifest_path.write_text(
        json.dumps(
            {
                "feature_names": [
                    "running_min",
                    "running_max",
                    "window_min",
                    "window_max",
                    "robust_min",
                    "robust_max",
                    "running_range",
                    "window_range",
                    "trimmed_range",
                    "window_mean",
                    "window_std",
                    "slope",
                    "monotonicity",
                    "sign_changes",
                ],
                "history_behavior": {
                    "running_min": "cumulative",
                    "running_max": "cumulative",
                    "window_min": "windowed",
                    "window_max": "windowed",
                    "robust_min": "windowed",
                    "robust_max": "windowed",
                    "running_range": "cumulative",
                    "window_range": "windowed",
                    "trimmed_range": "windowed",
                    "window_mean": "windowed",
                    "window_std": "windowed",
                    "slope": "windowed",
                    "monotonicity": "windowed",
                    "sign_changes": "windowed",
                },
                "feature_units": {
                    "running_min": "value",
                    "running_max": "value",
                    "window_min": "value",
                    "window_max": "value",
                    "robust_min": "value",
                    "robust_max": "value",
                    "running_range": "value",
                    "window_range": "value",
                    "trimmed_range": "value",
                    "window_mean": "value",
                    "window_std": "value",
                    "slope": "value / time",
                    "monotonicity": "unitless",
                    "sign_changes": "count",
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return WindowedBenchmarkArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        feature_matrix_path=feature_matrix_path,
        posterior_history_path=posterior_history_path,
        confusion_raw_path=confusion_raw_path,
        confusion_robust_path=confusion_robust_path,
        config_path=config_path,
        dataset_manifest_path=dataset_manifest_path,
        feature_manifest_path=feature_manifest_path,
        plot_png_path=plot_png_path,
    )
