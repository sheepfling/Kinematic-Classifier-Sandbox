from __future__ import annotations

import io
import random
from dataclasses import asdict, dataclass
from math import log
from pathlib import Path

from kinematic_classifier_sandbox.reports.markdown import MarkdownDocument

from ...utils.io import write_csv
from ...utils.math import (
    _gaussian_logpdf,
    _least_squares_slope,
    _mean,
    _normalize,
    _quadratic_fit,
    _std,
)
from ...utils.plotting import plt


@dataclass(frozen=True, slots=True)
class WindowRegimeTrajectory:
    trajectory_id: str
    true_class: str
    sampling_regime: str
    seed: int
    times: tuple[float, ...]
    measurements: tuple[float, ...]
    true_positions: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class WindowRegimeFeatureRow:
    trajectory_id: str
    true_class: str
    sampling_regime: str
    window_mode: str
    window_end: float
    window_start: float
    duration: float
    sample_count: int
    slope: float
    curvature_proxy: float
    position_range: float


@dataclass(frozen=True, slots=True)
class WindowRegimeSummaryRow:
    window_mode: str
    feature_name: str
    mean_regular_irregular_gap: float
    classification_accuracy: float
    mean_window_duration_irregular: float
    mean_window_sample_count_irregular: float


@dataclass(frozen=True, slots=True)
class WindowRegimeComparisonResult:
    trajectories: tuple[WindowRegimeTrajectory, ...]
    feature_rows: tuple[WindowRegimeFeatureRow, ...]
    summary_rows: tuple[WindowRegimeSummaryRow, ...]


@dataclass(frozen=True, slots=True)
class WindowRegimeArtifacts:
    run_dir: Path
    report_path: Path
    summary_path: Path
    features_path: Path
    plot_png_path: Path


REGULAR_TIMES = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
IRREGULAR_TIMES = (0.0, 0.7, 1.6, 2.8, 4.1, 5.0, 6.6, 7.4, 8.9, 10.0)


def _true_positions(class_name: str, times: tuple[float, ...]) -> tuple[float, ...]:
    if class_name == "constant_velocity":
        speed = 0.80
        return tuple(speed * time for time in times)
    if class_name == "constant_acceleration":
        speed0 = 0.30
        accel = 0.10
        return tuple(speed0 * time + 0.5 * accel * time * time for time in times)
    raise KeyError(class_name)


def generate_window_regime_trajectories(*, seed: int = 7, obs_sigma: float = 0.10, replicas: int = 12) -> tuple[WindowRegimeTrajectory, ...]:
    trajectories: list[WindowRegimeTrajectory] = []
    for class_index, class_name in enumerate(("constant_velocity", "constant_acceleration")):
        for regime_index, (sampling_regime, times) in enumerate((("regular", REGULAR_TIMES), ("irregular", IRREGULAR_TIMES))):
            for replica in range(replicas):
                local_seed = seed + class_index * 1000 + regime_index * 100 + replica
                rng = random.Random(local_seed)
                truth = _true_positions(class_name, times)
                measurements = tuple(value + rng.gauss(0.0, obs_sigma) for value in truth)
                trajectories.append(
                    WindowRegimeTrajectory(
                        trajectory_id=f"{class_name}_{sampling_regime}_{replica}",
                        true_class=class_name,
                        sampling_regime=sampling_regime,
                        seed=local_seed,
                        times=times,
                        measurements=measurements,
                        true_positions=truth,
                    )
                )
    return tuple(trajectories)


def _sample_count_window(trajectory: WindowRegimeTrajectory, sample_count: int) -> WindowRegimeFeatureRow:
    window_times = list(trajectory.times[-sample_count:])
    window_values = list(trajectory.measurements[-sample_count:])
    curvature = _quadratic_fit(window_times, window_values).curvature
    return WindowRegimeFeatureRow(
        trajectory_id=trajectory.trajectory_id,
        true_class=trajectory.true_class,
        sampling_regime=trajectory.sampling_regime,
        window_mode="sample_count",
        window_end=window_times[-1],
        window_start=window_times[0],
        duration=window_times[-1] - window_times[0],
        sample_count=len(window_values),
        slope=_least_squares_slope(window_times, window_values),
        curvature_proxy=2.0 * curvature,
        position_range=max(window_values) - min(window_values),
    )


def _duration_window(trajectory: WindowRegimeTrajectory, duration: float) -> WindowRegimeFeatureRow:
    end_time = trajectory.times[-1]
    keep = [
        index
        for index, time in enumerate(trajectory.times)
        if end_time - time <= duration + 1e-9
    ]
    window_times = [trajectory.times[index] for index in keep]
    window_values = [trajectory.measurements[index] for index in keep]
    if len(window_times) < 2:
        window_times = list(trajectory.times[-2:])
        window_values = list(trajectory.measurements[-2:])
    curvature = _quadratic_fit(window_times, window_values).curvature
    return WindowRegimeFeatureRow(
        trajectory_id=trajectory.trajectory_id,
        true_class=trajectory.true_class,
        sampling_regime=trajectory.sampling_regime,
        window_mode="duration",
        window_end=window_times[-1],
        window_start=window_times[0],
        duration=window_times[-1] - window_times[0],
        sample_count=len(window_values),
        slope=_least_squares_slope(window_times, window_values),
        curvature_proxy=2.0 * curvature,
        position_range=max(window_values) - min(window_values),
    )


def analyze_irregular_window_comparison(
    *,
    seed: int = 7,
    replicas: int = 12,
    obs_sigma: float = 0.10,
    sample_count_window: int = 5,
    duration_window: float = 5.0,
) -> WindowRegimeComparisonResult:
    trajectories = generate_window_regime_trajectories(seed=seed, obs_sigma=obs_sigma, replicas=replicas)
    feature_rows: list[WindowRegimeFeatureRow] = []
    for trajectory in trajectories:
        feature_rows.append(_sample_count_window(trajectory, sample_count_window))
        feature_rows.append(_duration_window(trajectory, duration_window))

    grouped: dict[tuple[str, str, int], dict[str, WindowRegimeFeatureRow]] = {}
    for row in feature_rows:
        parts = row.trajectory_id.split("_")
        replica = int(parts[-1])
        grouped.setdefault((row.true_class, row.window_mode, replica), {})[row.sampling_regime] = row

    summary_rows: list[WindowRegimeSummaryRow] = []
    for window_mode in ("sample_count", "duration"):
        gaps_by_feature = {"slope": [], "curvature_proxy": [], "position_range": []}
        irregular_rows = [row for row in feature_rows if row.window_mode == window_mode and row.sampling_regime == "irregular"]
        reference_rows = [row for row in feature_rows if row.window_mode == window_mode and row.sampling_regime == "regular"]
        class_means: dict[str, dict[str, float]] = {}
        class_sigmas: dict[str, dict[str, float]] = {}
        for class_name in ("constant_velocity", "constant_acceleration"):
            selected = [row for row in reference_rows if row.true_class == class_name]
            class_means[class_name] = {
                "slope": _mean([row.slope for row in selected]),
                "curvature_proxy": _mean([row.curvature_proxy for row in selected]),
                "position_range": _mean([row.position_range for row in selected]),
            }
            class_sigmas[class_name] = {
                "slope": max(_std([row.slope for row in selected]), 0.05),
                "curvature_proxy": max(_std([row.curvature_proxy for row in selected]), 0.05),
                "position_range": max(_std([row.position_range for row in selected]), 0.05),
            }
        hits: list[float] = []
        for row in irregular_rows:
            log_scores: dict[str, float] = {}
            for class_name in class_means:
                score = log(0.5)
                for feature_name in ("slope", "curvature_proxy", "position_range"):
                    score += _gaussian_logpdf(
                        float(getattr(row, feature_name)),
                        class_means[class_name][feature_name],
                        class_sigmas[class_name][feature_name],
                    )
                log_scores[class_name] = score
            weights = _normalize(log_scores)
            predicted = max(weights, key=weights.get)
            hits.append(1.0 if predicted == row.true_class else 0.0)
        for class_name in ("constant_velocity", "constant_acceleration"):
            for replica in range(replicas):
                pair = grouped[(class_name, window_mode, replica)]
                regular = pair["regular"]
                irregular = pair["irregular"]
                gaps_by_feature["slope"].append(abs(irregular.slope - regular.slope))
                gaps_by_feature["curvature_proxy"].append(abs(irregular.curvature_proxy - regular.curvature_proxy))
                gaps_by_feature["position_range"].append(abs(irregular.position_range - regular.position_range))
        mean_duration_irregular = _mean([row.duration for row in irregular_rows])
        mean_sample_count_irregular = _mean([float(row.sample_count) for row in irregular_rows])
        for feature_name, values in gaps_by_feature.items():
            summary_rows.append(
                WindowRegimeSummaryRow(
                    window_mode=window_mode,
                    feature_name=feature_name,
                    mean_regular_irregular_gap=_mean(values),
                    classification_accuracy=_mean(hits),
                    mean_window_duration_irregular=mean_duration_irregular,
                    mean_window_sample_count_irregular=mean_sample_count_irregular,
                )
            )
    return WindowRegimeComparisonResult(
        trajectories=trajectories,
        feature_rows=tuple(feature_rows),
        summary_rows=tuple(summary_rows),
    )


def render_irregular_window_report(result: WindowRegimeComparisonResult) -> str:
    report = MarkdownDocument("Irregular Window Comparison")
    report.paragraph(
        "Milestone 15 comparison between sample-count windows and elapsed-duration windows on matched regular and irregular tracks."
    )
    report.heading("Summary", level=2)
    report.table(
        [
            "window_mode",
            "feature_name",
            "mean_regular_irregular_gap",
            "classification_accuracy",
            "mean_window_duration_irregular",
            "mean_window_sample_count_irregular",
        ],
        [
            (
                row.window_mode,
                row.feature_name,
                f"{row.mean_regular_irregular_gap:.3f}",
                f"{row.classification_accuracy:.3f}",
                f"{row.mean_window_duration_irregular:.3f}",
                f"{row.mean_window_sample_count_irregular:.3f}",
            )
            for row in result.summary_rows
        ],
    )
    report.heading("Notes", level=2)
    report.bullet_list(
        [
            "Sample-count windows drift in elapsed duration under irregular sampling.",
            "Duration windows stabilize the time horizon and should reduce regular-vs-irregular feature mismatch.",
            "Classification is evaluated only on irregular tracks using regular-track feature references, so the comparison isolates window policy rather than class-model changes.",
        ]
    )
    return report.text()


def _build_figure(result: WindowRegimeComparisonResult):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    gap_ax, accuracy_ax = axes
    feature_order = ("slope", "curvature_proxy", "position_range")
    mode_order = ("sample_count", "duration")
    colors = {"sample_count": "#b45309", "duration": "#059669"}

    for idx, feature_name in enumerate(feature_order):
        values = {
            row.window_mode: row.mean_regular_irregular_gap
            for row in result.summary_rows
            if row.feature_name == feature_name
        }
        for mode_index, mode in enumerate(mode_order):
            gap_ax.bar(idx + (mode_index - 0.5) * 0.28, values[mode], width=0.24, color=colors[mode], label=mode if idx == 0 else None)
    gap_ax.set_xticks(range(len(feature_order)))
    gap_ax.set_xticklabels(["slope", "curvature", "range"])
    gap_ax.set_title("Regular vs irregular feature gap", loc="left", fontsize=12, fontweight="bold")
    gap_ax.grid(True, axis="y", alpha=0.25)
    gap_ax.legend(frameon=False)

    accuracies = {
        mode: _mean([row.classification_accuracy for row in result.summary_rows if row.window_mode == mode])
        for mode in mode_order
    }
    accuracy_ax.bar(range(len(mode_order)), [accuracies[mode] for mode in mode_order], color=[colors[mode] for mode in mode_order], width=0.5)
    accuracy_ax.set_xticks(range(len(mode_order)))
    accuracy_ax.set_xticklabels(["sample_count", "duration"])
    accuracy_ax.set_ylim(0.0, 1.0)
    accuracy_ax.set_title("Irregular-track classification", loc="left", fontsize=12, fontweight="bold")
    accuracy_ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("M15 Window Policy Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def write_irregular_window_artifacts(
    output_dir: str | Path,
    *,
    result: WindowRegimeComparisonResult | None = None,
) -> WindowRegimeArtifacts:
    analysis = result or analyze_irregular_window_comparison()
    run_dir = Path(output_dir) / "irregular_window_comparison_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "irregular_window_comparison_report.md"
    summary_path = run_dir / "irregular_window_summary.csv"
    features_path = run_dir / "irregular_window_features.csv"
    plot_png_path = run_dir / "irregular_window_comparison.png"

    report_path.write_text(render_irregular_window_report(analysis), encoding="utf-8")
    write_csv(
        summary_path,
        [asdict(row) for row in analysis.summary_rows],
        [
            "window_mode",
            "feature_name",
            "mean_regular_irregular_gap",
            "classification_accuracy",
            "mean_window_duration_irregular",
            "mean_window_sample_count_irregular",
        ],
    )
    write_csv(
        features_path,
        [asdict(row) for row in analysis.feature_rows],
        [
            "trajectory_id",
            "true_class",
            "sampling_regime",
            "window_mode",
            "window_end",
            "window_start",
            "duration",
            "sample_count",
            "slope",
            "curvature_proxy",
            "position_range",
        ],
    )
    fig = _build_figure(analysis)
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        plot_png_path.write_bytes(buffer.getvalue())
    finally:
        plt.close(fig)
    return WindowRegimeArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        summary_path=summary_path,
        features_path=features_path,
        plot_png_path=plot_png_path,
    )
