from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from kinematic_classifier_sandbox.analysis.common_dataset_comparison import (
    _class_expected_position,
    _kalman_predict,
    _rocket_proxy_predict,
    _windowed_predict,
    generate_shared_dynamics_dataset,
)
from kinematic_classifier_sandbox.analysis.common_dataset_comparison_contracts import SharedDynamicsTrajectory
from kinematic_classifier_sandbox.corpus.trajectory_exploration.comparison_surface import write_comparison_summary_csv
from kinematic_classifier_sandbox.utils.io import write_csv
from kinematic_classifier_sandbox.utils.plotting import _figure_to_png
from kinematic_classifier_sandbox.utils.plotting import plt


CLASS_NAMES = ("constant_velocity", "constant_acceleration")


@dataclass(frozen=True, slots=True)
class TSCArchivePredictionRow:
    trajectory_id: str
    scenario_name: str
    true_class: str
    method_name: str
    predicted_class: str
    confidence: float


@dataclass(frozen=True, slots=True)
class TSCArchiveMetricRow:
    method_name: str
    overall_accuracy: float
    short_noisy_accuracy: float
    outlier_accuracy: float
    endpoint_match_accuracy: float
    claim_level: str


@dataclass(frozen=True, slots=True)
class TSCArchiveFrontierResult:
    prediction_rows: tuple[TSCArchivePredictionRow, ...]
    metric_rows: tuple[TSCArchiveMetricRow, ...]
    metrics: dict[str, float | str | int]


@dataclass(frozen=True, slots=True)
class TSCArchiveFrontierArtifacts:
    run_dir: Path
    prediction_summary_path: Path
    metric_summary_path: Path
    summary_path: Path
    metrics_path: Path
    report_path: Path
    decision_card_path: Path
    plot_paths: tuple[Path, ...]


def _accuracy(rows: list[TSCArchivePredictionRow], *, scenario_name: str | None = None) -> float:
    selected = rows if scenario_name is None else [row for row in rows if row.scenario_name == scenario_name]
    return sum(1.0 if row.predicted_class == row.true_class else 0.0 for row in selected) / max(len(selected), 1)


def _ensemble_predict(trajectory: SharedDynamicsTrajectory) -> tuple[str, float]:
    members = (
        _windowed_predict(trajectory, robust=True),
        _rocket_proxy_predict(trajectory),
        _kalman_predict(trajectory),
    )
    scores = {class_name: 0.0 for class_name in CLASS_NAMES}
    for run in members:
        for class_name, probability in run.final_weights.items():
            scores[class_name] += probability
    total = max(sum(scores.values()), 1.0e-12)
    normalized = {class_name: value / total for class_name, value in scores.items()}
    predicted = max(normalized, key=normalized.get)
    return predicted, float(normalized[predicted])


def _interval_features(values: tuple[float, ...]) -> tuple[float, ...]:
    midpoint = len(values) // 2
    early = values[: midpoint + 1]
    late = values[midpoint:]
    center_start = max(0, midpoint - 1)
    center_stop = min(len(values), midpoint + 2)
    center = values[center_start:center_stop]
    early_slope = (early[-1] - early[0]) / max(len(early) - 1, 1)
    late_slope = (late[-1] - late[0]) / max(len(late) - 1, 1)
    center_range = max(center) - min(center)
    full_range = max(values) - min(values)
    return (
        float(early_slope),
        float(late_slope),
        float(center_range),
        float(full_range),
    )


def _drcif_proxy_predict(trajectory: SharedDynamicsTrajectory) -> tuple[str, float]:
    observed = _interval_features(trajectory.measurements)
    log_scores: dict[str, float] = {}
    for class_name in CLASS_NAMES:
        expected_series = tuple(
            _class_expected_position(class_name, time_value, trajectory.scenario_name)
            for time_value in trajectory.times
        )
        reference = _interval_features(expected_series)
        squared_distance = sum(
            (observed_value - reference_value) ** 2
            for observed_value, reference_value in zip(observed, reference, strict=True)
        )
        log_scores[class_name] = -0.5 * squared_distance
    score_max = max(log_scores.values())
    unnormalized = {label: pow(2.718281828459045, value - score_max) for label, value in log_scores.items()}
    total = max(sum(unnormalized.values()), 1.0e-12)
    normalized = {label: value / total for label, value in unnormalized.items()}
    predicted = max(normalized, key=normalized.get)
    return predicted, float(normalized[predicted])


def _symbolic_word(values: tuple[float, ...]) -> tuple[str, ...]:
    diffs = [values[index + 1] - values[index] for index in range(len(values) - 1)]
    mean_diff = sum(diffs) / max(len(diffs), 1)
    word: list[str] = []
    for diff in diffs:
        centered = diff - mean_diff
        if centered > 0.15:
            word.append("U")
        elif centered < -0.15:
            word.append("D")
        else:
            word.append("F")
    return tuple(word)


def _word_distance(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    shared = min(len(left), len(right))
    mismatch = sum(1.0 if left[index] != right[index] else 0.0 for index in range(shared))
    tail = abs(len(left) - len(right))
    return mismatch + tail


def _dictionary_proxy_predict(trajectory: SharedDynamicsTrajectory) -> tuple[str, float]:
    observed = _symbolic_word(trajectory.measurements)
    log_scores: dict[str, float] = {}
    for class_name in CLASS_NAMES:
        expected_series = tuple(
            _class_expected_position(class_name, time_value, trajectory.scenario_name)
            for time_value in trajectory.times
        )
        reference = _symbolic_word(expected_series)
        distance = _word_distance(observed, reference)
        log_scores[class_name] = -distance
    score_max = max(log_scores.values())
    unnormalized = {label: pow(2.718281828459045, value - score_max) for label, value in log_scores.items()}
    total = max(sum(unnormalized.values()), 1.0e-12)
    normalized = {label: value / total for label, value in unnormalized.items()}
    predicted = max(normalized, key=normalized.get)
    return predicted, float(normalized[predicted])


def analyze_tsc_archive_baseline_frontier(*, seed: int = 1009, trajectories_per_case: int = 8) -> TSCArchiveFrontierResult:
    trajectories = generate_shared_dynamics_dataset(seed=seed, trajectories_per_case=trajectories_per_case)
    prediction_rows: list[TSCArchivePredictionRow] = []
    for trajectory in trajectories:
        windowed_run = _windowed_predict(trajectory, robust=True)
        rocket_run = _rocket_proxy_predict(trajectory)
        kalman_run = _kalman_predict(trajectory)
        drcif_predicted, drcif_confidence = _drcif_proxy_predict(trajectory)
        dictionary_predicted, dictionary_confidence = _dictionary_proxy_predict(trajectory)
        methods = (
            ("windowed_robust", windowed_run.final_predicted_class, windowed_run.final_confidence),
            ("rocket_proxy", rocket_run.final_predicted_class, rocket_run.final_confidence),
            ("kalman_bank", kalman_run.final_predicted_class, kalman_run.final_confidence),
            ("drcif_proxy", drcif_predicted, drcif_confidence),
            ("dictionary_proxy", dictionary_predicted, dictionary_confidence),
        )
        for method_name, predicted_class, confidence in methods:
            prediction_rows.append(
                TSCArchivePredictionRow(
                    trajectory_id=trajectory.trajectory_id,
                    scenario_name=trajectory.scenario_name,
                    true_class=trajectory.true_class,
                    method_name=method_name,
                    predicted_class=predicted_class,
                    confidence=float(confidence),
                )
            )
        hive_predicted, hive_confidence = _ensemble_predict(trajectory)
        prediction_rows.append(
            TSCArchivePredictionRow(
                trajectory_id=trajectory.trajectory_id,
                scenario_name=trajectory.scenario_name,
                true_class=trajectory.true_class,
                method_name="hive_cote_proxy",
                predicted_class=hive_predicted,
                confidence=hive_confidence,
            )
        )

    metric_rows: list[TSCArchiveMetricRow] = []
    for method_name, claim_level in (
        ("windowed_robust", "baseline"),
        ("rocket_proxy", "implemented_proxy"),
        ("kalman_bank", "baseline"),
        ("drcif_proxy", "implemented_proxy"),
        ("dictionary_proxy", "implemented_proxy"),
        ("hive_cote_proxy", "implemented_proxy"),
    ):
        method_rows = [row for row in prediction_rows if row.method_name == method_name]
        metric_rows.append(
            TSCArchiveMetricRow(
                method_name=method_name,
                overall_accuracy=_accuracy(method_rows),
                short_noisy_accuracy=_accuracy(method_rows, scenario_name="short_noisy"),
                outlier_accuracy=_accuracy(method_rows, scenario_name="outlier"),
                endpoint_match_accuracy=_accuracy(method_rows, scenario_name="endpoint_match"),
                claim_level=claim_level,
            )
        )

    row_map = {row.method_name: row for row in metric_rows}
    metrics: dict[str, float | str | int] = {
        "study_id": "tsc_archive_baseline_frontier_v1",
        "seed": seed,
        "trajectory_count": len(trajectories),
        "rocket_overall_accuracy": row_map["rocket_proxy"].overall_accuracy,
        "drcif_overall_accuracy": row_map["drcif_proxy"].overall_accuracy,
        "dictionary_overall_accuracy": row_map["dictionary_proxy"].overall_accuracy,
        "hive_overall_accuracy": row_map["hive_cote_proxy"].overall_accuracy,
        "kalman_overall_accuracy": row_map["kalman_bank"].overall_accuracy,
        "promotion_decision": "hold_modern_tsc_at_proxy_stage",
    }
    return TSCArchiveFrontierResult(
        prediction_rows=tuple(prediction_rows),
        metric_rows=tuple(metric_rows),
        metrics=metrics,
    )


def _render_overall_accuracy(result: TSCArchiveFrontierResult):
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    labels = [row.method_name for row in result.metric_rows]
    values = [row.overall_accuracy for row in result.metric_rows]
    colors = ("#9ca3af", "#16a34a", "#2563eb", "#0891b2", "#be123c", "#7c3aed")
    ax.bar(range(len(labels)), values, color=colors, width=0.65)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("overall accuracy")
    ax.set_title("Modern TSC Proxy Frontier", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _render_scenario_slice(result: TSCArchiveFrontierResult):
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    labels = [row.method_name for row in result.metric_rows]
    x = list(range(len(labels)))
    width = 0.22
    short_noisy = [row.short_noisy_accuracy for row in result.metric_rows]
    outlier = [row.outlier_accuracy for row in result.metric_rows]
    endpoint = [row.endpoint_match_accuracy for row in result.metric_rows]
    ax.bar([value - width for value in x], short_noisy, width=width, label="short_noisy", color="#dc2626")
    ax.bar(x, outlier, width=width, label="outlier", color="#d97706")
    ax.bar([value + width for value in x], endpoint, width=width, label="endpoint_match", color="#2563eb")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title("Scenario Slice: TSC Proxy Frontier", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def write_tsc_archive_baseline_frontier_artifacts(
    output_dir: str | Path,
    *,
    result: TSCArchiveFrontierResult | None = None,
    seed: int = 1009,
    trajectories_per_case: int = 8,
) -> TSCArchiveFrontierArtifacts:
    payload = result or analyze_tsc_archive_baseline_frontier(
        seed=seed,
        trajectories_per_case=trajectories_per_case,
    )
    run_dir = Path(output_dir) / "tsc_archive_baseline_frontier_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    prediction_summary_path = run_dir / "prediction_summary.csv"
    metric_summary_path = run_dir / "metric_summary.csv"
    summary_path = run_dir / "summary.csv"
    metrics_path = run_dir / "metrics.csv"
    report_path = run_dir / "tsc_archive_baseline_frontier_report.md"
    decision_card_path = run_dir / "decision_card.md"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    overall_plot_path = plots_dir / "overall_accuracy.png"
    scenario_plot_path = plots_dir / "scenario_slice_accuracy.png"

    write_csv(prediction_summary_path, [asdict(row) for row in payload.prediction_rows], list(TSCArchivePredictionRow.__dataclass_fields__.keys()))
    write_csv(metric_summary_path, [asdict(row) for row in payload.metric_rows], list(TSCArchiveMetricRow.__dataclass_fields__.keys()))
    write_comparison_summary_csv(run_dir, [asdict(row) for row in payload.metric_rows], filename="summary.csv")
    write_csv(metrics_path, [payload.metrics], list(payload.metrics.keys()))

    report_lines = [
        "# TSC Archive Baseline Frontier",
        "",
        "- Study: `tsc_archive_baseline_frontier_v1`",
        "- Modern TSC proxies: `rocket_proxy`, `drcif_proxy`, `dictionary_proxy`, `hive_cote_proxy`",
        "- Baselines: `windowed_robust`, `kalman_bank`",
        "",
        "## Claim Boundary",
        "",
        "This packet provides an explicit proxy frontier for the modern TSC lane.",
        "It does not claim faithful MiniRocket, DrCIF, dictionary-method, or HIVE-COTE implementations.",
        "",
        f"- decision: `{payload.metrics['promotion_decision']}`",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    decision_lines = [
        "# Decision Card",
        "",
        "- Candidate methods: `minirocket_family`, `drcif_interval_forests`, `dictionary_tde_family`, `hive_cote`",
        "- Implementation state: explicit proxy frontier only",
        "- Decision: `hold at implemented until faithful external-method training or wrapping arrives`",
    ]
    decision_card_path.write_text("\n".join(decision_lines) + "\n", encoding="utf-8")

    overall_plot_path.write_bytes(_figure_to_png(_render_overall_accuracy(payload)))
    scenario_plot_path.write_bytes(_figure_to_png(_render_scenario_slice(payload)))

    return TSCArchiveFrontierArtifacts(
        run_dir=run_dir,
        prediction_summary_path=prediction_summary_path,
        metric_summary_path=metric_summary_path,
        summary_path=summary_path,
        metrics_path=metrics_path,
        report_path=report_path,
        decision_card_path=decision_card_path,
        plot_paths=(overall_plot_path, scenario_plot_path),
    )


__all__ = [
    "TSCArchiveFrontierArtifacts",
    "TSCArchiveFrontierMetricRow",
    "TSCArchiveFrontierResult",
    "TSCArchivePredictionRow",
    "analyze_tsc_archive_baseline_frontier",
    "write_tsc_archive_baseline_frontier_artifacts",
]
