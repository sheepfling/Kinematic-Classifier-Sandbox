from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from kinematic_classifier_sandbox.analysis.common_dataset_comparison import (
    _kalman_predict,
    _rocket_proxy_predict,
    _windowed_predict,
    generate_shared_dynamics_dataset,
)
from kinematic_classifier_sandbox.analysis.common_dataset_comparison_contracts import (
    SharedDynamicsTrajectory,
)
from kinematic_classifier_sandbox.analysis.embedding_baseline_frontier import (
    CLASS_NAMES,
    EMBEDDING_DIMENSION,
    Ts2VecProxyClassifier,
    fit_ts2vec_proxy_classifier,
    predict_ts2vec_proxy,
)
from kinematic_classifier_sandbox.corpus.trajectory_exploration.comparison_surface import (
    write_comparison_summary_csv,
)
from kinematic_classifier_sandbox.utils.io import write_csv
from kinematic_classifier_sandbox.utils.plotting import _figure_to_png, plt


def _trajectory_split(trajectory: SharedDynamicsTrajectory) -> str:
    return "train" if int(trajectory.trajectory_id.rsplit("_", 1)[-1]) < 4 else "test"


def _accuracy(rows: list["Ts2VecBackendParityPredictionRow"], *, scenario_name: str | None = None) -> float:
    selected = rows if scenario_name is None else [row for row in rows if row.scenario_name == scenario_name]
    return sum(1.0 if row.predicted_class == row.true_class else 0.0 for row in selected) / max(len(selected), 1)


@dataclass(frozen=True, slots=True)
class Ts2VecBackendParityPredictionRow:
    trajectory_id: str
    scenario_name: str
    split: str
    true_class: str
    method_name: str
    backend_name: str
    claim_level: str
    predicted_class: str
    confidence: float


@dataclass(frozen=True, slots=True)
class Ts2VecBackendParityMetricRow:
    method_name: str
    backend_name: str
    claim_level: str
    overall_accuracy: float
    test_accuracy: float
    short_noisy_accuracy: float
    endpoint_match_accuracy: float
    mean_confidence: float


@dataclass(frozen=True, slots=True)
class Ts2VecBackendParityResult:
    prediction_rows: tuple[Ts2VecBackendParityPredictionRow, ...]
    metric_rows: tuple[Ts2VecBackendParityMetricRow, ...]
    metrics: dict[str, float | int | str]


@dataclass(frozen=True, slots=True)
class Ts2VecBackendParityArtifacts:
    run_dir: Path
    prediction_summary_path: Path
    metric_summary_path: Path
    summary_path: Path
    metrics_path: Path
    report_path: Path
    decision_card_path: Path
    plot_paths: tuple[Path, ...]


def _baseline_predictions(
    trajectory: SharedDynamicsTrajectory,
) -> tuple[tuple[str, str, str, str, float], ...]:
    windowed_run = _windowed_predict(trajectory, robust=True)
    rocket_run = _rocket_proxy_predict(trajectory)
    kalman_run = _kalman_predict(trajectory)
    return (
        ("windowed_robust", "windowed_robust", "baseline", windowed_run.final_predicted_class, windowed_run.final_confidence),
        ("rocket_proxy", "rocket_proxy", "implemented_proxy", rocket_run.final_predicted_class, rocket_run.final_confidence),
        ("kalman_bank", "kalman_bank", "baseline", kalman_run.final_predicted_class, kalman_run.final_confidence),
    )


def _ts2vec_predictions(
    trajectory: SharedDynamicsTrajectory,
    *,
    variant_name: str,
    classifier: Ts2VecProxyClassifier,
    claim_level: str,
) -> tuple[tuple[str, str, str, str, float], ...]:
    centroid_predicted, centroid_confidence, _ = predict_ts2vec_proxy(
        trajectory,
        classifier=classifier,
        strategy="centroid",
    )
    nn_predicted, nn_confidence, _ = predict_ts2vec_proxy(
        trajectory,
        classifier=classifier,
        strategy="nn",
    )
    return (
        (
            f"{variant_name}_centroid",
            classifier.backend_name,
            claim_level,
            centroid_predicted,
            centroid_confidence,
        ),
        (
            f"{variant_name}_nn",
            classifier.backend_name,
            claim_level,
            nn_predicted,
            nn_confidence,
        ),
    )


def analyze_ts2vec_backend_parity(
    *,
    seed: int = 913,
    trajectories_per_case: int = 8,
    embedding_dimension: int = EMBEDDING_DIMENSION,
) -> Ts2VecBackendParityResult:
    trajectories = generate_shared_dynamics_dataset(seed=seed, trajectories_per_case=trajectories_per_case)
    proxy_classifier = fit_ts2vec_proxy_classifier(
        trajectories,
        embedding_dimension=embedding_dimension,
        backend_mode="proxy_only",
    )
    try:
        external_classifier = fit_ts2vec_proxy_classifier(
            trajectories,
            embedding_dimension=embedding_dimension,
            backend_mode="external_only",
        )
    except ValueError:
        external_classifier = None

    prediction_rows: list[Ts2VecBackendParityPredictionRow] = []
    for trajectory in trajectories:
        split = _trajectory_split(trajectory)
        for method_name, backend_name, claim_level, predicted_class, confidence in _baseline_predictions(trajectory):
            prediction_rows.append(
                Ts2VecBackendParityPredictionRow(
                    trajectory_id=trajectory.trajectory_id,
                    scenario_name=trajectory.scenario_name,
                    split=split,
                    true_class=trajectory.true_class,
                    method_name=method_name,
                    backend_name=backend_name,
                    claim_level=claim_level,
                    predicted_class=predicted_class,
                    confidence=float(confidence),
                )
            )
        for method_name, backend_name, claim_level, predicted_class, confidence in _ts2vec_predictions(
            trajectory,
            variant_name="ts2vec_proxy",
            classifier=proxy_classifier,
            claim_level="implemented_proxy",
        ):
            prediction_rows.append(
                Ts2VecBackendParityPredictionRow(
                    trajectory_id=trajectory.trajectory_id,
                    scenario_name=trajectory.scenario_name,
                    split=split,
                    true_class=trajectory.true_class,
                    method_name=method_name,
                    backend_name=backend_name,
                    claim_level=claim_level,
                    predicted_class=predicted_class,
                    confidence=float(confidence),
                )
            )
        if external_classifier is not None:
            for method_name, backend_name, claim_level, predicted_class, confidence in _ts2vec_predictions(
                trajectory,
                variant_name="ts2vec_external",
                classifier=external_classifier,
                claim_level="bounded_external",
            ):
                prediction_rows.append(
                    Ts2VecBackendParityPredictionRow(
                        trajectory_id=trajectory.trajectory_id,
                        scenario_name=trajectory.scenario_name,
                        split=split,
                        true_class=trajectory.true_class,
                        method_name=method_name,
                        backend_name=backend_name,
                        claim_level=claim_level,
                        predicted_class=predicted_class,
                        confidence=float(confidence),
                    )
                )

    method_names: list[str] = []
    for row in prediction_rows:
        if row.method_name not in method_names:
            method_names.append(row.method_name)
    metric_rows: list[Ts2VecBackendParityMetricRow] = []
    for method_name in method_names:
        method_rows = [row for row in prediction_rows if row.method_name == method_name]
        test_rows = [row for row in method_rows if row.split == "test"]
        representative = method_rows[0]
        metric_rows.append(
            Ts2VecBackendParityMetricRow(
                method_name=method_name,
                backend_name=representative.backend_name,
                claim_level=representative.claim_level,
                overall_accuracy=_accuracy(method_rows),
                test_accuracy=_accuracy(test_rows),
                short_noisy_accuracy=_accuracy(test_rows, scenario_name="short_noisy"),
                endpoint_match_accuracy=_accuracy(test_rows, scenario_name="endpoint_match"),
                mean_confidence=sum(row.confidence for row in test_rows) / max(len(test_rows), 1),
            )
        )

    row_map = {row.method_name: row for row in metric_rows}
    proxy_best_accuracy = max(
        row_map["ts2vec_proxy_centroid"].test_accuracy,
        row_map["ts2vec_proxy_nn"].test_accuracy,
    )
    external_best_accuracy = max(
        row.test_accuracy for row in metric_rows if row.method_name.startswith("ts2vec_external_")
    ) if external_classifier is not None else 0.0
    baseline_best_accuracy = max(
        row_map["windowed_robust"].test_accuracy,
        row_map["rocket_proxy"].test_accuracy,
        row_map["kalman_bank"].test_accuracy,
    )
    promotion_decision = (
        "external_ts2vec_candidate_signal"
        if external_classifier is not None
        and external_best_accuracy >= proxy_best_accuracy
        and external_best_accuracy >= baseline_best_accuracy
        else "external_ts2vec_bounded_gap"
        if external_classifier is not None
        else "external_ts2vec_unavailable"
    )
    metrics: dict[str, float | int | str] = {
        "study_id": "ts2vec_backend_parity_v1",
        "seed": seed,
        "trajectory_count": len(trajectories),
        "test_count": sum(1 for trajectory in trajectories if _trajectory_split(trajectory) == "test"),
        "class_count": len(CLASS_NAMES),
        "embedding_dimension": embedding_dimension,
        "proxy_backend": proxy_classifier.backend_name,
        "external_backend_available": "yes" if external_classifier is not None else "no",
        "external_backend": external_classifier.backend_name if external_classifier is not None else "unavailable",
        "proxy_best_test_accuracy": proxy_best_accuracy,
        "external_best_test_accuracy": external_best_accuracy,
        "best_baseline_test_accuracy": baseline_best_accuracy,
        "external_vs_proxy_best_delta": external_best_accuracy - proxy_best_accuracy if external_classifier is not None else 0.0,
        "promotion_decision": promotion_decision,
    }
    return Ts2VecBackendParityResult(
        prediction_rows=tuple(prediction_rows),
        metric_rows=tuple(metric_rows),
        metrics=metrics,
    )


def _render_accuracy_panel(result: Ts2VecBackendParityResult):
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    labels = [row.method_name for row in result.metric_rows]
    values = [row.test_accuracy for row in result.metric_rows]
    colors = [
        "#2563eb" if row.claim_level == "baseline"
        else "#0f766e" if row.method_name.startswith("ts2vec_proxy")
        else "#7c3aed" if row.method_name.startswith("ts2vec_external")
        else "#9ca3af"
        for row in result.metric_rows
    ]
    ax.bar(range(len(labels)), values, color=colors, width=0.65)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("test accuracy")
    ax.set_title("TS2Vec Backend Parity", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _render_gap_panel(result: Ts2VecBackendParityResult):
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    proxy = float(result.metrics["proxy_best_test_accuracy"])
    external = float(result.metrics["external_best_test_accuracy"])
    baseline = float(result.metrics["best_baseline_test_accuracy"])
    labels = ["proxy_best", "external_best", "baseline_best"]
    values = [proxy, external, baseline]
    colors = ["#0f766e", "#7c3aed", "#2563eb"]
    ax.bar(labels, values, color=colors, width=0.58)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("test accuracy")
    ax.set_title("TS2Vec Backend Gap Summary", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def write_ts2vec_backend_parity_artifacts(
    output_dir: str | Path,
    *,
    result: Ts2VecBackendParityResult | None = None,
    seed: int = 913,
    trajectories_per_case: int = 8,
    embedding_dimension: int = EMBEDDING_DIMENSION,
) -> Ts2VecBackendParityArtifacts:
    payload = result or analyze_ts2vec_backend_parity(
        seed=seed,
        trajectories_per_case=trajectories_per_case,
        embedding_dimension=embedding_dimension,
    )
    run_dir = Path(output_dir) / "ts2vec_backend_parity_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    prediction_summary_path = run_dir / "prediction_summary.csv"
    metric_summary_path = run_dir / "metric_summary.csv"
    summary_path = run_dir / "summary.csv"
    metrics_path = run_dir / "metrics.csv"
    report_path = run_dir / "ts2vec_backend_parity_report.md"
    decision_card_path = run_dir / "decision_card.md"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    accuracy_plot_path = plots_dir / "backend_accuracy.png"
    gap_plot_path = plots_dir / "backend_gap_summary.png"

    write_csv(
        prediction_summary_path,
        [asdict(row) for row in payload.prediction_rows],
        list(Ts2VecBackendParityPredictionRow.__dataclass_fields__.keys()),
    )
    write_csv(
        metric_summary_path,
        [asdict(row) for row in payload.metric_rows],
        list(Ts2VecBackendParityMetricRow.__dataclass_fields__.keys()),
    )
    write_comparison_summary_csv(run_dir, [payload.metrics], filename="summary.csv")
    write_csv(metrics_path, [payload.metrics], list(payload.metrics.keys()))

    report_lines = [
        "# TS2Vec Backend Parity",
        "",
        "- Study: `ts2vec_backend_parity_v1`",
        f"- Proxy backend: `{payload.metrics['proxy_backend']}`",
        f"- External backend available: `{payload.metrics['external_backend_available']}`",
        f"- External backend: `{payload.metrics['external_backend']}`",
        "- Compared methods: `windowed_robust`, `rocket_proxy`, `kalman_bank`, `ts2vec_proxy_*`, optional `ts2vec_external_*`",
        "",
        "## Purpose",
        "",
        "This packet checks proxy-versus-external TS2Vec behavior on the same shared 1D witness so the learned-embedding lane does not overclaim proxy or external coverage.",
        "",
        "## Read",
        "",
        f"- proxy best test accuracy: `{float(payload.metrics['proxy_best_test_accuracy']):.3f}`",
        f"- external best test accuracy: `{float(payload.metrics['external_best_test_accuracy']):.3f}`",
        f"- best baseline test accuracy: `{float(payload.metrics['best_baseline_test_accuracy']):.3f}`",
        f"- external minus proxy: `{float(payload.metrics['external_vs_proxy_best_delta']):.3f}`",
        f"- decision: `{payload.metrics['promotion_decision']}`",
        "",
        "This is a bounded parity witness, not a family-wide promotion claim.",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    decision_lines = [
        "# Decision Card",
        "",
        "- Candidate method: `ts2vec_external`",
        "- Comparator: `ts2vec_proxy` plus existing 1D baselines",
        "- Rule: `do not promote the external TS2Vec lane just because the package imports; require side-by-side parity evidence on the same witness`",
        f"- Proxy best: `{float(payload.metrics['proxy_best_test_accuracy']):.3f}`",
        f"- External best: `{float(payload.metrics['external_best_test_accuracy']):.3f}`",
        f"- Best baseline: `{float(payload.metrics['best_baseline_test_accuracy']):.3f}`",
        f"- Decision: `{payload.metrics['promotion_decision']}`",
    ]
    decision_card_path.write_text("\n".join(decision_lines) + "\n", encoding="utf-8")

    accuracy_plot_path.write_bytes(_figure_to_png(_render_accuracy_panel(payload)))
    gap_plot_path.write_bytes(_figure_to_png(_render_gap_panel(payload)))
    return Ts2VecBackendParityArtifacts(
        run_dir=run_dir,
        prediction_summary_path=prediction_summary_path,
        metric_summary_path=metric_summary_path,
        summary_path=summary_path,
        metrics_path=metrics_path,
        report_path=report_path,
        decision_card_path=decision_card_path,
        plot_paths=(accuracy_plot_path, gap_plot_path),
    )


__all__ = [
    "Ts2VecBackendParityArtifacts",
    "Ts2VecBackendParityMetricRow",
    "Ts2VecBackendParityPredictionRow",
    "Ts2VecBackendParityResult",
    "analyze_ts2vec_backend_parity",
    "write_ts2vec_backend_parity_artifacts",
]
