from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
import warnings

from kinematic_classifier_sandbox.analysis.common_dataset_comparison import (
    _kalman_predict,
    _windowed_predict,
    generate_shared_dynamics_dataset,
)
from kinematic_classifier_sandbox.analysis.common_dataset_comparison_contracts import (
    SharedDynamicsTrajectory,
)
from kinematic_classifier_sandbox.analysis.gradient_boosted_feature_witness import (
    analyze_feature_headroom_frontier,
)
from kinematic_classifier_sandbox.analysis.optional_external_backends import (
    fit_archive_classifier_with_outcome,
)
from kinematic_classifier_sandbox.corpus.trajectory_exploration.comparison_surface import (
    write_comparison_summary_csv,
)
from kinematic_classifier_sandbox.utils.io import write_csv
from kinematic_classifier_sandbox.utils.plotting import _figure_to_png
from kinematic_classifier_sandbox.utils.plotting import plt


ARCHIVE_METHODS = (
    "minirocket_family",
    "drcif_interval_forests",
    "dictionary_tde_family",
    "hive_cote",
)
PANEL_VARIANTS = (
    "raw_position",
    "normalized_position",
    "normalized_position_velocity",
    "normalized_position_velocity_acceleration",
)
RESAMPLE_LENGTHS = (16, 24, 32, 64)


@dataclass(frozen=True, slots=True)
class ArchiveDiagnosisRow:
    witness_id: str
    method_name: str
    panel_variant: str
    resample_length: int
    backend_name: str
    fit_succeeded: bool
    test_accuracy: float
    test_nll: float
    warning_count: int
    warning_types: str
    delta_vs_best_baseline: float
    diagnosis_read: str


@dataclass(frozen=True, slots=True)
class ArchiveDiagnosisSummaryRow:
    witness_id: str
    method_name: str
    best_panel_variant: str
    best_resample_length: int
    best_test_accuracy: float
    worst_test_accuracy: float
    best_delta_vs_baseline: float
    mean_warning_count: float
    warning_dominated: str


@dataclass(frozen=True, slots=True)
class ArchiveBackendDiagnosisResult:
    diagnosis_rows: tuple[ArchiveDiagnosisRow, ...]
    summary_rows: tuple[ArchiveDiagnosisSummaryRow, ...]
    metrics: dict[str, float | int | str]


@dataclass(frozen=True, slots=True)
class ArchiveBackendDiagnosisArtifacts:
    run_dir: Path
    diagnosis_path: Path
    summary_path: Path
    metrics_path: Path
    report_path: Path
    decision_card_path: Path
    plot_paths: tuple[Path, ...]


def _trajectory_split(trajectory: SharedDynamicsTrajectory) -> str:
    return "train" if int(trajectory.trajectory_id.rsplit("_", 1)[-1]) < 4 else "test"


def _shared_baseline_accuracy(trajectories: tuple[SharedDynamicsTrajectory, ...]) -> float:
    test_rows = [trajectory for trajectory in trajectories if _trajectory_split(trajectory) == "test"]
    windowed_accuracy = sum(
        1.0 if _windowed_predict(trajectory, robust=True).final_predicted_class == trajectory.true_class else 0.0
        for trajectory in test_rows
    ) / max(len(test_rows), 1)
    kalman_accuracy = sum(
        1.0 if _kalman_predict(trajectory).final_predicted_class == trajectory.true_class else 0.0
        for trajectory in test_rows
    ) / max(len(test_rows), 1)
    return max(windowed_accuracy, kalman_accuracy)


def _convert_feature_headroom_trajectory(trajectory) -> SharedDynamicsTrajectory:
    velocities = [trajectory.measurements[index + 1] - trajectory.measurements[index] for index in range(len(trajectory.measurements) - 1)]
    if velocities:
        velocities.append(velocities[-1])
    else:
        velocities = [0.0]
    accelerations = [velocities[index + 1] - velocities[index] for index in range(len(velocities) - 1)]
    accelerations.append(accelerations[-1] if accelerations else 0.0)
    return SharedDynamicsTrajectory(
        trajectory_id=trajectory.trajectory_id,
        true_class=trajectory.true_class,
        scenario_name="feature_headroom",
        seed=0,
        times=trajectory.times,
        measurements=trajectory.measurements,
        true_position=trajectory.true_position,
        true_velocity=tuple(velocities[: len(trajectory.times)]),
        true_acceleration=tuple(accelerations[: len(trajectory.times)]),
    )


def _feature_headroom_dataset(*, seed: int, trajectories_per_class: int) -> tuple[tuple[SharedDynamicsTrajectory, ...], float]:
    witness = analyze_feature_headroom_frontier(seed=seed, trajectories_per_class=trajectories_per_class)
    trajectories = tuple(_convert_feature_headroom_trajectory(trajectory) for trajectory in witness.trajectories)
    return trajectories, float(witness.metrics["boosted_test_accuracy"])


def _binary_nll(rows: list[tuple[float, bool]]) -> float:
    if not rows:
        return float("inf")
    return sum(-math.log(max(probability if correct else 1.0 - probability, 1.0e-12)) for probability, correct in rows) / len(rows)


def _warning_type_summary(caught_warnings: list[warnings.WarningMessage]) -> str:
    labels = sorted({warning.category.__name__ for warning in caught_warnings})
    return ",".join(labels)


def _evaluate_archive_method(
    *,
    witness_id: str,
    method_name: str,
    train_rows: tuple[SharedDynamicsTrajectory, ...],
    test_rows: tuple[SharedDynamicsTrajectory, ...],
    class_names: tuple[str, ...],
    best_baseline_accuracy: float,
    panel_variant: str,
    resample_length: int,
) -> ArchiveDiagnosisRow:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        outcome = fit_archive_classifier_with_outcome(
            method_name,
            train_rows,
            class_names=class_names,
            resample_length=resample_length,
            panel_variant=panel_variant,
        )
        probability_rows: list[tuple[float, bool]] = []
        accuracy_count = 0
        if outcome.adapter is not None:
            predictions = outcome.adapter.predict_many(test_rows)
            for (predicted_class, confidence), trajectory in zip(predictions, test_rows, strict=True):
                correct = predicted_class == trajectory.true_class
                probability_rows.append((float(confidence), correct))
                accuracy_count += 1 if correct else 0
    test_accuracy = accuracy_count / max(len(test_rows), 1)
    warning_count = len(caught)
    diagnosis_read = (
        "warning_dominated_external_stack"
        if warning_count > 0
        else "representation_or_data_regime_gap"
        if outcome.succeeded and test_accuracy < best_baseline_accuracy
        else "candidate_recovery_signal"
        if outcome.succeeded and test_accuracy >= best_baseline_accuracy
        else "fit_failed"
    )
    return ArchiveDiagnosisRow(
        witness_id=witness_id,
        method_name=method_name,
        panel_variant=panel_variant,
        resample_length=resample_length,
        backend_name=outcome.backend_name,
        fit_succeeded=outcome.succeeded,
        test_accuracy=test_accuracy,
        test_nll=_binary_nll(probability_rows),
        warning_count=warning_count,
        warning_types=_warning_type_summary(caught),
        delta_vs_best_baseline=test_accuracy - best_baseline_accuracy,
        diagnosis_read=diagnosis_read,
    )


def analyze_archive_backend_diagnosis(
    *,
    shared_seed: int = 1009,
    shared_trajectories_per_case: int = 8,
    feature_seed: int = 811,
    feature_trajectories_per_class: int = 12,
    methods: tuple[str, ...] = ARCHIVE_METHODS,
    panel_variants: tuple[str, ...] = PANEL_VARIANTS,
    resample_lengths: tuple[int, ...] = RESAMPLE_LENGTHS,
) -> ArchiveBackendDiagnosisResult:
    shared_trajectories = generate_shared_dynamics_dataset(
        seed=shared_seed,
        trajectories_per_case=shared_trajectories_per_case,
    )
    shared_best_baseline = _shared_baseline_accuracy(shared_trajectories)
    feature_trajectories, feature_best_baseline = _feature_headroom_dataset(
        seed=feature_seed,
        trajectories_per_class=feature_trajectories_per_class,
    )

    witness_specs = (
        (
            "shared_binary_dynamics",
            tuple(trajectory for trajectory in shared_trajectories if _trajectory_split(trajectory) == "train"),
            tuple(trajectory for trajectory in shared_trajectories if _trajectory_split(trajectory) == "test"),
            ("constant_velocity", "constant_acceleration"),
            shared_best_baseline,
        ),
        (
            "feature_headroom",
            tuple(trajectory for trajectory in feature_trajectories if "train" in trajectory.trajectory_id or int(trajectory.trajectory_id.rsplit("_", 1)[-1]) < 8),
            tuple(trajectory for trajectory in feature_trajectories if "train" not in trajectory.trajectory_id and int(trajectory.trajectory_id.rsplit("_", 1)[-1]) >= 8),
            ("early_push_late_brake", "early_brake_late_push"),
            feature_best_baseline,
        ),
    )

    diagnosis_rows: list[ArchiveDiagnosisRow] = []
    for witness_id, train_rows, test_rows, class_names, baseline_accuracy in witness_specs:
        for method_name in methods:
            for panel_variant in panel_variants:
                for resample_length in resample_lengths:
                    diagnosis_rows.append(
                        _evaluate_archive_method(
                            witness_id=witness_id,
                            method_name=method_name,
                            train_rows=train_rows,
                            test_rows=test_rows,
                            class_names=class_names,
                            best_baseline_accuracy=baseline_accuracy,
                            panel_variant=panel_variant,
                            resample_length=resample_length,
                        )
                    )

    summary_rows: list[ArchiveDiagnosisSummaryRow] = []
    for witness_id, *_ in witness_specs:
        for method_name in methods:
            rows = [
                row
                for row in diagnosis_rows
                if row.witness_id == witness_id and row.method_name == method_name
            ]
            best_row = max(rows, key=lambda row: row.test_accuracy)
            summary_rows.append(
                ArchiveDiagnosisSummaryRow(
                    witness_id=witness_id,
                    method_name=method_name,
                    best_panel_variant=best_row.panel_variant,
                    best_resample_length=best_row.resample_length,
                    best_test_accuracy=best_row.test_accuracy,
                    worst_test_accuracy=min(row.test_accuracy for row in rows),
                    best_delta_vs_baseline=max(row.delta_vs_best_baseline for row in rows),
                    mean_warning_count=float(mean(row.warning_count for row in rows)),
                    warning_dominated="yes" if any(row.warning_count > 0 for row in rows) else "no",
                )
            )

    recovery_count = sum(1 for row in diagnosis_rows if row.delta_vs_best_baseline >= 0.0)
    warning_rows = sum(1 for row in diagnosis_rows if row.warning_count > 0)
    metrics: dict[str, float | int | str] = {
        "study_id": "archive_backend_diagnosis_v1",
        "diagnosis_row_count": len(diagnosis_rows),
        "summary_row_count": len(summary_rows),
        "warning_row_count": warning_rows,
        "recovery_row_count": recovery_count,
        "diagnosis_read": (
            "bounded_variants_do_not_recover_archive_lane"
            if recovery_count == 0
            else "some_variants_recover_archive_rows"
        ),
    }
    return ArchiveBackendDiagnosisResult(
        diagnosis_rows=tuple(diagnosis_rows),
        summary_rows=tuple(summary_rows),
        metrics=metrics,
    )


def _render_best_accuracy_panel(result: ArchiveBackendDiagnosisResult):
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    labels = [f"{row.witness_id}:{row.method_name}" for row in result.summary_rows]
    values = [row.best_test_accuracy for row in result.summary_rows]
    colors = ["#7c3aed" if row.warning_dominated == "no" else "#dc2626" for row in result.summary_rows]
    ax.bar(range(len(labels)), values, color=colors, width=0.65)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("best test accuracy")
    ax.set_title("Archive Diagnosis: Best Variant Accuracy", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _render_warning_panel(result: ArchiveBackendDiagnosisResult):
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    labels = [f"{row.witness_id}:{row.method_name}" for row in result.summary_rows]
    values = [row.mean_warning_count for row in result.summary_rows]
    ax.bar(range(len(labels)), values, color="#d97706", width=0.65)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("mean warning count")
    ax.set_title("Archive Diagnosis: Warning Load", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def write_archive_backend_diagnosis_artifacts(
    output_dir: str | Path,
    *,
    result: ArchiveBackendDiagnosisResult | None = None,
) -> ArchiveBackendDiagnosisArtifacts:
    payload = result or analyze_archive_backend_diagnosis()
    run_dir = Path(output_dir) / "archive_backend_diagnosis_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    diagnosis_path = run_dir / "diagnosis_rows.csv"
    summary_path = run_dir / "summary_rows.csv"
    metrics_path = run_dir / "metrics.csv"
    report_path = run_dir / "archive_backend_diagnosis_report.md"
    decision_card_path = run_dir / "decision_card.md"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    best_accuracy_path = plots_dir / "best_accuracy_by_variant.png"
    warning_path = plots_dir / "warning_load_by_method.png"

    write_csv(diagnosis_path, [asdict(row) for row in payload.diagnosis_rows], list(ArchiveDiagnosisRow.__dataclass_fields__.keys()))
    write_csv(summary_path, [asdict(row) for row in payload.summary_rows], list(ArchiveDiagnosisSummaryRow.__dataclass_fields__.keys()))
    write_comparison_summary_csv(run_dir, [asdict(row) for row in payload.summary_rows], filename="summary.csv")
    write_csv(metrics_path, [payload.metrics], list(payload.metrics.keys()))

    report_lines = [
        "# Archive Backend Diagnosis",
        "",
        "- Study: `archive_backend_diagnosis_v1`",
        "- Purpose: test whether archive-lane failure is driven by panel construction, channel choice, resampling length, or external-transform fragility",
        "",
        "## Diagnosis Read",
        "",
        f"- diagnosis rows: `{payload.metrics['diagnosis_row_count']}`",
        f"- warning rows: `{payload.metrics['warning_row_count']}`",
        f"- recovery rows: `{payload.metrics['recovery_row_count']}`",
        f"- diagnosis read: `{payload.metrics['diagnosis_read']}`",
        "",
        "This packet is diagnostic, not promotional.",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    decision_lines = [
        "# Decision Card",
        "",
        "- Candidate family: `generic_time_series_benchmark_classifiers`",
        "- Packet: `archive_backend_diagnosis_v1`",
        f"- Diagnosis read: `{payload.metrics['diagnosis_read']}`",
        "- Promotion rule: `do not treat external execution as sufficient when bounded panel variants still fail to recover archive-family performance or warning load remains high`",
    ]
    decision_card_path.write_text("\n".join(decision_lines) + "\n", encoding="utf-8")

    best_accuracy_path.write_bytes(_figure_to_png(_render_best_accuracy_panel(payload)))
    warning_path.write_bytes(_figure_to_png(_render_warning_panel(payload)))
    return ArchiveBackendDiagnosisArtifacts(
        run_dir=run_dir,
        diagnosis_path=diagnosis_path,
        summary_path=summary_path,
        metrics_path=metrics_path,
        report_path=report_path,
        decision_card_path=decision_card_path,
        plot_paths=(best_accuracy_path, warning_path),
    )


__all__ = [
    "ArchiveBackendDiagnosisArtifacts",
    "ArchiveBackendDiagnosisResult",
    "ArchiveDiagnosisRow",
    "ArchiveDiagnosisSummaryRow",
    "analyze_archive_backend_diagnosis",
    "write_archive_backend_diagnosis_artifacts",
]
