from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev

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


CLASS_NAMES = ("early_push_late_brake", "early_brake_late_push")
ARCHIVE_METHODS = (
    "minirocket_family",
    "drcif_interval_forests",
    "dictionary_tde_family",
    "hive_cote",
)
BASELINE_METHODS = ("windowed_feature_summary", "gradient_boosted_features")


@dataclass(frozen=True, slots=True)
class ArchiveFeatureHeadroomMethodRow:
    method_name: str
    family_group: str
    backend_name: str
    claim_level: str
    test_accuracy: float
    test_nll: float
    test_ece: float
    seed_stability_read: str
    delta_vs_windowed_accuracy: float
    delta_vs_boosted_accuracy: float


@dataclass(frozen=True, slots=True)
class ArchiveFeatureHeadroomSeedSweepRow:
    method_name: str
    family_group: str
    seed_count: int
    mean_test_accuracy: float
    std_test_accuracy: float
    mean_test_nll: float
    mean_test_ece: float
    backend_names: str
    stability_read: str


@dataclass(frozen=True, slots=True)
class ArchiveFeatureHeadroomWitnessResult:
    method_rows: tuple[ArchiveFeatureHeadroomMethodRow, ...]
    seed_sweep_rows: tuple[ArchiveFeatureHeadroomSeedSweepRow, ...]
    metrics: dict[str, float | int | str]


@dataclass(frozen=True, slots=True)
class ArchiveFeatureHeadroomWitnessArtifacts:
    run_dir: Path
    method_summary_path: Path
    seed_sweep_path: Path
    summary_path: Path
    metrics_path: Path
    report_path: Path
    decision_card_path: Path
    plot_paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class _SingleSeedMethodMetrics:
    method_name: str
    family_group: str
    backend_name: str
    claim_level: str
    test_accuracy: float
    test_nll: float
    test_ece: float


def _family_group(method_name: str) -> str:
    if method_name in ARCHIVE_METHODS:
        return "archive_family"
    if method_name == "gradient_boosted_features":
        return "engineered_boosted_baseline"
    return "interpretable_baseline"


def _binary_nll(rows: list[tuple[float, bool]]) -> float:
    return sum(-math.log(max(probability if correct else 1.0 - probability, 1.0e-12)) for probability, correct in rows) / max(len(rows), 1)


def _binary_ece(rows: list[tuple[float, bool]], *, bins: int = 10) -> float:
    total = max(len(rows), 1)
    ece = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        selected = [
            (probability, correct)
            for probability, correct in rows
            if lower <= probability < upper or (bin_index == bins - 1 and probability == upper)
        ]
        if not selected:
            continue
        mean_confidence = sum(probability for probability, _ in selected) / len(selected)
        mean_accuracy = sum(1.0 if correct else 0.0 for _, correct in selected) / len(selected)
        ece += abs(mean_confidence - mean_accuracy) * (len(selected) / total)
    return float(ece)


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


def _single_seed_metrics(*, seed: int, trajectories_per_class: int) -> tuple[_SingleSeedMethodMetrics, ...]:
    witness = analyze_feature_headroom_frontier(seed=seed, trajectories_per_class=trajectories_per_class)
    prediction_map = {row.trajectory_id: row for row in witness.prediction_rows}
    train_rows = tuple(_convert_feature_headroom_trajectory(row) for row in witness.trajectories if row.split == "train")
    test_rows = tuple(_convert_feature_headroom_trajectory(row) for row in witness.trajectories if row.split == "test")

    rows: list[_SingleSeedMethodMetrics] = []
    for method_name in BASELINE_METHODS:
        probability_rows: list[tuple[float, bool]] = []
        accuracy_count = 0
        for trajectory in test_rows:
            prediction = prediction_map[trajectory.trajectory_id]
            if method_name == "windowed_feature_summary":
                predicted_class = prediction.windowed_predicted_class
                confidence = float(prediction.windowed_confidence)
                claim_level = "baseline"
            else:
                predicted_class = prediction.boosted_predicted_class
                confidence = float(prediction.boosted_confidence)
                claim_level = "promoted_engineered_feature_baseline"
            correct = predicted_class == trajectory.true_class
            probability_rows.append((confidence, correct))
            accuracy_count += 1 if correct else 0
        rows.append(
            _SingleSeedMethodMetrics(
                method_name=method_name,
                family_group=_family_group(method_name),
                backend_name="baseline",
                claim_level=claim_level,
                test_accuracy=accuracy_count / max(len(test_rows), 1),
                test_nll=_binary_nll(probability_rows),
                test_ece=_binary_ece(probability_rows),
            )
        )

    for method_name in ARCHIVE_METHODS:
        outcome = fit_archive_classifier_with_outcome(method_name, train_rows, class_names=CLASS_NAMES)
        probability_rows = []
        accuracy_count = 0
        if outcome.adapter is not None:
            predictions = outcome.adapter.predict_many(test_rows)
            for (predicted_class, confidence), trajectory in zip(predictions, test_rows, strict=True):
                correct = predicted_class == trajectory.true_class
                probability_rows.append((float(confidence), correct))
                accuracy_count += 1 if correct else 0
        rows.append(
            _SingleSeedMethodMetrics(
                method_name=method_name,
                family_group=_family_group(method_name),
                backend_name=outcome.backend_name,
                claim_level="trained_external" if outcome.succeeded else "external_failed",
                test_accuracy=accuracy_count / max(len(test_rows), 1),
                test_nll=_binary_nll(probability_rows) if probability_rows else float("inf"),
                test_ece=_binary_ece(probability_rows) if probability_rows else 1.0,
            )
        )
    return tuple(rows)


def analyze_archive_feature_headroom_witness(
    *,
    seed: int = 811,
    trajectories_per_class: int = 12,
    seed_sweep: tuple[int, ...] | None = None,
) -> ArchiveFeatureHeadroomWitnessResult:
    resolved_seed_sweep = seed_sweep or (seed, seed + 1)
    per_seed = [_single_seed_metrics(seed=sweep_seed, trajectories_per_class=trajectories_per_class) for sweep_seed in resolved_seed_sweep]
    primary = {row.method_name: row for row in per_seed[0]}
    windowed_accuracy = primary["windowed_feature_summary"].test_accuracy
    boosted_accuracy = primary["gradient_boosted_features"].test_accuracy

    method_rows: list[ArchiveFeatureHeadroomMethodRow] = []
    seed_sweep_rows: list[ArchiveFeatureHeadroomSeedSweepRow] = []
    archive_gate_pass_count = 0
    archive_external_count = 0
    for method_name in BASELINE_METHODS + ARCHIVE_METHODS:
        per_method_rows = [next(row for row in seed_rows if row.method_name == method_name) for seed_rows in per_seed]
        primary_row = primary[method_name]
        stability_read = (
            "narrow_seed_sweep_pass"
            if pstdev([row.test_accuracy for row in per_method_rows]) <= 0.20 and mean(row.test_ece for row in per_method_rows) <= 0.35
            else "narrow_seed_sweep_flags_instability"
        )
        if primary_row.family_group == "archive_family":
            archive_external_count += 1 if primary_row.backend_name != "local_proxy" else 0
            archive_gate_pass_count += 1 if stability_read == "narrow_seed_sweep_pass" and mean(row.test_ece for row in per_method_rows) <= 0.35 else 0
        method_rows.append(
            ArchiveFeatureHeadroomMethodRow(
                method_name=method_name,
                family_group=primary_row.family_group,
                backend_name=primary_row.backend_name,
                claim_level=primary_row.claim_level,
                test_accuracy=primary_row.test_accuracy,
                test_nll=primary_row.test_nll,
                test_ece=primary_row.test_ece,
                seed_stability_read=stability_read,
                delta_vs_windowed_accuracy=primary_row.test_accuracy - windowed_accuracy,
                delta_vs_boosted_accuracy=primary_row.test_accuracy - boosted_accuracy,
            )
        )
        seed_sweep_rows.append(
            ArchiveFeatureHeadroomSeedSweepRow(
                method_name=method_name,
                family_group=primary_row.family_group,
                seed_count=len(per_method_rows),
                mean_test_accuracy=float(mean(row.test_accuracy for row in per_method_rows)),
                std_test_accuracy=float(pstdev([row.test_accuracy for row in per_method_rows])),
                mean_test_nll=float(mean(row.test_nll for row in per_method_rows)),
                mean_test_ece=float(mean(row.test_ece for row in per_method_rows)),
                backend_names=",".join(sorted({row.backend_name for row in per_method_rows})),
                stability_read=stability_read,
            )
        )

    archive_rows = [row for row in method_rows if row.family_group == "archive_family"]
    archive_champion = max(archive_rows, key=lambda row: row.test_accuracy)
    promotion_decision = (
        "promote_archive_feature_headroom_witness_for_followon_review"
        if archive_external_count == len(ARCHIVE_METHODS)
        and archive_gate_pass_count == len(ARCHIVE_METHODS)
        and archive_champion.test_accuracy > boosted_accuracy
        else "record_archive_feature_headroom_witness_keep_gate_closed"
    )
    metrics: dict[str, float | int | str] = {
        "study_id": "archive_feature_headroom_witness_v1",
        "seed": seed,
        "seed_values": ",".join(str(value) for value in resolved_seed_sweep),
        "archive_external_family_count": archive_external_count,
        "archive_gate_pass_count": archive_gate_pass_count,
        "windowed_accuracy": windowed_accuracy,
        "boosted_accuracy": boosted_accuracy,
        "archive_champion_method": archive_champion.method_name,
        "archive_champion_accuracy": archive_champion.test_accuracy,
        "archive_champion_delta_vs_windowed": archive_champion.test_accuracy - windowed_accuracy,
        "archive_champion_delta_vs_boosted": archive_champion.test_accuracy - boosted_accuracy,
        "promotion_decision": promotion_decision,
        "next_gate": (
            "inspect_repeated_archive_failure_on_feature_headroom"
            if promotion_decision == "record_archive_feature_headroom_witness_keep_gate_closed"
            else "broaden_archive_timing_witnesses"
        ),
    }
    return ArchiveFeatureHeadroomWitnessResult(
        method_rows=tuple(method_rows),
        seed_sweep_rows=tuple(seed_sweep_rows),
        metrics=metrics,
    )


def _render_accuracy_panel(result: ArchiveFeatureHeadroomWitnessResult):
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    labels = [row.method_name for row in result.method_rows]
    values = [row.test_accuracy for row in result.method_rows]
    colors = [
        "#16a34a" if row.family_group == "engineered_boosted_baseline" else "#9ca3af" if row.family_group == "interpretable_baseline" else "#7c3aed"
        for row in result.method_rows
    ]
    ax.bar(range(len(labels)), values, color=colors, width=0.65)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("test accuracy")
    ax.set_title("Archive vs Feature Headroom Witness", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _render_gap_panel(result: ArchiveFeatureHeadroomWitnessResult):
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    archive_rows = [row for row in result.method_rows if row.family_group == "archive_family"]
    labels = [row.method_name for row in archive_rows]
    x = list(range(len(labels)))
    width = 0.35
    ax.bar([value - width / 2 for value in x], [row.delta_vs_windowed_accuracy for row in archive_rows], width=width, label="vs windowed", color="#2563eb")
    ax.bar([value + width / 2 for value in x], [row.delta_vs_boosted_accuracy for row in archive_rows], width=width, label="vs boosted", color="#dc2626")
    ax.axhline(0.0, color="#111827", linewidth=1.0, alpha=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("accuracy delta")
    ax.set_title("Archive Gaps on Feature Headroom", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def write_archive_feature_headroom_witness_artifacts(
    output_dir: str | Path,
    *,
    result: ArchiveFeatureHeadroomWitnessResult | None = None,
    seed: int = 811,
    trajectories_per_class: int = 12,
    seed_sweep: tuple[int, ...] | None = None,
) -> ArchiveFeatureHeadroomWitnessArtifacts:
    payload = result or analyze_archive_feature_headroom_witness(
        seed=seed,
        trajectories_per_class=trajectories_per_class,
        seed_sweep=seed_sweep,
    )
    run_dir = Path(output_dir) / "archive_feature_headroom_witness_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    method_summary_path = run_dir / "method_summary.csv"
    seed_sweep_path = run_dir / "seed_sweep_summary.csv"
    summary_path = run_dir / "summary.csv"
    metrics_path = run_dir / "metrics.csv"
    report_path = run_dir / "archive_feature_headroom_witness_report.md"
    decision_card_path = run_dir / "decision_card.md"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    accuracy_plot_path = plots_dir / "archive_feature_headroom_accuracy.png"
    gaps_plot_path = plots_dir / "archive_feature_headroom_gaps.png"

    write_csv(method_summary_path, [asdict(row) for row in payload.method_rows], list(ArchiveFeatureHeadroomMethodRow.__dataclass_fields__.keys()))
    write_csv(seed_sweep_path, [asdict(row) for row in payload.seed_sweep_rows], list(ArchiveFeatureHeadroomSeedSweepRow.__dataclass_fields__.keys()))
    write_comparison_summary_csv(run_dir, [asdict(row) for row in payload.method_rows], filename="summary.csv")
    write_csv(metrics_path, [payload.metrics], list(payload.metrics.keys()))

    report_lines = [
        "# Archive Feature Headroom Witness",
        "",
        "- Study: `archive_feature_headroom_witness_v1`",
        "- Task: timing-order / feature-headroom binary witness",
        "- Archive methods: `minirocket_family`, `drcif_interval_forests`, `dictionary_tde_family`, `hive_cote`",
        "- Baselines: `windowed_feature_summary`, `gradient_boosted_features`",
        "",
        "## Claim Boundary",
        "",
        "This packet asks whether the archive family can recover timing-order structure on a dataset where a simple global window fails.",
        "It does not promote the archive family because the external methods merely execute.",
        "",
        f"- seed sweep: `{payload.metrics['seed_values']}`",
        f"- windowed accuracy: `{float(payload.metrics['windowed_accuracy']):.4f}`",
        f"- boosted accuracy: `{float(payload.metrics['boosted_accuracy']):.4f}`",
        f"- archive champion: `{payload.metrics['archive_champion_method']}` @ `{float(payload.metrics['archive_champion_accuracy']):.4f}`",
        f"- archive champion delta vs windowed: `{float(payload.metrics['archive_champion_delta_vs_windowed']):.4f}`",
        f"- archive champion delta vs boosted: `{float(payload.metrics['archive_champion_delta_vs_boosted']):.4f}`",
        f"- decision: `{payload.metrics['promotion_decision']}`",
        f"- next gate: `{payload.metrics['next_gate']}`",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    decision_lines = [
        "# Decision Card",
        "",
        "- Candidate family: `generic_time_series_benchmark_classifiers`",
        "- Witness: `archive_feature_headroom_witness_v1`",
        f"- Decision: `{payload.metrics['promotion_decision']}`",
        f"- Next gate: `{payload.metrics['next_gate']}`",
        "- Promotion rule: `do not promote the archive family when real external rows still lose to the engineered timing-order baseline on the feature-headroom witness`",
    ]
    decision_card_path.write_text("\n".join(decision_lines) + "\n", encoding="utf-8")

    accuracy_plot_path.write_bytes(_figure_to_png(_render_accuracy_panel(payload)))
    gaps_plot_path.write_bytes(_figure_to_png(_render_gap_panel(payload)))
    return ArchiveFeatureHeadroomWitnessArtifacts(
        run_dir=run_dir,
        method_summary_path=method_summary_path,
        seed_sweep_path=seed_sweep_path,
        summary_path=summary_path,
        metrics_path=metrics_path,
        report_path=report_path,
        decision_card_path=decision_card_path,
        plot_paths=(accuracy_plot_path, gaps_plot_path),
    )


__all__ = [
    "ArchiveFeatureHeadroomMethodRow",
    "ArchiveFeatureHeadroomSeedSweepRow",
    "ArchiveFeatureHeadroomWitnessArtifacts",
    "ArchiveFeatureHeadroomWitnessResult",
    "analyze_archive_feature_headroom_witness",
    "write_archive_feature_headroom_witness_artifacts",
]
