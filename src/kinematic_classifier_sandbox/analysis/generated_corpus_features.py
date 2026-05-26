from __future__ import annotations
from kinematic_classifier_sandbox.utils.io import write_csv

from ..runtime_paths import prepare_matplotlib
from dataclasses import dataclass
import csv
import json
import os
from pathlib import Path
from io import BytesIO
from typing import Any

from ..corpus.exploration.backend_adapter_proof import AdapterExecutionRecord, BackendCandidateSpec
from ..markdown_builder import MarkdownDocument
from ..validation.class_validity import analyze_class_validity
from ..contracts import TrajectoryArtifact
from .feature_analysis import FEATURE_REGISTRY, analyze_feature_datasets
from ..corpus.exploration.objective_corpus_gym_runner import execute_objective_candidates_via_corpus_gym
from ..trajectory_generator import (
    DatasetTierDefinition,
    GeneratedTrajectoryDataset,
    default_dataset_tiers,
    default_trajectory_class_definitions,
)

plt = prepare_matplotlib()



@dataclass(frozen=True, slots=True)
class GeneratedCorpusRecord:
    candidate: BackendCandidateSpec
    execution: AdapterExecutionRecord
    label_status: str
    assigned_class: str
    validity_score: float


@dataclass(frozen=True, slots=True)
class GeneratedCorpusFeaturesResult:
    feature_manifest: dict[str, Any]
    feature_rows: tuple[dict[str, Any], ...]
    excitation_rows: tuple[dict[str, Any], ...]
    record_rows: tuple[dict[str, Any], ...]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class GeneratedCorpusFeaturesArtifacts:
    run_dir: Path
    feature_matrix_path: Path
    feature_manifest_path: Path
    excitation_scores_path: Path
    record_manifest_path: Path
    report_path: Path
    excitation_heatmap_path: Path
    coverage_plot_path: Path
    gallery_plot_path: Path


def _status_lookup() -> dict[str, dict[str, Any]]:
    return {str(row["candidate_id"]): row for row in analyze_class_validity().score_rows}


def collect_generated_corpus_records() -> tuple[GeneratedCorpusRecord, ...]:
    status_rows = _status_lookup()
    records: list[GeneratedCorpusRecord] = []
    for executed_record in execute_objective_candidates_via_corpus_gym():
        candidate = executed_record.candidate
        status_row = status_rows.get(candidate.candidate_id)
        if status_row is None:
            continue
        records.append(
            GeneratedCorpusRecord(
                candidate=candidate,
                execution=executed_record.execution,
                label_status=str(status_row["label_status"]),
                assigned_class=str(status_row["assigned_class"]),
                validity_score=float(status_row["validity_score"]),
            )
        )
    return tuple(records)


def select_generated_corpus_records(
    *,
    include_statuses: tuple[str, ...] = ("valid_target_class", "ambiguous", "relabel_candidate"),
) -> tuple[GeneratedCorpusRecord, ...]:
    selected: list[GeneratedCorpusRecord] = []
    for record in collect_generated_corpus_records():
        if record.label_status not in include_statuses:
            continue
        if not record.execution.trajectory_run.success:
            continue
        selected.append(record)
    return tuple(selected)


def _tier_definition_for_name(name: str) -> DatasetTierDefinition:
    for definition in default_dataset_tiers():
        if definition.name == name:
            return definition
    return DatasetTierDefinition(
        name=name,
        description=f"Generated tier {name}",
        trajectories_per_class=1,
        steps_range=(6, 32),
        dt_range=(0.1, 1.0),
        measurement_std_range=(0.01, 0.25),
        outlier_probability=0.0,
        dropout_probability=0.0,
        irregular_sampling_strength=0.2,
        parameter_mode="generated",
    )


def _record_to_trajectory_artifact(record: GeneratedCorpusRecord) -> TrajectoryArtifact:
    run = record.execution.trajectory_run
    observations = tuple(float(value) for value in run.observations.get("position", ()))
    truth = run.truth_state
    return TrajectoryArtifact(
        trajectory_id=run.run_id,
        true_class=record.assigned_class,
        scenario_id=run.scenario_id,
        seed=run.seed,
        times=tuple(float(value) for value in run.times),
        measurements=observations,
        measurement_dim=1,
        measurement_axes=("position",),
        coordinate_frame="scalar_line",
        measurement_std=record.candidate.measurement_std,
        true_position=tuple(float(value) for value in truth.get("position", ())),
        true_velocity=tuple(float(value) for value in truth.get("velocity", ())),
        true_acceleration=tuple(float(value) for value in truth.get("acceleration", ())),
        state_dim=1,
        state_axes=("position",),
        truth_series={
            "position": tuple(float(value) for value in truth.get("position", ())),
            "velocity": tuple(float(value) for value in truth.get("velocity", ())),
            "acceleration": tuple(float(value) for value in truth.get("acceleration", ())),
        },
        generator_parameters={
            "backend_id": run.backend_id,
            "candidate_id": record.candidate.candidate_id,
            "difficulty_tier": record.candidate.difficulty_tier,
            "scenario_family": record.candidate.scenario_family,
            "target_class": record.candidate.target_class,
            "assigned_class": record.assigned_class,
            "label_status": record.label_status,
            "validity_score": record.validity_score,
            "environment_id": record.candidate.provenance.get("environment_id", ""),
        },
    )


def generated_corpus_datasets(
    records: tuple[GeneratedCorpusRecord, ...] | None = None,
) -> tuple[GeneratedTrajectoryDataset, ...]:
    selected_records = records or select_generated_corpus_records()
    tier_names = sorted({record.candidate.difficulty_tier for record in selected_records})
    class_definitions = default_trajectory_class_definitions()
    datasets: list[GeneratedTrajectoryDataset] = []
    for tier_name in tier_names:
        tier_records = [record for record in selected_records if record.candidate.difficulty_tier == tier_name]
        datasets.append(
            GeneratedTrajectoryDataset(
                tier=tier_name,
                seed=min(record.candidate.seed for record in tier_records),
                class_definitions=class_definitions,
                tier_definition=_tier_definition_for_name(tier_name),
                trajectories=tuple(_record_to_trajectory_artifact(record) for record in tier_records),
            )
        )
    return tuple(datasets)


def analyze_generated_corpus_features() -> GeneratedCorpusFeaturesResult:
    records = select_generated_corpus_records()
    datasets = generated_corpus_datasets(records)
    analysis = analyze_feature_datasets(datasets=datasets, feature_set="all_engineered")
    feature_names = analysis.summary.feature_names

    feature_rows = [row.as_flat_dict(feature_names) for row in analysis.feature_rows]
    record_rows = [
        {
            "trajectory_id": record.execution.trajectory_run.run_id,
            "candidate_id": record.candidate.candidate_id,
            "backend_id": record.execution.backend_id,
            "target_class": record.candidate.target_class,
            "assigned_class": record.assigned_class,
            "label_status": record.label_status,
            "validity_score": record.validity_score,
            "difficulty_tier": record.candidate.difficulty_tier,
            "scenario_family": record.candidate.scenario_family,
            "environment_id": record.candidate.provenance.get("environment_id", ""),
        }
        for record in records
    ]
    feature_manifest = {
        "version": "m39_v1",
        "feature_set": analysis.summary.feature_set_name,
        "feature_names": list(feature_names),
        "feature_groups": {name: FEATURE_REGISTRY[name].group for name in feature_names},
        "selected_trajectory_count": len(records),
        "selected_class_counts": analysis.summary.class_counts,
        "top_features": list(analysis.summary.top_features),
        "top_confusing_pairs": [list(pair) for pair in analysis.summary.top_confusing_pairs],
    }
    report = MarkdownDocument()
    report.heading("Generated Corpus Feature Integration", level=1)
    report.heading("Summary", level=2)
    report.bullet_list(
        [
            f"selected generated trajectories: `{len(records)}`",
            f"feature rows: `{len(feature_rows)}`",
            f"feature count: `{len(feature_names)}`",
            f"class counts: `{analysis.summary.class_counts}`",
        ]
    )
    report.heading("Notes", level=2)
    report.bullet_list(
        [
            "This milestone routes objective-driven selected trajectories through the real feature pipeline rather than relying on proxy score columns.",
            "Trajectory labels come from the class-validity layer, so relabeled or ambiguous rows are preserved with explicit status metadata.",
        ]
    )
    report_markdown = report.text()
    return GeneratedCorpusFeaturesResult(
        feature_manifest=feature_manifest,
        feature_rows=tuple(feature_rows),
        excitation_rows=analysis.excitation_rows,
        record_rows=tuple(record_rows),
        report_markdown=report_markdown,
    )


def _render_excitation_heatmap(rows: tuple[dict[str, Any], ...]) -> bytes:
    selected = list(rows[:12])
    if not selected:
        return b""
    feature_names = [name for name in selected[0] if name.endswith("_level")][:8]
    value_map = {"not_excited": 0, "weak": 1, "moderate": 2, "strong": 3}
    matrix = [[value_map[str(row[name])] for name in feature_names] for row in selected]
    labels = [str(row["trajectory_id"]).split("_", 1)[-1] for row in selected]
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    image = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0, vmax=3)
    ax.set_xticks(range(len(feature_names)), labels=[name.replace("_level", "") for name in feature_names], rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels)), labels=labels, fontsize=7)
    ax.set_title("Generated Corpus Feature Excitation")
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.04)
    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def _render_coverage_plot(rows: tuple[dict[str, Any], ...]) -> bytes:
    tiers = sorted({str(row["tier"]) for row in rows})
    classes = sorted({str(row["true_class"]) for row in rows})
    matrix = []
    for tier in tiers:
        matrix.append([sum(1 for row in rows if row["tier"] == tier and row["true_class"] == class_name) for class_name in classes])
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    image = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(classes)), labels=classes, fontsize=8)
    ax.set_yticks(range(len(tiers)), labels=tiers, fontsize=8)
    ax.set_title("Feature-Space Coverage By Tier and Class")
    for row_index, row_values in enumerate(matrix):
        for column_index, value in enumerate(row_values):
            ax.text(column_index, row_index, str(value), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.04)
    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def _render_gallery(rows: tuple[dict[str, Any], ...]) -> bytes:
    selected = list(rows[:9])
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.axis("off")
    ax.text(0.50, 0.96, "Selected Generated Corpus Records", ha="center", va="center", fontsize=11)
    y = 0.84
    for row in selected:
        text = (
            f"{row['trajectory_id']}\n"
            f"class={row['assigned_class']} status={row['label_status']} backend={row['backend_id']}\n"
            f"tier={row['difficulty_tier']} family={row['scenario_family']}"
        )
        ax.text(0.50, y, text, ha="center", va="center", fontsize=8, bbox={"boxstyle": "round,pad=0.3", "facecolor": "#eef4f7"})
        y -= 0.09
    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def write_generated_corpus_feature_artifacts(
    base_dir: str | Path,
    *,
    result: GeneratedCorpusFeaturesResult | None = None,
) -> GeneratedCorpusFeaturesArtifacts:
    run_dir = Path(base_dir) / "generated_corpus_features"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = result or analyze_generated_corpus_features()

    feature_matrix_path = run_dir / "feature_matrix.csv"
    feature_manifest_path = run_dir / "feature_manifest.json"
    excitation_scores_path = run_dir / "feature_excitation_scores.csv"
    record_manifest_path = run_dir / "selected_record_manifest.csv"
    report_path = run_dir / "feature_generation_report.md"
    excitation_heatmap_path = run_dir / "feature_excitation_heatmap.png"
    coverage_plot_path = run_dir / "feature_space_coverage.png"
    gallery_plot_path = run_dir / "selected_feature_gallery.png"

    feature_manifest_path.write_text(json.dumps(payload.feature_manifest, indent=2), encoding="utf-8")
    report_path.write_text(payload.report_markdown, encoding="utf-8")
    write_csv(feature_matrix_path, list(payload.feature_rows), list(payload.feature_rows[0].keys()) if payload.feature_rows else [])
    write_csv(excitation_scores_path, list(payload.excitation_rows), list(payload.excitation_rows[0].keys()) if payload.excitation_rows else [])
    write_csv(record_manifest_path, list(payload.record_rows), list(payload.record_rows[0].keys()) if payload.record_rows else [])
    excitation_heatmap_path.write_bytes(_render_excitation_heatmap(payload.excitation_rows))
    coverage_plot_path.write_bytes(_render_coverage_plot(payload.feature_rows))
    gallery_plot_path.write_bytes(_render_gallery(payload.record_rows))

    return GeneratedCorpusFeaturesArtifacts(
        run_dir=run_dir,
        feature_matrix_path=feature_matrix_path,
        feature_manifest_path=feature_manifest_path,
        excitation_scores_path=excitation_scores_path,
        record_manifest_path=record_manifest_path,
        report_path=report_path,
        excitation_heatmap_path=excitation_heatmap_path,
        coverage_plot_path=coverage_plot_path,
        gallery_plot_path=gallery_plot_path,
    )
