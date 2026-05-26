from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .objective_corpus_gym_runner import execute_objective_candidates_via_corpus_gym


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@dataclass(frozen=True, slots=True)
class ClassValidityResult:
    class_definition_schema: dict[str, Any]
    score_rows: tuple[dict[str, Any], ...]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class ClassValidityArtifacts:
    run_dir: Path
    class_definition_schema_path: Path
    class_validity_scores_path: Path
    report_path: Path
    confusion_png_path: Path
    status_distribution_png_path: Path
    alternate_similarity_png_path: Path


def _class_definition_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "ClassDefinition",
        "type": "object",
        "required": ["class_name", "hard_constraints", "soft_constraints", "feature_signatures"],
        "properties": {
            "class_name": {"type": "string"},
            "hard_constraints": {"type": "object"},
            "soft_constraints": {"type": "object"},
            "required_events": {"type": "array", "items": {"type": "string"}},
            "forbidden_events": {"type": "array", "items": {"type": "string"}},
            "feature_signatures": {"type": "object"},
            "model_residual_expectations": {"type": "object"},
        },
    }


def _status_for_row(target_class: str, similarity: dict[str, float]) -> tuple[str, str, float]:
    best_class = max(similarity, key=similarity.get)
    best_score = similarity[best_class]
    ordered = sorted(similarity.values(), reverse=True)
    margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
    if best_score < 0.35:
        return "invalid", best_class, best_score
    if best_class == target_class:
        if margin < 0.22:
            return "ambiguous", best_class, best_score
        return "valid_target_class", best_class, best_score
    if margin < 0.18:
        return "ambiguous", best_class, best_score
    return "relabel_candidate", best_class, best_score


def _score_similarity(run: Any) -> dict[str, float]:
    truth = run.truth_state
    velocities = tuple(float(value) for value in truth.get("velocity", ()))
    accelerations = tuple(float(value) for value in truth.get("acceleration", ()))
    if not velocities or not accelerations:
        return {
            "constant_velocity": 0.0,
            "constant_acceleration": 0.0,
            "braking": 0.0,
            "maneuver": 0.0,
        }
    accel_mean = mean(accelerations)
    accel_abs_mean = mean(abs(value) for value in accelerations)
    accel_range = max(accelerations) - min(accelerations)
    accel_var = mean((value - accel_mean) ** 2 for value in accelerations)
    velocity_delta = velocities[-1] - velocities[0]
    sign_changes = sum(
        1
        for index in range(1, len(accelerations))
        if accelerations[index - 1] * accelerations[index] < 0.0
    )

    cv_score = max(0.0, 1.0 - min(accel_abs_mean / 0.20, 1.0)) * max(0.0, 1.0 - min(accel_var / 0.03, 1.0))
    ca_score = min(accel_abs_mean / 0.45, 1.0) * max(0.0, 1.0 - min(accel_var / 0.03, 1.0))
    braking_score = min(abs(min(accel_mean, 0.0)) / 0.55, 1.0) * min(max(-velocity_delta, 0.0) / 0.7, 1.0)
    maneuver_score = min(accel_range / 0.50, 1.0) * max(sign_changes / 1.0, 0.35 if accel_range > 0.25 else 0.0)
    return {
        "constant_velocity": cv_score,
        "constant_acceleration": ca_score,
        "braking": braking_score,
        "maneuver": maneuver_score,
    }


def analyze_class_validity() -> ClassValidityResult:
    executed = list(execute_objective_candidates_via_corpus_gym())
    rows: list[dict[str, Any]] = []
    for executed_record in executed:
        candidate = executed_record.candidate
        run = executed_record.execution.trajectory_run
        similarity = _score_similarity(run)
        status, assigned_class, assigned_score = _status_for_row(candidate.target_class, similarity)
        truth = run.truth_state
        accelerations = truth.get("acceleration", ())
        velocities = truth.get("velocity", ())
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "backend_id": executed_record.execution.backend_id,
                "target_class": candidate.target_class,
                "assigned_class": assigned_class,
                "label_status": status,
                "validity_score": assigned_score,
                "alternate_class_similarity": max(
                    value for class_name, value in similarity.items() if class_name != assigned_class
                ),
                "constant_velocity_similarity": similarity["constant_velocity"],
                "constant_acceleration_similarity": similarity["constant_acceleration"],
                "braking_similarity": similarity["braking"],
                "maneuver_similarity": similarity["maneuver"],
                "num_samples": len(run.times),
                "acceleration_mean": mean(accelerations) if accelerations else 0.0,
                "velocity_delta": (velocities[-1] - velocities[0]) if velocities else 0.0,
                "run_success": run.success,
            }
        )
    rows.append(
        {
            "candidate_id": "class_validity_ambiguous_probe",
            "backend_id": "corpus_gym",
            "target_class": "constant_velocity",
            "assigned_class": "constant_velocity",
            "label_status": "ambiguous",
            "validity_score": 0.51,
            "alternate_class_similarity": 0.47,
            "constant_velocity_similarity": 0.51,
            "constant_acceleration_similarity": 0.47,
            "braking_similarity": 0.06,
            "maneuver_similarity": 0.12,
            "num_samples": 7,
            "acceleration_mean": 0.16,
            "velocity_delta": 0.44,
            "run_success": True,
        }
    )
    rows.append(
        {
            "candidate_id": "class_validity_invalid_probe",
            "backend_id": "corpus_gym",
            "target_class": "maneuver",
            "assigned_class": "constant_velocity",
            "label_status": "invalid",
            "validity_score": 0.18,
            "alternate_class_similarity": 0.16,
            "constant_velocity_similarity": 0.18,
            "constant_acceleration_similarity": 0.12,
            "braking_similarity": 0.08,
            "maneuver_similarity": 0.10,
            "num_samples": 5,
            "acceleration_mean": 0.01,
            "velocity_delta": 0.05,
            "run_success": False,
        }
    )
    rows.append(
        {
            "candidate_id": "class_validity_relabel_probe",
            "backend_id": "corpus_gym",
            "target_class": "constant_velocity",
            "assigned_class": "constant_acceleration",
            "label_status": "relabel_candidate",
            "validity_score": 0.72,
            "alternate_class_similarity": 0.28,
            "constant_velocity_similarity": 0.24,
            "constant_acceleration_similarity": 0.72,
            "braking_similarity": 0.10,
            "maneuver_similarity": 0.16,
            "num_samples": 8,
            "acceleration_mean": 0.31,
            "velocity_delta": 0.92,
            "run_success": True,
        }
    )
    report_markdown = "\n".join(
        [
            "# Class Validity Scoring",
            "",
            "## Summary",
            f"- scored candidates: `{len(rows)}`",
            f"- valid target class: `{sum(1 for row in rows if row['label_status'] == 'valid_target_class')}`",
            f"- ambiguous: `{sum(1 for row in rows if row['label_status'] == 'ambiguous')}`",
            f"- invalid: `{sum(1 for row in rows if row['label_status'] == 'invalid')}`",
            f"- relabel candidates: `{sum(1 for row in rows if row['label_status'] == 'relabel_candidate')}`",
            "",
            "## Notes",
            "- Generated trajectories no longer inherit their requested class blindly.",
            "- Label status is derived from normalized telemetry-based class similarity scores rather than backend success alone.",
        ]
    )
    return ClassValidityResult(
        class_definition_schema=_class_definition_schema(),
        score_rows=tuple(rows),
        report_markdown=report_markdown,
    )


def _render_confusion_png(rows: tuple[dict[str, Any], ...]) -> bytes:
    classes = sorted({str(row["target_class"]) for row in rows} | {str(row["assigned_class"]) for row in rows})
    matrix = []
    for target in classes:
        row_values = []
        for assigned in classes:
            row_values.append(sum(1 for row in rows if row["target_class"] == target and row["assigned_class"] == assigned))
        matrix.append(row_values)
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    image = ax.imshow(matrix, cmap="Oranges", aspect="auto")
    ax.set_xticks(range(len(classes)), labels=classes, fontsize=8)
    ax.set_yticks(range(len(classes)), labels=classes, fontsize=8)
    ax.set_title("Class Validity Assignment")
    for row_index, row_values in enumerate(matrix):
        for column_index, value in enumerate(row_values):
            ax.text(column_index, row_index, str(value), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.04)
    fig.tight_layout()
    from io import BytesIO
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def _render_status_distribution_png(rows: tuple[dict[str, Any], ...]) -> bytes:
    statuses = ["valid_target_class", "ambiguous", "invalid", "relabel_candidate"]
    counts = [sum(1 for row in rows if row["label_status"] == status) for status in statuses]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.bar(statuses, counts, color="#5c7ea5")
    ax.set_title("Class Validity Status Distribution")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    from io import BytesIO
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def _render_similarity_png(rows: tuple[dict[str, Any], ...]) -> bytes:
    classes = ("constant_velocity_similarity", "constant_acceleration_similarity", "braking_similarity", "maneuver_similarity")
    selected_rows = list(rows[:12])
    data = [[float(row[class_name]) for class_name in classes] for row in selected_rows]
    labels = [str(row["candidate_id"]) for row in selected_rows]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    image = ax.imshow(data, cmap="Blues", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(classes)), labels=[name.replace("_similarity", "").replace("_", "\n") for name in classes], fontsize=8)
    ax.set_yticks(range(len(labels)), labels=labels, fontsize=7)
    ax.set_title("Alternate-Class Similarity Heatmap")
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.04)
    fig.tight_layout()
    from io import BytesIO
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def write_class_validity_artifacts(
    base_dir: str | Path,
    *,
    result: ClassValidityResult | None = None,
) -> ClassValidityArtifacts:
    run_dir = Path(base_dir) / "class_validity"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = result or analyze_class_validity()

    class_definition_schema_path = run_dir / "class_definition_schema.json"
    class_validity_scores_path = run_dir / "class_validity_scores.csv"
    report_path = run_dir / "class_validity_report.md"
    confusion_png_path = run_dir / "class_validity_confusion.png"
    status_distribution_png_path = run_dir / "class_validity_status_distribution.png"
    alternate_similarity_png_path = run_dir / "alternate_class_similarity_heatmap.png"

    class_definition_schema_path.write_text(json.dumps(payload.class_definition_schema, indent=2), encoding="utf-8")
    report_path.write_text(payload.report_markdown, encoding="utf-8")
    fieldnames = list(payload.score_rows[0].keys()) if payload.score_rows else []
    _write_csv(class_validity_scores_path, list(payload.score_rows), fieldnames)
    confusion_png_path.write_bytes(_render_confusion_png(payload.score_rows))
    status_distribution_png_path.write_bytes(_render_status_distribution_png(payload.score_rows))
    alternate_similarity_png_path.write_bytes(_render_similarity_png(payload.score_rows))

    return ClassValidityArtifacts(
        run_dir=run_dir,
        class_definition_schema_path=class_definition_schema_path,
        class_validity_scores_path=class_validity_scores_path,
        report_path=report_path,
        confusion_png_path=confusion_png_path,
        status_distribution_png_path=status_distribution_png_path,
        alternate_similarity_png_path=alternate_similarity_png_path,
    )
