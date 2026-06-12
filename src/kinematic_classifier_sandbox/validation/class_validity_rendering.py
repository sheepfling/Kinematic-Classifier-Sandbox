from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kinematic_classifier_sandbox.reports.markdown import MarkdownDocument

from ..utils.io import write_csv
from ..utils.plotting import figure_to_png_bytes, plt
from .class_validity_contracts import ClassValidityArtifacts, ClassValidityResult


def render_class_validity_report(result: ClassValidityResult) -> str:
    report = MarkdownDocument()
    report.heading("Class Validity Scoring", level=1)
    report.heading("Summary", level=2)
    report.bullet_list(
        [
            f"scored candidates: `{len(result.score_rows)}`",
            f"valid target class: `{sum(1 for row in result.score_rows if row['label_status'] == 'valid_target_class')}`",
            f"ambiguous: `{sum(1 for row in result.score_rows if row['label_status'] == 'ambiguous')}`",
            f"invalid: `{sum(1 for row in result.score_rows if row['label_status'] == 'invalid')}`",
            f"relabel candidates: `{sum(1 for row in result.score_rows if row['label_status'] == 'relabel_candidate')}`",
        ]
    )
    report.heading("Notes", level=2)
    report.bullet_list(
        [
            "Generated trajectories no longer inherit their requested class blindly.",
            "Label status is derived from normalized telemetry-based class similarity scores rather than backend success alone.",
        ]
    )
    return report.text()


def _render_confusion_png(rows: tuple[dict[str, Any], ...]) -> bytes:
    classes = sorted(
        {str(row["target_class"]) for row in rows}
        | {str(row["assigned_class"]) for row in rows}
    )
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
    return figure_to_png_bytes(fig, dpi=180)


def _render_status_distribution_png(rows: tuple[dict[str, Any], ...]) -> bytes:
    statuses = ["valid_target_class", "ambiguous", "invalid", "relabel_candidate"]
    counts = [sum(1 for row in rows if row["label_status"] == status) for status in statuses]

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.bar(statuses, counts, color="#5c7ea5")
    ax.set_title("Class Validity Status Distribution")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return figure_to_png_bytes(fig, dpi=180)


def _render_similarity_png(rows: tuple[dict[str, Any], ...]) -> bytes:
    classes = (
        "constant_velocity_similarity",
        "constant_acceleration_similarity",
        "braking_similarity",
        "maneuver_similarity",
    )
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
    return figure_to_png_bytes(fig, dpi=180)


def write_class_validity_artifacts(
    base_dir: str | Path,
    *,
    result: ClassValidityResult | None = None,
) -> ClassValidityArtifacts:
    run_dir = Path(base_dir) / "class_validity"
    run_dir.mkdir(parents=True, exist_ok=True)
    if result is None:
        from .class_validity import analyze_class_validity

        payload = analyze_class_validity()
    else:
        payload = result

    class_definition_schema_path = run_dir / "class_definition_schema.json"
    class_validity_scores_path = run_dir / "class_validity_scores.csv"
    report_path = run_dir / "class_validity_report.md"
    confusion_png_path = run_dir / "class_validity_confusion.png"
    status_distribution_png_path = run_dir / "class_validity_status_distribution.png"
    alternate_similarity_png_path = run_dir / "alternate_class_similarity_heatmap.png"

    class_definition_schema_path.write_text(json.dumps(payload.class_definition_schema, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(payload.report_markdown, encoding="utf-8")
    fieldnames = list(payload.score_rows[0].keys()) if payload.score_rows else []
    write_csv(class_validity_scores_path, list(payload.score_rows), fieldnames)
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


__all__ = [
    "ClassValidityArtifacts",
    "render_class_validity_report",
    "write_class_validity_artifacts",
]
