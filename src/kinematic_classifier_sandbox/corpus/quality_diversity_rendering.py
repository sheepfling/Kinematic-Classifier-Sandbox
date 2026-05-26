from __future__ import annotations

import json
from pathlib import Path

import yaml

from ..utils.io import write_csv
from ..utils.plotting import _figure_to_png, plt
from .quality_diversity_types import QualityDiversityCorpusArtifacts, QualityDiversityCorpusResult


def _render_archive_coverage_heatmap(result: QualityDiversityCorpusResult):
    tiers = sorted({str(row["target_tier"]) for row in result.archive_cell_rows})
    classes = sorted({str(row["generated_class"]) for row in result.archive_cell_rows})
    matrix = []
    for class_name in classes:
        row_values = []
        for tier_name in tiers:
            count = sum(
                1
                for row in result.archive_cell_rows
                if row["generated_class"] == class_name and row["target_tier"] == tier_name
            )
            row_values.append(count)
        matrix.append(row_values)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    image = ax.imshow(matrix, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(tiers)), tiers, rotation=20, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_title("Archive Coverage Heatmap", loc="left", fontweight="bold")
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            ax.text(col_index, row_index, str(value), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def _render_elite_score_distribution(result: QualityDiversityCorpusResult):
    values = [float(row["total_utility"]) for row in result.archive_elite_rows]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.hist(values, bins=min(10, max(4, len(values))), color="#2563eb", edgecolor="#111827", alpha=0.8)
    ax.set_title("Elite Score Distribution", loc="left", fontweight="bold")
    ax.set_xlabel("total utility")
    ax.set_ylabel("count")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _render_feature_cell_examples(result: QualityDiversityCorpusResult):
    rows = [
        row
        for row in result.archive_elite_rows
        if row["target_type"] == "target_feature_cell"
    ] or list(result.archive_elite_rows[:4])
    labels = [str(row["candidate_id"]).removeprefix("target_feature_cell_high_accel_low_monotonicity_") for row in rows[:5]]
    excitation = [float(row["feature_excitation"]) for row in rows[:5]]
    monotonicity = [float(row["monotonicity"]) for row in rows[:5]]
    accel = [float(row["acceleration_range"]) for row in rows[:5]]
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    positions = range(len(labels))
    ax.bar(positions, excitation, width=0.28, label="feature_excitation", color="#16a34a")
    ax.bar([index + 0.28 for index in positions], accel, width=0.28, label="acceleration_range", color="#2563eb")
    ax.bar([index + 0.56 for index in positions], monotonicity, width=0.28, label="monotonicity", color="#f59e0b")
    ax.set_xticks([index + 0.28 for index in positions], labels, rotation=25, ha="right")
    ax.set_title("Feature Cell Examples", loc="left", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def write_quality_diversity_corpus_artifacts(
    output_dir: str | Path,
    *,
    result: QualityDiversityCorpusResult | None = None,
) -> QualityDiversityCorpusArtifacts:
    if result is None:
        from .quality_diversity import analyze_quality_diversity_corpus

        result = analyze_quality_diversity_corpus()
    qd = result
    run_dir = Path(output_dir) / "quality_diversity_corpus"
    plots_dir = run_dir / "plots"
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "qd_config.yaml"
    archive_cells_path = run_dir / "archive_cells.csv"
    archive_elites_path = run_dir / "archive_elites.csv"
    archive_coverage_path = run_dir / "archive_coverage.csv"
    manifest_path = run_dir / "qd_corpus_manifest.json"
    report_path = run_dir / "qd_corpus_report.md"
    archive_coverage_heatmap_path = plots_dir / "archive_coverage_heatmap.png"
    elite_score_distribution_path = plots_dir / "elite_score_distribution.png"
    feature_cell_examples_path = plots_dir / "feature_cell_examples.png"

    config_path.write_text(yaml.safe_dump(qd.config, sort_keys=False), encoding="utf-8")
    write_csv(archive_cells_path, list(qd.archive_cell_rows), list(qd.archive_cell_rows[0].keys()))
    write_csv(archive_elites_path, list(qd.archive_elite_rows), list(qd.archive_elite_rows[0].keys()))
    write_csv(archive_coverage_path, list(qd.archive_coverage_rows), list(qd.archive_coverage_rows[0].keys()))
    manifest_path.write_text(json.dumps(qd.corpus_manifest, indent=2), encoding="utf-8")
    report_path.write_text(qd.report_markdown, encoding="utf-8")
    archive_coverage_heatmap_path.write_bytes(_figure_to_png(_render_archive_coverage_heatmap(qd)))
    elite_score_distribution_path.write_bytes(_figure_to_png(_render_elite_score_distribution(qd)))
    feature_cell_examples_path.write_bytes(_figure_to_png(_render_feature_cell_examples(qd)))

    return QualityDiversityCorpusArtifacts(
        run_dir=run_dir,
        config_path=config_path,
        archive_cells_path=archive_cells_path,
        archive_elites_path=archive_elites_path,
        archive_coverage_path=archive_coverage_path,
        manifest_path=manifest_path,
        report_path=report_path,
        archive_coverage_heatmap_path=archive_coverage_heatmap_path,
        elite_score_distribution_path=elite_score_distribution_path,
        feature_cell_examples_path=feature_cell_examples_path,
    )
