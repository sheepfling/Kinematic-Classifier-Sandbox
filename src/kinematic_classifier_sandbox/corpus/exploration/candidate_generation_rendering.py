from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from ...utils.plotting import plt
from ...utils.io import write_csv
from .candidate_generation_types import CandidateGenerationArtifacts, CandidateGenerationResult, CandidateGenerationRow


def _render_sampler_comparison_png(rows: tuple[CandidateGenerationRow, ...]) -> bytes:
    sampler_names = sorted({str(row["sampler_name"]) for row in rows})
    metric_names = ("total_utility", "feature_excitation", "coverage_gain", "boundary_closeness")
    means = {
        sampler_name: {
            metric_name: sum(float(row[metric_name]) for row in rows if row["sampler_name"] == sampler_name)
            / max(sum(1 for row in rows if row["sampler_name"] == sampler_name), 1)
            for metric_name in metric_names
        }
        for sampler_name in sampler_names
    }
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    width = 0.18
    xs = list(range(len(sampler_names)))
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    colors = ("#2563eb", "#16a34a", "#f59e0b", "#dc2626")
    for metric_name, offset, color in zip(metric_names, offsets, colors):
        ax.bar([x + offset for x in xs], [means[name][metric_name] for name in sampler_names], width=width, label=metric_name, color=color)
    ax.set_xticks(xs, sampler_names, rotation=15, ha="right")
    ax.set_ylabel("mean score")
    ax.set_title("Sampler Comparison", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def _render_coverage_png(rows: tuple[CandidateGenerationRow, ...]) -> bytes:
    target_types = sorted({str(row["target_type"]) for row in rows})
    methods = sorted({str(row["search_method"]) for row in rows})
    matrix = []
    for target_type in target_types:
        row_values = []
        for method in methods:
            selected = [
                row
                for row in rows
                if row["target_type"] == target_type and row["search_method"] == method
            ]
            row_values.append(sum(float(row["selected"]) for row in selected))
        matrix.append(row_values)
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    image = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(methods)), methods, rotation=15, ha="right")
    ax.set_yticks(range(len(target_types)), target_types)
    ax.set_title("Target Coverage by Search Method", loc="left", fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def _render_lineage_png(rows: tuple[CandidateGenerationRow, ...]) -> bytes:
    lineage_rows = [row for row in rows if str(row["parent_candidate_id"])]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for index, row in enumerate(lineage_rows[:20]):
        ax.plot([0, 1], [index, index], color="#94a3b8", linewidth=1.0)
        ax.text(0.0, index, str(row["parent_candidate_id"]), fontsize=7, ha="right", va="center")
        ax.text(1.0, index, str(row["candidate_id"]), fontsize=7, ha="left", va="center")
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-1, max(len(lineage_rows[:20]), 1))
    ax.axis("off")
    ax.set_title("Mutation Lineage", loc="left", fontweight="bold")
    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def write_candidate_generation_artifacts(
    output_dir: str | Path,
    *,
    result: CandidateGenerationResult | None = None,
) -> CandidateGenerationArtifacts:
    if result is None:
        from .candidate_generation import analyze_candidate_generation

        result = analyze_candidate_generation()
    payload = result
    run_dir = Path(output_dir) / "candidate_generation"
    plots_dir = run_dir / "plots"
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    sampler_manifest_path = run_dir / "sampler_manifest.json"
    generated_candidates_path = run_dir / "generated_candidates.csv"
    report_path = run_dir / "candidate_generation_report.md"
    sampler_comparison_png_path = plots_dir / "sampler_comparison.png"
    candidate_coverage_png_path = plots_dir / "candidate_coverage.png"
    mutation_lineage_png_path = plots_dir / "mutation_lineage.png"
    sampler_manifest_path.write_text(json.dumps(payload.sampler_manifest, indent=2), encoding="utf-8")
    write_csv(generated_candidates_path, list(payload.generated_candidate_rows), list(payload.generated_candidate_rows[0].keys()))
    report_path.write_text(payload.report_markdown, encoding="utf-8")
    sampler_comparison_png_path.write_bytes(_render_sampler_comparison_png(payload.generated_candidate_rows))
    candidate_coverage_png_path.write_bytes(_render_coverage_png(payload.generated_candidate_rows))
    mutation_lineage_png_path.write_bytes(_render_lineage_png(payload.generated_candidate_rows))
    return CandidateGenerationArtifacts(
        run_dir=run_dir,
        sampler_manifest_path=sampler_manifest_path,
        generated_candidates_path=generated_candidates_path,
        report_path=report_path,
        sampler_comparison_png_path=sampler_comparison_png_path,
        candidate_coverage_png_path=candidate_coverage_png_path,
        mutation_lineage_png_path=mutation_lineage_png_path,
    )
