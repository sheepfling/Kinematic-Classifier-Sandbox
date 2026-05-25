from __future__ import annotations

from dataclasses import dataclass
import csv
import io
import json
import os
from pathlib import Path
import random

import yaml

from .corpus_gym import CorpusGymAction, CorpusGymEnvironment, default_corpus_gym_targets
from .corpus_search_baseline import analyze_corpus_search_baseline


def _prepare_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp/kinematic-classifier-sandbox-mpl")))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _figure_to_png(fig) -> bytes:
    plt = _prepare_matplotlib()
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _bucket(value: float, thresholds: tuple[float, float]) -> str:
    if value < thresholds[0]:
        return "low"
    if value < thresholds[1]:
        return "medium"
    return "high"


@dataclass(frozen=True, slots=True)
class QualityDiversityCorpusResult:
    config: dict[str, object]
    archive_cell_rows: tuple[dict[str, object], ...]
    archive_elite_rows: tuple[dict[str, object], ...]
    archive_coverage_rows: tuple[dict[str, object], ...]
    corpus_manifest: dict[str, object]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class QualityDiversityCorpusArtifacts:
    run_dir: Path
    config_path: Path
    archive_cells_path: Path
    archive_elites_path: Path
    archive_coverage_path: Path
    manifest_path: Path
    report_path: Path
    archive_coverage_heatmap_path: Path
    elite_score_distribution_path: Path
    feature_cell_examples_path: Path


def _random_action_for_target(rng: random.Random, target_tier: str, seed: int) -> CorpusGymAction:
    return CorpusGymAction(
        seed=seed,
        tier_name=target_tier,
        duration_scale=rng.uniform(0.75, 1.30),
        measurement_scale=rng.uniform(0.80, 1.35),
        irregularity_scale=rng.uniform(0.75, 1.40),
        outlier_scale=rng.uniform(0.75, 1.40),
        step_scale=rng.uniform(0.80, 1.25),
    )


def _archive_cell_id(row: dict[str, object]) -> tuple[str, str, str, str, str]:
    return (
        str(row["generated_class"]),
        str(row["target_tier"]),
        _bucket(float(row["duration"]), (6.0, 12.0)),
        _bucket(float(row["acceleration_range"]), (0.35, 0.85)),
        _bucket(1.0 - float(row["monotonicity"]), (0.08, 0.22)),
    )


def _episode_row(
    *,
    iteration: int,
    candidate_id: str,
    target_id: str,
    target_type: str,
    target_tier: str,
    episode,
) -> dict[str, object]:
    reward = episode.reward
    diagnostics = episode.diagnostics
    return {
        "iteration": iteration,
        "candidate_id": candidate_id,
        "target_id": target_id,
        "target_type": target_type,
        "target_tier": target_tier,
        "trajectory_id": episode.trajectory.trajectory_id,
        "generated_class": episode.trajectory.true_class,
        "seed": episode.action.seed,
        "duration_scale": episode.action.duration_scale,
        "measurement_scale": episode.action.measurement_scale,
        "irregularity_scale": episode.action.irregularity_scale,
        "outlier_scale": episode.action.outlier_scale,
        "step_scale": episode.action.step_scale,
        "class_validity": reward.class_validity,
        "feature_excitation": reward.feature_excitation,
        "coverage_gain": reward.coverage_gain,
        "boundary_closeness": reward.boundary_closeness,
        "classifier_stress": reward.classifier_stress,
        "prior_sensitivity": reward.prior_sensitivity,
        "leakage_penalty": reward.leakage_penalty,
        "physical_invalidity_penalty": reward.physical_invalidity_penalty,
        "total_utility": reward.total_utility,
        "duration": diagnostics["duration"],
        "acceleration_range": diagnostics["acceleration_range"],
        "monotonicity": diagnostics["monotonicity"],
        "sampling_irregularity": diagnostics["sampling_irregularity"],
        "num_samples": diagnostics["num_samples"],
    }


def analyze_quality_diversity_corpus(
    *,
    seed: int = 7,
    iterations: int = 42,
) -> QualityDiversityCorpusResult:
    rng = random.Random(seed)
    environment = CorpusGymEnvironment()
    targets = default_corpus_gym_targets()
    baseline = analyze_corpus_search_baseline(seed=seed)
    feature_target_random_rows = [
        row
        for row in baseline.candidate_score_rows
        if row["search_method"] == "random" and row["target_type"] == "target_feature_cell"
    ]
    random_feature_excitation_mean = sum(float(row["feature_excitation"]) for row in feature_target_random_rows) / max(
        len(feature_target_random_rows),
        1,
    )

    archive: dict[tuple[str, str, str, str, str], dict[str, object]] = {}
    coverage_rows: list[dict[str, object]] = []

    for iteration in range(iterations):
        target = targets[iteration % len(targets)]
        environment.reset(target)
        action = _random_action_for_target(
            rng,
            target.target_tier or "realistic_v1",
            seed=seed * 100_000 + iteration,
        )
        episode = environment.simulate(action)
        row = _episode_row(
            iteration=iteration,
            candidate_id=f"{target.target_id}_qd_{iteration}",
            target_id=target.target_id,
            target_type=target.target_type,
            target_tier=target.target_tier or "realistic_v1",
            episode=episode,
        )
        cell_id = _archive_cell_id(row)
        incumbent = archive.get(cell_id)
        if incumbent is None or float(row["total_utility"]) > float(incumbent["total_utility"]):
            archive[cell_id] = row
        elite_values = list(archive.values())
        coverage_rows.append(
            {
                "iteration": iteration,
                "num_cells_filled": len(archive),
                "coverage_fraction": len(archive) / 81.0,
                "mean_elite_utility": sum(float(value["total_utility"]) for value in elite_values) / max(len(elite_values), 1),
                "mean_elite_feature_excitation": sum(float(value["feature_excitation"]) for value in elite_values) / max(len(elite_values), 1),
            }
        )

    archive_elite_rows = sorted(archive.values(), key=lambda row: float(row["total_utility"]), reverse=True)
    archive_cell_rows = []
    for cell_id, elite_row in sorted(archive.items()):
        archive_cell_rows.append(
            {
                "cell_id": "|".join(cell_id),
                "generated_class": cell_id[0],
                "target_tier": cell_id[1],
                "duration_bucket": cell_id[2],
                "acceleration_bucket": cell_id[3],
                "direction_change_bucket": cell_id[4],
                "elite_candidate_id": elite_row["candidate_id"],
                "elite_total_utility": elite_row["total_utility"],
            }
        )

    feature_target_elites = [row for row in archive_elite_rows if row["target_type"] == "target_feature_cell"]
    elite_feature_excitation_mean = sum(float(row["feature_excitation"]) for row in feature_target_elites) / max(
        len(feature_target_elites),
        1,
    )
    best_feature_target_excitation = max(
        (float(row["feature_excitation"]) for row in feature_target_elites),
        default=0.0,
    )
    num_feature_elites_above_random_mean = sum(
        1 for row in feature_target_elites if float(row["feature_excitation"]) > random_feature_excitation_mean
    )
    corpus_manifest = {
        "archive_id": "quality_diversity_corpus_v1",
        "seed": seed,
        "iterations": iterations,
        "num_archive_cells": len(archive_cell_rows),
        "num_archive_elites": len(archive_elite_rows),
        "feature_target_elite_excitation_mean": elite_feature_excitation_mean,
        "best_feature_target_excitation": best_feature_target_excitation,
        "random_feature_excitation_mean": random_feature_excitation_mean,
        "num_feature_elites_above_random_mean": num_feature_elites_above_random_mean,
        "improves_feature_excitation_over_baseline": best_feature_target_excitation > random_feature_excitation_mean,
    }
    report_markdown = "\n".join(
        [
            "# Quality-Diversity Corpus",
            "",
            "This artifact builds the first archive-style corpus layer on top of CorpusGym.",
            "",
            "## Summary",
            "",
            f"- Iterations: `{iterations}`",
            f"- Archive cells filled: `{len(archive_cell_rows)}`",
            f"- Archive elites retained: `{len(archive_elite_rows)}`",
            f"- Final coverage fraction: `{coverage_rows[-1]['coverage_fraction']:.3f}`",
            f"- Mean elite utility: `{coverage_rows[-1]['mean_elite_utility']:.3f}`",
            f"- Feature-target elite excitation mean: `{elite_feature_excitation_mean:.3f}`",
            f"- Best feature-target excitation: `{best_feature_target_excitation:.3f}`",
            f"- Random feature-target excitation mean: `{random_feature_excitation_mean:.3f}`",
            f"- Feature-target elites above random mean: `{num_feature_elites_above_random_mean}`",
            f"- Improves feature excitation over baseline: `{best_feature_target_excitation > random_feature_excitation_mean}`",
            "",
            "## Archive Policy",
            "",
            "- One elite is retained per behavior cell.",
            "- Cells are keyed by generated class, target tier, duration bucket, acceleration-range bucket, and direction-change bucket.",
            "- Elite replacement is based on total utility.",
            "",
            "## Reading Notes",
            "",
            "- This M27 slice is MAP-Elites-like but intentionally minimal: it proves archive coverage and elite retention before novelty search or crossover logic.",
            "- The archive is still driven by one-trajectory episodes from CorpusGym rather than multi-trajectory corpus optimization.",
            "- Boundary and adversarial tiers are explicitly retained as separate behavior-cell axes rather than collapsed into one stress bucket.",
        ]
    )
    config = {
        "archive_id": "quality_diversity_corpus_v1",
        "seed": seed,
        "iterations": iterations,
        "cell_axes": [
            "generated_class",
            "target_tier",
            "duration_bucket",
            "acceleration_bucket",
            "direction_change_bucket",
        ],
        "elite_selection_metric": "total_utility",
    }
    return QualityDiversityCorpusResult(
        config=config,
        archive_cell_rows=tuple(archive_cell_rows),
        archive_elite_rows=tuple(archive_elite_rows),
        archive_coverage_rows=tuple(coverage_rows),
        corpus_manifest=corpus_manifest,
        report_markdown=report_markdown,
    )


def _render_archive_coverage_heatmap(result: QualityDiversityCorpusResult):
    plt = _prepare_matplotlib()
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
    plt = _prepare_matplotlib()
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
    plt = _prepare_matplotlib()
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
    qd = result or analyze_quality_diversity_corpus()
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
    _write_csv(archive_cells_path, list(qd.archive_cell_rows), list(qd.archive_cell_rows[0].keys()))
    _write_csv(archive_elites_path, list(qd.archive_elite_rows), list(qd.archive_elite_rows[0].keys()))
    _write_csv(archive_coverage_path, list(qd.archive_coverage_rows), list(qd.archive_coverage_rows[0].keys()))
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
