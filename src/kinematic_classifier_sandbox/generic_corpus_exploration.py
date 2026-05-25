from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
import random
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .backend_adapter_proof import (
    BackendCandidateSpec,
    _adapter_map,
)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass(frozen=True, slots=True)
class GenericCorpusExplorationResult:
    exploration_manifest: dict[str, Any]
    candidate_score_rows: tuple[dict[str, Any], ...]
    archive_cell_rows: tuple[dict[str, Any], ...]
    selected_corpus_manifest: dict[str, Any]
    backend_comparison_rows: tuple[dict[str, Any], ...]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class GenericCorpusExplorationArtifacts:
    run_dir: Path
    exploration_manifest_path: Path
    candidate_scores_path: Path
    archive_cells_path: Path
    selected_corpus_manifest_path: Path
    backend_comparison_path: Path
    report_path: Path
    numeric_walkthrough_path: Path
    backend_coverage_png_path: Path
    archive_heatmap_png_path: Path
    score_parallel_png_path: Path
    selected_gallery_png_path: Path
    provenance_dashboard_png_path: Path


def _candidate_pool() -> tuple[BackendCandidateSpec, ...]:
    candidates: list[BackendCandidateSpec] = []
    for replicate in range(3):
        candidates.append(
            BackendCandidateSpec(
                candidate_id=f"boundary_param_{replicate}",
                scenario_id="shared_boundary_cv_ca",
                scenario_family="shared_boundary_case",
                target_class="constant_velocity",
                difficulty_tier="boundary_v1",
                seed=700 + replicate,
                duration=2.0 + 0.15 * replicate,
                sample_period=0.5,
                initial_position=0.0,
                initial_velocity=1.00 + 0.03 * replicate,
                acceleration=0.08 + 0.02 * replicate,
                measurement_std=0.03,
                provenance={"search_method": "dashboard_seeded", "search_iteration": len(candidates)},
            )
        )
        candidates.append(
            BackendCandidateSpec(
                candidate_id=f"switching_ctrl_{replicate}",
                scenario_id="switching_velocity_to_braking",
                scenario_family="switching_case",
                target_class="braking",
                difficulty_tier="stress_v1",
                seed=720 + replicate,
                duration=2.0,
                sample_period=0.5,
                initial_position=0.0,
                initial_velocity=1.35 + 0.04 * replicate,
                acceleration=0.0,
                measurement_std=0.04,
                switch_time=0.9 + 0.1 * replicate,
                acceleration_after_switch=-0.65 - 0.05 * replicate,
                provenance={"search_method": "dashboard_seeded", "search_iteration": len(candidates)},
            )
        )
        candidates.append(
            BackendCandidateSpec(
                candidate_id=f"env_accel_{replicate}",
                scenario_id="environment_density_gradient",
                scenario_family="environment_regime_case",
                target_class="constant_acceleration",
                difficulty_tier="realistic_v1",
                seed=740 + replicate,
                duration=2.0,
                sample_period=0.5,
                initial_position=0.0,
                initial_velocity=0.82 + 0.03 * replicate,
                acceleration=0.40 + 0.04 * replicate,
                measurement_std=0.03,
                drag_coefficient=0.18 + 0.04 * replicate,
                density_scale=1.02 - 0.08 * replicate,
                wind_bias=0.03 + 0.04 * replicate,
                provenance={"search_method": "dashboard_seeded", "search_iteration": len(candidates), "environment_id": f"env_regime_{replicate}"},
            )
        )
        candidates.append(
            BackendCandidateSpec(
                candidate_id=f"file_maneuver_{replicate}",
                scenario_id="file_backend_case",
                scenario_family="file_backend_case",
                target_class="maneuver",
                difficulty_tier="adversarial_v1",
                seed=760 + replicate,
                duration=2.0,
                sample_period=0.5,
                initial_position=0.0,
                initial_velocity=0.85 + 0.05 * replicate,
                acceleration=0.24 + 0.04 * replicate,
                measurement_std=0.03,
                input_deck_hash=f"file_case_hash_{replicate}",
                longitudinal_command=(0.4, 0.5, 0.1, -0.3, -0.4),
                provenance={"search_method": "dashboard_seeded", "search_iteration": len(candidates)},
            )
        )
    return tuple(candidates)


def _backends_for_candidate(candidate: BackendCandidateSpec) -> tuple[str, ...]:
    if candidate.scenario_family == "shared_boundary_case":
        return ("parameter_only_1d", "environment_aware_1d", "mock_file_backend_1d")
    if candidate.scenario_family == "switching_case":
        return ("controlled_1d", "mock_file_backend_1d")
    if candidate.scenario_family == "environment_regime_case":
        return ("environment_aware_1d",)
    if candidate.scenario_family == "file_backend_case":
        return ("mock_file_backend_1d",)
    return ()


def _provenance_completeness(metadata: dict[str, Any]) -> float:
    required = ("adapter_family", "candidate_id", "search_provenance")
    present = sum(1 for key in required if key in metadata and metadata[key] not in ("", None, {}))
    return present / len(required)


def _score_run(candidate: BackendCandidateSpec, backend_id: str, run: Any) -> dict[str, Any]:
    success = bool(run.success)
    provenance_score = _provenance_completeness(run.metadata)
    truth_state = run.truth_state
    positions = truth_state.get("position", ())
    velocities = truth_state.get("velocity", ())
    accelerations = truth_state.get("acceleration", ())
    speed_range = max(velocities) - min(velocities) if velocities else 0.0
    position_range = max(positions) - min(positions) if positions else 0.0
    acceleration_range = max(accelerations) - min(accelerations) if accelerations else 0.0
    validity_score = 1.0 if success else 0.0
    boundary_score = 0.85 if candidate.scenario_family == "shared_boundary_case" else 0.35
    stress_score = 0.90 if candidate.scenario_family == "switching_case" else 0.70 if candidate.scenario_family == "file_backend_case" else 0.45
    environment_score = 0.85 if candidate.scenario_family == "environment_regime_case" else 0.30
    coverage_novelty_score = {
        "shared_boundary_case": 0.70,
        "switching_case": 0.92,
        "environment_regime_case": 0.88,
        "file_backend_case": 0.78,
    }.get(candidate.scenario_family, 0.50)
    utility = (
        0.22 * validity_score
        + 0.18 * coverage_novelty_score
        + 0.18 * boundary_score
        + 0.18 * stress_score
        + 0.12 * environment_score
        + 0.12 * provenance_score
    )
    cell_id = f"{backend_id}|{candidate.scenario_family}|{candidate.target_class}|{candidate.difficulty_tier}"
    return {
        "candidate_id": candidate.candidate_id,
        "backend_id": backend_id,
        "trajectory_id": run.run_id,
        "scenario_family": candidate.scenario_family,
        "target_class": candidate.target_class,
        "difficulty_tier": candidate.difficulty_tier,
        "environment_id": candidate.provenance.get("environment_id", ""),
        "success": success,
        "validity_score": validity_score,
        "coverage_novelty_score": coverage_novelty_score,
        "boundary_score": boundary_score,
        "classifier_stress_score": stress_score,
        "environment_score": environment_score,
        "provenance_completeness": provenance_score,
        "total_utility": utility,
        "position_range": position_range,
        "speed_range": speed_range,
        "acceleration_range": acceleration_range,
        "num_samples": len(run.times),
        "cell_id": cell_id,
    }


def _archive_rows(candidate_rows: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    best_by_cell: dict[str, dict[str, Any]] = {}
    for row in candidate_rows:
        cell_id = str(row["cell_id"])
        if cell_id not in best_by_cell or float(row["total_utility"]) > float(best_by_cell[cell_id]["total_utility"]):
            best_by_cell[cell_id] = dict(row)
    rows = []
    for cell_id, row in sorted(best_by_cell.items()):
        rows.append(
            {
                "cell_id": cell_id,
                "backend_id": row["backend_id"],
                "scenario_family": row["scenario_family"],
                "target_class": row["target_class"],
                "difficulty_tier": row["difficulty_tier"],
                "elite_candidate_id": row["candidate_id"],
                "elite_total_utility": row["total_utility"],
            }
        )
    return tuple(rows)


def _selected_manifest_rows(candidate_rows: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    best_by_cell: dict[str, dict[str, Any]] = {}
    for row in candidate_rows:
        if not bool(row["success"]):
            continue
        cell_id = str(row["cell_id"])
        if cell_id not in best_by_cell or float(row["total_utility"]) > float(best_by_cell[cell_id]["total_utility"]):
            best_by_cell[cell_id] = dict(row)
    rows = sorted(best_by_cell.values(), key=lambda row: (-float(row["total_utility"]), str(row["backend_id"])))
    return tuple(rows[:6])


def _backend_comparison_rows(candidate_rows: tuple[dict[str, Any], ...], selected_rows: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    backend_ids = sorted({str(row["backend_id"]) for row in candidate_rows})
    for backend_id in backend_ids:
        backend_rows = [row for row in candidate_rows if row["backend_id"] == backend_id]
        selected_backend_rows = [row for row in selected_rows if row["backend_id"] == backend_id]
        rows.append(
            {
                "backend_id": backend_id,
                "candidate_count": len(backend_rows),
                "selected_count": len(selected_backend_rows),
                "success_rate": sum(1 for row in backend_rows if bool(row["success"])) / max(len(backend_rows), 1),
                "mean_total_utility": sum(float(row["total_utility"]) for row in backend_rows) / max(len(backend_rows), 1),
                "mean_provenance_completeness": sum(float(row["provenance_completeness"]) for row in backend_rows) / max(len(backend_rows), 1),
            }
        )
    return tuple(rows)


def _render_backend_coverage_png(rows: tuple[dict[str, Any], ...]) -> bytes:
    backend_ids = [str(row["backend_id"]) for row in rows]
    selected_counts = [int(row["selected_count"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.bar(backend_ids, selected_counts, color="#4d8f77")
    ax.set_ylabel("Selected Count")
    ax.set_title("Backend Coverage Comparison")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    from io import BytesIO
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def _render_archive_heatmap_png(rows: tuple[dict[str, Any], ...]) -> bytes:
    backends = sorted({str(row["backend_id"]) for row in rows})
    scenarios = sorted({str(row["scenario_family"]) for row in rows})
    matrix = []
    for backend in backends:
        backend_row = []
        for scenario in scenarios:
            backend_row.append(sum(1 for row in rows if row["backend_id"] == backend and row["scenario_family"] == scenario))
        matrix.append(backend_row)
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    image = ax.imshow(matrix, cmap="Purples", aspect="auto")
    ax.set_xticks(range(len(scenarios)), labels=scenarios, fontsize=8)
    ax.set_yticks(range(len(backends)), labels=backends, fontsize=8)
    ax.set_title("Archive Coverage Heatmap")
    for row_index, values in enumerate(matrix):
        for column_index, value in enumerate(values):
            ax.text(column_index, row_index, f"{value}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.04)
    fig.tight_layout()
    from io import BytesIO
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def _render_parallel_png(rows: tuple[dict[str, Any], ...]) -> bytes:
    metrics = (
        "validity_score",
        "coverage_novelty_score",
        "boundary_score",
        "classifier_stress_score",
        "environment_score",
        "provenance_completeness",
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for row in rows:
        values = [float(row[metric]) for metric in metrics]
        ax.plot(range(len(metrics)), values, marker="o", alpha=0.7, label=str(row["candidate_id"]))
    ax.set_xticks(range(len(metrics)), labels=[metric.replace("_", "\n") for metric in metrics], fontsize=8)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Normalized Score")
    ax.set_title("Score Component Parallel Coordinates")
    fig.tight_layout()
    from io import BytesIO
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def _render_selected_gallery_png(selected_rows: tuple[dict[str, Any,]], candidate_pool: tuple[BackendCandidateSpec, ...]) -> bytes:
    adapters = _adapter_map()
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.0), sharex=False)
    for axis, row in zip(axes.flat, selected_rows[:4]):
        candidate = next(candidate for candidate in candidate_pool if candidate.candidate_id == row["candidate_id"])
        run = adapters[str(row["backend_id"])].run(candidate).trajectory_run
        axis.plot(run.times, run.truth_state.get("position", ()), marker="o", label="position")
        if "velocity" in run.truth_state:
            axis.plot(run.times, run.truth_state["velocity"], marker="s", label="velocity")
        axis.set_title(f"{row['backend_id']}\n{row['scenario_family']}", fontsize=9)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Selected Trajectory Gallery", fontsize=11)
    fig.tight_layout()
    from io import BytesIO
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def _render_provenance_dashboard_png(rows: tuple[dict[str, Any], ...]) -> bytes:
    labels = [str(row["candidate_id"]) for row in rows]
    values = [float(row["provenance_completeness"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.bar(labels, values, color="#3e6a8a")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Completeness")
    ax.set_title("Provenance Completeness Dashboard")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    from io import BytesIO
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def analyze_generic_corpus_exploration(*, seed: int = 7) -> GenericCorpusExplorationResult:
    rng = random.Random(seed)
    adapters = _adapter_map()
    candidate_pool = _candidate_pool()
    candidate_rows: list[dict[str, Any]] = []
    for candidate in candidate_pool:
        for backend_id in _backends_for_candidate(candidate):
            run = adapters[backend_id].run(candidate).trajectory_run
            candidate_rows.append(_score_run(candidate, backend_id, run))

    candidate_rows.sort(key=lambda row: (-float(row["total_utility"]), str(row["backend_id"]), str(row["candidate_id"])))
    archive_rows = _archive_rows(tuple(candidate_rows))
    selected_rows = _selected_manifest_rows(tuple(candidate_rows))

    baseline_size = len(selected_rows)
    random_baseline = tuple(rng.sample(candidate_rows, baseline_size))
    selected_coverage = len({row["cell_id"] for row in selected_rows})
    random_coverage = len({row["cell_id"] for row in random_baseline})
    selected_backend_count = len({row["backend_id"] for row in selected_rows})
    selected_scenario_count = len({row["scenario_family"] for row in selected_rows})

    exploration_manifest = {
        "exploration_version": "m35_v1",
        "candidate_pool_size": len(candidate_rows),
        "archive_cell_count": len(archive_rows),
        "selected_corpus_size": len(selected_rows),
        "random_baseline_size": baseline_size,
        "selected_coverage": selected_coverage,
        "random_baseline_coverage": random_coverage,
        "coverage_improves_over_random": selected_coverage > random_coverage,
    }
    selected_corpus_manifest = {
        "selected_rows": list(selected_rows),
        "coverage_improves_over_random": selected_coverage > random_coverage,
        "selected_backend_count": selected_backend_count,
        "selected_scenario_count": selected_scenario_count,
        "includes_boundary_examples": any(row["scenario_family"] == "shared_boundary_case" for row in selected_rows),
        "includes_stress_examples": any(row["scenario_family"] == "switching_case" for row in selected_rows),
    }
    backend_comparison_rows = _backend_comparison_rows(tuple(candidate_rows), selected_rows)

    report_markdown = "\n".join(
        [
            "# Generic Corpus Exploration Dashboard",
            "",
            "## Summary",
            f"- candidate rows explored: `{len(candidate_rows)}`",
            f"- archive cells filled: `{len(archive_rows)}`",
            f"- selected corpus size: `{len(selected_rows)}`",
            f"- selected coverage: `{selected_coverage}`",
            f"- random baseline coverage: `{random_coverage}`",
            f"- selected backends represented: `{selected_backend_count}`",
            f"- selected scenario families represented: `{selected_scenario_count}`",
            "",
            "## Selected Corpus",
            "| Candidate | Backend | Scenario | Utility | Provenance |",
            "| --- | --- | --- | --- | --- |",
            *[
                f"| `{row['candidate_id']}` | `{row['backend_id']}` | `{row['scenario_family']}` | "
                f"`{row['total_utility']:.3f}` | `{row['provenance_completeness']:.2f}` |"
                for row in selected_rows
            ],
            "",
            "## Acceptance Notes",
            f"- coverage improves over random baseline: `{selected_coverage > random_coverage}`",
            f"- includes at least two backend types: `{selected_backend_count >= 2}`",
            f"- includes boundary examples: `{selected_corpus_manifest['includes_boundary_examples']}`",
            f"- includes stress examples: `{selected_corpus_manifest['includes_stress_examples']}`",
            "- All selected rows retain backend id, scenario family, candidate id, and provenance completeness scores.",
        ]
    )

    return GenericCorpusExplorationResult(
        exploration_manifest=exploration_manifest,
        candidate_score_rows=tuple(candidate_rows),
        archive_cell_rows=archive_rows,
        selected_corpus_manifest=selected_corpus_manifest,
        backend_comparison_rows=backend_comparison_rows,
        report_markdown=report_markdown,
    )


def render_generic_corpus_exploration_numeric_walkthrough_markdown(
    result: GenericCorpusExplorationResult | None = None,
) -> str:
    payload = result or analyze_generic_corpus_exploration()
    selected_rows = tuple(payload.selected_corpus_manifest["selected_rows"])
    selected_row = selected_rows[0]
    archive_row = next(
        row for row in payload.archive_cell_rows
        if row.get("archive_status") != "failed"
        and str(row.get("elite_candidate_id", "")) == str(selected_row["candidate_id"])
        and str(row.get("backend_id", "")) == str(selected_row["backend_id"])
    )
    selected_coverage = int(payload.exploration_manifest["selected_coverage"])
    random_coverage = int(payload.exploration_manifest["random_baseline_coverage"])
    coverage_delta = selected_coverage - random_coverage

    lines = [
        "# Generic Corpus Explorer Numeric Walkthrough",
        "",
        "This walkthrough expands one selected corpus row from",
        "`generic_corpus_exploration.py` into its utility components, archive",
        "cell interpretation, and selected-versus-random coverage comparison.",
        "",
        "## Selected Candidate",
        "",
        f"- `candidate_id`: `{selected_row['candidate_id']}`",
        f"- `backend_id`: `{selected_row['backend_id']}`",
        f"- `scenario_family`: `{selected_row['scenario_family']}`",
        f"- `target_class`: `{selected_row['target_class']}`",
        f"- `difficulty_tier`: `{selected_row['difficulty_tier']}`",
        f"- `cell_id`: `{selected_row['cell_id']}`",
        "",
        "## Implemented Explorer Utility",
        "",
        "```tex",
        "U_{\\text{explore}}",
        "= 0.22 \\cdot \\text{validity}",
        "+ 0.18 \\cdot \\text{coverage novelty}",
        "+ 0.18 \\cdot \\text{boundary score}",
        "+ 0.18 \\cdot \\text{classifier stress}",
        "+ 0.12 \\cdot \\text{environment score}",
        "+ 0.12 \\cdot \\text{provenance completeness}.",
        "```",
        "",
        "## Numeric Substitution",
        "",
        "```tex",
        "U_{\\text{explore}}",
        f"= 0.22 \\cdot {float(selected_row['validity_score']):.3f}",
        f"+ 0.18 \\cdot {float(selected_row['coverage_novelty_score']):.3f}",
        f"+ 0.18 \\cdot {float(selected_row['boundary_score']):.3f}",
        f"+ 0.18 \\cdot {float(selected_row['classifier_stress_score']):.3f}",
        f"+ 0.12 \\cdot {float(selected_row['environment_score']):.3f}",
        f"+ 0.12 \\cdot {float(selected_row['provenance_completeness']):.3f}",
        f"= {float(selected_row['total_utility']):.3f}.",
        "```",
        "",
        "## Utility Components",
        "",
        f"- `validity_score`: `{float(selected_row['validity_score']):.3f}`",
        f"- `coverage_novelty_score`: `{float(selected_row['coverage_novelty_score']):.3f}`",
        f"- `boundary_score`: `{float(selected_row['boundary_score']):.3f}`",
        f"- `classifier_stress_score`: `{float(selected_row['classifier_stress_score']):.3f}`",
        f"- `environment_score`: `{float(selected_row['environment_score']):.3f}`",
        f"- `provenance_completeness`: `{float(selected_row['provenance_completeness']):.3f}`",
        f"- `total_utility`: `{float(selected_row['total_utility']):.3f}`",
        "",
        "## Archive Cell Interpretation",
        "",
        f"- `archive cell`: `{archive_row['cell_id']}`",
        f"- `elite candidate`: `{archive_row['elite_candidate_id']}`",
        f"- `elite_total_utility`: `{float(archive_row['elite_total_utility']):.3f}`",
        "",
        "This row is selected because it is the elite for its backend/scenario/",
        "class/tier cell and therefore increases structured corpus coverage, not",
        "only scalar utility.",
        "",
        "## Coverage Comparison Against Random Baseline",
        "",
        "```tex",
        "\\Delta_{\\text{coverage}}",
        f"= {selected_coverage} - {random_coverage}",
        f"= {coverage_delta}.",
        "```",
        "",
        f"- `selected_coverage`: `{selected_coverage}`",
        f"- `random_baseline_coverage`: `{random_coverage}`",
        f"- `coverage_improves_over_random`: `{payload.exploration_manifest['coverage_improves_over_random']}`",
        "",
        "## Interpretation",
        "",
        "- The selected row is not just high-utility in isolation.",
        "- It is part of a selected set that covers more archive cells than a",
        "  same-size random baseline.",
        "- That is the core Explorer claim: corpus selection should improve both",
        "  utility and structural coverage.",
    ]
    return "\n".join(lines)


def write_generic_corpus_exploration_artifacts(
    base_dir: str | Path,
    *,
    result: GenericCorpusExplorationResult | None = None,
) -> GenericCorpusExplorationArtifacts:
    run_dir = Path(base_dir) / "generic_corpus_exploration"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = result or analyze_generic_corpus_exploration()

    exploration_manifest_path = run_dir / "exploration_manifest.json"
    candidate_scores_path = run_dir / "candidate_scores.csv"
    archive_cells_path = run_dir / "archive_cells.csv"
    selected_corpus_manifest_path = run_dir / "selected_corpus_manifest.json"
    backend_comparison_path = run_dir / "backend_comparison.csv"
    report_path = run_dir / "corpus_exploration_report.md"
    numeric_walkthrough_path = run_dir / "generic_corpus_explorer_numeric_walkthrough.md"
    backend_coverage_png_path = run_dir / "backend_coverage_comparison.png"
    archive_heatmap_png_path = run_dir / "archive_coverage_heatmap.png"
    score_parallel_png_path = run_dir / "score_component_parallel_coordinates.png"
    selected_gallery_png_path = run_dir / "selected_trajectory_gallery.png"
    provenance_dashboard_png_path = run_dir / "provenance_completeness_dashboard.png"

    exploration_manifest_path.write_text(json.dumps(payload.exploration_manifest, indent=2), encoding="utf-8")
    selected_corpus_manifest_path.write_text(json.dumps(payload.selected_corpus_manifest, indent=2), encoding="utf-8")
    report_path.write_text(payload.report_markdown, encoding="utf-8")
    numeric_walkthrough_path.write_text(render_generic_corpus_exploration_numeric_walkthrough_markdown(payload), encoding="utf-8")

    candidate_fieldnames = list(payload.candidate_score_rows[0].keys()) if payload.candidate_score_rows else []
    archive_fieldnames = list(payload.archive_cell_rows[0].keys()) if payload.archive_cell_rows else []
    backend_fieldnames = list(payload.backend_comparison_rows[0].keys()) if payload.backend_comparison_rows else []
    _write_csv(candidate_scores_path, list(payload.candidate_score_rows), candidate_fieldnames)
    _write_csv(archive_cells_path, list(payload.archive_cell_rows), archive_fieldnames)
    _write_csv(backend_comparison_path, list(payload.backend_comparison_rows), backend_fieldnames)

    candidate_pool = _candidate_pool()
    backend_coverage_png_path.write_bytes(_render_backend_coverage_png(payload.backend_comparison_rows))
    archive_heatmap_png_path.write_bytes(_render_archive_heatmap_png(payload.archive_cell_rows))
    selected_rows = tuple(payload.selected_corpus_manifest["selected_rows"])
    score_parallel_png_path.write_bytes(_render_parallel_png(selected_rows))
    selected_gallery_png_path.write_bytes(_render_selected_gallery_png(selected_rows, candidate_pool))
    provenance_dashboard_png_path.write_bytes(_render_provenance_dashboard_png(selected_rows))

    return GenericCorpusExplorationArtifacts(
        run_dir=run_dir,
        exploration_manifest_path=exploration_manifest_path,
        candidate_scores_path=candidate_scores_path,
        archive_cells_path=archive_cells_path,
        selected_corpus_manifest_path=selected_corpus_manifest_path,
        backend_comparison_path=backend_comparison_path,
        report_path=report_path,
        numeric_walkthrough_path=numeric_walkthrough_path,
        backend_coverage_png_path=backend_coverage_png_path,
        archive_heatmap_png_path=archive_heatmap_png_path,
        score_parallel_png_path=score_parallel_png_path,
        selected_gallery_png_path=selected_gallery_png_path,
        provenance_dashboard_png_path=provenance_dashboard_png_path,
    )
