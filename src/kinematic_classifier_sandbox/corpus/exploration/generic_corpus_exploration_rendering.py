from __future__ import annotations

from typing import Any

from kinematic_classifier_sandbox.utils.plotting import _figure_to_png, plt

from .generic_corpus_exploration_core import _set_jaccard


def _render_backend_coverage_png(rows: tuple[dict[str, Any], ...]) -> bytes:
    backend_ids = [str(row["backend_id"]) for row in rows]
    selected_counts = [int(row["selected_count"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.bar(backend_ids, selected_counts, color="#4d8f77")
    ax.set_ylabel("Selected Count")
    ax.set_title("Backend Coverage Comparison")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    return _figure_to_png(fig)


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
    return _figure_to_png(fig)


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
    return _figure_to_png(fig)


def _render_selected_gallery_png(selected_rows: tuple[dict[str, Any], ...], candidate_pool: tuple[Any, ...], adapter_map: dict[str, Any]) -> bytes:
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.0), sharex=False)
    for axis, row in zip(axes.flat, selected_rows[:4]):
        candidate = next(candidate for candidate in candidate_pool if candidate.candidate_id == row["candidate_id"])
        run = adapter_map[str(row["backend_id"])].run(candidate).trajectory_run
        axis.plot(run.times, run.truth_state.get("position", ()), marker="o", label="position")
        if "velocity" in run.truth_state:
            axis.plot(run.times, run.truth_state["velocity"], marker="s", label="velocity")
        axis.set_title(f"{row['backend_id']}\n{row['scenario_family']}", fontsize=9)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Selected Trajectory Gallery", fontsize=11)
    fig.tight_layout()
    return _figure_to_png(fig)


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
    return _figure_to_png(fig)


def _render_weight_sweep_tradeoff_png(rows: tuple[Any, ...]) -> bytes:
    variant_ids = [row.variant_id for row in rows]
    coverage_delta = [row.coverage_delta_vs_baseline for row in rows]
    candidate_jaccard = [row.candidate_jaccard_vs_baseline for row in rows]
    cell_jaccard = [row.cell_jaccard_vs_baseline for row in rows]
    fig, ax1 = plt.subplots(figsize=(10.0, 4.8))
    ax1.bar(variant_ids, coverage_delta, color="#4d8f77", alpha=0.7, label="Coverage Delta vs Baseline")
    ax1.set_ylabel("Coverage Delta vs Baseline")
    ax1.tick_params(axis="x", rotation=20)
    ax1.grid(axis="y", alpha=0.2)
    ax2 = ax1.twinx()
    ax2.plot(variant_ids, candidate_jaccard, color="#7c3aed", marker="o", linewidth=2.0, label="Candidate Jaccard")
    ax2.plot(variant_ids, cell_jaccard, color="#0f766e", marker="s", linewidth=2.0, label="Cell Jaccard")
    ax2.set_ylabel("Jaccard vs Baseline")
    ax2.set_ylim(0.0, 1.05)
    fig.suptitle("Explorer Weight Sweep Tradeoff", fontsize=11)
    fig.tight_layout()
    return _figure_to_png(fig)


def _render_weight_sweep_overlap_png(rows: tuple[Any, ...]) -> bytes:
    variant_ids = [row.variant_id for row in rows]
    candidate_sets = [set(row.selected_candidate_ids) for row in rows]
    cell_sets = [set(row.selected_cell_ids) for row in rows]
    candidate_matrix = [
        [_set_jaccard(left, right) for right in candidate_sets]
        for left in candidate_sets
    ]
    cell_matrix = [
        [_set_jaccard(left, right) for right in cell_sets]
        for left in cell_sets
    ]
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.2), constrained_layout=True)
    for axis, matrix, title in zip(
        axes,
        (candidate_matrix, cell_matrix),
        ("Candidate Overlap Heatmap", "Cell Overlap Heatmap"),
    ):
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
        axis.set_xticks(range(len(variant_ids)), labels=variant_ids, rotation=20, fontsize=7)
        axis.set_yticks(range(len(variant_ids)), labels=variant_ids, fontsize=7)
        axis.set_title(title)
        for row_index, values in enumerate(matrix):
            for column_index, value in enumerate(values):
                axis.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=6, color="white" if value < 0.6 else "black")
        fig.colorbar(image, ax=axis, fraction=0.04, pad=0.02)
    return _figure_to_png(fig)


def _render_weight_sweep_weight_matrix_png(rows: tuple[Any, ...]) -> bytes:
    variant_ids = [row.variant_id for row in rows]
    matrix = [
        [
            row.weight_validity,
            row.weight_coverage_novelty,
            row.weight_boundary,
            row.weight_stress,
            row.weight_environment,
            row.weight_provenance,
        ]
        for row in rows
    ]
    labels = ["validity", "coverage", "boundary", "stress", "environment", "provenance"]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=20, fontsize=8)
    ax.set_yticks(range(len(variant_ids)), labels=variant_ids, fontsize=8)
    ax.set_title("Weight Matrix")
    for row_index, values in enumerate(matrix):
        for column_index, value in enumerate(values):
            ax.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.04)
    fig.tight_layout()
    return _figure_to_png(fig)
