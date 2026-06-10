from __future__ import annotations

from io import BytesIO
from typing import Any

from ...utils.plotting import plt
from .environment_aware_corpus_core import _candidate_rows, _environment_adapter


def _render_coverage_heatmap_png(rows: tuple[dict[str, Any], ...]) -> bytes:
    environments = sorted({str(row["environment_id"]) for row in rows})
    classes = sorted({str(row["true_class"]) for row in rows})
    matrix = []
    for environment_id in environments:
        environment_row = []
        for true_class in classes:
            match = next(row for row in rows if row["environment_id"] == environment_id and row["true_class"] == true_class)
            environment_row.append(float(match["trajectory_count"]))
        matrix.append(environment_row)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    image = ax.imshow(matrix, cmap="YlOrBr", aspect="auto")
    ax.set_xticks(range(len(classes)), labels=classes, fontsize=9)
    ax.set_yticks(range(len(environments)), labels=environments, fontsize=9)
    ax.set_title("Environment Regime Coverage")
    for row_index, row_values in enumerate(matrix):
        for column_index, value in enumerate(row_values):
            ax.text(column_index, row_index, f"{value:.0f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.04)
    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def _render_leakage_plot_png(rows: tuple[dict[str, Any], ...]) -> bytes:
    labels = [f"{row['slice_id']}:{row['variable_name']}" for row in rows]
    values = [float(row["delta_ratio"]) for row in rows]
    colors = ["#ca5b4b" if bool(row["flagged_class_linkage"]) else "#4d8f77" for row in rows]

    fig, ax = plt.subplots(figsize=(10.0, 4.0))
    ax.bar(range(len(labels)), values, color=colors)
    ax.axhline(0.15, color="black", linestyle="--", linewidth=1.0, label="flag threshold")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Delta Ratio")
    ax.set_title("Environment Leakage Audit")
    ax.legend(fontsize=8)
    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def _render_trajectory_gallery_png() -> bytes:
    adapter = _environment_adapter()
    chosen_ids = (
        "constant_velocity_dense_calm_0",
        "constant_velocity_thin_windy_0",
        "constant_acceleration_dense_calm_0",
        "constant_acceleration_thin_windy_0",
    )
    chosen = [candidate for candidate in _candidate_rows() if candidate.candidate_id in chosen_ids]

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.0), sharex=True)
    for axis, candidate in zip(axes.flat, chosen):
        record = adapter.run(candidate)
        run = record.trajectory_run
        axis.plot(run.times, run.truth_state["position"], marker="o", label="position")
        axis.plot(run.times, run.truth_state["velocity"], marker="s", label="velocity")
        axis.set_title(f"{candidate.target_class}\n{candidate.provenance['environment_id']}", fontsize=9)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Environment-Conditioned Trajectory Gallery", fontsize=11)
    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()
