from __future__ import annotations

import io

from ...markdown_builder import MarkdownDocument
from ...runtime_paths import prepare_matplotlib
from ...utils.plotting import plt
from .contracts import CrossMethodPriorComparisonResult, PriorSensitivityResult


def render_prior_sensitivity_report(result: PriorSensitivityResult) -> str:
    report = MarkdownDocument("Prior Sensitivity and Bias Study")
    report.paragraph(
        f"This analysis sweeps binary class priors over the `{result.method_name}` classifier and records how final posteriors, "
        f"hard decisions, and confidence change as prior odds move."
    )
    report.heading("Summary", level=2)
    report.bullet_list(
        [
            f"Trajectories analyzed: {result.summary.trajectory_count}",
            f"Sweep rows: {result.summary.sweep_count}",
            f"Fraction flipped by +/- {result.prior_dominance_metrics['small_prior_delta']:.2f} prior perturbation: {result.summary.flipped_by_small_prior_fraction:.3f}",
            f"Median smallest prior shift to flip: {result.summary.median_smallest_prior_shift_to_flip if result.summary.median_smallest_prior_shift_to_flip is not None else 'n/a'}",
            f"Median smallest log-prior shift to flip: {result.summary.median_smallest_log_prior_shift_to_flip if result.summary.median_smallest_log_prior_shift_to_flip is not None else 'n/a'}",
            f"Ambiguous uniform-prior class: `{result.summary.ambiguous_uniform_class}`",
            f"Ambiguous minimum prior_A for class A: {result.summary.ambiguous_flip_threshold_for_a if result.summary.ambiguous_flip_threshold_for_a is not None else 'n/a'}",
        ]
    )
    report.heading("Flip Thresholds", level=2)
    report.table(
        ["trajectory_id", "scenario", "uniform_class", "uniform_confidence", "min_prior_A_for_A", "max_prior_A_for_B", "smallest_shift_to_flip"],
        [
            (
                row.trajectory_id,
                row.scenario_name,
                row.uniform_prior_class,
                f"{row.uniform_prior_confidence:.3f}",
                row.min_prior_a_for_a if row.min_prior_a_for_a is not None else "n/a",
                row.max_prior_a_for_b if row.max_prior_a_for_b is not None else "n/a",
                row.smallest_prior_shift_to_flip if row.smallest_prior_shift_to_flip is not None else "n/a",
            )
            for row in result.flip_thresholds
        ],
    )
    report.heading("Interpretation", level=2)
    report.bullet_list(
        [
            "Easy trajectories should remain evidence-driven, so prior sweeps should not flip the final class within a moderate prior range.",
            "Ambiguous trajectories should show prior-dominant regions where small prior changes alter the decision or move the run into abstain.",
            "With forgetting factor 1.0, the log-posterior odds decomposition is exact: log posterior odds = cumulative log-likelihood ratio + log prior odds.",
        ]
    )
    return report.text()


def _build_posterior_figure(result: PriorSensitivityResult):
    trajectory_ids = [getattr(trajectory, "trajectory_id") for trajectory in result.trajectories]
    selected_ids = trajectory_ids[:4]
    rows_by_trajectory: dict[str, list] = {trajectory_id: [] for trajectory_id in trajectory_ids}
    for row in result.sweep_rows:
        rows_by_trajectory[row.trajectory_id].append(row)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), sharex=True, sharey=True)
    for axis, trajectory_id in zip(axes.flat, selected_ids):
        rows = sorted(rows_by_trajectory[trajectory_id], key=lambda row: row.prior_a)
        axis.plot([row.prior_a for row in rows], [row.posterior_a for row in rows], color="#2563eb", linewidth=2.2, label="posterior_A")
        axis.plot([row.prior_a for row in rows], [row.posterior_b for row in rows], color="#dc2626", linewidth=2.2, label="posterior_B")
        axis.axvline(0.5, color="#6b7280", linestyle="--", linewidth=1.0)
        axis.axhline(0.75, color="#9ca3af", linestyle=":", linewidth=1.0)
        axis.set_title(trajectory_id, loc="left", fontsize=11, fontweight="bold")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.grid(True, alpha=0.25)
        axis.set_xlabel("prior_A")
        axis.set_ylabel("final posterior")
        axis.legend(frameon=False, fontsize=9)
    fig.suptitle(f"Prior Sweep: Final Posterior vs Prior ({result.method_name})", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def _build_flip_figure(result: PriorSensitivityResult):
    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    labels = [row.trajectory_id for row in result.flip_thresholds]
    values = [row.smallest_prior_shift_to_flip if row.smallest_prior_shift_to_flip is not None else 0.5 for row in result.flip_thresholds]
    colors = ["#d97706" if row.smallest_prior_shift_to_flip is not None else "#2563eb" for row in result.flip_thresholds]
    ax.bar(range(len(labels)), values, color=colors, alpha=0.9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0.0, 0.55)
    ax.set_ylabel("minimum |prior_A - 0.5| to flip")
    ax.set_title("Decision Fragility Under Prior Sweep", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _build_heatmap_figure(result: PriorSensitivityResult):
    trajectory_ids = [getattr(trajectory, "trajectory_id") for trajectory in result.trajectories]
    prior_values = sorted({row.prior_a for row in result.sweep_rows})
    matrix = []
    for trajectory_id in trajectory_ids:
        row_values = []
        for prior_a in prior_values:
            row = next(sweep_row for sweep_row in result.sweep_rows if sweep_row.trajectory_id == trajectory_id and abs(sweep_row.prior_a - prior_a) < 1e-9)
            row_values.append(row.posterior_a)
        matrix.append(row_values)
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=0.0, vmax=1.0)
    ax.set_title("Posterior_A Heatmap Across Prior Sweeps", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("prior_A")
    ax.set_ylabel("trajectory")
    ax.set_xticks(range(len(prior_values)))
    ax.set_xticklabels([f"{value:.2f}" for value in prior_values], rotation=45, ha="right")
    ax.set_yticks(range(len(trajectory_ids)))
    ax.set_yticklabels(trajectory_ids)
    fig.colorbar(image, ax=ax, label="final posterior_A")
    fig.tight_layout()
    return fig


def _build_decision_map_figure(result: PriorSensitivityResult):
    trajectory_ids = [getattr(trajectory, "trajectory_id") for trajectory in result.trajectories]
    prior_values = sorted({row.prior_a for row in result.sweep_rows})
    class_a, class_b = result.class_names
    value_map = {class_b: 0.0, "unknown": 0.5, class_a: 1.0}
    matrix = []
    for trajectory_id in trajectory_ids:
        row_values = []
        for prior_a in prior_values:
            row = next(sweep_row for sweep_row in result.sweep_rows if sweep_row.trajectory_id == trajectory_id and abs(sweep_row.prior_a - prior_a) < 1e-9)
            row_values.append(value_map[row.final_class])
        matrix.append(row_values)
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=0.0, vmax=1.0)
    ax.set_title("Prior Sensitivity Decision Map", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("prior_A")
    ax.set_ylabel("trajectory")
    ax.set_xticks(range(len(prior_values)))
    ax.set_xticklabels([f"{value:.2f}" for value in prior_values], rotation=45, ha="right")
    ax.set_yticks(range(len(trajectory_ids)))
    ax.set_yticklabels(trajectory_ids)
    colorbar = fig.colorbar(image, ax=ax, label="decision")
    colorbar.set_ticks([0.0, 0.5, 1.0])
    colorbar.set_ticklabels([class_b, "unknown", class_a])
    fig.tight_layout()
    return fig


def _build_decomposition_figure(result: PriorSensitivityResult):
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    selected_ids = ["easy_A_0", "ambiguous_mid", "late_flip"]
    colors = {"easy_A_0": "#2563eb", "ambiguous_mid": "#d97706", "late_flip": "#7c3aed"}
    for trajectory_id in selected_ids:
        rows = sorted([row for row in result.sweep_rows if row.trajectory_id == trajectory_id], key=lambda row: row.log_prior_odds)
        ax.plot([row.log_prior_odds for row in rows], [row.final_log_posterior_odds for row in rows], color=colors[trajectory_id], linewidth=2.2, label=trajectory_id)
    reference_rows = [row for row in result.sweep_rows if row.trajectory_id == "ambiguous_mid"]
    if reference_rows:
        xs = [row.log_prior_odds for row in sorted(reference_rows, key=lambda row: row.log_prior_odds)]
        ax.plot(xs, xs, color="#6b7280", linestyle="--", linewidth=1.2, label="posterior=prior reference")
    ax.axhline(0.0, color="#9ca3af", linestyle=":", linewidth=1.0)
    ax.axvline(0.0, color="#9ca3af", linestyle=":", linewidth=1.0)
    ax.set_title("Log-Odds Decomposition: Evidence + Prior", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("log prior odds (A/B)")
    ax.set_ylabel("final log posterior odds (A/B)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _build_pairwise_flip_heatmap_figure(result: PriorSensitivityResult):
    pair_label = f"{result.class_names[0]}_vs_{result.class_names[1]}"
    trajectory_ids = [row.trajectory_id for row in result.flip_thresholds]
    matrix = [[row.smallest_prior_shift_to_flip if row.smallest_prior_shift_to_flip is not None else 0.5] for row in result.flip_thresholds]
    fig, ax = plt.subplots(figsize=(5.8, max(4.5, 0.42 * len(trajectory_ids) + 1.8)))
    image = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=0.5)
    ax.set_title("Pairwise Flip Threshold Heatmap", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("class pair")
    ax.set_ylabel("trajectory")
    ax.set_xticks([0])
    ax.set_xticklabels([pair_label])
    ax.set_yticks(range(len(trajectory_ids)))
    ax.set_yticklabels(trajectory_ids)
    for row_index, values in enumerate(matrix):
        ax.text(0, row_index, f"{values[0]:.2f}" if values[0] < 0.5 else "n/a", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="minimum |prior shift| to flip")
    fig.tight_layout()
    return fig


def _build_fragility_overview_figure(result: PriorSensitivityResult):
    ordered = sorted(result.flip_thresholds, key=lambda row: (row.smallest_prior_shift_to_flip is None, row.smallest_prior_shift_to_flip if row.smallest_prior_shift_to_flip is not None else 1.0, row.trajectory_id))
    labels = [row.trajectory_id for row in ordered]
    values = [row.smallest_prior_shift_to_flip if row.smallest_prior_shift_to_flip is not None else 0.50 for row in ordered]
    colors = ["#d97706" if row.smallest_prior_shift_to_flip is not None else "#9ca3af" for row in ordered]
    fig, ax = plt.subplots(figsize=(8.8, max(4.6, 0.4 * len(ordered) + 1.5)))
    positions = list(range(len(ordered)))
    ax.barh(positions, values, color=colors, alpha=0.88)
    ax.axvline(0.25, color="#2563eb", linestyle="--", linewidth=1.2, label="small prior perturbation")
    ax.axvline(0.50, color="#6b7280", linestyle=":", linewidth=1.0, label="stable in sweep")
    ax.set_title("Trajectory Prior Fragility Overview", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("minimum absolute prior shift needed to flip")
    ax.set_ylabel("trajectory")
    ax.set_xlim(0.0, 0.52)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    for index, row in enumerate(ordered):
        label = "stable" if row.smallest_prior_shift_to_flip is None else f"{row.smallest_prior_shift_to_flip:.2f}"
        ax.text(min(values[index] + 0.012, 0.505), index, label, va="center", fontsize=8)
    ax.grid(True, axis="x", alpha=0.22)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def _to_svg(fig) -> str:
    buffer = io.StringIO()
    try:
        fig.savefig(buffer, format="svg", bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def _to_png(fig) -> bytes:
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def render_prior_sensitivity_posterior_svg(result: PriorSensitivityResult) -> str:
    return _to_svg(_build_posterior_figure(result))


def render_prior_sensitivity_posterior_png_bytes(result: PriorSensitivityResult) -> bytes:
    return _to_png(_build_posterior_figure(result))


def render_prior_sensitivity_flip_svg(result: PriorSensitivityResult) -> str:
    return _to_svg(_build_flip_figure(result))


def render_prior_sensitivity_flip_png_bytes(result: PriorSensitivityResult) -> bytes:
    return _to_png(_build_flip_figure(result))


def render_prior_sensitivity_heatmap_svg(result: PriorSensitivityResult) -> str:
    return _to_svg(_build_heatmap_figure(result))


def render_prior_sensitivity_heatmap_png_bytes(result: PriorSensitivityResult) -> bytes:
    return _to_png(_build_heatmap_figure(result))


def render_prior_sensitivity_decision_svg(result: PriorSensitivityResult) -> str:
    return _to_svg(_build_decision_map_figure(result))


def render_prior_sensitivity_decision_png_bytes(result: PriorSensitivityResult) -> bytes:
    return _to_png(_build_decision_map_figure(result))


def render_prior_sensitivity_decomposition_svg(result: PriorSensitivityResult) -> str:
    return _to_svg(_build_decomposition_figure(result))


def render_prior_sensitivity_decomposition_png_bytes(result: PriorSensitivityResult) -> bytes:
    return _to_png(_build_decomposition_figure(result))


def render_prior_sensitivity_pairwise_flip_svg(result: PriorSensitivityResult) -> str:
    return _to_svg(_build_pairwise_flip_heatmap_figure(result))


def render_prior_sensitivity_pairwise_flip_png_bytes(result: PriorSensitivityResult) -> bytes:
    return _to_png(_build_pairwise_flip_heatmap_figure(result))


def render_prior_sensitivity_fragility_svg(result: PriorSensitivityResult) -> str:
    return _to_svg(_build_fragility_overview_figure(result))


def render_prior_sensitivity_fragility_png_bytes(result: PriorSensitivityResult) -> bytes:
    return _to_png(_build_fragility_overview_figure(result))


def render_cross_method_prior_comparison_report(result: CrossMethodPriorComparisonResult) -> str:
    report = MarkdownDocument("Cross-Method Prior Sensitivity Comparison")
    report.paragraph(
        "This artifact compares smallest prior shifts needed to flip the final decision across the current baseline methods. "
        "Lower values mean higher prior fragility. `stable` means the scenario family exists for that method but no flip was observed "
        "anywhere in the swept prior range. `n/a` means that scenario family is not represented for that method."
    )
    table_rows: list[tuple[str, ...]] = []
    for row in result.rows:
        threshold_cells: list[str] = []
        for scenario_name in result.scenario_names:
            status = row[f"{scenario_name}_status"]
            if status == "missing":
                threshold_cells.append("n/a")
            elif status == "stable":
                threshold_cells.append("stable")
            else:
                threshold_cells.append(f"{row[scenario_name]:.2f}")
        table_rows.append((str(row["method_name"]), *threshold_cells, f"{row['fraction_flipped_by_small_prior_perturbation']:.3f}"))
    report.heading("Scenario Flip Thresholds", level=2)
    report.table(["method", *result.scenario_names, "small-prior-flip-fraction"], table_rows)
    report.heading("Interpretation", level=2)
    report.bullet_list(
        [
            "Lower threshold means a smaller prior change can flip the final decision.",
            "`stable` means the scenario family is present but no flip was observed within the swept prior range.",
            "`n/a` means that scenario family is not represented for that method.",
            "The final column summarizes how often each method flipped under the configured small prior perturbation.",
        ]
    )
    return report.text()


def _build_cross_method_prior_comparison_figure(result: CrossMethodPriorComparisonResult):
    matrix = []
    for row in result.rows:
        matrix.append([float("nan") if row[f"{scenario_name}_status"] == "missing" else float(row[scenario_name]) for scenario_name in result.scenario_names])
    fig, ax = plt.subplots(figsize=(max(8.2, 0.9 * len(result.scenario_names) + 2.2), 4.8))
    colormap = plt.get_cmap("YlOrRd").copy()
    colormap.set_bad(color="#e5e7eb")
    image = ax.imshow(matrix, aspect="auto", cmap=colormap, vmin=0.0, vmax=0.50)
    ax.set_title("Cross-Method Prior Fragility Heatmap", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("scenario")
    ax.set_ylabel("method")
    ax.set_xticks(range(len(result.scenario_names)))
    ax.set_xticklabels(list(result.scenario_names), rotation=35, ha="right")
    ax.set_yticks(range(len(result.rows)))
    ax.set_yticklabels([str(row["method_name"]) for row in result.rows])
    for row_index, row in enumerate(result.rows):
        for col_index, scenario_name in enumerate(result.scenario_names):
            status = row[f"{scenario_name}_status"]
            label = "n/a" if status == "missing" else ("stable" if status == "stable" else f"{row[scenario_name]:.2f}")
            ax.text(col_index, row_index, label, ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="minimum |prior shift| to flip")
    fig.tight_layout()
    return fig


def render_cross_method_prior_comparison_svg(result: CrossMethodPriorComparisonResult) -> str:
    return _to_svg(_build_cross_method_prior_comparison_figure(result))


def render_cross_method_prior_comparison_png_bytes(result: CrossMethodPriorComparisonResult) -> bytes:
    return _to_png(_build_cross_method_prior_comparison_figure(result))
