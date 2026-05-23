from __future__ import annotations

from dataclasses import dataclass, asdict
from math import exp, log
import csv
import io
import json
import os
from pathlib import Path

from .sequential_bayes_accumulator import (
    AccumulatorClassSpec,
    AccumulatorTrajectory,
    default_accumulator_class_specs,
    generate_accumulator_trajectories,
    run_accumulator,
)
from .pointwise_baseline import (
    PointwiseClassSpec,
    PointwiseTrajectory,
    default_pointwise_class_specs,
    generate_pointwise_benchmark_trajectories,
    run_pointwise_classifier,
)
from .windowed_baseline import (
    WindowedClassSpec,
    WindowedTrajectory,
    WindowedFeatureClassifier,
    default_windowed_class_specs,
    extract_windowed_feature_rows,
    generate_windowed_trajectories,
)


def _prepare_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp/kinematic-classifier-sandbox-mpl")))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _log_odds_from_prior(prior_a: float, prior_b: float) -> float:
    return log(max(prior_a, 1e-12) / max(prior_b, 1e-12))


def _binary_log_odds(weights: dict[str, float], class_a: str, class_b: str) -> float:
    return log(max(weights[class_a], 1e-12) / max(weights[class_b], 1e-12))


@dataclass(frozen=True, slots=True)
class PriorSweepRow:
    trajectory_id: str
    scenario_name: str
    true_class: str
    prior_a: float
    prior_b: float
    log_prior_odds: float
    final_class: str
    final_confidence: float
    abstained: bool
    posterior_a: float
    posterior_b: float
    final_log_posterior_odds: float
    cumulative_log_likelihood_ratio: float


@dataclass(frozen=True, slots=True)
class PriorFlipThreshold:
    trajectory_id: str
    scenario_name: str
    true_class: str
    uniform_prior_class: str
    uniform_prior_confidence: float
    min_prior_a_for_a: float | None
    max_prior_a_for_b: float | None
    smallest_prior_shift_to_flip: float | None
    smallest_log_prior_shift_to_flip: float | None


@dataclass(frozen=True, slots=True)
class PriorSensitivitySummary:
    trajectory_count: int
    sweep_count: int
    flipped_by_small_prior_fraction: float
    median_smallest_prior_shift_to_flip: float | None
    median_smallest_log_prior_shift_to_flip: float | None
    ambiguous_uniform_class: str
    ambiguous_flip_threshold_for_a: float | None


@dataclass(frozen=True, slots=True)
class PriorSensitivityResult:
    method_name: str
    class_names: tuple[str, ...]
    trajectories: tuple[object, ...]
    sweep_rows: tuple[PriorSweepRow, ...]
    flip_thresholds: tuple[PriorFlipThreshold, ...]
    summary: PriorSensitivitySummary
    prior_dominance_metrics: dict[str, object]


@dataclass(frozen=True, slots=True)
class PriorSensitivityArtifacts:
    run_dir: Path
    report_path: Path
    sweep_path: Path
    flip_thresholds_path: Path
    metrics_path: Path
    config_path: Path
    plot_posterior_svg_path: Path
    plot_posterior_png_path: Path
    plot_flip_svg_path: Path
    plot_flip_png_path: Path
    plot_heatmap_svg_path: Path
    plot_heatmap_png_path: Path
    plot_decision_svg_path: Path
    plot_decision_png_path: Path
    plot_decomposition_svg_path: Path
    plot_decomposition_png_path: Path
    plot_pairwise_flip_svg_path: Path
    plot_pairwise_flip_png_path: Path


@dataclass(frozen=True, slots=True)
class CrossMethodPriorComparisonResult:
    method_results: tuple[PriorSensitivityResult, ...]
    scenario_names: tuple[str, ...]
    rows: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class CrossMethodPriorComparisonArtifacts:
    run_dir: Path
    report_path: Path
    comparison_csv_path: Path
    status_csv_path: Path
    plot_svg_path: Path
    plot_png_path: Path


def _default_prior_grid(step: float = 0.05) -> tuple[float, ...]:
    values = []
    current = step
    while current < 1.0:
        values.append(round(current, 10))
        current += step
    return tuple(values)


def _trajectory_order_key(trajectory: AccumulatorTrajectory) -> tuple[int, str]:
    order = {"easy": 0, "ambiguous": 1, "late_flip": 2}
    return (order.get(trajectory.scenario_name, 99), trajectory.trajectory_id)


def _generic_trajectory_order_key(trajectory) -> tuple[int, str]:
    order = {
        "easy": 0,
        "overlap": 1,
        "ambiguous": 2,
        "late_flip": 3,
        "low_clean": 4,
        "high_clean": 5,
        "low_spike": 6,
        "high_dip": 7,
        "low_long": 8,
        "high_long": 9,
    }
    return (order.get(getattr(trajectory, "scenario_name", ""), 99), getattr(trajectory, "trajectory_id", ""))


def _analyze_generic_prior_sensitivity(
    *,
    method_name: str,
    class_names: tuple[str, str],
    trajectories: tuple[object, ...],
    runner,
    prior_grid: tuple[float, ...],
    confidence_threshold: float,
    forgetting_factor: float,
) -> PriorSensitivityResult:
    class_a, class_b = class_names
    sweep_rows: list[PriorSweepRow] = []
    rows_by_trajectory: dict[str, list[PriorSweepRow]] = {
        getattr(trajectory, "trajectory_id"): [] for trajectory in trajectories
    }
    for trajectory in trajectories:
        trajectory_id = getattr(trajectory, "trajectory_id")
        scenario_name = getattr(trajectory, "scenario_name")
        true_class = getattr(trajectory, "true_class")
        for prior_a in prior_grid:
            prior = {class_a: prior_a, class_b: 1.0 - prior_a}
            final_class, final_confidence, final_weights, cumulative_log_likelihood_ratio = runner(trajectory, prior)
            row = PriorSweepRow(
                trajectory_id=trajectory_id,
                scenario_name=scenario_name,
                true_class=true_class,
                prior_a=prior_a,
                prior_b=1.0 - prior_a,
                log_prior_odds=_log_odds_from_prior(prior_a, 1.0 - prior_a),
                final_class=final_class,
                final_confidence=final_confidence,
                abstained=final_class == "unknown",
                posterior_a=final_weights[class_a],
                posterior_b=final_weights[class_b],
                final_log_posterior_odds=_binary_log_odds(final_weights, class_a, class_b),
                cumulative_log_likelihood_ratio=cumulative_log_likelihood_ratio,
            )
            sweep_rows.append(row)
            rows_by_trajectory[trajectory_id].append(row)

    flip_thresholds: list[PriorFlipThreshold] = []
    flipped_by_small_prior = 0
    prior_shift_values: list[float] = []
    log_prior_shift_values: list[float] = []
    ambiguous_uniform_class = "n/a"
    ambiguous_flip_threshold_for_a = None
    small_prior_delta = 0.25
    uniform_prior = 0.50
    for trajectory in trajectories:
        trajectory_id = getattr(trajectory, "trajectory_id")
        scenario_name = getattr(trajectory, "scenario_name")
        true_class = getattr(trajectory, "true_class")
        rows = sorted(rows_by_trajectory[trajectory_id], key=lambda row: row.prior_a)
        uniform_row = min(rows, key=lambda row: abs(row.prior_a - uniform_prior))
        rows_for_a = [row.prior_a for row in rows if row.final_class == class_a]
        rows_for_b = [row.prior_a for row in rows if row.final_class == class_b]
        min_prior_a_for_a = min(rows_for_a) if rows_for_a else None
        max_prior_a_for_b = max(rows_for_b) if rows_for_b else None
        flip_candidates = [row for row in rows if row.final_class != uniform_row.final_class]
        smallest_prior_shift_to_flip = None
        smallest_log_prior_shift_to_flip = None
        if flip_candidates:
            smallest_prior_shift_to_flip = min(abs(row.prior_a - uniform_row.prior_a) for row in flip_candidates)
            prior_shift_values.append(smallest_prior_shift_to_flip)
            base_log_prior = _log_odds_from_prior(uniform_row.prior_a, 1.0 - uniform_row.prior_a)
            smallest_log_prior_shift_to_flip = min(abs(row.log_prior_odds - base_log_prior) for row in flip_candidates)
            log_prior_shift_values.append(smallest_log_prior_shift_to_flip)
        lower_row = min(rows, key=lambda row: abs(row.prior_a - (uniform_prior - small_prior_delta)))
        upper_row = min(rows, key=lambda row: abs(row.prior_a - (uniform_prior + small_prior_delta)))
        if lower_row.final_class != uniform_row.final_class or upper_row.final_class != uniform_row.final_class:
            flipped_by_small_prior += 1
        flip_thresholds.append(
            PriorFlipThreshold(
                trajectory_id=trajectory_id,
                scenario_name=scenario_name,
                true_class=true_class,
                uniform_prior_class=uniform_row.final_class,
                uniform_prior_confidence=uniform_row.final_confidence,
                min_prior_a_for_a=min_prior_a_for_a,
                max_prior_a_for_b=max_prior_a_for_b,
                smallest_prior_shift_to_flip=smallest_prior_shift_to_flip,
                smallest_log_prior_shift_to_flip=smallest_log_prior_shift_to_flip,
            )
        )
        if scenario_name in {"ambiguous", "overlap", "low_spike", "high_dip"}:
            ambiguous_uniform_class = uniform_row.final_class
            ambiguous_flip_threshold_for_a = min_prior_a_for_a

    summary = PriorSensitivitySummary(
        trajectory_count=len(trajectories),
        sweep_count=len(sweep_rows),
        flipped_by_small_prior_fraction=flipped_by_small_prior / max(len(trajectories), 1),
        median_smallest_prior_shift_to_flip=(
            sorted(prior_shift_values)[len(prior_shift_values) // 2] if prior_shift_values else None
        ),
        median_smallest_log_prior_shift_to_flip=(
            sorted(log_prior_shift_values)[len(log_prior_shift_values) // 2] if log_prior_shift_values else None
        ),
        ambiguous_uniform_class=ambiguous_uniform_class,
        ambiguous_flip_threshold_for_a=ambiguous_flip_threshold_for_a,
    )
    metrics = {
        "method_name": method_name,
        "forgetting_factor": forgetting_factor,
        "confidence_threshold": confidence_threshold,
        "binary_classes": [class_a, class_b],
        "small_prior_delta": small_prior_delta,
        "fraction_flipped_by_small_prior_perturbation": summary.flipped_by_small_prior_fraction,
        "median_smallest_prior_shift_to_flip": summary.median_smallest_prior_shift_to_flip,
        "median_smallest_log_prior_shift_to_flip": summary.median_smallest_log_prior_shift_to_flip,
        "ambiguous_uniform_class": summary.ambiguous_uniform_class,
        "ambiguous_flip_threshold_for_a": summary.ambiguous_flip_threshold_for_a,
        "decomposition_note": (
            "For recursive Bayes with forgetting_factor=1.0, final log posterior odds equal cumulative log-likelihood ratio plus log prior odds."
        ),
    }
    return PriorSensitivityResult(
        method_name=method_name,
        class_names=class_names,
        trajectories=trajectories,
        sweep_rows=tuple(sweep_rows),
        flip_thresholds=tuple(flip_thresholds),
        summary=summary,
        prior_dominance_metrics=metrics,
    )


def analyze_prior_sensitivity(
    *,
    seed: int = 7,
    forgetting_factor: float = 1.0,
    confidence_threshold: float = 0.75,
    trajectories_per_class: int = 3,
    class_specs: tuple[AccumulatorClassSpec, ...] | None = None,
    prior_grid: tuple[float, ...] | None = None,
) -> PriorSensitivityResult:
    specs = class_specs or default_accumulator_class_specs()
    if len(specs) != 2:
        raise ValueError("prior sensitivity analysis currently expects exactly two classes")
    grid = prior_grid or _default_prior_grid()
    trajectories = tuple(
        sorted(generate_accumulator_trajectories(seed=seed, trajectories_per_class=trajectories_per_class), key=_trajectory_order_key)
    )

    def _runner(trajectory: AccumulatorTrajectory, prior: dict[str, float]) -> tuple[str, float, dict[str, float], float]:
        run = run_accumulator(
            trajectory,
            specs,
            forgetting_factor=forgetting_factor,
            confidence_threshold=confidence_threshold,
            prior=prior,
        )
        class_a = specs[0].name
        class_b = specs[1].name
        cumulative_log_likelihood_ratio = sum(
            step.log_likelihood_terms[class_a] - step.log_likelihood_terms[class_b] for step in run.steps
        )
        return run.final_predicted_class, run.final_confidence, run.final_weights, cumulative_log_likelihood_ratio

    return _analyze_generic_prior_sensitivity(
        method_name="accumulator",
        class_names=(specs[0].name, specs[1].name),
        trajectories=trajectories,
        runner=_runner,
        prior_grid=grid,
        confidence_threshold=confidence_threshold,
        forgetting_factor=forgetting_factor,
    )


def analyze_pointwise_prior_sensitivity(
    *,
    seed: int = 7,
    class_specs: tuple[PointwiseClassSpec, ...] | None = None,
    prior_grid: tuple[float, ...] | None = None,
) -> PriorSensitivityResult:
    specs = class_specs or default_pointwise_class_specs()
    if len(specs) != 2:
        raise ValueError("pointwise prior sensitivity currently expects exactly two classes")
    grid = prior_grid or _default_prior_grid()
    trajectories = tuple(sorted(generate_pointwise_benchmark_trajectories(seed=seed), key=_generic_trajectory_order_key))

    def _runner(trajectory: PointwiseTrajectory, prior: dict[str, float]) -> tuple[str, float, dict[str, float], float]:
        run = run_pointwise_classifier(trajectory, specs, prior=prior)
        class_a = specs[0].name
        class_b = specs[1].name
        cumulative_log_likelihood_ratio = sum(
            step.log_likelihood_terms[class_a] - step.log_likelihood_terms[class_b] for step in run.steps
        )
        confidence = max(run.final_weights.values())
        return run.final_predicted_class, confidence, run.final_weights, cumulative_log_likelihood_ratio

    return _analyze_generic_prior_sensitivity(
        method_name="pointwise",
        class_names=(specs[0].name, specs[1].name),
        trajectories=trajectories,
        runner=_runner,
        prior_grid=grid,
        confidence_threshold=0.0,
        forgetting_factor=1.0,
    )


def analyze_windowed_prior_sensitivity(
    *,
    seed: int = 7,
    class_specs: tuple[WindowedClassSpec, ...] | None = None,
    prior_grid: tuple[float, ...] | None = None,
    feature_mode: str = "raw",
    window_size: int = 5,
    trim_fraction: float = 0.2,
) -> PriorSensitivityResult:
    specs = class_specs or default_windowed_class_specs()
    if len(specs) != 2:
        raise ValueError("windowed prior sensitivity currently expects exactly two classes")
    grid = prior_grid or _default_prior_grid()
    trajectories = tuple(sorted(generate_windowed_trajectories(seed=seed), key=_generic_trajectory_order_key))

    def _runner(trajectory: WindowedTrajectory, prior: dict[str, float]) -> tuple[str, float, dict[str, float], float]:
        classifier = WindowedFeatureClassifier(specs, feature_mode=feature_mode, prior=prior)
        classifier.reset(prior)
        feature_rows = extract_windowed_feature_rows(trajectory, window_size=window_size, trim_fraction=trim_fraction)
        for row in feature_rows:
            classifier.update(row)
        history = classifier.history()
        final_weights = classifier.posterior()
        final_class = classifier.predict()
        confidence = max(final_weights.values())
        class_a = specs[0].name
        class_b = specs[1].name
        cumulative_log_likelihood_ratio = sum(
            step.log_likelihood_terms[class_a] - step.log_likelihood_terms[class_b] for step in history
        )
        return final_class, confidence, final_weights, cumulative_log_likelihood_ratio

    return _analyze_generic_prior_sensitivity(
        method_name=f"windowed_{feature_mode}",
        class_names=(specs[0].name, specs[1].name),
        trajectories=trajectories,
        runner=_runner,
        prior_grid=grid,
        confidence_threshold=0.0,
        forgetting_factor=1.0,
    )


def render_prior_sensitivity_report(result: PriorSensitivityResult) -> str:
    lines = [
        "# Prior Sensitivity and Bias Study",
        "",
        f"This analysis sweeps binary class priors over the `{result.method_name}` classifier and records how final posteriors, hard decisions, and confidence change as prior odds move.",
        "",
        "## Summary",
        "",
        f"- Trajectories analyzed: {result.summary.trajectory_count}",
        f"- Sweep rows: {result.summary.sweep_count}",
        f"- Fraction flipped by +/- {result.prior_dominance_metrics['small_prior_delta']:.2f} prior perturbation: {result.summary.flipped_by_small_prior_fraction:.3f}",
        f"- Median smallest prior shift to flip: {result.summary.median_smallest_prior_shift_to_flip if result.summary.median_smallest_prior_shift_to_flip is not None else 'n/a'}",
        f"- Median smallest log-prior shift to flip: {result.summary.median_smallest_log_prior_shift_to_flip if result.summary.median_smallest_log_prior_shift_to_flip is not None else 'n/a'}",
        f"- Ambiguous uniform-prior class: `{result.summary.ambiguous_uniform_class}`",
        f"- Ambiguous minimum prior_A for class A: {result.summary.ambiguous_flip_threshold_for_a if result.summary.ambiguous_flip_threshold_for_a is not None else 'n/a'}",
        "",
        "## Flip Thresholds",
        "",
        "| trajectory_id | scenario | uniform_class | uniform_confidence | min_prior_A_for_A | max_prior_A_for_B | smallest_shift_to_flip |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in result.flip_thresholds:
        lines.append(
            "| "
            f"{row.trajectory_id} | {row.scenario_name} | {row.uniform_prior_class} | {row.uniform_prior_confidence:.3f} | "
            f"{row.min_prior_a_for_a if row.min_prior_a_for_a is not None else 'n/a'} | "
            f"{row.max_prior_a_for_b if row.max_prior_a_for_b is not None else 'n/a'} | "
            f"{row.smallest_prior_shift_to_flip if row.smallest_prior_shift_to_flip is not None else 'n/a'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Easy trajectories should remain evidence-driven, so prior sweeps should not flip the final class within a moderate prior range.",
            "- Ambiguous trajectories should show prior-dominant regions where small prior changes alter the decision or move the run into abstain.",
            "- With forgetting factor 1.0, the log-posterior odds decomposition is exact: log posterior odds = cumulative log-likelihood ratio + log prior odds.",
        ]
    )
    return "\n".join(lines)


def _build_posterior_figure(result: PriorSensitivityResult):
    plt = _prepare_matplotlib()
    trajectory_ids = [getattr(trajectory, "trajectory_id") for trajectory in result.trajectories]
    selected_ids = trajectory_ids[:4]
    rows_by_trajectory: dict[str, list[PriorSweepRow]] = {trajectory_id: [] for trajectory_id in trajectory_ids}
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
    plt = _prepare_matplotlib()
    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    labels = [row.trajectory_id for row in result.flip_thresholds]
    values = [
        row.smallest_prior_shift_to_flip if row.smallest_prior_shift_to_flip is not None else 0.5
        for row in result.flip_thresholds
    ]
    colors = [
        "#d97706" if row.smallest_prior_shift_to_flip is not None else "#2563eb"
        for row in result.flip_thresholds
    ]
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
    plt = _prepare_matplotlib()
    trajectory_ids = [getattr(trajectory, "trajectory_id") for trajectory in result.trajectories]
    prior_values = sorted({row.prior_a for row in result.sweep_rows})
    matrix = []
    for trajectory_id in trajectory_ids:
        row_values = []
        for prior_a in prior_values:
            row = next(
                sweep_row
                for sweep_row in result.sweep_rows
                if sweep_row.trajectory_id == trajectory_id and abs(sweep_row.prior_a - prior_a) < 1e-9
            )
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
    plt = _prepare_matplotlib()
    trajectory_ids = [getattr(trajectory, "trajectory_id") for trajectory in result.trajectories]
    prior_values = sorted({row.prior_a for row in result.sweep_rows})
    class_a, class_b = result.class_names
    value_map = {class_b: 0.0, "unknown": 0.5, class_a: 1.0}
    matrix = []
    for trajectory_id in trajectory_ids:
        row_values = []
        for prior_a in prior_values:
            row = next(
                sweep_row
                for sweep_row in result.sweep_rows
                if sweep_row.trajectory_id == trajectory_id and abs(sweep_row.prior_a - prior_a) < 1e-9
            )
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
    plt = _prepare_matplotlib()
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    selected_ids = ["easy_A_0", "ambiguous_mid", "late_flip"]
    colors = {"easy_A_0": "#2563eb", "ambiguous_mid": "#d97706", "late_flip": "#7c3aed"}
    for trajectory_id in selected_ids:
        rows = sorted(
            [row for row in result.sweep_rows if row.trajectory_id == trajectory_id],
            key=lambda row: row.log_prior_odds,
        )
        ax.plot(
            [row.log_prior_odds for row in rows],
            [row.final_log_posterior_odds for row in rows],
            color=colors[trajectory_id],
            linewidth=2.2,
            label=trajectory_id,
        )
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
    plt = _prepare_matplotlib()
    pair_label = f"{result.class_names[0]}_vs_{result.class_names[1]}"
    trajectory_ids = [row.trajectory_id for row in result.flip_thresholds]
    matrix = [
        [row.smallest_prior_shift_to_flip if row.smallest_prior_shift_to_flip is not None else 0.5]
        for row in result.flip_thresholds
    ]
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


def render_prior_sensitivity_posterior_svg(result: PriorSensitivityResult) -> str:
    plt = _prepare_matplotlib()
    fig = _build_posterior_figure(result)
    buffer = io.StringIO()
    try:
        fig.savefig(buffer, format="svg", bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def render_prior_sensitivity_posterior_png_bytes(result: PriorSensitivityResult) -> bytes:
    plt = _prepare_matplotlib()
    fig = _build_posterior_figure(result)
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def render_prior_sensitivity_flip_svg(result: PriorSensitivityResult) -> str:
    plt = _prepare_matplotlib()
    fig = _build_flip_figure(result)
    buffer = io.StringIO()
    try:
        fig.savefig(buffer, format="svg", bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def render_prior_sensitivity_flip_png_bytes(result: PriorSensitivityResult) -> bytes:
    plt = _prepare_matplotlib()
    fig = _build_flip_figure(result)
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def render_prior_sensitivity_heatmap_svg(result: PriorSensitivityResult) -> str:
    plt = _prepare_matplotlib()
    fig = _build_heatmap_figure(result)
    buffer = io.StringIO()
    try:
        fig.savefig(buffer, format="svg", bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def render_prior_sensitivity_heatmap_png_bytes(result: PriorSensitivityResult) -> bytes:
    plt = _prepare_matplotlib()
    fig = _build_heatmap_figure(result)
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def render_prior_sensitivity_decision_svg(result: PriorSensitivityResult) -> str:
    plt = _prepare_matplotlib()
    fig = _build_decision_map_figure(result)
    buffer = io.StringIO()
    try:
        fig.savefig(buffer, format="svg", bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def render_prior_sensitivity_decision_png_bytes(result: PriorSensitivityResult) -> bytes:
    plt = _prepare_matplotlib()
    fig = _build_decision_map_figure(result)
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def render_prior_sensitivity_decomposition_svg(result: PriorSensitivityResult) -> str:
    plt = _prepare_matplotlib()
    fig = _build_decomposition_figure(result)
    buffer = io.StringIO()
    try:
        fig.savefig(buffer, format="svg", bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def render_prior_sensitivity_decomposition_png_bytes(result: PriorSensitivityResult) -> bytes:
    plt = _prepare_matplotlib()
    fig = _build_decomposition_figure(result)
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def render_prior_sensitivity_pairwise_flip_svg(result: PriorSensitivityResult) -> str:
    plt = _prepare_matplotlib()
    fig = _build_pairwise_flip_heatmap_figure(result)
    buffer = io.StringIO()
    try:
        fig.savefig(buffer, format="svg", bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def render_prior_sensitivity_pairwise_flip_png_bytes(result: PriorSensitivityResult) -> bytes:
    plt = _prepare_matplotlib()
    fig = _build_pairwise_flip_heatmap_figure(result)
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def analyze_cross_method_prior_comparison(
    *,
    seed: int = 7,
) -> CrossMethodPriorComparisonResult:
    method_results = (
        analyze_prior_sensitivity(seed=seed),
        analyze_pointwise_prior_sensitivity(seed=seed),
        analyze_windowed_prior_sensitivity(seed=seed, feature_mode="raw"),
        analyze_windowed_prior_sensitivity(seed=seed, feature_mode="robust"),
    )
    scenario_names = ("easy", "boundary", "outlier", "transition", "long_history")
    scenario_family_map = {
        "easy": {"easy", "low_clean", "high_clean"},
        "boundary": {"ambiguous", "overlap"},
        "outlier": {"low_spike", "high_dip"},
        "transition": {"late_flip"},
        "long_history": {"low_long", "high_long"},
    }
    rows: list[dict[str, object]] = []
    for result in method_results:
        row = {"method_name": result.method_name}
        for scenario_name in scenario_names:
            matching = [
                item
                for item in result.flip_thresholds
                if item.scenario_name in scenario_family_map[scenario_name]
            ]
            if not matching:
                row[scenario_name] = None
                row[f"{scenario_name}_status"] = "missing"
                continue
            values = [
                item.smallest_prior_shift_to_flip
                for item in matching
                if item.smallest_prior_shift_to_flip is not None
            ]
            if values:
                row[scenario_name] = min(values)
                row[f"{scenario_name}_status"] = "flips"
            else:
                row[scenario_name] = 0.50
                row[f"{scenario_name}_status"] = "stable"
        row["fraction_flipped_by_small_prior_perturbation"] = result.summary.flipped_by_small_prior_fraction
        rows.append(row)
    return CrossMethodPriorComparisonResult(
        method_results=method_results,
        scenario_names=scenario_names,
        rows=tuple(rows),
    )


def render_cross_method_prior_comparison_report(result: CrossMethodPriorComparisonResult) -> str:
    lines = [
        "# Cross-Method Prior Sensitivity Comparison",
        "",
        "This artifact compares smallest prior shifts needed to flip the final decision across the current baseline methods. Lower values mean higher prior fragility. `stable` means the scenario family exists for that method but no flip was observed anywhere in the swept prior range. `n/a` means that scenario family is not represented for that method.",
        "",
        "## Scenario Flip Thresholds",
        "",
        "| method | " + " | ".join(result.scenario_names) + " | small-prior-flip-fraction |",
        "| --- | " + " | ".join("---:" for _ in result.scenario_names) + " | ---: |",
    ]
    for row in result.rows:
        threshold_cells = []
        for scenario_name in result.scenario_names:
            status = row[f"{scenario_name}_status"]
            if status == "missing":
                threshold_cells.append("n/a")
            elif status == "stable":
                threshold_cells.append("stable")
            else:
                threshold_cells.append(f"{row[scenario_name]:.2f}")
        lines.append(
            f"| {row['method_name']} | " + " | ".join(threshold_cells) + f" | {row['fraction_flipped_by_small_prior_perturbation']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Lower threshold means a smaller prior change can flip the final decision.",
            "- `stable` means the scenario family is present but no flip was observed within the swept prior range.",
            "- `n/a` means that scenario family is not represented for that method.",
            "- The final column summarizes how often each method flipped under the configured small prior perturbation.",
        ]
    )
    return "\n".join(lines)


def _build_cross_method_prior_comparison_figure(result: CrossMethodPriorComparisonResult):
    plt = _prepare_matplotlib()
    matrix = []
    for row in result.rows:
        matrix.append([
            float("nan") if row[f"{scenario_name}_status"] == "missing" else float(row[scenario_name])
            for scenario_name in result.scenario_names
        ])
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
            if status == "missing":
                label = "n/a"
            elif status == "stable":
                label = "stable"
            else:
                label = f"{row[scenario_name]:.2f}"
            ax.text(col_index, row_index, label, ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="minimum |prior shift| to flip")
    fig.tight_layout()
    return fig


def render_cross_method_prior_comparison_svg(result: CrossMethodPriorComparisonResult) -> str:
    plt = _prepare_matplotlib()
    fig = _build_cross_method_prior_comparison_figure(result)
    buffer = io.StringIO()
    try:
        fig.savefig(buffer, format="svg", bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def render_cross_method_prior_comparison_png_bytes(result: CrossMethodPriorComparisonResult) -> bytes:
    plt = _prepare_matplotlib()
    fig = _build_cross_method_prior_comparison_figure(result)
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def write_cross_method_prior_comparison_artifacts(
    output_dir: str | Path,
    *,
    seed: int = 7,
    result: CrossMethodPriorComparisonResult | None = None,
) -> CrossMethodPriorComparisonArtifacts:
    analysis = result or analyze_cross_method_prior_comparison(seed=seed)
    output_root = Path(output_dir)
    run_dir = output_root / "prior_sensitivity_cross_method_v1"
    run_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / "cross_method_prior_comparison_report.md"
    comparison_csv_path = run_dir / "cross_method_prior_comparison.csv"
    status_csv_path = run_dir / "cross_method_prior_comparison_status.csv"
    plot_svg_path = run_dir / "cross_method_prior_fragility_heatmap.svg"
    plot_png_path = run_dir / "cross_method_prior_fragility_heatmap.png"

    report_path.write_text(render_cross_method_prior_comparison_report(analysis), encoding="utf-8")
    _write_csv(
        comparison_csv_path,
        [
            {
                "method_name": row["method_name"],
                **{scenario_name: row[scenario_name] for scenario_name in analysis.scenario_names},
                "fraction_flipped_by_small_prior_perturbation": row["fraction_flipped_by_small_prior_perturbation"],
            }
            for row in analysis.rows
        ],
        ["method_name", *analysis.scenario_names, "fraction_flipped_by_small_prior_perturbation"],
    )
    _write_csv(
        status_csv_path,
        [
            {
                "method_name": row["method_name"],
                **{scenario_name: row[f"{scenario_name}_status"] for scenario_name in analysis.scenario_names},
            }
            for row in analysis.rows
        ],
        ["method_name", *analysis.scenario_names],
    )
    plot_svg_path.write_text(render_cross_method_prior_comparison_svg(analysis), encoding="utf-8")
    plot_png_path.write_bytes(render_cross_method_prior_comparison_png_bytes(analysis))
    return CrossMethodPriorComparisonArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        comparison_csv_path=comparison_csv_path,
        status_csv_path=status_csv_path,
        plot_svg_path=plot_svg_path,
        plot_png_path=plot_png_path,
    )


def write_prior_sensitivity_artifacts(
    output_dir: str | Path,
    *,
    seed: int = 7,
    forgetting_factor: float = 1.0,
    confidence_threshold: float = 0.75,
    trajectories_per_class: int = 3,
    result: PriorSensitivityResult | None = None,
) -> PriorSensitivityArtifacts:
    analysis = result or analyze_prior_sensitivity(
        seed=seed,
        forgetting_factor=forgetting_factor,
        confidence_threshold=confidence_threshold,
        trajectories_per_class=trajectories_per_class,
    )
    output_root = Path(output_dir)
    run_dir_name = "prior_sensitivity_v1" if analysis.method_name == "accumulator" else f"prior_sensitivity_{analysis.method_name}_v1"
    run_dir = output_root / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / "prior_sensitivity_report.md"
    sweep_path = run_dir / "prior_sensitivity.csv"
    flip_thresholds_path = run_dir / "prior_flip_thresholds.csv"
    metrics_path = run_dir / "prior_dominance_metrics.json"
    config_path = run_dir / "prior_sensitivity_config.yaml"
    plot_posterior_svg_path = run_dir / "posterior_vs_prior.svg"
    plot_posterior_png_path = run_dir / "posterior_vs_prior.png"
    plot_flip_svg_path = run_dir / "decision_flip_thresholds.svg"
    plot_flip_png_path = run_dir / "decision_flip_thresholds.png"
    plot_heatmap_svg_path = run_dir / "prior_dominance_heatmap.svg"
    plot_heatmap_png_path = run_dir / "prior_dominance_heatmap.png"
    plot_decision_svg_path = run_dir / "prior_decision_map.svg"
    plot_decision_png_path = run_dir / "prior_decision_map.png"
    plot_decomposition_svg_path = run_dir / "log_odds_decomposition.svg"
    plot_decomposition_png_path = run_dir / "log_odds_decomposition.png"
    plot_pairwise_flip_svg_path = run_dir / "pairwise_flip_threshold_heatmap.svg"
    plot_pairwise_flip_png_path = run_dir / "pairwise_flip_threshold_heatmap.png"

    report_path.write_text(render_prior_sensitivity_report(analysis), encoding="utf-8")
    sweep_path.write_text("", encoding="utf-8")
    flip_thresholds_path.write_text("", encoding="utf-8")
    metrics_path.write_text(json.dumps(analysis.prior_dominance_metrics, indent=2, sort_keys=True), encoding="utf-8")
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  name: prior_sensitivity_v1",
                f"  seed: {seed}",
                "classifier:",
                "  type: sequential_bayes_accumulator",
                f"  forgetting_factor: {forgetting_factor}",
                f"  confidence_threshold: {confidence_threshold}",
                "evaluation:",
                "  prior_grid: [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    plot_posterior_svg_path.write_text(render_prior_sensitivity_posterior_svg(analysis), encoding="utf-8")
    plot_posterior_png_path.write_bytes(render_prior_sensitivity_posterior_png_bytes(analysis))
    plot_flip_svg_path.write_text(render_prior_sensitivity_flip_svg(analysis), encoding="utf-8")
    plot_flip_png_path.write_bytes(render_prior_sensitivity_flip_png_bytes(analysis))
    plot_heatmap_svg_path.write_text(render_prior_sensitivity_heatmap_svg(analysis), encoding="utf-8")
    plot_heatmap_png_path.write_bytes(render_prior_sensitivity_heatmap_png_bytes(analysis))
    plot_decision_svg_path.write_text(render_prior_sensitivity_decision_svg(analysis), encoding="utf-8")
    plot_decision_png_path.write_bytes(render_prior_sensitivity_decision_png_bytes(analysis))
    plot_decomposition_svg_path.write_text(render_prior_sensitivity_decomposition_svg(analysis), encoding="utf-8")
    plot_decomposition_png_path.write_bytes(render_prior_sensitivity_decomposition_png_bytes(analysis))
    plot_pairwise_flip_svg_path.write_text(render_prior_sensitivity_pairwise_flip_svg(analysis), encoding="utf-8")
    plot_pairwise_flip_png_path.write_bytes(render_prior_sensitivity_pairwise_flip_png_bytes(analysis))

    sweep_rows = [asdict(row) for row in analysis.sweep_rows]
    flip_rows = [asdict(row) for row in analysis.flip_thresholds]
    _write_csv(
        sweep_path,
        sweep_rows,
        [
            "trajectory_id",
            "scenario_name",
            "true_class",
            "prior_a",
            "prior_b",
            "log_prior_odds",
            "final_class",
            "final_confidence",
            "abstained",
            "posterior_a",
            "posterior_b",
            "final_log_posterior_odds",
            "cumulative_log_likelihood_ratio",
        ],
    )
    _write_csv(
        flip_thresholds_path,
        flip_rows,
        [
            "trajectory_id",
            "scenario_name",
            "true_class",
            "uniform_prior_class",
            "uniform_prior_confidence",
            "min_prior_a_for_a",
            "max_prior_a_for_b",
            "smallest_prior_shift_to_flip",
            "smallest_log_prior_shift_to_flip",
        ],
    )

    return PriorSensitivityArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        sweep_path=sweep_path,
        flip_thresholds_path=flip_thresholds_path,
        metrics_path=metrics_path,
        config_path=config_path,
        plot_posterior_svg_path=plot_posterior_svg_path,
        plot_posterior_png_path=plot_posterior_png_path,
        plot_flip_svg_path=plot_flip_svg_path,
        plot_flip_png_path=plot_flip_png_path,
        plot_heatmap_svg_path=plot_heatmap_svg_path,
        plot_heatmap_png_path=plot_heatmap_png_path,
        plot_decision_svg_path=plot_decision_svg_path,
        plot_decision_png_path=plot_decision_png_path,
        plot_decomposition_svg_path=plot_decomposition_svg_path,
        plot_decomposition_png_path=plot_decomposition_png_path,
        plot_pairwise_flip_svg_path=plot_pairwise_flip_svg_path,
        plot_pairwise_flip_png_path=plot_pairwise_flip_png_path,
    )
