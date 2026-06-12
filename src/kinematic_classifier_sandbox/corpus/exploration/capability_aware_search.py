from __future__ import annotations

import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from kinematic_classifier_sandbox.reports.markdown import MarkdownDocument
from kinematic_classifier_sandbox.utils.io import write_csv

from ...utils.plotting import plt
from ..trajectory_backend_contract import (
    TrajectoryBackendCapabilities,
    default_backend_contract_definitions,
)


@dataclass(frozen=True, slots=True)
class CapabilityAwareSearchResult:
    search_planner_rules: dict[str, Any]
    selection_matrix_rows: tuple[dict[str, Any], ...]
    backend_plan_rows: tuple[dict[str, Any], ...]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class CapabilityAwareSearchArtifacts:
    run_dir: Path
    search_planner_rules_path: Path
    search_method_selection_matrix_path: Path
    backend_search_plan_path: Path
    report_path: Path
    selection_matrix_png_path: Path
    decision_tree_png_path: Path
    cost_coverage_frontier_png_path: Path


def _current_and_exemplar_capabilities() -> tuple[TrajectoryBackendCapabilities, ...]:
    current = tuple(definition.capabilities for definition in default_backend_contract_definitions())
    exemplars = (
        TrajectoryBackendCapabilities(
            backend_id="future_3plus3_backend",
            display_name="Future 3+3 Translational Backend",
            family="future_3plus3_backend",
            dimensionality="3d",
            fidelity="mid",
            input_modes=("design_variables", "control_schedule"),
            supports_environment=True,
            supports_sequential_control=False,
            supports_events=True,
            supports_stochastic_runs=True,
            runtime_class="medium",
            determinism="seeded",
            state_outputs=("position", "velocity", "acceleration"),
            observation_outputs=("position", "velocity"),
            event_outputs=("phase_change", "termination"),
            valid_search_methods=("sobol", "quality_diversity", "adaptive_stress"),
        ),
        TrajectoryBackendCapabilities(
            backend_id="future_6dof_backend",
            display_name="Future 6DOF Backend",
            family="future_6dof_backend",
            dimensionality="6dof",
            fidelity="high",
            input_modes=("input_deck", "design_variables", "control_schedule"),
            supports_environment=True,
            supports_sequential_control=True,
            supports_events=True,
            supports_stochastic_runs=False,
            runtime_class="expensive",
            determinism="deterministic",
            state_outputs=("position", "velocity", "attitude", "angular_rate", "mass"),
            observation_outputs=("position", "velocity", "sensor_measurements"),
            event_outputs=("phase_change", "constraint_violation", "termination"),
            valid_search_methods=("lhs", "surrogate_assisted", "active_learning", "quality_diversity"),
        ),
    )
    return current + exemplars


def _planner_rules() -> dict[str, Any]:
    return {
        "planner_version": "m34_v1",
        "runtime_rules": {
            "cheap": {
                "default": ["random", "lhs", "sobol", "quality_diversity"],
                "sequential_bonus": ["adaptive_stress", "cross_entropy"],
            },
            "medium": {
                "default": ["lhs", "sobol", "quality_diversity"],
                "file_backend_bonus": ["budgeted_doe", "surrogate_assisted"],
            },
            "expensive": {
                "default": ["small_doe", "surrogate_assisted", "active_learning"],
                "forbidden": ["broad_random_sweep", "large_qd_budget"],
            },
        },
        "environment_rules": {
            "supports_environment": ["leakage_aware_search", "environment_regime_targeting"],
            "no_environment": ["environment_regime_targeting_disabled"],
        },
        "sequential_control_rules": {
            "enabled": ["adaptive_stress", "cross_entropy", "future_rl_candidate"],
            "disabled": ["adaptive_stress_disabled", "future_rl_candidate_disabled"],
        },
        "stochastic_rules": {
            "stochastic_backend": ["repeat_seeded_sampling", "variance_estimation"],
            "deterministic_backend": ["single_pass_screening", "cache_priority"],
        },
    }


def _recommended_methods(capabilities: TrajectoryBackendCapabilities) -> tuple[list[str], list[str], str]:
    methods: list[str] = []
    reasons: list[str] = []
    rules = _planner_rules()

    runtime_rule = rules["runtime_rules"][capabilities.runtime_class]
    methods.extend(runtime_rule["default"])
    reasons.append(f"runtime_class={capabilities.runtime_class}")

    if capabilities.runtime_class == "cheap" and capabilities.supports_sequential_control:
        methods.extend(runtime_rule["sequential_bonus"])
        reasons.append("cheap_sequential_control")
    if capabilities.runtime_class == "medium" and "input_deck" in capabilities.input_modes:
        methods.extend(runtime_rule["file_backend_bonus"])
        reasons.append("file_backend_execution")
    if capabilities.runtime_class == "expensive":
        reasons.append("broad_search_avoided")

    if capabilities.supports_environment:
        methods.extend(rules["environment_rules"]["supports_environment"])
        reasons.append("environment_trace_available")
    else:
        reasons.append("environment_methods_not_needed")

    if capabilities.supports_sequential_control:
        methods.extend(rules["sequential_control_rules"]["enabled"])
        reasons.append("sequential_control_enabled")
    else:
        reasons.append("sequential_control_disabled")

    if capabilities.supports_stochastic_runs:
        methods.extend(rules["stochastic_rules"]["stochastic_backend"])
        reasons.append("stochastic_sampling_supported")
    else:
        methods.extend(rules["stochastic_rules"]["deterministic_backend"])
        reasons.append("deterministic_cache_priority")

    unique_methods = []
    for method in methods:
        if method not in unique_methods:
            unique_methods.append(method)

    budget_class = "broad" if capabilities.runtime_class == "cheap" else "budgeted" if capabilities.runtime_class == "medium" else "strict"
    return unique_methods, reasons, budget_class


def _selection_matrix_rows(capabilities_list: tuple[TrajectoryBackendCapabilities, ...]) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for capabilities in capabilities_list:
        methods, reasons, budget_class = _recommended_methods(capabilities)
        rows.append(
            {
                "backend_type": capabilities.family,
                "runtime_class": capabilities.runtime_class,
                "sequential_controls": capabilities.supports_sequential_control,
                "stochastic": capabilities.supports_stochastic_runs,
                "supports_environment": capabilities.supports_environment,
                "recommended_search": ", ".join(methods),
                "budget_class": budget_class,
                "rationale": "; ".join(reasons),
            }
        )
    return tuple(rows)


def _backend_plan_rows(capabilities_list: tuple[TrajectoryBackendCapabilities, ...]) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for capabilities in capabilities_list:
        methods, reasons, budget_class = _recommended_methods(capabilities)
        rows.append(
            {
                "backend_id": capabilities.backend_id,
                "family": capabilities.family,
                "runtime_class": capabilities.runtime_class,
                "dimensionality": capabilities.dimensionality,
                "supports_environment": capabilities.supports_environment,
                "supports_sequential_control": capabilities.supports_sequential_control,
                "supports_stochastic_runs": capabilities.supports_stochastic_runs,
                "search_budget_class": budget_class,
                "sequential_methods_enabled": capabilities.supports_sequential_control,
                "broad_expensive_search_avoided": capabilities.runtime_class == "expensive",
                "recommended_methods": ", ".join(methods),
                "planner_reasons": "; ".join(reasons),
            }
        )
    return tuple(rows)


def _render_selection_matrix_png(rows: tuple[dict[str, Any], ...]) -> bytes:
    metrics = ("supports_environment", "sequential_controls", "stochastic")
    data = [[1.0 if bool(row[metric]) else 0.0 for metric in metrics] for row in rows]
    labels = [str(row["backend_type"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    image = ax.imshow(data, cmap="Blues", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(metrics)), labels=[metric.replace("_", "\n") for metric in metrics], fontsize=8)
    ax.set_yticks(range(len(labels)), labels=labels, fontsize=8)
    ax.set_title("Search Selection Matrix")
    for row_index, row_values in enumerate(data):
        for column_index, value in enumerate(row_values):
            ax.text(column_index, row_index, f"{value:.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.04)
    fig.tight_layout()


    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def _render_decision_tree_png() -> bytes:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.axis("off")
    boxes = [
        (0.5, 0.92, "runtime_class?"),
        (0.2, 0.70, "cheap\nrandom + LHS + Sobol + QD"),
        (0.5, 0.70, "medium\nbudgeted DOE + QD + cache"),
        (0.8, 0.70, "expensive\nsmall DOE + surrogate + active learning"),
        (0.2, 0.42, "sequential?\nadd adaptive stress / CEM"),
        (0.5, 0.42, "environment?\nadd leakage-aware search"),
        (0.8, 0.42, "deterministic?\ncache-priority execution"),
    ]
    for x, y, label in boxes:
        ax.text(x, y, label, ha="center", va="center", fontsize=9, bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f2f0e6", "edgecolor": "#3f4f4f"})
    arrows = [
        ((0.5, 0.88), (0.2, 0.74)),
        ((0.5, 0.88), (0.5, 0.74)),
        ((0.5, 0.88), (0.8, 0.74)),
        ((0.2, 0.66), (0.2, 0.46)),
        ((0.5, 0.66), (0.5, 0.46)),
        ((0.8, 0.66), (0.8, 0.46)),
    ]
    for (x0, y0), (x1, y1) in arrows:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "->", "linewidth": 1.2})
    fig.tight_layout()


    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def _render_cost_coverage_frontier_png(rows: tuple[dict[str, Any], ...]) -> bytes:
    runtime_to_cost = {"cheap": 1.0, "medium": 2.0, "expensive": 3.2}
    budget_to_coverage = {"broad": 0.85, "budgeted": 0.62, "strict": 0.42}
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for row in rows:
        x_value = runtime_to_cost[str(row["runtime_class"])]
        y_value = budget_to_coverage[str(row["search_budget_class"])]
        ax.scatter(x_value, y_value, s=80)
        ax.text(x_value + 0.03, y_value + 0.01, str(row["family"]), fontsize=8)
    ax.set_xlabel("Relative Runtime Cost")
    ax.set_ylabel("Projected Coverage Breadth")
    ax.set_title("Projected Cost vs Coverage Frontier")
    ax.grid(alpha=0.25)
    fig.tight_layout()


    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def analyze_capability_aware_search() -> CapabilityAwareSearchResult:
    capabilities_list = _current_and_exemplar_capabilities()
    selection_rows = _selection_matrix_rows(capabilities_list)
    backend_plan_rows = _backend_plan_rows(capabilities_list)

    doc = MarkdownDocument("Capability-Aware Search Method Selection")
    doc.heading("Summary", level=2)
    doc.bullet_list(
        [
            f"backend profiles planned: `{len(capabilities_list)}`",
            f"current implemented backends covered: `{len(default_backend_contract_definitions())}`",
            f"expensive exemplar profiles included: `{sum(1 for item in capabilities_list if item.runtime_class == 'expensive')}`",
        ]
    )

    doc.heading("Search Method Selection Matrix", level=2)
    doc.table(
        ["Backend", "Runtime", "Sequential", "Environment", "Recommended Search"],
        [
            (
                f"`{row['backend_type']}`",
                f"`{row['runtime_class']}`",
                str(row['sequential_controls']),
                str(row['supports_environment']),
                f"`{row['recommended_search']}`",
            )
            for row in selection_rows
        ]
    )

    doc.heading("Notes", level=2)
    doc.bullet_list(
        [
            "Cheap parameter-only backends receive broad search plans and explicitly do not enable sequential-control search methods.",
            "Sequential-control backends gain adaptive stress and cross-entropy style methods because they can express time-varying behavior.",
            "Expensive exemplar profiles are included so the planner can prove it avoids broad brute-force search even before a real high-fidelity backend is integrated.",
        ]
    )

    return CapabilityAwareSearchResult(
        search_planner_rules=_planner_rules(),
        selection_matrix_rows=selection_rows,
        backend_plan_rows=backend_plan_rows,
        report_markdown=doc.text(),
    )


def write_capability_aware_search_artifacts(
    base_dir: str | Path,
    *,
    result: CapabilityAwareSearchResult | None = None,
) -> CapabilityAwareSearchArtifacts:
    run_dir = Path(base_dir) / "capability_aware_search"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = result or analyze_capability_aware_search()

    search_planner_rules_path = run_dir / "search_planner_rules.json"
    search_method_selection_matrix_path = run_dir / "search_method_selection_matrix.csv"
    backend_search_plan_path = run_dir / "backend_search_plan.csv"
    report_path = run_dir / "search_method_selection_report.md"
    selection_matrix_png_path = run_dir / "search_method_selection_matrix.png"
    decision_tree_png_path = run_dir / "backend_search_strategy_decision_tree.png"
    cost_coverage_frontier_png_path = run_dir / "projected_cost_coverage_frontier.png"

    search_planner_rules_path.write_text(json.dumps(payload.search_planner_rules, indent=2), encoding="utf-8")
    report_path.write_text(payload.report_markdown, encoding="utf-8")

    selection_fieldnames = list(payload.selection_matrix_rows[0].keys()) if payload.selection_matrix_rows else []
    backend_fieldnames = list(payload.backend_plan_rows[0].keys()) if payload.backend_plan_rows else []
    write_csv(search_method_selection_matrix_path, list(payload.selection_matrix_rows), selection_fieldnames)
    write_csv(backend_search_plan_path, list(payload.backend_plan_rows), backend_fieldnames)

    selection_matrix_png_path.write_bytes(_render_selection_matrix_png(payload.selection_matrix_rows))
    decision_tree_png_path.write_bytes(_render_decision_tree_png())
    cost_coverage_frontier_png_path.write_bytes(_render_cost_coverage_frontier_png(payload.backend_plan_rows))

    return CapabilityAwareSearchArtifacts(
        run_dir=run_dir,
        search_planner_rules_path=search_planner_rules_path,
        search_method_selection_matrix_path=search_method_selection_matrix_path,
        backend_search_plan_path=backend_search_plan_path,
        report_path=report_path,
        selection_matrix_png_path=selection_matrix_png_path,
        decision_tree_png_path=decision_tree_png_path,
        cost_coverage_frontier_png_path=cost_coverage_frontier_png_path,
    )
