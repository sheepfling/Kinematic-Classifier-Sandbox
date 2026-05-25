from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path

from .formal_math_registry import load_equation_registry


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class StrictEquationAuditRow:
    equation_id: str
    registry_status: str
    strict_label: str
    implementation: str
    exact_artifacts: tuple[str, ...]
    source_data: tuple[str, ...]
    notes: str


@dataclass(frozen=True, slots=True)
class StrictEquationAuditResult:
    rows: tuple[StrictEquationAuditRow, ...]
    summary: dict[str, object]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class StrictEquationAuditArtifacts:
    run_dir: Path
    report_path: Path
    summary_path: Path
    rows_path: Path
    status_plot_path: Path


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


IMPLEMENTED_SOURCE_DATA: dict[str, tuple[str, ...]] = {
    "bayes_logsumexp_update": (
        "artifacts/corpus_classifier_scoring/posterior_history.csv",
        "artifacts/bayesian_walkthroughs/bayesian_step_tables.csv",
    ),
    "two_class_log_odds": (
        "artifacts/bayesian_walkthroughs/posterior_flip_thresholds.csv",
        "artifacts/bayesian_walkthroughs/bayesian_step_tables.csv",
    ),
    "transition_matrix_update": (
        "artifacts/transition_matrix_accumulator_v1/posterior_history.csv",
        "artifacts/transition_matrix_accumulator_v1/scenario_summary.csv",
        "artifacts/transition_matrix_accumulator_v1/transition_matrix_numeric_walkthrough.md",
    ),
    "kalman_innovation_likelihood": (
        "artifacts/corpus_classifier_scoring/classifier_candidate_scores.csv",
        "artifacts/corpus_classifier_scoring/posterior_history.csv",
    ),
    "corpus_autodevelopment_score": (
        "artifacts/corpus_autodevelopment_v1/candidate_scores.csv",
        "artifacts/corpus_autodevelopment_v1/corpus_autodevelopment_numeric_walkthrough.md",
    ),
    "pareto_dominance": (
        "artifacts/corpus_autodevelopment_v1/pareto_front.csv",
        "artifacts/corpus_autodevelopment_v1/corpus_autodevelopment_report.md",
    ),
    "corpus_gym_reward": (
        "artifacts/corpus_gym/corpus_gym_report.md",
        "artifacts/corpus_gym/corpus_gym_numeric_walkthrough.md",
    ),
    "corpus_explorer_utility": (
        "artifacts/generic_corpus_exploration/generic_corpus_exploration_report.md",
        "artifacts/generic_corpus_exploration/generic_corpus_explorer_numeric_walkthrough.md",
    ),
    "qd_archive_utility": (
        "artifacts/quality_diversity_corpus_v1/archive_elites.csv",
        "artifacts/quality_diversity_corpus_v1/archive_cells.csv",
    ),
    "qd_cell_mapping": (
        "artifacts/quality_diversity_corpus_v1/archive_cells.csv",
    ),
    "sampler_mixture": (
        "artifacts/candidate_generation/generated_candidates.csv",
    ),
    "class_validity_status": (
        "artifacts/class_validity/class_validity_scores.csv",
    ),
    "advanced_filter_gate": (
        "artifacts/advanced_filter_decision_v1/advanced_filter_decision_report.md",
        "artifacts/advanced_filter_decision_v1/advanced_filter_decision_numeric_walkthrough.md",
    ),
}


IMPLEMENTED_EXACT_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "bayes_logsumexp_update": (
        "artifacts/corpus_classifier_scoring/posterior_history.csv",
        "artifacts/bayesian_walkthroughs/bayesian_step_tables.csv",
    ),
    "two_class_log_odds": (
        "artifacts/bayesian_walkthroughs/posterior_flip_thresholds.csv",
        "artifacts/bayesian_walkthroughs/bayesian_step_tables.csv",
    ),
    "transition_matrix_update": (
        "artifacts/transition_matrix_accumulator_v1/transition_matrix_accumulator_report.md",
        "artifacts/transition_matrix_accumulator_v1/transition_matrix_numeric_walkthrough.md",
    ),
    "kalman_innovation_likelihood": (
        "artifacts/corpus_classifier_scoring/classifier_candidate_scores.csv",
        "artifacts/corpus_classifier_scoring/posterior_history.csv",
    ),
    "corpus_autodevelopment_score": (
        "artifacts/corpus_autodevelopment_v1/candidate_scores.csv",
        "artifacts/corpus_autodevelopment_v1/corpus_autodevelopment_numeric_walkthrough.md",
    ),
    "pareto_dominance": (
        "artifacts/corpus_autodevelopment_v1/pareto_front.csv",
        "artifacts/corpus_autodevelopment_v1/corpus_autodevelopment_report.md",
    ),
    "corpus_gym_reward": (
        "artifacts/corpus_gym/corpus_gym_report.md",
        "artifacts/corpus_gym/corpus_gym_numeric_walkthrough.md",
    ),
    "corpus_explorer_utility": (
        "artifacts/generic_corpus_exploration/generic_corpus_exploration_report.md",
        "artifacts/generic_corpus_exploration/generic_corpus_explorer_numeric_walkthrough.md",
    ),
    "qd_archive_utility": (
        "artifacts/quality_diversity_corpus_v1/archive_elites.csv",
        "artifacts/quality_diversity_corpus_v1/archive_cells.csv",
    ),
    "qd_cell_mapping": (
        "artifacts/quality_diversity_corpus_v1/archive_cells.csv",
    ),
    "sampler_mixture": (
        "artifacts/candidate_generation/generated_candidates.csv",
    ),
    "class_validity_status": (
        "artifacts/class_validity/class_validity_scores.csv",
    ),
    "advanced_filter_gate": (
        "artifacts/advanced_filter_decision_v1/advanced_filter_decision_report.md",
        "artifacts/advanced_filter_decision_v1/advanced_filter_decision_numeric_walkthrough.md",
    ),
}


ILLUSTRATIVE_SOURCE_DATA: dict[str, tuple[str, ...]] = {
    "gaussian_feature_likelihood": (
        "artifacts/generated_corpus_features/feature_matrix.csv",
        "artifacts/generated_corpus_features/selected_record_manifest.csv",
        "artifacts/formal_math_visual_registry_v1/assets/gaussian_feature_likelihood.png",
    ),
    "calibration_metrics": (
        "artifacts/monte_carlo_accumulator/calibration_bins.csv",
        "artifacts/monte_carlo_accumulator/calibration_curve.png",
        "artifacts/formal_math_visual_registry_v1/assets/calibration_metrics.png",
    ),
    "pairwise_mahalanobis_distance": (
        "artifacts/feature_analysis_v1/pairwise_distance_matrix.csv",
        "artifacts/feature_analysis_v1/pairwise_distance_heatmap.png",
        "artifacts/formal_math_visual_registry_v1/assets/pairwise_mahalanobis_distance.png",
    ),
    "imm_mode_mixing_recursion": (
        "artifacts/imm_filter_v1/switching_detection_metrics.csv",
        "artifacts/advanced_filter_comparison_v1/advanced_filter_comparison_report.md",
        "artifacts/advanced_filter_comparison_v1/advanced_filter_decision_matrix.csv",
    ),
    "pf_importance_weight_update": (
        "artifacts/particle_filter_v1/pf_method_comparison.csv",
        "artifacts/particle_filter_v1/pf_report.md",
        "artifacts/advanced_filter_comparison_v1/method_comparison.csv",
    ),
    "pf_class_evidence_extraction": (
        "artifacts/particle_filter_v1/pf_method_comparison.csv",
        "artifacts/particle_filter_v1/posterior_history.csv",
        "artifacts/advanced_filter_comparison_v1/method_comparison.csv",
    ),
    "rbpf_conditional_weight_update": (
        "artifacts/rbpf_v1/rbpf_method_comparison.csv",
        "artifacts/rbpf_v1/rbpf_report.md",
        "artifacts/advanced_filter_comparison_v1/advanced_filter_decision_matrix.csv",
    ),
}


ILLUSTRATIVE_EXACT_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "gaussian_feature_likelihood": (
        "artifacts/formal_math_visual_registry_v1/assets/gaussian_feature_likelihood.png",
    ),
    "calibration_metrics": (
        "artifacts/formal_math_visual_registry_v1/assets/calibration_metrics.png",
    ),
    "pairwise_mahalanobis_distance": (
        "artifacts/formal_math_visual_registry_v1/assets/pairwise_mahalanobis_distance.png",
    ),
    "imm_mode_mixing_recursion": (
        "artifacts/imm_filter_v1/switching_detection_metrics.csv",
        "artifacts/advanced_filter_comparison_v1/advanced_filter_decision_matrix.csv",
    ),
    "pf_importance_weight_update": (
        "artifacts/particle_filter_v1/pf_method_comparison.csv",
        "artifacts/particle_filter_v1/pf_report.md",
    ),
    "pf_class_evidence_extraction": (
        "artifacts/particle_filter_v1/posterior_history.csv",
        "artifacts/particle_filter_v1/pf_method_comparison.csv",
    ),
    "rbpf_conditional_weight_update": (
        "artifacts/rbpf_v1/rbpf_method_comparison.csv",
        "artifacts/rbpf_v1/rbpf_report.md",
    ),
}


def _equation_implementation(row: dict[str, object]) -> str:
    implementation = row["implementation"]
    return f"{implementation['module']}::{implementation['function']}"


def analyze_strict_equation_audit() -> StrictEquationAuditResult:
    registry = load_equation_registry()
    rows: list[StrictEquationAuditRow] = []
    for entry in registry:
        equation_id = str(entry["id"])
        registry_status = str(entry["status"])
        implementation = _equation_implementation(entry)
        if registry_status == "implemented":
            strict_label = "implemented"
            exact_artifacts = IMPLEMENTED_EXACT_ARTIFACTS.get(equation_id, tuple(entry.get("artifacts", ())))
            source_data = IMPLEMENTED_SOURCE_DATA.get(equation_id, tuple())
            notes = "Registry-implemented and backed by code plus concrete artifact outputs."
        elif equation_id in ILLUSTRATIVE_SOURCE_DATA:
            strict_label = "illustrative"
            exact_artifacts = ILLUSTRATIVE_EXACT_ARTIFACTS.get(equation_id, tuple())
            source_data = ILLUSTRATIVE_SOURCE_DATA[equation_id]
            notes = "Registry marks this as conceptual, but the repo has a concrete artifact-backed visualization or numeric trace."
        else:
            strict_label = "missing"
            exact_artifacts = tuple(entry.get("artifacts", ()))
            source_data = ()
            notes = "No direct artifact trail or executable implementation is linked yet."
        rows.append(
            StrictEquationAuditRow(
                equation_id=equation_id,
                registry_status=registry_status,
                strict_label=strict_label,
                implementation=implementation,
                exact_artifacts=tuple(exact_artifacts),
                source_data=tuple(source_data),
                notes=notes,
            )
        )

    summary = {
        "equation_count": len(rows),
        "implemented_count": sum(1 for row in rows if row.strict_label == "implemented"),
        "illustrative_count": sum(1 for row in rows if row.strict_label == "illustrative"),
        "missing_count": sum(1 for row in rows if row.strict_label == "missing"),
    }
    report_markdown = render_strict_equation_audit_report(StrictEquationAuditResult(tuple(rows), summary, ""))
    return StrictEquationAuditResult(tuple(rows), summary, report_markdown)


def _status_summary_line(summary: dict[str, object]) -> str:
    return (
        f"Implemented: `{summary['implemented_count']}` | "
        f"Illustrative: `{summary['illustrative_count']}` | "
        f"Missing: `{summary['missing_count']}`"
    )


def render_strict_equation_audit_report(result: StrictEquationAuditResult) -> str:
    lines = [
        "# Formal Math Strict Audit",
        "",
        "This audit enumerates every equation in `docs/math/equation_registry.yaml` and labels it strictly as implemented, illustrative, or missing. Conceptual entries are not counted as implemented just because a nearby plot exists.",
        "",
        "## Summary",
        "",
        f"- Equation count: `{result.summary['equation_count']}`",
        f"- {_status_summary_line(result.summary)}",
        "",
        "## Equation Audit",
        "",
        "| equation_id | registry_status | strict_label | implementation | exact_artifacts | source_data | notes |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in result.rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.equation_id}`",
                    f"`{row.registry_status}`",
                    f"`{row.strict_label}`",
                    f"`{row.implementation}`",
                    "<br>".join(f"`{artifact}`" for artifact in row.exact_artifacts) or "_none_",
                    "<br>".join(f"`{data}`" for data in row.source_data) or "_none_",
                    row.notes,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_strict_equation_audit_artifacts(output_dir: str | Path) -> StrictEquationAuditArtifacts:
    payload = analyze_strict_equation_audit()
    output_root = Path(output_dir)
    run_dir = output_root / "formal_math_strict_audit_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "formal_math_strict_audit_report.md"
    summary_path = run_dir / "formal_math_strict_audit_summary.json"
    rows_path = run_dir / "formal_math_strict_audit.csv"
    status_plot_path = run_dir / "formal_math_strict_audit_status.png"

    report_path.write_text(payload.report_markdown, encoding="utf-8")
    summary_path.write_text(json.dumps(payload.summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(
        rows_path,
        [
            {
                "equation_id": row.equation_id,
                "registry_status": row.registry_status,
                "strict_label": row.strict_label,
                "implementation": row.implementation,
                "exact_artifacts": " | ".join(row.exact_artifacts),
                "source_data": " | ".join(row.source_data),
                "notes": row.notes,
            }
            for row in payload.rows
        ],
        [
            "equation_id",
            "registry_status",
            "strict_label",
            "implementation",
            "exact_artifacts",
            "source_data",
            "notes",
        ],
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"implemented": "#0f766e", "illustrative": "#d97706", "missing": "#dc2626"}
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    x = list(range(len(payload.rows)))
    heights = [1] * len(payload.rows)
    bar_colors = [colors[row.strict_label] for row in payload.rows]
    ax.bar(x, heights, color=bar_colors)
    ax.set_xticks(x)
    ax.set_xticklabels([row.equation_id for row in payload.rows], rotation=35, ha="right", fontsize=7)
    ax.set_yticks([])
    ax.set_title("Formal Math Strict Audit Status", loc="left", fontweight="bold")
    ax.set_xlim(-0.6, max(len(payload.rows) - 0.4, 0.4))
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    fig.savefig(status_plot_path, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    return StrictEquationAuditArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        summary_path=summary_path,
        rows_path=rows_path,
        status_plot_path=status_plot_path,
    )
