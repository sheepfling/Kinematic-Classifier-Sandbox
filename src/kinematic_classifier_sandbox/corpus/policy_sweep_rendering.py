from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from numpy import arange, array, float64, mean, ndarray, zeros

from kinematic_classifier_sandbox.utils.io import write_csv
from kinematic_classifier_sandbox.utils.plotting import prepare_matplotlib

from .policy_sweep_types import CorpusPolicyTuningArtifacts

plt = prepare_matplotlib()


def write_corpus_policy_tuning_artifacts(
    run_dir: Path,
    baseline: dict[str, Any],
    sweep_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
    perturbation_rows: list[dict[str, Any]],
    jaccard_rows: list[dict[str, Any]],
    rank_rows: list[dict[str, Any]],
    sampler_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
    dev_holdout_rows: list[dict[str, Any]],
    pareto_rows: list[dict[str, Any]],
    recommended: dict[str, Any],
    seed: int,
    plot: bool = True,
) -> CorpusPolicyTuningArtifacts:
    sweep_design_rows = [_design_row(row) for row in sweep_rows]
    sweep_design_path = run_dir / "sweep_design.csv"
    sweep_results_path = run_dir / "sweep_results.csv"
    ablation_path = run_dir / "ablation_results.csv"
    perturbation_path = run_dir / "local_perturbation_results.csv"
    jaccard_path = run_dir / "selected_set_jaccard.csv"
    rank_path = run_dir / "rank_stability.csv"
    sampler_path = run_dir / "sampler_budget_sweep.csv"
    gate_path = run_dir / "gate_threshold_sweep.csv"
    dev_holdout_path = run_dir / "dev_holdout_results.csv"
    pareto_path = run_dir / "pareto_front.csv"
    recommended_path = run_dir / "recommended_policy.yaml"
    report_path = run_dir / "corpus_hyperparameter_tuning_report.md"
    numeric_walkthrough_path = run_dir / "corpus_policy_numeric_walkthrough.md"
    sweep_config_path = run_dir / "sweep_config.yaml"

    all_rows = [baseline, *sweep_rows, *perturbation_rows]
    best_row = max(all_rows, key=lambda row: (float(row["policy_score"]), float(row.get("selected_jaccard_vs_default", 1.0))))

    write_csv(sweep_design_path, sweep_design_rows, list(sweep_design_rows[0]))
    write_csv(
        sweep_results_path,
        [_policy_row_for_csv(baseline), *[_policy_row_for_csv(row) for row in sweep_rows]],
        _policy_result_fields(),
    )
    write_csv(ablation_path, ablation_rows, list(ablation_rows[0]))
    perturbation_fields = [*_policy_result_fields(), "selected_jaccard_vs_default", "rank_spearman_vs_default", "rank_kendall_vs_default"]
    write_csv(
        perturbation_path,
        [
            _policy_row_for_csv(
                row,
                ["selected_jaccard_vs_default", "rank_spearman_vs_default", "rank_kendall_vs_default"],
            )
            for row in perturbation_rows
        ],
        perturbation_fields,
    )
    write_csv(jaccard_path, jaccard_rows, list(jaccard_rows[0]))
    write_csv(rank_path, rank_rows, list(rank_rows[0]))
    write_csv(sampler_path, sampler_rows, list(sampler_rows[0]))
    write_csv(gate_path, gate_rows, list(gate_rows[0]))
    write_csv(dev_holdout_path, dev_holdout_rows, list(dev_holdout_rows[0]))
    write_csv(pareto_path, pareto_rows, list(pareto_rows[0]))
    recommended_path.write_text(yaml.safe_dump(recommended, sort_keys=False), encoding="utf-8")
    sweep_config_path.write_text(
        yaml.safe_dump({"seed": seed, "policy_ids": [row["policy_id"] for row in [baseline, *sweep_rows]]}, sort_keys=False),
        encoding="utf-8",
    )
    report_path.write_text(_render_report(baseline, sweep_rows, ablation_rows, perturbation_rows, dev_holdout_rows, recommended), encoding="utf-8")
    numeric_walkthrough_path.write_text(
        _render_numeric_walkthrough(
            baseline=baseline,
            best_row=best_row,
            rank_rows=rank_rows,
            holdout_rows=dev_holdout_rows,
            recommended=recommended,
        ),
        encoding="utf-8",
    )

    if plot:
        _plot_tornado(run_dir / "weight_sensitivity_tornado.png", ablation_rows)
        _plot_heatmap(run_dir / "selected_set_jaccard_heatmap.png", jaccard_rows, "selected-set Jaccard")
        _plot_rank_heatmap(run_dir / "rank_correlation_heatmap.png", rank_rows)
        _plot_ablation(run_dir / "ablation_tradeoff_bars.png", ablation_rows)
        _plot_pareto(run_dir / "pareto_tradeoff_scatter.png", pareto_rows)
        _plot_sampler(run_dir / "sampler_budget_efficiency.png", sampler_rows)
        _plot_gate(run_dir / "gate_sensitivity_curves.png", gate_rows)
        _plot_dev_holdout(run_dir / "dev_vs_holdout_policy_scores.png", dev_holdout_rows)

    return CorpusPolicyTuningArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        recommended_policy_path=recommended_path,
        sweep_results_path=sweep_results_path,
        ablation_results_path=ablation_path,
        stability_path=jaccard_path,
        numeric_walkthrough_path=numeric_walkthrough_path,
    )


def _policy_row_for_csv(row: dict[str, Any], extra_fields: list[str] | None = None) -> dict[str, Any]:
    fields = set(_policy_result_fields())
    if extra_fields:
        fields.update(extra_fields)
    return {key: row[key] for key in fields if key in row}


def _policy_result_fields() -> list[str]:
    return [
        "policy_id",
        "objective_id",
        "selected_candidate_count",
        "candidate_count",
        "selected_set",
        "ranked_ids",
        "mean_total_utility",
        "validity",
        "feature_excitation",
        "boundary_coverage",
        "classifier_stress",
        "provenance_completeness",
        "leakage",
        "triviality",
        "degeneracy",
        "adequacy_score",
        "downstream_proxy",
        "policy_score",
        "weight_validity",
        "weight_coverage_novelty",
        "weight_boundary_score",
        "weight_classifier_stress",
        "weight_environment_score",
        "weight_provenance_completeness",
    ]


def _design_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in _policy_result_fields() if key.startswith("policy") or key.startswith("weight")}


def _render_report(
    baseline: dict[str, Any],
    sweep_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
    perturbation_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    recommended: dict[str, Any],
) -> str:
    most_influential = max(ablation_rows, key=lambda row: abs(float(row["adequacy_delta_vs_default"])))
    most_dangerous = max(ablation_rows, key=lambda row: (bool(row["unsafe_flag"]), float(row["leakage"])))
    mean_jaccard = float(mean([float(row["selected_jaccard_vs_default"]) for row in perturbation_rows]))
    return "\n".join(
        [
            "# Corpus Hyperparameter Tuning V1",
            "",
            "This report lifts corpus scoring constants into an auditable `CorpusPolicySpec` and evaluates the default policy under sweeps, ablations, perturbations, sampler budgets, gate thresholds, and dev/holdout splits.",
            "",
            "## Default Policy",
            "",
            f"- Baseline policy: `{baseline['policy_id']}`",
            f"- Baseline adequacy score: `{float(baseline['adequacy_score']):.3f}`",
            f"- Baseline selected count: `{baseline['selected_candidate_count']}`",
            "",
            "## Stability",
            "",
            f"- Mean local selected-set Jaccard: `{mean_jaccard:.3f}`",
            "- Low Jaccard means selected corpora are fragile under reasonable policy perturbations.",
            "",
            "## Ablation Findings",
            "",
            f"- Most influential term: `{most_influential['removed_term']}`",
            f"- Most dangerous ablation: `{most_dangerous['removed_term']}`",
            "",
            "## Dev/Holdout",
            "",
            f"- Evaluated policies: `{len(holdout_rows)}`",
            "- Holdout results are reported separately from development scores.",
            "",
            "## Recommended Policy",
            "",
            f"- Recommended policy: `{recommended['recommendation']['recommended_policy_id']}`",
            f"- Known limitation: {recommended['recommendation']['known_limitation']}",
            "",
            "## Interpretation Rule",
            "",
            "Treat tuned weights as corpus policy, not truth. Preserve metric vectors and Pareto rows when interpreting downstream classifier claims.",
        ]
    )


def _render_numeric_walkthrough(
    *,
    baseline: dict[str, Any],
    best_row: dict[str, Any],
    rank_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    recommended: dict[str, Any],
) -> str:
    rank_lookup = {str(row["policy_id"]): row for row in rank_rows}
    holdout_lookup = {str(row["policy_id"]): row for row in holdout_rows}
    best_policy_id = str(best_row["policy_id"])
    rank_row = rank_lookup.get(best_policy_id, {})
    holdout_row = holdout_lookup.get(best_policy_id, {})
    feature_term = min(float(best_row["feature_excitation"]) / 1.5, 1.0)
    adequacy = float(best_row["adequacy_score"])
    policy_score = float(best_row["policy_score"])
    return "\n".join(
        [
            "# Corpus Policy Numeric Walkthrough",
            "",
            "This artifact expands one real evaluated policy row from `corpus/policy_sweep.py`.",
            "",
            "## Recommended Policy",
            "",
            f"- Policy id: `{best_policy_id}`",
            f"- Recommendation basis: {recommended['recommendation']['basis']}",
            "",
            "## Adequacy Proxy Substitution",
            "",
            "The implemented adequacy proxy is:",
            "",
            "```tex",
            "A_{\\text{policy}}",
            "= 0.25\\,\\text{validity}",
            "+ 0.20\\,\\text{boundary}",
            "+ 0.20\\min(\\text{feature}/1.5, 1)",
            "+ 0.15\\,\\text{stress}",
            "+ 0.20\\,\\text{provenance}",
            "- 0.20\\,\\text{leakage}.",
            "```",
            "",
            "Substituting the evaluated row:",
            "",
            "```tex",
            f"A_{{\\text{{policy}}}} = 0.25({float(best_row['validity']):.6f})",
            f"+ 0.20({float(best_row['boundary_coverage']):.6f})",
            f"+ 0.20({feature_term:.6f})",
            f"+ 0.15({float(best_row['classifier_stress']):.6f})",
            f"+ 0.20({float(best_row['provenance_completeness']):.6f})",
            f"- 0.20({float(best_row['leakage']):.6f})",
            f"= {adequacy:.6f}.",
            "```",
            "",
            "The bounded policy score then adds the stress bonus:",
            "",
            "```tex",
            f"J_{{\\text{{policy}}}} = \\operatorname{{clip}}({adequacy:.6f} + 0.10({float(best_row['classifier_stress']):.6f}), 0, 1) = {policy_score:.6f}.",
            "```",
            "",
            "## Stability And Holdout",
            "",
            f"- Selected-set Jaccard vs default: `{float(rank_row.get('selected_set_jaccard_vs_default', 1.0)):.6f}`",
            f"- Spearman vs default: `{float(rank_row.get('spearman_vs_default', 1.0)):.6f}`",
            f"- Kendall tau vs default: `{float(rank_row.get('kendall_tau_vs_default', 1.0)):.6f}`",
            f"- Dev score: `{float(holdout_row.get('dev_score', policy_score)):.6f}`",
            f"- Holdout score: `{float(holdout_row.get('holdout_score', policy_score)):.6f}`",
            f"- Dev-holdout gap: `{float(holdout_row.get('dev_holdout_gap', 0.0)):.6f}`",
            "",
            "## Comparison To Default",
            "",
            f"- Default policy id: `{baseline['policy_id']}`",
            f"- Default policy score: `{float(baseline['policy_score']):.6f}`",
            f"- Recommended policy score: `{policy_score:.6f}`",
            f"- Delta vs default: `{policy_score - float(baseline['policy_score']):.6f}`",
            "",
            "## Interpretation",
            "",
            "- A higher policy score is only meaningful when the selected-set and rank-stability metrics remain acceptable.",
            "- The holdout score is the anti-overfitting check; a tuned policy should not be judged only by the development objective.",
        ]
    )


def _plot_tornado(path: Path, rows: list[dict[str, Any]]) -> None:
    labels = [str(row["removed_term"]) for row in rows]
    values = [float(row["adequacy_delta_vs_default"]) for row in rows]
    _barh(path, labels, values, "Weight Sensitivity Tornado", "adequacy delta")


def _plot_ablation(path: Path, rows: list[dict[str, Any]]) -> None:
    labels = [str(row["removed_term"]) for row in rows]
    values = [float(row["leakage"]) for row in rows]
    _barh(path, labels, values, "Ablation Tradeoff Bars", "leakage")


def _barh(path: Path, labels: list[str], values: list[float], title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.barh(labels, values, color="#2563eb")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_heatmap(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    labels = sorted({str(row["policy_id_a"]) for row in rows})
    matrix = zeros((len(labels), len(labels)), dtype=float64)
    index = {label: i for i, label in enumerate(labels)}
    for row in rows:
        matrix[index[str(row["policy_id_a"])]] [index[str(row["policy_id_b"])] ] = float(row["selected_set_jaccard"])
    _imshow(path, matrix, labels, title)


def _plot_rank_heatmap(path: Path, rows: list[dict[str, Any]]) -> None:
    labels = [str(row["policy_id"]) for row in rows]
    matrix = array([[float(row["spearman_vs_default"]), float(row["kendall_tau_vs_default"])] for row in rows])
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.imshow(matrix, vmin=-1, vmax=1, cmap="viridis")
    ax.set_yticks(range(len(labels)), labels=labels, fontsize=6)
    ax.set_xticks([0, 1], labels=["spearman", "kendall"])
    ax.set_title("Rank Correlation Heatmap", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _imshow(path: Path, matrix: "ndarray", labels: list[str], title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    ax.imshow(matrix, vmin=0, vmax=1, cmap="magma")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=90, fontsize=5)
    ax.set_yticks(range(len(labels)), labels=labels, fontsize=5)
    ax.set_title(title, loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_pareto(path: Path, rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    colors = ["#15803d" if row["pareto_front"] else "#64748b" for row in rows]
    ax.scatter([float(row["leakage"]) for row in rows], [float(row["adequacy_score"]) for row in rows], c=colors)
    ax.set_xlabel("leakage")
    ax.set_ylabel("adequacy")
    ax.set_title("Pareto Tradeoff Scatter", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_sampler(path: Path, rows: list[dict[str, Any]]) -> None:
    _barh(path, [str(row["sampler_family"]) for row in rows], [float(row["useful_discovery_rate"]) for row in rows], "Sampler Budget Efficiency", "useful discovery rate")


def _plot_gate(path: Path, rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    for gate in sorted({str(row["gate"]) for row in rows}):
        subset = [row for row in rows if row["gate"] == gate]
        ax.plot([float(row["threshold_value"]) for row in subset], [float(row["accepted_count"]) for row in subset], marker="o", label=gate)
    ax.set_title("Gate Sensitivity Curves", loc="left", fontweight="bold")
    ax.set_xlabel("threshold")
    ax.set_ylabel("accepted count")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_dev_holdout(path: Path, rows: list[dict[str, Any]]) -> None:
    labels = [str(row["policy_id"]) for row in rows]
    x = arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.bar(x - 0.18, [float(row["dev_score"]) for row in rows], width=0.36, label="dev")
    ax.bar(x + 0.18, [float(row["holdout_score"]) for row in rows], width=0.36, label="holdout")
    ax.set_xticks(x, labels=labels, rotation=35, ha="right", fontsize=7)
    ax.set_title("Dev vs Holdout Policy Scores", loc="left", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
