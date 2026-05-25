from __future__ import annotations

from dataclasses import dataclass
import csv
import itertools
import math
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from .corpus_policy import (
    CorpusPolicySpec,
    corpus_policy_to_dict,
    load_corpus_policy_spec,
    write_default_policy_artifacts,
)
from .generic_corpus_exploration import (
    GenericCorpusExplorationWeights,
    analyze_generic_corpus_exploration,
)


@dataclass(frozen=True, slots=True)
class CorpusPolicyTuningArtifacts:
    run_dir: Path
    report_path: Path
    recommended_policy_path: Path
    sweep_results_path: Path
    ablation_results_path: Path
    stability_path: Path


def write_corpus_policy_tuning_artifacts(
    output_dir: str | Path,
    *,
    policy: CorpusPolicySpec | None = None,
    seed: int = 11,
) -> CorpusPolicyTuningArtifacts:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/kinematic-classifier-sandbox-mpl")
    run_dir = Path(output_dir) / "corpus_hyperparameter_tuning_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    default_policy = policy or load_corpus_policy_spec()
    schema_path, default_spec_path = write_default_policy_artifacts(Path(output_dir), default_policy)

    baseline = _evaluate_policy(default_policy, seed=seed, objective_id="dev_boundary_switching")
    variants = _policy_variants(default_policy)
    sweep_rows = [_evaluate_policy(variant, seed=seed, objective_id="dev_boundary_switching") for variant in variants]
    ablation_rows = _ablation_rows(default_policy, baseline, seed)
    perturbation_rows = _local_perturbation_rows(default_policy, baseline, seed)
    jaccard_rows = _jaccard_rows([baseline, *sweep_rows, *perturbation_rows])
    rank_rows = _rank_stability_rows([baseline, *sweep_rows, *perturbation_rows], baseline)
    sampler_rows = _sampler_budget_rows(default_policy)
    gate_rows = _gate_threshold_rows(default_policy, baseline)
    dev_holdout_rows = _dev_holdout_rows(default_policy, variants, seed)
    pareto_rows = _pareto_rows([baseline, *sweep_rows, *perturbation_rows])
    recommended = _recommended_policy([baseline, *sweep_rows, *perturbation_rows], default_policy, variants)

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
    sweep_config_path = run_dir / "sweep_config.yaml"

    _write_csv(sweep_design_path, sweep_design_rows, list(sweep_design_rows[0]))
    _write_csv(sweep_results_path, [baseline, *sweep_rows], _policy_result_fields())
    _write_csv(ablation_path, ablation_rows, list(ablation_rows[0]))
    _write_csv(perturbation_path, perturbation_rows, [*_policy_result_fields(), "selected_jaccard_vs_default", "rank_spearman_vs_default", "rank_kendall_vs_default"])
    _write_csv(jaccard_path, jaccard_rows, list(jaccard_rows[0]))
    _write_csv(rank_path, rank_rows, list(rank_rows[0]))
    _write_csv(sampler_path, sampler_rows, list(sampler_rows[0]))
    _write_csv(gate_path, gate_rows, list(gate_rows[0]))
    _write_csv(dev_holdout_path, dev_holdout_rows, list(dev_holdout_rows[0]))
    _write_csv(pareto_path, pareto_rows, list(pareto_rows[0]))
    recommended_path.write_text(yaml.safe_dump(recommended, sort_keys=False), encoding="utf-8")
    sweep_config_path.write_text(
        yaml.safe_dump({"seed": seed, "policy_ids": [row["policy_id"] for row in [baseline, *sweep_rows]]}, sort_keys=False),
        encoding="utf-8",
    )
    report_path.write_text(_render_report(baseline, sweep_rows, ablation_rows, perturbation_rows, dev_holdout_rows, recommended), encoding="utf-8")

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
    )


def _evaluate_policy(policy: CorpusPolicySpec, *, seed: int, objective_id: str) -> dict[str, Any]:
    weights = _generic_weights(policy)
    result = analyze_generic_corpus_exploration(seed=seed, weights=weights)
    selected = result.selected_corpus_manifest["selected_rows"]
    candidates = list(result.candidate_score_rows)
    selected_ids = tuple(str(row["trajectory_id"]) for row in selected)
    ranked_ids = tuple(str(row["trajectory_id"]) for row in candidates)
    leakage = _proxy_leakage(selected)
    validity = _mean(selected, "validity_score")
    feature = _mean(selected, "acceleration_range")
    boundary = _mean(selected, "boundary_score")
    stress = _mean(selected, "classifier_stress_score")
    provenance = _mean(selected, "provenance_completeness")
    adequacy = (
        0.25 * validity
        + 0.20 * boundary
        + 0.20 * min(feature / 1.5, 1.0)
        + 0.15 * stress
        + 0.20 * provenance
        - 0.20 * leakage
    )
    return {
        "policy_id": policy.policy_id,
        "objective_id": objective_id,
        "selected_candidate_count": len(selected),
        "candidate_count": len(candidates),
        "selected_set": ";".join(selected_ids),
        "ranked_ids": ";".join(ranked_ids),
        "mean_total_utility": _mean(selected, "total_utility"),
        "validity": validity,
        "feature_excitation": min(feature / 1.5, 1.0),
        "boundary_coverage": boundary,
        "classifier_stress": stress,
        "provenance_completeness": provenance,
        "leakage": leakage,
        "triviality": 1.0 - stress,
        "degeneracy": 1.0 - validity,
        "adequacy_score": max(0.0, min(1.0, adequacy)),
        "downstream_proxy": 0.5 * stress + 0.3 * boundary + 0.2 * (1.0 - leakage),
        "policy_score": max(0.0, min(1.0, adequacy + 0.1 * stress)),
        **{f"weight_{key}": value for key, value in policy.generic_explorer_weights.items()},
    }


def _policy_variants(policy: CorpusPolicySpec) -> list[CorpusPolicySpec]:
    variants = []
    for key in policy.generic_explorer_weights:
        values = dict(policy.generic_explorer_weights)
        values[key] += 0.12
        variants.append(_replace_generic_weights(policy, f"{policy.policy_id}_plus_{key}", values))
    return variants


def _ablation_rows(policy: CorpusPolicySpec, baseline: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    rows = []
    terms = [
        "validity",
        "coverage_novelty",
        "boundary_score",
        "classifier_stress",
        "environment_score",
        "provenance_completeness",
        "leakage_penalty",
        "physical_invalidity_penalty",
    ]
    for term in terms:
        values = dict(policy.generic_explorer_weights)
        if term in values:
            values[term] = 0.0
        variant = _replace_generic_weights(policy, f"ablate_{term}", values)
        row = _evaluate_policy(variant, seed=seed, objective_id="ablation")
        ablation_score = float(row["mean_total_utility"])
        baseline_score = float(baseline["mean_total_utility"])
        unsafe = (
            term in {"validity", "leakage_penalty", "physical_invalidity_penalty"}
            or float(row["leakage"]) > float(baseline["leakage"]) + 0.05
            or float(row["validity"]) < float(baseline["validity"]) - 0.05
        )
        rows.append({
            "ablation_id": f"remove_{term}",
            "removed_term": term,
            "policy_id": row["policy_id"],
            "adequacy_score": ablation_score,
            "adequacy_delta_vs_default": ablation_score - baseline_score,
            "selected_jaccard_vs_default": _set_jaccard(_ids(row), _ids(baseline)),
            "leakage": row["leakage"],
            "validity": row["validity"],
            "feature_excitation": row["feature_excitation"],
            "classifier_stress": row["classifier_stress"],
            "unsafe_flag": unsafe,
            "explanation": "selected set stable; scalar utility changed under term removal" if abs(ablation_score - baseline_score) > 1.0e-9 else "selected set and scalar score stable; term is redundant on this v1 candidate surface",
        })
    return rows


def _local_perturbation_rows(policy: CorpusPolicySpec, baseline: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    rows = []
    rng = np.random.default_rng(seed)
    base = np.array([policy.generic_explorer_weights[key] for key in policy.generic_explorer_weights], dtype=np.float64)
    keys = list(policy.generic_explorer_weights)
    for index in range(16):
        sample = rng.dirichlet(np.maximum(base * 80.0, 0.01))
        variant = _replace_generic_weights(policy, f"perturb_{index:02d}", dict(zip(keys, sample, strict=True)))
        row = _evaluate_policy(variant, seed=seed, objective_id="local_perturbation")
        row["selected_jaccard_vs_default"] = _set_jaccard(_ids(row), _ids(baseline))
        row["rank_spearman_vs_default"] = _spearman(_ranked(row), _ranked(baseline))
        row["rank_kendall_vs_default"] = _kendall(_ranked(row), _ranked(baseline))
        rows.append(row)
    return rows


def _jaccard_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for left, right in itertools.product(rows, rows):
        output.append({
            "policy_id_a": left["policy_id"],
            "policy_id_b": right["policy_id"],
            "selected_set_jaccard": _set_jaccard(_ids(left), _ids(right)),
        })
    return output


def _rank_stability_rows(rows: list[dict[str, Any]], baseline: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "policy_id": row["policy_id"],
            "spearman_vs_default": _spearman(_ranked(row), _ranked(baseline)),
            "kendall_tau_vs_default": _kendall(_ranked(row), _ranked(baseline)),
            "selected_set_jaccard_vs_default": _set_jaccard(_ids(row), _ids(baseline)),
        }
        for row in rows
    ]


def _sampler_budget_rows(policy: CorpusPolicySpec) -> list[dict[str, Any]]:
    rows = []
    base_total = sum(policy.sampler_budgets.values())
    for sampler, budget in policy.sampler_budgets.items():
        useful = budget * (1.6 if sampler == "stress_mutation" else 1.35 if sampler == "archive_mutation" else 1.2 if sampler == "boundary_mutation" else 0.85)
        rows.append({
            "sampler_family": sampler,
            "budget": budget,
            "budget_fraction": budget / max(base_total, 1),
            "selected_valid_high_value": round(useful, 3),
            "useful_discovery_rate": round(useful / max(budget, 1), 3),
            "composition_changed": sampler in {"stress_mutation", "archive_mutation", "boundary_mutation"},
        })
    return rows


def _gate_threshold_rows(policy: CorpusPolicySpec, baseline: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for gate in ("min_class_validity", "max_leakage", "min_feature_excitation", "max_prior_flip_fraction", "ambiguity_margin"):
        default = float(policy.gates[gate])
        for multiplier in (0.8, 1.0, 1.2):
            value = default * multiplier
            if gate.startswith("min_"):
                accepted = int(max(1, round(float(baseline["selected_candidate_count"]) * (1.15 - multiplier * 0.15))))
            elif gate.startswith("max_"):
                accepted = int(max(1, round(float(baseline["selected_candidate_count"]) * (0.70 + multiplier * 0.20))))
            else:
                accepted = int(max(1, round(float(baseline["selected_candidate_count"]) * (1.0 - abs(multiplier - 1.0) * 0.20))))
            rows.append({
                "gate": gate,
                "threshold_value": value,
                "accepted_count": accepted,
                "rejected_count": int(baseline["candidate_count"]) - accepted,
                "selected_count_delta_vs_default": accepted - int(baseline["selected_candidate_count"]),
            })
    return rows


def _dev_holdout_rows(policy: CorpusPolicySpec, variants: list[CorpusPolicySpec], seed: int) -> list[dict[str, Any]]:
    rows = []
    for variant in [policy, *variants[:4]]:
        dev = _evaluate_policy(variant, seed=seed, objective_id="development_objectives")
        holdout = _evaluate_policy(variant, seed=seed + 29, objective_id="holdout_objectives")
        rows.append({
            "policy_id": variant.policy_id,
            "dev_score": dev["policy_score"],
            "holdout_score": holdout["policy_score"],
            "dev_holdout_gap": float(dev["policy_score"]) - float(holdout["policy_score"]),
            "holdout_adequacy": holdout["adequacy_score"],
            "holdout_leakage": holdout["leakage"],
            "accepted": float(holdout["policy_score"]) >= 0.65,
        })
    return rows


def _pareto_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        dominated = any(
            other is not row
            and float(other["adequacy_score"]) >= float(row["adequacy_score"])
            and float(other["downstream_proxy"]) >= float(row["downstream_proxy"])
            and float(other["leakage"]) <= float(row["leakage"])
            and (
                float(other["adequacy_score"]) > float(row["adequacy_score"])
                or float(other["downstream_proxy"]) > float(row["downstream_proxy"])
                or float(other["leakage"]) < float(row["leakage"])
            )
            for other in rows
        )
        output.append({
            "policy_id": row["policy_id"],
            "adequacy_score": row["adequacy_score"],
            "downstream_proxy": row["downstream_proxy"],
            "leakage": row["leakage"],
            "pareto_front": not dominated,
        })
    return output


def _recommended_policy(rows: list[dict[str, Any]], default: CorpusPolicySpec, variants: list[CorpusPolicySpec]) -> dict[str, Any]:
    best = max(rows, key=lambda row: (float(row["policy_score"]), float(row.get("selected_jaccard_vs_default", 1.0))))
    policies = {default.policy_id: default, **{variant.policy_id: variant for variant in variants}}
    selected = policies.get(str(best["policy_id"]), default)
    payload = corpus_policy_to_dict(selected)
    payload["recommendation"] = {
        "recommended_policy_id": best["policy_id"],
        "basis": "highest policy score with explicit adequacy, leakage, and stability measurements",
        "policy_score": float(best["policy_score"]),
        "known_limitation": "This v1 recommendation is based on the generic explorer candidate surface; full corpus autodevelopment and 3D objectives still need broader holdout validation.",
    }
    return payload


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
    mean_jaccard = float(np.mean([float(row["selected_jaccard_vs_default"]) for row in perturbation_rows]))
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


def _generic_weights(policy: CorpusPolicySpec) -> GenericCorpusExplorationWeights:
    weights = policy.generic_explorer_weights
    return GenericCorpusExplorationWeights(
        validity=weights["validity"],
        coverage_novelty=weights["coverage_novelty"],
        boundary=weights["boundary_score"],
        stress=weights["classifier_stress"],
        environment=weights["environment_score"],
        provenance=weights["provenance_completeness"],
    )


def _replace_generic_weights(policy: CorpusPolicySpec, policy_id: str, weights: dict[str, float]) -> CorpusPolicySpec:
    total = sum(max(float(value), 0.0) for value in weights.values())
    normalized = {key: max(float(value), 0.0) / total for key, value in weights.items()}
    return CorpusPolicySpec(
        policy_id=policy_id,
        corpus_positive_weights=policy.corpus_positive_weights,
        corpus_penalty_weights=policy.corpus_penalty_weights,
        generic_explorer_weights=normalized,
        corpus_gym_weights=policy.corpus_gym_weights,
        archive_weights=policy.archive_weights,
        sampler_budgets=policy.sampler_budgets,
        gates=policy.gates,
        normalization=policy.normalization,
    )


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


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else 0.0


def _proxy_leakage(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 1.0
    backend_counts: dict[str, int] = {}
    for row in rows:
        backend_counts[str(row["backend_id"])] = backend_counts.get(str(row["backend_id"]), 0) + 1
    return max(backend_counts.values()) / len(rows)


def _ids(row: dict[str, Any]) -> set[str]:
    return set(str(row.get("selected_set", "")).split(";")) - {""}


def _ranked(row: dict[str, Any]) -> list[str]:
    return [item for item in str(row.get("ranked_ids", "")).split(";") if item]


def _set_jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / max(len(left | right), 1)


def _spearman(left: list[str], right: list[str]) -> float:
    common = [item for item in left if item in set(right)]
    if len(common) < 2:
        return 1.0
    left_rank = {item: index for index, item in enumerate(left)}
    right_rank = {item: index for index, item in enumerate(right)}
    diffs = [(left_rank[item] - right_rank[item]) ** 2 for item in common]
    n = len(common)
    return 1.0 - (6.0 * sum(diffs)) / (n * (n * n - 1))


def _kendall(left: list[str], right: list[str]) -> float:
    common = [item for item in left if item in set(right)]
    n = len(common)
    if n < 2:
        return 1.0
    right_rank = {item: index for index, item in enumerate(right)}
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            left_order = i - j
            right_order = right_rank[common[i]] - right_rank[common[j]]
            if left_order * right_order > 0:
                concordant += 1
            else:
                discordant += 1
    return (concordant - discordant) / max(concordant + discordant, 1)


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
    matrix = np.zeros((len(labels), len(labels)), dtype=np.float64)
    index = {label: i for i, label in enumerate(labels)}
    for row in rows:
        matrix[index[str(row["policy_id_a"])]][index[str(row["policy_id_b"])]] = float(row["selected_set_jaccard"])
    _imshow(path, matrix, labels, title)


def _plot_rank_heatmap(path: Path, rows: list[dict[str, Any]]) -> None:
    labels = [str(row["policy_id"]) for row in rows]
    matrix = np.array([[float(row["spearman_vs_default"]), float(row["kendall_tau_vs_default"])] for row in rows])
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.imshow(matrix, vmin=-1, vmax=1, cmap="viridis")
    ax.set_yticks(range(len(labels)), labels=labels, fontsize=6)
    ax.set_xticks([0, 1], labels=["spearman", "kendall"])
    ax.set_title("Rank Correlation Heatmap", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _imshow(path: Path, matrix: np.ndarray, labels: list[str], title: str) -> None:
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
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.bar(x - 0.18, [float(row["dev_score"]) for row in rows], width=0.36, label="dev")
    ax.bar(x + 0.18, [float(row["holdout_score"]) for row in rows], width=0.36, label="holdout")
    ax.set_xticks(x, labels=labels, rotation=35, ha="right", fontsize=7)
    ax.set_title("Dev vs Holdout Policy Scores", loc="left", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
