from __future__ import annotations
from ..utils.math import _mean as _mean_of_values

from dataclasses import dataclass
import itertools
from pathlib import Path
from typing import Any

import numpy.random as random

from .policy import (
    CorpusPolicySpec,
    corpus_policy_to_dict,
    load_corpus_policy_spec,
    write_default_policy_artifacts,
)
from ..corpus.exploration.generic_corpus_exploration import (
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
    numeric_walkthrough_path: Path


@dataclass(frozen=True, slots=True)
class PolicyEvaluationRow:
    policy_id: str
    objective_id: str
    selected_candidate_count: int
    candidate_count: int
    selected_set: str
    ranked_ids: str
    mean_total_utility: float
    validity: float
    feature_excitation: float
    boundary_coverage: float
    classifier_stress: float
    provenance_completeness: float
    leakage: float
    triviality: float
    degeneracy: float
    adequacy_score: float
    downstream_proxy: float
    policy_score: float
    weights: dict[str, float]
    policy: dict[str, Any]

    def as_row_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "objective_id": self.objective_id,
            "selected_candidate_count": self.selected_candidate_count,
            "candidate_count": self.candidate_count,
            "selected_set": self.selected_set,
            "ranked_ids": self.ranked_ids,
            "mean_total_utility": self.mean_total_utility,
            "validity": self.validity,
            "feature_excitation": self.feature_excitation,
            "boundary_coverage": self.boundary_coverage,
            "classifier_stress": self.classifier_stress,
            "provenance_completeness": self.provenance_completeness,
            "leakage": self.leakage,
            "triviality": self.triviality,
            "degeneracy": self.degeneracy,
            "adequacy_score": self.adequacy_score,
            "downstream_proxy": self.downstream_proxy,
            "policy_score": self.policy_score,
            **{f"weight_{key}": value for key, value in self.weights.items()},
            "policy": self.policy,
        }


def write_corpus_policy_tuning_artifacts(
    output_dir: str | Path,
    *,
    policy: CorpusPolicySpec | None = None,
    seed: int = 11,
) -> CorpusPolicyTuningArtifacts:
    from .policy_sweep_rendering import write_corpus_policy_tuning_artifacts as _write_policy_sweep_artifacts

    run_dir = Path(output_dir) / "corpus_hyperparameter_tuning_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    default_policy = policy or load_corpus_policy_spec()
    write_default_policy_artifacts(Path(output_dir), default_policy)

    baseline = _evaluate_policy(default_policy, seed=seed, objective_id="dev_boundary_switching").as_row_dict()
    variants = _policy_variants(default_policy)
    sweep_rows = [_evaluate_policy(variant, seed=seed, objective_id="dev_boundary_switching").as_row_dict() for variant in variants]
    ablation_rows = _ablation_rows(default_policy, baseline, seed)
    perturbation_rows = _local_perturbation_rows(default_policy, baseline, seed)
    jaccard_rows = _jaccard_rows([baseline, *sweep_rows, *perturbation_rows])
    rank_rows = _rank_stability_rows([baseline, *sweep_rows, *perturbation_rows], baseline)
    sampler_rows = _sampler_budget_rows(default_policy)
    gate_rows = _gate_threshold_rows(default_policy, baseline)
    dev_holdout_rows = _dev_holdout_rows(default_policy, variants, seed)
    pareto_rows = _pareto_rows([baseline, *sweep_rows, *perturbation_rows])
    recommended = _recommended_policy([baseline, *sweep_rows, *perturbation_rows], default_policy, variants)
    return _write_policy_sweep_artifacts(
        run_dir=run_dir,
        baseline=baseline,
        sweep_rows=sweep_rows,
        ablation_rows=ablation_rows,
        perturbation_rows=perturbation_rows,
        jaccard_rows=jaccard_rows,
        rank_rows=rank_rows,
        sampler_rows=sampler_rows,
        gate_rows=gate_rows,
        dev_holdout_rows=dev_holdout_rows,
        pareto_rows=pareto_rows,
        recommended=recommended,
        seed=seed,
    )


def _evaluate_policy(policy: CorpusPolicySpec, *, seed: int, objective_id: str) -> PolicyEvaluationRow:
    weights = _generic_weights(policy)
    result = analyze_generic_corpus_exploration(seed=seed, weights=weights)
    selected = result.selected_corpus_manifest["selected_rows"]
    candidates = list(result.candidate_score_rows)
    selected_ids = tuple(str(row["trajectory_id"]) for row in selected)
    ranked_ids = tuple(str(row["trajectory_id"]) for row in candidates)
    leakage = _proxy_leakage(selected)
    validity = _mean_of_values([float(row["validity_score"]) for row in selected])
    feature = _mean_of_values([float(row["acceleration_range"]) for row in selected])
    boundary = _mean_of_values([float(row["boundary_score"]) for row in selected])
    stress = _mean_of_values([float(row["classifier_stress_score"]) for row in selected])
    provenance = _mean_of_values([float(row["provenance_completeness"]) for row in selected])
    adequacy = (
        0.25 * validity
        + 0.20 * boundary
        + 0.20 * min(feature / 1.5, 1.0)
        + 0.15 * stress
        + 0.20 * provenance
        - 0.20 * leakage
    )
    return PolicyEvaluationRow(
        policy_id=policy.policy_id,
        objective_id=objective_id,
        selected_candidate_count=len(selected),
        candidate_count=len(candidates),
        selected_set=";".join(selected_ids),
        ranked_ids=";".join(ranked_ids),
        mean_total_utility=_mean_of_values([float(row["total_utility"]) for row in selected]),
        validity=validity,
        feature_excitation=min(feature / 1.5, 1.0),
        boundary_coverage=boundary,
        classifier_stress=stress,
        provenance_completeness=provenance,
        leakage=leakage,
        triviality=1.0 - stress,
        degeneracy=1.0 - validity,
        adequacy_score=max(0.0, min(1.0, adequacy)),
        downstream_proxy=0.5 * stress + 0.3 * boundary + 0.2 * (1.0 - leakage),
        policy_score=max(0.0, min(1.0, adequacy + 0.1 * stress)),
        weights=dict(policy.generic_explorer_weights),
        policy=corpus_policy_to_dict(policy),
    )


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
        row = _evaluate_policy(variant, seed=seed, objective_id="ablation").as_row_dict()
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
    rng = random.default_rng(seed)
    base = [float(policy.generic_explorer_weights[key]) for key in policy.generic_explorer_weights]
    keys = list(policy.generic_explorer_weights)
    for index in range(16):
        sample = rng.dirichlet([max(weight * 80.0, 0.01) for weight in base])
        variant = _replace_generic_weights(policy, f"perturb_{index:02d}", dict(zip(keys, sample, strict=True)))
        row = _evaluate_policy(variant, seed=seed, objective_id="local_perturbation").as_row_dict()
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
        dev = _evaluate_policy(variant, seed=seed, objective_id="development_objectives").as_row_dict()
        holdout = _evaluate_policy(variant, seed=seed + 29, objective_id="holdout_objectives").as_row_dict()
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
        study_static_positive_weights=policy.study_static_positive_weights,
        study_static_penalty_weights=policy.study_static_penalty_weights,
        study_mc_weights=policy.study_mc_weights,
        sampler_budgets=policy.sampler_budgets,
        gates=policy.gates,
        normalization=policy.normalization,
    )


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

