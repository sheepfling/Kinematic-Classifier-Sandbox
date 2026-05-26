from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import yaml

from ..analysis.feature_analysis import FeatureAnalysisResult
from ..trajectory_generator import DatasetTierDefinition, default_dataset_tiers
from ..utils.math import _clamp
from .adequacy_audit import CorpusAdequacyResult

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OBJECTIVES_PATH = ROOT / "experiments" / "corpus_objectives" / "common_1d_corpus_objectives.yaml"


@dataclass(frozen=True, slots=True)
class CorpusCandidateSpec:
    candidate_id: str
    description: str
    sampling_method: str
    seed: int
    tier_counts: dict[str, int]
    measurement_scale: float = 1.0
    irregularity_scale: float = 1.0
    outlier_scale: float = 1.0
    step_scale: float = 1.0
    per_tier_measurement_scale: dict[str, float] | None = None
    per_tier_irregularity_scale: dict[str, float] | None = None
    per_tier_outlier_scale: dict[str, float] | None = None
    per_tier_step_scale: dict[str, float] | None = None


def _status_score(status: str) -> float:
    return {"green": 1.0, "yellow": 0.5, "red": 0.0}.get(status, 0.0)


def _scale_range(bounds: tuple[float, float], scale: float, *, integral: bool = False) -> tuple[float, float]:
    lower = bounds[0] * scale
    upper = bounds[1] * scale
    if integral:
        lower_i = max(2, int(round(lower)))
        upper_i = max(lower_i + 1, int(round(upper)))
        return lower_i, upper_i
    return max(0.0, lower), max(max(0.0, lower), upper)


def load_corpus_objectives(path: str | Path = DEFAULT_OBJECTIVES_PATH) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return dict(payload)


def _default_candidate_specs(seed: int) -> tuple[CorpusCandidateSpec, ...]:
    return (
        CorpusCandidateSpec(
            candidate_id="baseline_uniform",
            description="Uniform tier counts with default generator settings.",
            sampling_method="baseline",
            seed=seed,
            tier_counts={"easy_v1": 5, "boundary_v1": 5, "adversarial_v1": 5, "stress_v1": 5, "realistic_v1": 5},
        ),
        CorpusCandidateSpec(
            candidate_id="boundary_boosted",
            description="Extra boundary and adversarial coverage to strengthen hard class-pair evidence.",
            sampling_method="boundary_focused",
            seed=seed + 101,
            tier_counts={"easy_v1": 4, "boundary_v1": 7, "adversarial_v1": 6, "stress_v1": 4, "realistic_v1": 4},
            measurement_scale=1.05,
            irregularity_scale=1.10,
        ),
        CorpusCandidateSpec(
            candidate_id="leakage_reduced",
            description="Softer timing irregularity and outlier rate to reduce covariate leakage.",
            sampling_method="grid_search",
            seed=seed + 202,
            tier_counts={"easy_v1": 4, "boundary_v1": 6, "adversarial_v1": 5, "stress_v1": 3, "realistic_v1": 4},
            measurement_scale=0.95,
            irregularity_scale=0.75,
            outlier_scale=0.70,
        ),
        CorpusCandidateSpec(
            candidate_id="excitation_boosted",
            description="Heavier stress and outlier exposure to improve feature excitation coverage.",
            sampling_method="adversarial_template",
            seed=seed + 303,
            tier_counts={"easy_v1": 4, "boundary_v1": 6, "adversarial_v1": 7, "stress_v1": 6, "realistic_v1": 4},
            measurement_scale=1.20,
            irregularity_scale=1.20,
            outlier_scale=1.35,
            step_scale=0.92,
        ),
        CorpusCandidateSpec(
            candidate_id="realistic_balanced",
            description="Balanced corpus with more realistic and boundary coverage.",
            sampling_method="latin_hypercube",
            seed=seed + 404,
            tier_counts={"easy_v1": 5, "boundary_v1": 6, "adversarial_v1": 5, "stress_v1": 4, "realistic_v1": 6},
            measurement_scale=1.00,
            irregularity_scale=0.95,
            outlier_scale=0.90,
        ),
        CorpusCandidateSpec(
            candidate_id="stress_frontier",
            description="Aggressively hard corpus to probe identifiability and robustness limits.",
            sampling_method="rejection_sampling",
            seed=seed + 505,
            tier_counts={"easy_v1": 3, "boundary_v1": 7, "adversarial_v1": 6, "stress_v1": 7, "realistic_v1": 3},
            measurement_scale=1.25,
            irregularity_scale=1.30,
            outlier_scale=1.40,
            step_scale=0.85,
        ),
    )


def _candidate_tier_definitions(spec: CorpusCandidateSpec) -> tuple[DatasetTierDefinition, ...]:
    rows: list[DatasetTierDefinition] = []
    for tier in default_dataset_tiers():
        measurement_scale = spec.per_tier_measurement_scale.get(tier.name, spec.measurement_scale) if spec.per_tier_measurement_scale else spec.measurement_scale
        irregularity_scale = spec.per_tier_irregularity_scale.get(tier.name, spec.irregularity_scale) if spec.per_tier_irregularity_scale else spec.irregularity_scale
        outlier_scale = spec.per_tier_outlier_scale.get(tier.name, spec.outlier_scale) if spec.per_tier_outlier_scale else spec.outlier_scale
        step_scale = spec.per_tier_step_scale.get(tier.name, spec.step_scale) if spec.per_tier_step_scale else spec.step_scale
        trajectories_per_class = int(spec.tier_counts.get(tier.name, tier.trajectories_per_class))
        rows.append(
            DatasetTierDefinition(
                name=tier.name,
                description=tier.description,
                trajectories_per_class=max(2, trajectories_per_class),
                steps_range=_scale_range(tier.steps_range, step_scale, integral=True),
                dt_range=tier.dt_range,
                measurement_std_range=_scale_range(tier.measurement_std_range, measurement_scale),
                outlier_probability=_clamp(tier.outlier_probability * outlier_scale, 0.0, 0.30),
                dropout_probability=tier.dropout_probability,
                irregular_sampling_strength=_clamp(tier.irregular_sampling_strength * irregularity_scale, 0.0, 1.0),
                parameter_mode=tier.parameter_mode,
            )
        )
    return tuple(rows)


def _difficulty_distribution_rows(datasets) -> list[dict[str, object]]:
    total = sum(len(dataset.trajectories) for dataset in datasets)
    rows = []
    for dataset in datasets:
        count = len(dataset.trajectories)
        rows.append(
            {
                "tier": dataset.tier,
                "trajectory_count": count,
                "fraction": count / max(total, 1),
            }
        )
    return rows


def _difficulty_diversity_score(distribution_rows: list[dict[str, object]], objectives: dict[str, object]) -> float:
    target = objectives.get("difficulty_distribution", {})
    total_abs_error = 0.0
    for row in distribution_rows:
        tier = str(row["tier"]).replace("_v1", "")
        target_fraction = float(target.get(f"{tier}_fraction", 0.0))
        total_abs_error += abs(float(row["fraction"]) - target_fraction)
    return _clamp(1.0 - total_abs_error / 2.0, 0.0, 1.0)


def _feature_excitation_score(adequacy: CorpusAdequacyResult) -> float:
    if not adequacy.feature_set_rows:
        return 0.0
    score_terms = []
    for row in adequacy.feature_set_rows:
        score_terms.append(0.65 * float(row["moderate_or_strong_fraction"]) + 0.35 * _status_score(str(row["status"])))
    return _clamp(mean(score_terms), 0.0, 1.0)


def _boundary_coverage_score(adequacy: CorpusAdequacyResult) -> float:
    if not adequacy.class_pair_rows:
        return 0.0
    return _clamp(mean(_status_score(str(row["status"])) for row in adequacy.class_pair_rows), 0.0, 1.0)


def _balance_score(adequacy: CorpusAdequacyResult) -> float:
    if not adequacy.class_balance_rows:
        return 0.0
    return _clamp(mean(_status_score(str(row["status"])) for row in adequacy.class_balance_rows), 0.0, 1.0)


def _leakage_penalty(adequacy: CorpusAdequacyResult, objectives: dict[str, object]) -> float:
    leakage_objectives = objectives.get("covariate_leakage", {})
    limit = max(
        float(leakage_objectives.get("max_duration_class_correlation", 0.20)),
        float(leakage_objectives.get("max_sample_count_class_correlation", 0.20)),
        float(leakage_objectives.get("max_noise_class_correlation", 0.20)),
    )
    penalties = []
    for row in adequacy.covariate_rows:
        auc_excess = max(0.0, float(row["max_pairwise_auc"]) - (0.5 + limit))
        spread_excess = max(0.0, float(row["spread_ratio"]) - 1.0)
        penalties.append(1.5 * auc_excess + 0.35 * spread_excess + (1.0 - _status_score(str(row["status"]))) * 0.5)
    return _clamp(mean(penalties) if penalties else 0.0, 0.0, 1.0)


def _triviality_penalty(adequacy: CorpusAdequacyResult) -> float:
    penalties = []
    for row in adequacy.class_pair_rows:
        difficulty = str(row["expected_difficulty"])
        auc = float(row["pairwise_auc"])
        overlap = float(row["overlap_estimate"])
        if difficulty in {"hard", "short_horizon_boundary"}:
            penalties.append(max(0.0, auc - 0.95) + max(0.0, 0.12 - overlap))
        elif difficulty == "easy":
            penalties.append(max(0.0, 0.90 - auc) + max(0.0, overlap - 0.25))
        else:
            penalties.append(max(0.0, auc - 0.97) * 0.5)
    return _clamp(mean(penalties) if penalties else 0.0, 0.0, 1.0)


def _degeneracy_penalty(adequacy: CorpusAdequacyResult) -> float:
    if not adequacy.feature_set_rows:
        return 1.0
    red_fraction = sum(1 for row in adequacy.feature_set_rows if row["status"] == "red") / len(adequacy.feature_set_rows)
    low_excitation_fraction = sum(1 for row in adequacy.feature_set_rows if float(row["moderate_or_strong_fraction"]) < 0.20) / len(adequacy.feature_set_rows)
    return _clamp(0.7 * red_fraction + 0.3 * low_excitation_fraction, 0.0, 1.0)


def _pareto_objectives(score_row: dict[str, object]) -> dict[str, float]:
    return {
        "balance_score": float(score_row["balance_score"]),
        "boundary_coverage_score": float(score_row["boundary_coverage_score"]),
        "feature_excitation_score": float(score_row["feature_excitation_score"]),
        "difficulty_diversity_score": float(score_row["difficulty_diversity_score"]),
        "negative_leakage_penalty": -float(score_row["leakage_penalty"]),
        "negative_triviality_penalty": -float(score_row["triviality_penalty"]),
        "negative_degeneracy_penalty": -float(score_row["degeneracy_penalty"]),
    }


def _is_dominated(left: dict[str, float], right: dict[str, float]) -> bool:
    return all(right[key] >= left[key] for key in left) and any(right[key] > left[key] for key in left)


def _pareto_front_rows(evaluations: list) -> list[dict[str, object]]:
    rows = []
    for evaluation in evaluations:
        dominated = False
        for other in evaluations:
            if other.spec.candidate_id == evaluation.spec.candidate_id:
                continue
            if _is_dominated(evaluation.pareto_objectives, other.pareto_objectives):
                dominated = True
                break
        if not dominated:
            rows.append(
                {
                    "candidate_id": evaluation.spec.candidate_id,
                    "overall_score": evaluation.score_row["overall_score"],
                    "balance_score": evaluation.score_row["balance_score"],
                    "boundary_coverage_score": evaluation.score_row["boundary_coverage_score"],
                    "feature_excitation_score": evaluation.score_row["feature_excitation_score"],
                    "difficulty_diversity_score": evaluation.score_row["difficulty_diversity_score"],
                    "leakage_penalty": evaluation.score_row["leakage_penalty"],
                    "triviality_penalty": evaluation.score_row["triviality_penalty"],
                    "degeneracy_penalty": evaluation.score_row["degeneracy_penalty"],
                }
            )
    rows.sort(key=lambda row: float(row["overall_score"]), reverse=True)
    return rows


def _candidate_manifest_row(spec: CorpusCandidateSpec, distribution_rows: list[dict[str, object]]) -> dict[str, object]:
    row = {
        "candidate_id": spec.candidate_id,
        "description": spec.description,
        "sampling_method": spec.sampling_method,
        "seed": spec.seed,
        "measurement_scale": spec.measurement_scale,
        "irregularity_scale": spec.irregularity_scale,
        "outlier_scale": spec.outlier_scale,
        "step_scale": spec.step_scale,
    }
    for tier_row in distribution_rows:
        tier = str(tier_row["tier"])
        row[f"{tier}_count"] = int(tier_row["trajectory_count"])
        row[f"{tier}_fraction"] = float(tier_row["fraction"])
    return row


def _feature_excitation_comparison_rows(candidate_id: str, adequacy: CorpusAdequacyResult) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in adequacy.feature_set_rows:
        grouped.setdefault(str(row["feature_set"]), []).append(row)
    rows = []
    for feature_set, feature_rows in sorted(grouped.items()):
        rows.append(
            {
                "candidate_id": candidate_id,
                "feature_set": feature_set,
                "mean_moderate_or_strong_fraction": mean(float(row["moderate_or_strong_fraction"]) for row in feature_rows),
                "strong_feature_fraction": sum(1 for row in feature_rows if int(row["strong_count"]) >= 5) / max(len(feature_rows), 1),
                "status_score": mean(_status_score(str(row["status"])) for row in feature_rows),
            }
        )
    return rows
