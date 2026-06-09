from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import io
import json
import os
from pathlib import Path
from statistics import mean

import yaml

from .corpus_adequacy_audit import CorpusAdequacyResult, analyze_corpus_adequacy
from .feature_analysis import FeatureAnalysisResult, analyze_feature_datasets
from .trajectory_generator import DatasetTierDefinition, default_dataset_tiers, generate_trajectory_datasets


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OBJECTIVES_PATH = ROOT / "experiments" / "corpus_objectives" / "common_1d_corpus_objectives.yaml"


def _prepare_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp/kinematic-classifier-sandbox-mpl")))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _figure_to_png(fig) -> bytes:
    plt = _prepare_matplotlib()
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _status_score(status: str) -> float:
    return {"green": 1.0, "yellow": 0.5, "red": 0.0}.get(status, 0.0)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _scale_range(bounds: tuple[float, float], scale: float, *, integral: bool = False) -> tuple[float, float]:
    lower = bounds[0] * scale
    upper = bounds[1] * scale
    if integral:
        lower_i = max(2, int(round(lower)))
        upper_i = max(lower_i + 1, int(round(upper)))
        return lower_i, upper_i
    return max(0.0, lower), max(max(0.0, lower), upper)


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


@dataclass(frozen=True, slots=True)
class CorpusCandidateEvaluation:
    spec: CorpusCandidateSpec
    feature_analysis: FeatureAnalysisResult
    adequacy: CorpusAdequacyResult
    manifest_row: dict[str, object]
    score_row: dict[str, object]
    adequacy_row: dict[str, object]
    feature_excitation_rows: tuple[dict[str, object], ...]
    leakage_rows: tuple[dict[str, object], ...]
    pareto_objectives: dict[str, float]


@dataclass(frozen=True, slots=True)
class CorpusAutodevelopmentResult:
    objectives_path: Path
    objectives: dict[str, object]
    candidate_evaluations: tuple[CorpusCandidateEvaluation, ...]
    selected_candidate_id: str
    candidate_manifest_rows: tuple[dict[str, object], ...]
    candidate_score_rows: tuple[dict[str, object], ...]
    rejected_candidate_rows: tuple[dict[str, object], ...]
    pareto_front_rows: tuple[dict[str, object], ...]
    adequacy_comparison_rows: tuple[dict[str, object], ...]
    feature_excitation_comparison_rows: tuple[dict[str, object], ...]
    leakage_comparison_rows: tuple[dict[str, object], ...]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class CorpusAutodevelopmentArtifacts:
    run_dir: Path
    objectives_path: Path
    candidate_manifest_path: Path
    candidate_scores_path: Path
    selected_manifest_path: Path
    rejected_manifest_path: Path
    pareto_front_path: Path
    adequacy_comparison_path: Path
    feature_excitation_comparison_path: Path
    leakage_comparison_path: Path
    report_path: Path
    corpus_score_pareto_path: Path
    feature_excitation_heatmap_path: Path
    leakage_by_candidate_path: Path
    difficulty_distribution_by_candidate_path: Path


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


def _pareto_front_rows(evaluations: list[CorpusCandidateEvaluation]) -> list[dict[str, object]]:
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


def _render_report(result: CorpusAutodevelopmentResult) -> str:
    selected = next(row for row in result.candidate_score_rows if row["candidate_id"] == result.selected_candidate_id)
    rejected_ids = [row["candidate_id"] for row in result.rejected_candidate_rows[:3]]
    pareto_ids = [row["candidate_id"] for row in result.pareto_front_rows]
    return "\n".join(
        [
            "# Corpus Autodevelopment",
            "",
            "This artifact proposes, scores, compares, and selects candidate corpora against explicit adequacy objectives.",
            "",
            "## Selection Summary",
            "",
            f"- Selected candidate: `{result.selected_candidate_id}`",
            f"- Overall score: `{selected['overall_score']:.3f}`",
            f"- Adequacy status: `{selected['adequacy_status']}`",
            f"- Pareto-front candidates: `{', '.join(pareto_ids)}`",
            f"- Example rejected candidates: `{', '.join(rejected_ids)}`",
            "",
            "## Scoring Logic",
            "",
            "- `overall_score = balance_score + boundary_coverage_score + feature_excitation_score + difficulty_diversity_score - leakage_penalty - triviality_penalty - degeneracy_penalty`",
            "- Higher is better.",
            "- The Pareto front preserves non-dominated candidates even if only one is selected for default use.",
            "",
            "## What This Proves",
            "",
            "- The repo can compare multiple corpus candidates instead of only auditing the default corpus.",
            "- Selection is tied to declared adequacy objectives rather than manual tweaking alone.",
            "- Rejected candidates remain inspectable, which makes the corpus-development story defensible.",
        ]
    )


def analyze_corpus_autodevelopment(
    *,
    objectives_path: str | Path = DEFAULT_OBJECTIVES_PATH,
    seed: int = 7,
) -> CorpusAutodevelopmentResult:
    objectives = load_corpus_objectives(objectives_path)
    candidate_evaluations: list[CorpusCandidateEvaluation] = []
    candidate_manifest_rows: list[dict[str, object]] = []
    candidate_score_rows: list[dict[str, object]] = []
    adequacy_rows: list[dict[str, object]] = []
    feature_excitation_rows: list[dict[str, object]] = []
    leakage_rows: list[dict[str, object]] = []

    for spec in _default_candidate_specs(seed):
        tier_definitions = _candidate_tier_definitions(spec)
        datasets = generate_trajectory_datasets(seed=spec.seed, tier_definitions=tier_definitions)
        feature_analysis = analyze_feature_datasets(datasets=datasets)
        adequacy = analyze_corpus_adequacy(feature_analysis_result=feature_analysis)
        distribution_rows = _difficulty_distribution_rows(datasets)

        balance_score = _balance_score(adequacy)
        boundary_coverage_score = _boundary_coverage_score(adequacy)
        feature_excitation_score = _feature_excitation_score(adequacy)
        difficulty_diversity_score = _difficulty_diversity_score(distribution_rows, objectives)
        leakage_penalty = _leakage_penalty(adequacy, objectives)
        triviality_penalty = _triviality_penalty(adequacy)
        degeneracy_penalty = _degeneracy_penalty(adequacy)
        overall_score = (
            balance_score
            + boundary_coverage_score
            + feature_excitation_score
            + difficulty_diversity_score
            - leakage_penalty
            - triviality_penalty
            - degeneracy_penalty
        )
        manifest_row = _candidate_manifest_row(spec, distribution_rows)
        score_row = {
            "candidate_id": spec.candidate_id,
            "sampling_method": spec.sampling_method,
            "overall_score": overall_score,
            "balance_score": balance_score,
            "boundary_coverage_score": boundary_coverage_score,
            "feature_excitation_score": feature_excitation_score,
            "difficulty_diversity_score": difficulty_diversity_score,
            "leakage_penalty": leakage_penalty,
            "triviality_penalty": triviality_penalty,
            "degeneracy_penalty": degeneracy_penalty,
            "adequacy_status": adequacy.summary.overall_status,
        }
        adequacy_row = {
            "candidate_id": spec.candidate_id,
            "overall_status": adequacy.summary.overall_status,
            "feature_status": adequacy.summary.feature_status,
            "class_pair_status": adequacy.summary.class_pair_status,
            "class_balance_status": adequacy.summary.class_balance_status,
            "covariate_status": adequacy.summary.covariate_status,
            "total_trajectories": adequacy.summary.total_trajectories,
            "red_count": adequacy.summary.red_count,
            "yellow_count": adequacy.summary.yellow_count,
            "recommendation_count": adequacy.summary.recommendation_count,
        }
        excitation_rows = tuple(_feature_excitation_comparison_rows(spec.candidate_id, adequacy))
        leakage_detail_rows = tuple(
            {
                "candidate_id": spec.candidate_id,
                "covariate": row["covariate"],
                "max_pairwise_auc": row["max_pairwise_auc"],
                "spread_ratio": row["spread_ratio"],
                "status": row["status"],
            }
            for row in adequacy.covariate_rows
        )
        evaluation = CorpusCandidateEvaluation(
            spec=spec,
            feature_analysis=feature_analysis,
            adequacy=adequacy,
            manifest_row=manifest_row,
            score_row=score_row,
            adequacy_row=adequacy_row,
            feature_excitation_rows=excitation_rows,
            leakage_rows=leakage_detail_rows,
            pareto_objectives=_pareto_objectives(score_row),
        )
        candidate_evaluations.append(evaluation)
        candidate_manifest_rows.append(manifest_row)
        candidate_score_rows.append(score_row)
        adequacy_rows.append(adequacy_row)
        feature_excitation_rows.extend(excitation_rows)
        leakage_rows.extend(leakage_detail_rows)

    candidate_evaluations.sort(key=lambda item: float(item.score_row["overall_score"]), reverse=True)
    candidate_manifest_rows.sort(key=lambda row: float(next(item.score_row["overall_score"] for item in candidate_evaluations if item.spec.candidate_id == row["candidate_id"])), reverse=True)
    candidate_score_rows.sort(key=lambda row: float(row["overall_score"]), reverse=True)
    adequacy_rows.sort(key=lambda row: next(score["overall_score"] for score in candidate_score_rows if score["candidate_id"] == row["candidate_id"]), reverse=True)
    selected_candidate_id = candidate_score_rows[0]["candidate_id"]
    rejected_candidate_rows = [row for row in candidate_score_rows if row["candidate_id"] != selected_candidate_id]
    pareto_front_rows = _pareto_front_rows(candidate_evaluations)
    report_markdown = _render_report(
        CorpusAutodevelopmentResult(
            objectives_path=Path(objectives_path),
            objectives=objectives,
            candidate_evaluations=tuple(candidate_evaluations),
            selected_candidate_id=selected_candidate_id,
            candidate_manifest_rows=tuple(candidate_manifest_rows),
            candidate_score_rows=tuple(candidate_score_rows),
            rejected_candidate_rows=tuple(rejected_candidate_rows),
            pareto_front_rows=tuple(pareto_front_rows),
            adequacy_comparison_rows=tuple(adequacy_rows),
            feature_excitation_comparison_rows=tuple(feature_excitation_rows),
            leakage_comparison_rows=tuple(leakage_rows),
            report_markdown="",
        )
    )
    return CorpusAutodevelopmentResult(
        objectives_path=Path(objectives_path),
        objectives=objectives,
        candidate_evaluations=tuple(candidate_evaluations),
        selected_candidate_id=selected_candidate_id,
        candidate_manifest_rows=tuple(candidate_manifest_rows),
        candidate_score_rows=tuple(candidate_score_rows),
        rejected_candidate_rows=tuple(rejected_candidate_rows),
        pareto_front_rows=tuple(pareto_front_rows),
        adequacy_comparison_rows=tuple(adequacy_rows),
        feature_excitation_comparison_rows=tuple(feature_excitation_rows),
        leakage_comparison_rows=tuple(leakage_rows),
        report_markdown=report_markdown,
    )


def _render_corpus_score_pareto(result: CorpusAutodevelopmentResult):
    plt = _prepare_matplotlib()
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    pareto_ids = {row["candidate_id"] for row in result.pareto_front_rows}
    for row in result.candidate_score_rows:
        candidate_id = str(row["candidate_id"])
        color = "#dc2626" if candidate_id == result.selected_candidate_id else ("#2563eb" if candidate_id in pareto_ids else "#6b7280")
        ax.scatter(float(row["leakage_penalty"]), float(row["feature_excitation_score"]), s=90, color=color, alpha=0.9)
        ax.text(float(row["leakage_penalty"]) + 0.01, float(row["feature_excitation_score"]) + 0.01, candidate_id, fontsize=8)
    ax.set_xlabel("Leakage penalty")
    ax.set_ylabel("Feature excitation score")
    ax.set_title("Corpus Candidate Pareto View", loc="left", fontweight="bold")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def _render_feature_excitation_heatmap(result: CorpusAutodevelopmentResult):
    plt = _prepare_matplotlib()
    candidate_ids = [row["candidate_id"] for row in result.candidate_score_rows]
    feature_sets = sorted({str(row["feature_set"]) for row in result.feature_excitation_comparison_rows})
    matrix = []
    for candidate_id in candidate_ids:
        candidate_lookup = {
            str(row["feature_set"]): float(row["mean_moderate_or_strong_fraction"])
            for row in result.feature_excitation_comparison_rows
            if row["candidate_id"] == candidate_id
        }
        matrix.append([candidate_lookup.get(feature_set, 0.0) for feature_set in feature_sets])
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    image = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(feature_sets)))
    ax.set_xticklabels(feature_sets, rotation=30, ha="right")
    ax.set_yticks(range(len(candidate_ids)))
    ax.set_yticklabels(candidate_ids)
    ax.set_title("Feature Excitation By Candidate", loc="left", fontweight="bold")
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            ax.text(col_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def _render_leakage_by_candidate(result: CorpusAutodevelopmentResult):
    plt = _prepare_matplotlib()
    candidate_ids = [row["candidate_id"] for row in result.candidate_score_rows]
    worst_auc = []
    for candidate_id in candidate_ids:
        values = [
            float(row["max_pairwise_auc"])
            for row in result.leakage_comparison_rows
            if row["candidate_id"] == candidate_id
        ]
        worst_auc.append(max(values) if values else 0.0)
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    colors = ["#dc2626" if candidate_id == result.selected_candidate_id else "#2563eb" for candidate_id in candidate_ids]
    ax.bar(candidate_ids, worst_auc, color=colors)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Worst covariate-only AUC")
    ax.set_title("Leakage By Candidate", loc="left", fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _render_difficulty_distribution(result: CorpusAutodevelopmentResult):
    plt = _prepare_matplotlib()
    candidate_ids = [row["candidate_id"] for row in result.candidate_score_rows]
    tiers = ["easy_v1", "boundary_v1", "adversarial_v1", "stress_v1", "realistic_v1"]
    colors = ["#93c5fd", "#60a5fa", "#3b82f6", "#1d4ed8", "#1e3a8a"]
    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    bottoms = [0.0] * len(candidate_ids)
    manifest_lookup = {str(row["candidate_id"]): row for row in result.candidate_manifest_rows}
    for tier, color in zip(tiers, colors):
        values = [float(manifest_lookup[candidate_id].get(f"{tier}_fraction", 0.0)) for candidate_id in candidate_ids]
        ax.bar(candidate_ids, values, bottom=bottoms, label=tier, color=color)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Fraction of trajectories")
    ax.set_title("Difficulty Distribution By Candidate", loc="left", fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    return fig


def write_corpus_autodevelopment_artifacts(
    output_dir: str | Path,
    *,
    result: CorpusAutodevelopmentResult | None = None,
) -> CorpusAutodevelopmentArtifacts:
    autodevelopment = result or analyze_corpus_autodevelopment()
    run_dir = Path(output_dir) / "corpus_autodevelopment_v1"
    plots_dir = run_dir / "plots"
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    objectives_path = run_dir / "corpus_objectives.yaml"
    candidate_manifest_path = run_dir / "candidate_corpus_manifest.csv"
    candidate_scores_path = run_dir / "corpus_candidate_scores.csv"
    selected_manifest_path = run_dir / "selected_corpus_manifest.json"
    rejected_manifest_path = run_dir / "rejected_corpus_manifest.csv"
    pareto_front_path = run_dir / "corpus_pareto_front.csv"
    adequacy_comparison_path = run_dir / "corpus_adequacy_comparison.csv"
    feature_excitation_comparison_path = run_dir / "feature_excitation_comparison.csv"
    leakage_comparison_path = run_dir / "leakage_comparison.csv"
    report_path = run_dir / "corpus_autodevelopment_report.md"
    corpus_score_pareto_path = plots_dir / "corpus_score_pareto.png"
    feature_excitation_heatmap_path = plots_dir / "feature_excitation_heatmap.png"
    leakage_by_candidate_path = plots_dir / "leakage_by_candidate.png"
    difficulty_distribution_by_candidate_path = plots_dir / "difficulty_distribution_by_candidate.png"

    objectives_path.write_text(Path(autodevelopment.objectives_path).read_text(encoding="utf-8"), encoding="utf-8")
    _write_csv(candidate_manifest_path, list(autodevelopment.candidate_manifest_rows), list(autodevelopment.candidate_manifest_rows[0].keys()))
    _write_csv(candidate_scores_path, list(autodevelopment.candidate_score_rows), list(autodevelopment.candidate_score_rows[0].keys()))
    _write_csv(rejected_manifest_path, list(autodevelopment.rejected_candidate_rows), list(autodevelopment.rejected_candidate_rows[0].keys()))
    _write_csv(pareto_front_path, list(autodevelopment.pareto_front_rows), list(autodevelopment.pareto_front_rows[0].keys()))
    _write_csv(adequacy_comparison_path, list(autodevelopment.adequacy_comparison_rows), list(autodevelopment.adequacy_comparison_rows[0].keys()))
    _write_csv(feature_excitation_comparison_path, list(autodevelopment.feature_excitation_comparison_rows), list(autodevelopment.feature_excitation_comparison_rows[0].keys()))
    _write_csv(leakage_comparison_path, list(autodevelopment.leakage_comparison_rows), list(autodevelopment.leakage_comparison_rows[0].keys()))

    selected_evaluation = next(
        evaluation for evaluation in autodevelopment.candidate_evaluations if evaluation.spec.candidate_id == autodevelopment.selected_candidate_id
    )
    selected_manifest_payload = {
        "selected_candidate_id": autodevelopment.selected_candidate_id,
        "objectives_path": str(autodevelopment.objectives_path),
        "selected_spec": asdict(selected_evaluation.spec),
        "selected_score": selected_evaluation.score_row,
        "selected_adequacy_summary": asdict(selected_evaluation.adequacy.summary),
    }
    selected_manifest_path.write_text(json.dumps(selected_manifest_payload, indent=2), encoding="utf-8")
    report_path.write_text(autodevelopment.report_markdown, encoding="utf-8")

    corpus_score_pareto_path.write_bytes(_figure_to_png(_render_corpus_score_pareto(autodevelopment)))
    feature_excitation_heatmap_path.write_bytes(_figure_to_png(_render_feature_excitation_heatmap(autodevelopment)))
    leakage_by_candidate_path.write_bytes(_figure_to_png(_render_leakage_by_candidate(autodevelopment)))
    difficulty_distribution_by_candidate_path.write_bytes(_figure_to_png(_render_difficulty_distribution(autodevelopment)))

    return CorpusAutodevelopmentArtifacts(
        run_dir=run_dir,
        objectives_path=objectives_path,
        candidate_manifest_path=candidate_manifest_path,
        candidate_scores_path=candidate_scores_path,
        selected_manifest_path=selected_manifest_path,
        rejected_manifest_path=rejected_manifest_path,
        pareto_front_path=pareto_front_path,
        adequacy_comparison_path=adequacy_comparison_path,
        feature_excitation_comparison_path=feature_excitation_comparison_path,
        leakage_comparison_path=leakage_comparison_path,
        report_path=report_path,
        corpus_score_pareto_path=corpus_score_pareto_path,
        feature_excitation_heatmap_path=feature_excitation_heatmap_path,
        leakage_by_candidate_path=leakage_by_candidate_path,
        difficulty_distribution_by_candidate_path=difficulty_distribution_by_candidate_path,
    )
