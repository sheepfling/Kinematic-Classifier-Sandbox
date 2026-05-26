from __future__ import annotations

from ..utils.math import _mean
from .adaptive_stress import analyze_adaptive_stress_corpus
from .gym import CorpusGymAction, CorpusGymEnvironment, default_corpus_gym_targets
from .quality_diversity import analyze_quality_diversity_corpus
from .rl_backend_decision import analyze_rl_backend_decision
from .search_baseline import analyze_corpus_search_baseline
from .synthesis_comparison_artifact_io import write_corpus_synthesis_comparison_artifacts
from .synthesis_comparison_contracts import (
    CorpusSynthesisComparisonArtifacts,
    CorpusSynthesisComparisonResult,
)


def _bucket(value: float, thresholds: tuple[float, float]) -> str:
    if value < thresholds[0]:
        return "low"
    if value < thresholds[1]:
        return "medium"
    return "high"


def _archive_cell_id(row: dict[str, object]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("true_class") or row.get("generated_class") or ""),
        str(row.get("tier_name") or row.get("target_tier") or ""),
        _bucket(float(row.get("duration", 0.0)), (6.0, 12.0)),
        _bucket(float(row.get("acceleration_range", 0.0)), (0.35, 0.85)),
        _bucket(float(row.get("sampling_irregularity", 0.0) + float(row.get("outlier_score", 0.0)) * 0.05), (0.12, 0.35)),
    )


def _manual_generator_rows(*, seed: int = 7, replicas_per_target: int = 3) -> list[dict[str, object]]:
    environment = CorpusGymEnvironment()
    rows: list[dict[str, object]] = []
    for target_index, target in enumerate(default_corpus_gym_targets()):
        for replica in range(replicas_per_target):
            action = CorpusGymAction(
                seed=seed * 1000 + target_index * 20 + replica,
                tier_name=target.target_tier or "realistic_v1",
                duration_scale=1.0,
                measurement_scale=1.0,
                irregularity_scale=1.0,
                outlier_scale=1.0,
                step_scale=1.0,
            )
            environment.reset(target)
            episode = environment.simulate(action)
            rows.append(
                {
                    "method_name": "manual_generator",
                    "true_class": episode.trajectory.true_class,
                    "target_tier": target.target_tier or "",
                    "duration": float(episode.diagnostics["duration"]),
                    "acceleration_range": float(episode.diagnostics["acceleration_range"]),
                    "sampling_irregularity": float(episode.diagnostics["sampling_irregularity"]),
                    "outlier_score": float(episode.diagnostics["outlier_score"]),
                    "class_validity": float(episode.reward.class_validity),
                    "feature_excitation": float(episode.reward.feature_excitation),
                    "boundary_closeness": float(episode.reward.boundary_closeness),
                    "classifier_stress": float(episode.reward.classifier_stress),
                    "prior_sensitivity": float(episode.reward.prior_sensitivity),
                    "leakage_penalty": float(episode.reward.leakage_penalty),
                    "physical_invalidity_penalty": float(episode.reward.physical_invalidity_penalty),
                    "total_utility": float(episode.reward.total_utility),
                    "stress_score": float(episode.reward.classifier_stress),
                }
            )
    return rows


def _rows_from_search(search, search_method: str, method_name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in search.candidate_score_rows:
        if row["search_method"] != search_method:
            continue
        rows.append(
            {
                "method_name": method_name,
                "true_class": row["generated_class"],
                "target_tier": row["target_tier"],
                "duration": float(row["duration"]),
                "acceleration_range": float(row["acceleration_range"]),
                "sampling_irregularity": float(row["sampling_irregularity"]),
                "outlier_score": float(row.get("outlier_score", 0.0) or 0.0),
                "class_validity": float(row["class_validity"]),
                "feature_excitation": float(row["feature_excitation"]),
                "boundary_closeness": float(row["boundary_closeness"]),
                "classifier_stress": float(row["classifier_stress"]),
                "prior_sensitivity": float(row["prior_sensitivity"]),
                "leakage_penalty": float(row["leakage_penalty"]),
                "physical_invalidity_penalty": float(row["physical_invalidity_penalty"]),
                "total_utility": float(row["total_utility"]),
                "stress_score": float(row["classifier_stress"]),
            }
        )
    return rows


def _rows_from_qd(qd) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in qd.archive_elite_rows:
        rows.append(
            {
                "method_name": "quality_diversity",
                "true_class": row["generated_class"],
                "target_tier": row["target_tier"],
                "duration": float(row["duration"]),
                "acceleration_range": float(row["acceleration_range"]),
                "sampling_irregularity": float(row["sampling_irregularity"]),
                "outlier_score": float(row.get("outlier_score", 0.0) or 0.0),
                "class_validity": float(row["class_validity"]),
                "feature_excitation": float(row["feature_excitation"]),
                "boundary_closeness": float(row["boundary_closeness"]),
                "classifier_stress": float(row["classifier_stress"]),
                "prior_sensitivity": float(row["prior_sensitivity"]),
                "leakage_penalty": float(row["leakage_penalty"]),
                "physical_invalidity_penalty": float(row["physical_invalidity_penalty"]),
                "total_utility": float(row["total_utility"]),
                "stress_score": float(row["classifier_stress"]),
            }
        )
    return rows


def _rows_from_stress(stress) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in stress.stress_score_rows:
        rows.append(
            {
                "method_name": "adaptive_stress",
                "true_class": row["true_class"],
                "target_tier": row["tier_name"],
                "duration": float(row["duration"]),
                "acceleration_range": float(row["acceleration_range"]),
                "sampling_irregularity": float(row["sampling_irregularity"]),
                "outlier_score": float(row["outlier_score"]),
                "class_validity": float(row["class_validity"]),
                "feature_excitation": float(row.get("feature_excitation", 0.0) or 0.0),
                "boundary_closeness": float(row["boundary_closeness"]),
                "classifier_stress": float(row["classifier_stress"]),
                "prior_sensitivity": float(row["prior_sensitivity"]),
                "leakage_penalty": float(row["leakage_penalty"]),
                "physical_invalidity_penalty": float(row["physical_invalidity_penalty"]),
                "total_utility": float(row["total_utility"]),
                "stress_score": float(row["stress_score"]),
            }
        )
    return rows


def _metric_row(method_name: str, rows: list[dict[str, object]]) -> dict[str, object]:
    class_counts: dict[str, int] = {}
    for row in rows:
        class_counts[str(row["true_class"])] = class_counts.get(str(row["true_class"]), 0) + 1
    total = sum(class_counts.values())
    if total == 0:
        class_balance = 0.0
    else:
        target_share = 1.0 / max(len(class_counts), 1)
        class_balance = 1.0 - max(abs((count / total) - target_share) for count in class_counts.values())
    archive_coverage = len({_archive_cell_id(row) for row in rows}) / max(len(rows), 1)
    invalid_rate = sum(1.0 for row in rows if float(row["class_validity"]) < 0.45 or float(row["physical_invalidity_penalty"]) > 0.0) / max(len(rows), 1)
    stress_rate = sum(1.0 for row in rows if float(row["stress_score"]) >= 0.40) / max(len(rows), 1)
    prior_rate = sum(1.0 for row in rows if float(row["prior_sensitivity"]) >= 0.15) / max(len(rows), 1)
    return {
        "method_name": method_name,
        "num_rows": len(rows),
        "class_balance": class_balance,
        "feature_excitation": _mean([float(row["feature_excitation"]) for row in rows]),
        "archive_coverage": archive_coverage,
        "boundary_coverage": sum(1.0 for row in rows if float(row["boundary_closeness"]) >= 0.35) / max(len(rows), 1),
        "leakage_penalty": _mean([float(row["leakage_penalty"]) for row in rows]),
        "invalid_trajectory_rate": invalid_rate,
        "classifier_error_discovery_rate": stress_rate,
        "prior_sensitive_case_discovery_rate": prior_rate,
        "mean_total_utility": _mean([float(row["total_utility"]) for row in rows]),
        "mean_stress_score": _mean([float(row["stress_score"]) for row in rows]),
    }


def analyze_corpus_synthesis_comparison(*, seed: int = 7) -> CorpusSynthesisComparisonResult:
    search = analyze_corpus_search_baseline(seed=seed)
    qd = analyze_quality_diversity_corpus(seed=seed, iterations=42)
    stress = analyze_adaptive_stress_corpus(seed=seed, random_candidates_per_mode=8, guided_candidates_per_mode=14)
    rl = analyze_rl_backend_decision()

    row_groups: dict[str, list[dict[str, object]]] = {
        "manual_generator": _manual_generator_rows(seed=seed),
        "random_search": _rows_from_search(search, "random", "random_search"),
        "doe_search": _rows_from_search(search, "doe_grid", "doe_search"),
        "rejection_search": _rows_from_search(search, "rejection_search", "rejection_search"),
        "quality_diversity": _rows_from_qd(qd),
        "adaptive_stress": _rows_from_stress(stress),
    }

    generator_rows = []
    for method_name, rows in row_groups.items():
        generator_rows.append(_metric_row(method_name, rows))
    generator_rows.append(
        {
            "method_name": "rl_backend",
            "num_rows": 0,
            "class_balance": "",
            "feature_excitation": "",
            "archive_coverage": "",
            "boundary_coverage": "",
            "leakage_penalty": "",
            "invalid_trajectory_rate": "",
            "classifier_error_discovery_rate": "",
            "prior_sensitive_case_discovery_rate": "",
            "mean_total_utility": "",
            "mean_stress_score": "",
            "rl_justified": rl.rl_justified,
        }
    )

    corpus_quality_rows = [
        {
            "method_name": row["method_name"],
            "class_balance": row["class_balance"],
            "archive_coverage": row["archive_coverage"],
            "leakage_penalty": row["leakage_penalty"],
            "invalid_trajectory_rate": row["invalid_trajectory_rate"],
            "mean_total_utility": row["mean_total_utility"],
        }
        for row in generator_rows
        if row["method_name"] != "rl_backend"
    ]
    feature_excitation_rows = [
        {
            "method_name": row["method_name"],
            "feature_excitation": row["feature_excitation"],
            "boundary_coverage": row["boundary_coverage"],
            "prior_sensitive_case_discovery_rate": row["prior_sensitive_case_discovery_rate"],
        }
        for row in generator_rows
        if row["method_name"] != "rl_backend"
    ]
    classifier_stress_rows = [
        {
            "method_name": row["method_name"],
            "classifier_error_discovery_rate": row["classifier_error_discovery_rate"],
            "mean_stress_score": row["mean_stress_score"],
        }
        for row in generator_rows
        if row["method_name"] != "rl_backend"
    ]

    best_quality = max(corpus_quality_rows, key=lambda row: float(row["mean_total_utility"]))
    best_coverage = max(corpus_quality_rows, key=lambda row: float(row["archive_coverage"]))
    best_stress = max(classifier_stress_rows, key=lambda row: float(row["mean_stress_score"]))
    report_markdown = "\n".join(
        [
            "# Corpus Synthesis Comparison",
            "",
            "Milestone 30 comparison across the currently implemented corpus-synthesis families.",
            "",
            "## Summary",
            "",
            f"- Best mean corpus utility: `{best_quality['method_name']}` at `{float(best_quality['mean_total_utility']):.3f}`",
            f"- Best archive coverage proxy: `{best_coverage['method_name']}` at `{float(best_coverage['archive_coverage']):.3f}`",
            f"- Best classifier stress discovery: `{best_stress['method_name']}` at `{float(best_stress['mean_stress_score']):.3f}`",
            f"- RL backend justified now: `{rl.rl_justified}`",
            "",
            "## Notes",
            "",
            "- `manual_generator` is the fixed-parameter CorpusGym baseline with no search pressure.",
            "- `random_search`, `doe_search`, and `rejection_search` come from the M26 search baseline.",
            "- `quality_diversity` uses archive elites from M27.",
            "- `adaptive_stress` uses the full M28 stress-search score surface.",
            "- `rl_backend` is included as a decision status row rather than a performance row because RL is currently a no-go and no trained policy exists.",
        ]
    )
    return CorpusSynthesisComparisonResult(
        generator_rows=tuple(generator_rows),
        corpus_quality_rows=tuple(corpus_quality_rows),
        feature_excitation_rows=tuple(feature_excitation_rows),
        classifier_stress_rows=tuple(classifier_stress_rows),
        report_markdown=report_markdown,
    )
