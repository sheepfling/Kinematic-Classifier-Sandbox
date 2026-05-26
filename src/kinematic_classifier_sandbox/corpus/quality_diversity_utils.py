from __future__ import annotations

import random

from .gym import CorpusGymAction, CorpusGymEnvironment, default_corpus_gym_targets
from .search_baseline import analyze_corpus_search_baseline


def _bucket(value: float, thresholds: tuple[float, float]) -> str:
    if value < thresholds[0]:
        return "low"
    if value < thresholds[1]:
        return "medium"
    return "high"


def _random_action_for_target(rng: random.Random, target_tier: str, seed: int) -> CorpusGymAction:
    return CorpusGymAction(
        seed=seed,
        tier_name=target_tier,
        duration_scale=rng.uniform(0.75, 1.30),
        measurement_scale=rng.uniform(0.80, 1.35),
        irregularity_scale=rng.uniform(0.75, 1.40),
        outlier_scale=rng.uniform(0.75, 1.40),
        step_scale=rng.uniform(0.80, 1.25),
    )


def _archive_cell_id(row: dict[str, object]) -> tuple[str, str, str, str, str]:
    return (
        str(row["generated_class"]),
        str(row["target_tier"]),
        _bucket(float(row["duration"]), (6.0, 12.0)),
        _bucket(float(row["acceleration_range"]), (0.35, 0.85)),
        _bucket(1.0 - float(row["monotonicity"]), (0.08, 0.22)),
    )


def _episode_row(
    *,
    iteration: int,
    candidate_id: str,
    target_id: str,
    target_type: str,
    target_tier: str,
    episode,
) -> dict[str, object]:
    reward = episode.reward
    diagnostics = episode.diagnostics
    return {
        "iteration": iteration,
        "candidate_id": candidate_id,
        "target_id": target_id,
        "target_type": target_type,
        "target_tier": target_tier,
        "trajectory_id": episode.trajectory.trajectory_id,
        "generated_class": episode.trajectory.true_class,
        "seed": episode.action.seed,
        "duration_scale": episode.action.duration_scale,
        "measurement_scale": episode.action.measurement_scale,
        "irregularity_scale": episode.action.irregularity_scale,
        "outlier_scale": episode.action.outlier_scale,
        "step_scale": episode.action.step_scale,
        "class_validity": reward.class_validity,
        "feature_excitation": reward.feature_excitation,
        "coverage_gain": reward.coverage_gain,
        "boundary_closeness": reward.boundary_closeness,
        "classifier_stress": reward.classifier_stress,
        "prior_sensitivity": reward.prior_sensitivity,
        "leakage_penalty": reward.leakage_penalty,
        "physical_invalidity_penalty": reward.physical_invalidity_penalty,
        "total_utility": reward.total_utility,
        "duration": diagnostics["duration"],
        "acceleration_range": diagnostics["acceleration_range"],
        "monotonicity": diagnostics["monotonicity"],
        "sampling_irregularity": diagnostics["sampling_irregularity"],
        "num_samples": diagnostics["num_samples"],
    }


def build_quality_diversity_corpus(
    *,
    seed: int = 7,
    iterations: int = 42,
) -> tuple[
    dict[str, object],
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
    dict[str, object],
    str,
]:
    rng = random.Random(seed)
    environment = CorpusGymEnvironment()
    targets = default_corpus_gym_targets()
    baseline = analyze_corpus_search_baseline(seed=seed)
    feature_target_random_rows = [
        row
        for row in baseline.candidate_score_rows
        if row["search_method"] == "random" and row["target_type"] == "target_feature_cell"
    ]
    random_feature_excitation_mean = sum(float(row["feature_excitation"]) for row in feature_target_random_rows) / max(
        len(feature_target_random_rows),
        1,
    )

    archive: dict[tuple[str, str, str, str, str], dict[str, object]] = {}
    coverage_rows: list[dict[str, object]] = []

    for iteration in range(iterations):
        target = targets[iteration % len(targets)]
        environment.reset(target)
        action = _random_action_for_target(
            rng,
            target.target_tier or "realistic_v1",
            seed=seed * 100_000 + iteration,
        )
        episode = environment.simulate(action)
        row = _episode_row(
            iteration=iteration,
            candidate_id=f"{target.target_id}_qd_{iteration}",
            target_id=target.target_id,
            target_type=target.target_type,
            target_tier=target.target_tier or "realistic_v1",
            episode=episode,
        )
        cell_id = _archive_cell_id(row)
        incumbent = archive.get(cell_id)
        if incumbent is None or float(row["total_utility"]) > float(incumbent["total_utility"]):
            archive[cell_id] = row
        elite_values = list(archive.values())
        coverage_rows.append(
            {
                "iteration": iteration,
                "num_cells_filled": len(archive),
                "coverage_fraction": len(archive) / 81.0,
                "mean_elite_utility": sum(float(value["total_utility"]) for value in elite_values) / max(len(elite_values), 1),
                "mean_elite_feature_excitation": sum(float(value["feature_excitation"]) for value in elite_values) / max(len(elite_values), 1),
            }
        )

    archive_elite_rows = sorted(archive.values(), key=lambda row: float(row["total_utility"]), reverse=True)
    archive_cell_rows = []
    for cell_id, elite_row in sorted(archive.items()):
        archive_cell_rows.append(
            {
                "cell_id": "|".join(cell_id),
                "generated_class": cell_id[0],
                "target_tier": cell_id[1],
                "duration_bucket": cell_id[2],
                "acceleration_bucket": cell_id[3],
                "direction_change_bucket": cell_id[4],
                "elite_candidate_id": elite_row["candidate_id"],
                "elite_total_utility": elite_row["total_utility"],
            }
        )

    feature_target_elites = [row for row in archive_elite_rows if row["target_type"] == "target_feature_cell"]
    elite_feature_excitation_mean = sum(float(row["feature_excitation"]) for row in feature_target_elites) / max(
        len(feature_target_elites),
        1,
    )
    best_feature_target_excitation = max(
        (float(row["feature_excitation"]) for row in feature_target_elites),
        default=0.0,
    )
    num_feature_elites_above_random_mean = sum(
        1 for row in feature_target_elites if float(row["feature_excitation"]) > random_feature_excitation_mean
    )
    corpus_manifest = {
        "archive_id": "quality_diversity_corpus_v1",
        "seed": seed,
        "iterations": iterations,
        "num_archive_cells": len(archive_cell_rows),
        "num_archive_elites": len(archive_elite_rows),
        "feature_target_elite_excitation_mean": elite_feature_excitation_mean,
        "best_feature_target_excitation": best_feature_target_excitation,
        "random_feature_excitation_mean": random_feature_excitation_mean,
        "num_feature_elites_above_random_mean": num_feature_elites_above_random_mean,
        "improves_feature_excitation_over_baseline": best_feature_target_excitation > random_feature_excitation_mean,
    }
    report_markdown = "\n".join(
        [
            "# Quality-Diversity Corpus",
            "",
            "This artifact builds the first archive-style corpus layer on top of CorpusGym.",
            "",
            "## Summary",
            "",
            f"- Iterations: `{iterations}`",
            f"- Archive cells filled: `{len(archive_cell_rows)}`",
            f"- Archive elites retained: `{len(archive_elite_rows)}`",
            f"- Final coverage fraction: `{coverage_rows[-1]['coverage_fraction']:.3f}`",
            f"- Mean elite utility: `{coverage_rows[-1]['mean_elite_utility']:.3f}`",
            f"- Feature-target elite excitation mean: `{elite_feature_excitation_mean:.3f}`",
            f"- Best feature-target excitation: `{best_feature_target_excitation:.3f}`",
            f"- Random feature-target excitation mean: `{random_feature_excitation_mean:.3f}`",
            f"- Feature-target elites above random mean: `{num_feature_elites_above_random_mean}`",
            f"- Improves feature excitation over baseline: `{best_feature_target_excitation > random_feature_excitation_mean}`",
            "",
            "## Archive Policy",
            "",
            "- One elite is retained per behavior cell.",
            "- Cells are keyed by generated class, target tier, duration bucket, acceleration-range bucket, and direction-change bucket.",
            "- Elite replacement is based on total utility.",
            "",
            "## Reading Notes",
            "",
            "- This M27 slice is MAP-Elites-like but intentionally minimal: it proves archive coverage and elite retention before novelty search or crossover logic.",
            "- The archive is still driven by one-trajectory episodes from CorpusGym rather than multi-trajectory corpus optimization.",
            "- Boundary and adversarial tiers are explicitly retained as separate behavior-cell axes rather than collapsed into one stress bucket.",
        ]
    )
    config = {
        "archive_id": "quality_diversity_corpus_v1",
        "seed": seed,
        "iterations": iterations,
        "cell_axes": [
            "generated_class",
            "target_tier",
            "duration_bucket",
            "acceleration_bucket",
            "direction_change_bucket",
        ],
        "elite_selection_metric": "total_utility",
    }
    return (
        config,
        tuple(archive_cell_rows),
        tuple(archive_elite_rows),
        tuple(coverage_rows),
        corpus_manifest,
        report_markdown,
    )
