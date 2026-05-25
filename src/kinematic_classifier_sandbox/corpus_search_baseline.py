from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from pathlib import Path
import random

import yaml

from .corpus_gym import (
    CorpusGymAction,
    CorpusGymEnvironment,
    CorpusGymEpisode,
    CorpusGymTarget,
    default_corpus_gym_targets,
)


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass(frozen=True, slots=True)
class CorpusSearchBaselineResult:
    config: dict[str, object]
    generated_candidate_rows: tuple[dict[str, object], ...]
    candidate_score_rows: tuple[dict[str, object], ...]
    selected_candidate_rows: tuple[dict[str, object], ...]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class CorpusSearchBaselineArtifacts:
    run_dir: Path
    search_config_path: Path
    generated_candidates_path: Path
    candidate_scores_path: Path
    selected_candidates_path: Path
    report_path: Path


def _random_action(rng: random.Random, target: CorpusGymTarget, *, seed: int) -> CorpusGymAction:
    return CorpusGymAction(
        seed=seed,
        tier_name=target.target_tier or "realistic_v1",
        duration_scale=rng.uniform(0.75, 1.25),
        measurement_scale=rng.uniform(0.80, 1.30),
        irregularity_scale=rng.uniform(0.75, 1.35),
        outlier_scale=rng.uniform(0.75, 1.35),
        step_scale=rng.uniform(0.80, 1.20),
    )


def _doe_actions(target: CorpusGymTarget, *, base_seed: int) -> tuple[CorpusGymAction, ...]:
    grid = (
        (0.85, 0.90, 0.90, 0.85, 0.90),
        (0.95, 1.00, 1.10, 1.00, 0.95),
        (1.05, 1.10, 0.90, 1.20, 1.05),
        (1.15, 1.20, 1.25, 1.10, 1.10),
    )
    actions = []
    for index, (duration_scale, measurement_scale, irregularity_scale, outlier_scale, step_scale) in enumerate(grid):
        actions.append(
            CorpusGymAction(
                seed=base_seed + index,
                tier_name=target.target_tier or "realistic_v1",
                duration_scale=duration_scale,
                measurement_scale=measurement_scale,
                irregularity_scale=irregularity_scale,
                outlier_scale=outlier_scale,
                step_scale=step_scale,
            )
        )
    return tuple(actions)


def _episode_row(
    *,
    candidate_id: str,
    target: CorpusGymTarget,
    search_method: str,
    episode: CorpusGymEpisode,
) -> dict[str, object]:
    diagnostics = episode.diagnostics
    reward = episode.reward
    trajectory = episode.trajectory
    return {
        "candidate_id": candidate_id,
        "target_id": target.target_id,
        "target_type": target.target_type,
        "target_tier": target.target_tier or "",
        "target_class": target.class_name or "",
        "target_class_pair": " vs ".join(target.class_pair) if target.class_pair else "",
        "search_method": search_method,
        "seed": episode.action.seed,
        "tier_name": episode.action.tier_name,
        "duration_scale": episode.action.duration_scale,
        "measurement_scale": episode.action.measurement_scale,
        "irregularity_scale": episode.action.irregularity_scale,
        "outlier_scale": episode.action.outlier_scale,
        "step_scale": episode.action.step_scale,
        "trajectory_id": trajectory.trajectory_id,
        "generated_class": trajectory.true_class,
        "generated_tier": str(trajectory.generator_parameters.get("tier", "")),
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
        "position_range": diagnostics["position_range"],
        "speed_range": diagnostics["speed_range"],
        "acceleration_range": diagnostics["acceleration_range"],
        "monotonicity": diagnostics["monotonicity"],
        "sampling_irregularity": diagnostics["sampling_irregularity"],
        "num_samples": diagnostics["num_samples"],
    }


def analyze_corpus_search_baseline(
    *,
    seed: int = 7,
    random_candidates_per_target: int = 8,
    rejection_pool_per_target: int = 12,
) -> CorpusSearchBaselineResult:
    rng = random.Random(seed)
    environment = CorpusGymEnvironment()
    targets = default_corpus_gym_targets()
    generated_rows: list[dict[str, object]] = []

    config = {
        "environment_id": "corpus_gym_v1",
        "seed": seed,
        "search_methods": ["random", "doe_grid", "rejection_search"],
        "random_candidates_per_target": random_candidates_per_target,
        "rejection_pool_per_target": rejection_pool_per_target,
        "selection_policy": "best_total_utility_per_target_with_random_baseline_comparison",
    }

    for target_index, target in enumerate(targets):
        environment.reset(target)

        for candidate_index in range(random_candidates_per_target):
            action = _random_action(
                rng,
                target,
                seed=seed * 10_000 + target_index * 100 + candidate_index,
            )
            episode = environment.simulate(action)
            generated_rows.append(
                _episode_row(
                    candidate_id=f"{target.target_id}_random_{candidate_index}",
                    target=target,
                    search_method="random",
                    episode=episode,
                )
            )

        for candidate_index, action in enumerate(_doe_actions(target, base_seed=seed * 20_000 + target_index * 100)):
            episode = environment.simulate(action)
            generated_rows.append(
                _episode_row(
                    candidate_id=f"{target.target_id}_doe_{candidate_index}",
                    target=target,
                    search_method="doe_grid",
                    episode=episode,
                )
            )

        rejection_rows: list[dict[str, object]] = []
        for candidate_index in range(rejection_pool_per_target):
            action = _random_action(
                rng,
                target,
                seed=seed * 30_000 + target_index * 100 + candidate_index,
            )
            episode = environment.simulate(action)
            row = _episode_row(
                candidate_id=f"{target.target_id}_rejection_{candidate_index}",
                target=target,
                search_method="rejection_search",
                episode=episode,
            )
            rejection_rows.append(row)
        threshold = sorted(float(row["total_utility"]) for row in rejection_rows)[max(len(rejection_rows) // 2, 0)]
        generated_rows.extend(
            row for row in rejection_rows if float(row["class_validity"]) >= 0.45 and float(row["total_utility"]) >= threshold
        )

    generated_rows.sort(
        key=lambda row: (
            str(row["target_id"]),
            str(row["search_method"]),
            -float(row["total_utility"]),
        )
    )
    candidate_score_rows = [dict(row) for row in generated_rows]

    selected_rows: list[dict[str, object]] = []
    for target in targets:
        target_rows = [row for row in candidate_score_rows if row["target_id"] == target.target_id]
        random_rows = [row for row in target_rows if row["search_method"] == "random"]
        non_random_rows = [row for row in target_rows if row["search_method"] != "random"]
        random_average_utility = sum(float(row["total_utility"]) for row in random_rows) / max(len(random_rows), 1)
        if target.target_type == "target_feature_cell":
            best_row = max(
                non_random_rows or target_rows,
                key=lambda row: (float(row["feature_excitation"]), float(row["total_utility"])),
            )
        else:
            best_row = max(non_random_rows or target_rows, key=lambda row: float(row["total_utility"]))
        feature_excitation_gain = float(best_row["feature_excitation"]) - (
            sum(float(row["feature_excitation"]) for row in random_rows) / max(len(random_rows), 1)
        )
        selected_rows.append(
            {
                **best_row,
                "random_average_utility": random_average_utility,
                "utility_gain_vs_random": float(best_row["total_utility"]) - random_average_utility,
                "feature_excitation_gain_vs_random": feature_excitation_gain,
                "selected": True,
            }
        )

    mean_random_utility = sum(float(row["total_utility"]) for row in candidate_score_rows if row["search_method"] == "random") / max(
        sum(1 for row in candidate_score_rows if row["search_method"] == "random"),
        1,
    )
    mean_selected_utility = sum(float(row["total_utility"]) for row in selected_rows) / max(len(selected_rows), 1)
    best_candidate = max(selected_rows, key=lambda row: float(row["total_utility"]))
    report_markdown = "\n".join(
        [
            "# Corpus Search Baseline",
            "",
            "This artifact runs the first non-RL corpus-search baseline on top of CorpusGym.",
            "",
            "## Summary",
            "",
            f"- Targets evaluated: `{len(targets)}`",
            f"- Generated candidates: `{len(generated_rows)}`",
            f"- Mean random utility: `{mean_random_utility:.3f}`",
            f"- Mean selected utility: `{mean_selected_utility:.3f}`",
            f"- Best selected candidate: `{best_candidate['candidate_id']}` with utility `{float(best_candidate['total_utility']):.3f}`",
            "",
            "## Search Ladder Used",
            "",
            "- `random`: unguided baseline",
            "- `doe_grid`: deterministic design-of-experiments parameter sweep",
            "- `rejection_search`: oversample then keep only viable high-utility candidates",
            "",
            "## Acceptance Signal",
            "",
            f"- Selected candidates beat the random-average utility: `{mean_selected_utility > mean_random_utility}`",
            f"- At least one selected candidate improves feature excitation versus random baseline: `{any(float(row['feature_excitation_gain_vs_random']) > 0.0 for row in selected_rows)}`",
            "",
            "## Reading Notes",
            "",
            "- This M26 slice is parameter search only; it does not yet maintain archive cells or novelty pressure.",
            "- Rejection search currently filters on class validity and top-half total utility within its candidate pool.",
            "- The output tables are intended to feed M27 quality-diversity and M28 stress-search work rather than replace them.",
        ]
    )

    return CorpusSearchBaselineResult(
        config=config,
        generated_candidate_rows=tuple(generated_rows),
        candidate_score_rows=tuple(candidate_score_rows),
        selected_candidate_rows=tuple(selected_rows),
        report_markdown=report_markdown,
    )


def write_corpus_search_baseline_artifacts(
    output_dir: str | Path,
    *,
    result: CorpusSearchBaselineResult | None = None,
) -> CorpusSearchBaselineArtifacts:
    baseline = result or analyze_corpus_search_baseline()
    run_dir = Path(output_dir) / "corpus_search_baseline"
    run_dir.mkdir(parents=True, exist_ok=True)
    search_config_path = run_dir / "search_config.yaml"
    generated_candidates_path = run_dir / "generated_candidates.csv"
    candidate_scores_path = run_dir / "candidate_scores.csv"
    selected_candidates_path = run_dir / "selected_candidates.csv"
    report_path = run_dir / "search_baseline_report.md"

    search_config_path.write_text(yaml.safe_dump(baseline.config, sort_keys=False), encoding="utf-8")
    _write_csv(
        generated_candidates_path,
        list(baseline.generated_candidate_rows),
        list(baseline.generated_candidate_rows[0].keys()),
    )
    _write_csv(
        candidate_scores_path,
        list(baseline.candidate_score_rows),
        list(baseline.candidate_score_rows[0].keys()),
    )
    _write_csv(
        selected_candidates_path,
        list(baseline.selected_candidate_rows),
        list(baseline.selected_candidate_rows[0].keys()),
    )
    report_path.write_text(baseline.report_markdown, encoding="utf-8")

    return CorpusSearchBaselineArtifacts(
        run_dir=run_dir,
        search_config_path=search_config_path,
        generated_candidates_path=generated_candidates_path,
        candidate_scores_path=candidate_scores_path,
        selected_candidates_path=selected_candidates_path,
        report_path=report_path,
    )
