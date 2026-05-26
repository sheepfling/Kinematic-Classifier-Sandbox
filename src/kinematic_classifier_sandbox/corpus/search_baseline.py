from __future__ import annotations

import random

from .gym import CorpusGymEnvironment, default_corpus_gym_targets
from .search_baseline_artifact_io import write_corpus_search_baseline_artifacts
from .search_baseline_contracts import (
    CorpusSearchBaselineArtifacts,
    CorpusSearchBaselineResult,
)
from .search_baseline_utils import _doe_actions, _episode_row, _random_action


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
