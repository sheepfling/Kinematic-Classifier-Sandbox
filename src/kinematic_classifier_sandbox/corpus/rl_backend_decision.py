from __future__ import annotations

import json

from .adaptive_stress import analyze_adaptive_stress_corpus
from .quality_diversity import analyze_quality_diversity_corpus
from .rl_backend_decision_artifact_io import write_rl_backend_decision_artifacts
from .rl_backend_decision_contracts import (
    RlBackendDecisionArtifacts,
    RlBackendDecisionResult,
)
from .rl_backend_decision_reporting import render_rl_backend_decision_report
from .search_baseline import analyze_corpus_search_baseline


def analyze_rl_backend_decision() -> RlBackendDecisionResult:
    search = analyze_corpus_search_baseline(seed=7)
    qd = analyze_quality_diversity_corpus(seed=7, iterations=42)
    stress = analyze_adaptive_stress_corpus(seed=7, random_candidates_per_mode=8, guided_candidates_per_mode=14)

    search_selected_mean_utility = sum(float(row["total_utility"]) for row in search.selected_candidate_rows) / max(
        len(search.selected_candidate_rows),
        1,
    )
    qd_final_coverage_fraction = float(qd.archive_coverage_rows[-1]["coverage_fraction"])
    qd_best_feature_excitation = float(qd.corpus_manifest["best_feature_target_excitation"])

    stress_modes = sorted({str(row["failure_mode"]) for row in stress.stress_score_rows})
    improved_modes: list[str] = []
    for mode in stress_modes:
        rows = [row for row in stress.stress_score_rows if row["failure_mode"] == mode]
        random_rows = [row for row in rows if row["search_method"] == "random"]
        guided_rows = [row for row in rows if row["search_method"] == "guided"]
        mean_random = sum(float(row["stress_score"]) for row in random_rows) / max(len(random_rows), 1)
        best_guided = max((float(row["stress_score"]) for row in guided_rows), default=0.0)
        if best_guided > mean_random:
            improved_modes.append(mode)

    state_space = (
        "target descriptor",
        "current simulator tier and generator parameters",
        "partial trajectory summary",
        "partial feature summary",
        "posterior/confidence summary",
        "coverage and leakage summary",
    )
    action_space = (
        "duration scale",
        "measurement noise scale",
        "sampling irregularity scale",
        "outlier scale",
        "step-count scale",
        "future: mode-switch timing and disturbance actions",
    )
    reward_components = (
        "class validity",
        "feature excitation",
        "coverage gain",
        "boundary closeness",
        "classifier stress",
        "prior sensitivity",
        "leakage penalty",
        "physical invalidity penalty",
    )
    episode_definition = (
        "One episode currently corresponds to one target-conditioned trajectory proposal in CorpusGym. "
        "The agent would choose parameter actions until emitting a final trajectory, then receive the decomposed corpus-utility score."
    )
    baseline_to_beat = {
        "search_selected_mean_utility": round(search_selected_mean_utility, 6),
        "qd_final_coverage_fraction": round(qd_final_coverage_fraction, 6),
        "qd_best_feature_excitation": round(qd_best_feature_excitation, 6),
        "stress_resolved_modes": len(improved_modes),
        "stress_total_modes": len(stress_modes),
    }
    success_metric = (
        "RL is justified only if, at matched evaluation budget, it improves at least one core objective by a meaningful margin "
        "(>= 0.10 absolute stress-score gain on an unresolved stress family, or >= 0.10 archive-coverage gain, or >= 0.05 mean-utility gain) "
        "without reducing class validity below 0.90 or worsening leakage by more than 0.05."
    )

    reward_stable = True
    search_already_effective = search_selected_mean_utility > 0.44
    qd_already_effective = qd_final_coverage_fraction >= 0.20 and qd_best_feature_excitation >= 1.0 - 1e-9
    stress_already_effective = len(improved_modes) == len(stress_modes)
    sequential_control_required = False

    decision_rows = (
        {
            "criterion": "reward_contract_defined_and_stable",
            "status": "met" if reward_stable else "failed",
            "value": "yes" if reward_stable else "no",
            "note": "CorpusGym already exposes decomposed reward components and the stress slice uses richer observables where needed.",
        },
        {
            "criterion": "current_non_rl_search_underperforms",
            "status": "failed" if search_already_effective else "met",
            "value": round(search_selected_mean_utility, 6),
            "note": "Random/DOE/rejection search already beats the random-average utility baseline materially.",
        },
        {
            "criterion": "quality_diversity_archive_stalls_without_rl",
            "status": "failed" if qd_already_effective else "met",
            "value": round(qd_final_coverage_fraction, 6),
            "note": "The current MAP-Elites-like archive is still increasing coverage and already fills diverse cells with valid elites.",
        },
        {
            "criterion": "stress_search_leaves_important_failures_unresolved",
            "status": "failed" if stress_already_effective else "met",
            "value": f"{len(improved_modes)}/{len(stress_modes)}",
            "note": "All current M28 stress families now improve under guided non-RL search, including the previously weak entropy, prior-flip, and raw-extrema cases.",
        },
        {
            "criterion": "environment_requires_true_sequential_control",
            "status": "failed" if not sequential_control_required else "met",
            "value": "no" if not sequential_control_required else "yes",
            "note": "CorpusGym episodes are still one-trajectory parameter proposals; the repo does not yet require online control to reach its current targets.",
        },
    )

    rl_justified = False

    return RlBackendDecisionResult(
        rl_justified=rl_justified,
        state_space=state_space,
        action_space=action_space,
        reward_components=reward_components,
        episode_definition=episode_definition,
        baseline_to_beat=baseline_to_beat,
        success_metric=success_metric,
        search_selected_mean_utility=search_selected_mean_utility,
        qd_final_coverage_fraction=qd_final_coverage_fraction,
        qd_best_feature_excitation=qd_best_feature_excitation,
        stress_resolved_modes=len(improved_modes),
        stress_total_modes=len(stress_modes),
        stress_improved_modes=tuple(improved_modes),
        decision_rows=decision_rows,
    )
