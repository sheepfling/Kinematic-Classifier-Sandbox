from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

import yaml

from kinematic_classifier_sandbox.utils.io import write_csv

from ..inference.irregular_window_comparison import (
    WindowRegimeTrajectory,
    _duration_window,
    _sample_count_window,
    generate_window_regime_trajectories,
)
from ..inference.transition_matrix_accumulator import (
    _run_mode_accumulator,
    default_switching_mode_specs,
    default_transition_matrix,
    generate_transition_switching_scenarios,
)
from ..utils.math import _union_fieldnames
from ..utils.plotting import _figure_to_png
from .adaptive_stress_rendering import (
    _plot_feature_traces,
    _plot_posterior_timelines,
    _plot_prior_flip_examples,
)
from .adaptive_stress_types import AdaptiveStressCorpusArtifacts, AdaptiveStressCorpusResult
from .adaptive_stress_utils import (
    _infer_shared_scenario_name,
    _guided_action,
    _random_action,
    _reference_window_stats,
    _stress_targets,
    _to_shared_trajectory,
)
from .adaptive_stress_scoring import (
    _accumulator_trace,
    _high_entropy_score,
    _irregular_window_failure_score,
    _kalman_mismatch_score,
    _prior_flip_score,
    _raw_extrema_failure_score,
    _static_candidate_row,
    _transition_delay_candidates,
    _wrong_classification_score,
)
from .gym import CorpusGymAction, CorpusGymEnvironment, CorpusGymTarget

def analyze_adaptive_stress_corpus(
    *,
    seed: int = 7,
    random_candidates_per_mode: int = 8,
    guided_candidates_per_mode: int = 14,
) -> AdaptiveStressCorpusResult:
    rng = random.Random(seed)
    environment = CorpusGymEnvironment()
    sample_stats, duration_stats = _reference_window_stats()

    score_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    posterior_payloads: list[dict[str, object]] = []
    feature_payloads: list[dict[str, object]] = []
    prior_flip_payloads: list[dict[str, object]] = []

    failure_modes = {
        "wrong_classification": _wrong_classification_score,
        "high_entropy": _high_entropy_score,
        "prior_flip": _prior_flip_score,
        "raw_extrema_failure": _raw_extrema_failure_score,
        "irregular_window_failure": lambda shared: _irregular_window_failure_score(shared, sample_stats, duration_stats),
        "kalman_mismatch": _kalman_mismatch_score,
    }
    targets = _stress_targets()

    for target in targets:
        mode_name = target.target_id.removeprefix("stress_")
        evaluator = failure_modes[mode_name]
        mode_rows: list[dict[str, object]] = []
        for candidate_index in range(random_candidates_per_mode + guided_candidates_per_mode):
            target_for_candidate = target
            if target.class_pair is not None:
                target_for_candidate = replace(
                    target,
                    class_name=target.class_pair[candidate_index % len(target.class_pair)],
                )
            search_method = "random" if candidate_index < random_candidates_per_mode else "guided"
            action_seed = seed * 100_000 + len(score_rows) * 17 + candidate_index
            action = (
                _random_action(rng, target_for_candidate, seed=action_seed)
                if search_method == "random"
                else _guided_action(rng, mode_name, target_for_candidate, seed=action_seed)
            )
            environment.reset(target_for_candidate)
            episode = environment.simulate(action)
            if episode.reward.class_validity < 0.45:
                continue
            if episode.trajectory.true_class not in ("constant_velocity", "constant_acceleration"):
                continue
            shared = _to_shared_trajectory(episode.trajectory)
            stress_score, details, payload = evaluator(shared)
            if mode_name == "prior_flip":
                stress_score = max(stress_score, float(episode.reward.prior_sensitivity))
            row = _static_candidate_row(
                failure_mode=mode_name,
                search_method=search_method,
                target=target,
                episode=episode,
                score=stress_score,
                details=details,
            )
            mode_rows.append(row)
            score_rows.append(row)
            if mode_name in ("wrong_classification", "high_entropy", "transition_delay"):
                posterior_payloads.append(payload)
            if mode_name in ("raw_extrema_failure", "irregular_window_failure", "kalman_mismatch"):
                feature_payloads.append(payload)
            if mode_name == "prior_flip":
                prior_flip_payloads.append(payload)
        selected_rows.extend(sorted(mode_rows, key=lambda row: float(row["stress_score"]), reverse=True)[:2])

    transition_bundle = _transition_delay_candidates(
        seed=seed + 101,
        random_candidates=random_candidates_per_mode,
        guided_candidates=guided_candidates_per_mode,
    )
    transition_rows = transition_bundle.rows
    transition_posteriors = transition_bundle.posterior_payloads
    transition_features = transition_bundle.feature_payloads
    score_rows.extend(transition_rows)
    selected_rows.extend(sorted(transition_rows, key=lambda row: float(row["stress_score"]), reverse=True)[:2])
    posterior_payloads.extend(transition_posteriors[:2])

    selected_rows.sort(key=lambda row: (str(row["failure_mode"]), -float(row["stress_score"])))
    report_lines = [
        "# Adaptive Stress Corpus",
        "",
        "This artifact runs the first failure-targeted corpus search layer on top of CorpusGym and the existing classifier/filter diagnostics.",
        "",
        "## Summary",
        "",
    ]
    for failure_mode in (
        "wrong_classification",
        "high_entropy",
        "prior_flip",
        "raw_extrema_failure",
        "irregular_window_failure",
        "kalman_mismatch",
        "transition_delay",
    ):
        rows = [row for row in score_rows if row["failure_mode"] == failure_mode]
        random_rows = [row for row in rows if row["search_method"] == "random"]
        guided_rows = [row for row in rows if row["search_method"] == "guided"]
        mean_random = sum(float(row["stress_score"]) for row in random_rows) / max(len(random_rows), 1)
        best_guided = max((float(row["stress_score"]) for row in guided_rows), default=0.0)
        status = "resolved" if best_guided > mean_random else "not_yet_resolved"
        report_lines.append(f"- `{failure_mode}`: random mean `{mean_random:.3f}`, best guided `{best_guided:.3f}`, status `{status}`")
    report_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Wrong-classification, entropy, prior-flip, raw-extrema, irregular-window, and Kalman-mismatch cases are scored on generated CV/CA trajectories using existing repo comparators.",
            "- Transition-delay cases are searched over switching scenarios using the transition-matrix accumulator itself, because the current CorpusGym environment is still single-trajectory rather than switching-sequence native.",
            "- Guided search currently means failure-mode-shaped parameter sampling plus rejection and selection, not RL.",
            "- `high_entropy`, `prior_flip`, and `raw_extrema_failure` now use richer observables than the original shared-classifier hooks: local quadratic acceleration evidence for ambiguity and prior flips, plus explicit raw-versus-robust feature inflation for extrema stress.",
        ]
    )
    config = {
        "search_id": "adaptive_stress_corpus_v1",
        "seed": seed,
        "random_candidates_per_mode": random_candidates_per_mode,
        "guided_candidates_per_mode": guided_candidates_per_mode,
        "failure_modes": [
            "wrong_classification",
            "high_entropy",
            "prior_flip",
            "raw_extrema_failure",
            "irregular_window_failure",
            "kalman_mismatch",
            "transition_delay",
        ],
    }
    return AdaptiveStressCorpusResult(
        config=config,
        stress_case_rows=tuple(selected_rows),
        stress_score_rows=tuple(score_rows),
        report_markdown="\n".join(report_lines),
        posterior_trace_payloads=tuple(posterior_payloads),
        feature_trace_payloads=tuple(feature_payloads),
        prior_flip_payloads=tuple(prior_flip_payloads),
    )



def write_adaptive_stress_corpus_artifacts(
    output_dir: str | Path,
    *,
    result: AdaptiveStressCorpusResult | None = None,
) -> AdaptiveStressCorpusArtifacts:
    analysis = result or analyze_adaptive_stress_corpus()
    run_dir = Path(output_dir) / "adaptive_stress_corpus"
    plots_dir = run_dir / "plots"
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "stress_search_config.yaml"
    stress_cases_path = run_dir / "stress_cases.csv"
    stress_scores_path = run_dir / "stress_case_scores.csv"
    report_path = run_dir / "stress_case_report.md"
    posterior_timelines_path = plots_dir / "stress_case_posterior_timelines.png"
    feature_traces_path = plots_dir / "stress_case_feature_traces.png"
    prior_flip_examples_path = plots_dir / "prior_flip_examples.png"

    config_path.write_text(yaml.safe_dump(analysis.config, sort_keys=False), encoding="utf-8")
    if analysis.stress_case_rows:
        write_csv(stress_cases_path, list(analysis.stress_case_rows), _union_fieldnames(analysis.stress_case_rows))
    else:
        write_csv(stress_cases_path, [], ["failure_mode", "candidate_id"])
    if analysis.stress_score_rows:
        write_csv(stress_scores_path, list(analysis.stress_score_rows), _union_fieldnames(analysis.stress_score_rows))
    else:
        write_csv(stress_scores_path, [], ["failure_mode", "candidate_id"])
    report_path.write_text(analysis.report_markdown, encoding="utf-8")
    posterior_timelines_path.write_bytes(_figure_to_png(_plot_posterior_timelines(analysis)))
    feature_traces_path.write_bytes(_figure_to_png(_plot_feature_traces(analysis)))
    prior_flip_examples_path.write_bytes(_figure_to_png(_plot_prior_flip_examples(analysis)))

    return AdaptiveStressCorpusArtifacts(
        run_dir=run_dir,
        config_path=config_path,
        stress_cases_path=stress_cases_path,
        stress_scores_path=stress_scores_path,
        report_path=report_path,
        posterior_timelines_path=posterior_timelines_path,
        feature_traces_path=feature_traces_path,
        prior_flip_examples_path=prior_flip_examples_path,
    )
