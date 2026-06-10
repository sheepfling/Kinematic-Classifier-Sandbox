from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from ..analysis.common_dataset_comparison import (
    CommonComparisonResult,
    analyze_common_dataset_comparison,
)
from ..analysis.feature_analysis import load_feature_set_manifest
from ..scenarios import get_scenario_family as _registry_scenario_family
from ..scenarios import get_scenario_tier as _registry_scenario_tier
from .adapters import ExecutablePairSpec, ExecutableTrajectory
from .config import (
    CommonExperimentConfig,
    load_common_experiment_config,
    resolve_common_study_adapter,
)
from .contracts import CommonExperimentResult, CommonExperimentSummary
from .feature_set_studies import (
    build_feature_set_comparison_rows as _feature_set_comparison_rows,
)
from .feature_set_studies import (
    build_irregular_window_comparison_rows as _irregular_window_comparison_rows,
)
from .pair_evaluation import (
    evaluate_executable_pairs as _pair_eval_evaluate_executable_pairs,
)
from .pair_evaluation import (
    feature_set_scores_for_prefix as _pair_eval_feature_set_scores_for_prefix,
)
from .pair_evaluation import (
    feature_set_scores_for_window as _pair_eval_feature_set_scores_for_window,
)
from .pair_evaluation import (
    pair_priors as _pair_eval_pair_priors,
)
from .summary_rows import (
    class_pair_duration_rows as _class_pair_duration_rows,
)
from .summary_rows import (
    class_pair_scenario_rows as _class_pair_scenario_rows,
)
from .summary_rows import (
    covariate_rows as _covariate_rows,
)
from .summary_rows import (
    feature_excitation_rows as _feature_excitation_rows,
)
from .summary_rows import (
    identifiability_rows as _identifiability_rows,
)
from .summary_rows import (
    metrics_by_classifier as _metrics_by_classifier,
)
from .summary_rows import (
    metrics_by_sensor_regime as _metrics_by_sensor_regime,
)
from .summary_rows import (
    oracle_rows as _oracle_rows,
)


def _scenario_family(scenario_id: str) -> str:
    try:
        return _registry_scenario_family(scenario_id)
    except KeyError:
        return "other"


def _scenario_tier(scenario_id: str) -> str:
    try:
        return _registry_scenario_tier(scenario_id)
    except KeyError:
        return "other_v1"


def _analyze_common_trajectory_corpus(
    *,
    config: CommonExperimentConfig,
    comparison: CommonComparisonResult,
    pair_specs: tuple[ExecutablePairSpec, ...],
    trajectories: tuple[ExecutableTrajectory, ...],
    trajectories_per_case: int,
) -> CommonExperimentResult:
    feature_manifest = load_feature_set_manifest(config.feature_sets_path)
    (
        pair_prediction_rows,
        posterior_history_rows,
        likelihood_history_rows,
        feature_rows,
        metrics_by_class_pair_rows,
        prior_sensitivity_rows,
        metrics_by_classifier_and_feature_set_rows,
    ) = _pair_eval_evaluate_executable_pairs(
        config=config,
        pair_specs=pair_specs,
        trajectories=trajectories,
        scenario_family_fn=_scenario_family,
        scenario_tier_fn=_scenario_tier,
    )
    metrics_by_classifier_rows = _metrics_by_classifier(pair_prediction_rows)
    metrics_by_sensor_regime_rows = _metrics_by_sensor_regime(pair_prediction_rows)
    pair_prediction_row_dicts = tuple(asdict(row) for row in pair_prediction_rows)
    posterior_history_row_dicts = tuple(asdict(row) for row in posterior_history_rows)
    likelihood_history_row_dicts = tuple(asdict(row) for row in likelihood_history_rows)
    feature_row_dicts = tuple(row.as_flat_dict() for row in feature_rows)
    feature_set_comparison_rows = _feature_set_comparison_rows(
        config=config,
        pair_specs=pair_specs,
        trajectories=trajectories,
        pair_priors=_pair_eval_pair_priors,
        feature_set_scores_for_prefix=_pair_eval_feature_set_scores_for_prefix,
    )
    irregular_window_rows = _irregular_window_comparison_rows(
        config=config,
        pair_specs=pair_specs,
        trajectories=trajectories,
        pair_priors=_pair_eval_pair_priors,
        feature_set_scores_for_window=_pair_eval_feature_set_scores_for_window,
    )
    class_pair_duration_rows = _class_pair_duration_rows(posterior_history_row_dicts)
    class_pair_scenario_rows = _class_pair_scenario_rows(pair_prediction_row_dicts)
    covariate_rows = _covariate_rows(
        trajectories,
        scenario_tier_fn=_scenario_tier,
        scenario_family_fn=_scenario_family,
    )
    feature_excitation_rows = _feature_excitation_rows(feature_row_dicts)
    identifiability_rows = _identifiability_rows(feature_row_dicts, feature_manifest=feature_manifest)
    oracle_rows = _oracle_rows(feature_row_dicts, feature_manifest=feature_manifest)
    summary = CommonExperimentSummary(
        experiment_name=config.experiment_name,
        study_adapter_id=config.study_adapter_id,
        executable_class_pairs=tuple(spec.pair_id for spec in pair_specs),
        trajectories_per_case=trajectories_per_case,
        num_pair_trajectories=len(trajectories),
        num_pair_predictions=len(pair_prediction_rows),
    )
    return CommonExperimentResult(
        config=config,
        summary=summary,
        comparison=comparison,
        pair_prediction_rows=pair_prediction_row_dicts,
        posterior_history_rows=posterior_history_row_dicts,
        likelihood_history_rows=likelihood_history_row_dicts,
        feature_rows=feature_row_dicts,
        metrics_by_classifier_rows=metrics_by_classifier_rows,
        metrics_by_sensor_regime_rows=metrics_by_sensor_regime_rows,
        metrics_by_classifier_and_feature_set_rows=metrics_by_classifier_and_feature_set_rows,
        metrics_by_class_pair_rows=metrics_by_class_pair_rows,
        prior_sensitivity_rows=prior_sensitivity_rows,
        feature_set_comparison_rows=feature_set_comparison_rows,
        irregular_window_rows=irregular_window_rows,
        class_pair_duration_rows=class_pair_duration_rows,
        class_pair_scenario_rows=class_pair_scenario_rows,
        covariate_rows=covariate_rows,
        feature_excitation_rows=feature_excitation_rows,
        identifiability_rows=identifiability_rows,
        oracle_rows=oracle_rows,
    )


def analyze_common_experiment(
    *,
    config_path: str | Path | None = None,
    seed: int = 7,
    trajectories_per_case: int = 8,
) -> CommonExperimentResult:
    config = load_common_experiment_config(config_path)
    study_adapter = resolve_common_study_adapter(config)
    comparison = analyze_common_dataset_comparison(seed=seed, trajectories_per_case=trajectories_per_case)
    pair_specs = study_adapter.pair_spec_builder(config)
    trajectories = study_adapter.trajectory_generator(pair_specs, seed, trajectories_per_case)
    return _analyze_common_trajectory_corpus(
        config=config,
        comparison=comparison,
        pair_specs=pair_specs,
        trajectories=trajectories,
        trajectories_per_case=trajectories_per_case,
    )


def analyze_common_trajectory_corpus(
    *,
    pair_specs: tuple[ExecutablePairSpec, ...],
    trajectories: tuple[ExecutableTrajectory, ...],
    config_path: str | Path | None = None,
    seed: int = 7,
    trajectories_per_case: int | None = None,
    include_comparison: bool = True,
) -> CommonExperimentResult:
    config = load_common_experiment_config(config_path)
    comparison = (
        analyze_common_dataset_comparison(
            seed=seed,
            trajectories_per_case=trajectories_per_case or max(len(trajectories), 1),
        )
        if include_comparison
        else CommonComparisonResult(trajectories=(), runs=(), rows=(), method_specs=())
    )
    return _analyze_common_trajectory_corpus(
        config=config,
        comparison=comparison,
        pair_specs=pair_specs,
        trajectories=trajectories,
        trajectories_per_case=trajectories_per_case or max(len(trajectories), 1),
    )
