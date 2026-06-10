from __future__ import annotations

from pathlib import Path
from typing import Any

from ..policy import load_corpus_policy_spec
from .backend_adapter_proof_types import BackendCandidateSpec
from .generic_corpus_exploration_types import (
    GenericCorpusExplorationResult,
    GenericCorpusExplorationSweepConfig,
    GenericCorpusExplorationSweepVariant,
    GenericCorpusExplorationWeights,
)


def _set_jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / max(len(left | right), 1)


def _default_generic_corpus_exploration_weights() -> GenericCorpusExplorationWeights:
    weights = load_corpus_policy_spec().generic_explorer_weights
    return GenericCorpusExplorationWeights(
        validity=weights["validity"],
        coverage_novelty=weights["coverage_novelty"],
        boundary=weights["boundary_score"],
        stress=weights["classifier_stress"],
        environment=weights["environment_score"],
        provenance=weights["provenance_completeness"],
    )


DEFAULT_GENERIC_CORPUS_EXPLORATION_WEIGHTS = _default_generic_corpus_exploration_weights()


def _normalize_weights(weights: GenericCorpusExplorationWeights) -> GenericCorpusExplorationWeights:
    values = {
        "validity": float(weights.validity),
        "coverage_novelty": float(weights.coverage_novelty),
        "boundary": float(weights.boundary),
        "stress": float(weights.stress),
        "environment": float(weights.environment),
        "provenance": float(weights.provenance),
    }
    total = sum(values.values())
    if total <= 0.0:
        raise ValueError("generic corpus exploration weights must sum to a positive value")
    return GenericCorpusExplorationWeights(**{key: value / total for key, value in values.items()})


def _perturb_weights(
    weights: GenericCorpusExplorationWeights,
    *,
    focus: str,
    delta: float,
) -> GenericCorpusExplorationWeights:
    if focus not in {"validity", "coverage_novelty", "boundary", "stress", "environment", "provenance"}:
        raise KeyError(f"unknown weight component: {focus}")
    values = {
        "validity": float(weights.validity),
        "coverage_novelty": float(weights.coverage_novelty),
        "boundary": float(weights.boundary),
        "stress": float(weights.stress),
        "environment": float(weights.environment),
        "provenance": float(weights.provenance),
    }
    values[focus] = max(values[focus] + delta, 1e-6)
    total = sum(values.values())
    return GenericCorpusExplorationWeights(**{key: value / total for key, value in values.items()})


def _candidate_pool() -> tuple[BackendCandidateSpec, ...]:
    candidates: list[BackendCandidateSpec] = []
    for replicate in range(3):
        candidates.append(
            BackendCandidateSpec(
                candidate_id=f"boundary_param_{replicate}",
                scenario_id="shared_boundary_cv_ca",
                scenario_family="shared_boundary_case",
                target_class="constant_velocity",
                difficulty_tier="boundary_v1",
                seed=700 + replicate,
                duration=2.0 + 0.15 * replicate,
                sample_period=0.5,
                initial_position=0.0,
                initial_velocity=1.00 + 0.03 * replicate,
                acceleration=0.08 + 0.02 * replicate,
                measurement_std=0.03,
                provenance={"search_method": "dashboard_seeded", "search_iteration": len(candidates)},
            )
        )
        candidates.append(
            BackendCandidateSpec(
                candidate_id=f"switching_ctrl_{replicate}",
                scenario_id="switching_velocity_to_braking",
                scenario_family="switching_case",
                target_class="braking",
                difficulty_tier="stress_v1",
                seed=720 + replicate,
                duration=2.0,
                sample_period=0.5,
                initial_position=0.0,
                initial_velocity=1.35 + 0.04 * replicate,
                acceleration=0.0,
                measurement_std=0.04,
                switch_time=0.9 + 0.1 * replicate,
                acceleration_after_switch=-0.65 - 0.05 * replicate,
                provenance={"search_method": "dashboard_seeded", "search_iteration": len(candidates)},
            )
        )
        candidates.append(
            BackendCandidateSpec(
                candidate_id=f"env_accel_{replicate}",
                scenario_id="environment_density_gradient",
                scenario_family="environment_regime_case",
                target_class="constant_acceleration",
                difficulty_tier="realistic_v1",
                seed=740 + replicate,
                duration=2.0,
                sample_period=0.5,
                initial_position=0.0,
                initial_velocity=0.82 + 0.03 * replicate,
                acceleration=0.40 + 0.04 * replicate,
                measurement_std=0.03,
                drag_coefficient=0.18 + 0.04 * replicate,
                density_scale=1.02 - 0.08 * replicate,
                wind_bias=0.03 + 0.04 * replicate,
                provenance={"search_method": "dashboard_seeded", "search_iteration": len(candidates), "environment_id": f"env_regime_{replicate}"},
            )
        )
        candidates.append(
            BackendCandidateSpec(
                candidate_id=f"file_maneuver_{replicate}",
                scenario_id="file_backend_case",
                scenario_family="file_backend_case",
                target_class="maneuver",
                difficulty_tier="adversarial_v1",
                seed=760 + replicate,
                duration=2.0,
                sample_period=0.5,
                initial_position=0.0,
                initial_velocity=0.85 + 0.05 * replicate,
                acceleration=0.24 + 0.04 * replicate,
                measurement_std=0.03,
                input_deck_hash=f"file_case_hash_{replicate}",
                longitudinal_command=(0.4, 0.5, 0.1, -0.3, -0.4),
                provenance={"search_method": "dashboard_seeded", "search_iteration": len(candidates)},
            )
        )
    return tuple(candidates)


def _backends_for_candidate(candidate: BackendCandidateSpec) -> tuple[str, ...]:
    if candidate.scenario_family == "shared_boundary_case":
        return ("parameter_only_1d", "environment_aware_1d", "mock_file_backend_1d")
    if candidate.scenario_family == "switching_case":
        return ("controlled_1d", "mock_file_backend_1d")
    if candidate.scenario_family == "environment_regime_case":
        return ("environment_aware_1d",)
    if candidate.scenario_family == "file_backend_case":
        return ("mock_file_backend_1d",)
    return ()


def _provenance_completeness(metadata: dict[str, Any]) -> float:
    required = ("adapter_family", "candidate_id", "search_provenance")
    present = sum(1 for key in required if key in metadata and metadata[key] not in ("", None, {}))
    return present / len(required)


def _score_run(
    candidate: BackendCandidateSpec,
    backend_id: str,
    run: Any,
    weights: GenericCorpusExplorationWeights,
) -> dict[str, Any]:
    success = bool(run.success)
    provenance_score = _provenance_completeness(run.metadata)
    truth_state = run.truth_state
    positions = truth_state.get("position", ())
    velocities = truth_state.get("velocity", ())
    accelerations = truth_state.get("acceleration", ())
    speed_range = max(velocities) - min(velocities) if velocities else 0.0
    position_range = max(positions) - min(positions) if positions else 0.0
    acceleration_range = max(accelerations) - min(accelerations) if accelerations else 0.0
    validity_score = 1.0 if success else 0.0
    boundary_score = 0.85 if candidate.scenario_family == "shared_boundary_case" else 0.35
    stress_score = 0.90 if candidate.scenario_family == "switching_case" else 0.70 if candidate.scenario_family == "file_backend_case" else 0.45
    environment_score = 0.85 if candidate.scenario_family == "environment_regime_case" else 0.30
    coverage_novelty_score = {
        "shared_boundary_case": 0.70,
        "switching_case": 0.92,
        "environment_regime_case": 0.88,
        "file_backend_case": 0.78,
    }.get(candidate.scenario_family, 0.50)
    utility = (
        weights.validity * validity_score
        + weights.coverage_novelty * coverage_novelty_score
        + weights.boundary * boundary_score
        + weights.stress * stress_score
        + weights.environment * environment_score
        + weights.provenance * provenance_score
    )
    cell_id = f"{backend_id}|{candidate.scenario_family}|{candidate.target_class}|{candidate.difficulty_tier}"
    return {
        "candidate_id": candidate.candidate_id,
        "backend_id": backend_id,
        "trajectory_id": run.run_id,
        "scenario_family": candidate.scenario_family,
        "target_class": candidate.target_class,
        "difficulty_tier": candidate.difficulty_tier,
        "environment_id": candidate.provenance.get("environment_id", ""),
        "success": success,
        "validity_score": validity_score,
        "coverage_novelty_score": coverage_novelty_score,
        "boundary_score": boundary_score,
        "classifier_stress_score": stress_score,
        "environment_score": environment_score,
        "provenance_completeness": provenance_score,
        "total_utility": utility,
        "utility_weights": _weights_to_dict(weights),
        "position_range": position_range,
        "speed_range": speed_range,
        "acceleration_range": acceleration_range,
        "num_samples": len(run.times),
        "cell_id": cell_id,
    }


def _archive_rows(candidate_rows: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    best_by_cell: dict[str, dict[str, Any]] = {}
    for row in candidate_rows:
        cell_id = str(row["cell_id"])
        if cell_id not in best_by_cell or float(row["total_utility"]) > float(best_by_cell[cell_id]["total_utility"]):
            best_by_cell[cell_id] = dict(row)
    rows = []
    for cell_id, row in sorted(best_by_cell.items()):
        rows.append(
            {
                "cell_id": cell_id,
                "backend_id": row["backend_id"],
                "scenario_family": row["scenario_family"],
                "target_class": row["target_class"],
                "difficulty_tier": row["difficulty_tier"],
                "elite_candidate_id": row["candidate_id"],
                "elite_total_utility": row["total_utility"],
            }
        )
    return tuple(rows)


def _selected_manifest_rows(candidate_rows: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    best_by_cell: dict[str, dict[str, Any]] = {}
    for row in candidate_rows:
        if not bool(row["success"]):
            continue
        cell_id = str(row["cell_id"])
        if cell_id not in best_by_cell or float(row["total_utility"]) > float(best_by_cell[cell_id]["total_utility"]):
            best_by_cell[cell_id] = dict(row)
    rows = sorted(best_by_cell.values(), key=lambda row: (-float(row["total_utility"]), str(row["backend_id"])))
    return tuple(rows[:6])


def _backend_comparison_rows(candidate_rows: tuple[dict[str, Any], ...], selected_rows: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    backend_ids = sorted({str(row["backend_id"]) for row in candidate_rows})
    for backend_id in backend_ids:
        backend_rows = [row for row in candidate_rows if row["backend_id"] == backend_id]
        selected_backend_rows = [row for row in selected_rows if row["backend_id"] == backend_id]
        rows.append(
            {
                "backend_id": backend_id,
                "candidate_count": len(backend_rows),
                "selected_count": len(selected_backend_rows),
                "success_rate": sum(1 for row in backend_rows if bool(row["success"])) / max(len(backend_rows), 1),
                "mean_total_utility": sum(float(row["total_utility"]) for row in backend_rows) / max(len(backend_rows), 1),
                "mean_provenance_completeness": sum(float(row["provenance_completeness"]) for row in backend_rows) / max(len(backend_rows), 1),
            }
        )
    return tuple(rows)


def _generic_corpus_exploration_weight_variants() -> tuple[GenericCorpusExplorationSweepVariant, ...]:
    baseline = _normalize_weights(DEFAULT_GENERIC_CORPUS_EXPLORATION_WEIGHTS)
    return (
        GenericCorpusExplorationSweepVariant(
            variant_id="baseline",
            description="Current production weights",
            weights=baseline,
        ),
        GenericCorpusExplorationSweepVariant(
            variant_id="validity_plus_5",
            description="Increase validity emphasis by 5 percentage points",
            weights=_perturb_weights(baseline, focus="validity", delta=0.05),
        ),
        GenericCorpusExplorationSweepVariant(
            variant_id="coverage_plus_5",
            description="Increase coverage-novelty emphasis by 5 percentage points",
            weights=_perturb_weights(baseline, focus="coverage_novelty", delta=0.05),
        ),
        GenericCorpusExplorationSweepVariant(
            variant_id="boundary_plus_5",
            description="Increase boundary emphasis by 5 percentage points",
            weights=_perturb_weights(baseline, focus="boundary", delta=0.05),
        ),
        GenericCorpusExplorationSweepVariant(
            variant_id="stress_plus_5",
            description="Increase stress emphasis by 5 percentage points",
            weights=_perturb_weights(baseline, focus="stress", delta=0.05),
        ),
        GenericCorpusExplorationSweepVariant(
            variant_id="environment_plus_5",
            description="Increase environment emphasis by 5 percentage points",
            weights=_perturb_weights(baseline, focus="environment", delta=0.05),
        ),
        GenericCorpusExplorationSweepVariant(
            variant_id="provenance_plus_5",
            description="Increase provenance emphasis by 5 percentage points",
            weights=_perturb_weights(baseline, focus="provenance", delta=0.05),
        ),
    )


def _resolve_generic_corpus_exploration_weight_sweep_config(
    *,
    config: GenericCorpusExplorationSweepConfig | None = None,
    config_path: str | Path | None = None,
    variants: tuple[GenericCorpusExplorationSweepVariant, ...] | None = None,
) -> GenericCorpusExplorationSweepConfig:
    if config is not None:
        return config
    if config_path is not None:
        from .generic_corpus_exploration import load_generic_corpus_exploration_weight_sweep_config

        return load_generic_corpus_exploration_weight_sweep_config(config_path)
    if variants is not None:
        if not variants:
            raise ValueError("at least one weight variant is required")
        return GenericCorpusExplorationSweepConfig(
            baseline_variant_id=variants[0].variant_id,
            variants=variants,
            config_path=None,
        )
    from .generic_corpus_exploration import (
        DEFAULT_GENERIC_CORPUS_EXPLORATION_WEIGHT_SWEEP_CONFIG_PATH,
        load_generic_corpus_exploration_weight_sweep_config,
    )

    if DEFAULT_GENERIC_CORPUS_EXPLORATION_WEIGHT_SWEEP_CONFIG_PATH.exists():
        return load_generic_corpus_exploration_weight_sweep_config(
            DEFAULT_GENERIC_CORPUS_EXPLORATION_WEIGHT_SWEEP_CONFIG_PATH
        )
    return GenericCorpusExplorationSweepConfig(
        baseline_variant_id="baseline",
        variants=_generic_corpus_exploration_weight_variants(),
        config_path=None,
    )


def _exploration_result_set(result: GenericCorpusExplorationResult) -> tuple[set[str], set[str]]:
    selected_rows = tuple(result.selected_corpus_manifest["selected_rows"])
    candidate_ids = {str(row["candidate_id"]) for row in selected_rows}
    cell_ids = {str(row["cell_id"]) for row in selected_rows}
    return candidate_ids, cell_ids


def _weights_to_dict(weights: GenericCorpusExplorationWeights) -> dict[str, float]:
    return {
        "validity": float(weights.validity),
        "coverage_novelty": float(weights.coverage_novelty),
        "boundary": float(weights.boundary),
        "stress": float(weights.stress),
        "environment": float(weights.environment),
        "provenance": float(weights.provenance),
    }
