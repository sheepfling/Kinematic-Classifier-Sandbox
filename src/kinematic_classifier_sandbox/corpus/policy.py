from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import yaml

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CORPUS_POLICY_PATH = ROOT / "experiments" / "corpus_policies" / "default_corpus_policy_v1.yaml"


@dataclass(frozen=True, slots=True)
class CorpusPolicySpec:
    policy_id: str
    corpus_positive_weights: dict[str, float]
    corpus_penalty_weights: dict[str, float]
    generic_explorer_weights: dict[str, float]
    corpus_gym_weights: dict[str, float]
    archive_weights: dict[str, float]
    study_static_positive_weights: dict[str, float]
    study_static_penalty_weights: dict[str, float]
    study_mc_weights: dict[str, float]
    sampler_budgets: dict[str, int]
    gates: dict[str, float | int]
    normalization: dict[str, bool]


class PolicyArtifactPaths(NamedTuple):
    schema_path: Path
    default_path: Path


GENERIC_EXPLORER_TERMS = (
    "validity",
    "coverage_novelty",
    "boundary_score",
    "classifier_stress",
    "environment_score",
    "provenance_completeness",
)

CORPUS_POSITIVE_TERMS = (
    "balance",
    "boundary_coverage",
    "feature_excitation",
    "difficulty_diversity",
    "provenance_completeness",
)

CORPUS_PENALTY_TERMS = ("leakage", "triviality", "degeneracy")

CORPUS_GYM_TERMS = (
    "class_validity",
    "feature_excitation",
    "coverage_gain",
    "boundary_closeness",
    "classifier_stress",
    "prior_sensitivity",
    "leakage_penalty",
    "physical_invalidity_penalty",
)

ARCHIVE_TERMS = (
    "validity",
    "acceleration_range_pressure",
    "classifier_stress",
    "mean_margin_pressure",
)

STUDY_STATIC_POSITIVE_TERMS = (
    "feature_class_compatibility",
    "expected_separability",
    "classifier_assumption_fit",
    "corpus_coverage",
    "dimensional_transfer",
    "implementation_readiness",
    "feature_independence",
)

STUDY_STATIC_PENALTY_TERMS = (
    "cumulative_double_counting_risk",
    "prior_sensitivity_risk",
)

STUDY_MC_TERMS = (
    "accuracy",
    "prior_stability",
    "oracle_gap_closure",
)

SAMPLER_BUDGET_TERMS = (
    "random",
    "grid",
    "lhs",
    "boundary_mutation",
    "archive_mutation",
    "stress_mutation",
)

GATE_TERMS = (
    "min_class_validity",
    "ambiguity_margin",
    "max_leakage",
    "max_triviality",
    "min_feature_excitation",
    "max_prior_flip_fraction",
    "boundary_margin_min",
    "boundary_margin_max",
    "selected_corpus_size",
)


def load_corpus_policy_spec(path: str | Path = DEFAULT_CORPUS_POLICY_PATH) -> CorpusPolicySpec:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    policy = CorpusPolicySpec(
        policy_id=str(payload["policy_id"]),
        corpus_positive_weights=_float_mapping(payload["corpus_autodevelopment"]["positive_weights"], CORPUS_POSITIVE_TERMS),
        corpus_penalty_weights=_float_mapping(payload["corpus_autodevelopment"]["penalty_weights"], CORPUS_PENALTY_TERMS),
        generic_explorer_weights=_float_mapping(payload["generic_explorer"]["weights"], GENERIC_EXPLORER_TERMS),
        corpus_gym_weights=_float_mapping(payload["corpus_gym"]["weights"], CORPUS_GYM_TERMS),
        archive_weights=_float_mapping(payload["archive"]["weights"], ARCHIVE_TERMS),
        study_static_positive_weights=_float_mapping(payload["study_candidate"]["static_positive_weights"], STUDY_STATIC_POSITIVE_TERMS),
        study_static_penalty_weights=_float_mapping(payload["study_candidate"]["static_penalty_weights"], STUDY_STATIC_PENALTY_TERMS),
        study_mc_weights=_float_mapping(payload["study_candidate"]["monte_carlo_weights"], STUDY_MC_TERMS),
        sampler_budgets=_int_mapping(payload["sampler_budgets"], SAMPLER_BUDGET_TERMS),
        gates=_float_or_int_mapping(payload["gates"], GATE_TERMS),
        normalization={key: bool(value) for key, value in dict(payload.get("normalization", {})).items()},
    )
    return normalize_corpus_policy_spec(validate_corpus_policy_spec(policy))


def validate_corpus_policy_spec(policy: CorpusPolicySpec) -> CorpusPolicySpec:
    for group_name, values in (
        ("corpus_positive_weights", policy.corpus_positive_weights),
        ("corpus_penalty_weights", policy.corpus_penalty_weights),
        ("generic_explorer_weights", policy.generic_explorer_weights),
        ("corpus_gym_weights", policy.corpus_gym_weights),
        ("archive_weights", policy.archive_weights),
        ("study_static_positive_weights", policy.study_static_positive_weights),
        ("study_static_penalty_weights", policy.study_static_penalty_weights),
        ("study_mc_weights", policy.study_mc_weights),
    ):
        for key, value in values.items():
            if value < 0.0:
                raise ValueError(f"{group_name}.{key} must be non-negative")
        if sum(values.values()) <= 0.0:
            raise ValueError(f"{group_name} must sum to a positive value")
    for key, value in policy.sampler_budgets.items():
        if value < 0:
            raise ValueError(f"sampler_budgets.{key} must be non-negative")
    for key, value in policy.gates.items():
        if key == "selected_corpus_size" and int(value) <= 0:
            raise ValueError("gates.selected_corpus_size must be positive")
    return policy


def normalize_corpus_policy_spec(policy: CorpusPolicySpec) -> CorpusPolicySpec:
    return CorpusPolicySpec(
        policy_id=policy.policy_id,
        corpus_positive_weights=_normalize_mapping(policy.corpus_positive_weights)
        if policy.normalization.get("positive_weights_sum_to_one", True)
        else dict(policy.corpus_positive_weights),
        corpus_penalty_weights=_normalize_mapping(policy.corpus_penalty_weights)
        if policy.normalization.get("penalty_weights_sum_to_one", True)
        else dict(policy.corpus_penalty_weights),
        generic_explorer_weights=_normalize_mapping(policy.generic_explorer_weights),
        corpus_gym_weights=dict(policy.corpus_gym_weights),
        archive_weights=_normalize_mapping(policy.archive_weights),
        study_static_positive_weights=dict(policy.study_static_positive_weights),
        study_static_penalty_weights=dict(policy.study_static_penalty_weights),
        study_mc_weights=dict(policy.study_mc_weights),
        sampler_budgets=dict(policy.sampler_budgets),
        gates=dict(policy.gates),
        normalization=dict(policy.normalization),
    )


def corpus_policy_to_dict(policy: CorpusPolicySpec) -> dict[str, Any]:
    return {
        "policy_id": policy.policy_id,
        "corpus_autodevelopment": {
            "positive_weights": policy.corpus_positive_weights,
            "penalty_weights": policy.corpus_penalty_weights,
        },
        "generic_explorer": {"weights": policy.generic_explorer_weights},
        "corpus_gym": {"weights": policy.corpus_gym_weights},
        "archive": {"weights": policy.archive_weights},
        "study_candidate": {
            "static_positive_weights": policy.study_static_positive_weights,
            "static_penalty_weights": policy.study_static_penalty_weights,
            "monte_carlo_weights": policy.study_mc_weights,
        },
        "sampler_budgets": policy.sampler_budgets,
        "gates": policy.gates,
        "normalization": policy.normalization,
    }


def corpus_policy_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "CorpusPolicySpec",
        "type": "object",
        "required": [
            "policy_id",
            "corpus_autodevelopment",
            "generic_explorer",
            "corpus_gym",
            "archive",
            "study_candidate",
            "sampler_budgets",
            "gates",
        ],
        "properties": {
            "policy_id": {"type": "string"},
            "corpus_autodevelopment": _weight_group_schema(CORPUS_POSITIVE_TERMS, CORPUS_PENALTY_TERMS),
            "generic_explorer": _weights_schema(GENERIC_EXPLORER_TERMS),
            "corpus_gym": _weights_schema(CORPUS_GYM_TERMS),
            "archive": _weights_schema(ARCHIVE_TERMS),
            "study_candidate": {
                "type": "object",
                "required": ["static_positive_weights", "static_penalty_weights", "monte_carlo_weights"],
                "properties": {
                    "static_positive_weights": {
                        "type": "object",
                        "required": list(STUDY_STATIC_POSITIVE_TERMS),
                        "properties": {key: {"type": "number", "minimum": 0.0} for key in STUDY_STATIC_POSITIVE_TERMS},
                    },
                    "static_penalty_weights": {
                        "type": "object",
                        "required": list(STUDY_STATIC_PENALTY_TERMS),
                        "properties": {key: {"type": "number", "minimum": 0.0} for key in STUDY_STATIC_PENALTY_TERMS},
                    },
                    "monte_carlo_weights": {
                        "type": "object",
                        "required": list(STUDY_MC_TERMS),
                        "properties": {key: {"type": "number", "minimum": 0.0} for key in STUDY_MC_TERMS},
                    },
                },
            },
            "sampler_budgets": {
                "type": "object",
                "required": list(SAMPLER_BUDGET_TERMS),
                "properties": {key: {"type": "integer", "minimum": 0} for key in SAMPLER_BUDGET_TERMS},
            },
            "gates": {
                "type": "object",
                "required": list(GATE_TERMS),
                "properties": {key: {"type": "number"} for key in GATE_TERMS},
            },
        },
    }


def write_default_policy_artifacts(output_dir: str | Path, policy: CorpusPolicySpec | None = None) -> PolicyArtifactPaths:
    run_dir = Path(output_dir) / "corpus_hyperparameter_tuning_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    schema_path = run_dir / "weight_spec_schema.json"
    default_path = run_dir / "default_weight_spec.yaml"
    resolved = policy or load_corpus_policy_spec()
    schema_path.write_text(json.dumps(corpus_policy_schema(), indent=2), encoding="utf-8")
    default_path.write_text(yaml.safe_dump(corpus_policy_to_dict(resolved), sort_keys=False), encoding="utf-8")
    return PolicyArtifactPaths(schema_path=schema_path, default_path=default_path)


def _float_mapping(payload: dict[str, Any], required: tuple[str, ...]) -> dict[str, float]:
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"missing policy keys: {', '.join(missing)}")
    return {key: float(payload[key]) for key in required}


def _int_mapping(payload: dict[str, Any], required: tuple[str, ...]) -> dict[str, int]:
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"missing policy keys: {', '.join(missing)}")
    return {key: int(payload[key]) for key in required}


def _float_or_int_mapping(payload: dict[str, Any], required: tuple[str, ...]) -> dict[str, float | int]:
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"missing policy keys: {', '.join(missing)}")
    values: dict[str, float | int] = {}
    for key in required:
        values[key] = int(payload[key]) if key == "selected_corpus_size" else float(payload[key])
    return values


def _normalize_mapping(values: dict[str, float]) -> dict[str, float]:
    total = sum(float(value) for value in values.values())
    if total <= 0.0:
        raise ValueError("cannot normalize weights with non-positive total")
    return {key: float(value) / total for key, value in values.items()}


def score_corpus_autodevelopment_candidate(
    policy: CorpusPolicySpec,
    *,
    balance_score: float,
    boundary_coverage_score: float,
    feature_excitation_score: float,
    difficulty_diversity_score: float,
    provenance_completeness_score: float = 0.0,
    leakage_penalty: float,
    triviality_penalty: float,
    degeneracy_penalty: float,
) -> float:
    positives = {
        "balance": balance_score,
        "boundary_coverage": boundary_coverage_score,
        "feature_excitation": feature_excitation_score,
        "difficulty_diversity": difficulty_diversity_score,
        "provenance_completeness": provenance_completeness_score,
    }
    penalties = {
        "leakage": leakage_penalty,
        "triviality": triviality_penalty,
        "degeneracy": degeneracy_penalty,
    }
    active_positive_terms = sum(1 for value in policy.corpus_positive_weights.values() if value > 0.0)
    active_penalty_terms = sum(1 for value in policy.corpus_penalty_weights.values() if value > 0.0)
    positive_score = active_positive_terms * sum(policy.corpus_positive_weights[key] * positives[key] for key in CORPUS_POSITIVE_TERMS)
    penalty_score = active_penalty_terms * sum(policy.corpus_penalty_weights[key] * penalties[key] for key in CORPUS_PENALTY_TERMS)
    return positive_score - penalty_score


def score_corpus_gym_reward(
    policy: CorpusPolicySpec,
    *,
    class_validity: float,
    feature_excitation: float,
    coverage_gain: float,
    boundary_closeness: float,
    classifier_stress: float,
    prior_sensitivity: float,
    leakage_penalty: float,
    physical_invalidity_penalty: float,
) -> float:
    weights = policy.corpus_gym_weights
    return _clamp_policy_score(
        weights["class_validity"] * class_validity
        + weights["feature_excitation"] * feature_excitation
        + weights["coverage_gain"] * coverage_gain
        + weights["boundary_closeness"] * boundary_closeness
        + weights["classifier_stress"] * classifier_stress
        + weights["prior_sensitivity"] * prior_sensitivity
        - weights["leakage_penalty"] * leakage_penalty
        - weights["physical_invalidity_penalty"] * physical_invalidity_penalty
    )


def score_qd_archive_elite(
    policy: CorpusPolicySpec,
    *,
    validity_score: float,
    acceleration_range_pressure: float,
    classifier_stress: float,
    mean_margin_pressure: float,
) -> float:
    weights = policy.archive_weights
    return (
        weights["validity"] * validity_score
        + weights["acceleration_range_pressure"] * acceleration_range_pressure
        + weights["classifier_stress"] * classifier_stress
        + weights["mean_margin_pressure"] * mean_margin_pressure
    )


def score_study_candidate_static(
    policy: CorpusPolicySpec,
    *,
    feature_class_compatibility: float,
    expected_separability: float,
    classifier_assumption_fit: float,
    corpus_coverage: float,
    dimensional_transfer: float,
    implementation_readiness: float,
    feature_dependency_risk: float,
    cumulative_double_counting_risk: float,
    prior_sensitivity_risk: float,
) -> float:
    positive = policy.study_static_positive_weights
    penalty = policy.study_static_penalty_weights
    return _clamp_policy_score(
        positive["feature_class_compatibility"] * feature_class_compatibility
        + positive["expected_separability"] * expected_separability
        + positive["classifier_assumption_fit"] * classifier_assumption_fit
        + positive["corpus_coverage"] * corpus_coverage
        + positive["dimensional_transfer"] * dimensional_transfer
        + positive["implementation_readiness"] * implementation_readiness
        + positive["feature_independence"] * (1.0 - feature_dependency_risk)
        - penalty["cumulative_double_counting_risk"] * cumulative_double_counting_risk
        - penalty["prior_sensitivity_risk"] * prior_sensitivity_risk
    )


def score_study_candidate_monte_carlo(
    policy: CorpusPolicySpec,
    *,
    accuracy: float,
    prior_flip_fraction: float,
    oracle_gap: float,
) -> float:
    weights = policy.study_mc_weights
    return _clamp_policy_score(
        weights["accuracy"] * accuracy
        + weights["prior_stability"] * (1.0 - prior_flip_fraction)
        + weights["oracle_gap_closure"] * (1.0 - max(0.0, oracle_gap))
    )


def _clamp_policy_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _weights_schema(required: tuple[str, ...]) -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["weights"],
        "properties": {
            "weights": {
                "type": "object",
                "required": list(required),
                "properties": {key: {"type": "number", "minimum": 0.0} for key in required},
            }
        },
    }


def _weight_group_schema(positive: tuple[str, ...], penalty: tuple[str, ...]) -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["positive_weights", "penalty_weights"],
        "properties": {
            "positive_weights": {
                "type": "object",
                "required": list(positive),
                "properties": {key: {"type": "number", "minimum": 0.0} for key in positive},
            },
            "penalty_weights": {
                "type": "object",
                "required": list(penalty),
                "properties": {key: {"type": "number", "minimum": 0.0} for key in penalty},
            },
        },
    }
