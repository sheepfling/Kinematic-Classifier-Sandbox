from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_POLICY_PATH = ROOT / "experiments" / "corpus_policies" / "default_corpus_policy_v1.yaml"


@dataclass(frozen=True, slots=True)
class CorpusPolicySpec:
    policy_id: str
    corpus_positive_weights: dict[str, float]
    corpus_penalty_weights: dict[str, float]
    generic_explorer_weights: dict[str, float]
    corpus_gym_weights: dict[str, float]
    archive_weights: dict[str, float]
    sampler_budgets: dict[str, int]
    gates: dict[str, float | int]
    normalization: dict[str, bool]


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
            "sampler_budgets",
            "gates",
        ],
        "properties": {
            "policy_id": {"type": "string"},
            "corpus_autodevelopment": _weight_group_schema(CORPUS_POSITIVE_TERMS, CORPUS_PENALTY_TERMS),
            "generic_explorer": _weights_schema(GENERIC_EXPLORER_TERMS),
            "corpus_gym": _weights_schema(CORPUS_GYM_TERMS),
            "archive": _weights_schema(ARCHIVE_TERMS),
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


def write_default_policy_artifacts(output_dir: str | Path, policy: CorpusPolicySpec | None = None) -> tuple[Path, Path]:
    run_dir = Path(output_dir) / "corpus_hyperparameter_tuning_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    schema_path = run_dir / "weight_spec_schema.json"
    default_path = run_dir / "default_weight_spec.yaml"
    resolved = policy or load_corpus_policy_spec()
    schema_path.write_text(json.dumps(corpus_policy_schema(), indent=2), encoding="utf-8")
    default_path.write_text(yaml.safe_dump(corpus_policy_to_dict(resolved), sort_keys=False), encoding="utf-8")
    return schema_path, default_path


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

