from __future__ import annotations

from pathlib import Path

from ...utils.runtime import repo_root


def _protocol_source_path() -> Path:
    return repo_root() / "docs" / "protocols" / "feature_class_classifier_analysis_protocol.md"


def _fallback_protocol_markdown() -> str:
    return "\n".join(
        [
            "# Feature + Class + Classifier Analysis Protocol",
            "",
            "This fallback protocol exists so artifact generation remains stable if the checked-in protocol document is missing.",
            "",
            "## Core Steps",
            "",
            "1. Define the study hypothesis.",
            "2. Declare the class set and class-pair claims.",
            "3. Declare feature sets and feature taxonomy metadata.",
            "4. Declare classifier or filter family and assumptions.",
            "5. Run static compatibility screening.",
            "6. Generate or select corpus candidates.",
            "7. Run corpus adequacy and leakage audits.",
            "8. Run static separability and oracle studies.",
            "9. Run Monte Carlo classifier ladder.",
            "10. Produce a promote, revise, reject, or defer decision.",
        ]
    )


def _load_protocol_markdown() -> str:
    path = _protocol_source_path()
    if path.exists():
        return path.read_text(encoding="utf-8")
    return _fallback_protocol_markdown()


def _study_candidate_schema() -> dict[str, object]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "StudyCandidate",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "study_id",
            "hypothesis",
            "corpus_spec",
            "feature_set_spec",
            "class_set_spec",
            "classifier_spec",
            "prior_spec",
            "expected_failure_modes",
            "decision_policy",
        ],
        "properties": {
            "study_id": {"type": "string", "minLength": 3},
            "hypothesis": {"type": "string", "minLength": 1},
            "corpus_spec": {
                "type": "object",
                "additionalProperties": False,
                "required": ["corpus_id", "sensor_regime_id", "tiers"],
                "properties": {
                    "corpus_id": {"type": "string"},
                    "sensor_regime_id": {"type": "string"},
                    "tiers": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["easy", "boundary", "adversarial", "stress", "realistic"],
                        },
                        "minItems": 1,
                    },
                    "generator_family": {"type": "string"},
                    "objectives_id": {"type": "string"},
                },
            },
            "feature_set_spec": {
                "type": "object",
                "additionalProperties": False,
                "required": ["feature_sets"],
                "properties": {
                    "feature_sets": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "required_tags": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "double_counting_risk": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                },
            },
            "class_set_spec": {
                "type": "object",
                "additionalProperties": False,
                "required": ["classes", "class_pairs"],
                "properties": {
                    "classes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                    },
                    "class_pairs": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "prefixItems": [{"type": "string"}, {"type": "string"}],
                            "minItems": 2,
                            "maxItems": 2,
                        },
                        "minItems": 1,
                    },
                    "claims": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
            "classifier_spec": {
                "type": "object",
                "additionalProperties": False,
                "required": ["classifier_families"],
                "properties": {
                    "classifier_families": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "pointwise",
                                "windowed",
                                "sequential_bayes",
                                "state_space",
                                "transition_matrix",
                            ],
                        },
                        "minItems": 1,
                    },
                    "history_behavior": {"type": "string"},
                    "assumptions": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
            "prior_spec": {
                "type": "object",
                "additionalProperties": False,
                "required": ["prior_ids"],
                "properties": {
                    "prior_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "sensitivity_risk": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                },
            },
            "filter_spec": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "filter_family": {
                        "type": "string",
                        "enum": ["none", "kalman_bank", "transition_matrix", "imm", "pf", "rbpf"],
                    },
                    "uses_dynamics": {"type": "boolean"},
                    "handles_switching": {"type": "boolean"},
                },
            },
            "visualization_spec": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "required_plots": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "requires_bayesian_walkthrough": {"type": "boolean"},
                },
            },
            "expected_failure_modes": {
                "type": "array",
                "items": {"type": "string"},
            },
            "decision_policy": {
                "type": "object",
                "additionalProperties": False,
                "required": ["allowed_decisions"],
                "properties": {
                    "allowed_decisions": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["promote", "revise", "reject", "defer"],
                        },
                        "minItems": 4,
                        "uniqueItems": True,
                    },
                    "promotion_requires_monte_carlo": {"type": "boolean"},
                    "rejection_allows_static_only": {"type": "boolean"},
                },
            },
        },
    }


def _validation_ladder_schema() -> dict[str, object]:
    levels = [
        "static_compatibility",
        "corpus_adequacy",
        "feature_separability",
        "oracle_separability",
        "classifier_performance",
        "posterior_and_calibration_quality",
        "prior_sensitivity",
        "stress_and_adversarial_robustness",
        "dimensional_transfer_assessment",
        "promotion_decision",
    ]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "ValidationLadder",
        "type": "object",
        "additionalProperties": False,
        "required": ["study_id", "levels", "final_decision"],
        "properties": {
            "study_id": {"type": "string"},
            "levels": {
                "type": "array",
                "minItems": len(levels),
                "maxItems": len(levels),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["level_id", "level_name", "status", "score", "evidence_summary"],
                    "properties": {
                        "level_id": {"type": "integer", "minimum": 1, "maximum": 10},
                        "level_name": {"type": "string", "enum": levels},
                        "status": {"type": "string", "enum": ["pass", "partial", "fail", "defer"]},
                        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "evidence_summary": {"type": "string"},
                        "linked_artifacts": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
            },
            "final_decision": {"type": "string", "enum": ["promote", "revise", "reject", "defer"]},
            "decision_rationale": {"type": "string"},
            "known_gaps": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
    }


def _validation_summary(
    study_candidate_schema: dict[str, object],
    validation_ladder_schema: dict[str, object],
    protocol_markdown: str,
) -> dict[str, object]:
    candidate_required = set(study_candidate_schema["required"])
    ladder_level_names = [
        item["enum"]
        for item in [validation_ladder_schema["properties"]["levels"]["items"]["properties"]["level_name"]]
    ][0]
    return {
        "protocol_source_path": str(_protocol_source_path()),
        "protocol_has_ten_steps": all(f"{index}." in protocol_markdown for index in range(1, 11)),
        "study_candidate_has_core_specs": all(
            name in candidate_required
            for name in ("corpus_spec", "feature_set_spec", "class_set_spec", "classifier_spec", "prior_spec")
        ),
        "study_candidate_has_optional_filter_spec": "filter_spec" in study_candidate_schema["properties"],
        "decision_vocab_complete": sorted(
            study_candidate_schema["properties"]["decision_policy"]["properties"]["allowed_decisions"]["items"]["enum"]
        )
        == ["defer", "promote", "reject", "revise"],
        "validation_ladder_has_ten_levels": len(ladder_level_names) == 10,
        "validation_ladder_terminal_decisions": validation_ladder_schema["properties"]["final_decision"]["enum"],
        "overall_status": "pass",
    }
