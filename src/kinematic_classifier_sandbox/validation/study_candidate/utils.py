from __future__ import annotations


def _pair_id_from_pair(pair: tuple[str, str] | list[str]) -> str:
    class_a, class_b = (str(name) for name in pair)
    return f"{class_a}_vs_{class_b}"


def _normalize_pair_id(value: str) -> str:
    return value.replace(" vs ", "_vs_")


def _feature_set_for_classifier(classifier_entry: dict[str, object]) -> str | None:
    if "feature_set_id" in classifier_entry:
        return str(classifier_entry["feature_set_id"])
    if "requires_feature_set" in classifier_entry:
        return str(classifier_entry["requires_feature_set"])
    return None


def _alias_matches(primary_separator: str, feature_names: set[str], groups: set[str], dependency_tags: set[str]) -> bool:
    separator = primary_separator.lower()
    direct_tokens = {separator, separator.replace(" ", "_")}
    if direct_tokens & feature_names:
        return True
    if direct_tokens & groups:
        return True
    if "linear_slope" in direct_tokens and {"slope", "velocity", "linear_fit"} & groups:
        return True
    if "model_residual" in direct_tokens and {"innovation", "state_residual", "log_likelihood"} & groups:
        return True
    if "velocity_residual" in direct_tokens and {"velocity", "linear_fit"} & dependency_tags:
        return True
    if "time_to_stop_proxy" in direct_tokens and "duration" in feature_names:
        return True
    if "innovation_sequence" in direct_tokens and {"innovation", "log_likelihood"} & groups:
        return True
    return False


def _feature_class_compatibility_score(
    *,
    primary_separators: list[str],
    feature_names: tuple[str, ...],
    groups: tuple[str, ...],
    dependency_tags: tuple[str, ...],
) -> float:
    feature_name_set = {name.lower() for name in feature_names}
    group_set = {name.lower() for name in groups}
    dependency_tag_set = {name.lower() for name in dependency_tags}
    matches = sum(
        1
        for separator in primary_separators
        if _alias_matches(separator, feature_name_set, group_set, dependency_tag_set)
    )
    if not primary_separators:
        return 0.5
    return matches / len(primary_separators)


def _history_risk(history_behavior: str) -> float:
    return {
        "instantaneous": 0.10,
        "windowed": 0.25,
        "model_residual": 0.20,
        "cumulative": 0.85,
        "cumulative_and_windowed": 0.75,
        "mixed": 0.65,
    }.get(history_behavior, 0.50)


def _classifier_assumption_fit(
    *,
    classifier_family: str,
    expected_difficulty: str,
    compatible: bool,
) -> float:
    if not compatible:
        return 0.0
    family_scores = {
        "pointwise": {
            "easy": 0.95,
            "duration_dependent": 0.45,
            "hard": 0.30,
            "short_horizon_boundary": 0.40,
        },
        "windowed": {
            "easy": 0.80,
            "duration_dependent": 0.75,
            "hard": 0.65,
            "short_horizon_boundary": 0.70,
        },
        "sequential_bayes": {
            "easy": 0.85,
            "duration_dependent": 0.90,
            "hard": 0.78,
            "short_horizon_boundary": 0.82,
        },
        "state_space": {
            "easy": 0.82,
            "duration_dependent": 0.95,
            "hard": 0.72,
            "short_horizon_boundary": 0.76,
        },
    }
    return family_scores.get(classifier_family, {}).get(expected_difficulty, 0.50)
