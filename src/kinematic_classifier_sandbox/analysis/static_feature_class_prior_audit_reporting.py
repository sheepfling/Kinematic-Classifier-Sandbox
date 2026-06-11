from __future__ import annotations

from .static_feature_class_prior_audit_contracts import StaticFeatureClassPriorAuditResult


def _format_float(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _markdown_table(rows: tuple[dict[str, object], ...], columns: tuple[str, ...]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_float(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def render_static_decision_card(result: StaticFeatureClassPriorAuditResult) -> str:
    return _markdown_table(
        result.decision_card_rows,
        ("lane", "score", "hardest_pair_or_feature", "status", "next_action"),
    )


def render_static_feature_class_prior_audit_report(
    result: StaticFeatureClassPriorAuditResult,
) -> str:
    decision = result.static_decision
    blockers = decision.get("blockers", ())
    warnings = decision.get("warnings", ())
    next_work = decision.get("next_work", ())
    lines = [
        "# Static Feature/Class/Prior Audit",
        "",
        f"- Study: `{result.study_name}`",
        f"- Classes: `{', '.join(result.class_names)}`",
        f"- Features: `{', '.join(result.feature_names)}`",
        f"- Decision: `{decision['status']}`",
        f"- Adequacy label: `{decision['adequacy_label']}`",
        "",
        "## Static Audit Decision Card",
        "",
        render_static_decision_card(result),
        "",
        "## Blockers",
        "",
    ]
    if blockers:
        lines.extend(f"- {blocker}" for blocker in blockers)
    else:
        lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- none")
    lines.extend(["", "## Next Work", ""])
    if next_work:
        lines.extend(f"- {item}" for item in next_work)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Hardest Class Pairs",
            "",
            _markdown_table(
                tuple(result.class_pair_rows[:6]),
                (
                    "class_a",
                    "class_b",
                    "pairwise_auc",
                    "mahalanobis_distance",
                    "overlap_coefficient",
                    "status",
                ),
            ),
            "",
            "## Feature Relevance",
            "",
            _markdown_table(
                tuple(result.feature_relevance_rows[:8]),
                (
                    "feature",
                    "mi_with_class",
                    "max_pairwise_auc",
                    "mean_effect_size",
                    "recommended_status",
                ),
            ),
            "",
            "## Prior Pathology",
            "",
            _markdown_table(
                tuple(result.prior_pathology_rows[:8]),
                (
                    "class_a",
                    "class_b",
                    "prior_odds_log",
                    "observed_log_lr_min",
                    "observed_log_lr_max",
                    "flip_possible",
                    "pathology_flag",
                ),
            ),
        ]
    )
    return "\n".join(lines) + "\n"
