from __future__ import annotations

from ..utils.plotting import plt
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


def render_static_decision_card_figure(result: StaticFeatureClassPriorAuditResult):
    rows = list(result.decision_card_rows)
    status_colors = {
        "pass": "#d8f3dc",
        "candidate": "#fff3bf",
        "warning": "#ffe8cc",
        "blocker": "#ffc9c9",
        "promote_to_corpus_explorer": "#d8f3dc",
        "revise_feature_set": "#ffe8cc",
        "revise_class_set": "#ffe8cc",
        "revise_prior": "#ffe8cc",
        "reject": "#ffc9c9",
    }
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    ax.axis("off")
    columns = ("lane", "score", "hardest_pair_or_feature", "status", "next_action")
    cell_text = [[_format_float(row.get(column, "")) for column in columns] for row in rows]
    table = ax.table(
        cellText=cell_text,
        colLabels=("lane", "score", "hardest pair / feature", "status", "next action"),
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.18, 0.10, 0.30, 0.18, 0.24],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.45)
    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_edgecolor("#ced4da")
        if row_index == 0:
            cell.set_facecolor("#212529")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif col_index == 3:
            status = str(rows[row_index - 1].get("status", ""))
            cell.set_facecolor(status_colors.get(status, "#f8f9fa"))
        else:
            cell.set_facecolor("#ffffff" if row_index % 2 else "#f8f9fa")
    ax.set_title(
        "Static Feature/Class/Prior Audit Card",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    return fig


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
