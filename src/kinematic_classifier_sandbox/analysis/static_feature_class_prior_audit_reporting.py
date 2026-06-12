from __future__ import annotations

import math

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


def _status_color(status: str) -> str:
    return {
        "easy": "#d8f3dc",
        "medium": "#fff3bf",
        "hard": "#ffc9c9",
        "pass": "#d8f3dc",
        "ok": "#d8f3dc",
        "keep": "#d8f3dc",
        "weak": "#ffe8cc",
        "pair_limited": "#fff3bf",
        "high_redundancy": "#ffe8cc",
        "synergy_candidate": "#fff3bf",
        "ordinary": "#f8f9fa",
        "prior_domination": "#ffc9c9",
        "posterior_collapse_risk": "#ffe8cc",
        "warning": "#ffe8cc",
        "blocker": "#ffc9c9",
    }.get(status, "#f8f9fa")


def _class_pair_value(result: StaticFeatureClassPriorAuditResult, key: str) -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], float] = {}
    for row in result.class_pair_rows:
        class_a = str(row["class_a"])
        class_b = str(row["class_b"])
        value = float(row.get(key, 0.0))
        values[(class_a, class_b)] = value
        values[(class_b, class_a)] = value
    return values


def render_class_confusability_figure(result: StaticFeatureClassPriorAuditResult):
    labels = list(result.class_names)
    values = _class_pair_value(result, "overlap_coefficient")
    matrix = [[0.0 if left == right else values.get((left, right), 0.0) for right in labels] for left in labels]
    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    image = ax.imshow(matrix, cmap="YlOrRd", vmin=0.0, vmax=max(0.7, max(max(row) for row in matrix)))
    ax.set_xticks(range(len(labels)), [label.replace("_", "\n") for label in labels], fontsize=8)
    ax.set_yticks(range(len(labels)), [label.replace("_", "\n") for label in labels], fontsize=8)
    status_by_pair = {(str(row["class_a"]), str(row["class_b"])): str(row["status"]) for row in result.class_pair_rows}
    for i, left in enumerate(labels):
        for j, right in enumerate(labels):
            if i == j:
                text = "-"
            else:
                text = status_by_pair.get((left, right), status_by_pair.get((right, left), f"{matrix[i][j]:.2f}"))
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color="#17202A")
    ax.set_title("Hard class pairs are identified before blaming algorithms.", loc="left", fontsize=15, fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03, label="overlap / confusability")
    return fig


def render_feature_relevance_figure(result: StaticFeatureClassPriorAuditResult):
    rows = sorted(result.feature_relevance_rows, key=lambda row: float(row["mi_with_class"]), reverse=True)
    labels = [str(row["feature"]) for row in rows]
    mi_values = [float(row["mi_with_class"]) for row in rows]
    auc_values = [max(float(row["max_pairwise_auc"]) - 0.5, 0.0) for row in rows]
    y = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(10.8, max(4.8, 0.48 * len(rows) + 1.2)))
    ax.barh([value + 0.18 for value in y], mi_values, height=0.32, color="#2E86AB", label="mutual information")
    ax.barh([value - 0.18 for value in y], auc_values, height=0.32, color="#1B998B", label="AUC lift over 0.5")
    for index, row in enumerate(rows):
        ax.text(max(mi_values[index], auc_values[index]) + 0.01, index, str(row["recommended_status"]), va="center", fontsize=8)
    ax.set_yticks(y, [label.replace("_", " ") for label in labels], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("score")
    ax.set_title("Feature relevance is measured before classifier work begins.", loc="left", fontsize=15, fontweight="bold")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="x", alpha=0.2)
    return fig


def render_feature_redundancy_graph_figure(result: StaticFeatureClassPriorAuditResult):
    features = list(result.feature_names)
    count = len(features)
    radius = 0.38
    center = (0.5, 0.52)
    positions = {
        feature: (
            center[0] + radius * math.cos(2.0 * math.pi * index / max(count, 1)),
            center[1] + radius * math.sin(2.0 * math.pi * index / max(count, 1)),
        )
        for index, feature in enumerate(features)
    }
    fig, ax = plt.subplots(figsize=(9.8, 7.0))
    ax.set_axis_off()
    for row in result.feature_redundancy_rows:
        left = str(row["feature_a"])
        right = str(row["feature_b"])
        strength = abs(float(row["spearman_corr"]))
        if strength < 0.65:
            continue
        x0, y0 = positions[left]
        x1, y1 = positions[right]
        color = "#E67E22" if strength >= 0.90 else "#85929E"
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=1.0 + 3.0 * strength, alpha=0.72, zorder=1)
    for feature, (x, y) in positions.items():
        high_degree = sum(
            1
            for row in result.feature_redundancy_rows
            if feature in {str(row["feature_a"]), str(row["feature_b"])}
            and abs(float(row["spearman_corr"])) >= 0.90
        )
        ax.scatter([x], [y], s=520, color="#F7F9F9", edgecolor="#2E86AB", linewidth=2.0, zorder=2)
        ax.text(x, y, feature.replace("_", "\n"), ha="center", va="center", fontsize=8, zorder=3)
        if high_degree:
            ax.text(x, y - 0.075, "cluster", ha="center", va="center", fontsize=7, color="#E67E22")
    ax.text(0.03, 0.95, "Redundant feature families are clustered before they inflate confidence.", fontsize=15, fontweight="bold", color="#17202A")
    ax.text(0.03, 0.89, "Orange edges: high redundancy; gray edges: moderate redundancy.", fontsize=9, color="#5D6D7E")
    return fig


def render_feature_synergy_map_figure(result: StaticFeatureClassPriorAuditResult):
    features = list(result.feature_names)
    gain_by_pair: dict[tuple[str, str], float] = {}
    for row in result.feature_synergy_rows:
        left = str(row["feature_a"])
        right = str(row["feature_b"])
        gain = float(row["pair_gain"])
        gain_by_pair[(left, right)] = gain
        gain_by_pair[(right, left)] = gain
    matrix = [[0.0 if left == right else gain_by_pair.get((left, right), 0.0) for right in features] for left in features]
    vmax = max(0.05, max(max(row) for row in matrix))
    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0.0, vmax=vmax)
    ax.set_xticks(range(len(features)), [feature.replace("_", "\n") for feature in features], fontsize=8)
    ax.set_yticks(range(len(features)), [feature.replace("_", "\n") for feature in features], fontsize=8)
    for i, left in enumerate(features):
        for j, right in enumerate(features):
            if i != j and matrix[i][j] >= 0.05:
                ax.text(j, i, "candidate", ha="center", va="center", fontsize=7)
    ax.set_title("Some features are only useful together; synergy is labeled as candidate evidence.", loc="left", fontsize=14, fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03, label="joint MI gain")
    return fig


def render_prior_pathology_surface_figure(result: StaticFeatureClassPriorAuditResult):
    regimes = {
        "uniform": 1.0,
        "operational_skew": 1.3,
        "rare_maneuver": 1.7,
        "stationary_dominant": 2.0,
        "switching_sparse": 2.4,
    }
    pairs = [f"{row['class_a']} vs {row['class_b']}" for row in result.prior_pathology_rows]
    base = [abs(float(row["prior_odds_log"])) + float(row["posterior_collapse_rate"]) for row in result.prior_pathology_rows]
    matrix = [[min(1.0, value * scale / 4.0) for value in base] for scale in regimes.values()]
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    image = ax.imshow(matrix, cmap="YlOrRd", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_yticks(range(len(regimes)), list(regimes), fontsize=9)
    ax.set_xticks(range(len(pairs)), [pair.replace(" vs ", "\nvs ") for pair in pairs], rotation=25, ha="right", fontsize=8)
    ax.set_title("A prior can make an otherwise separable setup pathological.", loc="left", fontsize=15, fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03, label="prior-domination proxy")
    return fig


def render_prior_flip_thresholds_figure(result: StaticFeatureClassPriorAuditResult):
    rows = list(result.prior_pathology_rows)
    labels = [f"{row['class_a']} vs {row['class_b']}" for row in rows]
    y = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(10.8, max(4.6, 0.5 * len(rows) + 1.2)))
    for index, row in enumerate(rows):
        left = float(row["observed_log_lr_min"])
        right = float(row["observed_log_lr_max"])
        threshold = float(row["flip_threshold_log_lr"])
        color = "#2E7D32" if str(row["flip_possible"]) == "True" else "#C0392B"
        ax.hlines(index, left, right, color=color, linewidth=5, alpha=0.75)
        ax.plot([threshold], [index], marker="|", markersize=18, color="#17202A", markeredgewidth=2)
        ax.text(right + 0.1, index, str(row["pathology_flag"]), va="center", fontsize=8)
    ax.axvline(0.0, color="#D5DBDB", linewidth=1)
    ax.set_yticks(y, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("log likelihood ratio")
    ax.set_title("Prior odds are compared against achievable feature evidence.", loc="left", fontsize=15, fontweight="bold")
    ax.grid(axis="x", alpha=0.22)
    return fig


def render_static_coverage_feasibility_figure(result: StaticFeatureClassPriorAuditResult):
    classes = list(result.class_names)
    features = list(result.feature_names)
    by_cell = {(str(row["class_name"]), str(row["feature"])): 1.0 - float(row["empty_bin_rate"]) for row in result.coverage_rows}
    matrix = [[by_cell.get((class_name, feature), 0.0) for feature in features] for class_name in classes]
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    image = ax.imshow(matrix, cmap="YlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_yticks(range(len(classes)), [class_name.replace("_", " ") for class_name in classes], fontsize=9)
    ax.set_xticks(range(len(features)), [feature.replace("_", "\n") for feature in features], fontsize=8)
    for i, class_name in enumerate(classes):
        for j, feature in enumerate(features):
            value = matrix[i][j]
            ax.text(j, i, "covered" if value >= 0.6 else "thin", ha="center", va="center", fontsize=7)
    ax.set_title("The static stage checks whether intended boundary regions are reachable.", loc="left", fontsize=14, fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03, label="occupied-bin fraction")
    return fig


def render_static_leakage_provenance_figure(result: StaticFeatureClassPriorAuditResult):
    rows = list(result.leakage_rows)
    columns = ("feature", "online_available", "future_dependency_flag", "metadata_leakage_flag", "label_rule_overlap_flag", "status")
    fig, ax = plt.subplots(figsize=(12.5, max(4.2, 0.38 * len(rows) + 1.3)))
    ax.axis("off")
    table = ax.table(
        cellText=[[str(row.get(column, "")) for column in columns] for row in rows],
        colLabels=("feature", "online?", "future?", "metadata?", "label rule?", "status"),
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.25)
    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_edgecolor("#D5DBDB")
        if row_index == 0:
            cell.set_facecolor("#17202A")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif col_index == len(columns) - 1:
            cell.set_facecolor(_status_color(str(rows[row_index - 1].get("status", ""))))
    ax.set_title("Leakage and feature provenance are checked before the classifier sees the data.", fontsize=14, fontweight="bold", pad=12)
    return fig


def render_static_audit_action_router_figure(result: StaticFeatureClassPriorAuditResult):
    routes = [
        ("class overlap", "revise class definition"),
        ("feature blindness", "revise feature set"),
        ("redundancy", "cluster / drop / regularize"),
        ("synergy candidate", "test pair / add interaction"),
        ("prior domination", "revise prior / evidence threshold"),
        ("coverage gap", "send objective to Corpus Explorer"),
        ("leakage risk", "block study"),
        ("pass", "promote to Corpus Explorer"),
    ]
    fig, ax = plt.subplots(figsize=(12.0, 6.5))
    ax.set_axis_off()
    ax.text(0.04, 0.93, "Static audit findings become actions, not just metrics.", fontsize=16, fontweight="bold", color="#17202A")
    for index, (finding, action) in enumerate(routes):
        row = index // 2
        col = index % 2
        y = 0.78 - row * 0.18
        x0 = 0.06 + col * 0.48
        ax.add_patch(plt.Rectangle((x0, y), 0.18, 0.09, facecolor="#F7F9F9", edgecolor="#2E86AB", linewidth=1.5))
        ax.add_patch(plt.Rectangle((x0 + 0.27, y), 0.21, 0.09, facecolor="#F7F9F9", edgecolor="#1B998B", linewidth=1.5))
        ax.annotate("", xy=(x0 + 0.27, y + 0.045), xytext=(x0 + 0.18, y + 0.045), arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "#5D6D7E"})
        ax.text(x0 + 0.09, y + 0.045, finding, ha="center", va="center", fontsize=9, fontweight="bold", color="#2E86AB")
        ax.text(x0 + 0.375, y + 0.045, action, ha="center", va="center", fontsize=8, color="#17202A")
    ax.text(0.04, 0.06, f"Current static decision: {result.static_decision['status']}", fontsize=11, color="#5D6D7E")
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
