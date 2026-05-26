from __future__ import annotations

from ..markdown_builder import MarkdownDocument
from .rl_backend_decision_contracts import RlBackendDecisionResult


def render_rl_backend_decision_report(result: RlBackendDecisionResult) -> str:
    doc = MarkdownDocument("RL Backend Decision Report")
    doc.paragraph(
        "Milestone 29 decision gate for whether CorpusGym should advance from search, quality-diversity, and adaptive stress methods to an RL backend."
    )

    doc.heading("Decision", level=2)
    doc.bullet_list([f"RL justified now: `{result.rl_justified}`"])

    doc.heading("Current Formulation", level=2)
    doc.bullet_list(
        [
            f"State space: `{', '.join(result.state_space)}`",
            f"Action space: `{', '.join(result.action_space)}`",
            f"Reward components: `{', '.join(result.reward_components)}`",
            f"Episode definition: {result.episode_definition}",
        ]
    )

    doc.heading("Baseline To Beat", level=2)
    doc.bullet_list(
        [
            f"Selected-search mean utility: `{result.search_selected_mean_utility:.3f}`",
            f"QD final coverage fraction: `{result.qd_final_coverage_fraction:.3f}`",
            f"QD best feature-target excitation: `{result.qd_best_feature_excitation:.3f}`",
            f"Stress modes improved over random baseline: `{result.stress_resolved_modes}/{result.stress_total_modes}`",
        ]
    )

    doc.heading("Success Metric Required To Justify RL", level=2)
    doc.bullet_list([result.success_metric])

    doc.heading("Gate Table", level=2)
    doc.table(
        ["criterion", "status", "value", "note"],
        [
            (row["criterion"], row["status"], str(row["value"]), row["note"])
            for row in result.decision_rows
        ],
    )

    doc.heading("Recommendation", level=2)
    doc.bullet_list(
        [
            "Keep RL as a no-go for now.",
            "The current repo already gets measurable gains from non-RL methods across M26, M27, and M28.",
            "Revisit RL only after adding a genuinely sequential CorpusGym environment, or after a future corpus objective remains unresolved under matched-budget search and quality-diversity baselines.",
        ]
    )

    return doc.text()
