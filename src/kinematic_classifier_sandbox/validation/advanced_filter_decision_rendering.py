from __future__ import annotations

from kinematic_classifier_sandbox.reports.markdown import MarkdownDocument

from .advanced_filter_decision_contracts import (
    AdvancedFilterDecisionArtifacts,
    AdvancedFilterDecisionResult,
)


def render_advanced_filter_decision_report(result: AdvancedFilterDecisionResult) -> str:
    report = MarkdownDocument("Advanced Filter Decision Report")
    report.paragraph(
        "Milestone 17 decision gate for whether the repo should advance from the current ladder to IMM or "
        "particle filtering."
    )
    report.heading("Decision", level=2)
    report.bullet_list([f"IMM justified now: `{result.imm_justified}`", f"Particle filter justified now: `{result.particle_filter_justified}`"])
    report.heading("Key Evidence", level=2)
    report.bullet_list(
        [
            f"Transition-matrix post-switch gain over static accumulator: `{result.transition_post_switch_gain:.3f}`",
            f"Transition-matrix overall gain over static accumulator: `{result.transition_overall_gain:.3f}`",
            f"Transition-matrix post-switch gain over Kalman mode bank: `{result.transition_vs_kalman_post_switch_gain:.3f}`",
            f"Transition-matrix overall gain over Kalman mode bank: `{result.transition_vs_kalman_overall_gain:.3f}`",
            f"Short-horizon mean normalized gap at nominal noise: `{result.short_horizon_mean_gap_sigma:.3f}` sigma",
            f"Short-horizon final normalized gap at nominal noise: `{result.short_horizon_final_gap_sigma:.3f}` sigma",
            f"Velocity-aided short-noisy gain over position-only Kalman: `{result.velocity_aided_short_noisy_gain:.3f}`",
            f"Best outlier accuracy among current Kalman variants: `{result.best_kalman_outlier_accuracy:.3f}`",
        ]
    )
    report.heading("Gate Table", level=2)
    report.table(
        ["gate", "criterion", "status", "value", "note"],
        [
            (str(row["gate"]), str(row["criterion"]), str(row["status"]), str(row["value"]), str(row["note"]))
            for row in result.evidence_rows
        ],
    )
    report.heading("Recommendation", level=2)
    report.bullet_list(
        [
            "Defer IMM for now. The transition-matrix accumulator already improves switching behavior and currently beats "
            "the switching-mode Kalman bank on post-switch accuracy, so the repo still lacks evidence that the simpler "
            "transition model is insufficient.",
            "Defer particle filtering for now. The strongest current hard case is `short_noisy`, and that case is "
            "evidence-limited: direct velocity sensing helps materially, while the identifiability study shows position-only "
            "separation stays near or below one sigma for much of the horizon.",
            "Revisit IMM only after a switching-mode Kalman or multiple-model variant matches or beats the transition-matrix "
            "accumulator and still leaves the switching scenarios inadequately explained.",
            "Revisit particle filtering only after adding a documented nonlinear or non-Gaussian benchmark where robust "
            "Kalman-style methods still fail for reasons other than sensing limits or feature excitation.",
        ]
    )
    return report.text()


def render_advanced_filter_decision_numeric_walkthrough_markdown(result: AdvancedFilterDecisionResult) -> str:
    imm_rows = [row for row in result.evidence_rows if row["gate"] == "IMM"]
    pf_rows = [row for row in result.evidence_rows if row["gate"] == "Particle Filter"]

    def _implication_for_imm(row: dict[str, object]) -> str:
        return (
            "supports deferral because the simpler transition layer still adds value"
            if row["status"] == "met"
            else "would push the repo toward a stronger switching backend"
        )

    def _implication_for_pf(row: dict[str, object]) -> str:
        return (
            "blocks PF because the required failure evidence is still missing"
            if row["status"] in {"missing", "failed"}
            else "would support PF if other gates aligned"
        )

    report = MarkdownDocument("Advanced Filter Decision Numeric Walkthrough")
    report.paragraph(
        "This worked example uses the exact evidence values from `analyze_advanced_filter_decision()` to show why "
        "the repo currently defers IMM and particle filtering."
    )
    report.heading("IMM Gate", level=2)
    report.paragraph("The implemented IMM decision logic is currently conservative:")
    report.fence(r"\text{IMM justified now} = \text{False}", language="tex")
    report.paragraph("but it is based on measured switching gains:")
    report.fence(
        rf"\Delta_{{\mathrm{{post-switch}}}}^{{\mathrm{{TM-static}}}} = {result.transition_post_switch_gain:.3f}",
        language="tex",
    )
    report.fence(
        rf"\Delta_{{\mathrm{{overall}}}}^{{\mathrm{{TM-static}}}} = {result.transition_overall_gain:.3f}",
        language="tex",
    )
    report.fence(
        rf"\Delta_{{\mathrm{{post-switch}}}}^{{\mathrm{{TM-kalman}}}} = {result.transition_vs_kalman_post_switch_gain:.3f}",
        language="tex",
    )
    report.fence(
        rf"\Delta_{{\mathrm{{overall}}}}^{{\mathrm{{TM-kalman}}}} = {result.transition_vs_kalman_overall_gain:.3f}",
        language="tex",
    )
    report.paragraph("Interpretation:")
    report.table(
        ["criterion", "status", "value", "implication"],
        [
            (str(row["criterion"]), str(row["status"]), str(row["value"]), _implication_for_imm(row))
            for row in imm_rows
        ],
    )
    report.paragraph(
        "The crucial number is the post-switch comparison against the current switching Kalman bank. "
        "Because the value stays positive, the simpler transition-matrix accumulator still outperforms "
        "the current model-based switching alternative on the exact regime where IMM would need to justify itself."
    )
    report.heading("Particle-Filter Gate", level=2)
    report.paragraph("The particle-filter gate is also evidence-driven rather than aspirational.")
    report.paragraph("At nominal noise, the position-only identifiability gap is:")
    report.fence(rf"\text{{mean normalized gap}} = {result.short_horizon_mean_gap_sigma:.3f}\sigma", language="tex")
    report.paragraph("and the final-step gap is:")
    report.fence(rf"\text{{final normalized gap}} = {result.short_horizon_final_gap_sigma:.3f}\sigma", language="tex")
    report.paragraph("The direct-velocity measurement gain on the `short_noisy` case is:")
    report.fence(
        rf"\Delta_{{\mathrm{{short\_noisy}}}}^{{\mathrm{{vel-aided}}}} = {result.velocity_aided_short_noisy_gain:.3f}",
        language="tex",
    )
    report.paragraph("The best outlier accuracy among current Kalman variants is:")
    report.fence(
        rf"A_{{\mathrm{{outlier}}}}^{{\mathrm{{best\ Kalman}}}} = {result.best_kalman_outlier_accuracy:.3f}",
        language="tex",
    )
    report.table(
        ["criterion", "status", "value", "implication"],
        [
            (str(row["criterion"]), str(row["status"]), str(row["value"]), _implication_for_pf(row))
            for row in pf_rows
        ],
    )
    report.heading("Why The Decision Is `defer`", level=2)
    report.paragraph(
        "The strongest hard case remains sensing-limited rather than inference-limited. Direct velocity helps materially on "
        "`short_noisy`, which means the current bottleneck is still evidence quality."
    )
    report.paragraph(
        "At the same time, the robust Kalman variants already recover substantial outlier performance, so the repo "
        "does not yet have a clean nonlinear or non-Gaussian witness problem that simpler methods fail."
    )
    report.paragraph("That is why the current implemented decisions are:")
    report.bullet_list(
        [
            f"IMM justified now: `{result.imm_justified}`",
            f"Particle filter justified now: `{result.particle_filter_justified}`",
        ]
    )
    report.paragraph(
        "The correct next proof burden is therefore not “add PF anyway,” but “construct a benchmark where switching "
        "or non-Gaussian structure defeats the current ladder for reasons that sensing improvements and robust Kalman "
        "variants cannot explain away."
    )
    return report.text()

__all__ = [
    "AdvancedFilterDecisionArtifacts",
    "render_advanced_filter_decision_report",
    "render_advanced_filter_decision_numeric_walkthrough_markdown",
]
