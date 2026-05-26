from __future__ import annotations

import json
from pathlib import Path

from .advanced_filter_decision_contracts import (
    AdvancedFilterDecisionArtifacts,
    AdvancedFilterDecisionResult,
)
from .advanced_filter_decision_rendering import (
    render_advanced_filter_decision_numeric_walkthrough_markdown,
    render_advanced_filter_decision_report,
)
from .advanced_filter_decision_runner import analyze_advanced_filter_decision


def write_advanced_filter_decision_artifacts(
    output_dir: str | Path,
    *,
    result: AdvancedFilterDecisionResult | None = None,
) -> AdvancedFilterDecisionArtifacts:
    analysis = result or analyze_advanced_filter_decision()

    run_dir = Path(output_dir) / "advanced_filter_decision_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "advanced_filter_decision_report.md"
    summary_path = run_dir / "advanced_filter_decision_summary.json"
    evidence_path = run_dir / "advanced_filter_decision_evidence.json"
    numeric_walkthrough_path = run_dir / "advanced_filter_decision_numeric_walkthrough.md"

    report_path.write_text(render_advanced_filter_decision_report(analysis), encoding="utf-8")
    numeric_walkthrough_path.write_text(
        render_advanced_filter_decision_numeric_walkthrough_markdown(analysis),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(
            {
                "imm_justified": analysis.imm_justified,
                "particle_filter_justified": analysis.particle_filter_justified,
                "transition_post_switch_gain": analysis.transition_post_switch_gain,
                "transition_overall_gain": analysis.transition_overall_gain,
                "transition_vs_kalman_post_switch_gain": analysis.transition_vs_kalman_post_switch_gain,
                "transition_vs_kalman_overall_gain": analysis.transition_vs_kalman_overall_gain,
                "short_horizon_mean_gap_sigma": analysis.short_horizon_mean_gap_sigma,
                "short_horizon_final_gap_sigma": analysis.short_horizon_final_gap_sigma,
                "velocity_aided_short_noisy_gain": analysis.velocity_aided_short_noisy_gain,
                "best_kalman_outlier_accuracy": analysis.best_kalman_outlier_accuracy,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    evidence_path.write_text(json.dumps(list(analysis.evidence_rows), indent=2), encoding="utf-8")

    return AdvancedFilterDecisionArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        summary_path=summary_path,
        evidence_path=evidence_path,
        numeric_walkthrough_path=numeric_walkthrough_path,
    )


__all__ = ["write_advanced_filter_decision_artifacts"]
