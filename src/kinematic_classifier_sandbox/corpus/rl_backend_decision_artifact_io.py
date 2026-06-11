from __future__ import annotations

import json
from pathlib import Path

from .rl_backend_decision_contracts import RlBackendDecisionArtifacts, RlBackendDecisionResult


def write_rl_backend_decision_artifacts(
    output_dir: str | Path,
    *,
    result: RlBackendDecisionResult | None = None,
) -> RlBackendDecisionArtifacts:
    from .rl_backend_decision import analyze_rl_backend_decision
    from .rl_backend_decision_reporting import render_rl_backend_decision_report

    analysis = result or analyze_rl_backend_decision()
    run_dir = Path(output_dir) / "rl_corpus_agent"
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "rl_backend_decision_report.md"
    summary_path = run_dir / "rl_backend_decision_summary.json"
    evidence_path = run_dir / "rl_backend_decision_evidence.json"

    report_path.write_text(render_rl_backend_decision_report(analysis), encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "rl_justified": analysis.rl_justified,
                "baseline_to_beat": analysis.baseline_to_beat,
                "success_metric": analysis.success_metric,
                "stress_improved_modes": list(analysis.stress_improved_modes),
                "offpolicy_mean_best_policy_minus_best_baseline": analysis.offpolicy_mean_best_policy_minus_best_baseline,
                "offpolicy_seed_promotion_rate": analysis.offpolicy_seed_promotion_rate,
                "offpolicy_best_policy_backend": analysis.offpolicy_best_policy_backend,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    evidence_path.write_text(
        json.dumps(
            [
                {
                    "criterion": row.criterion,
                    "status": row.status,
                    "value": row.value,
                    "note": row.note,
                }
                for row in analysis.decision_rows
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    return RlBackendDecisionArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        summary_path=summary_path,
        evidence_path=evidence_path,
    )
