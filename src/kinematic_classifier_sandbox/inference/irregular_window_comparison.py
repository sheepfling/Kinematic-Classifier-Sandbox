from __future__ import annotations

from kinematic_classifier_sandbox.witnesses.benchmarks.irregular_window_comparison import (
    WindowRegimeArtifacts,
    WindowRegimeComparisonResult,
    WindowRegimeFeatureRow,
    WindowRegimeSummaryRow,
    WindowRegimeTrajectory,
    _duration_window,
    _gaussian_logpdf,
    _normalize,
    _sample_count_window,
    analyze_irregular_window_comparison,
    generate_window_regime_trajectories,
    render_irregular_window_report,
    write_irregular_window_artifacts,
)

__all__ = [
    "WindowRegimeArtifacts",
    "WindowRegimeComparisonResult",
    "WindowRegimeFeatureRow",
    "WindowRegimeSummaryRow",
    "WindowRegimeTrajectory",
    "generate_window_regime_trajectories",
    "analyze_irregular_window_comparison",
    "render_irregular_window_report",
    "write_irregular_window_artifacts",
    "_sample_count_window",
    "_duration_window",
    "_gaussian_logpdf",
    "_normalize",
]
