"""Curated corpus package surface with explicit imports and wrappers.

This package groups corpus-generation, corpus-evaluation, and corpus-decision
helpers. The public surface is intentionally curated around the main analysis
and artifact entrypoints without using dynamic attribute hooks.
"""

from __future__ import annotations

from .adequacy_contracts import (
    CorpusAdequacyArtifacts,
    CorpusAdequacyResult,
    CorpusAdequacyScorecard,
    CorpusAdequacySummary,
    CorpusAdequacyThresholds,
)
from .classifier_scoring_types import (
    CorpusClassifierScoringArtifacts,
    CorpusClassifierScoringResult,
)
from .coverage_contracts import (
    CoverageReportArtifacts,
    CoverageReportResult,
    CoverageReportSummary,
)
from .rl_backend_decision_contracts import (
    RlBackendDecisionArtifacts,
    RlBackendDecisionGateRow,
    RlBackendDecisionResult,
)
from .search_baseline_contracts import (
    CorpusSearchBaselineArtifacts,
    CorpusSearchBaselineResult,
)
from .trajectory_exploration.contracts import (
    TrajectoryExplorationArtifacts,
    TrajectoryExplorationBenchmarkResult,
    TrajectoryExplorationEvaluation,
    TrajectoryExplorationObjective,
    TrajectoryExplorationProposal,
    TrajectoryExplorationResult,
)
from .selected_generated_corpus_contracts import (
    SelectedGeneratedCorpusArtifacts,
    SelectedGeneratedCorpusResult,
)
from .synthesis_comparison_contracts import (
    CorpusSynthesisComparisonArtifacts,
    CorpusSynthesisComparisonResult,
)

__all__ = [
    "CorpusAdequacyArtifacts",
    "CorpusAdequacyResult",
    "CorpusAdequacyScorecard",
    "CorpusAdequacySummary",
    "CorpusAdequacyThresholds",
    "CorpusClassifierScoringArtifacts",
    "CorpusClassifierScoringResult",
    "CorpusSearchBaselineArtifacts",
    "CorpusSearchBaselineResult",
    "CorpusSynthesisComparisonArtifacts",
    "CorpusSynthesisComparisonResult",
    "CoverageReportArtifacts",
    "CoverageReportResult",
    "CoverageReportSummary",
    "RlBackendDecisionArtifacts",
    "RlBackendDecisionGateRow",
    "RlBackendDecisionResult",
    "TrajectoryExplorationArtifacts",
    "TrajectoryExplorationBenchmarkResult",
    "TrajectoryExplorationEvaluation",
    "TrajectoryExplorationObjective",
    "TrajectoryExplorationProposal",
    "TrajectoryExplorationResult",
    "SelectedGeneratedCorpusArtifacts",
    "SelectedGeneratedCorpusResult",
    "analyze_corpus_adequacy",
    "analyze_corpus_classifier_scoring",
    "analyze_corpus_search_baseline",
    "analyze_corpus_synthesis_comparison",
    "analyze_coverage_report",
    "analyze_rl_backend_decision",
    "analyze_selected_generated_corpus",
    "analyze_trajectory_exploration_benchmarks",
    "render_corpus_adequacy_report",
    "render_coverage_report",
    "render_rl_backend_decision_report",
    "write_corpus_adequacy_artifacts",
    "write_corpus_classifier_scoring_artifacts",
    "write_corpus_search_baseline_artifacts",
    "write_corpus_synthesis_comparison_artifacts",
    "write_coverage_report_artifacts",
    "write_rl_backend_decision_artifacts",
    "write_selected_generated_corpus_artifacts",
    "write_trajectory_exploration_artifacts",
]


def analyze_corpus_adequacy(*args, **kwargs):
    from .adequacy_audit import analyze_corpus_adequacy as _impl

    return _impl(*args, **kwargs)


def analyze_corpus_classifier_scoring(*args, **kwargs):
    from .classifier_scoring import analyze_corpus_classifier_scoring as _impl

    return _impl(*args, **kwargs)


def analyze_corpus_search_baseline(*args, **kwargs):
    from .search_baseline import analyze_corpus_search_baseline as _impl

    return _impl(*args, **kwargs)


def analyze_corpus_synthesis_comparison(*args, **kwargs):
    from .synthesis_comparison import analyze_corpus_synthesis_comparison as _impl

    return _impl(*args, **kwargs)


def analyze_coverage_report(*args, **kwargs):
    from .coverage_report import analyze_coverage_report as _impl

    return _impl(*args, **kwargs)


def analyze_rl_backend_decision(*args, **kwargs):
    from .rl_backend_decision import analyze_rl_backend_decision as _impl

    return _impl(*args, **kwargs)


def analyze_selected_generated_corpus(*args, **kwargs):
    from .selected_generated_corpus import analyze_selected_generated_corpus as _impl

    return _impl(*args, **kwargs)


def analyze_trajectory_exploration_benchmarks(*args, **kwargs):
    from .trajectory_exploration import analyze_trajectory_exploration_benchmarks as _impl

    return _impl(*args, **kwargs)


def render_corpus_adequacy_report(*args, **kwargs):
    from .adequacy_audit import render_corpus_adequacy_report as _impl

    return _impl(*args, **kwargs)


def render_coverage_report(*args, **kwargs):
    from .coverage_report import render_coverage_report as _impl

    return _impl(*args, **kwargs)


def render_rl_backend_decision_report(*args, **kwargs):
    from .rl_backend_decision_reporting import render_rl_backend_decision_report as _impl

    return _impl(*args, **kwargs)


def write_corpus_adequacy_artifacts(*args, **kwargs):
    from .adequacy_artifact_io import write_corpus_adequacy_artifacts as _impl

    return _impl(*args, **kwargs)


def write_corpus_classifier_scoring_artifacts(*args, **kwargs):
    from .classifier_scoring_artifact_io import write_corpus_classifier_scoring_artifacts as _impl

    return _impl(*args, **kwargs)


def write_corpus_search_baseline_artifacts(*args, **kwargs):
    from .search_baseline_artifact_io import write_corpus_search_baseline_artifacts as _impl

    return _impl(*args, **kwargs)


def write_corpus_synthesis_comparison_artifacts(*args, **kwargs):
    from .synthesis_comparison_artifact_io import write_corpus_synthesis_comparison_artifacts as _impl

    return _impl(*args, **kwargs)


def write_coverage_report_artifacts(*args, **kwargs):
    from .coverage_artifact_io import write_coverage_report_artifacts as _impl

    return _impl(*args, **kwargs)


def write_rl_backend_decision_artifacts(*args, **kwargs):
    from .rl_backend_decision_artifact_io import write_rl_backend_decision_artifacts as _impl

    return _impl(*args, **kwargs)


def write_selected_generated_corpus_artifacts(*args, **kwargs):
    from .selected_generated_corpus_artifact_io import (
        write_selected_generated_corpus_artifacts as _impl,
    )

    return _impl(*args, **kwargs)


def write_trajectory_exploration_artifacts(*args, **kwargs):
    from .trajectory_exploration import write_trajectory_exploration_artifacts as _impl

    return _impl(*args, **kwargs)
