from __future__ import annotations

from kinematic_classifier_sandbox.witnesses.benchmarks.sequential_bayes_accumulator import (
    AccumulatorBenchmarkArtifacts,
    AccumulatorBenchmarkResult,
    AccumulatorBenchmarkSummary,
    AccumulatorClassSpec,
    AccumulatorPosteriorStep,
    AccumulatorRun,
    AccumulatorTrajectory,
    SequentialBayesAccumulator,
)

__all__ = [
    "AccumulatorBenchmarkArtifacts",
    "AccumulatorBenchmarkResult",
    "AccumulatorBenchmarkSummary",
    "AccumulatorClassSpec",
    "AccumulatorPosteriorStep",
    "AccumulatorRun",
    "AccumulatorTrajectory",
    "SequentialBayesAccumulator",
    "default_accumulator_class_specs",
    "generate_accumulator_trajectories",
    "run_accumulator",
    "run_accumulator_benchmark",
    "render_accumulator_report",
    "render_accumulator_svg",
    "render_accumulator_png_bytes",
    "write_accumulator_artifacts",
]


def default_accumulator_class_specs(*args, **kwargs):
    from kinematic_classifier_sandbox.witnesses.benchmarks.sequential_bayes_accumulator import (
        default_accumulator_class_specs as _impl,
    )

    return _impl(*args, **kwargs)


def generate_accumulator_trajectories(*args, **kwargs):
    from kinematic_classifier_sandbox.witnesses.benchmarks.sequential_bayes_accumulator import (
        generate_accumulator_trajectories as _impl,
    )

    return _impl(*args, **kwargs)


def run_accumulator(*args, **kwargs):
    from kinematic_classifier_sandbox.witnesses.benchmarks.sequential_bayes_accumulator import (
        run_accumulator as _impl,
    )

    return _impl(*args, **kwargs)


def run_accumulator_benchmark(*args, **kwargs):
    from kinematic_classifier_sandbox.witnesses.benchmarks.sequential_bayes_accumulator import (
        run_accumulator_benchmark as _impl,
    )

    return _impl(*args, **kwargs)


def render_accumulator_report(*args, **kwargs):
    from kinematic_classifier_sandbox.witnesses.benchmarks.sequential_bayes_accumulator import (
        render_accumulator_report as _impl,
    )

    return _impl(*args, **kwargs)


def render_accumulator_svg(*args, **kwargs):
    from kinematic_classifier_sandbox.witnesses.benchmarks.sequential_bayes_accumulator import (
        render_accumulator_svg as _impl,
    )

    return _impl(*args, **kwargs)


def render_accumulator_png_bytes(*args, **kwargs):
    from kinematic_classifier_sandbox.witnesses.benchmarks.sequential_bayes_accumulator import (
        render_accumulator_png_bytes as _impl,
    )

    return _impl(*args, **kwargs)


def write_accumulator_artifacts(*args, **kwargs):
    from kinematic_classifier_sandbox.witnesses.benchmarks.sequential_bayes_accumulator import (
        write_accumulator_artifacts as _impl,
    )

    return _impl(*args, **kwargs)
