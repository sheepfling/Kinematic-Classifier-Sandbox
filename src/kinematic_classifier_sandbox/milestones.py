from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .contracts import write_milestone0_sample_run_artifacts
from .feature_analysis import write_feature_analysis_artifacts
from .kalman_filter_bank import write_kalman_bank_artifacts
from .monte_carlo_benchmark import write_monte_carlo_artifacts
from .pca_analysis import write_pca_analysis_artifacts
from .pointwise_baseline import run_pointwise_benchmark, write_pointwise_benchmark_artifacts
from .sequential_bayes_accumulator import run_accumulator_benchmark, write_accumulator_artifacts
from .trajectory_generator import write_trajectory_generator_artifacts
from .windowed_baseline import run_windowed_benchmark, write_windowed_benchmark_artifacts


@dataclass(frozen=True, slots=True)
class MilestoneEntry:
    milestone_id: str
    title: str
    status: str
    summary: str
    artifact_dir_name: str
    command_example: str


@dataclass(frozen=True, slots=True)
class MilestoneRunResult:
    milestone_id: str
    artifact_dir: Path
    report_path: Path | None


def _run_m0(output_root: Path) -> MilestoneRunResult:
    artifacts = write_milestone0_sample_run_artifacts(output_root)
    return MilestoneRunResult("m0", artifacts.run_dir, artifacts.report_path)


def _run_m1(output_root: Path) -> MilestoneRunResult:
    result = run_pointwise_benchmark(seed=7)
    artifacts = write_pointwise_benchmark_artifacts(output_root, result=result)
    return MilestoneRunResult("m1", artifacts.run_dir, artifacts.report_path)


def _run_m2(output_root: Path) -> MilestoneRunResult:
    result = run_windowed_benchmark(seed=7)
    artifacts = write_windowed_benchmark_artifacts(output_root, result=result)
    return MilestoneRunResult("m2", artifacts.run_dir, artifacts.report_path)


def _run_m3(output_root: Path) -> MilestoneRunResult:
    result = run_accumulator_benchmark(seed=7)
    artifacts = write_accumulator_artifacts(output_root, result=result)
    return MilestoneRunResult("m3", artifacts.run_dir, artifacts.report_path)


def _run_m4(output_root: Path) -> MilestoneRunResult:
    artifacts = write_monte_carlo_artifacts(output_root, result=None)
    return MilestoneRunResult("m4", artifacts.run_dir, artifacts.report_path)


def _run_m5(output_root: Path) -> MilestoneRunResult:
    artifacts = write_trajectory_generator_artifacts(output_root, seed=7)
    return MilestoneRunResult("m5", artifacts.run_dir, artifacts.report_path)


def _run_m6(output_root: Path) -> MilestoneRunResult:
    artifacts = write_feature_analysis_artifacts(output_root, seed=7, trajectories_per_class=5)
    return MilestoneRunResult("m6", artifacts.run_dir, artifacts.report_path)


def _run_m7(output_root: Path) -> MilestoneRunResult:
    artifacts = write_kalman_bank_artifacts(output_root)
    return MilestoneRunResult("m7", artifacts.run_dir, artifacts.report_path)


def _run_m8(output_root: Path) -> MilestoneRunResult:
    artifacts = write_pca_analysis_artifacts(output_root, seed=7, trajectories_per_class=5, n_components=3)
    return MilestoneRunResult("m8", artifacts.run_dir, artifacts.report_path)


def _run_m9(output_root: Path) -> MilestoneRunResult:
    artifacts = write_trajectory_generator_artifacts(output_root, seed=7, trajectories_per_class=5)
    return MilestoneRunResult("m9", artifacts.run_dir, artifacts.report_path)


MILESTONE_REGISTRY: tuple[MilestoneEntry, ...] = (
    MilestoneEntry(
        milestone_id="m0",
        title="Contracts and Sample Artifact Validation",
        status="done",
        summary="Writes the baseline contract demo run directory and validates the core artifact schema.",
        artifact_dir_name="milestone0_contract_demo",
        command_example="python3 scripts/run_milestone.py m0",
    ),
    MilestoneEntry(
        milestone_id="m1",
        title="Pointwise Baseline",
        status="done",
        summary="Runs the Gaussian pointwise benchmark and writes its report, confusion outputs, and posterior history.",
        artifact_dir_name="pointwise_baseline",
        command_example="python3 scripts/run_milestone.py m1",
    ),
    MilestoneEntry(
        milestone_id="m2",
        title="Windowed Feature Baseline",
        status="done",
        summary="Runs the raw and robust windowed-feature benchmark and writes feature, posterior, confusion, and plot artifacts.",
        artifact_dir_name="windowed_baseline",
        command_example="python3 scripts/run_milestone.py m2",
    ),
    MilestoneEntry(
        milestone_id="m3",
        title="Sequential Bayesian Accumulator",
        status="done",
        summary="Runs the accumulator benchmark and emits posterior, log-odds, confidence-crossing, and prior-sensitivity artifacts.",
        artifact_dir_name="bayes_accumulator",
        command_example="python3 scripts/run_milestone.py m3",
    ),
    MilestoneEntry(
        milestone_id="m4",
        title="Monte Carlo Pack",
        status="done",
        summary="Builds the accumulator Monte Carlo report with calibration, confusion, time-to-confidence, and time-to-correct plots.",
        artifact_dir_name="monte_carlo_accumulator",
        command_example="python3 scripts/run_milestone.py m4",
    ),
    MilestoneEntry(
        milestone_id="m5",
        title="Trajectory Generator Foundation",
        status="done",
        summary="Writes the 1D class definitions, tier manifests, generated trajectories, true states, and overview plot.",
        artifact_dir_name="trajectory_generator_v1",
        command_example="python3 scripts/run_milestone.py m5",
    ),
    MilestoneEntry(
        milestone_id="m6",
        title="Identifiability and Feature Analysis",
        status="done",
        summary="Writes feature excitation, separability, overlap, confusability, and feature-ranking artifacts.",
        artifact_dir_name="feature_analysis_v1",
        command_example="python3 scripts/run_milestone.py m6",
    ),
    MilestoneEntry(
        milestone_id="m7",
        title="Kalman Filter Bank",
        status="done",
        summary="Runs the Kalman filter bank benchmark and writes innovation, state-estimate, posterior, and confusion artifacts.",
        artifact_dir_name="kalman_filter_bank",
        command_example="python3 scripts/run_milestone.py m7",
    ),
    MilestoneEntry(
        milestone_id="m8",
        title="PCA and Principal-Feature Analysis",
        status="done",
        summary="Runs PCA on engineered features and writes coordinate, loading, explained-variance, and plot artifacts.",
        artifact_dir_name="pca_analysis_v1",
        command_example="python3 scripts/run_milestone.py m8",
    ),
    MilestoneEntry(
        milestone_id="m9",
        title="Generator Stack Graduation Surface",
        status="done",
        summary="Reruns the generator-stack artifact surface including tier datasets plus explicit short-horizon, perturbation-sweep, and switching scenario libraries.",
        artifact_dir_name="trajectory_generator_v1",
        command_example="python3 scripts/run_milestone.py m9",
    ),
)


_RUNNERS: dict[str, Callable[[Path], MilestoneRunResult]] = {
    "m0": _run_m0,
    "m1": _run_m1,
    "m2": _run_m2,
    "m3": _run_m3,
    "m4": _run_m4,
    "m5": _run_m5,
    "m6": _run_m6,
    "m7": _run_m7,
    "m8": _run_m8,
    "m9": _run_m9,
}


def list_milestones() -> tuple[MilestoneEntry, ...]:
    return MILESTONE_REGISTRY


def resolve_milestone_ids(requested: str) -> tuple[str, ...]:
    normalized = requested.strip().lower()
    if normalized == "all":
        return tuple(entry.milestone_id for entry in MILESTONE_REGISTRY if entry.milestone_id != "m0")
    if normalized == "m1-m9":
        return tuple(f"m{index}" for index in range(1, 10))
    if normalized == "m0-m9":
        return tuple(f"m{index}" for index in range(10))
    if normalized in _RUNNERS:
        return (normalized,)
    raise KeyError(f"unknown milestone selection: {requested}")


def run_milestones(
    output_dir: str | Path,
    *,
    selection: str,
) -> tuple[MilestoneRunResult, ...]:
    output_root = Path(output_dir)
    results: list[MilestoneRunResult] = []
    for milestone_id in resolve_milestone_ids(selection):
        results.append(_RUNNERS[milestone_id](output_root))
    return tuple(results)
