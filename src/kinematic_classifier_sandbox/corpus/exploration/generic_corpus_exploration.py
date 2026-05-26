from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from kinematic_classifier_sandbox.utils.io import write_csv
from kinematic_classifier_sandbox.utils.plotting import _figure_to_png
from kinematic_classifier_sandbox.runtime_paths import prepare_matplotlib
from kinematic_classifier_sandbox.utils.runtime import repo_root

from ...markdown_builder import MarkdownDocument
from ..policy import load_corpus_policy_spec
from .backend_adapter_proof import (
    BackendCandidateSpec,
    _adapter_map,
)

ROOT = repo_root()
DEFAULT_GENERIC_CORPUS_EXPLORATION_WEIGHT_SWEEP_CONFIG_PATH = (
    ROOT / "experiments" / "generic_corpus_exploration_weight_sweep" / "generic_corpus_exploration_weight_sweep.yaml"
)

plt = prepare_matplotlib()


def _normalize_weights(weights: GenericCorpusExplorationWeights) -> GenericCorpusExplorationWeights:
    values = asdict(weights)
    total = sum(float(value) for value in values.values())
    if total <= 0.0:
        raise ValueError("generic corpus exploration weights must sum to a positive value")
    return GenericCorpusExplorationWeights(**{key: float(value) / total for key, value in values.items()})


def _perturb_weights(
    weights: GenericCorpusExplorationWeights,
    *,
    focus: str,
    delta: float,
) -> GenericCorpusExplorationWeights:
    if focus not in asdict(weights):
        raise KeyError(f"unknown weight component: {focus}")
    values = asdict(weights)
    values[focus] = max(float(values[focus]) + delta, 1e-6)
    total = sum(float(value) for value in values.values())
    return GenericCorpusExplorationWeights(**{key: float(value) / total for key, value in values.items()})


def _weights_to_dict(weights: GenericCorpusExplorationWeights) -> dict[str, float]:
    return {key: float(value) for key, value in asdict(weights).items()}


def _weights_from_mapping(mapping: dict[str, Any]) -> GenericCorpusExplorationWeights:
    values = asdict(DEFAULT_GENERIC_CORPUS_EXPLORATION_WEIGHTS)
    missing = [key for key in values if key not in mapping]
    if missing:
        raise ValueError(f"missing weight keys: {', '.join(sorted(missing))}")
    return GenericCorpusExplorationWeights(
        validity=float(mapping["validity"]),
        coverage_novelty=float(mapping["coverage_novelty"]),
        boundary=float(mapping["boundary"]),
        stress=float(mapping["stress"]),
        environment=float(mapping["environment"]),
        provenance=float(mapping["provenance"]),
    )


def _sweep_variant_from_mapping(mapping: dict[str, Any]) -> GenericCorpusExplorationSweepVariant:
    variant_id = str(mapping["variant_id"])
    description = str(mapping.get("description", variant_id))
    weights_payload = mapping.get("weights")
    if not isinstance(weights_payload, dict):
        raise ValueError(f"variant {variant_id} must define a weights mapping")
    return GenericCorpusExplorationSweepVariant(
        variant_id=variant_id,
        description=description,
        weights=_weights_from_mapping(weights_payload),
    )


def _sweep_config_to_dict(config: GenericCorpusExplorationSweepConfig) -> dict[str, Any]:
    return {
        "baseline_variant_id": config.baseline_variant_id,
        "variants": [
            {
                "variant_id": variant.variant_id,
                "description": variant.description,
                "weights": _weights_to_dict(variant.weights),
            }
            for variant in config.variants
        ],
    }


def load_generic_corpus_exploration_weight_sweep_config(
    path: str | Path,
) -> GenericCorpusExplorationSweepConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    variant_rows = payload.get("variants", [])
    if not isinstance(variant_rows, list) or not variant_rows:
        raise ValueError("generic corpus exploration weight sweep config must define a non-empty variants list")
    variants = tuple(_sweep_variant_from_mapping(dict(item)) for item in variant_rows)
    baseline_variant_id = str(payload.get("baseline_variant_id", variants[0].variant_id))
    if not any(variant.variant_id == baseline_variant_id for variant in variants):
        raise ValueError(f"baseline variant id {baseline_variant_id!r} is not present in the variants list")
    return GenericCorpusExplorationSweepConfig(
        baseline_variant_id=baseline_variant_id,
        variants=variants,
        config_path=config_path,
    )


def _set_jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / max(len(left | right), 1)


@dataclass(frozen=True, slots=True)
class GenericCorpusExplorationResult:
    exploration_manifest: dict[str, Any]
    candidate_score_rows: tuple[dict[str, Any], ...]
    archive_cell_rows: tuple[dict[str, Any], ...]
    selected_corpus_manifest: dict[str, Any]
    backend_comparison_rows: tuple[dict[str, Any], ...]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class GenericCorpusExplorationArtifacts:
    run_dir: Path
    exploration_manifest_path: Path
    candidate_scores_path: Path
    archive_cells_path: Path
    selected_corpus_manifest_path: Path
    backend_comparison_path: Path
    report_path: Path
    numeric_walkthrough_path: Path
    backend_coverage_png_path: Path
    archive_heatmap_png_path: Path
    score_parallel_png_path: Path
    selected_gallery_png_path: Path
    provenance_dashboard_png_path: Path


@dataclass(frozen=True, slots=True)
class GenericCorpusExplorationWeights:
    validity: float = 0.22
    coverage_novelty: float = 0.18
    boundary: float = 0.18
    stress: float = 0.18
    environment: float = 0.12
    provenance: float = 0.12


@dataclass(frozen=True, slots=True)
class GenericCorpusExplorationSweepVariant:
    variant_id: str
    description: str
    weights: GenericCorpusExplorationWeights


@dataclass(frozen=True, slots=True)
class GenericCorpusExplorationSweepConfig:
    baseline_variant_id: str
    variants: tuple[GenericCorpusExplorationSweepVariant, ...]
    config_path: Path | None = None


@dataclass(frozen=True, slots=True)
class GenericCorpusExplorationSweepRow:
    variant_id: str
    description: str
    weight_validity: float
    weight_coverage_novelty: float
    weight_boundary: float
    weight_stress: float
    weight_environment: float
    weight_provenance: float
    selected_coverage: int
    random_baseline_coverage: int
    coverage_delta_vs_random: int
    coverage_delta_vs_baseline: int
    selected_backend_count: int
    selected_scenario_count: int
    selected_candidate_count: int
    selected_cell_count: int
    mean_total_utility: float
    mean_total_utility_delta_vs_baseline: float
    mean_provenance_completeness: float
    candidate_jaccard_vs_baseline: float
    cell_jaccard_vs_baseline: float
    selected_candidate_ids: tuple[str, ...]
    selected_cell_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GenericCorpusExplorationSweepResult:
    baseline_variant_id: str
    variants: tuple[GenericCorpusExplorationSweepVariant, ...]
    rows: tuple[GenericCorpusExplorationSweepRow, ...]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class GenericCorpusExplorationSweepArtifacts:
    run_dir: Path
    config_path: Path
    report_path: Path
    summary_path: Path
    rows_path: Path
    overlap_matrix_path: Path
    weight_matrix_path: Path
    tradeoff_png_path: Path
    selected_set_png_path: Path
    baseline_manifest_path: Path


def _default_generic_corpus_exploration_weights() -> GenericCorpusExplorationWeights:
    weights = load_corpus_policy_spec().generic_explorer_weights
    return GenericCorpusExplorationWeights(
        validity=weights["validity"],
        coverage_novelty=weights["coverage_novelty"],
        boundary=weights["boundary_score"],
        stress=weights["classifier_stress"],
        environment=weights["environment_score"],
        provenance=weights["provenance_completeness"],
    )


DEFAULT_GENERIC_CORPUS_EXPLORATION_WEIGHTS = _default_generic_corpus_exploration_weights()


def _candidate_pool() -> tuple[BackendCandidateSpec, ...]:
    candidates: list[BackendCandidateSpec] = []
    for replicate in range(3):
        candidates.append(
            BackendCandidateSpec(
                candidate_id=f"boundary_param_{replicate}",
                scenario_id="shared_boundary_cv_ca",
                scenario_family="shared_boundary_case",
                target_class="constant_velocity",
                difficulty_tier="boundary_v1",
                seed=700 + replicate,
                duration=2.0 + 0.15 * replicate,
                sample_period=0.5,
                initial_position=0.0,
                initial_velocity=1.00 + 0.03 * replicate,
                acceleration=0.08 + 0.02 * replicate,
                measurement_std=0.03,
                provenance={"search_method": "dashboard_seeded", "search_iteration": len(candidates)},
            )
        )
        candidates.append(
            BackendCandidateSpec(
                candidate_id=f"switching_ctrl_{replicate}",
                scenario_id="switching_velocity_to_braking",
                scenario_family="switching_case",
                target_class="braking",
                difficulty_tier="stress_v1",
                seed=720 + replicate,
                duration=2.0,
                sample_period=0.5,
                initial_position=0.0,
                initial_velocity=1.35 + 0.04 * replicate,
                acceleration=0.0,
                measurement_std=0.04,
                switch_time=0.9 + 0.1 * replicate,
                acceleration_after_switch=-0.65 - 0.05 * replicate,
                provenance={"search_method": "dashboard_seeded", "search_iteration": len(candidates)},
            )
        )
        candidates.append(
            BackendCandidateSpec(
                candidate_id=f"env_accel_{replicate}",
                scenario_id="environment_density_gradient",
                scenario_family="environment_regime_case",
                target_class="constant_acceleration",
                difficulty_tier="realistic_v1",
                seed=740 + replicate,
                duration=2.0,
                sample_period=0.5,
                initial_position=0.0,
                initial_velocity=0.82 + 0.03 * replicate,
                acceleration=0.40 + 0.04 * replicate,
                measurement_std=0.03,
                drag_coefficient=0.18 + 0.04 * replicate,
                density_scale=1.02 - 0.08 * replicate,
                wind_bias=0.03 + 0.04 * replicate,
                provenance={"search_method": "dashboard_seeded", "search_iteration": len(candidates), "environment_id": f"env_regime_{replicate}"},
            )
        )
        candidates.append(
            BackendCandidateSpec(
                candidate_id=f"file_maneuver_{replicate}",
                scenario_id="file_backend_case",
                scenario_family="file_backend_case",
                target_class="maneuver",
                difficulty_tier="adversarial_v1",
                seed=760 + replicate,
                duration=2.0,
                sample_period=0.5,
                initial_position=0.0,
                initial_velocity=0.85 + 0.05 * replicate,
                acceleration=0.24 + 0.04 * replicate,
                measurement_std=0.03,
                input_deck_hash=f"file_case_hash_{replicate}",
                longitudinal_command=(0.4, 0.5, 0.1, -0.3, -0.4),
                provenance={"search_method": "dashboard_seeded", "search_iteration": len(candidates)},
            )
        )
    return tuple(candidates)


def _backends_for_candidate(candidate: BackendCandidateSpec) -> tuple[str, ...]:
    if candidate.scenario_family == "shared_boundary_case":
        return ("parameter_only_1d", "environment_aware_1d", "mock_file_backend_1d")
    if candidate.scenario_family == "switching_case":
        return ("controlled_1d", "mock_file_backend_1d")
    if candidate.scenario_family == "environment_regime_case":
        return ("environment_aware_1d",)
    if candidate.scenario_family == "file_backend_case":
        return ("mock_file_backend_1d",)
    return ()


def _provenance_completeness(metadata: dict[str, Any]) -> float:
    required = ("adapter_family", "candidate_id", "search_provenance")
    present = sum(1 for key in required if key in metadata and metadata[key] not in ("", None, {}))
    return present / len(required)


def _score_run(
    candidate: BackendCandidateSpec,
    backend_id: str,
    run: Any,
    weights: GenericCorpusExplorationWeights,
) -> dict[str, Any]:
    success = bool(run.success)
    provenance_score = _provenance_completeness(run.metadata)
    truth_state = run.truth_state
    positions = truth_state.get("position", ())
    velocities = truth_state.get("velocity", ())
    accelerations = truth_state.get("acceleration", ())
    speed_range = max(velocities) - min(velocities) if velocities else 0.0
    position_range = max(positions) - min(positions) if positions else 0.0
    acceleration_range = max(accelerations) - min(accelerations) if accelerations else 0.0
    validity_score = 1.0 if success else 0.0
    boundary_score = 0.85 if candidate.scenario_family == "shared_boundary_case" else 0.35
    stress_score = 0.90 if candidate.scenario_family == "switching_case" else 0.70 if candidate.scenario_family == "file_backend_case" else 0.45
    environment_score = 0.85 if candidate.scenario_family == "environment_regime_case" else 0.30
    coverage_novelty_score = {
        "shared_boundary_case": 0.70,
        "switching_case": 0.92,
        "environment_regime_case": 0.88,
        "file_backend_case": 0.78,
    }.get(candidate.scenario_family, 0.50)
    utility = (
        weights.validity * validity_score
        + weights.coverage_novelty * coverage_novelty_score
        + weights.boundary * boundary_score
        + weights.stress * stress_score
        + weights.environment * environment_score
        + weights.provenance * provenance_score
    )
    cell_id = f"{backend_id}|{candidate.scenario_family}|{candidate.target_class}|{candidate.difficulty_tier}"
    return {
        "candidate_id": candidate.candidate_id,
        "backend_id": backend_id,
        "trajectory_id": run.run_id,
        "scenario_family": candidate.scenario_family,
        "target_class": candidate.target_class,
        "difficulty_tier": candidate.difficulty_tier,
        "environment_id": candidate.provenance.get("environment_id", ""),
        "success": success,
        "validity_score": validity_score,
        "coverage_novelty_score": coverage_novelty_score,
        "boundary_score": boundary_score,
        "classifier_stress_score": stress_score,
        "environment_score": environment_score,
        "provenance_completeness": provenance_score,
        "total_utility": utility,
        "utility_weights": _weights_to_dict(weights),
        "position_range": position_range,
        "speed_range": speed_range,
        "acceleration_range": acceleration_range,
        "num_samples": len(run.times),
        "cell_id": cell_id,
    }


def _archive_rows(candidate_rows: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    best_by_cell: dict[str, dict[str, Any]] = {}
    for row in candidate_rows:
        cell_id = str(row["cell_id"])
        if cell_id not in best_by_cell or float(row["total_utility"]) > float(best_by_cell[cell_id]["total_utility"]):
            best_by_cell[cell_id] = dict(row)
    rows = []
    for cell_id, row in sorted(best_by_cell.items()):
        rows.append(
            {
                "cell_id": cell_id,
                "backend_id": row["backend_id"],
                "scenario_family": row["scenario_family"],
                "target_class": row["target_class"],
                "difficulty_tier": row["difficulty_tier"],
                "elite_candidate_id": row["candidate_id"],
                "elite_total_utility": row["total_utility"],
            }
        )
    return tuple(rows)


def _selected_manifest_rows(candidate_rows: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    best_by_cell: dict[str, dict[str, Any]] = {}
    for row in candidate_rows:
        if not bool(row["success"]):
            continue
        cell_id = str(row["cell_id"])
        if cell_id not in best_by_cell or float(row["total_utility"]) > float(best_by_cell[cell_id]["total_utility"]):
            best_by_cell[cell_id] = dict(row)
    rows = sorted(best_by_cell.values(), key=lambda row: (-float(row["total_utility"]), str(row["backend_id"])))
    return tuple(rows[:6])


def _backend_comparison_rows(candidate_rows: tuple[dict[str, Any], ...], selected_rows: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    backend_ids = sorted({str(row["backend_id"]) for row in candidate_rows})
    for backend_id in backend_ids:
        backend_rows = [row for row in candidate_rows if row["backend_id"] == backend_id]
        selected_backend_rows = [row for row in selected_rows if row["backend_id"] == backend_id]
        rows.append(
            {
                "backend_id": backend_id,
                "candidate_count": len(backend_rows),
                "selected_count": len(selected_backend_rows),
                "success_rate": sum(1 for row in backend_rows if bool(row["success"])) / max(len(backend_rows), 1),
                "mean_total_utility": sum(float(row["total_utility"]) for row in backend_rows) / max(len(backend_rows), 1),
                "mean_provenance_completeness": sum(float(row["provenance_completeness"]) for row in backend_rows) / max(len(backend_rows), 1),
            }
        )
    return tuple(rows)


def _render_backend_coverage_png(rows: tuple[dict[str, Any], ...]) -> bytes:
    backend_ids = [str(row["backend_id"]) for row in rows]
    selected_counts = [int(row["selected_count"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.bar(backend_ids, selected_counts, color="#4d8f77")
    ax.set_ylabel("Selected Count")
    ax.set_title("Backend Coverage Comparison")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    return _figure_to_png(fig)


def _render_archive_heatmap_png(rows: tuple[dict[str, Any], ...]) -> bytes:
    backends = sorted({str(row["backend_id"]) for row in rows})
    scenarios = sorted({str(row["scenario_family"]) for row in rows})
    matrix = []
    for backend in backends:
        backend_row = []
        for scenario in scenarios:
            backend_row.append(sum(1 for row in rows if row["backend_id"] == backend and row["scenario_family"] == scenario))
        matrix.append(backend_row)
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    image = ax.imshow(matrix, cmap="Purples", aspect="auto")
    ax.set_xticks(range(len(scenarios)), labels=scenarios, fontsize=8)
    ax.set_yticks(range(len(backends)), labels=backends, fontsize=8)
    ax.set_title("Archive Coverage Heatmap")
    for row_index, values in enumerate(matrix):
        for column_index, value in enumerate(values):
            ax.text(column_index, row_index, f"{value}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.04)
    fig.tight_layout()
    return _figure_to_png(fig)


def _render_parallel_png(rows: tuple[dict[str, Any], ...]) -> bytes:
    metrics = (
        "validity_score",
        "coverage_novelty_score",
        "boundary_score",
        "classifier_stress_score",
        "environment_score",
        "provenance_completeness",
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for row in rows:
        values = [float(row[metric]) for metric in metrics]
        ax.plot(range(len(metrics)), values, marker="o", alpha=0.7, label=str(row["candidate_id"]))
    ax.set_xticks(range(len(metrics)), labels=[metric.replace("_", "\n") for metric in metrics], fontsize=8)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Normalized Score")
    ax.set_title("Score Component Parallel Coordinates")
    fig.tight_layout()
    return _figure_to_png(fig)


def _render_selected_gallery_png(selected_rows: tuple[dict[str, Any,]], candidate_pool: tuple[BackendCandidateSpec, ...]) -> bytes:
    adapters = _adapter_map()
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.0), sharex=False)
    for axis, row in zip(axes.flat, selected_rows[:4]):
        candidate = next(candidate for candidate in candidate_pool if candidate.candidate_id == row["candidate_id"])
        run = adapters[str(row["backend_id"])].run(candidate).trajectory_run
        axis.plot(run.times, run.truth_state.get("position", ()), marker="o", label="position")
        if "velocity" in run.truth_state:
            axis.plot(run.times, run.truth_state["velocity"], marker="s", label="velocity")
        axis.set_title(f"{row['backend_id']}\n{row['scenario_family']}", fontsize=9)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Selected Trajectory Gallery", fontsize=11)
    fig.tight_layout()
    return _figure_to_png(fig)


def _render_provenance_dashboard_png(rows: tuple[dict[str, Any], ...]) -> bytes:
    labels = [str(row["candidate_id"]) for row in rows]
    values = [float(row["provenance_completeness"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.bar(labels, values, color="#3e6a8a")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Completeness")
    ax.set_title("Provenance Completeness Dashboard")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return _figure_to_png(fig)


def analyze_generic_corpus_exploration(
    *,
    seed: int = 7,
    weights: GenericCorpusExplorationWeights | None = None,
) -> GenericCorpusExplorationResult:
    rng = random.Random(seed)
    adapters = _adapter_map()
    candidate_pool = _candidate_pool()
    utility_weights = _normalize_weights(weights or DEFAULT_GENERIC_CORPUS_EXPLORATION_WEIGHTS)
    candidate_rows: list[dict[str, Any]] = []
    for candidate in candidate_pool:
        for backend_id in _backends_for_candidate(candidate):
            run = adapters[backend_id].run(candidate).trajectory_run
            candidate_rows.append(_score_run(candidate, backend_id, run, utility_weights))

    candidate_rows.sort(key=lambda row: (-float(row["total_utility"]), str(row["backend_id"]), str(row["candidate_id"])))
    archive_rows = _archive_rows(tuple(candidate_rows))
    selected_rows = _selected_manifest_rows(tuple(candidate_rows))

    baseline_size = len(selected_rows)
    random_baseline = tuple(rng.sample(candidate_rows, baseline_size))
    selected_coverage = len({row["cell_id"] for row in selected_rows})
    random_coverage = len({row["cell_id"] for row in random_baseline})
    selected_backend_count = len({row["backend_id"] for row in selected_rows})
    selected_scenario_count = len({row["scenario_family"] for row in selected_rows})

    exploration_manifest = {
        "exploration_version": "m35_v1",
        "utility_weights": _weights_to_dict(utility_weights),
        "candidate_pool_size": len(candidate_rows),
        "archive_cell_count": len(archive_rows),
        "selected_corpus_size": len(selected_rows),
        "random_baseline_size": baseline_size,
        "selected_coverage": selected_coverage,
        "random_baseline_coverage": random_coverage,
        "coverage_improves_over_random": selected_coverage > random_coverage,
    }
    selected_corpus_manifest = {
        "utility_weights": _weights_to_dict(utility_weights),
        "selected_rows": list(selected_rows),
        "coverage_improves_over_random": selected_coverage > random_coverage,
        "selected_backend_count": selected_backend_count,
        "selected_scenario_count": selected_scenario_count,
        "includes_boundary_examples": any(row["scenario_family"] == "shared_boundary_case" for row in selected_rows),
        "includes_stress_examples": any(row["scenario_family"] == "switching_case" for row in selected_rows),
    }
    backend_comparison_rows = _backend_comparison_rows(tuple(candidate_rows), selected_rows)

    doc = MarkdownDocument("Generic Corpus Exploration Dashboard")
    doc.heading("Summary", level=2)
    doc.bullet_list(
        [
            f"candidate rows explored: `{len(candidate_rows)}`",
            f"archive cells filled: `{len(archive_rows)}`",
            f"selected corpus size: `{len(selected_rows)}`",
            f"selected coverage: `{selected_coverage}`",
            f"random baseline coverage: `{random_coverage}`",
            f"selected backends represented: `{selected_backend_count}`",
            f"selected scenario families represented: `{selected_scenario_count}`",
            f"utility weights: `{_weights_to_dict(utility_weights)}`",
        ]
    )
    
    doc.heading("Selected Corpus", level=2)
    doc.table(
        ["Candidate", "Backend", "Scenario", "Utility", "Provenance"],
        [
            (
                f"`{row['candidate_id']}`",
                f"`{row['backend_id']}`",
                f"`{row['scenario_family']}`",
                f"`{row['total_utility']:.3f}`",
                f"`{row['provenance_completeness']:.2f}`",
            )
            for row in selected_rows
        ]
    )

    doc.heading("Acceptance Notes", level=2)
    doc.bullet_list(
        [
            f"coverage improves over random baseline: `{selected_coverage > random_coverage}`",
            f"includes at least two backend types: `{selected_backend_count >= 2}`",
            f"includes boundary examples: `{selected_corpus_manifest['includes_boundary_examples']}`",
            f"includes stress examples: `{selected_corpus_manifest['includes_stress_examples']}`",
            "All selected rows retain backend id, scenario family, candidate id, and provenance completeness scores.",
        ]
    )

    return GenericCorpusExplorationResult(
        exploration_manifest=exploration_manifest,
        candidate_score_rows=tuple(candidate_rows),
        archive_cell_rows=archive_rows,
        selected_corpus_manifest=selected_corpus_manifest,
        backend_comparison_rows=backend_comparison_rows,
        report_markdown=doc.text(),
    )


def render_generic_corpus_exploration_numeric_walkthrough_markdown(
    result: GenericCorpusExplorationResult | None = None,
) -> str:
    payload = result or analyze_generic_corpus_exploration()
    weight_values = payload.exploration_manifest.get("utility_weights", _weights_to_dict(DEFAULT_GENERIC_CORPUS_EXPLORATION_WEIGHTS))
    selected_rows = tuple(payload.selected_corpus_manifest["selected_rows"])
    selected_row = selected_rows[0]
    archive_row = next(
        row for row in payload.archive_cell_rows
        if row.get("archive_status") != "failed"
        and str(row.get("elite_candidate_id", "")) == str(selected_row["candidate_id"])
        and str(row.get("backend_id", "")) == str(selected_row["backend_id"])
    )
    selected_coverage = int(payload.exploration_manifest["selected_coverage"])
    random_coverage = int(payload.exploration_manifest["random_baseline_coverage"])
    coverage_delta = selected_coverage - random_coverage

    lines = [
        "# Generic Corpus Explorer Numeric Walkthrough",
        "",
        "This walkthrough expands one selected corpus row from",
        "`generic_corpus_exploration.py` into its utility components, archive",
        "cell interpretation, and selected-versus-random coverage comparison.",
        "",
        "## Selected Candidate",
        "",
        f"- `candidate_id`: `{selected_row['candidate_id']}`",
        f"- `backend_id`: `{selected_row['backend_id']}`",
        f"- `scenario_family`: `{selected_row['scenario_family']}`",
        f"- `target_class`: `{selected_row['target_class']}`",
        f"- `difficulty_tier`: `{selected_row['difficulty_tier']}`",
        f"- `cell_id`: `{selected_row['cell_id']}`",
        "",
        "## Implemented Explorer Utility",
        "",
        "```tex",
        "U_{\\text{explore}}",
        f"= {float(weight_values['validity']):.2f} \\cdot \\text{{validity}}",
        f"+ {float(weight_values['coverage_novelty']):.2f} \\cdot \\text{{coverage novelty}}",
        f"+ {float(weight_values['boundary']):.2f} \\cdot \\text{{boundary score}}",
        f"+ {float(weight_values['stress']):.2f} \\cdot \\text{{classifier stress}}",
        f"+ {float(weight_values['environment']):.2f} \\cdot \\text{{environment score}}",
        f"+ {float(weight_values['provenance']):.2f} \\cdot \\text{{provenance completeness}}.",
        "```",
        "",
        "## Numeric Substitution",
        "",
        "```tex",
        "U_{\\text{explore}}",
        f"= {float(weight_values['validity']):.2f} \\cdot {float(selected_row['validity_score']):.3f}",
        f"+ {float(weight_values['coverage_novelty']):.2f} \\cdot {float(selected_row['coverage_novelty_score']):.3f}",
        f"+ {float(weight_values['boundary']):.2f} \\cdot {float(selected_row['boundary_score']):.3f}",
        f"+ {float(weight_values['stress']):.2f} \\cdot {float(selected_row['classifier_stress_score']):.3f}",
        f"+ {float(weight_values['environment']):.2f} \\cdot {float(selected_row['environment_score']):.3f}",
        f"+ {float(weight_values['provenance']):.2f} \\cdot {float(selected_row['provenance_completeness']):.3f}",
        f"= {float(selected_row['total_utility']):.3f}.",
        "```",
        "",
        "## Utility Components",
        "",
        f"- `validity_score`: `{float(selected_row['validity_score']):.3f}`",
        f"- `coverage_novelty_score`: `{float(selected_row['coverage_novelty_score']):.3f}`",
        f"- `boundary_score`: `{float(selected_row['boundary_score']):.3f}`",
        f"- `classifier_stress_score`: `{float(selected_row['classifier_stress_score']):.3f}`",
        f"- `environment_score`: `{float(selected_row['environment_score']):.3f}`",
        f"- `provenance_completeness`: `{float(selected_row['provenance_completeness']):.3f}`",
        f"- `total_utility`: `{float(selected_row['total_utility']):.3f}`",
        "",
        "## Archive Cell Interpretation",
        "",
        f"- `archive cell`: `{archive_row['cell_id']}`",
        f"- `elite candidate`: `{archive_row['elite_candidate_id']}`",
        f"- `elite_total_utility`: `{float(archive_row['elite_total_utility']):.3f}`",
        "",
        "This row is selected because it is the elite for its backend/scenario/",
        "class/tier cell and therefore increases structured corpus coverage, not",
        "only scalar utility.",
        "",
        "## Coverage Comparison Against Random Baseline",
        "",
        "```tex",
        "\\Delta_{\\text{coverage}}",
        f"= {selected_coverage} - {random_coverage}",
        f"= {coverage_delta}.",
        "```",
        "",
        f"- `selected_coverage`: `{selected_coverage}`",
        f"- `random_baseline_coverage`: `{random_coverage}`",
        f"- `coverage_improves_over_random`: `{payload.exploration_manifest['coverage_improves_over_random']}`",
        "",
        "## Interpretation",
        "",
        "- The selected row is not just high-utility in isolation.",
        "- It is part of a selected set that covers more archive cells than a",
        "  same-size random baseline.",
        "- That is the core Explorer claim: corpus selection should improve both",
        "  utility and structural coverage.",
    ]
    return "\n".join(lines)


def _generic_corpus_exploration_weight_variants() -> tuple[GenericCorpusExplorationSweepVariant, ...]:
    if DEFAULT_GENERIC_CORPUS_EXPLORATION_WEIGHT_SWEEP_CONFIG_PATH.exists():
        return load_generic_corpus_exploration_weight_sweep_config(
            DEFAULT_GENERIC_CORPUS_EXPLORATION_WEIGHT_SWEEP_CONFIG_PATH
        ).variants
    baseline = _normalize_weights(DEFAULT_GENERIC_CORPUS_EXPLORATION_WEIGHTS)
    return (
        GenericCorpusExplorationSweepVariant(
            variant_id="baseline",
            description="Current production weights",
            weights=baseline,
        ),
        GenericCorpusExplorationSweepVariant(
            variant_id="validity_plus_5",
            description="Increase validity emphasis by 5 percentage points",
            weights=_perturb_weights(baseline, focus="validity", delta=0.05),
        ),
        GenericCorpusExplorationSweepVariant(
            variant_id="coverage_plus_5",
            description="Increase coverage-novelty emphasis by 5 percentage points",
            weights=_perturb_weights(baseline, focus="coverage_novelty", delta=0.05),
        ),
        GenericCorpusExplorationSweepVariant(
            variant_id="boundary_plus_5",
            description="Increase boundary emphasis by 5 percentage points",
            weights=_perturb_weights(baseline, focus="boundary", delta=0.05),
        ),
        GenericCorpusExplorationSweepVariant(
            variant_id="stress_plus_5",
            description="Increase stress emphasis by 5 percentage points",
            weights=_perturb_weights(baseline, focus="stress", delta=0.05),
        ),
        GenericCorpusExplorationSweepVariant(
            variant_id="environment_plus_5",
            description="Increase environment emphasis by 5 percentage points",
            weights=_perturb_weights(baseline, focus="environment", delta=0.05),
        ),
        GenericCorpusExplorationSweepVariant(
            variant_id="provenance_plus_5",
            description="Increase provenance emphasis by 5 percentage points",
            weights=_perturb_weights(baseline, focus="provenance", delta=0.05),
        ),
    )


def _resolve_generic_corpus_exploration_weight_sweep_config(
    *,
    config: GenericCorpusExplorationSweepConfig | None = None,
    config_path: str | Path | None = None,
    variants: tuple[GenericCorpusExplorationSweepVariant, ...] | None = None,
) -> GenericCorpusExplorationSweepConfig:
    if config is not None:
        return config
    if config_path is not None:
        return load_generic_corpus_exploration_weight_sweep_config(config_path)
    if variants is not None:
        if not variants:
            raise ValueError("at least one weight variant is required")
        return GenericCorpusExplorationSweepConfig(
            baseline_variant_id=variants[0].variant_id,
            variants=variants,
            config_path=None,
        )
    if DEFAULT_GENERIC_CORPUS_EXPLORATION_WEIGHT_SWEEP_CONFIG_PATH.exists():
        return load_generic_corpus_exploration_weight_sweep_config(
            DEFAULT_GENERIC_CORPUS_EXPLORATION_WEIGHT_SWEEP_CONFIG_PATH
        )
    return GenericCorpusExplorationSweepConfig(
        baseline_variant_id="baseline",
        variants=_generic_corpus_exploration_weight_variants(),
        config_path=None,
    )


def _exploration_result_set(result: GenericCorpusExplorationResult) -> tuple[set[str], set[str]]:
    selected_rows = tuple(result.selected_corpus_manifest["selected_rows"])
    candidate_ids = {str(row["candidate_id"]) for row in selected_rows}
    cell_ids = {str(row["cell_id"]) for row in selected_rows}
    return candidate_ids, cell_ids


def analyze_generic_corpus_exploration_weight_sweep(
    *,
    seed: int = 7,
    config_path: str | Path | None = None,
    config: GenericCorpusExplorationSweepConfig | None = None,
    variants: tuple[GenericCorpusExplorationSweepVariant, ...] | None = None,
) -> GenericCorpusExplorationSweepResult:
    sweep_config = _resolve_generic_corpus_exploration_weight_sweep_config(
        config=config,
        config_path=config_path,
        variants=variants,
    )
    sweep_variants = sweep_config.variants
    baseline_variant = next(
        (variant for variant in sweep_variants if variant.variant_id == sweep_config.baseline_variant_id),
        sweep_variants[0],
    )
    baseline_weights = _normalize_weights(baseline_variant.weights)
    baseline_result = analyze_generic_corpus_exploration(seed=seed, weights=baseline_weights)
    baseline_candidate_ids, baseline_cell_ids = _exploration_result_set(baseline_result)
    baseline_selected_coverage = int(baseline_result.exploration_manifest["selected_coverage"])
    baseline_selected_rows = tuple(baseline_result.selected_corpus_manifest["selected_rows"])
    baseline_mean_total_utility = sum(float(row["total_utility"]) for row in baseline_selected_rows) / max(len(baseline_selected_rows), 1)

    rows: list[GenericCorpusExplorationSweepRow] = []
    for variant in sweep_variants:
        normalized_weights = _normalize_weights(variant.weights)
        result = baseline_result if variant.variant_id == baseline_variant.variant_id else analyze_generic_corpus_exploration(seed=seed, weights=normalized_weights)
        candidate_ids, cell_ids = _exploration_result_set(result)
        selected_rows = tuple(result.selected_corpus_manifest["selected_rows"])
        mean_total_utility = sum(float(row["total_utility"]) for row in selected_rows) / max(len(selected_rows), 1)
        rows.append(
            GenericCorpusExplorationSweepRow(
                variant_id=variant.variant_id,
                description=variant.description,
                weight_validity=float(normalized_weights.validity),
                weight_coverage_novelty=float(normalized_weights.coverage_novelty),
                weight_boundary=float(normalized_weights.boundary),
                weight_stress=float(normalized_weights.stress),
                weight_environment=float(normalized_weights.environment),
                weight_provenance=float(normalized_weights.provenance),
                selected_coverage=int(result.exploration_manifest["selected_coverage"]),
                random_baseline_coverage=int(result.exploration_manifest["random_baseline_coverage"]),
                coverage_delta_vs_random=int(result.exploration_manifest["selected_coverage"]) - int(result.exploration_manifest["random_baseline_coverage"]),
                coverage_delta_vs_baseline=int(result.exploration_manifest["selected_coverage"]) - baseline_selected_coverage,
                selected_backend_count=int(result.selected_corpus_manifest["selected_backend_count"]),
                selected_scenario_count=int(result.selected_corpus_manifest["selected_scenario_count"]),
                selected_candidate_count=len(selected_rows),
                selected_cell_count=len(cell_ids),
                mean_total_utility=mean_total_utility,
                mean_total_utility_delta_vs_baseline=mean_total_utility - baseline_mean_total_utility,
                mean_provenance_completeness=sum(float(row["provenance_completeness"]) for row in selected_rows) / max(len(selected_rows), 1),
                candidate_jaccard_vs_baseline=_set_jaccard(candidate_ids, baseline_candidate_ids),
                cell_jaccard_vs_baseline=_set_jaccard(cell_ids, baseline_cell_ids),
                selected_candidate_ids=tuple(sorted(candidate_ids)),
                selected_cell_ids=tuple(sorted(cell_ids)),
            )
        )

    doc = MarkdownDocument("Generic Corpus Exploration Weight Sweep")
    doc.paragraph(
        "This sweep keeps the underlying candidate pool fixed and perturbs the corpus-explorer utility weights to test whether selection is stable or brittle."
    )
    
    doc.heading("Baseline", level=2)
    doc.bullet_list(
        [
            f"baseline variant: `{baseline_variant.variant_id}`",
            f"baseline selected coverage: `{baseline_selected_coverage}`",
            f"baseline weights: `{_weights_to_dict(baseline_weights)}`",
            f"sweep config source: `{sweep_config.config_path.name if sweep_config.config_path else 'generic_corpus_exploration_weight_sweep.yaml'}`",
        ]
    )

    doc.heading("Variant Comparison", level=2)
    doc.table(
        ["Variant", "Description", "Selected Coverage", "Random Coverage", "Delta vs Random", "Delta vs Baseline", "Candidate Jaccard vs Baseline", "Cell Jaccard vs Baseline", "Mean Utility", "Mean Utility Delta"],
        [
            (
                f"`{row.variant_id}`",
                f"`{row.description}`",
                f"`{row.selected_coverage}`",
                f"`{row.random_baseline_coverage}`",
                f"`{row.coverage_delta_vs_random}`",
                f"`{row.coverage_delta_vs_baseline}`",
                f"`{row.candidate_jaccard_vs_baseline:.2f}`",
                f"`{row.cell_jaccard_vs_baseline:.2f}`",
                f"`{row.mean_total_utility:.3f}`",
                f"`{row.mean_total_utility_delta_vs_baseline:.3f}`",
            )
            for row in rows
        ]
    )

    doc.heading("Interpretation", level=2)
    doc.bullet_list(
        [
            "If nearby weight perturbations keep the selected corpus stable, the heuristic is robust.",
            "If small perturbations reorder the selected cells heavily, the utility is acting like a fragile hand-tuned score and should be revisited.",
            "The baseline row is included so the same analysis path can be used for direct before/after comparison.",
        ]
    )

    return GenericCorpusExplorationSweepResult(
        baseline_variant_id=baseline_variant.variant_id,
        variants=sweep_variants,
        rows=tuple(rows),
        report_markdown=doc.text(),
    )


def render_generic_corpus_exploration_weight_sweep_markdown(
    result: GenericCorpusExplorationSweepResult | None = None,
) -> str:
    payload = result or analyze_generic_corpus_exploration_weight_sweep()
    return payload.report_markdown


def _render_weight_sweep_tradeoff_png(rows: tuple[GenericCorpusExplorationSweepRow, ...]) -> bytes:
    variant_ids = [row.variant_id for row in rows]
    coverage_delta = [row.coverage_delta_vs_baseline for row in rows]
    candidate_jaccard = [row.candidate_jaccard_vs_baseline for row in rows]
    cell_jaccard = [row.cell_jaccard_vs_baseline for row in rows]
    fig, ax1 = plt.subplots(figsize=(10.0, 4.8))
    ax1.bar(variant_ids, coverage_delta, color="#4d8f77", alpha=0.7, label="Coverage Delta vs Baseline")
    ax1.set_ylabel("Coverage Delta vs Baseline")
    ax1.tick_params(axis="x", rotation=20)
    ax1.grid(axis="y", alpha=0.2)
    ax2 = ax1.twinx()
    ax2.plot(variant_ids, candidate_jaccard, color="#7c3aed", marker="o", linewidth=2.0, label="Candidate Jaccard")
    ax2.plot(variant_ids, cell_jaccard, color="#0f766e", marker="s", linewidth=2.0, label="Cell Jaccard")
    ax2.set_ylabel("Jaccard vs Baseline")
    ax2.set_ylim(0.0, 1.05)
    fig.suptitle("Explorer Weight Sweep Tradeoff", fontsize=11)
    fig.tight_layout()
    return _figure_to_png(fig)


def _render_weight_sweep_overlap_png(rows: tuple[GenericCorpusExplorationSweepRow, ...]) -> bytes:
    variant_ids = [row.variant_id for row in rows]
    candidate_sets = [set(row.selected_candidate_ids) for row in rows]
    cell_sets = [set(row.selected_cell_ids) for row in rows]
    candidate_matrix = [
        [_set_jaccard(left, right) for right in candidate_sets]
        for left in candidate_sets
    ]
    cell_matrix = [
        [_set_jaccard(left, right) for right in cell_sets]
        for left in cell_sets
    ]
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.2), constrained_layout=True)
    for axis, matrix, title in zip(
        axes,
        (candidate_matrix, cell_matrix),
        ("Candidate Overlap Heatmap", "Cell Overlap Heatmap"),
    ):
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
        axis.set_xticks(range(len(variant_ids)), labels=variant_ids, rotation=20, fontsize=7)
        axis.set_yticks(range(len(variant_ids)), labels=variant_ids, fontsize=7)
        axis.set_title(title)
    for row_index, values in enumerate(matrix):
        for column_index, value in enumerate(values):
            axis.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=6, color="white" if value < 0.6 else "black")
    fig.colorbar(image, ax=axis, fraction=0.04, pad=0.02)
    return _figure_to_png(fig)


def _render_weight_sweep_weight_matrix_png(rows: tuple[GenericCorpusExplorationSweepRow, ...]) -> bytes:
    variant_ids = [row.variant_id for row in rows]
    matrix = [
        [
            row.weight_validity,
            row.weight_coverage_novelty,
            row.weight_boundary,
            row.weight_stress,
            row.weight_environment,
            row.weight_provenance,
        ]
        for row in rows
    ]
    labels = ["validity", "coverage", "boundary", "stress", "environment", "provenance"]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=20, fontsize=8)
    ax.set_yticks(range(len(variant_ids)), labels=variant_ids, fontsize=8)
    ax.set_title("Weight Matrix")
    for row_index, values in enumerate(matrix):
        for column_index, value in enumerate(values):
            ax.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.04)
    fig.tight_layout()
    return _figure_to_png(fig)


def write_generic_corpus_exploration_weight_sweep_artifacts(
    base_dir: str | Path,
    *,
    result: GenericCorpusExplorationSweepResult | None = None,
    seed: int = 7,
    config_path: str | Path | None = None,
    config: GenericCorpusExplorationSweepConfig | None = None,
    variants: tuple[GenericCorpusExplorationSweepVariant, ...] | None = None,
) -> GenericCorpusExplorationSweepArtifacts:
    run_dir = Path(base_dir) / "generic_corpus_exploration_weight_sweep_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    if result is not None and config is None and config_path is None and variants is None:
        sweep_config = GenericCorpusExplorationSweepConfig(
            baseline_variant_id=result.baseline_variant_id,
            variants=result.variants,
            config_path=None,
        )
    else:
        sweep_config = _resolve_generic_corpus_exploration_weight_sweep_config(
            config=config,
            config_path=config_path,
            variants=variants,
        )
    payload = result or analyze_generic_corpus_exploration_weight_sweep(seed=seed, config=sweep_config)

    config_output_path = run_dir / "generic_corpus_exploration_weight_sweep_config.yaml"
    report_path = run_dir / "generic_corpus_exploration_weight_sweep_report.md"
    summary_path = run_dir / "generic_corpus_exploration_weight_sweep_summary.json"
    rows_path = run_dir / "generic_corpus_exploration_weight_sweep.csv"
    overlap_matrix_path = run_dir / "generic_corpus_exploration_weight_sweep_overlap_matrix.csv"
    weight_matrix_path = run_dir / "generic_corpus_exploration_weight_sweep_weight_matrix.csv"
    tradeoff_png_path = run_dir / "generic_corpus_exploration_weight_sweep_tradeoff.png"
    selected_set_png_path = run_dir / "generic_corpus_exploration_weight_sweep_overlap_heatmap.png"
    baseline_manifest_path = run_dir / "generic_corpus_exploration_weight_sweep_baseline_manifest.json"

    rows = list(payload.rows)
    config_output_path.write_text(
        yaml.safe_dump(_sweep_config_to_dict(sweep_config), sort_keys=False),
        encoding="utf-8",
    )
    report_path.write_text(payload.report_markdown, encoding="utf-8")
    summary = {
        "baseline_variant_id": payload.baseline_variant_id,
        "variant_count": len(payload.variants),
        "row_count": len(rows),
        "baseline_weights": _weights_to_dict(next(variant.weights for variant in payload.variants if variant.variant_id == payload.baseline_variant_id)),
        "config_path": str(config_output_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    baseline_row = next(row for row in rows if row.variant_id == payload.baseline_variant_id)
    baseline_manifest_path.write_text(
        json.dumps(
            {
                "variant_id": baseline_row.variant_id,
                "description": baseline_row.description,
                "selected_candidate_ids": list(baseline_row.selected_candidate_ids),
                "selected_cell_ids": list(baseline_row.selected_cell_ids),
                "selected_coverage": baseline_row.selected_coverage,
                "coverage_delta_vs_random": baseline_row.coverage_delta_vs_random,
                "coverage_delta_vs_baseline": baseline_row.coverage_delta_vs_baseline,
                "weights": {
                    "validity": baseline_row.weight_validity,
                    "coverage_novelty": baseline_row.weight_coverage_novelty,
                    "boundary": baseline_row.weight_boundary,
                    "stress": baseline_row.weight_stress,
                    "environment": baseline_row.weight_environment,
                    "provenance": baseline_row.weight_provenance,
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    write_csv(
        rows_path,
        [
            {
                "variant_id": row.variant_id,
                "description": row.description,
                "weight_validity": row.weight_validity,
                "weight_coverage_novelty": row.weight_coverage_novelty,
                "weight_boundary": row.weight_boundary,
                "weight_stress": row.weight_stress,
                "weight_environment": row.weight_environment,
                "weight_provenance": row.weight_provenance,
                "selected_coverage": row.selected_coverage,
                "random_baseline_coverage": row.random_baseline_coverage,
                "coverage_delta_vs_random": row.coverage_delta_vs_random,
                "coverage_delta_vs_baseline": row.coverage_delta_vs_baseline,
                "selected_backend_count": row.selected_backend_count,
                "selected_scenario_count": row.selected_scenario_count,
                "selected_candidate_count": row.selected_candidate_count,
                "selected_cell_count": row.selected_cell_count,
                "mean_total_utility": row.mean_total_utility,
                "mean_total_utility_delta_vs_baseline": row.mean_total_utility_delta_vs_baseline,
                "mean_provenance_completeness": row.mean_provenance_completeness,
                "candidate_jaccard_vs_baseline": row.candidate_jaccard_vs_baseline,
                "cell_jaccard_vs_baseline": row.cell_jaccard_vs_baseline,
                "selected_candidate_ids": " | ".join(row.selected_candidate_ids),
                "selected_cell_ids": " | ".join(row.selected_cell_ids),
            }
            for row in rows
        ],
        [
            "variant_id",
            "description",
            "weight_validity",
            "weight_coverage_novelty",
            "weight_boundary",
            "weight_stress",
            "weight_environment",
            "weight_provenance",
            "selected_coverage",
            "random_baseline_coverage",
            "coverage_delta_vs_random",
            "coverage_delta_vs_baseline",
            "selected_backend_count",
            "selected_scenario_count",
            "selected_candidate_count",
            "selected_cell_count",
            "mean_total_utility",
            "mean_total_utility_delta_vs_baseline",
            "mean_provenance_completeness",
            "candidate_jaccard_vs_baseline",
            "cell_jaccard_vs_baseline",
            "selected_candidate_ids",
            "selected_cell_ids",
        ],
    )

    overlap_rows = []
    for left in rows:
        left_candidates = set(left.selected_candidate_ids)
        left_cells = set(left.selected_cell_ids)
        for right in rows:
            overlap_rows.append(
                {
                    "left_variant_id": left.variant_id,
                    "right_variant_id": right.variant_id,
                    "candidate_jaccard": _set_jaccard(left_candidates, set(right.selected_candidate_ids)),
                    "cell_jaccard": _set_jaccard(left_cells, set(right.selected_cell_ids)),
                }
            )
    write_csv(
        overlap_matrix_path,
        overlap_rows,
        ["left_variant_id", "right_variant_id", "candidate_jaccard", "cell_jaccard"],
    )

    weight_rows = [
        {
            "variant_id": row.variant_id,
            "validity": row.weight_validity,
            "coverage_novelty": row.weight_coverage_novelty,
            "boundary": row.weight_boundary,
            "stress": row.weight_stress,
            "environment": row.weight_environment,
            "provenance": row.weight_provenance,
        }
        for row in rows
    ]
    write_csv(
        weight_matrix_path,
        weight_rows,
        ["variant_id", "validity", "coverage_novelty", "boundary", "stress", "environment", "provenance"],
    )

    tradeoff_png_path.write_bytes(_render_weight_sweep_tradeoff_png(tuple(rows)))
    selected_set_png_path.write_bytes(_render_weight_sweep_overlap_png(tuple(rows)))

    return GenericCorpusExplorationSweepArtifacts(
        run_dir=run_dir,
        config_path=config_output_path,
        report_path=report_path,
        summary_path=summary_path,
        rows_path=rows_path,
        overlap_matrix_path=overlap_matrix_path,
        weight_matrix_path=weight_matrix_path,
        tradeoff_png_path=tradeoff_png_path,
        selected_set_png_path=selected_set_png_path,
        baseline_manifest_path=baseline_manifest_path,
    )


def write_generic_corpus_exploration_artifacts(
    base_dir: str | Path,
    *,
    result: GenericCorpusExplorationResult | None = None,
) -> GenericCorpusExplorationArtifacts:
    run_dir = Path(base_dir) / "generic_corpus_exploration"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = result or analyze_generic_corpus_exploration()

    exploration_manifest_path = run_dir / "exploration_manifest.json"
    candidate_scores_path = run_dir / "candidate_scores.csv"
    archive_cells_path = run_dir / "archive_cells.csv"
    selected_corpus_manifest_path = run_dir / "selected_corpus_manifest.json"
    backend_comparison_path = run_dir / "backend_comparison.csv"
    report_path = run_dir / "corpus_exploration_report.md"
    numeric_walkthrough_path = run_dir / "generic_corpus_explorer_numeric_walkthrough.md"
    backend_coverage_png_path = run_dir / "backend_coverage_comparison.png"
    archive_heatmap_png_path = run_dir / "archive_coverage_heatmap.png"
    score_parallel_png_path = run_dir / "score_component_parallel_coordinates.png"
    selected_gallery_png_path = run_dir / "selected_trajectory_gallery.png"
    provenance_dashboard_png_path = run_dir / "provenance_completeness_dashboard.png"

    exploration_manifest_path.write_text(json.dumps(payload.exploration_manifest, indent=2), encoding="utf-8")
    selected_corpus_manifest_path.write_text(json.dumps(payload.selected_corpus_manifest, indent=2), encoding="utf-8")
    report_path.write_text(payload.report_markdown, encoding="utf-8")
    numeric_walkthrough_path.write_text(render_generic_corpus_exploration_numeric_walkthrough_markdown(payload), encoding="utf-8")

    candidate_fieldnames = list(payload.candidate_score_rows[0].keys()) if payload.candidate_score_rows else []
    archive_fieldnames = list(payload.archive_cell_rows[0].keys()) if payload.archive_cell_rows else []
    backend_fieldnames = list(payload.backend_comparison_rows[0].keys()) if payload.backend_comparison_rows else []
    write_csv(candidate_scores_path, list(payload.candidate_score_rows), candidate_fieldnames)
    write_csv(archive_cells_path, list(payload.archive_cell_rows), archive_fieldnames)
    write_csv(backend_comparison_path, list(payload.backend_comparison_rows), backend_fieldnames)

    candidate_pool = _candidate_pool()
    backend_coverage_png_path.write_bytes(_render_backend_coverage_png(payload.backend_comparison_rows))
    archive_heatmap_png_path.write_bytes(_render_archive_heatmap_png(payload.archive_cell_rows))
    selected_rows = tuple(payload.selected_corpus_manifest["selected_rows"])
    score_parallel_png_path.write_bytes(_render_parallel_png(selected_rows))
    selected_gallery_png_path.write_bytes(_render_selected_gallery_png(selected_rows, candidate_pool))
    provenance_dashboard_png_path.write_bytes(_render_provenance_dashboard_png(selected_rows))

    return GenericCorpusExplorationArtifacts(
        run_dir=run_dir,
        exploration_manifest_path=exploration_manifest_path,
        candidate_scores_path=candidate_scores_path,
        archive_cells_path=archive_cells_path,
        selected_corpus_manifest_path=selected_corpus_manifest_path,
        backend_comparison_path=backend_comparison_path,
        report_path=report_path,
        numeric_walkthrough_path=numeric_walkthrough_path,
        backend_coverage_png_path=backend_coverage_png_path,
        archive_heatmap_png_path=archive_heatmap_png_path,
        score_parallel_png_path=score_parallel_png_path,
        selected_gallery_png_path=selected_gallery_png_path,
        provenance_dashboard_png_path=provenance_dashboard_png_path,
    )
