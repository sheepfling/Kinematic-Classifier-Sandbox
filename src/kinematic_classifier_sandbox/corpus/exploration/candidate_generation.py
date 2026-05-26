from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import random
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from .backend_adapter_proof import BackendCandidateSpec
from .capability_aware_search import analyze_capability_aware_search
from .corpus_objectives import CorpusObjectiveSpec, default_corpus_objectives, load_corpus_objectives_from_yaml


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@dataclass(frozen=True, slots=True)
class CandidateGenerationResult:
    sampler_manifest: dict[str, Any]
    generated_candidate_rows: tuple[dict[str, Any], ...]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class CandidateGenerationArtifacts:
    run_dir: Path
    sampler_manifest_path: Path
    generated_candidates_path: Path
    report_path: Path
    sampler_comparison_png_path: Path
    candidate_coverage_png_path: Path
    mutation_lineage_png_path: Path


def _backend_constraints_for_objective(objective: CorpusObjectiveSpec) -> tuple[str, ...]:
    if objective.backend_constraints:
        return objective.backend_constraints
    return ("parameter_only_1d",)


def _base_class_parameters(class_name: str) -> dict[str, float]:
    defaults = {
        "constant_velocity": {"initial_velocity": 1.0, "acceleration": 0.05},
        "constant_acceleration": {"initial_velocity": 0.8, "acceleration": 0.40},
        "braking": {"initial_velocity": 1.35, "acceleration": 0.0},
        "maneuver": {"initial_velocity": 0.9, "acceleration": 0.28},
    }
    return defaults.get(class_name, {"initial_velocity": 1.0, "acceleration": 0.10})


def _objective_primary_class(objective: CorpusObjectiveSpec) -> str:
    if objective.target_class is not None:
        return objective.target_class
    assert objective.target_class_pair is not None
    return objective.target_class_pair[0]


def _base_candidate(objective: CorpusObjectiveSpec, backend_id: str, index: int) -> BackendCandidateSpec:
    class_name = _objective_primary_class(objective)
    base = _base_class_parameters(class_name)
    scenario_family = (
        "environment_regime_case"
        if objective.target_environment_regimes
        else "switching_case"
        if objective.target_class == "braking" and "controlled_1d" in backend_id
        else "shared_boundary_case"
    )
    candidate_id = f"{objective.objective_id}_{backend_id}_{index}"
    environment_id = objective.target_environment_regimes[index % max(len(objective.target_environment_regimes), 1)] if objective.target_environment_regimes else ""
    return BackendCandidateSpec(
        candidate_id=candidate_id,
        scenario_id=objective.objective_id,
        scenario_family=scenario_family,
        target_class=class_name,
        difficulty_tier=objective.target_difficulty,
        seed=1_000 + index,
        duration=2.0,
        sample_period=0.5,
        initial_position=0.0,
        initial_velocity=base["initial_velocity"],
        acceleration=base["acceleration"],
        measurement_std=0.03,
        switch_time=1.0 if scenario_family == "switching_case" else None,
        acceleration_after_switch=-0.7 if scenario_family == "switching_case" else None,
        drag_coefficient=0.20 if scenario_family == "environment_regime_case" else None,
        density_scale=1.0 if environment_id == "nominal_mixed" else 1.1 if environment_id == "dense_calm" else 0.82 if environment_id else None,
        wind_bias=0.05 if environment_id == "nominal_mixed" else 0.0 if environment_id == "dense_calm" else 0.12 if environment_id else None,
        input_deck_hash=f"{candidate_id}_deck" if "mock_file_backend_1d" in backend_id else None,
        longitudinal_command=(0.0, 0.0, -0.7, -0.7, -0.7) if scenario_family == "switching_case" else (),
        provenance={"objective_id": objective.objective_id, "sampler_name": "", "backend_id": backend_id, "parent_candidate_id": "", "environment_id": environment_id},
    )


def _candidate_row(candidate: BackendCandidateSpec, sampler_name: str, parent_candidate_id: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "objective_id": candidate.provenance["objective_id"],
        "sampler_name": sampler_name,
        "backend_id": candidate.provenance["backend_id"],
        "scenario_family": candidate.scenario_family,
        "target_class": candidate.target_class,
        "difficulty_tier": candidate.difficulty_tier,
        "seed": candidate.seed,
        "duration": candidate.duration,
        "sample_period": candidate.sample_period,
        "initial_velocity": candidate.initial_velocity,
        "acceleration": candidate.acceleration,
        "measurement_std": candidate.measurement_std,
        "environment_id": candidate.provenance.get("environment_id", ""),
        "parent_candidate_id": parent_candidate_id,
    }


def _random_sampler(objective: CorpusObjectiveSpec, backend_id: str, budget: int, rng: random.Random) -> list[BackendCandidateSpec]:
    candidates = []
    for index in range(max(2, budget // 6)):
        base = _base_candidate(objective, backend_id, index)
        candidates.append(
            _replace_candidate(
                base,
                seed=base.seed + 10 * index,
                duration=1.7 + 0.6 * rng.random(),
                measurement_std=0.02 + 0.05 * rng.random(),
                initial_velocity=base.initial_velocity + 0.15 * (rng.random() - 0.5),
                acceleration=base.acceleration + 0.15 * (rng.random() - 0.5),
                sampler_name="random",
                parent_candidate_id="",
            )
        )
    return candidates


def _grid_sampler(objective: CorpusObjectiveSpec, backend_id: str, budget: int) -> list[BackendCandidateSpec]:
    grid = [(1.8, 0.03), (2.0, 0.04), (2.2, 0.05)]
    candidates = []
    for index, (duration, measurement_std) in enumerate(grid[: max(2, budget // 6)]):
        base = _base_candidate(objective, backend_id, 100 + index)
        candidates.append(_replace_candidate(base, duration=duration, measurement_std=measurement_std, sampler_name="grid", parent_candidate_id=""))
    return candidates


def _lhs_sampler(objective: CorpusObjectiveSpec, backend_id: str, budget: int) -> list[BackendCandidateSpec]:
    spans = [(0.15, 0.25), (0.35, 0.50), (0.55, 0.75)]
    candidates = []
    for index, (alpha, beta) in enumerate(spans[: max(2, budget // 6)]):
        base = _base_candidate(objective, backend_id, 200 + index)
        duration = 1.6 + alpha
        measurement_std = 0.02 + 0.04 * beta
        candidates.append(
            _replace_candidate(
                base,
                duration=duration,
                measurement_std=measurement_std,
                initial_velocity=base.initial_velocity + 0.1 * alpha,
                acceleration=base.acceleration + 0.1 * (beta - 0.5),
                sampler_name="lhs",
                parent_candidate_id="",
            )
        )
    return candidates


def _boundary_mutation_sampler(objective: CorpusObjectiveSpec, backend_id: str, budget: int) -> list[BackendCandidateSpec]:
    seeds = _grid_sampler(objective, backend_id, budget)
    candidates = []
    for index, seed_candidate in enumerate(seeds):
        candidates.append(
            _replace_candidate(
                seed_candidate,
                candidate_id=f"{seed_candidate.candidate_id}_boundary_mut",
                duration=max(1.5, seed_candidate.duration * 0.92),
                acceleration=seed_candidate.acceleration * 0.82,
                measurement_std=seed_candidate.measurement_std * 1.10,
                sampler_name="boundary_mutation",
                parent_candidate_id=seed_candidate.candidate_id,
            )
        )
    return candidates


def _archive_mutation_sampler(objective: CorpusObjectiveSpec, backend_id: str, budget: int) -> list[BackendCandidateSpec]:
    seeds = _lhs_sampler(objective, backend_id, budget)
    candidates = []
    for index, seed_candidate in enumerate(seeds):
        candidates.append(
            _replace_candidate(
                seed_candidate,
                candidate_id=f"{seed_candidate.candidate_id}_archive_mut",
                duration=seed_candidate.duration * 1.08,
                measurement_std=seed_candidate.measurement_std * 0.95,
                acceleration=seed_candidate.acceleration + 0.04 * ((index % 2) * 2 - 1),
                sampler_name="archive_mutation",
                parent_candidate_id=seed_candidate.candidate_id,
            )
        )
    return candidates


def _stress_mutation_sampler(objective: CorpusObjectiveSpec, backend_id: str, budget: int) -> list[BackendCandidateSpec]:
    seeds = _random_sampler(objective, backend_id, budget, random.Random(99))
    candidates = []
    for seed_candidate in seeds:
        candidates.append(
            _replace_candidate(
                seed_candidate,
                candidate_id=f"{seed_candidate.candidate_id}_stress_mut",
                duration=max(1.4, seed_candidate.duration * 0.88),
                measurement_std=seed_candidate.measurement_std * 1.35,
                acceleration=seed_candidate.acceleration * 0.9,
                sampler_name="stress_mutation",
                parent_candidate_id=seed_candidate.candidate_id,
            )
        )
    return candidates


def _replace_candidate(
    candidate: BackendCandidateSpec,
    *,
    candidate_id: str | None = None,
    seed: int | None = None,
    duration: float | None = None,
    measurement_std: float | None = None,
    initial_velocity: float | None = None,
    acceleration: float | None = None,
    sampler_name: str,
    parent_candidate_id: str,
) -> BackendCandidateSpec:
    provenance = dict(candidate.provenance)
    provenance["sampler_name"] = sampler_name
    provenance["parent_candidate_id"] = parent_candidate_id
    return BackendCandidateSpec(
        candidate_id=candidate_id or candidate.candidate_id,
        scenario_id=candidate.scenario_id,
        scenario_family=candidate.scenario_family,
        target_class=candidate.target_class,
        difficulty_tier=candidate.difficulty_tier,
        seed=seed if seed is not None else candidate.seed,
        duration=duration if duration is not None else candidate.duration,
        sample_period=candidate.sample_period,
        initial_position=candidate.initial_position,
        initial_velocity=initial_velocity if initial_velocity is not None else candidate.initial_velocity,
        acceleration=acceleration if acceleration is not None else candidate.acceleration,
        measurement_std=measurement_std if measurement_std is not None else candidate.measurement_std,
        switch_time=candidate.switch_time,
        acceleration_after_switch=candidate.acceleration_after_switch,
        drag_coefficient=candidate.drag_coefficient,
        density_scale=candidate.density_scale,
        wind_bias=candidate.wind_bias,
        input_deck_hash=candidate.input_deck_hash,
        longitudinal_command=candidate.longitudinal_command,
        provenance=provenance,
    )


def generate_candidates_from_objectives(objectives: tuple[CorpusObjectiveSpec, ...] | None = None) -> tuple[BackendCandidateSpec, ...]:
    objective_list = objectives or default_corpus_objectives()
    planner = analyze_capability_aware_search()
    planner_backends = {row["family"]: row for row in planner.backend_plan_rows}
    rng = random.Random(7)
    candidates: list[BackendCandidateSpec] = []
    for objective in objective_list:
        budget = int(objective.runtime_budget.get("candidate_budget", 12))
        for backend_id in _backend_constraints_for_objective(objective):
            if backend_id not in planner_backends:
                continue
            candidates.extend(_random_sampler(objective, backend_id, budget, rng))
            candidates.extend(_grid_sampler(objective, backend_id, budget))
            candidates.extend(_lhs_sampler(objective, backend_id, budget))
            candidates.extend(_boundary_mutation_sampler(objective, backend_id, budget))
            candidates.extend(_archive_mutation_sampler(objective, backend_id, budget))
            candidates.extend(_stress_mutation_sampler(objective, backend_id, budget))
    return tuple(candidates)


def generate_candidates_from_objective_file(path: str | Path) -> tuple[BackendCandidateSpec, ...]:
    return generate_candidates_from_objectives(load_corpus_objectives_from_yaml(path))


def analyze_candidate_generation() -> CandidateGenerationResult:
    objectives = default_corpus_objectives()
    candidates = generate_candidates_from_objectives(objectives)
    candidate_rows = [_candidate_row(candidate, str(candidate.provenance["sampler_name"]), str(candidate.provenance["parent_candidate_id"])) for candidate in candidates]
    sampler_manifest = {
        "version": "m37_v1",
        "objective_source": "default_corpus_objectives",
        "samplers": [
            {"name": "random", "kind": "stochastic"},
            {"name": "grid", "kind": "deterministic"},
            {"name": "lhs", "kind": "space_filling"},
            {"name": "boundary_mutation", "kind": "mutation"},
            {"name": "archive_mutation", "kind": "mutation"},
            {"name": "stress_mutation", "kind": "mutation"},
        ],
        "candidate_count": len(candidates),
        "objective_count": len(objectives),
    }
    report_markdown = "\n".join(
        [
            "# Candidate Generation From Objectives",
            "",
            "## Summary",
            f"- objectives loaded: `{len(objectives)}`",
            f"- candidates generated: `{len(candidates)}`",
            f"- sampler families used: `{len(sampler_manifest['samplers'])}`",
            "",
            "## Objective Backends",
            "| Objective | Backends | Budget |",
            "| --- | --- | --- |",
            *[
                f"| `{objective.objective_id}` | `{', '.join(_backend_constraints_for_objective(objective))}` | `{objective.runtime_budget}` |"
                for objective in objectives
            ],
            "",
            "## Notes",
            "- Candidate generation is now objective-driven and no longer relies on the fixed `_candidate_pool()` path from PLN-019.",
            "- Mutation samplers preserve parent lineage in candidate provenance so later archive logic can build mutation histories.",
        ]
    )
    return CandidateGenerationResult(
        sampler_manifest=sampler_manifest,
        generated_candidate_rows=tuple(candidate_rows),
        report_markdown=report_markdown,
    )


def _render_sampler_comparison_png(rows: tuple[dict[str, Any], ...]) -> bytes:
    counts: dict[str, int] = {}
    for row in rows:
        sampler = str(row["sampler_name"])
        counts[sampler] = counts.get(sampler, 0) + 1
    labels = list(counts.keys())
    values = [counts[label] for label in labels]
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.bar(labels, values, color="#4d8f77")
    ax.set_ylabel("Candidate Count")
    ax.set_title("Sampler Family Comparison")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    from io import BytesIO
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def _render_coverage_png(rows: tuple[dict[str, Any], ...]) -> bytes:
    scenarios = sorted({str(row["scenario_family"]) for row in rows})
    samplers = sorted({str(row["sampler_name"]) for row in rows})
    matrix = []
    for scenario in scenarios:
        scenario_row = []
        for sampler in samplers:
            scenario_row.append(sum(1 for row in rows if row["scenario_family"] == scenario and row["sampler_name"] == sampler))
        matrix.append(scenario_row)
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    image = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(samplers)), labels=samplers, fontsize=8)
    ax.set_yticks(range(len(scenarios)), labels=scenarios, fontsize=8)
    ax.set_title("Generated Candidate Coverage")
    for row_index, row_values in enumerate(matrix):
        for column_index, value in enumerate(row_values):
            ax.text(column_index, row_index, f"{value}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.04)
    fig.tight_layout()
    from io import BytesIO
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def _render_lineage_png(rows: tuple[dict[str, Any], ...]) -> bytes:
    sampled = [row for row in rows if str(row["parent_candidate_id"])]
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.axis("off")
    y = 0.90
    ax.text(0.50, 0.98, "Mutation Lineage Preview", ha="center", va="center", fontsize=11)
    for row in sampled[:8]:
        parent = str(row["parent_candidate_id"])
        child = str(row["candidate_id"])
        sampler = str(row["sampler_name"])
        ax.text(0.22, y, parent, ha="center", va="center", fontsize=8, bbox={"boxstyle": "round,pad=0.2", "facecolor": "#eee"})
        ax.text(0.50, y, sampler, ha="center", va="center", fontsize=8)
        ax.text(0.78, y, child, ha="center", va="center", fontsize=8, bbox={"boxstyle": "round,pad=0.2", "facecolor": "#e6f2ea"})
        ax.annotate("", xy=(0.70, y), xytext=(0.30, y), arrowprops={"arrowstyle": "->", "linewidth": 1.0})
        y -= 0.10
    fig.tight_layout()
    from io import BytesIO
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def write_candidate_generation_artifacts(
    base_dir: str | Path,
    *,
    result: CandidateGenerationResult | None = None,
) -> CandidateGenerationArtifacts:
    run_dir = Path(base_dir) / "candidate_generation"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = result or analyze_candidate_generation()

    sampler_manifest_path = run_dir / "sampler_manifest.json"
    generated_candidates_path = run_dir / "generated_candidates.csv"
    report_path = run_dir / "candidate_generation_report.md"
    sampler_comparison_png_path = run_dir / "sampler_family_comparison.png"
    candidate_coverage_png_path = run_dir / "generated_candidate_coverage.png"
    mutation_lineage_png_path = run_dir / "mutation_lineage_preview.png"

    sampler_manifest_path.write_text(json.dumps(payload.sampler_manifest, indent=2), encoding="utf-8")
    report_path.write_text(payload.report_markdown, encoding="utf-8")
    fieldnames = list(payload.generated_candidate_rows[0].keys()) if payload.generated_candidate_rows else []
    _write_csv(generated_candidates_path, list(payload.generated_candidate_rows), fieldnames)
    sampler_comparison_png_path.write_bytes(_render_sampler_comparison_png(payload.generated_candidate_rows))
    candidate_coverage_png_path.write_bytes(_render_coverage_png(payload.generated_candidate_rows))
    mutation_lineage_png_path.write_bytes(_render_lineage_png(payload.generated_candidate_rows))

    return CandidateGenerationArtifacts(
        run_dir=run_dir,
        sampler_manifest_path=sampler_manifest_path,
        generated_candidates_path=generated_candidates_path,
        report_path=report_path,
        sampler_comparison_png_path=sampler_comparison_png_path,
        candidate_coverage_png_path=candidate_coverage_png_path,
        mutation_lineage_png_path=mutation_lineage_png_path,
    )
