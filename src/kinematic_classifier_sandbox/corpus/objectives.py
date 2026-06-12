from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from kinematic_classifier_sandbox.reports.markdown import MarkdownDocument

from ..utils.plotting import plt


class LeakageConstraintSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    variable_name: str
    max_delta_ratio: float


class CorpusObjectiveSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    objective_id: str
    description: str
    target_class: str | None = None
    target_class_pair: tuple[str, str] | None = None
    target_feature_excitation: dict[str, dict[str, float]] = Field(default_factory=dict)
    target_difficulty: str = "realistic_v1"
    target_posterior_entropy: str | None = None
    target_environment_regimes: tuple[str, ...] = Field(default_factory=tuple)
    leakage_constraints: tuple[LeakageConstraintSpec, ...] = Field(default_factory=tuple)
    backend_constraints: tuple[str, ...] = Field(default_factory=tuple)
    runtime_budget: dict[str, int] = Field(default_factory=dict)


class CorpusObjectiveArtifacts(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_dir: Path
    schema_path: Path
    example_objectives_path: Path
    report_path: Path
    relationship_png_path: Path
    coverage_png_path: Path


class CorpusObjectiveResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    contract_schema: dict[str, Any]
    example_objectives: tuple[dict[str, Any], ...]
    validation_rows: tuple[dict[str, Any], ...]
    report_markdown: str


def _schema() -> dict[str, Any]:
    return CorpusObjectiveSpec.model_json_schema()


def default_corpus_objectives() -> tuple[CorpusObjectiveSpec, ...]:
    return (
        CorpusObjectiveSpec(
            objective_id="cv_vs_ca_boundary_entropy",
            description="Boundary search for constant velocity versus constant acceleration with elevated ambiguity.",
            target_class_pair=("constant_velocity", "constant_acceleration"),
            target_feature_excitation={
                "acceleration_range": {"min": 0.08, "max": 0.40},
                "sampling_irregularity": {"max": 0.35},
            },
            target_difficulty="boundary_v1",
            target_posterior_entropy="high",
            leakage_constraints=(
                LeakageConstraintSpec(variable_name="duration", max_delta_ratio=0.20),
                LeakageConstraintSpec(variable_name="measurement_std", max_delta_ratio=0.20),
            ),
            backend_constraints=("parameter_only_1d", "environment_aware_1d", "mock_file_backend_1d"),
            runtime_budget={"candidate_budget": 18, "archive_iterations": 8},
        ),
        CorpusObjectiveSpec(
            objective_id="switching_transition_delay",
            description="Stress switching trajectories where transition-sensitive methods should have measurable value.",
            target_class="braking",
            target_feature_excitation={
                "acceleration_sign_changes": {"min": 1.0},
            },
            target_difficulty="stress_v1",
            target_posterior_entropy="medium",
            leakage_constraints=(LeakageConstraintSpec(variable_name="num_samples", max_delta_ratio=0.25),),
            backend_constraints=("controlled_1d", "mock_file_backend_1d"),
            runtime_budget={"candidate_budget": 16, "archive_iterations": 10},
        ),
        CorpusObjectiveSpec(
            objective_id="environment_regime_balanced_accel",
            description="Generate balanced constant-acceleration examples across environment regimes with leakage limits.",
            target_class="constant_acceleration",
            target_feature_excitation={
                "acceleration_range": {"min": 0.30},
                "position_range": {"min": 1.0},
            },
            target_difficulty="realistic_v1",
            target_environment_regimes=("dense_calm", "nominal_mixed", "thin_windy"),
            leakage_constraints=(
                LeakageConstraintSpec(variable_name="density_scale", max_delta_ratio=0.15),
                LeakageConstraintSpec(variable_name="wind_bias", max_delta_ratio=0.15),
            ),
            backend_constraints=("environment_aware_1d",),
            runtime_budget={"candidate_budget": 12, "archive_iterations": 6},
        ),
    )


def validate_corpus_objective(spec: CorpusObjectiveSpec) -> list[str]:
    errors: list[str] = []
    if not spec.objective_id:
        errors.append("objective_id is required")
    if not spec.description:
        errors.append("description is required")
    if spec.target_class is None and spec.target_class_pair is None:
        errors.append("either target_class or target_class_pair is required")
    if spec.target_class is not None and spec.target_class_pair is not None:
        errors.append("target_class and target_class_pair must not both be set")
    if not spec.runtime_budget:
        errors.append("runtime_budget is required")
    if "candidate_budget" in spec.runtime_budget and int(spec.runtime_budget["candidate_budget"]) <= 0:
        errors.append("candidate_budget must be positive")
    for constraint in spec.leakage_constraints:
        if constraint.max_delta_ratio <= 0.0:
            errors.append(f"leakage constraint for {constraint.variable_name} must be positive")
    return errors


def analyze_corpus_objectives() -> CorpusObjectiveResult:
    objectives = default_corpus_objectives()
    validation_rows = []
    for objective in objectives:
        errors = validate_corpus_objective(objective)
        validation_rows.append(
            {
                "objective_id": objective.objective_id,
                "valid": not errors,
                "error_count": len(errors),
                "errors": errors,
                "has_class_target": objective.target_class is not None,
                "has_class_pair_target": objective.target_class_pair is not None,
                "has_feature_targets": bool(objective.target_feature_excitation),
                "has_environment_targets": bool(objective.target_environment_regimes),
                "has_leakage_constraints": bool(objective.leakage_constraints),
            }
        )
    report = MarkdownDocument()
    report.heading("Corpus Objective Validation", level=1)
    report.heading("Summary", level=2)
    report.bullet_list(
        [
            f"objectives declared: `{len(objectives)}`",
            f"valid objectives: `{sum(1 for row in validation_rows if row['valid'])}`",
        ]
    )
    report.heading("Objective Coverage", level=2)
    report.table(
        ["Objective", "Class", "Class Pair", "Features", "Environment", "Leakage", "Budget"],
        [
            (
                f"`{objective.objective_id}`",
                f"`{objective.target_class or ''}`",
                f"`{' vs '.join(objective.target_class_pair) if objective.target_class_pair else ''}`",
                f"`{len(objective.target_feature_excitation)}`",
                f"`{len(objective.target_environment_regimes)}`",
                f"`{len(objective.leakage_constraints)}`",
                f"`{objective.runtime_budget}`",
            )
            for objective in objectives
        ],
    )
    report.heading("Notes", level=2)
    report.bullet_list(
        [
            "Objectives are the driving contract for sampler-driven corpus generation in PLN-021.",
            "They intentionally combine class, feature, environment, leakage, and runtime constraints so the explorer can synthesize candidate pools from declared goals instead of fixed scenario lists.",
        ]
    )
    report_markdown = report.text()
    return CorpusObjectiveResult(
        contract_schema=_schema(),
        example_objectives=tuple(_objective_to_dict(objective) for objective in objectives),
        validation_rows=tuple(validation_rows),
        report_markdown=report_markdown,
    )


def _objective_to_dict(objective: CorpusObjectiveSpec) -> dict[str, Any]:
    return objective.model_dump()


def load_corpus_objectives_from_yaml(path: str | Path) -> tuple[CorpusObjectiveSpec, ...]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    objectives = []
    for item in payload.get("objectives", []):
        objectives.append(
            CorpusObjectiveSpec(
                objective_id=str(item["objective_id"]),
                description=str(item["description"]),
                target_class=item.get("target_class"),
                target_class_pair=tuple(item["target_class_pair"]) if item.get("target_class_pair") else None,
                target_feature_excitation=dict(item.get("target_feature_excitation", {})),
                target_difficulty=str(item.get("target_difficulty", "realistic_v1")),
                target_posterior_entropy=item.get("target_posterior_entropy"),
                target_environment_regimes=tuple(item.get("target_environment_regimes", ())),
                leakage_constraints=tuple(
                    LeakageConstraintSpec(
                        variable_name=str(constraint["variable_name"]),
                        max_delta_ratio=float(constraint["max_delta_ratio"]),
                    )
                    for constraint in item.get("leakage_constraints", ())
                ),
                backend_constraints=tuple(item.get("backend_constraints", ())),
                runtime_budget=dict(item.get("runtime_budget", {})),
            )
        )
    return tuple(objectives)


def _render_relationship_png() -> bytes:
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.axis("off")
    nodes = [
        (0.12, 0.75, "Class / Class Pair"),
        (0.38, 0.75, "Feature Targets"),
        (0.64, 0.75, "Environment Regimes"),
        (0.88, 0.75, "Leakage Constraints"),
        (0.50, 0.40, "CorpusObjectiveSpec"),
        (0.25, 0.10, "Sampler Layer"),
        (0.50, 0.10, "Class Validity"),
        (0.75, 0.10, "Feature + Classifier Scoring"),
    ]
    for x, y, label in nodes:
        ax.text(x, y, label, ha="center", va="center", fontsize=9, bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f5f2e8", "edgecolor": "#445"})
    arrows = [
        ((0.12, 0.70), (0.44, 0.47)),
        ((0.38, 0.70), (0.47, 0.47)),
        ((0.64, 0.70), (0.53, 0.47)),
        ((0.88, 0.70), (0.56, 0.47)),
        ((0.50, 0.34), (0.25, 0.16)),
        ((0.50, 0.34), (0.50, 0.16)),
        ((0.50, 0.34), (0.75, 0.16)),
    ]
    for (x0, y0), (x1, y1) in arrows:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "->", "linewidth": 1.2})
    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def _render_coverage_png(result: CorpusObjectiveResult) -> bytes:
    rows = list(result.validation_rows)
    categories = ("has_class_target", "has_class_pair_target", "has_feature_targets", "has_environment_targets", "has_leakage_constraints")
    data = [[1.0 if bool(row[category]) else 0.0 for category in categories] for row in rows]
    labels = [str(row["objective_id"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    image = ax.imshow(data, cmap="Greens", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(categories)), labels=[category.replace("has_", "").replace("_", "\n") for category in categories], fontsize=8)
    ax.set_yticks(range(len(labels)), labels=labels, fontsize=8)
    ax.set_title("Objective Coverage Map")
    for row_index, row_values in enumerate(data):
        for column_index, value in enumerate(row_values):
            ax.text(column_index, row_index, f"{value:.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.04)
    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def write_corpus_objective_artifacts(
    base_dir: str | Path,
    *,
    result: CorpusObjectiveResult | None = None,
) -> CorpusObjectiveArtifacts:
    run_dir = Path(base_dir) / "corpus_objectives"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = result or analyze_corpus_objectives()

    schema_path = run_dir / "corpus_objective_schema.json"
    example_objectives_path = run_dir / "example_objectives.yaml"
    report_path = run_dir / "objective_validation_report.md"
    relationship_png_path = run_dir / "objective_field_relationship.png"
    coverage_png_path = run_dir / "example_objective_coverage_map.png"

    schema_path.write_text(json.dumps(payload.contract_schema, indent=2), encoding="utf-8")
    example_objectives_path.write_text(yaml.safe_dump({"objectives": list(payload.example_objectives)}, sort_keys=False), encoding="utf-8")
    report_path.write_text(payload.report_markdown, encoding="utf-8")
    relationship_png_path.write_bytes(_render_relationship_png())
    coverage_png_path.write_bytes(_render_coverage_png(payload))

    return CorpusObjectiveArtifacts(
        run_dir=run_dir,
        schema_path=schema_path,
        example_objectives_path=example_objectives_path,
        report_path=report_path,
        relationship_png_path=relationship_png_path,
        coverage_png_path=coverage_png_path,
    )
