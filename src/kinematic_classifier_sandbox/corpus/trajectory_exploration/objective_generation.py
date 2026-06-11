from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

from ...utils.io import _write_json, _write_text, write_csv
from ..gym_types import CorpusGymTarget
from .contracts import TrajectoryExplorationObjective
from .objective_scoring import ObjectiveProofArtifacts, write_objective_proof_artifacts


@dataclass(frozen=True, slots=True)
class FeatureBinSpec:
    bin_id: str
    min_value: float | None = None
    max_value: float | None = None

    def as_constraints(self) -> dict[str, float]:
        constraints: dict[str, float] = {}
        if self.min_value is not None:
            constraints["min"] = self.min_value
        if self.max_value is not None:
            constraints["max"] = self.max_value
        return constraints


@dataclass(frozen=True, slots=True)
class FeatureAxisSpec:
    feature_name: str
    bins: tuple[FeatureBinSpec, ...]


@dataclass(frozen=True, slots=True)
class FeatureRowSpec:
    row_id: str
    axis_name: str
    bin_id: str
    target_tier: str = "boundary_v1"
    preferred_class: str = "maneuver"


@dataclass(frozen=True, slots=True)
class ClassPairRegionSpec:
    pair_id: str
    class_pair: tuple[str, str]
    target_tier: str = "boundary_v1"
    target_prior_sensitivity: str | None = None


@dataclass(frozen=True, slots=True)
class PosteriorTargetRegionSpec:
    region_id: str
    posterior_distribution: dict[str, float]
    target_tier: str = "boundary_v1"
    evidence_provider_id: str = "class_similarity_proxy_v1"


@dataclass(frozen=True, slots=True)
class NoveltyRegionSpec:
    region_id: str
    description: str
    class_name: str | None = None
    class_pair: tuple[str, str] | None = None
    feature_constraints: dict[str, dict[str, float]] | None = None
    target_tier: str = "adversarial_v1"


@dataclass(frozen=True, slots=True)
class TrajectoryObjectiveGenerationSpec:
    spec_id: str
    description: str
    feature_axes: tuple[FeatureAxisSpec, ...]
    feature_rows: tuple[FeatureRowSpec, ...]
    class_pair_regions: tuple[ClassPairRegionSpec, ...]
    posterior_target_regions: tuple[PosteriorTargetRegionSpec, ...]
    novelty_regions: tuple[NoveltyRegionSpec, ...]
    include_full_feature_space: bool = True
    include_feature_rows: bool = True
    include_class_pair_regions: bool = True
    include_posterior_target_regions: bool = True
    include_novelty_regions: bool = True
    default_evaluation_budget: int = 18


@dataclass(frozen=True, slots=True)
class GeneratedTrajectoryObjectiveSuite:
    spec: TrajectoryObjectiveGenerationSpec
    objectives: tuple[TrajectoryExplorationObjective, ...]
    objective_rows: tuple[dict[str, object], ...]
    manifest: dict[str, object]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class GeneratedTrajectoryObjectiveArtifacts:
    run_dir: Path
    spec_path: Path
    manifest_path: Path
    objectives_path: Path
    objective_table_path: Path
    report_path: Path
    proof_artifacts: ObjectiveProofArtifacts | None = None


def _weights(
    *,
    validity: float,
    excitation: float,
    coverage: float,
    geometry: float,
    stress: float,
    prior: float,
    leakage: float,
    physical: float,
) -> dict[str, float]:
    return {
        "validity": validity,
        "feature_excitation": excitation,
        "coverage_gain": coverage,
        "geometry_score": geometry,
        "classifier_stress": stress,
        "prior_sensitivity": prior,
        "leakage_penalty": leakage,
        "physical_invalidity_penalty": physical,
    }


def default_objective_generation_spec() -> TrajectoryObjectiveGenerationSpec:
    return TrajectoryObjectiveGenerationSpec(
        spec_id="trajectory_feature_class_space_v1",
        description="Mechanical objective pack covering feature cells, feature rows, class-pair regions, and novelty-biased trajectory zones.",
        feature_axes=(
            FeatureAxisSpec(
                feature_name="acceleration_range",
                bins=(
                    FeatureBinSpec("mid", min_value=0.20, max_value=0.70),
                    FeatureBinSpec("high", min_value=0.70),
                ),
            ),
            FeatureAxisSpec(
                feature_name="monotonicity",
                bins=(
                    FeatureBinSpec("low", max_value=0.90),
                    FeatureBinSpec("high", min_value=0.90),
                ),
            ),
        ),
        feature_rows=(
            FeatureRowSpec(row_id="accel_high_row", axis_name="acceleration_range", bin_id="high"),
            FeatureRowSpec(row_id="monotonicity_low_row", axis_name="monotonicity", bin_id="low"),
        ),
        class_pair_regions=(
            ClassPairRegionSpec("cv_vs_ca", ("constant_velocity", "constant_acceleration")),
            ClassPairRegionSpec("cv_vs_braking", ("constant_velocity", "braking"), target_prior_sensitivity="high"),
            ClassPairRegionSpec("ca_vs_maneuver", ("constant_acceleration", "maneuver")),
        ),
        posterior_target_regions=(
            PosteriorTargetRegionSpec(
                region_id="cv_ca_50_50",
                posterior_distribution={"constant_velocity": 0.5, "constant_acceleration": 0.5},
            ),
            PosteriorTargetRegionSpec(
                region_id="cv_braking_50_50",
                posterior_distribution={"constant_velocity": 0.5, "braking": 0.5},
            ),
        ),
        novelty_regions=(
            NoveltyRegionSpec(
                region_id="novel_maneuver_feature_zone",
                description="Novel maneuver trajectories with high acceleration range and low monotonicity not well covered by scripted families.",
                class_name="maneuver",
                feature_constraints={
                    "acceleration_range": {"min": 0.75},
                    "monotonicity": {"max": 0.88},
                    "sampling_irregularity": {"min": 0.10},
                },
            ),
            NoveltyRegionSpec(
                region_id="novel_cv_ca_boundary_zone",
                description="Novel CV/CA boundary witnesses with short duration and high prior sensitivity.",
                class_pair=("constant_velocity", "constant_acceleration"),
                feature_constraints={
                    "duration": {"max": 3.2},
                    "acceleration_variance": {"min": 0.01, "max": 0.24},
                },
                target_tier="boundary_v1",
            ),
        ),
    )


def _feature_axis_lookup(spec: TrajectoryObjectiveGenerationSpec) -> dict[str, FeatureAxisSpec]:
    return {axis.feature_name: axis for axis in spec.feature_axes}


def _feature_bin_lookup(spec: TrajectoryObjectiveGenerationSpec) -> dict[tuple[str, str], FeatureBinSpec]:
    return {
        (axis.feature_name, feature_bin.bin_id): feature_bin
        for axis in spec.feature_axes
        for feature_bin in axis.bins
    }


def _feature_cell_objectives(spec: TrajectoryObjectiveGenerationSpec) -> list[TrajectoryExplorationObjective]:
    objectives: list[TrajectoryExplorationObjective] = []
    if not spec.include_full_feature_space:
        return objectives
    for bin_tuple in product(*[axis.bins for axis in spec.feature_axes]):
        feature_constraints = {
            axis.feature_name: feature_bin.as_constraints()
            for axis, feature_bin in zip(spec.feature_axes, bin_tuple, strict=True)
        }
        bin_suffix = "_".join(f"{axis.feature_name}_{feature_bin.bin_id}" for axis, feature_bin in zip(spec.feature_axes, bin_tuple, strict=True))
        target = CorpusGymTarget(
            target_id=f"generated_feature_cell_{bin_suffix}",
            target_type="target_feature_space_cell",
            description=f"Mechanically generated feature-space cell objective for {bin_suffix}.",
            class_name="maneuver",
            feature_constraints=feature_constraints,
            target_tier="adversarial_v1",
        )
        objectives.append(
            TrajectoryExplorationObjective(
                objective_id=f"feature_cell__{bin_suffix}",
                mode="feature_space_cell",
                geometry_target="fill_underexcited_feature_cells",
                description=target.description,
                target=target,
                reward_weights=_weights(validity=0.24, excitation=0.34, coverage=0.18, geometry=0.16, stress=0.08, prior=0.04, leakage=0.12, physical=0.12),
                thresholds={"min_class_validity": 0.45, "min_feature_excitation": 0.65, "max_leakage_penalty": 0.40},
                evaluation_budget=spec.default_evaluation_budget,
                backend_constraints={"mechanically_generated": True, "generation_scope": "feature_cell"},
                classifier_family="feature_space_explorer",
            )
        )
    return objectives


def _feature_row_objectives(spec: TrajectoryObjectiveGenerationSpec) -> list[TrajectoryExplorationObjective]:
    objectives: list[TrajectoryExplorationObjective] = []
    if not spec.include_feature_rows:
        return objectives
    bins = _feature_bin_lookup(spec)
    for row_spec in spec.feature_rows:
        row_bin = bins[(row_spec.axis_name, row_spec.bin_id)]
        target = CorpusGymTarget(
            target_id=f"generated_feature_row_{row_spec.row_id}",
            target_type="target_feature_space_row",
            description=f"Mechanically generated feature-row objective anchored on {row_spec.axis_name}={row_spec.bin_id}.",
            class_name=row_spec.preferred_class,
            feature_constraints={row_spec.axis_name: row_bin.as_constraints()},
            target_tier=row_spec.target_tier,
        )
        objectives.append(
            TrajectoryExplorationObjective(
                objective_id=f"feature_row__{row_spec.row_id}",
                mode="feature_space_row",
                geometry_target="sweep_feature_row_novelty",
                description=target.description,
                target=target,
                reward_weights=_weights(validity=0.20, excitation=0.26, coverage=0.18, geometry=0.18, stress=0.08, prior=0.04, leakage=0.10, physical=0.10),
                thresholds={"min_class_validity": 0.40, "min_feature_excitation": 0.55, "max_leakage_penalty": 0.45},
                evaluation_budget=spec.default_evaluation_budget,
                backend_constraints={"mechanically_generated": True, "generation_scope": "feature_row", "row_axis": row_spec.axis_name},
                classifier_family="feature_row_explorer",
            )
        )
    return objectives


def _class_pair_objectives(spec: TrajectoryObjectiveGenerationSpec) -> list[TrajectoryExplorationObjective]:
    objectives: list[TrajectoryExplorationObjective] = []
    if not spec.include_class_pair_regions:
        return objectives
    for pair_spec in spec.class_pair_regions:
        target = CorpusGymTarget(
            target_id=f"generated_class_pair_{pair_spec.pair_id}",
            target_type="target_class_space_region",
            description=f"Mechanically generated class-pair boundary objective for {pair_spec.class_pair[0]} vs {pair_spec.class_pair[1]}.",
            class_pair=pair_spec.class_pair,
            target_tier=pair_spec.target_tier,
            target_prior_sensitivity=pair_spec.target_prior_sensitivity,
        )
        objectives.append(
            TrajectoryExplorationObjective(
                objective_id=f"class_pair__{pair_spec.pair_id}",
                mode="class_pair_region",
                geometry_target="create_ambiguous_boundary_trajectories",
                description=target.description,
                target=target,
                reward_weights=_weights(validity=0.20, excitation=0.16, coverage=0.14, geometry=0.28, stress=0.14, prior=0.10, leakage=0.12, physical=0.12),
                thresholds={"min_class_validity": 0.45, "min_boundary_closeness": 0.50, "max_leakage_penalty": 0.40},
                evaluation_budget=spec.default_evaluation_budget,
                backend_constraints={"mechanically_generated": True, "generation_scope": "class_pair_region"},
                classifier_family="class_boundary_explorer",
            )
        )
    return objectives


def _posterior_target_objectives(spec: TrajectoryObjectiveGenerationSpec) -> list[TrajectoryExplorationObjective]:
    objectives: list[TrajectoryExplorationObjective] = []
    if not spec.include_posterior_target_regions:
        return objectives
    for target_spec in spec.posterior_target_regions:
        class_names = tuple(target_spec.posterior_distribution)
        target = CorpusGymTarget(
            target_id=f"generated_posterior_target_{target_spec.region_id}",
            target_type="target_posterior_distribution",
            description=f"Mechanically generated posterior target objective for {target_spec.region_id}.",
            class_pair=(class_names[0], class_names[1]) if len(class_names) == 2 else None,
            target_tier=target_spec.target_tier,
        )
        objectives.append(
            TrajectoryExplorationObjective(
                objective_id=f"posterior_target__{target_spec.region_id}",
                mode="posterior_target_distribution",
                geometry_target="match_target_posterior_distribution",
                description=target.description,
                target=target,
                reward_weights=_weights(validity=0.22, excitation=0.12, coverage=0.12, geometry=0.34, stress=0.08, prior=0.08, leakage=0.12, physical=0.12),
                thresholds={"min_class_validity": 0.35, "max_posterior_tv_error": 0.12, "max_leakage_penalty": 0.45},
                evaluation_budget=spec.default_evaluation_budget,
                backend_constraints={
                    "mechanically_generated": True,
                    "generation_scope": "posterior_target_distribution",
                    "posterior_target_distribution": target_spec.posterior_distribution,
                    "evidence_provider_id": target_spec.evidence_provider_id,
                },
                classifier_family="posterior_target_explorer",
            )
        )
    return objectives


def _novelty_objectives(spec: TrajectoryObjectiveGenerationSpec) -> list[TrajectoryExplorationObjective]:
    objectives: list[TrajectoryExplorationObjective] = []
    if not spec.include_novelty_regions:
        return objectives
    for novelty_spec in spec.novelty_regions:
        target = CorpusGymTarget(
            target_id=f"generated_novelty_region_{novelty_spec.region_id}",
            target_type="target_novel_feature_class_region",
            description=novelty_spec.description,
            class_name=novelty_spec.class_name,
            class_pair=novelty_spec.class_pair,
            feature_constraints=novelty_spec.feature_constraints,
            target_tier=novelty_spec.target_tier,
        )
        objectives.append(
            TrajectoryExplorationObjective(
                objective_id=f"novelty_region__{novelty_spec.region_id}",
                mode="novelty_region",
                geometry_target="discover_novel_feature_class_region",
                description=target.description,
                target=target,
                reward_weights=_weights(validity=0.18, excitation=0.24, coverage=0.20, geometry=0.20, stress=0.10, prior=0.06, leakage=0.10, physical=0.10),
                thresholds={"min_class_validity": 0.40, "min_feature_excitation": 0.55, "max_leakage_penalty": 0.45},
                evaluation_budget=spec.default_evaluation_budget,
                backend_constraints={
                    "mechanically_generated": True,
                    "generation_scope": "novelty_region",
                    "novelty_reference_families": ("scripted_profiles", "doe_schedule_bank"),
                },
                classifier_family="novelty_zone_explorer",
            )
        )
    return objectives


def generate_trajectory_exploration_objective_suite(
    spec: TrajectoryObjectiveGenerationSpec | None = None,
) -> GeneratedTrajectoryObjectiveSuite:
    resolved = spec or default_objective_generation_spec()
    objectives = tuple(
        _feature_cell_objectives(resolved)
        + _feature_row_objectives(resolved)
        + _class_pair_objectives(resolved)
        + _posterior_target_objectives(resolved)
        + _novelty_objectives(resolved)
    )
    objective_rows = tuple(
        {
            "objective_id": objective.objective_id,
            "mode": objective.mode,
            "geometry_target": objective.geometry_target,
            "target_id": objective.target.target_id,
            "target_type": objective.target.target_type,
            "target_class": objective.target.class_name or "",
            "target_class_pair": " vs ".join(objective.target.class_pair) if objective.target.class_pair else "",
            "target_tier": objective.target.target_tier or "",
            "feature_constraints": str(objective.target.feature_constraints or {}),
            "generation_scope": str(objective.backend_constraints.get("generation_scope", "")),
            "evaluation_budget": objective.evaluation_budget,
        }
        for objective in objectives
    )
    manifest = {
        "spec_id": resolved.spec_id,
        "objective_count": len(objectives),
        "feature_cell_count": len([row for row in objective_rows if row["generation_scope"] == "feature_cell"]),
        "feature_row_count": len([row for row in objective_rows if row["generation_scope"] == "feature_row"]),
        "class_pair_region_count": len([row for row in objective_rows if row["generation_scope"] == "class_pair_region"]),
        "posterior_target_region_count": len([row for row in objective_rows if row["generation_scope"] == "posterior_target_distribution"]),
        "novelty_region_count": len([row for row in objective_rows if row["generation_scope"] == "novelty_region"]),
        "feature_axes": [asdict(axis) for axis in resolved.feature_axes],
        "feature_rows": [asdict(row) for row in resolved.feature_rows],
        "class_pair_regions": [asdict(region) for region in resolved.class_pair_regions],
        "posterior_target_regions": [asdict(region) for region in resolved.posterior_target_regions],
        "novelty_regions": [asdict(region) for region in resolved.novelty_regions],
    }
    report_markdown = "\n".join(
        [
            "# Generated Trajectory Objective Suite",
            "",
            resolved.description,
            "",
            f"- objective count: `{manifest['objective_count']}`",
            f"- feature cells: `{manifest['feature_cell_count']}`",
            f"- feature rows: `{manifest['feature_row_count']}`",
            f"- class-pair regions: `{manifest['class_pair_region_count']}`",
            f"- posterior-target regions: `{manifest['posterior_target_region_count']}`",
            f"- novelty regions: `{manifest['novelty_region_count']}`",
            "",
            "This bundle mechanically generates objective functions for targeted feature-space excitation, class-boundary exploration, and novelty-seeking zones that are not limited to pre-scripted trajectory families.",
        ]
    )
    return GeneratedTrajectoryObjectiveSuite(
        spec=resolved,
        objectives=objectives,
        objective_rows=objective_rows,
        manifest=manifest,
        report_markdown=report_markdown,
    )


def generated_trajectory_exploration_objectives(
    spec: TrajectoryObjectiveGenerationSpec | None = None,
) -> tuple[TrajectoryExplorationObjective, ...]:
    return generate_trajectory_exploration_objective_suite(spec).objectives


def resolve_generated_trajectory_objective(
    objective_id: str,
    *,
    spec: TrajectoryObjectiveGenerationSpec | None = None,
) -> TrajectoryExplorationObjective:
    suite = generate_trajectory_exploration_objective_suite(spec)
    for objective in suite.objectives:
        if objective.objective_id == objective_id:
            return objective
    raise KeyError(f"unknown generated trajectory objective: {objective_id}")


def write_generated_trajectory_objective_artifacts(
    output_dir: str | Path,
    *,
    spec: TrajectoryObjectiveGenerationSpec | None = None,
) -> GeneratedTrajectoryObjectiveArtifacts:
    suite = generate_trajectory_exploration_objective_suite(spec)
    run_dir = Path(output_dir) / "trajectory_exploration_objectives"
    run_dir.mkdir(parents=True, exist_ok=True)
    spec_path = run_dir / "objective_generation_spec.json"
    manifest_path = run_dir / "objective_manifest.json"
    objectives_path = run_dir / "objectives.json"
    objective_table_path = run_dir / "objective_table.csv"
    report_path = run_dir / "report.md"
    _write_json(spec_path, asdict(suite.spec))
    _write_json(manifest_path, suite.manifest)
    _write_json(objectives_path, [asdict(objective) for objective in suite.objectives])
    write_csv(objective_table_path, list(suite.objective_rows), list(suite.objective_rows[0].keys()))
    _write_text(report_path, suite.report_markdown)
    proof_artifacts = write_objective_proof_artifacts(output_dir)
    return GeneratedTrajectoryObjectiveArtifacts(
        run_dir=run_dir,
        spec_path=spec_path,
        manifest_path=manifest_path,
        objectives_path=objectives_path,
        objective_table_path=objective_table_path,
        report_path=report_path,
        proof_artifacts=proof_artifacts,
    )
