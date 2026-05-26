from __future__ import annotations

import json
from typing import Any

from kinematic_classifier_sandbox.utils.plotting import figure_to_png_bytes
from kinematic_classifier_sandbox.utils.plotting import plt
from ..runtime_paths import prepare_matplotlib

from ..analysis.generated_corpus_features import collect_generated_corpus_records
from ..common_experiment.adapters import ExecutablePairSpec, ExecutableTrajectory
from ..common_experiment.runner import analyze_common_trajectory_corpus
from ..corpus.exploration.objective_driven_qd_archive import analyze_objective_driven_qd_archive
from .classifier_scoring import analyze_corpus_classifier_scoring
from .objectives import default_corpus_objectives
from .selected_generated_corpus_artifact_io import write_selected_generated_corpus_artifacts
from .selected_generated_corpus_contracts import (
    SelectedGeneratedCorpusArtifacts,
    SelectedGeneratedCorpusResult,
)


def _objective_lookup() -> dict[str, Any]:
    return {objective.objective_id: objective for objective in default_corpus_objectives()}


def _canonical_pair(record) -> tuple[str, str, str]:
    objective = _objective_lookup().get(str(record.candidate.provenance.get("objective_id", "")))
    if objective is not None and objective.target_class_pair is not None:
        class_a, class_b = objective.target_class_pair
        return f"{class_a}_vs_{class_b}", class_a, class_b
    target = record.candidate.target_class
    mapping = {
        "constant_acceleration": ("constant_velocity_vs_constant_acceleration", "constant_velocity", "constant_acceleration"),
        "braking": ("constant_velocity_vs_braking", "constant_velocity", "braking"),
        "maneuver": ("constant_acceleration_vs_maneuver", "constant_acceleration", "maneuver"),
        "constant_velocity": ("stationary_vs_constant_velocity", "stationary", "constant_velocity"),
    }
    return mapping.get(target, ("constant_velocity_vs_constant_acceleration", "constant_velocity", "constant_acceleration"))


def _canonical_scenario_id(record) -> str:
    mapping = {
        "boundary_v1": "endpoint_match",
        "stress_v1": "short_noisy",
        "adversarial_v1": "outlier",
        "realistic_v1": "irregular",
        "easy_v1": "easy",
    }
    return mapping.get(record.candidate.difficulty_tier, "easy")


def _record_to_executable(record) -> ExecutableTrajectory:
    run = record.execution.trajectory_run
    pair_id, class_a, class_b = _canonical_pair(record)
    scenario_id = _canonical_scenario_id(record)
    truth = run.truth_state
    return ExecutableTrajectory(
        trajectory_id=run.run_id,
        class_pair_id=pair_id,
        class_a=class_a,
        class_b=class_b,
        true_class=record.assigned_class,
        scenario_id=scenario_id,
        seed=run.seed,
        times=tuple(float(value) for value in run.times),
        measurements=tuple(float(value) for value in run.observations.get("position", ())),
        true_position=tuple(float(value) for value in truth.get("position", ())),
        true_velocity=tuple(float(value) for value in truth.get("velocity", ())),
        true_acceleration=tuple(float(value) for value in truth.get("acceleration", ())),
        measurement_dim=1,
        coordinate_frame="scalar_line",
    )


def analyze_selected_generated_corpus() -> SelectedGeneratedCorpusResult:
    qd = analyze_objective_driven_qd_archive()
    all_records = {record.candidate.candidate_id: record for record in collect_generated_corpus_records()}
    selected_records = [
        all_records[str(row["candidate_id"])]
        for row in qd.archive_elite_rows
        if str(row["candidate_id"]) in all_records
    ]
    executable_trajectories = tuple(_record_to_executable(record) for record in selected_records)
    pair_specs_map: dict[str, ExecutablePairSpec] = {}
    for trajectory in executable_trajectories:
        pair_specs_map.setdefault(
            trajectory.class_pair_id,
            ExecutablePairSpec(
                pair_id=trajectory.class_pair_id,
                class_a=trajectory.class_a,
                class_b=trajectory.class_b,
                expected_difficulty="generated",
            ),
        )
    common_result = analyze_common_trajectory_corpus(
        pair_specs=tuple(pair_specs_map.values()),
        trajectories=executable_trajectories,
        trajectories_per_case=max(len(executable_trajectories), 1),
    )

    scoring = analyze_corpus_classifier_scoring()
    feature_rows = [row for row in common_result.feature_rows if str(row["trajectory_id"]) in {trajectory.trajectory_id for trajectory in executable_trajectories}]
    classifier_rows = [row for row in scoring.candidate_score_rows if str(row["trajectory_id"]) in {trajectory.trajectory_id for trajectory in executable_trajectories}]
    posterior_rows = [row for row in scoring.posterior_rows if str(row["trajectory_id"]) in {trajectory.trajectory_id for trajectory in executable_trajectories}]

    trajectory_rows = []
    observation_rows = []
    truth_rows = []
    event_rows = []
    environment_rows = []
    class_validity_rows = []
    for record, trajectory in zip(selected_records, executable_trajectories):
        run = record.execution.trajectory_run
        trajectory_rows.append(
            {
                "trajectory_id": trajectory.trajectory_id,
                "candidate_id": record.candidate.candidate_id,
                "backend_id": record.execution.backend_id,
                "class_pair_id": trajectory.class_pair_id,
                "class_a": trajectory.class_a,
                "class_b": trajectory.class_b,
                "true_class": trajectory.true_class,
                "scenario_id": trajectory.scenario_id,
                "difficulty_tier": record.candidate.difficulty_tier,
                "label_status": record.label_status,
                "validity_score": record.validity_score,
            }
        )
        class_validity_rows.append(
            {
                "trajectory_id": trajectory.trajectory_id,
                "candidate_id": record.candidate.candidate_id,
                "target_class": record.candidate.target_class,
                "assigned_class": record.assigned_class,
                "label_status": record.label_status,
                "validity_score": record.validity_score,
            }
        )
        for index, time in enumerate(trajectory.times):
            observation_rows.append(
                {
                    "trajectory_id": trajectory.trajectory_id,
                    "time": time,
                    "measurement_position": trajectory.measurements[index],
                }
            )
            truth_rows.append(
                {
                    "trajectory_id": trajectory.trajectory_id,
                    "time": time,
                    "true_position": trajectory.true_position[index],
                    "true_velocity": trajectory.true_velocity[index],
                    "true_acceleration": trajectory.true_acceleration[index],
                }
            )
            environment_rows.append(
                {
                    "trajectory_id": trajectory.trajectory_id,
                    "time": time,
                    "environment_id": str(record.candidate.provenance.get("environment_id", "")),
                    "density_scale": record.candidate.density_scale if record.candidate.density_scale is not None else "",
                    "wind_bias": record.candidate.wind_bias if record.candidate.wind_bias is not None else "",
                    "drag_coefficient": record.candidate.drag_coefficient if record.candidate.drag_coefficient is not None else "",
                }
            )
        for event in run.events:
            event_rows.append(
                {
                    "trajectory_id": trajectory.trajectory_id,
                    "time": event.get("time", ""),
                    "event_type": event.get("event_type", ""),
                    "event_value": event.get("event_value", ""),
                }
            )

    manifest = {
        "corpus_id": "selected_generated_corpus",
        "selected_trajectory_count": len(executable_trajectories),
        "selected_pair_ids": sorted(pair_specs_map),
        "harness_experiment_name": common_result.summary.experiment_name,
        "harness_prediction_rows": len(common_result.pair_prediction_rows),
        "harness_feature_rows": len(common_result.feature_rows),
        "consumable_by_common_harness": len(common_result.pair_prediction_rows) > 0 and len(common_result.feature_rows) > 0,
    }
    report_markdown = "\n".join(
        [
            "# Selected Generated Corpus",
            "",
            "## Summary",
            f"- selected trajectories: `{len(executable_trajectories)}`",
            f"- pair specs represented: `{len(pair_specs_map)}`",
            f"- common harness prediction rows: `{len(common_result.pair_prediction_rows)}`",
            f"- common harness feature rows: `{len(common_result.feature_rows)}`",
            f"- consumable by common harness: `{manifest['consumable_by_common_harness']}`",
            "",
            "## Notes",
            "- This artifact materializes the selected generated corpus as normalized trajectory tables plus feature and classifier outputs.",
            "- Consumability is proven by routing the selected executable trajectories through the common experiment scoring path rather than by a manifest-only claim.",
        ]
    )
    return SelectedGeneratedCorpusResult(
        corpus_manifest=manifest,
        trajectory_rows=tuple(trajectory_rows),
        observation_rows=tuple(observation_rows),
        truth_state_rows=tuple(truth_rows),
        event_rows=tuple(event_rows),
        environment_rows=tuple(environment_rows),
        feature_rows=tuple(feature_rows),
        class_validity_rows=tuple(class_validity_rows),
        classifier_score_rows=tuple(classifier_rows),
        posterior_rows=tuple(posterior_rows),
        report_markdown=report_markdown,
    )


def _render_summary(rows: tuple[dict[str, Any], ...]) -> bytes:
    classes = sorted({str(row["true_class"]) for row in rows})
    counts = [sum(1 for row in rows if row["true_class"] == class_name) for class_name in classes]
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.bar(classes, counts, color="#5c7ea5")
    ax.set_title("Selected Corpus Summary")
    ax.set_ylabel("Trajectory Count")
    fig.tight_layout()

    return figure_to_png_bytes(fig, dpi=180)


def _render_validity(rows: tuple[dict[str, Any], ...]) -> bytes:
    statuses = sorted({str(row["label_status"]) for row in rows})
    counts = [sum(1 for row in rows if row["label_status"] == status) for status in statuses]
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.bar(statuses, counts, color="#b56b4d")
    ax.set_title("Selected Corpus Class-Validity Breakdown")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()

    return figure_to_png_bytes(fig, dpi=180)


def _render_score_gallery(feature_rows: tuple[dict[str, Any], ...], classifier_rows: tuple[dict[str, Any], ...]) -> bytes:
    selected_features = list(feature_rows[:8])
    selected_scores = list(classifier_rows[:8])
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.axis("off")
    ax.text(0.50, 0.96, "Feature and Classifier Score Gallery", ha="center", va="center", fontsize=11)
    y = 0.84
    for feature_row, score_row in zip(selected_features, selected_scores):
        text = (
            f"{feature_row['trajectory_id']}\n"
            f"feature_set={feature_row['feature_set_id']} accel_range={float(feature_row['acceleration_range']):.3f}\n"
            f"method={score_row['method_name']} stress={float(score_row['measured_classifier_stress']):.3f}"
        )
        ax.text(0.50, y, text, ha="center", va="center", fontsize=8, bbox={"boxstyle": "round,pad=0.3", "facecolor": "#eef4f7"})
        y -= 0.09
    fig.tight_layout()

    return figure_to_png_bytes(fig, dpi=180)
