from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

from kinematic_classifier_sandbox.utils.plotting import figure_to_png_bytes
from kinematic_classifier_sandbox.utils.plotting import plt

from ..analysis.generated_corpus_features import (
    collect_generated_corpus_records,
    generated_corpus_datasets,
)
from ..common_experiment.adapters import ExecutablePairSpec
from ..common_experiment.runner import analyze_common_trajectory_corpus
from ..corpus.exploration.objective_driven_qd_archive import analyze_objective_driven_qd_archive
from .adequacy_audit import analyze_corpus_adequacy
from .classifier_scoring import analyze_corpus_classifier_scoring
from .selected_generated_corpus_artifact_io import write_selected_generated_corpus_artifacts
from .selected_generated_corpus_contracts import (
    SelectedGeneratedCorpusResult,
)
from .selected_generated_corpus_rows import _record_to_executable


def _with_missing_boundary_fault(datasets):
    faulted = []
    for dataset in datasets:
        kept = tuple(
            trajectory
            for trajectory in dataset.trajectories
            if not (
                trajectory.true_class in {"constant_acceleration", "maneuver"}
                and dataset.tier in {"boundary", "adversarial"}
            )
        )
        faulted.append(replace(dataset, trajectories=kept))
    return tuple(faulted)


def _with_leakage_fault(datasets):
    faulted = []
    for dataset in datasets:
        updated_trajectories = []
        for trajectory in dataset.trajectories:
            class_scale = 0.02 if trajectory.true_class in {"stationary", "constant_velocity"} else 0.24
            generator_parameters = dict(trajectory.generator_parameters)
            generator_parameters["injected_leakage_fault"] = True
            updated_trajectories.append(
                replace(
                    trajectory,
                    measurement_std=class_scale,
                    generator_parameters=generator_parameters,
                )
            )
        faulted.append(replace(dataset, trajectories=tuple(updated_trajectories)))
    return tuple(faulted)


def _regression_rows(base_datasets):
    rows = []
    for regression_id, builder in (
        ("missing_boundary", _with_missing_boundary_fault),
        ("leakage", _with_leakage_fault),
    ):
        result = analyze_corpus_adequacy(datasets=builder(base_datasets))
        rows.append(
            {
                "regression_id": regression_id,
                "overall_status": result.summary.overall_status,
                "overall_pass": result.summary.overall_pass,
                "q_corpus": result.summary.q_corpus,
                "leakage_penalty": result.summary.leakage_penalty,
                "red_count": result.summary.red_count,
                "recommendation_count": result.summary.recommendation_count,
            }
        )
    return tuple(rows)


def analyze_selected_generated_corpus() -> SelectedGeneratedCorpusResult:
    qd = analyze_objective_driven_qd_archive()
    all_records = {record.candidate.candidate_id: record for record in collect_generated_corpus_records()}
    selected_records = [
        all_records[str(row["candidate_id"])]
        for row in qd.archive_elite_rows
        if str(row["candidate_id"]) in all_records
    ]
    selected_datasets = generated_corpus_datasets(tuple(selected_records))
    adequacy = analyze_corpus_adequacy(datasets=selected_datasets)
    regression_rows = _regression_rows(selected_datasets)
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
        "adequacy_overall_status": adequacy.summary.overall_status,
        "adequacy_q_corpus": adequacy.summary.q_corpus,
        "adequacy_leakage_penalty": adequacy.summary.leakage_penalty,
        "adequacy_recommendation_count": adequacy.summary.recommendation_count,
        "regression_checks": list(regression_rows),
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
            f"- adequacy overall status: `{adequacy.summary.overall_status}`",
            f"- adequacy q_corpus: `{adequacy.summary.q_corpus:.3f}`",
            f"- adequacy leakage penalty: `{adequacy.summary.leakage_penalty:.3f}`",
            "",
            "## Closed-Loop Adequacy",
            f"- recommendations emitted: `{adequacy.summary.recommendation_count}`",
            f"- regression checks run: `{len(regression_rows)}`",
            "",
            "## Regression Checks",
            *[
                f"- `{row['regression_id']}` -> status=`{row['overall_status']}`, leakage=`{row['leakage_penalty']:.3f}`, q_corpus=`{row['q_corpus']:.3f}`"
                for row in regression_rows
            ],
            "",
            "## Notes",
            "- This artifact materializes the selected generated corpus as normalized trajectory tables plus feature and classifier outputs.",
            "- Consumability is proven by routing the selected executable trajectories through the common experiment scoring path rather than by a manifest-only claim.",
            "- Corpus selection is followed immediately by an adequacy rerun so boundary coverage, leakage, and recommendations remain attached to the selected corpus itself.",
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
        adequacy_result=adequacy,
        adequacy_summary={
            "summary": asdict(adequacy.summary),
            "scorecard": asdict(adequacy.scorecard),
        },
        adequacy_recommendations=adequacy.recommendations,
        regression_rows=regression_rows,
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
