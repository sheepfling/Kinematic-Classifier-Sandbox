from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev

from kinematic_classifier_sandbox.analysis.common_dataset_comparison import (
    _class_expected_position,
    _kalman_predict,
    _rocket_proxy_predict,
    _windowed_predict,
    generate_shared_dynamics_dataset,
)
from kinematic_classifier_sandbox.analysis.common_dataset_comparison_contracts import (
    SharedDynamicsTrajectory,
)
from kinematic_classifier_sandbox.analysis.optional_external_backends import (
    fit_archive_classifier_with_outcome,
)
from kinematic_classifier_sandbox.analysis.tsc_archive_backend_smoke import (
    analyze_tsc_archive_backend_smoke,
)
from kinematic_classifier_sandbox.corpus.trajectory_exploration.comparison_surface import (
    write_comparison_summary_csv,
)
from kinematic_classifier_sandbox.utils.io import write_csv
from kinematic_classifier_sandbox.utils.plotting import _figure_to_png
from kinematic_classifier_sandbox.utils.plotting import plt


CLASS_NAMES = ("constant_velocity", "constant_acceleration")


@dataclass(frozen=True, slots=True)
class TSCArchivePredictionRow:
    trajectory_id: str
    scenario_name: str
    true_class: str
    split: str
    method_name: str
    backend_name: str
    predicted_class: str
    confidence: float
    truth_probability: float
    negative_log_likelihood: float


@dataclass(frozen=True, slots=True)
class TSCArchiveMetricRow:
    method_name: str
    backend_name: str
    overall_accuracy: float
    test_accuracy: float
    short_noisy_accuracy: float
    outlier_accuracy: float
    endpoint_match_accuracy: float
    test_nll: float
    test_ece: float
    seed_stability_read: str
    claim_level: str


@dataclass(frozen=True, slots=True)
class TSCArchiveSeedSweepRow:
    method_name: str
    claim_level: str
    seed_count: int
    mean_test_accuracy: float
    std_test_accuracy: float
    mean_test_nll: float
    std_test_nll: float
    mean_test_ece: float
    std_test_ece: float
    backend_names: str
    supported_backend_fraction: float
    stability_read: str


@dataclass(frozen=True, slots=True)
class TSCArchiveFrontierResult:
    prediction_rows: tuple[TSCArchivePredictionRow, ...]
    metric_rows: tuple[TSCArchiveMetricRow, ...]
    seed_sweep_rows: tuple[TSCArchiveSeedSweepRow, ...]
    metrics: dict[str, float | str | int]


@dataclass(frozen=True, slots=True)
class TSCArchiveFrontierArtifacts:
    run_dir: Path
    prediction_summary_path: Path
    metric_summary_path: Path
    seed_sweep_path: Path
    summary_path: Path
    metrics_path: Path
    report_path: Path
    decision_card_path: Path
    plot_paths: tuple[Path, ...]


def _trajectory_split(trajectory: SharedDynamicsTrajectory) -> str:
    return "train" if int(trajectory.trajectory_id.rsplit("_", 1)[-1]) < 4 else "test"


def _accuracy(rows: list[TSCArchivePredictionRow], *, scenario_name: str | None = None, split: str | None = None) -> float:
    selected = rows
    if scenario_name is not None:
        selected = [row for row in selected if row.scenario_name == scenario_name]
    if split is not None:
        selected = [row for row in selected if row.split == split]
    return sum(1.0 if row.predicted_class == row.true_class else 0.0 for row in selected) / max(len(selected), 1)


def _mean_nll(rows: list[TSCArchivePredictionRow], *, scenario_name: str | None = None, split: str | None = None) -> float:
    selected = rows
    if scenario_name is not None:
        selected = [row for row in selected if row.scenario_name == scenario_name]
    if split is not None:
        selected = [row for row in selected if row.split == split]
    return sum(row.negative_log_likelihood for row in selected) / max(len(selected), 1)


def _test_probability_panel(rows: list[TSCArchivePredictionRow]) -> tuple[list[list[float]], list[int]]:
    probability_rows: list[list[float]] = []
    label_rows: list[int] = []
    for row in rows:
        if row.predicted_class == CLASS_NAMES[0]:
            probability_rows.append([row.confidence, 1.0 - row.confidence])
        else:
            probability_rows.append([1.0 - row.confidence, row.confidence])
        label_rows.append(CLASS_NAMES.index(row.true_class))
    return probability_rows, label_rows


def _evaluate_ece(probability_rows: list[list[float]], labels: list[int], *, bins: int = 10) -> float:
    if not probability_rows:
        return 0.0
    total = max(len(labels), 1)
    ece = 0.0
    confidences = [max(row) for row in probability_rows]
    predictions = [0 if row[0] >= row[1] else 1 for row in probability_rows]
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        selected = [
            index
            for index, confidence in enumerate(confidences)
            if lower <= confidence < upper or (bin_index == bins - 1 and confidence == upper)
        ]
        if not selected:
            continue
        mean_confidence = sum(confidences[index] for index in selected) / len(selected)
        mean_accuracy = sum(1.0 if predictions[index] == labels[index] else 0.0 for index in selected) / len(selected)
        ece += abs(mean_confidence - mean_accuracy) * (len(selected) / total)
    return float(ece)


def _ensemble_predict(trajectory: SharedDynamicsTrajectory) -> tuple[str, float]:
    members = (
        _windowed_predict(trajectory, robust=True),
        _rocket_proxy_predict(trajectory),
        _kalman_predict(trajectory),
    )
    scores = {class_name: 0.0 for class_name in CLASS_NAMES}
    for run in members:
        for class_name, probability in run.final_weights.items():
            scores[class_name] += probability
    total = max(sum(scores.values()), 1.0e-12)
    normalized = {class_name: value / total for class_name, value in scores.items()}
    predicted = max(normalized, key=normalized.get)
    return predicted, float(normalized[predicted])


def _interval_features(values: tuple[float, ...]) -> tuple[float, ...]:
    midpoint = len(values) // 2
    early = values[: midpoint + 1]
    late = values[midpoint:]
    center_start = max(0, midpoint - 1)
    center_stop = min(len(values), midpoint + 2)
    center = values[center_start:center_stop]
    early_slope = (early[-1] - early[0]) / max(len(early) - 1, 1)
    late_slope = (late[-1] - late[0]) / max(len(late) - 1, 1)
    center_range = max(center) - min(center)
    full_range = max(values) - min(values)
    return (
        float(early_slope),
        float(late_slope),
        float(center_range),
        float(full_range),
    )


def _local_drcif_predict(trajectory: SharedDynamicsTrajectory) -> tuple[str, float]:
    observed = _interval_features(trajectory.measurements)
    log_scores: dict[str, float] = {}
    for class_name in CLASS_NAMES:
        expected_series = tuple(
            _class_expected_position(class_name, time_value, trajectory.scenario_name)
            for time_value in trajectory.times
        )
        reference = _interval_features(expected_series)
        squared_distance = sum(
            (observed_value - reference_value) ** 2
            for observed_value, reference_value in zip(observed, reference, strict=True)
        )
        log_scores[class_name] = -0.5 * squared_distance
    return _normalize_log_scores(log_scores)


def _symbolic_word(values: tuple[float, ...]) -> tuple[str, ...]:
    diffs = [values[index + 1] - values[index] for index in range(len(values) - 1)]
    mean_diff = sum(diffs) / max(len(diffs), 1)
    word: list[str] = []
    for diff in diffs:
        centered = diff - mean_diff
        if centered > 0.15:
            word.append("U")
        elif centered < -0.15:
            word.append("D")
        else:
            word.append("F")
    return tuple(word)


def _word_distance(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    shared = min(len(left), len(right))
    mismatch = sum(1.0 if left[index] != right[index] else 0.0 for index in range(shared))
    tail = abs(len(left) - len(right))
    return mismatch + tail


def _local_dictionary_predict(trajectory: SharedDynamicsTrajectory) -> tuple[str, float]:
    observed = _symbolic_word(trajectory.measurements)
    log_scores: dict[str, float] = {}
    for class_name in CLASS_NAMES:
        expected_series = tuple(
            _class_expected_position(class_name, time_value, trajectory.scenario_name)
            for time_value in trajectory.times
        )
        reference = _symbolic_word(expected_series)
        distance = _word_distance(observed, reference)
        log_scores[class_name] = -distance
    return _normalize_log_scores(log_scores)


def _normalize_log_scores(log_scores: dict[str, float]) -> tuple[str, float]:
    score_max = max(log_scores.values())
    unnormalized = {label: pow(2.718281828459045, value - score_max) for label, value in log_scores.items()}
    total = max(sum(unnormalized.values()), 1.0e-12)
    normalized = {label: value / total for label, value in unnormalized.items()}
    predicted = max(normalized, key=normalized.get)
    return predicted, float(normalized[predicted])


def _analyze_single_seed_frontier(
    *,
    seed: int = 1009,
    trajectories_per_case: int = 8,
    backend_smoke_timeout_seconds: float = 20.0,
) -> TSCArchiveFrontierResult:
    trajectories = generate_shared_dynamics_dataset(seed=seed, trajectories_per_case=trajectories_per_case)
    train_rows = tuple(trajectory for trajectory in trajectories if _trajectory_split(trajectory) == "train")
    smoke_result = analyze_tsc_archive_backend_smoke(timeout_seconds=backend_smoke_timeout_seconds)
    smoke_row_map = {row.method_family: row for row in smoke_result.rows}

    def _gated_fit(method_family: str):
        smoke_row = smoke_row_map[method_family]
        if smoke_row.succeeded:
            return smoke_row.detail, fit_archive_classifier_with_outcome(method_family, train_rows, class_names=CLASS_NAMES)
        return smoke_row.detail, None

    minirocket_smoke_detail, minirocket_fit_outcome = _gated_fit("minirocket_family")
    drcif_smoke_detail, drcif_fit_outcome = _gated_fit("drcif_interval_forests")
    dictionary_smoke_detail, dictionary_fit_outcome = _gated_fit("dictionary_tde_family")
    hive_smoke_detail, hive_fit_outcome = _gated_fit("hive_cote")

    minirocket_outcome = minirocket_fit_outcome
    drcif_outcome = drcif_fit_outcome
    dictionary_outcome = dictionary_fit_outcome
    hive_outcome = hive_fit_outcome
    minirocket_adapter = minirocket_outcome.adapter if minirocket_outcome is not None else None
    drcif_adapter = drcif_outcome.adapter if drcif_outcome is not None else None
    dictionary_adapter = dictionary_outcome.adapter if dictionary_outcome is not None else None
    hive_adapter = hive_outcome.adapter if hive_outcome is not None else None

    backend_lookup = {
        "minirocket_family": minirocket_outcome.backend_name if minirocket_adapter is not None else "local_proxy",
        "drcif_interval_forests": drcif_outcome.backend_name if drcif_adapter is not None else "local_proxy",
        "dictionary_tde_family": dictionary_outcome.backend_name if dictionary_adapter is not None else "local_proxy",
        "hive_cote": hive_outcome.backend_name if hive_adapter is not None else "local_proxy",
    }
    minirocket_predictions = (
        minirocket_adapter.predict_many(trajectories) if minirocket_adapter is not None else None
    )
    drcif_predictions = drcif_adapter.predict_many(trajectories) if drcif_adapter is not None else None
    dictionary_predictions = (
        dictionary_adapter.predict_many(trajectories) if dictionary_adapter is not None else None
    )
    hive_predictions = hive_adapter.predict_many(trajectories) if hive_adapter is not None else None

    prediction_rows: list[TSCArchivePredictionRow] = []
    for trajectory_index, trajectory in enumerate(trajectories):
        split = _trajectory_split(trajectory)
        windowed_run = _windowed_predict(trajectory, robust=True)
        kalman_run = _kalman_predict(trajectory)
        minirocket_predicted, minirocket_confidence = (
            minirocket_predictions[trajectory_index]
            if minirocket_predictions is not None
            else (lambda run=_rocket_proxy_predict(trajectory): (run.final_predicted_class, run.final_confidence))()
        )
        drcif_predicted, drcif_confidence = (
            drcif_predictions[trajectory_index] if drcif_predictions is not None else _local_drcif_predict(trajectory)
        )
        dictionary_predicted, dictionary_confidence = (
            dictionary_predictions[trajectory_index]
            if dictionary_predictions is not None
            else _local_dictionary_predict(trajectory)
        )
        hive_predicted, hive_confidence = (
            hive_predictions[trajectory_index] if hive_predictions is not None else _ensemble_predict(trajectory)
        )
        for method_name, backend_name, predicted_class, confidence in (
            ("windowed_robust", "baseline", windowed_run.final_predicted_class, windowed_run.final_confidence),
            ("kalman_bank", "baseline", kalman_run.final_predicted_class, kalman_run.final_confidence),
            ("minirocket_family", backend_lookup["minirocket_family"], minirocket_predicted, minirocket_confidence),
            ("drcif_interval_forests", backend_lookup["drcif_interval_forests"], drcif_predicted, drcif_confidence),
            ("dictionary_tde_family", backend_lookup["dictionary_tde_family"], dictionary_predicted, dictionary_confidence),
            ("hive_cote", backend_lookup["hive_cote"], hive_predicted, hive_confidence),
        ):
            prediction_rows.append(
                TSCArchivePredictionRow(
                    trajectory_id=trajectory.trajectory_id,
                    scenario_name=trajectory.scenario_name,
                    true_class=trajectory.true_class,
                    split=split,
                    method_name=method_name,
                    backend_name=backend_name,
                    predicted_class=predicted_class,
                    confidence=float(confidence),
                    truth_probability=float(confidence if predicted_class == trajectory.true_class else 1.0 - confidence),
                    negative_log_likelihood=float(
                        -math.log(max(confidence if predicted_class == trajectory.true_class else 1.0 - confidence, 1.0e-12))
                    ),
                )
            )

    metric_rows: list[TSCArchiveMetricRow] = []
    claim_level_lookup = {
        "windowed_robust": "baseline",
        "kalman_bank": "baseline",
        "minirocket_family": "trained_external" if minirocket_adapter is not None else "local_proxy",
        "drcif_interval_forests": "trained_external" if drcif_adapter is not None else "local_proxy",
        "dictionary_tde_family": "trained_external" if dictionary_adapter is not None else "local_proxy",
        "hive_cote": "trained_external" if hive_adapter is not None else "local_proxy",
    }
    for method_name in (
        "windowed_robust",
        "kalman_bank",
        "minirocket_family",
        "drcif_interval_forests",
        "dictionary_tde_family",
        "hive_cote",
    ):
        method_rows = [row for row in prediction_rows if row.method_name == method_name]
        test_rows = [row for row in method_rows if row.split == "test"]
        probability_rows, label_rows = _test_probability_panel(test_rows)
        metric_rows.append(
            TSCArchiveMetricRow(
                method_name=method_name,
                backend_name=method_rows[0].backend_name,
                overall_accuracy=_accuracy(method_rows),
                test_accuracy=_accuracy(test_rows),
                short_noisy_accuracy=_accuracy(test_rows, scenario_name="short_noisy"),
                outlier_accuracy=_accuracy(test_rows, scenario_name="outlier"),
                endpoint_match_accuracy=_accuracy(test_rows, scenario_name="endpoint_match"),
                test_nll=_mean_nll(test_rows),
                test_ece=_evaluate_ece(probability_rows, label_rows),
                seed_stability_read="not_run",
                claim_level=claim_level_lookup[method_name],
            )
        )

    row_map = {row.method_name: row for row in metric_rows}
    archive_method_names = (
        "minirocket_family",
        "drcif_interval_forests",
        "dictionary_tde_family",
        "hive_cote",
    )
    external_family_count = sum(1 for name in archive_method_names if row_map[name].backend_name != "local_proxy")
    fallback_family_count = len(archive_method_names) - external_family_count
    attempted_family_count = sum(
        1
        for outcome in (minirocket_outcome, drcif_outcome, dictionary_outcome, hive_outcome)
        if outcome is not None and outcome.attempted
    )
    failed_external_family_count = sum(
        1
        for outcome in (minirocket_outcome, drcif_outcome, dictionary_outcome, hive_outcome)
        if outcome is not None and outcome.attempted and not outcome.succeeded
    )
    integration_read = (
        "all_external"
        if external_family_count == len(archive_method_names)
        else "mixed_external_and_fallback"
        if external_family_count > 0
        else "wrapper_stage_only"
    )
    metrics: dict[str, float | str | int] = {
        "study_id": "tsc_archive_baseline_frontier_v1",
        "seed": seed,
        "trajectory_count": len(trajectories),
        "train_count": len(train_rows),
        "test_count": len(trajectories) - len(train_rows),
        "minirocket_test_accuracy": row_map["minirocket_family"].test_accuracy,
        "drcif_test_accuracy": row_map["drcif_interval_forests"].test_accuracy,
        "dictionary_test_accuracy": row_map["dictionary_tde_family"].test_accuracy,
        "hive_test_accuracy": row_map["hive_cote"].test_accuracy,
        "kalman_test_accuracy": row_map["kalman_bank"].test_accuracy,
        "minirocket_backend": row_map["minirocket_family"].backend_name,
        "drcif_backend": row_map["drcif_interval_forests"].backend_name,
        "dictionary_backend": row_map["dictionary_tde_family"].backend_name,
        "hive_backend": row_map["hive_cote"].backend_name,
        "minirocket_external_available": int(smoke_row_map["minirocket_family"].available),
        "drcif_external_available": int(smoke_row_map["drcif_interval_forests"].available),
        "dictionary_external_available": int(smoke_row_map["dictionary_tde_family"].available),
        "hive_external_available": int(smoke_row_map["hive_cote"].available),
        "minirocket_external_attempted": int(minirocket_outcome.attempted if minirocket_outcome is not None else False),
        "drcif_external_attempted": int(drcif_outcome.attempted if drcif_outcome is not None else False),
        "dictionary_external_attempted": int(dictionary_outcome.attempted if dictionary_outcome is not None else False),
        "hive_external_attempted": int(hive_outcome.attempted if hive_outcome is not None else False),
        "minirocket_external_succeeded": int(minirocket_outcome.succeeded if minirocket_outcome is not None else False),
        "drcif_external_succeeded": int(drcif_outcome.succeeded if drcif_outcome is not None else False),
        "dictionary_external_succeeded": int(dictionary_outcome.succeeded if dictionary_outcome is not None else False),
        "hive_external_succeeded": int(hive_outcome.succeeded if hive_outcome is not None else False),
        "minirocket_smoke_detail": minirocket_smoke_detail,
        "drcif_smoke_detail": drcif_smoke_detail,
        "dictionary_smoke_detail": dictionary_smoke_detail,
        "hive_smoke_detail": hive_smoke_detail,
        "minirocket_backend_detail": minirocket_outcome.detail if minirocket_outcome is not None else minirocket_smoke_detail,
        "drcif_backend_detail": drcif_outcome.detail if drcif_outcome is not None else drcif_smoke_detail,
        "dictionary_backend_detail": dictionary_outcome.detail if dictionary_outcome is not None else dictionary_smoke_detail,
        "hive_backend_detail": hive_outcome.detail if hive_outcome is not None else hive_smoke_detail,
        "archive_smoke_integration_read": smoke_result.metrics["integration_read"],
        "archive_attempted_family_count": attempted_family_count,
        "archive_external_family_count": external_family_count,
        "archive_fallback_family_count": fallback_family_count,
        "archive_failed_external_family_count": failed_external_family_count,
        "archive_integration_read": integration_read,
        "promotion_decision": (
            "promote_faithful_archive_wrappers"
            if external_family_count == len(archive_method_names)
            else "record_partial_external_execution_keep_gate_closed"
            if external_family_count > 0
            else "hold_modern_tsc_at_optional_wrapper_stage"
        ),
    }
    return TSCArchiveFrontierResult(
        prediction_rows=tuple(prediction_rows),
        metric_rows=tuple(metric_rows),
        seed_sweep_rows=(),
        metrics=metrics,
    )


def analyze_tsc_archive_baseline_frontier(
    *,
    seed: int = 1009,
    trajectories_per_case: int = 8,
    backend_smoke_timeout_seconds: float = 20.0,
    seed_sweep: tuple[int, ...] | None = None,
) -> TSCArchiveFrontierResult:
    resolved_seed_sweep = seed_sweep or (seed, seed + 1)
    single_seed_results: list[TSCArchiveFrontierResult] = []
    primary_result: TSCArchiveFrontierResult | None = None
    for sweep_seed in resolved_seed_sweep:
        result = _analyze_single_seed_frontier(
            seed=sweep_seed,
            trajectories_per_case=trajectories_per_case,
            backend_smoke_timeout_seconds=backend_smoke_timeout_seconds,
        )
        single_seed_results.append(result)
        if sweep_seed == seed:
            primary_result = result
    if primary_result is None:
        primary_result = single_seed_results[0]

    method_names = tuple(row.method_name for row in primary_result.metric_rows)
    metric_rows: list[TSCArchiveMetricRow] = []
    seed_sweep_rows: list[TSCArchiveSeedSweepRow] = []
    stability_pass_count = 0
    calibration_pass_count = 0
    for method_name in method_names:
        per_seed_metric_rows = [
            next(row for row in result.metric_rows if row.method_name == method_name)
            for result in single_seed_results
        ]
        accuracy_values = [row.test_accuracy for row in per_seed_metric_rows]
        nll_values = [row.test_nll for row in per_seed_metric_rows]
        ece_values = [row.test_ece for row in per_seed_metric_rows]
        backend_names = sorted({row.backend_name for row in per_seed_metric_rows})
        supported_backend_fraction = sum(1.0 if row.backend_name != "local_proxy" else 0.0 for row in per_seed_metric_rows) / max(
            len(per_seed_metric_rows),
            1,
        )
        stability_read = (
            "narrow_seed_sweep_pass"
            if pstdev(accuracy_values) <= 0.20 and mean(ece_values) <= 0.35
            else "narrow_seed_sweep_flags_instability"
        )
        if stability_read == "narrow_seed_sweep_pass":
            stability_pass_count += 1
        if mean(ece_values) <= 0.35:
            calibration_pass_count += 1
        primary_metric_row = next(row for row in primary_result.metric_rows if row.method_name == method_name)
        metric_rows.append(
            TSCArchiveMetricRow(
                method_name=primary_metric_row.method_name,
                backend_name=primary_metric_row.backend_name,
                overall_accuracy=primary_metric_row.overall_accuracy,
                test_accuracy=primary_metric_row.test_accuracy,
                short_noisy_accuracy=primary_metric_row.short_noisy_accuracy,
                outlier_accuracy=primary_metric_row.outlier_accuracy,
                endpoint_match_accuracy=primary_metric_row.endpoint_match_accuracy,
                test_nll=primary_metric_row.test_nll,
                test_ece=primary_metric_row.test_ece,
                seed_stability_read=stability_read,
                claim_level=primary_metric_row.claim_level,
            )
        )
        seed_sweep_rows.append(
            TSCArchiveSeedSweepRow(
                method_name=method_name,
                claim_level=primary_metric_row.claim_level,
                seed_count=len(per_seed_metric_rows),
                mean_test_accuracy=float(mean(accuracy_values)),
                std_test_accuracy=float(pstdev(accuracy_values)),
                mean_test_nll=float(mean(nll_values)),
                std_test_nll=float(pstdev(nll_values)),
                mean_test_ece=float(mean(ece_values)),
                std_test_ece=float(pstdev(ece_values)),
                backend_names=",".join(backend_names),
                supported_backend_fraction=float(supported_backend_fraction),
                stability_read=stability_read,
            )
        )

    metrics = dict(primary_result.metrics)
    metrics["archive_seed_count"] = len(resolved_seed_sweep)
    metrics["archive_seed_values"] = ",".join(str(value) for value in resolved_seed_sweep)
    metrics["archive_seed_stability_pass_count"] = stability_pass_count
    metrics["archive_calibration_pass_count"] = calibration_pass_count
    metrics["archive_seed_robustness_read"] = (
        "narrow_seed_sweep_passes"
        if stability_pass_count == len(seed_sweep_rows)
        else "narrow_seed_sweep_flags_instability"
    )
    metrics["archive_calibration_read"] = (
        "all_methods_within_bounded_binary_ece_band"
        if calibration_pass_count == len(seed_sweep_rows)
        else "bounded_binary_calibration_gaps_present"
    )
    metrics["next_gate_decision"] = (
        "ready_for_named_witness_comparison"
        if metrics["archive_external_family_count"] == 4 and metrics["archive_seed_robustness_read"] == "narrow_seed_sweep_passes"
        else "keep_generic_tsc_gate_closed"
    )

    return TSCArchiveFrontierResult(
        prediction_rows=primary_result.prediction_rows,
        metric_rows=tuple(metric_rows),
        seed_sweep_rows=tuple(seed_sweep_rows),
        metrics=metrics,
    )


def _render_overall_accuracy(result: TSCArchiveFrontierResult):
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    labels = [row.method_name for row in result.metric_rows]
    values = [row.test_accuracy for row in result.metric_rows]
    colors = ("#9ca3af", "#2563eb", "#16a34a", "#0891b2", "#be123c", "#7c3aed")
    ax.bar(range(len(labels)), values, color=colors, width=0.65)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("test accuracy")
    ax.set_title("Modern TSC Archive Frontier", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _render_scenario_slice(result: TSCArchiveFrontierResult):
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    labels = [row.method_name for row in result.metric_rows]
    x = list(range(len(labels)))
    width = 0.22
    short_noisy = [row.short_noisy_accuracy for row in result.metric_rows]
    outlier = [row.outlier_accuracy for row in result.metric_rows]
    endpoint = [row.endpoint_match_accuracy for row in result.metric_rows]
    ax.bar([value - width for value in x], short_noisy, width=width, label="short_noisy", color="#dc2626")
    ax.bar(x, outlier, width=width, label="outlier", color="#d97706")
    ax.bar([value + width for value in x], endpoint, width=width, label="endpoint_match", color="#2563eb")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title("Scenario Slice: Archive Methods", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _render_calibration_frontier(result: TSCArchiveFrontierResult):
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    labels = [row.method_name for row in result.metric_rows]
    x = list(range(len(labels)))
    width = 0.35
    nll_values = [row.test_nll for row in result.metric_rows]
    ece_values = [row.test_ece for row in result.metric_rows]
    ax.bar([value - width / 2 for value in x], nll_values, width=width, label="test_nll", color="#2563eb")
    ax.bar([value + width / 2 for value in x], ece_values, width=width, label="test_ece", color="#d97706")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("metric value")
    ax.set_title("Archive Calibration Surface", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _render_seed_stability(result: TSCArchiveFrontierResult):
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    labels = [row.method_name for row in result.seed_sweep_rows]
    x = list(range(len(labels)))
    width = 0.35
    mean_accuracy = [row.mean_test_accuracy for row in result.seed_sweep_rows]
    std_accuracy = [row.std_test_accuracy for row in result.seed_sweep_rows]
    ax.bar([value - width / 2 for value in x], mean_accuracy, width=width, label="mean_test_accuracy", color="#16a34a")
    ax.bar([value + width / 2 for value in x], std_accuracy, width=width, label="std_test_accuracy", color="#be123c")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("value")
    ax.set_title("Archive Seed Stability", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def write_tsc_archive_baseline_frontier_artifacts(
    output_dir: str | Path,
    *,
    result: TSCArchiveFrontierResult | None = None,
    seed: int = 1009,
    trajectories_per_case: int = 8,
    backend_smoke_timeout_seconds: float = 20.0,
    seed_sweep: tuple[int, ...] | None = None,
) -> TSCArchiveFrontierArtifacts:
    payload = result or analyze_tsc_archive_baseline_frontier(
        seed=seed,
        trajectories_per_case=trajectories_per_case,
        backend_smoke_timeout_seconds=backend_smoke_timeout_seconds,
        seed_sweep=seed_sweep,
    )
    run_dir = Path(output_dir) / "tsc_archive_baseline_frontier_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    prediction_summary_path = run_dir / "prediction_summary.csv"
    metric_summary_path = run_dir / "metric_summary.csv"
    seed_sweep_path = run_dir / "seed_sweep_summary.csv"
    summary_path = run_dir / "summary.csv"
    metrics_path = run_dir / "metrics.csv"
    report_path = run_dir / "tsc_archive_baseline_frontier_report.md"
    decision_card_path = run_dir / "decision_card.md"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    overall_plot_path = plots_dir / "overall_accuracy.png"
    scenario_plot_path = plots_dir / "scenario_slice_accuracy.png"
    calibration_plot_path = plots_dir / "calibration_surface.png"
    stability_plot_path = plots_dir / "seed_stability.png"

    write_csv(
        prediction_summary_path,
        [asdict(row) for row in payload.prediction_rows],
        list(TSCArchivePredictionRow.__dataclass_fields__.keys()),
    )
    write_csv(
        metric_summary_path,
        [asdict(row) for row in payload.metric_rows],
        list(TSCArchiveMetricRow.__dataclass_fields__.keys()),
    )
    write_csv(
        seed_sweep_path,
        [asdict(row) for row in payload.seed_sweep_rows],
        list(TSCArchiveSeedSweepRow.__dataclass_fields__.keys()),
    )
    write_comparison_summary_csv(run_dir, [asdict(row) for row in payload.metric_rows], filename="summary.csv")
    write_csv(metrics_path, [payload.metrics], list(payload.metrics.keys()))

    report_lines = [
        "# TSC Archive Baseline Frontier",
        "",
        "- Study: `tsc_archive_baseline_frontier_v1`",
        "- Methods: `minirocket_family`, `drcif_interval_forests`, `dictionary_tde_family`, `hive_cote`",
        "- Baselines: `windowed_robust`, `kalman_bank`",
        "",
        "## Claim Boundary",
        "",
        "This packet now records backend provenance per archive family.",
        f"If the tiny backend smoke packet says a family is timing out or unavailable, the shared frontier keeps that family on fallback instead of retrying the same failing path.",
        "If a family-appropriate optional backend is installed, the family can run through a trained external wrapper.",
        "If not, the frontier stays on local fallbacks and the promotion gate remains closed.",
        "",
        f"- smoke integration read: `{payload.metrics['archive_smoke_integration_read']}`",
        f"- integration read: `{payload.metrics['archive_integration_read']}`",
        f"- attempted family count: `{payload.metrics['archive_attempted_family_count']}`",
        f"- external family count: `{payload.metrics['archive_external_family_count']}`",
        f"- fallback family count: `{payload.metrics['archive_fallback_family_count']}`",
        f"- failed external family count: `{payload.metrics['archive_failed_external_family_count']}`",
        f"- decision: `{payload.metrics['promotion_decision']}`",
        f"- seed sweep: `{payload.metrics['archive_seed_values']}`",
        f"- seed robustness read: `{payload.metrics['archive_seed_robustness_read']}`",
        f"- calibration read: `{payload.metrics['archive_calibration_read']}`",
        f"- next gate: `{payload.metrics['next_gate_decision']}`",
        "",
        "## Backend Provenance",
        "",
        f"- `minirocket_family`: smoke=`{payload.metrics['minirocket_smoke_detail']}` frontier=`{payload.metrics['minirocket_backend']}` / `{payload.metrics['minirocket_backend_detail']}`",
        f"- `drcif_interval_forests`: smoke=`{payload.metrics['drcif_smoke_detail']}` frontier=`{payload.metrics['drcif_backend']}` / `{payload.metrics['drcif_backend_detail']}`",
        f"- `dictionary_tde_family`: smoke=`{payload.metrics['dictionary_smoke_detail']}` frontier=`{payload.metrics['dictionary_backend']}` / `{payload.metrics['dictionary_backend_detail']}`",
        f"- `hive_cote`: smoke=`{payload.metrics['hive_smoke_detail']}` frontier=`{payload.metrics['hive_backend']}` / `{payload.metrics['hive_backend_detail']}`",
        "",
        "## Bounded Robustness Read",
        "",
        "The shared archive frontier now also carries a narrow seed sweep and binary-calibration read.",
        "This is still not a named witness promotion packet for the archive families.",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    decision_lines = [
        "# Decision Card",
        "",
        "- Candidate methods: `minirocket_family`, `drcif_interval_forests`, `dictionary_tde_family`, `hive_cote`",
        "- Implementation state: optional family-appropriate wrappers with local fallback backends",
        f"- Integration read: `{payload.metrics['archive_integration_read']}`",
        f"- Seed robustness read: `{payload.metrics['archive_seed_robustness_read']}`",
        f"- Calibration read: `{payload.metrics['archive_calibration_read']}`",
        f"- Decision: `{payload.metrics['promotion_decision']}`",
        f"- Next gate: `{payload.metrics['next_gate_decision']}`",
        "- Promotion rule: `do not promote the family lane from wrapper execution alone; require external execution, bounded robustness, calibration evidence, and named witness comparison`",
    ]
    decision_card_path.write_text("\n".join(decision_lines) + "\n", encoding="utf-8")

    overall_plot_path.write_bytes(_figure_to_png(_render_overall_accuracy(payload)))
    scenario_plot_path.write_bytes(_figure_to_png(_render_scenario_slice(payload)))
    calibration_plot_path.write_bytes(_figure_to_png(_render_calibration_frontier(payload)))
    stability_plot_path.write_bytes(_figure_to_png(_render_seed_stability(payload)))

    return TSCArchiveFrontierArtifacts(
        run_dir=run_dir,
        prediction_summary_path=prediction_summary_path,
        metric_summary_path=metric_summary_path,
        seed_sweep_path=seed_sweep_path,
        summary_path=summary_path,
        metrics_path=metrics_path,
        report_path=report_path,
        decision_card_path=decision_card_path,
        plot_paths=(overall_plot_path, scenario_plot_path, calibration_plot_path, stability_plot_path),
    )


__all__ = [
    "TSCArchiveFrontierArtifacts",
    "TSCArchiveFrontierResult",
    "TSCArchiveMetricRow",
    "TSCArchivePredictionRow",
    "TSCArchiveSeedSweepRow",
    "analyze_tsc_archive_baseline_frontier",
    "write_tsc_archive_baseline_frontier_artifacts",
]
