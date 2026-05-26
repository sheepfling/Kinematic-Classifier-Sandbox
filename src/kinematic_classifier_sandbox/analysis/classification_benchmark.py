from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

TRun = TypeVar("TRun")
TStep = TypeVar("TStep")


@dataclass(frozen=True, slots=True)
class ClassificationOutcomeSummary:
    total_runs: int
    overall_accuracy: float
    transient_accuracy: float
    terminal_accuracy: float
    per_class_accuracy: dict[str, float]
    confusion_counts: dict[str, dict[str, int]]
    transient_confusion_counts: dict[str, dict[str, int]]
    terminal_confusion_counts: dict[str, dict[str, int]]
    scenario_confusion_counts: dict[str, dict[str, int]]


@dataclass(frozen=True, slots=True)
class ClassificationFeatureSummary:
    class_feature_detection_counts: dict[str, dict[str, int]]
    feature_confusion_counts: dict[str, dict[str, int]]
    true_feature_predicted_class_counts: dict[str, dict[str, int]]
    detected_feature_predicted_class_counts: dict[str, dict[str, int]]
    entropy_mean_by_step: tuple[float, ...]
    mean_feature_probability_by_step: dict[str, tuple[float, ...]]


def summarize_classification_outcomes(
    runs: Sequence[TRun],
    *,
    true_class_fn: Callable[[TRun], str],
    aggregate_pred_fn: Callable[[TRun], str],
    transient_pred_fn: Callable[[TRun], str],
    terminal_pred_fn: Callable[[TRun], str],
    scenario_group_fn: Callable[[TRun], str],
) -> ClassificationOutcomeSummary:
    true_classes = sorted({true_class_fn(run) for run in runs})
    predicted_classes = sorted({aggregate_pred_fn(run) for run in runs} | set(true_classes))
    confusion_counts = {
        true_class: {predicted_class: 0 for predicted_class in predicted_classes}
        for true_class in true_classes
    }
    transient_confusion_counts = {
        true_class: {predicted_class: 0 for predicted_class in predicted_classes}
        for true_class in true_classes
    }
    terminal_confusion_counts = {
        true_class: {predicted_class: 0 for predicted_class in predicted_classes}
        for true_class in true_classes
    }
    scenario_confusion_counts = {
        scenario_group_fn(run): {predicted_class: 0 for predicted_class in predicted_classes}
        for run in runs
    }
    class_totals = {true_class: 0 for true_class in true_classes}
    correct = 0
    transient_correct = 0
    terminal_correct = 0

    for run in runs:
        true_class = true_class_fn(run)
        aggregate_class = aggregate_pred_fn(run)
        transient_class = transient_pred_fn(run)
        terminal_class = terminal_pred_fn(run)
        scenario_name = scenario_group_fn(run)
        class_totals[true_class] += 1
        confusion_counts[true_class][aggregate_class] += 1
        transient_confusion_counts[true_class][transient_class] += 1
        terminal_confusion_counts[true_class][terminal_class] += 1
        scenario_confusion_counts[scenario_name][aggregate_class] += 1
        if aggregate_class == true_class:
            correct += 1
        if transient_class == true_class:
            transient_correct += 1
        if terminal_class == true_class:
            terminal_correct += 1

    per_class_accuracy = {
        true_class: (
            confusion_counts[true_class][true_class] / class_totals[true_class]
            if class_totals[true_class]
            else 0.0
        )
        for true_class in true_classes
    }
    total_runs = len(runs)
    return ClassificationOutcomeSummary(
        total_runs=total_runs,
        overall_accuracy=(correct / total_runs) if total_runs else 0.0,
        transient_accuracy=(transient_correct / total_runs) if total_runs else 0.0,
        terminal_accuracy=(terminal_correct / total_runs) if total_runs else 0.0,
        per_class_accuracy=per_class_accuracy,
        confusion_counts=confusion_counts,
        transient_confusion_counts=transient_confusion_counts,
        terminal_confusion_counts=terminal_confusion_counts,
        scenario_confusion_counts=scenario_confusion_counts,
    )


def summarize_classification_features(
    runs: Sequence[TRun],
    *,
    feature_names: Sequence[str],
    true_class_fn: Callable[[TRun], str],
    aggregate_pred_fn: Callable[[TRun], str],
    true_features_fn: Callable[[TRun], Sequence[str]],
    detected_features_fn: Callable[[TRun], Sequence[str]],
    step_iter_fn: Callable[[TRun], Sequence[TStep]],
    step_entropy_fn: Callable[[TStep], float],
    step_feature_probability_fn: Callable[[TStep], dict[str, float]],
) -> ClassificationFeatureSummary:
    true_classes = sorted({true_class_fn(run) for run in runs})
    predicted_classes = sorted({aggregate_pred_fn(run) for run in runs} | set(true_classes))
    class_feature_detection_counts = {
        true_class: {feature_name: 0 for feature_name in feature_names}
        for true_class in true_classes
    }
    feature_confusion_counts = {
        feature_name: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for feature_name in feature_names
    }
    true_feature_predicted_class_counts = {
        feature_name: {predicted_class: 0 for predicted_class in predicted_classes}
        for feature_name in feature_names
    }
    detected_feature_predicted_class_counts = {
        feature_name: {predicted_class: 0 for predicted_class in predicted_classes}
        for feature_name in feature_names
    }
    step_count = max((len(step_iter_fn(run)) for run in runs), default=0)
    entropy_by_step: list[list[float]] = [[] for _ in range(step_count)]
    feature_probabilities_by_step = {feature_name: [[] for _ in range(step_count)] for feature_name in feature_names}

    for run in runs:
        true_class = true_class_fn(run)
        aggregate_class = aggregate_pred_fn(run)
        true_features = set(true_features_fn(run))
        detected_features = set(detected_features_fn(run))
        for feature_name in feature_names:
            true_present = feature_name in true_features
            detected_present = feature_name in detected_features
            if true_present and detected_present:
                feature_confusion_counts[feature_name]["tp"] += 1
            elif true_present and not detected_present:
                feature_confusion_counts[feature_name]["fn"] += 1
            elif not true_present and detected_present:
                feature_confusion_counts[feature_name]["fp"] += 1
            else:
                feature_confusion_counts[feature_name]["tn"] += 1
            if detected_present:
                class_feature_detection_counts[true_class][feature_name] += 1
                detected_feature_predicted_class_counts[feature_name][aggregate_class] += 1
            if true_present:
                true_feature_predicted_class_counts[feature_name][aggregate_class] += 1
        for step_index, step in enumerate(step_iter_fn(run)):
            entropy_by_step[step_index].append(step_entropy_fn(step))
            feature_probabilities = step_feature_probability_fn(step)
            for feature_name in feature_names:
                feature_probabilities_by_step[feature_name][step_index].append(feature_probabilities[feature_name])

    entropy_mean = tuple(
        (sum(step_values) / len(step_values)) if step_values else 0.0
        for step_values in entropy_by_step
    )
    mean_feature_probability = {
        feature_name: tuple(
            (sum(values) / len(values)) if values else 0.0
            for values in feature_probabilities_by_step[feature_name]
        )
        for feature_name in feature_names
    }
    return ClassificationFeatureSummary(
        class_feature_detection_counts=class_feature_detection_counts,
        feature_confusion_counts=feature_confusion_counts,
        true_feature_predicted_class_counts=true_feature_predicted_class_counts,
        detected_feature_predicted_class_counts=detected_feature_predicted_class_counts,
        entropy_mean_by_step=entropy_mean,
        mean_feature_probability_by_step=mean_feature_probability,
    )
