from __future__ import annotations

import io
import json
import random
from dataclasses import dataclass
from math import log
from pathlib import Path

from kinematic_classifier_sandbox.utils.io import write_csv
from kinematic_classifier_sandbox.utils.math import (
    gaussian_logpdf as _gaussian_logpdf,
)
from kinematic_classifier_sandbox.utils.math import (
    normalize_log_scores as _normalize_log_scores,
)
from kinematic_classifier_sandbox.utils.plotting import plt

from ..markdown_builder import MarkdownDocument


@dataclass(frozen=True, slots=True)
class PointwiseClassSpec:
    name: str
    mean: float
    sigma: float
    prior_weight: float


@dataclass(frozen=True, slots=True)
class PointwiseTrajectory:
    trajectory_id: str
    true_class: str
    scenario_name: str
    seed: int
    times: tuple[float, ...]
    measurements: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class PointwisePosteriorStep:
    time: float
    measurement: float
    posterior_weights: dict[str, float]
    log_likelihood_terms: dict[str, float]
    predicted_class: str
    confidence: float


@dataclass(frozen=True, slots=True)
class PointwiseClassificationRun:
    trajectory_id: str
    true_class: str
    scenario_name: str
    steps: tuple[PointwisePosteriorStep, ...]
    final_weights: dict[str, float]
    final_predicted_class: str


@dataclass(frozen=True, slots=True)
class PointwiseBenchmarkSummary:
    total_trajectories: int
    final_accuracy: float
    per_class_accuracy: dict[str, float]
    confusion_counts: dict[str, dict[str, int]]


@dataclass(frozen=True, slots=True)
class PointwiseBenchmarkResult:
    class_specs: tuple[PointwiseClassSpec, ...]
    trajectories: tuple[PointwiseTrajectory, ...]
    runs: tuple[PointwiseClassificationRun, ...]
    summary: PointwiseBenchmarkSummary


@dataclass(frozen=True, slots=True)
class PointwiseBenchmarkArtifacts:
    run_dir: Path
    report_path: Path
    posterior_history_path: Path
    confusion_matrix_path: Path
    plot_png_path: Path
    config_path: Path
    dataset_manifest_path: Path
    class_definitions_path: Path


class GaussianPointwiseClassifier:
    def __init__(self, class_specs: tuple[PointwiseClassSpec, ...], prior: dict[str, float] | None = None) -> None:
        self._class_specs = class_specs
        total_prior = sum(spec.prior_weight for spec in class_specs)
        self._prior = prior or {spec.name: spec.prior_weight / total_prior for spec in class_specs}
        self._posterior = dict(self._prior)
        self._history: list[PointwisePosteriorStep] = []

    def reset(self, prior: dict[str, float] | None = None) -> None:
        if prior is not None:
            self._posterior = dict(prior)
        else:
            self._posterior = dict(self._prior)
        self._history.clear()

    def update(self, time: float, measurement: float) -> PointwisePosteriorStep:
        log_scores = {
            spec.name: log(max(self._posterior[spec.name], 1e-12))
            + _gaussian_logpdf(measurement, spec.mean, spec.sigma * spec.sigma)
            for spec in self._class_specs
        }
        posterior = _normalize_log_scores(log_scores)
        predicted_class = max(posterior, key=posterior.get)
        step = PointwisePosteriorStep(
            time=time,
            measurement=measurement,
            posterior_weights=posterior,
            log_likelihood_terms=log_scores,
            predicted_class=predicted_class,
            confidence=posterior[predicted_class],
        )
        self._posterior = posterior
        self._history.append(step)
        return step

    def posterior(self) -> dict[str, float]:
        return dict(self._posterior)

    def predict(self) -> str:
        return max(self._posterior, key=self._posterior.get)

    def history(self) -> tuple[PointwisePosteriorStep, ...]:
        return tuple(self._history)


def default_pointwise_class_specs() -> tuple[PointwiseClassSpec, ...]:
    return (
        PointwiseClassSpec("A", mean=-1.0, sigma=0.35, prior_weight=0.5),
        PointwiseClassSpec("B", mean=1.0, sigma=0.35, prior_weight=0.5),
    )


def generate_pointwise_trajectories(
    *,
    classes: tuple[PointwiseClassSpec, ...] | None = None,
    seed: int = 7,
    trajectories_per_class: int = 24,
    steps_per_trajectory: int = 6,
    obs_sigma: float = 0.25,
    scenario_name: str = "easy",
) -> tuple[PointwiseTrajectory, ...]:
    class_specs = classes or default_pointwise_class_specs()
    rng = random.Random(seed)
    trajectories: list[PointwiseTrajectory] = []
    for class_index, spec in enumerate(class_specs):
        for trajectory_index in range(trajectories_per_class):
            trajectory_rng = random.Random(rng.randrange(1 << 30) + class_index * 1000 + trajectory_index)
            times = tuple(float(step) for step in range(steps_per_trajectory))
            measurements = tuple(trajectory_rng.gauss(spec.mean, obs_sigma) for _ in range(steps_per_trajectory))
            trajectories.append(
                PointwiseTrajectory(
                    trajectory_id=f"{scenario_name}_{spec.name}_{trajectory_index}",
                    true_class=spec.name,
                    scenario_name=scenario_name,
                    seed=seed + class_index + trajectory_index,
                    times=times,
                    measurements=measurements,
                )
            )
    return tuple(trajectories)


def generate_pointwise_benchmark_trajectories(
    *,
    seed: int = 7,
) -> tuple[PointwiseTrajectory, ...]:
    easy = generate_pointwise_trajectories(seed=seed, trajectories_per_class=18, steps_per_trajectory=6, obs_sigma=0.22, scenario_name="easy")
    overlap_classes = (
        PointwiseClassSpec("A", mean=0.0, sigma=1.0, prior_weight=0.5),
        PointwiseClassSpec("B", mean=0.5, sigma=1.0, prior_weight=0.5),
    )
    overlap = generate_pointwise_trajectories(
        classes=overlap_classes,
        seed=seed + 1,
        trajectories_per_class=18,
        steps_per_trajectory=6,
        obs_sigma=1.0,
        scenario_name="overlap",
    )
    return easy + overlap


def run_pointwise_classifier(
    trajectory: PointwiseTrajectory,
    class_specs: tuple[PointwiseClassSpec, ...],
    *,
    prior: dict[str, float] | None = None,
) -> PointwiseClassificationRun:
    classifier = GaussianPointwiseClassifier(class_specs, prior=prior)
    classifier.reset(prior)
    for time, measurement in zip(trajectory.times, trajectory.measurements):
        classifier.update(time, measurement)
    return PointwiseClassificationRun(
        trajectory_id=trajectory.trajectory_id,
        true_class=trajectory.true_class,
        scenario_name=trajectory.scenario_name,
        steps=classifier.history(),
        final_weights=classifier.posterior(),
        final_predicted_class=classifier.predict(),
    )


def _summarize_runs(runs: tuple[PointwiseClassificationRun, ...], class_names: tuple[str, ...]) -> PointwiseBenchmarkSummary:
    confusion_counts = {true_name: {predicted_name: 0 for predicted_name in class_names} for true_name in class_names}
    per_class_correct = {name: 0 for name in class_names}
    per_class_total = {name: 0 for name in class_names}
    correct_total = 0
    for run in runs:
        confusion_counts[run.true_class][run.final_predicted_class] += 1
        per_class_total[run.true_class] += 1
        if run.final_predicted_class == run.true_class:
            correct_total += 1
            per_class_correct[run.true_class] += 1
    per_class_accuracy = {
        name: per_class_correct[name] / per_class_total[name] if per_class_total[name] else 0.0
        for name in class_names
    }
    return PointwiseBenchmarkSummary(
        total_trajectories=len(runs),
        final_accuracy=correct_total / max(len(runs), 1),
        per_class_accuracy=per_class_accuracy,
        confusion_counts=confusion_counts,
    )


def run_pointwise_benchmark(
    *,
    seed: int = 7,
    class_specs: tuple[PointwiseClassSpec, ...] | None = None,
) -> PointwiseBenchmarkResult:
    specs = class_specs or default_pointwise_class_specs()
    trajectories = generate_pointwise_benchmark_trajectories(seed=seed)
    runs = tuple(run_pointwise_classifier(trajectory, specs) for trajectory in trajectories)
    summary = _summarize_runs(runs, tuple(spec.name for spec in specs))
    return PointwiseBenchmarkResult(class_specs=specs, trajectories=trajectories, runs=runs, summary=summary)


def _format_confusion_rows(summary: PointwiseBenchmarkSummary) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    class_names = tuple(summary.confusion_counts)
    for true_name in class_names:
        row = {"true_class": true_name}
        row.update(summary.confusion_counts[true_name])
        rows.append(row)
    return rows


def render_pointwise_benchmark_report(result: PointwiseBenchmarkResult) -> str:
    summary = result.summary
    report = MarkdownDocument("Pointwise Gaussian Benchmark")
    report.paragraph(
        "This milestone-1 benchmark classifies each measurement with a Gaussian class likelihood and "
        "updates the posterior sequentially across the trajectory."
    )
    report.heading("Summary", level=2)
    report.bullet_list(
        [
            f"Trajectories: {summary.total_trajectories}",
            f"Final accuracy: {summary.final_accuracy:.3f}",
        ]
    )
    report.heading("Per-Class Accuracy", level=2)
    report.bullet_list([f"`{class_name}`: {accuracy:.3f}" for class_name, accuracy in summary.per_class_accuracy.items()])
    report.heading("Confusion Counts", level=2)
    report.table(
        ["True \\ Pred"] + list(summary.confusion_counts),
        [
            (true_name,) + tuple(str(summary.confusion_counts[true_name][predicted]) for predicted in summary.confusion_counts)
            for true_name in summary.confusion_counts
        ],
    )
    report.heading("Acceptance Notes", level=2)
    report.bullet_list(
        [
            "Easy class separation should yield high final accuracy.",
            "Overlap scenarios should remain uncertain and less overconfident.",
        ]
    )
    return report.text()


def _build_diagnostic_figure(result: PointwiseBenchmarkResult):
    easy_run = next(run for run in result.runs if run.scenario_name == "easy" and run.true_class == result.class_specs[0].name)
    overlap_run = next(run for run in result.runs if run.scenario_name == "overlap" and run.true_class == result.class_specs[0].name)
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 7.5), sharex="col")
    colors = {spec.name: color for spec, color in zip(result.class_specs, ("#2563eb", "#7c3aed"))}

    for row_axes, run, title in ((axes[0], easy_run, "Easy"), (axes[1], overlap_run, "Overlap")):
        measurement_ax, posterior_ax = row_axes
        steps = list(range(len(run.steps)))
        measurements = [step.measurement for step in run.steps]
        measurement_ax.plot(steps, measurements, color="#111827", linewidth=2.0, marker="o")
        for spec in result.class_specs:
            measurement_ax.axhline(spec.mean, color=colors[spec.name], linestyle="--", linewidth=1.2, label=spec.name)
        measurement_ax.set_title(f"{title} measurements", loc="left", fontsize=12, fontweight="bold")
        measurement_ax.set_ylabel("value")
        measurement_ax.grid(True, alpha=0.25)
        measurement_ax.legend(frameon=False)

        for spec in result.class_specs:
            posterior_ax.plot(steps, [step.posterior_weights[spec.name] for step in run.steps], linewidth=2.2, label=spec.name, color=colors[spec.name])
        posterior_ax.set_ylim(0.0, 1.0)
        posterior_ax.set_title(f"{title} posterior", loc="left", fontsize=12, fontweight="bold")
        posterior_ax.set_ylabel("probability")
        posterior_ax.grid(True, alpha=0.25)
        posterior_ax.legend(frameon=False)

    axes[1][0].set_xlabel("step")
    axes[1][1].set_xlabel("step")
    fig.suptitle("Pointwise Gaussian Baseline Diagnostics", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def render_pointwise_benchmark_svg(result: PointwiseBenchmarkResult) -> str:
    fig = _build_diagnostic_figure(result)
    buffer = io.StringIO()
    try:
        fig.savefig(buffer, format="svg", bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def render_pointwise_benchmark_png_bytes(result: PointwiseBenchmarkResult) -> bytes:
    fig = _build_diagnostic_figure(result)
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def write_pointwise_benchmark_artifacts(
    output_dir: str | Path,
    *,
    seed: int = 7,
    result: PointwiseBenchmarkResult | None = None,
) -> PointwiseBenchmarkArtifacts:
    benchmark_result = result or run_pointwise_benchmark(seed=seed)
    output_root = Path(output_dir)
    run_dir = output_root / "pointwise_baseline"
    run_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / "pointwise_baseline_report.md"
    posterior_history_path = run_dir / "posterior_history.csv"
    confusion_matrix_path = run_dir / "confusion_final.csv"
    plot_png_path = run_dir / "pointwise_baseline_diagnostics.png"
    config_path = run_dir / "pointwise_classifier_config.yaml"
    dataset_manifest_path = run_dir / "dataset_manifest.json"
    class_definitions_path = run_dir / "likelihood_parameters.json"

    report_path.write_text(render_pointwise_benchmark_report(benchmark_result), encoding="utf-8")
    plot_png_path.write_bytes(render_pointwise_benchmark_png_bytes(benchmark_result))

    posterior_rows: list[dict[str, object]] = []
    for trajectory, run in zip(benchmark_result.trajectories, benchmark_result.runs):
        for step_index, step in enumerate(run.steps):
            posterior_rows.append(
                {
                    "trajectory_id": trajectory.trajectory_id,
                    "scenario_name": trajectory.scenario_name,
                    "step": step_index,
                    "time": step.time,
                    "measurement": step.measurement,
                    "true_class": trajectory.true_class,
                    "predicted_class": step.predicted_class,
                    "confidence": step.confidence,
                    **{f"posterior_{name}": step.posterior_weights[name] for name in benchmark_result.summary.confusion_counts},
                    **{f"log_likelihood_{name}": step.log_likelihood_terms[name] for name in benchmark_result.summary.confusion_counts},
                }
            )
    posterior_fieldnames = [
        "trajectory_id",
        "scenario_name",
        "step",
        "time",
        "measurement",
        "true_class",
        "predicted_class",
        "confidence",
        *[f"posterior_{name}" for name in benchmark_result.summary.confusion_counts],
        *[f"log_likelihood_{name}" for name in benchmark_result.summary.confusion_counts],
    ]
    write_csv(posterior_history_path, posterior_rows, posterior_fieldnames)
    write_csv(confusion_matrix_path, _format_confusion_rows(benchmark_result.summary), ["true_class", *benchmark_result.summary.confusion_counts.keys()])

    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  name: pointwise_baseline",
                f"  seed: {seed}",
                "classifier:",
                "  type: gaussian_pointwise",
                "  prior: uniform",
                "dataset:",
                "  scenarios: [easy, overlap]",
                "  trajectories_per_class: 18",
                "  steps_per_trajectory: 6",
                "  easy_obs_sigma: 0.22",
                "  overlap_obs_sigma: 1.0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    dataset_manifest_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "scenario_names": ["easy", "overlap"],
                "trajectory_count": benchmark_result.summary.total_trajectories,
                "easy_accuracy": benchmark_result.summary.per_class_accuracy[benchmark_result.class_specs[0].name],
                "overlap_accuracy": benchmark_result.summary.per_class_accuracy[benchmark_result.class_specs[1].name],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    class_definitions_path.write_text(
        json.dumps(
            {
                "class_names": [spec.name for spec in benchmark_result.class_specs],
                "means": {spec.name: spec.mean for spec in benchmark_result.class_specs},
                "sigmas": {spec.name: spec.sigma for spec in benchmark_result.class_specs},
                "priors": {spec.name: spec.prior_weight for spec in benchmark_result.class_specs},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return PointwiseBenchmarkArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        posterior_history_path=posterior_history_path,
        confusion_matrix_path=confusion_matrix_path,
        plot_png_path=plot_png_path,
        config_path=config_path,
        dataset_manifest_path=dataset_manifest_path,
        class_definitions_path=class_definitions_path,
    )
