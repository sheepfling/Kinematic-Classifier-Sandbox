from __future__ import annotations

import io
import json
import random
from dataclasses import dataclass
from math import exp, log
from pathlib import Path

from kinematic_classifier_sandbox.reports.markdown import MarkdownDocument

from ...utils.io import write_csv
from ...utils.math import _gaussian_logpdf, _normalize_log_scores
from ...utils.plotting import plt


@dataclass(frozen=True, slots=True)
class AccumulatorClassSpec:
    name: str
    mean: float
    sigma: float
    prior_weight: float


@dataclass(frozen=True, slots=True)
class AccumulatorTrajectory:
    trajectory_id: str
    true_class: str
    scenario_name: str
    seed: int
    times: tuple[float, ...]
    measurements: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class AccumulatorPosteriorStep:
    time: float
    measurement: float
    prior_weights: dict[str, float]
    posterior_weights: dict[str, float]
    log_likelihood_terms: dict[str, float]
    log_odds: dict[str, float]
    predicted_class: str
    confidence: float
    abstained: bool


@dataclass(frozen=True, slots=True)
class AccumulatorRun:
    trajectory_id: str
    true_class: str
    scenario_name: str
    forgetting_factor: float
    confidence_threshold: float
    steps: tuple[AccumulatorPosteriorStep, ...]
    final_weights: dict[str, float]
    final_predicted_class: str
    final_confidence: float
    abstain_rate: float


@dataclass(frozen=True, slots=True)
class AccumulatorBenchmarkSummary:
    total_trajectories: int
    final_accuracy: float
    abstain_rate: float
    per_class_accuracy: dict[str, float]
    confusion_counts: dict[str, dict[str, int]]
    confidence_crossings: int


@dataclass(frozen=True, slots=True)
class AccumulatorBenchmarkResult:
    class_specs: tuple[AccumulatorClassSpec, ...]
    trajectories: tuple[AccumulatorTrajectory, ...]
    runs: tuple[AccumulatorRun, ...]
    summary: AccumulatorBenchmarkSummary
    prior_sensitivity: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class AccumulatorBenchmarkArtifacts:
    run_dir: Path
    report_path: Path
    posterior_history_path: Path
    log_odds_history_path: Path
    confidence_crossings_path: Path
    prior_sensitivity_path: Path
    config_path: Path
    dataset_manifest_path: Path
    class_definitions_path: Path
    plot_png_path: Path


class SequentialBayesAccumulator:
    def __init__(
        self,
        class_specs: tuple[AccumulatorClassSpec, ...],
        *,
        forgetting_factor: float = 1.0,
        confidence_threshold: float = 0.75,
        prior: dict[str, float] | None = None,
        abstain_label: str = "unknown",
    ) -> None:
        if forgetting_factor <= 0.0:
            raise ValueError("forgetting_factor must be positive")
        self._class_specs = class_specs
        self._forgetting_factor = forgetting_factor
        self._confidence_threshold = confidence_threshold
        self._abstain_label = abstain_label
        total_prior = sum(spec.prior_weight for spec in class_specs)
        self._prior = prior or {spec.name: spec.prior_weight / total_prior for spec in class_specs}
        self._posterior = dict(self._prior)
        self._history: list[AccumulatorPosteriorStep] = []

    def reset(self, prior: dict[str, float] | None = None) -> None:
        self._posterior = dict(prior or self._prior)
        self._history.clear()

    def update_with_likelihoods(self, time: float, likelihoods: dict[str, float]) -> AccumulatorPosteriorStep:
        prior_weights = dict(self._posterior)
        log_scores = {
            name: self._forgetting_factor * log(max(prior_weights[name], 1e-12))
            + log(max(likelihoods[name], 1e-12))
            for name in likelihoods
        }
        posterior = _normalize_log_scores(log_scores)
        predicted_class = max(posterior, key=posterior.get)
        confidence = posterior[predicted_class]
        abstained = confidence < self._confidence_threshold
        step = AccumulatorPosteriorStep(
            time=time,
            measurement=float("nan"),
            prior_weights=prior_weights,
            posterior_weights=posterior,
            log_likelihood_terms={name: log(max(likelihoods[name], 1e-12)) for name in likelihoods},
            log_odds=_log_odds(posterior),
            predicted_class=self._abstain_label if abstained else predicted_class,
            confidence=confidence,
            abstained=abstained,
        )
        self._posterior = posterior
        self._history.append(step)
        return step

    def update(self, time: float, measurement: float) -> AccumulatorPosteriorStep:
        likelihoods = {
            spec.name: exp(_gaussian_logpdf(measurement, spec.mean, spec.sigma * spec.sigma))
            for spec in self._class_specs
        }
        step = self.update_with_likelihoods(time, likelihoods)
        self._history[-1] = AccumulatorPosteriorStep(
            time=step.time,
            measurement=measurement,
            prior_weights=step.prior_weights,
            posterior_weights=step.posterior_weights,
            log_likelihood_terms=step.log_likelihood_terms,
            log_odds=step.log_odds,
            predicted_class=step.predicted_class,
            confidence=step.confidence,
            abstained=step.abstained,
        )
        return self._history[-1]

    def posterior(self) -> dict[str, float]:
        return dict(self._posterior)

    def predict(self) -> str:
        predicted = max(self._posterior, key=self._posterior.get)
        return self._abstain_label if self._posterior[predicted] < self._confidence_threshold else predicted

    def history(self) -> tuple[AccumulatorPosteriorStep, ...]:
        return tuple(self._history)


def _log_odds(posterior: dict[str, float]) -> dict[str, float]:
    names = list(posterior)
    if len(names) < 2:
        return {}
    anchor = names[0]
    return {f"{anchor}_vs_{name}": log(max(posterior[anchor], 1e-12) / max(posterior[name], 1e-12)) for name in names[1:]}


def default_accumulator_class_specs() -> tuple[AccumulatorClassSpec, ...]:
    return (
        AccumulatorClassSpec("A", mean=-1.0, sigma=0.55, prior_weight=0.5),
        AccumulatorClassSpec("B", mean=1.0, sigma=0.55, prior_weight=0.5),
    )


def _make_sequence(
    *,
    trajectory_id: str,
    true_class: str,
    scenario_name: str,
    seed: int,
    values: tuple[float, ...],
) -> AccumulatorTrajectory:
    times = tuple(float(index) for index in range(len(values)))
    return AccumulatorTrajectory(
        trajectory_id=trajectory_id,
        true_class=true_class,
        scenario_name=scenario_name,
        seed=seed,
        times=times,
        measurements=values,
    )


def generate_accumulator_trajectories(*, seed: int = 7, trajectories_per_class: int = 3) -> tuple[AccumulatorTrajectory, ...]:
    rng = random.Random(seed)
    trajectories: list[AccumulatorTrajectory] = []
    specs = default_accumulator_class_specs()
    for spec in specs:
        for index in range(trajectories_per_class):
            series = tuple(rng.gauss(spec.mean, 0.28) for _ in range(7))
            trajectories.append(
                _make_sequence(
                    trajectory_id=f"easy_{spec.name}_{index}",
                    true_class=spec.name,
                    scenario_name="easy",
                    seed=seed + index,
                    values=series,
                )
            )
    ambiguous = tuple(0.0 for _ in range(7))
    switch_series = (-1.0, -1.1, -0.9, -0.4, 0.2, 0.8, 1.0)
    trajectories.extend(
        [
            _make_sequence(
                trajectory_id="ambiguous_mid",
                true_class="A",
                scenario_name="ambiguous",
                seed=seed + 10,
                values=ambiguous,
            ),
            _make_sequence(
                trajectory_id="late_flip",
                true_class="B",
                scenario_name="late_flip",
                seed=seed + 11,
                values=switch_series,
            ),
        ]
    )
    return tuple(trajectories)


def run_accumulator(
    trajectory: AccumulatorTrajectory,
    class_specs: tuple[AccumulatorClassSpec, ...],
    *,
    forgetting_factor: float = 1.0,
    confidence_threshold: float = 0.75,
    prior: dict[str, float] | None = None,
) -> AccumulatorRun:
    accumulator = SequentialBayesAccumulator(
        class_specs,
        forgetting_factor=forgetting_factor,
        confidence_threshold=confidence_threshold,
        prior=prior,
    )
    accumulator.reset(prior)
    for time, measurement in zip(trajectory.times, trajectory.measurements):
        accumulator.update(time, measurement)
    steps = accumulator.history()
    final_weights = accumulator.posterior()
    predicted = accumulator.predict()
    confidence = max(final_weights.values())
    abstain_rate = sum(1 for step in steps if step.abstained) / max(len(steps), 1)
    return AccumulatorRun(
        trajectory_id=trajectory.trajectory_id,
        true_class=trajectory.true_class,
        scenario_name=trajectory.scenario_name,
        forgetting_factor=forgetting_factor,
        confidence_threshold=confidence_threshold,
        steps=steps,
        final_weights=final_weights,
        final_predicted_class=predicted,
        final_confidence=confidence,
        abstain_rate=abstain_rate,
    )


def _summarize_runs(runs: tuple[AccumulatorRun, ...], class_names: tuple[str, ...]) -> AccumulatorBenchmarkSummary:
    confusion = {name: {predicted: 0 for predicted in (*class_names, "unknown")} for name in class_names}
    correct = {name: 0 for name in class_names}
    total = {name: 0 for name in class_names}
    abstains = 0
    confidence_crossings = 0
    for run in runs:
        total[run.true_class] += 1
        confusion[run.true_class][run.final_predicted_class] += 1
        if run.final_predicted_class == run.true_class:
            correct[run.true_class] += 1
        if run.final_predicted_class == "unknown":
            abstains += 1
        confidence_crossings += sum(1 for step in run.steps if step.confidence >= run.confidence_threshold)
    return AccumulatorBenchmarkSummary(
        total_trajectories=len(runs),
        final_accuracy=sum(correct.values()) / max(len(runs), 1),
        abstain_rate=abstains / max(len(runs), 1),
        per_class_accuracy={name: correct[name] / total[name] if total[name] else 0.0 for name in class_names},
        confusion_counts=confusion,
        confidence_crossings=confidence_crossings,
    )


def _prior_sweep(
    trajectory: AccumulatorTrajectory,
    class_specs: tuple[AccumulatorClassSpec, ...],
    *,
    forgetting_factor: float,
    confidence_threshold: float,
) -> tuple[dict[str, object], ...]:
    sweep: list[dict[str, object]] = []
    if len(class_specs) < 2:
        return tuple(sweep)
    class_a, class_b = class_specs[:2]
    for prior_a in (0.1, 0.25, 0.5, 0.75, 0.9):
        prior = {class_a.name: prior_a, class_b.name: 1.0 - prior_a}
        run = run_accumulator(
            trajectory,
            class_specs,
            forgetting_factor=forgetting_factor,
            confidence_threshold=confidence_threshold,
            prior=prior,
        )
        sweep.append(
            {
                "trajectory_id": trajectory.trajectory_id,
                "prior_a": prior_a,
                "final_class": run.final_predicted_class,
                "final_confidence": run.final_confidence,
                f"posterior_{class_a.name}": run.final_weights[class_a.name],
                f"posterior_{class_b.name}": run.final_weights[class_b.name],
            }
        )
    return tuple(sweep)


def run_accumulator_benchmark(
    *,
    seed: int = 7,
    forgetting_factor: float = 1.0,
    confidence_threshold: float = 0.75,
    trajectories_per_class: int = 3,
    class_specs: tuple[AccumulatorClassSpec, ...] | None = None,
) -> AccumulatorBenchmarkResult:
    specs = class_specs or default_accumulator_class_specs()
    trajectories = generate_accumulator_trajectories(seed=seed, trajectories_per_class=trajectories_per_class)
    runs = tuple(
        run_accumulator(
            trajectory,
            specs,
            forgetting_factor=forgetting_factor,
            confidence_threshold=confidence_threshold,
        )
        for trajectory in trajectories
    )
    summary = _summarize_runs(runs, tuple(spec.name for spec in specs))
    prior_sensitivity = _prior_sweep(
        next(trajectory for trajectory in trajectories if trajectory.scenario_name == "ambiguous"),
        specs,
        forgetting_factor=forgetting_factor,
        confidence_threshold=confidence_threshold,
    )
    return AccumulatorBenchmarkResult(
        class_specs=specs,
        trajectories=trajectories,
        runs=runs,
        summary=summary,
        prior_sensitivity=prior_sensitivity,
    )


def _render_report(result: AccumulatorBenchmarkResult) -> str:
    summary = result.summary
    report = MarkdownDocument("Sequential Bayesian Accumulator")
    report.paragraph(
        "This benchmark accumulates class evidence sequentially in log space, supports configurable "
        "forgetting, and exposes prior sensitivity on an ambiguous track."
    )
    report.heading("Summary", level=2)
    report.bullet_list(
        [
            f"Trajectories: {summary.total_trajectories}",
            f"Final accuracy: {summary.final_accuracy:.3f}",
            f"Abstain rate: {summary.abstain_rate:.3f}",
            f"Confidence crossings: {summary.confidence_crossings}",
        ]
    )
    report.heading("Per-Class Accuracy", level=2)
    report.bullet_list([f"`{class_name}`: {value:.3f}" for class_name, value in summary.per_class_accuracy.items()])
    report.heading("Prior Sensitivity", level=2)
    report.table(
        ["prior_A", "final_class", "final_confidence", "posterior_A", "posterior_B"],
        [
            (row["prior_a"], row["final_class"], row["final_confidence"], row["posterior_A"], row["posterior_B"])
            for row in result.prior_sensitivity
        ],
    )
    report.heading("Acceptance Notes", level=2)
    report.bullet_list(
        [
            "Equal likelihoods should leave the posterior equal to the prior.",
            "Repeated identical evidence should monotonically reinforce the matching class.",
            "Lower forgetting factors should allow the posterior to switch faster after evidence changes.",
        ]
    )
    return report.text()


def _build_figure(result: AccumulatorBenchmarkResult):
    class_names = [spec.name for spec in result.class_specs]
    selected = []
    for target in ("easy_A_0", "easy_B_0", "ambiguous_mid", "late_flip"):
        selected.append(next(run for run in result.runs if run.trajectory_id == target))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=False)
    colors = {name: color for name, color in zip(class_names, ("#2563eb", "#7c3aed"))}
    for axis, run in zip(axes.flat, selected):
        steps = list(range(len(run.steps)))
        for class_name in class_names:
            axis.plot(steps, [step.posterior_weights[class_name] for step in run.steps], label=class_name, color=colors[class_name], linewidth=2.2)
        axis.set_title(f"{run.scenario_name}: {run.trajectory_id}", loc="left", fontsize=12, fontweight="bold")
        axis.set_ylim(0.0, 1.0)
        axis.grid(True, alpha=0.25)
        axis.set_xlabel("step")
        axis.set_ylabel("posterior")
        axis.legend(frameon=False)
    fig.suptitle("Sequential Bayesian Accumulator", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def render_accumulator_report(result: AccumulatorBenchmarkResult) -> str:
    return _render_report(result)


def render_accumulator_svg(result: AccumulatorBenchmarkResult) -> str:
    fig = _build_figure(result)
    buffer = io.StringIO()
    try:
        fig.savefig(buffer, format="svg", bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def render_accumulator_png_bytes(result: AccumulatorBenchmarkResult) -> bytes:
    fig = _build_figure(result)
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def write_accumulator_artifacts(
    output_dir: str | Path,
    *,
    seed: int = 7,
    forgetting_factor: float = 1.0,
    confidence_threshold: float = 0.75,
    result: AccumulatorBenchmarkResult | None = None,
) -> AccumulatorBenchmarkArtifacts:
    benchmark_result = result or run_accumulator_benchmark(
        seed=seed,
        forgetting_factor=forgetting_factor,
        confidence_threshold=confidence_threshold,
    )
    output_root = Path(output_dir)
    run_dir = output_root / "bayes_accumulator"
    run_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / "bayes_accumulator_report.md"
    posterior_history_path = run_dir / "posterior_history.csv"
    log_odds_history_path = run_dir / "log_odds_history.csv"
    confidence_crossings_path = run_dir / "confidence_crossings.csv"
    prior_sensitivity_path = run_dir / "prior_sensitivity.csv"
    config_path = run_dir / "bayes_accumulator_config.yaml"
    dataset_manifest_path = run_dir / "dataset_manifest.json"
    class_definitions_path = run_dir / "class_definitions.json"
    plot_png_path = run_dir / "bayes_accumulator_diagnostics.png"

    report_path.write_text(render_accumulator_report(benchmark_result), encoding="utf-8")
    plot_png_path.write_bytes(render_accumulator_png_bytes(benchmark_result))

    posterior_rows: list[dict[str, object]] = []
    log_odds_rows: list[dict[str, object]] = []
    confidence_rows: list[dict[str, object]] = []
    for run in benchmark_result.runs:
        for step_index, step in enumerate(run.steps):
            posterior_rows.append(
                {
                    "trajectory_id": run.trajectory_id,
                    "scenario_name": run.scenario_name,
                    "step": step_index,
                    "time": step.time,
                    "measurement": step.measurement,
                    "true_class": run.true_class,
                    "predicted_class": step.predicted_class,
                    "confidence": step.confidence,
                    "abstained": step.abstained,
                    **{f"posterior_{name}": step.posterior_weights[name] for name in benchmark_result.summary.confusion_counts},
                    **{f"log_likelihood_{name}": step.log_likelihood_terms[name] for name in benchmark_result.summary.confusion_counts},
                }
            )
            log_odds_rows.append(
                {
                    "trajectory_id": run.trajectory_id,
                    "scenario_name": run.scenario_name,
                    "step": step_index,
                    "time": step.time,
                    **{name: value for name, value in step.log_odds.items()},
                }
            )
            if step.confidence >= run.confidence_threshold:
                confidence_rows.append(
                    {
                        "trajectory_id": run.trajectory_id,
                        "scenario_name": run.scenario_name,
                        "step": step_index,
                        "time": step.time,
                        "confidence": step.confidence,
                        "predicted_class": step.predicted_class,
                        "true_class": run.true_class,
                    }
                )

    class_names = list(benchmark_result.summary.confusion_counts)
    posterior_fieldnames = [
        "trajectory_id",
        "scenario_name",
        "step",
        "time",
        "measurement",
        "true_class",
        "predicted_class",
        "confidence",
        "abstained",
        *[f"posterior_{name}" for name in class_names],
        *[f"log_likelihood_{name}" for name in class_names],
    ]
    log_odds_keys = []
    for row in log_odds_rows:
        for key in row:
            if key not in {"trajectory_id", "scenario_name", "step", "time"} and key not in log_odds_keys:
                log_odds_keys.append(key)
    log_odds_fieldnames = ["trajectory_id", "scenario_name", "step", "time", *log_odds_keys]
    confidence_fieldnames = ["trajectory_id", "scenario_name", "step", "time", "confidence", "predicted_class", "true_class"]

    write_csv(posterior_history_path, posterior_rows, posterior_fieldnames)
    write_csv(log_odds_history_path, log_odds_rows, log_odds_fieldnames)
    write_csv(confidence_crossings_path, confidence_rows, confidence_fieldnames)
    write_csv(prior_sensitivity_path, [dict(row) for row in benchmark_result.prior_sensitivity], ["trajectory_id", "prior_a", "final_class", "final_confidence", "posterior_A", "posterior_B"])

    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  name: sequential_bayes_accumulator",
                f"  seed: {seed}",
                "classifier:",
                f"  forgetting_factor: {forgetting_factor}",
                f"  confidence_threshold: {confidence_threshold}",
                "dataset:",
                "  scenarios: [easy, ambiguous, late_flip]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    dataset_manifest_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "trajectory_count": benchmark_result.summary.total_trajectories,
                "scenario_names": [trajectory.scenario_name for trajectory in benchmark_result.trajectories],
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

    return AccumulatorBenchmarkArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        posterior_history_path=posterior_history_path,
        log_odds_history_path=log_odds_history_path,
        confidence_crossings_path=confidence_crossings_path,
        prior_sensitivity_path=prior_sensitivity_path,
        config_path=config_path,
        dataset_manifest_path=dataset_manifest_path,
        class_definitions_path=class_definitions_path,
        plot_png_path=plot_png_path,
    )
