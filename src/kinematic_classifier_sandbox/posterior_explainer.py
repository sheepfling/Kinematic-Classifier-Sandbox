from __future__ import annotations

from dataclasses import dataclass
import io
import os
from pathlib import Path

from .toy_1d import (
    ClassificationRun,
    ToyBenchmarkResult,
    run_toy_benchmark,
)


@dataclass(frozen=True, slots=True)
class PosteriorWalkthrough:
    title: str
    class_names: tuple[str, ...]
    run: ClassificationRun
    step_index: int
    measurement: float
    prior_weights: dict[str, float]
    posterior_weights: dict[str, float]
    log_terms: dict[str, dict[str, float]]


def _prepare_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp/kinematic-classifier-sandbox-mpl")))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _track_for_run(result: ToyBenchmarkResult, run: ClassificationRun):
    return next(track for track in result.dataset.tracks if track.seed == run.seed)


def _build_walkthrough(result: ToyBenchmarkResult, run: ClassificationRun, *, title: str) -> PosteriorWalkthrough:
    class_names = tuple(spec.name for spec in result.dataset.class_specs)
    track = _track_for_run(result, run)
    if run.aggregate_map_class == run.true_class:
        step_index = min(max(2, len(run.steps) // 3), len(run.steps) - 1)
    else:
        misclassified_indices = [index for index, step in enumerate(run.steps) if step.map_class != run.true_class]
        step_index = misclassified_indices[min(len(misclassified_indices) // 2, len(misclassified_indices) - 1)] if misclassified_indices else len(run.steps) - 1
    step = run.steps[step_index]
    return PosteriorWalkthrough(
        title=title,
        class_names=class_names,
        run=run,
        step_index=step_index,
        measurement=track.positions_obs[step_index + 1],
        prior_weights=step.predicted_class_weights,
        posterior_weights=step.updated_class_weights,
        log_terms=step.log_likelihood_terms,
    )


def _select_success_walkthrough(result: ToyBenchmarkResult) -> PosteriorWalkthrough:
    preferred_order = ("drift", "powered", "unknown", "coast", "maneuver", "brake")
    run = next(
        (
            result.representative_runs[class_name]
            for class_name in preferred_order
            if class_name in result.representative_runs
            and result.representative_runs[class_name].steps
            and result.representative_runs[class_name].aggregate_map_class == class_name
        ),
        next(run for run in result.runs if run.steps and run.aggregate_map_class == run.true_class),
    )
    return _build_walkthrough(result, run, title="Posterior Update Walkthrough")


def _select_failure_walkthrough(result: ToyBenchmarkResult) -> PosteriorWalkthrough:
    preferred_pairs = (
        ("brake", "powered"),
        ("maneuver", "unknown"),
        ("coast", "drift"),
    )
    for true_class, predicted_class in preferred_pairs:
        for run in result.runs:
            if run.true_class == true_class and run.aggregate_map_class == predicted_class and run.steps:
                return _build_walkthrough(result, run, title="Posterior Failure Walkthrough")
    run = next(run for run in result.runs if run.steps and run.aggregate_map_class != run.true_class)
    return _build_walkthrough(result, run, title="Posterior Failure Walkthrough")


def _render_walkthrough_markdown(result: ToyBenchmarkResult, walkthrough: PosteriorWalkthrough) -> str:
    step = walkthrough.run.steps[walkthrough.step_index]
    lines = [
        f"# {walkthrough.title}",
        "",
        "This artifact illustrates the class-posterior update used in the toy benchmark for one concrete measurement step.",
        "",
        "## Measurement Model",
        "",
        "- Latent state: `x_t = [position, velocity, acceleration]`.",
        "- Measurement: `z_t = H x_t + v_t` with `H = [1, 0, 0]` and `v_t ~ N(0, R)`.",
        "- In this toy setup, `z_t` is the observed position sample.",
        "",
        "## Bayesian Update",
        "",
        "- Predict each class-conditioned filter state forward one step.",
        "- Score the measurement innovation with a Gaussian likelihood `p(z_t | s_i, history)`.",
        "- Multiply by soft envelope and behavior terms to form a composite class likelihood.",
        "- Update class weights with `p(s_i | z_{1:t}) ∝ p(s_i | z_{1:t-1}) * L_i(z_t)` and normalize across classes.",
        "",
        "## Worked Example",
        "",
        f"- True class: `{walkthrough.run.true_class}`",
        f"- Scenario: `{walkthrough.run.scenario_name}`",
        f"- Aggregate predicted class: `{walkthrough.run.aggregate_map_class}`",
        f"- Step index: `{walkthrough.step_index + 1}` of `{len(walkthrough.run.steps)}` posterior updates",
        f"- Observed measurement `z_t`: `{walkthrough.measurement:.3f}`",
        f"- MAP class after update: `{step.map_class}`",
        "",
        "## Composite Log-Likelihood Terms by Class",
        "",
        "| Class | Prior | dyn | speed | accel | behavior | observed | mode | total | Posterior |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for class_name in walkthrough.class_names:
        prior = walkthrough.prior_weights[class_name]
        posterior = walkthrough.posterior_weights[class_name]
        terms = walkthrough.log_terms[class_name]
        behavior_total = sum(terms.get(name, 0.0) for name in ("velocity_center", "accel_center", "direction", "oscillation"))
        observed_total = sum(terms.get(name, 0.0) for name in ("obs_velocity", "obs_accel"))
        mode_total = terms.get("mode_mix", 0.0)
        lines.append(
            f"| `{class_name}` | {prior:.3f} | {terms.get('dyn', 0.0):.3f} | {terms.get('speed', 0.0):.3f} | {terms.get('accel', 0.0):.3f} | {behavior_total:.3f} | {observed_total:.3f} | {mode_total:.3f} | {terms.get('total', 0.0):.3f} | {posterior:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `dyn` is the Gaussian innovation log likelihood from the position measurement residual.",
            "- `speed` and `accel` are soft envelope probabilities, not chopped PDFs. They are interval probabilities from Gaussian CDF mass.",
            "- `behavior` collects class-shape terms on latent velocity and acceleration.",
            "- `observed` uses finite-difference velocity and acceleration estimates from recent measurements.",
            "- `mode` is a within-class mixture term for transient classes such as `brake` and `maneuver`.",
            "",
            "## Posterior Formula Used in Practice",
            "",
            "`log w_t(s_i) = log w_{t-1}(s_i) + log_dyn_i + log_speed_i + log_accel_i + log_behavior_i + log_observed_i + log_mode_i - log Z_t`",
            "",
            "where `log Z_t` is the across-class normalizer from `logsumexp`.",
        ]
    )
    return "\n".join(lines)


def render_posterior_explainer_markdown(result: ToyBenchmarkResult) -> str:
    return _render_walkthrough_markdown(result, _select_success_walkthrough(result))


def render_posterior_failure_markdown(result: ToyBenchmarkResult) -> str:
    return _render_walkthrough_markdown(result, _select_failure_walkthrough(result))


def _build_posterior_explainer_figure(walkthrough: PosteriorWalkthrough):
    plt = _prepare_matplotlib()
    class_names = list(walkthrough.class_names)
    step = walkthrough.run.steps[walkthrough.step_index]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    prior_ax, score_ax, posterior_ax = axes
    x = list(range(len(class_names)))

    prior_ax.bar(x, [walkthrough.prior_weights[name] for name in class_names], color="#2563eb")
    prior_ax.set_title("Prior Class Weights", loc="left", fontsize=12, fontweight="bold")
    prior_ax.set_xticks(x, class_names, rotation=45, ha="right")
    prior_ax.set_ylim(0.0, 1.0)
    prior_ax.grid(True, axis="y", alpha=0.25)

    dyn_scores = [walkthrough.log_terms[name].get("dyn", 0.0) for name in class_names]
    behavior_scores = [
        sum(walkthrough.log_terms[name].get(term, 0.0) for term in ("speed", "accel", "velocity_center", "accel_center", "direction", "oscillation", "obs_velocity", "obs_accel", "mode_mix"))
        for name in class_names
    ]
    score_ax.bar(x, dyn_scores, color="#0f766e", label="innovation")
    score_ax.bar(x, behavior_scores, bottom=dyn_scores, color="#f59e0b", label="composite extras")
    score_ax.set_title("Composite Log Score", loc="left", fontsize=12, fontweight="bold")
    score_ax.set_xticks(x, class_names, rotation=45, ha="right")
    score_ax.grid(True, axis="y", alpha=0.25)
    score_ax.legend(frameon=False)

    posterior_ax.bar(x, [walkthrough.posterior_weights[name] for name in class_names], color="#7c3aed")
    posterior_ax.set_title("Posterior Class Weights", loc="left", fontsize=12, fontweight="bold")
    posterior_ax.set_xticks(x, class_names, rotation=45, ha="right")
    posterior_ax.set_ylim(0.0, 1.0)
    posterior_ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        f"{walkthrough.title}: {walkthrough.run.scenario_name} step {walkthrough.step_index + 1}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def render_posterior_explainer_png_bytes(result: ToyBenchmarkResult) -> bytes:
    plt = _prepare_matplotlib()
    fig = _build_posterior_explainer_figure(_select_success_walkthrough(result))
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def render_posterior_failure_png_bytes(result: ToyBenchmarkResult) -> bytes:
    plt = _prepare_matplotlib()
    fig = _build_posterior_explainer_figure(_select_failure_walkthrough(result))
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def render_posterior_comparison_markdown(result: ToyBenchmarkResult) -> str:
    success = _select_success_walkthrough(result)
    failure = _select_failure_walkthrough(result)
    lines = [
        "# Toy 1D Posterior Comparison",
        "",
        "This artifact compares one successful posterior update against one confused posterior update using the same composite score decomposition.",
        "",
        "## Cases",
        "",
        f"- Success case: `{success.run.scenario_name}` with true class `{success.run.true_class}` and aggregate prediction `{success.run.aggregate_map_class}` at step `{success.step_index + 1}`",
        f"- Failure case: `{failure.run.scenario_name}` with true class `{failure.run.true_class}` and aggregate prediction `{failure.run.aggregate_map_class}` at step `{failure.step_index + 1}`",
        "",
        "## Side-by-Side Posterior Terms",
        "",
        "| Class | Success prior | Success total | Success posterior | Failure prior | Failure total | Failure posterior |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for class_name in success.class_names:
        success_terms = success.log_terms[class_name]
        failure_terms = failure.log_terms[class_name]
        lines.append(
            f"| `{class_name}` | {success.prior_weights[class_name]:.3f} | {success_terms.get('total', 0.0):.3f} | {success.posterior_weights[class_name]:.3f} | {failure.prior_weights[class_name]:.3f} | {failure_terms.get('total', 0.0):.3f} | {failure.posterior_weights[class_name]:.3f} |"
        )

    def margin_text(walkthrough: PosteriorWalkthrough) -> str:
        ranked = sorted(walkthrough.posterior_weights.items(), key=lambda item: item[1], reverse=True)
        if len(ranked) < 2:
            return "n/a"
        return f"{ranked[0][0]} over {ranked[1][0]} by {ranked[0][1] - ranked[1][1]:.3f}"

    lines.extend(
        [
            "",
            "## Decision Margins",
            "",
            f"- Success margin: {margin_text(success)}",
            f"- Failure margin: {margin_text(failure)}",
            "",
            "## Largest Score Shifts Between Cases",
            "",
            "| Class | Delta total (success - failure) | Delta posterior (success - failure) |",
            "| --- | ---: | ---: |",
        ]
    )
    for class_name in success.class_names:
        success_total = success.log_terms[class_name].get("total", 0.0)
        failure_total = failure.log_terms[class_name].get("total", 0.0)
        delta_total = success_total - failure_total
        delta_posterior = success.posterior_weights[class_name] - failure.posterior_weights[class_name]
        lines.append(f"| `{class_name}` | {delta_total:.3f} | {delta_posterior:.3f} |")

    lines.extend(
        [
            "",
            "## Reading Guide",
            "",
            "- `dyn` is the pure Gaussian innovation likelihood from the measurement residual.",
            "- `speed` and `accel` are soft envelope terms based on Gaussian interval probabilities.",
            "- `behavior`, `observed`, and `mode` are the extra class-discriminative terms that usually decide the winner when raw innovation scores are close.",
            "- The failure case is most useful when the wrong class has a consistently better composite total even though the true class remains physically plausible.",
        ]
    )
    return "\n".join(lines)


def _build_posterior_comparison_figure(result: ToyBenchmarkResult):
    plt = _prepare_matplotlib()
    success = _select_success_walkthrough(result)
    failure = _select_failure_walkthrough(result)
    class_names = list(success.class_names)
    x = list(range(len(class_names)))
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))
    score_ax, posterior_ax = axes

    success_totals = [success.log_terms[name].get("total", 0.0) for name in class_names]
    failure_totals = [failure.log_terms[name].get("total", 0.0) for name in class_names]
    score_ax.bar([value - width / 2 for value in x], success_totals, width=width, color="#0f766e", label="success")
    score_ax.bar([value + width / 2 for value in x], failure_totals, width=width, color="#b91c1c", label="failure")
    score_ax.set_title("Composite Log Totals", loc="left", fontsize=12, fontweight="bold")
    score_ax.set_xticks(x, class_names, rotation=45, ha="right")
    score_ax.grid(True, axis="y", alpha=0.25)
    score_ax.legend(frameon=False)

    success_posteriors = [success.posterior_weights[name] for name in class_names]
    failure_posteriors = [failure.posterior_weights[name] for name in class_names]
    posterior_ax.bar([value - width / 2 for value in x], success_posteriors, width=width, color="#2563eb", label="success")
    posterior_ax.bar([value + width / 2 for value in x], failure_posteriors, width=width, color="#7c3aed", label="failure")
    posterior_ax.set_title("Posterior Weights", loc="left", fontsize=12, fontweight="bold")
    posterior_ax.set_xticks(x, class_names, rotation=45, ha="right")
    posterior_ax.set_ylim(0.0, 1.0)
    posterior_ax.grid(True, axis="y", alpha=0.25)
    posterior_ax.legend(frameon=False)

    fig.suptitle("Posterior Success vs Failure Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def render_posterior_comparison_png_bytes(result: ToyBenchmarkResult) -> bytes:
    plt = _prepare_matplotlib()
    fig = _build_posterior_comparison_figure(result)
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def render_posterior_margin_trace_markdown(result: ToyBenchmarkResult) -> str:
    failure = _select_failure_walkthrough(result)
    true_class = failure.run.true_class
    predicted_class = failure.run.aggregate_map_class
    track = _track_for_run(result, failure.run)
    lines = [
        "# Toy 1D Posterior Margin Trace",
        "",
        "This artifact traces the margin between the true class and the winning confused class across time for one failure run.",
        "",
        "## Case",
        "",
        f"- Scenario: `{failure.run.scenario_name}`",
        f"- True class: `{true_class}`",
        f"- Aggregate predicted class: `{predicted_class}`",
        "",
        "## Margin Definition",
        "",
        f"- Posterior margin: `p({true_class} | z_1:t) - p({predicted_class} | z_1:t)`",
        f"- Score margin: `log_total({true_class}) - log_total({predicted_class})`",
        "",
        "## Stepwise Margins",
        "",
        "| Step | z_t | True posterior | Predicted posterior | Posterior margin | Score margin | Dominant negative term |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for step_index, step in enumerate(failure.run.steps):
        measurement = track.positions_obs[step_index + 1]
        true_terms = step.log_likelihood_terms[true_class]
        predicted_terms = step.log_likelihood_terms[predicted_class]
        posterior_margin = step.updated_class_weights[true_class] - step.updated_class_weights[predicted_class]
        score_margin = true_terms["total"] - predicted_terms["total"]
        component_deltas = {
            "dyn": true_terms.get("dyn", 0.0) - predicted_terms.get("dyn", 0.0),
            "speed": true_terms.get("speed", 0.0) - predicted_terms.get("speed", 0.0),
            "accel": true_terms.get("accel", 0.0) - predicted_terms.get("accel", 0.0),
            "behavior": sum(true_terms.get(name, 0.0) - predicted_terms.get(name, 0.0) for name in ("velocity_center", "accel_center", "direction", "oscillation")),
            "observed": sum(true_terms.get(name, 0.0) - predicted_terms.get(name, 0.0) for name in ("obs_velocity", "obs_accel")),
            "mode": true_terms.get("mode_mix", 0.0) - predicted_terms.get("mode_mix", 0.0),
        }
        dominant_negative_term = min(component_deltas.items(), key=lambda item: item[1])[0]
        lines.append(
            f"| {step_index + 1} | {measurement:.3f} | {step.updated_class_weights[true_class]:.3f} | {step.updated_class_weights[predicted_class]:.3f} | {posterior_margin:.3f} | {score_margin:.3f} | {dominant_negative_term} |"
        )

    crossover_index = next(
        (
            index + 1
            for index, step in enumerate(failure.run.steps)
            if step.updated_class_weights[predicted_class] > step.updated_class_weights[true_class]
        ),
        None,
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- First posterior crossover step: `{crossover_index}`" if crossover_index is not None else "- No crossover found",
            f"- The trace is most useful for seeing whether the wrong class wins because of raw innovation fit, envelope terms, behavior terms, observed finite-difference terms, or within-class mode terms.",
        ]
    )
    return "\n".join(lines)


def _build_posterior_margin_trace_figure(result: ToyBenchmarkResult):
    plt = _prepare_matplotlib()
    failure = _select_failure_walkthrough(result)
    true_class = failure.run.true_class
    predicted_class = failure.run.aggregate_map_class
    track = _track_for_run(result, failure.run)
    steps = list(range(1, len(failure.run.steps) + 1))
    posterior_margins = [
        step.updated_class_weights[true_class] - step.updated_class_weights[predicted_class]
        for step in failure.run.steps
    ]
    score_margins = [
        step.log_likelihood_terms[true_class]["total"] - step.log_likelihood_terms[predicted_class]["total"]
        for step in failure.run.steps
    ]
    measurements = list(track.positions_obs[1:])

    fig, axes = plt.subplots(3, 1, figsize=(12, 8.2), sharex=True)
    measurement_ax, posterior_ax, score_ax = axes

    measurement_ax.plot(steps, measurements, color="#111827", linewidth=2.0)
    measurement_ax.set_title("Observed Measurement z_t", loc="left", fontsize=12, fontweight="bold")
    measurement_ax.set_ylabel("position")
    measurement_ax.grid(True, alpha=0.25)

    posterior_ax.plot(steps, posterior_margins, color="#2563eb", linewidth=2.2)
    posterior_ax.axhline(0.0, color="#991b1b", linestyle="--", linewidth=1.0)
    posterior_ax.set_title(f"Posterior Margin: {true_class} - {predicted_class}", loc="left", fontsize=12, fontweight="bold")
    posterior_ax.set_ylabel("probability")
    posterior_ax.grid(True, alpha=0.25)

    score_ax.plot(steps, score_margins, color="#7c3aed", linewidth=2.2)
    score_ax.axhline(0.0, color="#991b1b", linestyle="--", linewidth=1.0)
    score_ax.set_title(f"Composite Score Margin: {true_class} - {predicted_class}", loc="left", fontsize=12, fontweight="bold")
    score_ax.set_xlabel("step")
    score_ax.set_ylabel("log-score")
    score_ax.grid(True, alpha=0.25)

    fig.suptitle(f"Posterior Margin Trace: {failure.run.scenario_name}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def render_posterior_margin_trace_png_bytes(result: ToyBenchmarkResult) -> bytes:
    plt = _prepare_matplotlib()
    fig = _build_posterior_margin_trace_figure(result)
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def write_posterior_explainer_artifacts(
    output_dir: str | Path,
    *,
    result: ToyBenchmarkResult | None = None,
) -> tuple[Path, Path]:
    benchmark_result = result or run_toy_benchmark()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    markdown_path = output_root / "toy_1d_posterior_walkthrough.md"
    png_path = output_root / "toy_1d_posterior_walkthrough.png"
    markdown_path.write_text(render_posterior_explainer_markdown(benchmark_result), encoding="utf-8")
    png_path.write_bytes(render_posterior_explainer_png_bytes(benchmark_result))
    return markdown_path, png_path


def write_posterior_failure_artifacts(
    output_dir: str | Path,
    *,
    result: ToyBenchmarkResult | None = None,
) -> tuple[Path, Path]:
    benchmark_result = result or run_toy_benchmark()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    markdown_path = output_root / "toy_1d_posterior_failure_walkthrough.md"
    png_path = output_root / "toy_1d_posterior_failure_walkthrough.png"
    markdown_path.write_text(render_posterior_failure_markdown(benchmark_result), encoding="utf-8")
    png_path.write_bytes(render_posterior_failure_png_bytes(benchmark_result))
    return markdown_path, png_path


def write_posterior_comparison_artifacts(
    output_dir: str | Path,
    *,
    result: ToyBenchmarkResult | None = None,
) -> tuple[Path, Path]:
    benchmark_result = result or run_toy_benchmark()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    markdown_path = output_root / "toy_1d_posterior_comparison.md"
    png_path = output_root / "toy_1d_posterior_comparison.png"
    markdown_path.write_text(render_posterior_comparison_markdown(benchmark_result), encoding="utf-8")
    png_path.write_bytes(render_posterior_comparison_png_bytes(benchmark_result))
    return markdown_path, png_path


def write_posterior_margin_trace_artifacts(
    output_dir: str | Path,
    *,
    result: ToyBenchmarkResult | None = None,
) -> tuple[Path, Path]:
    benchmark_result = result or run_toy_benchmark()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    markdown_path = output_root / "toy_1d_posterior_margin_trace.md"
    png_path = output_root / "toy_1d_posterior_margin_trace.png"
    markdown_path.write_text(render_posterior_margin_trace_markdown(benchmark_result), encoding="utf-8")
    png_path.write_bytes(render_posterior_margin_trace_png_bytes(benchmark_result))
    return markdown_path, png_path
