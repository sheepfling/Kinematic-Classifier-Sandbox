from __future__ import annotations

import io
import math
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

from matplotlib.patches import FancyBboxPatch

from kinematic_classifier_sandbox.reports.markdown import MarkdownDocument

from .registry.catalog import METHOD_CATALOG, MethodEntry
from .utils.math import _logsumexp
from .utils.plotting import plt
from .utils.runtime import repo_root as package_repo_root
from .witnesses.identity_1d.core import IdentityBenchmarkResult, run_identity_benchmark
from .witnesses.identity_1d.posterior_explainer import (
    _select_failure_walkthrough as _select_identity_failure_walkthrough,
)
from .witnesses.toy_1d.core import ToyBenchmarkResult, run_toy_benchmark
from .witnesses.toy_1d.posterior_explainer import (
    _select_failure_walkthrough as _select_toy_failure_walkthrough,
)


class ArtifactPaths(NamedTuple):
    markdown_path: Path
    png_path: Path


class EquationCaseRows(NamedTuple):
    rows: list[dict[str, float | str]]
    log_norm: float


def render_method_survey_markdown(entries: tuple[MethodEntry, ...] = METHOD_CATALOG) -> str:
    grouped: dict[str, list[MethodEntry]] = defaultdict(list)
    for entry in entries:
        grouped[entry.family].append(entry)

    report = MarkdownDocument("Kinematic Method Survey Summary")
    report.paragraph("This generated summary groups the initial sandbox method landscape by family.")

    for family in sorted(grouped):
        report.heading(family.replace("_", " ").title(), level=2)
        for entry in grouped[family]:
            strengths = "; ".join(entry.strengths)
            limits = "; ".join(entry.limits)
            use_cases = ", ".join(entry.typical_use_cases)
            inputs = ", ".join(entry.typical_inputs)
            report.heading(entry.name, level=3)
            report.bullet_list(
                [
                    f"Style: `{entry.style}`",
                    f"Typical inputs: {inputs}",
                    f"Typical use cases: {use_cases}",
                    f"Strengths: {strengths}",
                    f"Limits: {limits}",
                ]
            )

    report.heading("Current Recommended Sandbox Baseline", level=2)
    report.paragraph(
        "The strongest initial model-based baseline is a Bayesian joint tracking and"
        "classification stack with a class-matched filter bank, IMM-style within-class"
        "mode switching, covariance-aware constraint likelihoods, optional aerodynamic"
        "parameter evidence, and explicit unknown-class handling."
    )
    return report.text()


def write_method_survey_artifact(output_dir: str | Path) -> Path:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "method_survey_summary.md"
    output_path.write_text(render_method_survey_markdown(), encoding="utf-8")
    return output_path


def render_posterior_math_markdown() -> str:
    repo_root = package_repo_root()
    note_path = repo_root / "docs" / "surveys" / "posterior_update_math.md"
    return note_path.read_text(encoding="utf-8")


def _draw_box(ax, xy: tuple[float, float], width: float, height: float, text: str, facecolor: str) -> None:

    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.4,
        edgecolor="#1f2937",
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2.0,
        xy[1] + height / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=9,
        color="#111827",
        wrap=True,
    )


def _draw_arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "->", "linewidth": 1.6, "color": "#374151"},
    )


def _build_posterior_math_figure():
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    toy_ax, identity_ax = axes

    for ax in axes:
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.axis("off")

    toy_ax.set_title("Toy 1D: Latent-State Class Posterior Update", loc="left", fontsize=13, fontweight="bold")
    _draw_box(toy_ax, (0.03, 0.58), 0.18, 0.20, "Measurement\n$z_t = p_t + \\nu_t$", "#dbeafe")
    _draw_box(toy_ax, (0.28, 0.58), 0.18, 0.20, "Per-class prediction\n$\\hat{x}_{i,t}^-, P_{i,t}^-$", "#e0e7ff")
    _draw_box(toy_ax, (0.53, 0.58), 0.19, 0.20, "Innovation term\n$\\log L^{dyn}_{i,t}$", "#ede9fe")
    _draw_box(toy_ax, (0.78, 0.58), 0.18, 0.20, "Posterior update\n$\\log \\tilde{w}_{i,t} = \\log w_{i,t-1} + \\log L_{i,t}$", "#dcfce7")
    _draw_arrow(toy_ax, (0.21, 0.68), (0.28, 0.68))
    _draw_arrow(toy_ax, (0.46, 0.68), (0.53, 0.68))
    _draw_arrow(toy_ax, (0.72, 0.68), (0.78, 0.68))

    _draw_box(
        toy_ax,
        (0.14, 0.18),
        0.24,
        0.20,
        "Soft envelope terms\n$\\log L^{speed}_{i,t}$\n$\\log L^{accel}_{i,t}$",
        "#fef3c7",
    )
    _draw_box(
        toy_ax,
        (0.41, 0.18),
        0.24,
        0.20,
        "Behavior and observed terms\nvelocity/accel centers\nfinite-difference kinematics",
        "#fde68a",
    )
    _draw_box(
        toy_ax,
        (0.68, 0.18),
        0.22,
        0.20,
        "Mode and unknown terms\n$\\log L^{mode}_{i,t}$\n$-\\gamma_{unknown}$",
        "#fecaca",
    )
    _draw_arrow(toy_ax, (0.26, 0.38), (0.58, 0.58))
    _draw_arrow(toy_ax, (0.53, 0.38), (0.61, 0.58))
    _draw_arrow(toy_ax, (0.79, 0.38), (0.67, 0.58))
    toy_ax.text(
        0.02,
        0.03,
        "Composite toy score: dyn + speed + accel + behavior + observed + mode - unknown penalty",
        fontsize=9.5,
        color="#374151",
    )

    identity_ax.set_title("Identity 1D: Direct Speed-Class Posterior Update", loc="left", fontsize=13, fontweight="bold")
    _draw_box(identity_ax, (0.03, 0.58), 0.18, 0.20, "Measurement\n$z_t =$ observed speed", "#dbeafe")
    _draw_box(identity_ax, (0.28, 0.58), 0.20, 0.20, "Base speed shape\n$\\log L^{speed-shape}_{i,t}$", "#e0e7ff")
    _draw_box(identity_ax, (0.55, 0.58), 0.18, 0.20, "History/mode/dynamics\nshort-window class fit", "#ede9fe")
    _draw_box(identity_ax, (0.79, 0.58), 0.17, 0.20, "Posterior update\nnormalize across\nbike / horse / car", "#dcfce7")
    _draw_arrow(identity_ax, (0.21, 0.68), (0.28, 0.68))
    _draw_arrow(identity_ax, (0.48, 0.68), (0.55, 0.68))
    _draw_arrow(identity_ax, (0.73, 0.68), (0.79, 0.68))

    _draw_box(
        identity_ax,
        (0.18, 0.18),
        0.24,
        0.20,
        "Soft validity\n$1.4 \\log P(z_t \\leq v_{max,i}+\\delta_i)$",
        "#fef3c7",
    )
    _draw_box(
        identity_ax,
        (0.47, 0.18),
        0.20,
        0.20,
        "Mode mixture\ncruise / pack / border\nor class-specific regimes",
        "#fde68a",
    )
    _draw_box(
        identity_ax,
        (0.72, 0.18),
        0.22,
        0.20,
        "Diagnostics only\nfeatures, entropy,\nconfusion matrices",
        "#fecaca",
    )
    _draw_arrow(identity_ax, (0.30, 0.38), (0.37, 0.58))
    _draw_arrow(identity_ax, (0.57, 0.38), (0.61, 0.58))
    _draw_arrow(identity_ax, (0.83, 0.38), (0.87, 0.58))
    identity_ax.text(
        0.02,
        0.03,
        "Composite identity score: speed_shape + speed_validity + history_shape + mode_shape + dynamics_shape",
        fontsize=9.5,
        color="#374151",
    )

    fig.suptitle("Posterior Update Math Overview", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def render_posterior_math_png_bytes() -> bytes:
    fig = _build_posterior_math_figure()
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=170, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def write_posterior_math_artifacts(output_dir: str | Path) -> ArtifactPaths:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    markdown_path = output_root / "posterior_update_math.md"
    png_path = output_root / "posterior_update_math.png"
    markdown_path.write_text(render_posterior_math_markdown(), encoding="utf-8")
    png_path.write_bytes(render_posterior_math_png_bytes())
    return ArtifactPaths(markdown_path=markdown_path, png_path=png_path)


def _normal_pdf(x: float, mean: float, sigma: float) -> float:
    variance = sigma * sigma
    return math.exp(-0.5 * ((x - mean) ** 2) / variance) / math.sqrt(2.0 * math.pi * variance)


def _normal_cdf(x: float, mean: float, sigma: float) -> float:
    z = (x - mean) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def render_probability_primitives_markdown() -> str:
    report = MarkdownDocument("Probability Primitive Charts")
    report.paragraph("This artifact isolates the simple probability shapes used inside the 1D classifiers.")
    report.heading("What The Charts Show", level=2)
    report.bullet_list(
        [
            "`Innovation density`: a Gaussian measurement-fit term. This is a PDF-shaped score on the residual itself.",
            "`Symmetric soft limit`: interval probability mass inside `[-c, c]`. This is how toy envelope terms act on velocity or acceleration.",
            "`One-sided soft limit`: cumulative Gaussian mass below an upper bound. This is how identity speed validity acts.",
            "`Prior plus score to posterior`: one concrete bar-chart example showing how prior class mass and composite score combine into posterior mass.",
        ]
    )
    report.heading("PDF Versus CDF", level=2)
    report.bullet_list(
        [
            "The innovation chart is a PDF or log-PDF style term: it scores one exact residual.",
            "The soft-limit charts are CDF-derived region-probability terms: they integrate mass over an allowed region.",
        ]
    )
    report.heading("Why This Matters", level=2)
    report.bullet_list(
        [
            "Farther residuals get lower innovation likelihood.",
            "Broader uncertainty can still earn useful soft-limit mass if much of the distribution remains inside the valid region.",
            "Small violations of a hard bound are not collapsed to zero. They are penalized softly through remaining probability mass.",
        ]
    )
    return report.text()


def _build_probability_primitives_figure():
    toy_benchmark = run_toy_benchmark()
    identity_benchmark = run_identity_benchmark()
    toy_walkthrough = _select_toy_failure_walkthrough(toy_benchmark)
    identity_walkthrough = _select_identity_failure_walkthrough(identity_benchmark)

    toy_bundle = _equation_case_rows(
        toy_walkthrough.class_names,
        toy_walkthrough.prior_weights,
        toy_walkthrough.posterior_weights,
        toy_walkthrough.log_terms,
    )
    toy_rows = toy_bundle.rows
    toy_log_norm = toy_bundle.log_norm
    identity_bundle = _equation_case_rows(
        identity_walkthrough.class_names,
        identity_walkthrough.prior_weights,
        identity_walkthrough.posterior_weights,
        identity_walkthrough.log_terms,
    )
    identity_log_norm = identity_bundle.log_norm

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    innovation_ax, interval_ax, upper_ax, posterior_ax = axes.flat

    sigma = 1.0
    xs = [(-4.0 + 0.02 * i) for i in range(401)]
    pdf_values = [_normal_pdf(x, 0.0, sigma) for x in xs]
    innovation_ax.plot(xs, pdf_values, color="#2563eb", linewidth=2.3)
    for residual, color, label in ((0.3, "#16a34a", "small residual"), (1.4, "#f59e0b", "larger residual")):
        innovation_ax.axvline(residual, color=color, linestyle="--", linewidth=1.5)
        innovation_ax.scatter([residual], [_normal_pdf(residual, 0.0, sigma)], color=color, zorder=3)
        innovation_ax.text(residual + 0.08, _normal_pdf(residual, 0.0, sigma) + 0.01, label, fontsize=8, color=color)
    innovation_ax.set_title("Innovation Likelihood", loc="left", fontsize=12, fontweight="bold")
    innovation_ax.set_xlabel("innovation residual")
    innovation_ax.set_ylabel("Gaussian density")
    innovation_ax.grid(True, alpha=0.25)

    mean_interval = 0.6
    sigma_interval = 0.9
    limit = 1.2
    interval_pdf = [_normal_pdf(x, mean_interval, sigma_interval) for x in xs]
    interval_ax.plot(xs, interval_pdf, color="#0f766e", linewidth=2.3)
    fill_x = [x for x in xs if -limit <= x <= limit]
    fill_y = [_normal_pdf(x, mean_interval, sigma_interval) for x in fill_x]
    interval_ax.fill_between(fill_x, fill_y, color="#99f6e4", alpha=0.9)
    interval_ax.axvline(-limit, color="#991b1b", linestyle="--", linewidth=1.3)
    interval_ax.axvline(limit, color="#991b1b", linestyle="--", linewidth=1.3)
    interval_mass = _normal_cdf(limit, mean_interval, sigma_interval) - _normal_cdf(-limit, mean_interval, sigma_interval)
    interval_ax.text(0.03, 0.92, f"soft mass = {interval_mass:.3f}", transform=interval_ax.transAxes, fontsize=9, color="#134e4a")
    interval_ax.set_title("Toy Symmetric Soft Limit", loc="left", fontsize=12, fontweight="bold")
    interval_ax.set_xlabel("state component")
    interval_ax.set_ylabel("Gaussian density")
    interval_ax.grid(True, alpha=0.25)

    mean_upper = 22.5
    sigma_upper = 2.4
    upper_limit = 24.0
    upper_xs = [10.0 + 0.08 * i for i in range(301)]
    upper_pdf = [_normal_pdf(x, mean_upper, sigma_upper) for x in upper_xs]
    upper_ax.plot(upper_xs, upper_pdf, color="#7c3aed", linewidth=2.3)
    fill_upper_x = [x for x in upper_xs if x <= upper_limit]
    fill_upper_y = [_normal_pdf(x, mean_upper, sigma_upper) for x in fill_upper_x]
    upper_ax.fill_between(fill_upper_x, fill_upper_y, color="#ddd6fe", alpha=0.95)
    upper_ax.axvline(upper_limit, color="#991b1b", linestyle="--", linewidth=1.3)
    upper_mass = _normal_cdf(upper_limit, mean_upper, sigma_upper)
    upper_ax.text(0.03, 0.92, f"soft mass = {upper_mass:.3f}", transform=upper_ax.transAxes, fontsize=9, color="#4c1d95")
    upper_ax.set_title("Identity One-Sided Soft Limit", loc="left", fontsize=12, fontweight="bold")
    upper_ax.set_xlabel("observed speed")
    upper_ax.set_ylabel("Gaussian density")
    upper_ax.grid(True, alpha=0.25)

    class_names = [str(row["class_name"]) for row in toy_rows[:3]]
    prior_vals = [float(row["prior"]) for row in toy_rows[:3]]
    total_vals = [float(row["total"]) for row in toy_rows[:3]]
    posterior_vals = [float(row["posterior"]) for row in toy_rows[:3]]
    x = list(range(len(class_names)))
    posterior_ax.bar([i - 0.25 for i in x], prior_vals, width=0.22, label="prior", color="#60a5fa")
    posterior_ax.bar(x, total_vals, width=0.22, label="total score", color="#f59e0b")
    posterior_ax.bar([i + 0.25 for i in x], posterior_vals, width=0.22, label="posterior", color="#8b5cf6")
    posterior_ax.axhline(0.0, color="#9ca3af", linewidth=1.0)
    posterior_ax.set_xticks(x, class_names)
    posterior_ax.set_title("Prior + Score -> Posterior", loc="left", fontsize=12, fontweight="bold")
    posterior_ax.set_ylabel("mixed scale view")
    posterior_ax.grid(True, axis="y", alpha=0.25)
    posterior_ax.legend(frameon=False, fontsize=8)
    posterior_ax.text(
        0.03,
        0.04,
        f"toy log Z = {toy_log_norm:.3f}\nidentity log Z = {identity_log_norm:.3f}",
        transform=posterior_ax.transAxes,
        fontsize=8.5,
        color="#374151",
    )

    fig.suptitle("Probability Primitives Behind the 1D Classifiers", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def render_probability_primitives_png_bytes() -> bytes:
    fig = _build_probability_primitives_figure()
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=170, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def write_probability_primitives_artifacts(output_dir: str | Path) -> ArtifactPaths:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    markdown_path = output_root / "probability_primitives.md"
    png_path = output_root / "probability_primitives.png"
    markdown_path.write_text(render_probability_primitives_markdown(), encoding="utf-8")
    png_path.write_bytes(render_probability_primitives_png_bytes())
    return ArtifactPaths(markdown_path=markdown_path, png_path=png_path)


def _equation_case_rows(
    class_names: tuple[str, ...],
    prior_weights: dict[str, float],
    posterior_weights: dict[str, float],
    log_terms: dict[str, dict[str, float]],
) -> EquationCaseRows:
    rows: list[dict[str, float | str]] = []
    log_numerators: list[float] = []
    for class_name in class_names:
        prior = max(float(prior_weights[class_name]), 1e-12)
        total = float(log_terms[class_name]["total"])
        log_prior = math.log(prior)
        log_numerator = log_prior + total
        log_numerators.append(log_numerator)
        rows.append(
            {
                "class_name": class_name,
                "prior": prior,
                "log_prior": log_prior,
                "total": total,
                "log_numerator": log_numerator,
                "posterior": float(posterior_weights[class_name]),
            }
        )
    return EquationCaseRows(rows=rows, log_norm=_logsumexp(log_numerators))


def _render_equation_rows(
    rows: list[dict[str, float | str]],
    highlight_classes: tuple[str, ...],
) -> list[tuple[str, ...]]:
    row_by_name = {str(row["class_name"]): row for row in rows}
    ordered_names = list(highlight_classes) + [name for name in row_by_name if name not in highlight_classes]
    table_rows = []
    for class_name in ordered_names:
        row = row_by_name[class_name]
        table_rows.append(
            (
                f"`{class_name}`",
                f"{row['prior']:.3f}",
                f"{row['log_prior']:.3f}",
                f"{row['total']:.3f}",
                f"{row['log_numerator']:.3f}",
                f"{row['posterior']:.3f}",
            )
        )
    return table_rows


def render_posterior_numeric_walkthrough_markdown(
    toy_result: ToyBenchmarkResult | None = None,
    identity_result: IdentityBenchmarkResult | None = None,
) -> str:
    toy_benchmark = toy_result or run_toy_benchmark()
    identity_benchmark = identity_result or run_identity_benchmark()
    toy_walkthrough = _select_toy_failure_walkthrough(toy_benchmark)
    identity_walkthrough = _select_identity_failure_walkthrough(identity_benchmark)

    toy_bundle = _equation_case_rows(
        toy_walkthrough.class_names,
        toy_walkthrough.prior_weights,
        toy_walkthrough.posterior_weights,
        toy_walkthrough.log_terms,
    )
    toy_rows = toy_bundle.rows
    toy_log_norm = toy_bundle.log_norm
    identity_bundle = _equation_case_rows(
        identity_walkthrough.class_names,
        identity_walkthrough.prior_weights,
        identity_walkthrough.posterior_weights,
        identity_walkthrough.log_terms,
    )
    identity_rows = identity_bundle.rows
    identity_log_norm = identity_bundle.log_norm

    toy_true = toy_walkthrough.run.true_class
    toy_pred = toy_walkthrough.run.aggregate_map_class
    identity_true = identity_walkthrough.run.expected_class
    identity_pred = identity_walkthrough.run.aggregate_map_class

    report = MarkdownDocument("Posterior Equation Numeric Walkthrough")
    report.paragraph(
        "This artifact ties the implemented posterior equations directly to real numeric values from one toy failure case and one identity failure case."
    )
    report.heading("Generic Update", level=2)
    report.paragraph("`log numerator_i = log prior_i + total_i`")
    report.paragraph("`posterior_i = exp(log numerator_i - log Z)`")
    report.paragraph("where `log Z = logsumexp(log numerator_j)` across all classes.")
    report.heading("Toy Case", level=2)
    report.bullet_list(
        [
            f"Scenario: `{toy_walkthrough.run.scenario_name}`",
            f"True class: `{toy_true}`",
            f"Aggregate predicted class: `{toy_pred}`",
            f"Step: `{toy_walkthrough.step_index + 1}`",
            f"Measurement: `{toy_walkthrough.measurement:.3f}`",
        ]
    )
    report.paragraph("Toy total score definition:")
    report.paragraph("`total_i = dyn_i + speed_i + accel_i + behavior_i + observed_i + mode_i - unknown_penalty_i`")
    report.heading("Toy numeric substitution", level=3)
    report.table(
        ["Class", "Prior", "log prior", "total score", "log numerator", "posterior"],
        _render_equation_rows(toy_rows, (toy_true, toy_pred)),
    )
    report.paragraph(f"Normalizer: `log Z = {toy_log_norm:.3f}`")
    report.bullet_list(
        [
            f"`{toy_true}` loses when its `log prior + total` stays below `{toy_pred}` after normalization.",
            "The existing toy posterior artifacts break `total` into `dyn`, `speed`, `accel`, behavior, observed, and mode terms.",
        ]
    )
    report.heading("Identity Case", level=2)
    report.bullet_list(
        [
            f"Scenario: `{identity_walkthrough.run.scenario_name}`",
            f"True class: `{identity_true}`",
            f"Aggregate predicted class: `{identity_pred}`",
            f"Step: `{identity_walkthrough.step_index + 1}`",
            f"Measurement: `{identity_walkthrough.measurement:.3f} mph`",
        ]
    )
    report.paragraph("Identity total score definition:")
    report.paragraph("`total_i = speed_shape_i + speed_validity_i + history_shape_i + mode_shape_i + dynamics_shape_i`")
    report.heading("Identity numeric substitution", level=3)
    report.table(
        ["Class", "Prior", "log prior", "total score", "log numerator", "posterior"],
        _render_equation_rows(identity_rows, (identity_true, identity_pred)),
    )
    report.paragraph(f"Normalizer: `log Z = {identity_log_norm:.3f}`")
    report.bullet_list(
        [
            f"`{identity_true}` loses when its combined speed-shape and short-window terms remain below `{identity_pred}`.",
            "In the current identity boundary failures, the decisive term is usually the base `speed_shape` score.",
        ]
    )
    report.heading("How To Read This With The Other Artifacts", level=2)
    report.bullet_list(
        [
            "Use this note for the scalar algebra at one update step.",
            "Use the posterior walkthrough artifacts for the full per-term decomposition by class.",
            "Use the margin-trace artifacts to see when the sign of the decision margin flips over time.",
            "Use the confusion matrices to see how those local update behaviors accumulate over many runs.",
        ]
    )
    return report.text()


def _build_numeric_walkthrough_table(ax, title: str, rows: list[dict[str, float | str]], log_norm: float) -> None:
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    table_rows = [
        [
            str(row["class_name"]),
            f"{row['prior']:.3f}",
            f"{row['log_prior']:.3f}",
            f"{row['total']:.3f}",
            f"{row['log_numerator']:.3f}",
            f"{row['posterior']:.3f}",
        ]
        for row in rows
    ]
    table = ax.table(
        cellText=table_rows,
        colLabels=["class", "prior", "log prior", "total", "log num", "posterior"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.35)
    ax.text(0.02, 0.05, f"log Z = {log_norm:.3f}", transform=ax.transAxes, fontsize=9, color="#374151")


def render_posterior_numeric_walkthrough_png_bytes(
    toy_result: ToyBenchmarkResult | None = None,
    identity_result: IdentityBenchmarkResult | None = None,
) -> bytes:
    toy_benchmark = toy_result or run_toy_benchmark()
    identity_benchmark = identity_result or run_identity_benchmark()
    toy_walkthrough = _select_toy_failure_walkthrough(toy_benchmark)
    identity_walkthrough = _select_identity_failure_walkthrough(identity_benchmark)
    toy_bundle = _equation_case_rows(
        toy_walkthrough.class_names,
        toy_walkthrough.prior_weights,
        toy_walkthrough.posterior_weights,
        toy_walkthrough.log_terms,
    )
    toy_rows = toy_bundle.rows
    toy_log_norm = toy_bundle.log_norm
    identity_bundle = _equation_case_rows(
        identity_walkthrough.class_names,
        identity_walkthrough.prior_weights,
        identity_walkthrough.posterior_weights,
        identity_walkthrough.log_terms,
    )
    identity_rows = identity_bundle.rows
    identity_log_norm = identity_bundle.log_norm

    fig, axes = plt.subplots(2, 1, figsize=(12, 8.5))
    _build_numeric_walkthrough_table(
        axes[0],
        f"Toy: {toy_walkthrough.run.scenario_name} | true={toy_walkthrough.run.true_class} | predicted={toy_walkthrough.run.aggregate_map_class}",
        toy_rows,
        toy_log_norm,
    )
    _build_numeric_walkthrough_table(
        axes[1],
        f"Identity: {identity_walkthrough.run.scenario_name} | true={identity_walkthrough.run.expected_class} | predicted={identity_walkthrough.run.aggregate_map_class}",
        identity_rows,
        identity_log_norm,
    )
    fig.suptitle("Posterior Equation Numeric Walkthrough", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=170, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def write_posterior_numeric_walkthrough_artifacts(
    output_dir: str | Path,
    *,
    toy_result: ToyBenchmarkResult | None = None,
    identity_result: IdentityBenchmarkResult | None = None,
) -> ArtifactPaths:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    markdown_path = output_root / "posterior_numeric_walkthrough.md"
    png_path = output_root / "posterior_numeric_walkthrough.png"
    markdown_path.write_text(
        render_posterior_numeric_walkthrough_markdown(toy_result=toy_result, identity_result=identity_result),
        encoding="utf-8",
    )
    png_path.write_bytes(
        render_posterior_numeric_walkthrough_png_bytes(toy_result=toy_result, identity_result=identity_result)
    )
    return ArtifactPaths(markdown_path=markdown_path, png_path=png_path)
