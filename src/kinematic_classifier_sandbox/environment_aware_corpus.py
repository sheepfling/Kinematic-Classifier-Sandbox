from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .backend_adapter_proof import (
    BackendCandidateSpec,
    EnvironmentAware1DAdapter,
    _environment_candidate,
)
from .trajectory_backend_contract import default_backend_contract_definitions


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass(frozen=True, slots=True)
class EnvironmentAwareCorpusResult:
    environment_manifest: dict[str, Any]
    environment_coverage_rows: tuple[dict[str, Any], ...]
    environment_leakage_rows: tuple[dict[str, Any], ...]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class EnvironmentAwareCorpusArtifacts:
    run_dir: Path
    environment_manifest_path: Path
    environment_coverage_path: Path
    environment_leakage_audit_path: Path
    report_path: Path
    coverage_heatmap_png_path: Path
    leakage_plot_png_path: Path
    trajectory_gallery_png_path: Path


def _environment_adapter() -> EnvironmentAware1DAdapter:
    definition = {
        definition.capabilities.backend_id: definition
        for definition in default_backend_contract_definitions()
    }["environment_aware_1d"]
    return EnvironmentAware1DAdapter(definition)


def _environment_regimes() -> tuple[dict[str, Any], ...]:
    return (
        {
            "environment_id": "dense_calm",
            "density_scale": 1.10,
            "wind_bias": 0.00,
            "drag_coefficient": 0.28,
            "description": "Dense and calm reference regime.",
        },
        {
            "environment_id": "nominal_mixed",
            "density_scale": 1.00,
            "wind_bias": 0.05,
            "drag_coefficient": 0.20,
            "description": "Nominal density with mild wind bias.",
        },
        {
            "environment_id": "thin_windy",
            "density_scale": 0.82,
            "wind_bias": 0.12,
            "drag_coefficient": 0.12,
            "description": "Thin and wind-biased regime.",
        },
    )


def _class_specs() -> tuple[dict[str, Any], ...]:
    return (
        {"target_class": "constant_velocity", "acceleration": 0.04, "initial_velocity": 1.05},
        {"target_class": "constant_acceleration", "acceleration": 0.42, "initial_velocity": 0.78},
    )


def _candidate_rows() -> tuple[BackendCandidateSpec, ...]:
    base = _environment_candidate()
    candidates: list[BackendCandidateSpec] = []
    seed = 500
    for class_index, class_spec in enumerate(_class_specs()):
        for regime_index, regime in enumerate(_environment_regimes()):
            for replicate in range(2):
                candidates.append(
                    BackendCandidateSpec(
                        candidate_id=f"{class_spec['target_class']}_{regime['environment_id']}_{replicate}",
                        scenario_id=f"environment_{class_spec['target_class']}_{regime['environment_id']}",
                        scenario_family="environment_regime_case",
                        target_class=str(class_spec["target_class"]),
                        difficulty_tier="realistic_v1",
                        seed=seed + class_index * 100 + regime_index * 10 + replicate,
                        duration=base.duration,
                        sample_period=base.sample_period,
                        initial_position=base.initial_position,
                        initial_velocity=float(class_spec["initial_velocity"]) + 0.03 * replicate,
                        acceleration=float(class_spec["acceleration"]),
                        measurement_std=base.measurement_std,
                        drag_coefficient=float(regime["drag_coefficient"]),
                        density_scale=float(regime["density_scale"]),
                        wind_bias=float(regime["wind_bias"]),
                        provenance={
                            "search_method": "environment_regime_targeting",
                            "search_iteration": len(candidates),
                            "environment_id": regime["environment_id"],
                        },
                    )
                )
    return tuple(candidates)


def _trajectory_summary_row(candidate: BackendCandidateSpec, run: dict[str, Any]) -> dict[str, Any]:
    density_trace = run["environment_trace"]["density_scale"]
    wind_trace = run["environment_trace"]["wind_bias"]
    velocities = run["truth_state"]["velocity"]
    positions = run["truth_state"]["position"]
    return {
        "trajectory_id": f"{candidate.candidate_id}_trajectory",
        "candidate_id": candidate.candidate_id,
        "true_class": candidate.target_class,
        "environment_id": candidate.provenance["environment_id"],
        "seed": candidate.seed,
        "duration": candidate.duration,
        "num_samples": len(run["times"]),
        "density_mean": mean(density_trace),
        "wind_bias_mean": mean(wind_trace),
        "drag_coefficient": candidate.drag_coefficient,
        "position_final": positions[-1],
        "speed_final": velocities[-1],
        "speed_range": max(velocities) - min(velocities),
        "position_range": max(positions) - min(positions),
        "environment_trace_available": True,
        "environment_feature_view": "available",
        "agnostic_feature_view": "available",
    }


def _coverage_rows(summary_rows: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in summary_rows:
        key = (str(row["environment_id"]), str(row["true_class"]))
        grouped.setdefault(key, []).append(row)
    rows: list[dict[str, Any]] = []
    for (environment_id, true_class), members in sorted(grouped.items()):
        rows.append(
            {
                "environment_id": environment_id,
                "true_class": true_class,
                "trajectory_count": len(members),
                "mean_duration": mean(float(member["duration"]) for member in members),
                "mean_density": mean(float(member["density_mean"]) for member in members),
                "mean_wind_bias": mean(float(member["wind_bias_mean"]) for member in members),
                "mean_speed_range": mean(float(member["speed_range"]) for member in members),
                "mean_position_range": mean(float(member["position_range"]) for member in members),
            }
        )
    return tuple(rows)


def _leakage_rows(summary_rows: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    selected_rows = list(summary_rows)
    biased_control_rows = [
        row for row in summary_rows
        if (row["true_class"] == "constant_velocity" and row["environment_id"] == "dense_calm")
        or (row["true_class"] == "constant_acceleration" and row["environment_id"] == "thin_windy")
    ]
    slices = {
        "selected_corpus": selected_rows,
        "biased_control_slice": biased_control_rows,
    }
    variables = ("density_mean", "wind_bias_mean", "drag_coefficient")
    rows: list[dict[str, Any]] = []
    for slice_id, members in slices.items():
        for variable in variables:
            cv_values = [float(row[variable]) for row in members if row["true_class"] == "constant_velocity"]
            ca_values = [float(row[variable]) for row in members if row["true_class"] == "constant_acceleration"]
            if not cv_values or not ca_values:
                continue
            cv_mean = mean(cv_values)
            ca_mean = mean(ca_values)
            delta_ratio = abs(cv_mean - ca_mean) / max(abs(cv_mean) + abs(ca_mean), 1e-6)
            flagged = delta_ratio >= 0.15
            rows.append(
                {
                    "slice_id": slice_id,
                    "variable_name": variable,
                    "class_a": "constant_velocity",
                    "class_b": "constant_acceleration",
                    "class_a_mean": cv_mean,
                    "class_b_mean": ca_mean,
                    "delta_ratio": delta_ratio,
                    "flagged_class_linkage": flagged,
                }
            )
    return tuple(rows)


def _render_coverage_heatmap_png(rows: tuple[dict[str, Any], ...]) -> bytes:
    environments = sorted({str(row["environment_id"]) for row in rows})
    classes = sorted({str(row["true_class"]) for row in rows})
    matrix = []
    for environment_id in environments:
        environment_row = []
        for true_class in classes:
            match = next(row for row in rows if row["environment_id"] == environment_id and row["true_class"] == true_class)
            environment_row.append(float(match["trajectory_count"]))
        matrix.append(environment_row)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    image = ax.imshow(matrix, cmap="YlOrBr", aspect="auto")
    ax.set_xticks(range(len(classes)), labels=classes, fontsize=9)
    ax.set_yticks(range(len(environments)), labels=environments, fontsize=9)
    ax.set_title("Environment Regime Coverage")
    for row_index, row_values in enumerate(matrix):
        for column_index, value in enumerate(row_values):
            ax.text(column_index, row_index, f"{value:.0f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.04)
    fig.tight_layout()

    from io import BytesIO

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def _render_leakage_plot_png(rows: tuple[dict[str, Any], ...]) -> bytes:
    labels = [f"{row['slice_id']}:{row['variable_name']}" for row in rows]
    values = [float(row["delta_ratio"]) for row in rows]
    colors = ["#ca5b4b" if bool(row["flagged_class_linkage"]) else "#4d8f77" for row in rows]

    fig, ax = plt.subplots(figsize=(10.0, 4.0))
    ax.bar(range(len(labels)), values, color=colors)
    ax.axhline(0.15, color="black", linestyle="--", linewidth=1.0, label="flag threshold")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Delta Ratio")
    ax.set_title("Environment Leakage Audit")
    ax.legend(fontsize=8)
    fig.tight_layout()

    from io import BytesIO

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def _render_trajectory_gallery_png() -> bytes:
    adapter = _environment_adapter()
    chosen_ids = (
        "constant_velocity_dense_calm_0",
        "constant_velocity_thin_windy_0",
        "constant_acceleration_dense_calm_0",
        "constant_acceleration_thin_windy_0",
    )
    chosen = [candidate for candidate in _candidate_rows() if candidate.candidate_id in chosen_ids]

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.0), sharex=True)
    for axis, candidate in zip(axes.flat, chosen):
        record = adapter.run(candidate)
        run = record.trajectory_run
        axis.plot(run.times, run.truth_state["position"], marker="o", label="position")
        axis.plot(run.times, run.truth_state["velocity"], marker="s", label="velocity")
        axis.set_title(f"{candidate.target_class}\n{candidate.provenance['environment_id']}", fontsize=9)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Environment-Conditioned Trajectory Gallery", fontsize=11)
    fig.tight_layout()

    from io import BytesIO

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def analyze_environment_aware_corpus() -> EnvironmentAwareCorpusResult:
    adapter = _environment_adapter()
    candidates = _candidate_rows()
    summary_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        record = adapter.run(candidate)
        run = record.trajectory_run
        summary_rows.append(_trajectory_summary_row(candidate, {"times": run.times, "truth_state": run.truth_state, "environment_trace": run.environment_trace}))

    coverage_rows = _coverage_rows(tuple(summary_rows))
    leakage_rows = _leakage_rows(tuple(summary_rows))
    selected_flagged = sum(1 for row in leakage_rows if row["slice_id"] == "selected_corpus" and bool(row["flagged_class_linkage"]))
    control_flagged = sum(1 for row in leakage_rows if row["slice_id"] == "biased_control_slice" and bool(row["flagged_class_linkage"]))

    environment_manifest = {
        "environment_regimes": _environment_regimes(),
        "candidate_count": len(candidates),
        "selected_classes": sorted({candidate.target_class for candidate in candidates}),
        "feature_views": {
            "agnostic": "truth_state-only summaries remain available even when environment fields are ignored",
            "environment_aware": "density, wind, and drag metadata are available from normalized environment_trace",
        },
        "search_objective": "target environment regimes while preserving class balance and traceable environment metadata",
    }

    report_markdown = "\n".join(
        [
            "# Atmosphere-Like 1D Environment Corpus",
            "",
            "## Summary",
            f"- environment regimes targeted: `{len(_environment_regimes())}`",
            f"- generated trajectories: `{len(candidates)}`",
            f"- selected corpus leakage flags: `{selected_flagged}`",
            f"- biased control leakage flags: `{control_flagged}`",
            "",
            "## Environment Coverage",
            "| Environment | Class | Count | Mean Density | Mean Wind Bias |",
            "| --- | --- | --- | --- | --- |",
            *[
                f"| `{row['environment_id']}` | `{row['true_class']}` | `{row['trajectory_count']}` | "
                f"`{row['mean_density']:.3f}` | `{row['mean_wind_bias']:.3f}` |"
                for row in coverage_rows
            ],
            "",
            "## Leakage Audit",
            "| Slice | Variable | Delta Ratio | Flagged |",
            "| --- | --- | --- | --- |",
            *[
                f"| `{row['slice_id']}` | `{row['variable_name']}` | `{row['delta_ratio']:.3f}` | `{row['flagged_class_linkage']}` |"
                for row in leakage_rows
            ],
            "",
            "## Notes",
            "- The selected corpus is balanced across environment regimes and classes, so the main slice should remain mostly unflagged.",
            "- A biased control slice is audited alongside it to prove that the leakage logic can surface class-linked environment variables when they are present.",
            "- The same normalized runs support both environment-agnostic and environment-aware feature views because `truth_state` and `environment_trace` are preserved separately.",
        ]
    )

    return EnvironmentAwareCorpusResult(
        environment_manifest=environment_manifest,
        environment_coverage_rows=coverage_rows,
        environment_leakage_rows=leakage_rows,
        report_markdown=report_markdown,
    )


def write_environment_aware_corpus_artifacts(
    base_dir: str | Path,
    *,
    result: EnvironmentAwareCorpusResult | None = None,
) -> EnvironmentAwareCorpusArtifacts:
    run_dir = Path(base_dir) / "environment_aware_corpus"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = result or analyze_environment_aware_corpus()

    environment_manifest_path = run_dir / "environment_manifest.json"
    environment_coverage_path = run_dir / "environment_coverage.csv"
    environment_leakage_audit_path = run_dir / "environment_leakage_audit.csv"
    report_path = run_dir / "atmosphere_like_1d_report.md"
    coverage_heatmap_png_path = run_dir / "environment_regime_coverage_heatmap.png"
    leakage_plot_png_path = run_dir / "environment_variable_leakage_by_class.png"
    trajectory_gallery_png_path = run_dir / "environment_conditioned_trajectory_gallery.png"

    environment_manifest_path.write_text(json.dumps(payload.environment_manifest, indent=2), encoding="utf-8")
    report_path.write_text(payload.report_markdown, encoding="utf-8")

    coverage_fieldnames = list(payload.environment_coverage_rows[0].keys()) if payload.environment_coverage_rows else []
    leakage_fieldnames = list(payload.environment_leakage_rows[0].keys()) if payload.environment_leakage_rows else []
    _write_csv(environment_coverage_path, list(payload.environment_coverage_rows), coverage_fieldnames)
    _write_csv(environment_leakage_audit_path, list(payload.environment_leakage_rows), leakage_fieldnames)

    coverage_heatmap_png_path.write_bytes(_render_coverage_heatmap_png(payload.environment_coverage_rows))
    leakage_plot_png_path.write_bytes(_render_leakage_plot_png(payload.environment_leakage_rows))
    trajectory_gallery_png_path.write_bytes(_render_trajectory_gallery_png())

    return EnvironmentAwareCorpusArtifacts(
        run_dir=run_dir,
        environment_manifest_path=environment_manifest_path,
        environment_coverage_path=environment_coverage_path,
        environment_leakage_audit_path=environment_leakage_audit_path,
        report_path=report_path,
        coverage_heatmap_png_path=coverage_heatmap_png_path,
        leakage_plot_png_path=leakage_plot_png_path,
        trajectory_gallery_png_path=trajectory_gallery_png_path,
    )
