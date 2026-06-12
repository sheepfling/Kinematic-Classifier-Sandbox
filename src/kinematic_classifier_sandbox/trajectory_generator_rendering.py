from __future__ import annotations

import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from kinematic_classifier_sandbox.reports.markdown import MarkdownDocument
from kinematic_classifier_sandbox.utils.io import write_csv
from kinematic_classifier_sandbox.utils.math import mean as _mean
from kinematic_classifier_sandbox.utils.plotting import plt

from .contracts import TrajectoryArtifact, validate_trajectory_artifact
from .trajectory_generator import (
    GeneratedTrajectoryDataset,
    default_trajectory_class_definitions,
    generate_trajectory_datasets,
)
from .witnesses.trajectory_scenarios import (
    generate_perturbation_sweep_scenarios,
    generate_short_horizon_scenarios,
    generate_switching_scenarios,
)


@dataclass(frozen=True, slots=True)
class TrajectoryGeneratorArtifacts:
    run_dir: Path
    report_path: Path
    class_definitions_path: Path
    config_path: Path
    dataset_manifest_paths: dict[str, Path]
    generated_trajectories_paths: dict[str, Path]
    true_states_paths: dict[str, Path]
    supplemental_manifest_paths: dict[str, Path]
    supplemental_generated_paths: dict[str, Path]
    supplemental_true_states_paths: dict[str, Path]
    plot_png_path: Path


def write_trajectory_generator_artifacts(
    output_dir: str | Path,
    *,
    seed: int = 7,
    trajectories_per_class: int | None = None,
) -> TrajectoryGeneratorArtifacts:
    datasets = generate_trajectory_datasets(seed=seed, trajectories_per_class=trajectories_per_class)
    output_root = Path(output_dir)
    run_dir = output_root / "trajectory_generator_v1"
    run_dir.mkdir(parents=True, exist_ok=True)

    class_definitions_path = run_dir / "class_definitions.json"
    config_path = run_dir / "trajectory_generator_config.yaml"
    report_path = run_dir / "trajectory_generator_report.md"
    plot_png_path = run_dir / "trajectory_generator_overview.png"

    class_definitions_path.write_text(
        json.dumps([asdict(class_definition) for class_definition in default_trajectory_class_definitions()], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  name: trajectory_generator_v1",
                f"  seed: {seed}",
                f"  trajectories_per_class: {trajectories_per_class if trajectories_per_class is not None else 'default'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    report_path.write_text(_render_trajectory_generator_report(datasets), encoding="utf-8")
    plot_png_path.write_bytes(_render_figure_png(_render_dataset_plot(datasets[0])))

    dataset_manifest_paths: dict[str, Path] = {}
    generated_paths: dict[str, Path] = {}
    true_state_paths: dict[str, Path] = {}
    for dataset in datasets:
        manifest_path = run_dir / f"{dataset.tier}_dataset_manifest.json"
        trajectories_path = run_dir / f"{dataset.tier}_generated_trajectories.csv"
        true_states_path = run_dir / f"{dataset.tier}_true_states.csv"
        dataset_manifest_paths[dataset.tier] = manifest_path
        generated_paths[dataset.tier] = trajectories_path
        true_state_paths[dataset.tier] = true_states_path
        manifest_path.write_text(
            json.dumps(_dataset_manifest(dataset), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        write_csv(
            trajectories_path,
            _trajectory_rows(dataset),
            [
                "trajectory_id",
                "tier",
                "scenario_id",
                "true_class",
                "seed",
                "step",
                "time",
                "measurement",
                "true_position",
                "true_velocity",
                "true_acceleration",
            ],
        )
        write_csv(
            true_states_path,
            _true_state_rows(dataset),
            [
                "trajectory_id",
                "tier",
                "scenario_id",
                "true_class",
                "seed",
                "step",
                "time",
                "true_position",
                "true_velocity",
                "true_acceleration",
            ],
        )
        for trajectory in dataset.trajectories:
            errors = validate_trajectory_artifact(trajectory)
            if errors:
                raise ValueError(f"invalid generated trajectory {trajectory.trajectory_id}: {errors}")

    supplemental_specs = {
        "short_horizon_v1": (
            generate_short_horizon_scenarios(seed=seed),
            [
                "Explicit short-horizon cases for CV, CA, braking, and maneuver discrimination.",
                "These scenarios complement the tiered datasets with named few-sample boundary cases.",
            ],
        ),
        "perturbation_sweeps_v1": (
            generate_perturbation_sweep_scenarios(seed=seed),
            [
                "Explicit noise, outlier, and irregular-`dt` sweep scenarios.",
                "These scenarios make perturbation response rerunnable without reading tier internals.",
            ],
        ),
        "switching_scenarios_v1": (
            generate_switching_scenarios(seed=seed),
            [
                "Explicit switching-mode trajectories for later M16 transition studies.",
                "These scenarios close the remaining M9 gap around switching coverage.",
            ],
        ),
    }
    supplemental_manifest_paths: dict[str, Path] = {}
    supplemental_generated_paths: dict[str, Path] = {}
    supplemental_true_state_paths: dict[str, Path] = {}
    for name, (trajectories, notes) in supplemental_specs.items():
        manifest_path = run_dir / f"{name}_manifest.json"
        trajectories_path = run_dir / f"{name}_generated_trajectories.csv"
        true_states_path = run_dir / f"{name}_true_states.csv"
        supplemental_manifest_paths[name] = manifest_path
        supplemental_generated_paths[name] = trajectories_path
        supplemental_true_state_paths[name] = true_states_path
        manifest_path.write_text(
            json.dumps(_trajectory_manifest(name, trajectories, notes=notes), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        write_csv(
            trajectories_path,
            [
                {
                    "trajectory_id": trajectory.trajectory_id,
                    "tier": trajectory.generator_parameters.get("tier", ""),
                    "scenario_id": trajectory.scenario_id,
                    "true_class": trajectory.true_class,
                    "seed": trajectory.seed,
                    "step": index,
                    "time": time,
                    "measurement": trajectory.measurements[index],
                    "true_position": trajectory.true_position[index] if trajectory.true_position else "",
                    "true_velocity": trajectory.true_velocity[index] if trajectory.true_velocity else "",
                    "true_acceleration": trajectory.true_acceleration[index] if trajectory.true_acceleration else "",
                }
                for trajectory in trajectories
                for index, time in enumerate(trajectory.times)
            ],
            [
                "trajectory_id",
                "tier",
                "scenario_id",
                "true_class",
                "seed",
                "step",
                "time",
                "measurement",
                "true_position",
                "true_velocity",
                "true_acceleration",
            ],
        )
        write_csv(
            true_states_path,
            [
                {
                    "trajectory_id": trajectory.trajectory_id,
                    "tier": trajectory.generator_parameters.get("tier", ""),
                    "scenario_id": trajectory.scenario_id,
                    "true_class": trajectory.true_class,
                    "seed": trajectory.seed,
                    "step": index,
                    "time": time,
                    "true_position": trajectory.true_position[index] if trajectory.true_position else "",
                    "true_velocity": trajectory.true_velocity[index] if trajectory.true_velocity else "",
                    "true_acceleration": trajectory.true_acceleration[index] if trajectory.true_acceleration else "",
                }
                for trajectory in trajectories
                for index, time in enumerate(trajectory.times)
            ],
            [
                "trajectory_id",
                "tier",
                "scenario_id",
                "true_class",
                "seed",
                "step",
                "time",
                "true_position",
                "true_velocity",
                "true_acceleration",
            ],
        )
        for trajectory in trajectories:
            errors = validate_trajectory_artifact(trajectory)
            if errors:
                raise ValueError(f"invalid supplemental trajectory {trajectory.trajectory_id}: {errors}")

    return TrajectoryGeneratorArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        class_definitions_path=class_definitions_path,
        config_path=config_path,
        dataset_manifest_paths=dataset_manifest_paths,
        generated_trajectories_paths=generated_paths,
        true_states_paths=true_state_paths,
        supplemental_manifest_paths=supplemental_manifest_paths,
        supplemental_generated_paths=supplemental_generated_paths,
        supplemental_true_states_paths=supplemental_true_state_paths,
        plot_png_path=plot_png_path,
    )


def _render_dataset_plot(dataset: GeneratedTrajectoryDataset):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
    class_names = [class_definition.name for class_definition in dataset.class_definitions]
    colors = {
        name: color
        for name, color in zip(class_names, ("#2563eb", "#16a34a", "#7c3aed", "#d97706", "#db2777", "#0f766e", "#dc2626"))
    }
    for class_name in class_names:
        class_trajectories = [trajectory for trajectory in dataset.trajectories if trajectory.true_class == class_name][:2]
        for trajectory in class_trajectories:
            axes[0].plot(trajectory.times, trajectory.measurements, color=colors[class_name], alpha=0.7, linewidth=1.5)
            axes[1].plot(trajectory.times, trajectory.true_position, color=colors[class_name], alpha=0.7, linewidth=1.5)
    axes[0].set_title(f"{dataset.tier} measurements", loc="left", fontweight="bold")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("measurement")
    axes[0].grid(True, alpha=0.2)
    axes[1].set_title(f"{dataset.tier} true position", loc="left", fontweight="bold")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("position")
    axes[1].grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def _trajectory_rows(dataset: GeneratedTrajectoryDataset) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for trajectory in dataset.trajectories:
        for index, time in enumerate(trajectory.times):
            rows.append(
                {
                    "trajectory_id": trajectory.trajectory_id,
                    "tier": dataset.tier,
                    "scenario_id": trajectory.scenario_id,
                    "true_class": trajectory.true_class,
                    "seed": trajectory.seed,
                    "step": index,
                    "time": time,
                    "measurement": trajectory.measurements[index],
                    "true_position": trajectory.true_position[index] if trajectory.true_position else "",
                    "true_velocity": trajectory.true_velocity[index] if trajectory.true_velocity else "",
                    "true_acceleration": trajectory.true_acceleration[index] if trajectory.true_acceleration else "",
                }
            )
    return rows


def _true_state_rows(dataset: GeneratedTrajectoryDataset) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for trajectory in dataset.trajectories:
        for index, time in enumerate(trajectory.times):
            rows.append(
                {
                    "trajectory_id": trajectory.trajectory_id,
                    "tier": dataset.tier,
                    "scenario_id": trajectory.scenario_id,
                    "true_class": trajectory.true_class,
                    "seed": trajectory.seed,
                    "step": index,
                    "time": time,
                    "true_position": trajectory.true_position[index] if trajectory.true_position else "",
                    "true_velocity": trajectory.true_velocity[index] if trajectory.true_velocity else "",
                    "true_acceleration": trajectory.true_acceleration[index] if trajectory.true_acceleration else "",
                }
            )
    return rows


def _dataset_manifest(dataset: GeneratedTrajectoryDataset) -> dict[str, object]:
    class_counts: dict[str, int] = {}
    scenario_counts: dict[str, int] = {}
    step_counts: list[int] = []
    dt_values: list[float] = []
    measurement_values: list[float] = []
    for trajectory in dataset.trajectories:
        class_counts[trajectory.true_class] = class_counts.get(trajectory.true_class, 0) + 1
        scenario_counts[trajectory.scenario_id] = scenario_counts.get(trajectory.scenario_id, 0) + 1
        step_counts.append(len(trajectory.times))
        if len(trajectory.times) >= 2:
            dt_values.extend(
                trajectory.times[index] - trajectory.times[index - 1]
                for index in range(1, len(trajectory.times))
            )
        if trajectory.measurement_std is not None:
            measurement_values.append(trajectory.measurement_std)
    return {
        "tier": dataset.tier,
        "generator_version": "trajectory_generator_v1",
        "trajectory_count": len(dataset.trajectories),
        "class_counts": class_counts,
        "scenario_counts": scenario_counts,
        "steps": {
            "min": min(step_counts) if step_counts else 0,
            "max": max(step_counts) if step_counts else 0,
            "mean": _mean(step_counts) if step_counts else 0.0,
        },
        "dt": {
            "min": min(dt_values) if dt_values else 0.0,
            "max": max(dt_values) if dt_values else 0.0,
            "mean": _mean(dt_values) if dt_values else 0.0,
        },
        "measurement_std": {
            "min": min(measurement_values) if measurement_values else 0.0,
            "max": max(measurement_values) if measurement_values else 0.0,
            "mean": _mean(measurement_values) if measurement_values else 0.0,
        },
        "notes": [
            "Trajectories are one-dimensional and deterministic under seed control.",
            "Outlier and timing irregularity are included for adversarial and stress tiers.",
            "Measurement dropouts are deferred until the contracts support explicit missing-data masks.",
        ],
    }


def _trajectory_manifest(
    name: str,
    trajectories: tuple[TrajectoryArtifact, ...],
    *,
    notes: list[str],
) -> dict[str, object]:
    step_counts = [len(trajectory.times) for trajectory in trajectories]
    dt_values = [
        trajectory.times[index] - trajectory.times[index - 1]
        for trajectory in trajectories
        for index in range(1, len(trajectory.times))
    ]
    measurement_values = [trajectory.measurement_std for trajectory in trajectories if trajectory.measurement_std is not None]
    class_counts: dict[str, int] = {}
    scenario_counts: dict[str, int] = {}
    for trajectory in trajectories:
        class_counts[trajectory.true_class] = class_counts.get(trajectory.true_class, 0) + 1
        scenario_counts[trajectory.scenario_id] = scenario_counts.get(trajectory.scenario_id, 0) + 1
    return {
        "name": name,
        "generator_version": "trajectory_generator_v1",
        "trajectory_count": len(trajectories),
        "class_counts": class_counts,
        "scenario_counts": scenario_counts,
        "steps": {
            "min": min(step_counts) if step_counts else 0,
            "max": max(step_counts) if step_counts else 0,
            "mean": _mean(step_counts) if step_counts else 0.0,
        },
        "dt": {
            "min": min(dt_values) if dt_values else 0.0,
            "max": max(dt_values) if dt_values else 0.0,
            "mean": _mean(dt_values) if dt_values else 0.0,
        },
        "measurement_std": {
            "min": min(measurement_values) if measurement_values else 0.0,
            "max": max(measurement_values) if measurement_values else 0.0,
            "mean": _mean(measurement_values) if measurement_values else 0.0,
        },
        "notes": notes,
    }


def _render_trajectory_generator_report(datasets: tuple[GeneratedTrajectoryDataset, ...]) -> str:
    report = MarkdownDocument("Trajectory Generator v1")
    report.paragraph("This generator defines explicit 1D class models and produces tiered synthetic datasets for the roadmap baseline.")
    report.heading("Class Definitions", level=2)
    for class_definition in default_trajectory_class_definitions():
        report.heading(class_definition.name, level=3)
        report.bullet_list(
            [
                f"kind: `{class_definition.kind}`",
                f"description: {class_definition.description}",
                f"nominal steps: `{class_definition.nominal_steps[0]}..{class_definition.nominal_steps[1]}`",
                f"dt range: `{class_definition.dt_range[0]:.2f}..{class_definition.dt_range[1]:.2f}`",
                f"measurement std range: `{class_definition.measurement_std_range[0]:.2f}..{class_definition.measurement_std_range[1]:.2f}`",
            ]
        )
    report.heading("Dataset Tiers", level=2)
    for dataset in datasets:
        manifest = _dataset_manifest(dataset)
        report.heading(dataset.tier, level=3)
        report.bullet_list(
            [
                f"trajectories: {manifest['trajectory_count']}",
                f"steps: `{manifest['steps']['min']}..{manifest['steps']['max']}`",
                f"dt mean: `{manifest['dt']['mean']:.3f}`",
                f"measurement std mean: `{manifest['measurement_std']['mean']:.3f}`",
                f"parameter mode: `{dataset.tier_definition.parameter_mode}`",
            ]
        )
        report.table(
            ["class", "count"],
            [(class_name, count) for class_name, count in sorted(manifest["class_counts"].items())],
        )
    report.heading("Supplemental Scenario Libraries", level=2)
    report.bullet_list(
        [
            "`short_horizon_v1`: explicit short-horizon separability cases with only a few samples.",
            "`perturbation_sweeps_v1`: explicit noise, outlier, and irregular-`dt` sweep scenarios.",
            "`switching_scenarios_v1`: explicit mode-switching tracks such as stationary-then-moving and velocity-then-braking.",
        ]
    )
    report.heading("Validation Notes", level=2)
    report.bullet_list(
        [
            "Trajectories are validated with the shared trajectory artifact contract.",
            "The easy tier is cleanly separated; boundary and adversarial tiers intentionally stress overlap.",
            "Stress tier emphasizes short sequences, irregular timing, and larger measurement noise.",
        ]
    )
    return report.text()


def _render_figure_svg(fig) -> str:
    buffer = io.StringIO()
    try:
        fig.savefig(buffer, format="svg", bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def _render_figure_png(fig) -> bytes:
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)
