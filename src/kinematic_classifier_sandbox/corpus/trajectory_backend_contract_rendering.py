from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from kinematic_classifier_sandbox.reports.markdown import (
    MarkdownDocument,
    MermaidEdge,
    MermaidFlow,
    MermaidNode,
)

from ..utils.io import write_csv
from ..utils.plotting import plt
from .trajectory_backend_contract_types import (
    BackendContractDefinition,
    TrajectoryBackendContractResult,
)


class TrajectoryBackendContractArtifacts(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_dir: Path
    backend_contract_path: Path
    backend_capability_schema_path: Path
    scenario_spec_schema_path: Path
    design_variable_schema_path: Path
    control_policy_schema_path: Path
    environment_spec_schema_path: Path
    trajectory_run_schema_path: Path
    capability_matrix_csv_path: Path
    capability_matrix_png_path: Path
    report_path: Path


def _render_capability_matrix_png(rows: tuple[dict[str, Any], ...]) -> bytes:
    metric_names = (
        "supports_environment",
        "supports_sequential_control",
        "supports_events",
        "supports_stochastic_runs",
        "state_output_count",
        "observation_output_count",
        "event_output_count",
        "search_method_count",
    )
    data = [[float(row[name]) for name in metric_names] for row in rows]
    backend_labels = [str(row["family"]) for row in rows]

    fig, ax = plt.subplots(figsize=(10, 3.6))
    image = ax.imshow(data, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(metric_names)), labels=[name.replace("_", "\n") for name in metric_names], fontsize=8)
    ax.set_yticks(range(len(backend_labels)), labels=backend_labels, fontsize=9)
    ax.set_title("Backend Capability Matrix")
    for row_index, row_values in enumerate(data):
        for column_index, value in enumerate(row_values):
            ax.text(column_index, row_index, f"{value:.0f}", ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.04)
    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def _render_contract_markdown(
    definitions: tuple[BackendContractDefinition, ...],
    *,
    valid_count: int,
    sequential_count: int,
    environment_count: int,
) -> str:
    report = MarkdownDocument()
    report.heading("Trajectory Backend Contract", level=1)
    report.heading("Summary", level=2)
    report.bullet_list(
        [
            f"backend families declared: `{len(definitions)}`",
            f"fully valid contract declarations: `{valid_count}`",
            f"sequential-control backends: `{sequential_count}`",
            f"environment-aware backends: `{environment_count}`",
        ]
    )
    report.heading("Backend Families", level=2)
    report.table(
        ["Backend", "Family", "Runtime", "Sequential", "Environment", "Search Methods"],
        [
            (
                f"`{definition.capabilities.display_name}`",
                f"`{definition.capabilities.family}`",
                f"`{definition.capabilities.runtime_class}`",
                f"`{definition.capabilities.supports_sequential_control}`",
                f"`{definition.capabilities.supports_environment}`",
                f"`{', '.join(definition.capabilities.valid_search_methods)}`",
            )
            for definition in definitions
        ],
    )
    report.heading("Relationship Diagram", level=2)
    report.mermaid(
        MermaidFlow(
            nodes=(
                MermaidNode("A", "ScenarioSpec"),
                MermaidNode("B", "DesignVariableSpec"),
                MermaidNode("C", "ControlPolicySpec"),
                MermaidNode("D", "EnvironmentSpec"),
                MermaidNode("E", "TrajectoryBackendCapabilities"),
                MermaidNode("F", "TrajectoryBackend Adapter"),
                MermaidNode("G", "TrajectoryRun"),
                MermaidNode("H", "Features / Labels / Classifiers"),
                MermaidNode("I", "Search / Archive / Corpus Selection"),
            ),
            edges=(
                MermaidEdge("A", "F"),
                MermaidEdge("B", "F"),
                MermaidEdge("C", "F"),
                MermaidEdge("D", "F"),
                MermaidEdge("E", "F"),
                MermaidEdge("F", "G"),
                MermaidEdge("G", "H"),
                MermaidEdge("H", "I"),
            ),
        )
    )
    report.heading("Notes", level=2)
    report.bullet_list(
        [
            "This milestone defines the typed contract only. It does not yet prove full adapter execution across multiple engines.",
            "The common search layer should only inspect capability descriptors and normalized `TrajectoryRun` objects, never simulator-specific control names directly.",
            "The mock file backend exists to prove future file-in/file-out simulator integration patterns without binding to a real external tool yet.",
        ]
    )
    return report.text()


def write_trajectory_backend_contract_artifacts(
    base_dir: str | Path,
    *,
    result: TrajectoryBackendContractResult | None = None,
) -> TrajectoryBackendContractArtifacts:
    from .trajectory_backend_contract import analyze_trajectory_backend_contract

    run_dir = Path(base_dir) / "trajectory_backend_contract"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = result or analyze_trajectory_backend_contract()

    backend_contract_path = run_dir / "backend_contract.json"
    backend_capability_schema_path = run_dir / "backend_capability_schema.json"
    scenario_spec_schema_path = run_dir / "scenario_spec_schema.json"
    design_variable_schema_path = run_dir / "design_variable_schema.json"
    control_policy_schema_path = run_dir / "control_policy_schema.json"
    environment_spec_schema_path = run_dir / "environment_spec_schema.json"
    trajectory_run_schema_path = run_dir / "trajectory_run_schema.json"
    capability_matrix_csv_path = run_dir / "capability_matrix.csv"
    capability_matrix_png_path = run_dir / "capability_matrix.png"
    report_path = run_dir / "backend_contract_report.md"

    backend_contract_path.write_text(json.dumps(payload.backend_contract, indent=2), encoding="utf-8")
    backend_capability_schema_path.write_text(json.dumps(payload.backend_capability_schema, indent=2), encoding="utf-8")
    scenario_spec_schema_path.write_text(json.dumps(payload.scenario_spec_schema, indent=2), encoding="utf-8")
    design_variable_schema_path.write_text(json.dumps(payload.design_variable_schema, indent=2), encoding="utf-8")
    control_policy_schema_path.write_text(json.dumps(payload.control_policy_schema, indent=2), encoding="utf-8")
    environment_spec_schema_path.write_text(json.dumps(payload.environment_spec_schema, indent=2), encoding="utf-8")
    trajectory_run_schema_path.write_text(json.dumps(payload.trajectory_run_schema, indent=2), encoding="utf-8")
    report_path.write_text(payload.report_markdown, encoding="utf-8")

    fieldnames = list(payload.capability_matrix_rows[0].keys()) if payload.capability_matrix_rows else []
    write_csv(capability_matrix_csv_path, list(payload.capability_matrix_rows), fieldnames)

    capability_matrix_png_path.write_bytes(_render_capability_matrix_png(payload.capability_matrix_rows))

    return TrajectoryBackendContractArtifacts(
        run_dir=run_dir,
        backend_contract_path=backend_contract_path,
        backend_capability_schema_path=backend_capability_schema_path,
        scenario_spec_schema_path=scenario_spec_schema_path,
        design_variable_schema_path=design_variable_schema_path,
        control_policy_schema_path=control_policy_schema_path,
        environment_spec_schema_path=environment_spec_schema_path,
        trajectory_run_schema_path=trajectory_run_schema_path,
        capability_matrix_csv_path=capability_matrix_csv_path,
        capability_matrix_png_path=capability_matrix_png_path,
        report_path=report_path,
    )
