from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from kinematic_classifier_sandbox.reports.markdown import MarkdownDocument

from ...utils.io import _write_json, _write_text, write_csv
from ...utils.plotting import _figure_to_png, plt
from .comparison_surface import write_comparison_summary_csv


@dataclass(frozen=True, slots=True)
class ExplorationBackendSpec:
    backend_id: str
    display_name: str
    family: str
    implementation_status: str
    evaluation_phase: str
    search_mode: str
    requires_training: bool
    sequential_control: bool
    diversity_native: bool
    works_at_small_budget: bool
    ready_1d: bool
    lift_ready_3d: bool
    benchmark_priority: str
    notes: str


@dataclass(frozen=True, slots=True)
class ExplorationBackendRegistryResult:
    specs: tuple[ExplorationBackendSpec, ...]
    family_rows: tuple[dict[str, object], ...]
    summary: dict[str, object]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class ExplorationBackendRegistryArtifacts:
    run_dir: Path
    report_path: Path
    summary_path: Path
    spec_table_path: Path
    inventory_path: Path
    family_summary_path: Path
    capability_plot_path: Path


def default_exploration_backend_specs() -> tuple[ExplorationBackendSpec, ...]:
    return (
        ExplorationBackendSpec(
            backend_id="heuristic_search",
            display_name="Random / DOE heuristic search",
            family="baseline_search",
            implementation_status="implemented",
            evaluation_phase="phase_1",
            search_mode="parameter_proposal",
            requires_training=False,
            sequential_control=False,
            diversity_native=False,
            works_at_small_budget=True,
            ready_1d=True,
            lift_ready_3d=True,
            benchmark_priority="default_control",
            notes="Current cheap baseline covering random search plus design-of-experiments seeding.",
        ),
        ExplorationBackendSpec(
            backend_id="blackbox_optimizer",
            display_name="Cross-entropy black-box optimizer",
            family="black_box_optimization",
            implementation_status="implemented",
            evaluation_phase="phase_1",
            search_mode="parameter_proposal",
            requires_training=False,
            sequential_control=False,
            diversity_native=False,
            works_at_small_budget=True,
            ready_1d=True,
            lift_ready_3d=True,
            benchmark_priority="default_black_box",
            notes="Current CEM-style optimizer baseline for fixed-budget continuous parameter search.",
        ),
        ExplorationBackendSpec(
            backend_id="rl_policy",
            display_name="Stateless RL-shaped policy search",
            family="reinforcement_learning",
            implementation_status="implemented",
            evaluation_phase="phase_1",
            search_mode="parameter_proposal",
            requires_training=True,
            sequential_control=False,
            diversity_native=False,
            works_at_small_budget=False,
            ready_1d=True,
            lift_ready_3d=True,
            benchmark_priority="experimental_control",
            notes="Current RL-shaped comparator; remains experimental unless it beats heuristic and black-box search at matched budget.",
        ),
        ExplorationBackendSpec(
            backend_id="latin_hypercube",
            display_name="Grid / Latin hypercube search",
            family="baseline_search",
            implementation_status="implemented",
            evaluation_phase="phase_1",
            search_mode="parameter_proposal",
            requires_training=False,
            sequential_control=False,
            diversity_native=False,
            works_at_small_budget=True,
            ready_1d=True,
            lift_ready_3d=True,
            benchmark_priority="near_term",
            notes="Explicit low-variance stratified proposal baseline for disciplined fixed-budget comparisons.",
        ),
        ExplorationBackendSpec(
            backend_id="cmaes",
            display_name="CMA-ES",
            family="black_box_optimization",
            implementation_status="implemented",
            evaluation_phase="phase_1",
            search_mode="parameter_proposal",
            requires_training=False,
            sequential_control=False,
            diversity_native=False,
            works_at_small_budget=True,
            ready_1d=True,
            lift_ready_3d=True,
            benchmark_priority="near_term",
            notes="Diagonal covariance-adapting optimizer for difficult nonconvex witness and corpus objectives under the shared proposal contract.",
        ),
        ExplorationBackendSpec(
            backend_id="bayesian_optimization",
            display_name="Bayesian optimization",
            family="black_box_optimization",
            implementation_status="implemented",
            evaluation_phase="phase_2",
            search_mode="parameter_proposal",
            requires_training=False,
            sequential_control=False,
            diversity_native=False,
            works_at_small_budget=True,
            ready_1d=True,
            lift_ready_3d=True,
            benchmark_priority="sample_efficiency",
            notes="Acquisition-guided proposal search for settings where evaluation count matters more than brute-force population size.",
        ),
        ExplorationBackendSpec(
            backend_id="map_elites",
            display_name="MAP-Elites / quality-diversity archive",
            family="quality_diversity",
            implementation_status="implemented",
            evaluation_phase="phase_1",
            search_mode="archive_search",
            requires_training=False,
            sequential_control=False,
            diversity_native=True,
            works_at_small_budget=False,
            ready_1d=True,
            lift_ready_3d=True,
            benchmark_priority="coverage_default",
            notes="Current archive-search lane for behavior-space coverage rather than single-solution optimization.",
        ),
        ExplorationBackendSpec(
            backend_id="ppo",
            display_name="PPO",
            family="reinforcement_learning",
            implementation_status="implemented",
            evaluation_phase="phase_3",
            search_mode="sequential_control",
            requires_training=True,
            sequential_control=True,
            diversity_native=False,
            works_at_small_budget=False,
            ready_1d=True,
            lift_ready_3d=True,
            benchmark_priority="sequential_control_baseline",
            notes="First-class sequential-control baseline for witnesses where control timing matters rather than static parameter proposals.",
        ),
        ExplorationBackendSpec(
            backend_id="sac",
            display_name="SAC",
            family="reinforcement_learning",
            implementation_status="implemented",
            evaluation_phase="phase_3",
            search_mode="sequential_control",
            requires_training=True,
            sequential_control=True,
            diversity_native=False,
            works_at_small_budget=False,
            ready_1d=True,
            lift_ready_3d=True,
            benchmark_priority="research_candidate",
            notes="First-class off-policy continuous-control lane for better sample reuse than PPO.",
        ),
        ExplorationBackendSpec(
            backend_id="td3",
            display_name="TD3",
            family="reinforcement_learning",
            implementation_status="implemented",
            evaluation_phase="phase_3",
            search_mode="sequential_control",
            requires_training=True,
            sequential_control=True,
            diversity_native=False,
            works_at_small_budget=False,
            ready_1d=True,
            lift_ready_3d=True,
            benchmark_priority="research_candidate",
            notes="Alternative continuous-control policy search lane tracked with SAC so PPO is not the only first-class generator.",
        ),
        ExplorationBackendSpec(
            backend_id="mpc_adversarial",
            display_name="MPC-style adversarial generator",
            family="trajectory_optimization",
            implementation_status="planned",
            evaluation_phase="phase_3",
            search_mode="sequential_control",
            requires_training=False,
            sequential_control=True,
            diversity_native=False,
            works_at_small_budget=False,
            ready_1d=True,
            lift_ready_3d=True,
            benchmark_priority="research_candidate",
            notes="Control-optimization lane for explicit adversarial witness construction under dynamics constraints.",
        ),
    )


def exploration_backend_family_summary() -> tuple[dict[str, object], ...]:
    family_rows: dict[str, dict[str, object]] = {}
    for spec in default_exploration_backend_specs():
        row = family_rows.setdefault(
            spec.family,
            {
                "family": spec.family,
                "backend_count": 0,
                "implemented_count": 0,
                "sequential_control_count": 0,
                "ready_1d_count": 0,
            },
        )
        row["backend_count"] += 1
        row["implemented_count"] += 1 if spec.implementation_status == "implemented" else 0
        row["sequential_control_count"] += 1 if spec.sequential_control else 0
        row["ready_1d_count"] += 1 if spec.ready_1d else 0
    return tuple(family_rows[family] for family in sorted(family_rows))


def analyze_exploration_backend_registry() -> ExplorationBackendRegistryResult:
    specs = default_exploration_backend_specs()
    family_rows = exploration_backend_family_summary()
    summary = {
        "backend_count": len(specs),
        "family_count": len(family_rows),
        "implemented_count": sum(1 for spec in specs if spec.implementation_status == "implemented"),
        "planned_count": sum(1 for spec in specs if spec.implementation_status == "planned"),
        "experimental_count": sum(1 for spec in specs if spec.implementation_status == "experimental"),
        "sequential_control_count": sum(1 for spec in specs if spec.sequential_control),
        "diversity_native_count": sum(1 for spec in specs if spec.diversity_native),
    }
    return ExplorationBackendRegistryResult(
        specs=specs,
        family_rows=family_rows,
        summary=summary,
        report_markdown=render_exploration_backend_registry_report(specs, family_rows, summary),
    )


def render_exploration_backend_registry_report(
    specs: tuple[ExplorationBackendSpec, ...],
    family_rows: tuple[dict[str, object], ...],
    summary: dict[str, object],
) -> str:
    report = MarkdownDocument("Trajectory Exploration Backend Registry")
    report.paragraph(
        "This registry records the exploration and generator backends the repo supports, prototypes, or explicitly tracks for future evaluation. It separates parameter-only search, archive search, sequential-control RL, and trajectory-optimization lanes so comparisons stay honest."
    )
    report.heading("Summary", level=2)
    report.bullet_list(
        [
            f"Tracked backends: `{summary['backend_count']}`",
            f"Families: `{summary['family_count']}`",
            f"Implemented rows: `{summary['implemented_count']}`",
            f"Planned rows: `{summary['planned_count']}`",
            f"Experimental rows: `{summary['experimental_count']}`",
            f"Sequential-control rows: `{summary['sequential_control_count']}`",
        ]
    )
    report.heading("Family Summary", level=2)
    report.table(
        ["Family", "Backends", "Implemented", "Sequential control", "1D-ready"],
        [
            (
                row["family"],
                row["backend_count"],
                row["implemented_count"],
                row["sequential_control_count"],
                row["ready_1d_count"],
            )
            for row in family_rows
        ],
    )
    report.heading("Tracked Backends", level=2)
    report.table(
        ["Backend", "Family", "Status", "Phase", "Search mode", "Priority"],
        [
            (
                spec.display_name,
                spec.family,
                spec.implementation_status,
                spec.evaluation_phase,
                spec.search_mode,
                spec.benchmark_priority,
            )
            for spec in specs
        ],
    )
    report.heading("Reading Rule", level=2)
    report.bullet_list(
        [
            "Implemented rows have concrete code paths or artifact-producing surfaces in the repo.",
            "Planned rows reserve a slot in the benchmark architecture without claiming implementation.",
            "Sequential-control rows should not be compared naively against fixed-budget parameter proposal backends.",
            "Quality-diversity rows optimize coverage and archive breadth, not just single-solution utility.",
        ]
    )
    return report.text()


def _render_capability_plot(specs: tuple[ExplorationBackendSpec, ...]):
    columns = ("sequential_control", "requires_training", "diversity_native", "works_at_small_budget", "ready_1d", "lift_ready_3d")
    matrix = [[1 if getattr(spec, column) else 0 for column in columns] for spec in specs]
    fig_height = max(4.5, 0.4 * len(specs))
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    image = ax.imshow(matrix, aspect="auto", cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([column.replace("_", "\n") for column in columns], fontsize=8)
    ax.set_yticks(range(len(specs)))
    ax.set_yticklabels([spec.backend_id for spec in specs], fontsize=8)
    ax.set_title("Exploration Backend Capability Matrix", loc="left", fontweight="bold")
    ax.set_xlabel("Capability")
    ax.set_ylabel("Backend")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.028, pad=0.02)
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(["no", "yes"])
    fig.tight_layout()
    return fig


def write_exploration_backend_registry_artifacts(
    output_dir: str | Path,
    *,
    result: ExplorationBackendRegistryResult | None = None,
) -> ExplorationBackendRegistryArtifacts:
    payload = result or analyze_exploration_backend_registry()
    output_root = Path(output_dir)
    run_dir = output_root / "trajectory_exploration_backend_registry_v1"
    run_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / "report.md"
    summary_path = run_dir / "summary.json"
    spec_table_path = run_dir / "backend_registry.csv"
    inventory_path = run_dir / "backend_registry.json"
    family_summary_path = run_dir / "family_summary.csv"
    capability_plot_path = run_dir / "capability_matrix.png"

    _write_text(report_path, payload.report_markdown)
    _write_json(summary_path, payload.summary)
    write_comparison_summary_csv(run_dir, payload.family_rows, filename="summary.csv")
    write_csv(spec_table_path, [asdict(spec) for spec in payload.specs], list(asdict(payload.specs[0]).keys()))
    _write_json(inventory_path, [asdict(spec) for spec in payload.specs])
    write_csv(
        family_summary_path,
        list(payload.family_rows),
        ["family", "backend_count", "implemented_count", "sequential_control_count", "ready_1d_count"],
    )
    capability_plot_path.write_bytes(_figure_to_png(_render_capability_plot(payload.specs)))
    return ExplorationBackendRegistryArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        summary_path=summary_path,
        spec_table_path=spec_table_path,
        inventory_path=inventory_path,
        family_summary_path=family_summary_path,
        capability_plot_path=capability_plot_path,
    )
