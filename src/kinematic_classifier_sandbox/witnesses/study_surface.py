from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

from .surface import WitnessSurface, run_surface

ResultT = TypeVar("ResultT")
ClassSpecT = TypeVar("ClassSpecT")


@dataclass(frozen=True, slots=True, kw_only=True)
class NamedArtifactWriter(Generic[ResultT]):
    filename: str
    write: Callable[[ResultT, Path], Path]


@dataclass(frozen=True, slots=True, kw_only=True)
class OneDWitnessSurface(WitnessSurface[ResultT, "OneDWitnessSurfaceArtifacts"], Generic[ResultT, ClassSpecT]):
    class_specs: tuple[ClassSpecT, ...]
    feature_names: tuple[str, ...]
    render_markdown: Callable[[ResultT], str]
    render_png_bytes: Callable[[ResultT], bytes]
    write_trace_csv: Callable[[ResultT, Path], Path] | None = None
    extra_artifact_writers: tuple[NamedArtifactWriter[ResultT], ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class OneDWitnessSurfaceArtifacts:
    run_dir: Path
    summary_path: Path | None = None
    plot_path: Path | None = None
    trace_path: Path | None = None
    extra_paths: dict[str, Path] = field(default_factory=dict)


def write_one_d_surface_artifacts(
    surface: OneDWitnessSurface[ResultT, Any],
    output_dir: str | Path,
    *,
    result: ResultT | None = None,
    run_kwargs: dict[str, Any] | None = None,
    summary_filename: str,
    plot_filename: str,
    write_summary: bool = True,
    write_plot: bool = True,
    write_trace: bool = True,
    write_extra: bool = True,
    nest_under_study_id: bool = True,
) -> OneDWitnessSurfaceArtifacts:
    benchmark_result = result or run_surface(surface, **(run_kwargs or {}))
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / surface.study_id if nest_under_study_id else output_root
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / summary_filename if write_summary else None
    plot_path = run_dir / plot_filename if write_plot else None
    trace_path = None
    extra_paths: dict[str, Path] = {}

    if write_summary and summary_path is not None:
        summary_path.write_text(surface.render_markdown(benchmark_result), encoding="utf-8")
    if write_plot and plot_path is not None:
        plot_path.write_bytes(surface.render_png_bytes(benchmark_result))
    if write_trace and surface.write_trace_csv is not None:
        trace_path = surface.write_trace_csv(benchmark_result, run_dir)
    if write_extra:
        for writer in surface.extra_artifact_writers:
            extra_paths[writer.filename] = writer.write(benchmark_result, run_dir / writer.filename)

    return OneDWitnessSurfaceArtifacts(
        run_dir=run_dir,
        summary_path=summary_path,
        plot_path=plot_path,
        trace_path=trace_path,
        extra_paths=extra_paths,
    )


write_surface_artifacts = write_one_d_surface_artifacts
