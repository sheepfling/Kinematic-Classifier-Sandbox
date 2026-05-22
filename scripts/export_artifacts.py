from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
existing_pythonpath = os.environ.get("PYTHONPATH")
os.environ["PYTHONPATH"] = (
    str(SRC) if not existing_pythonpath else f"{SRC}{os.pathsep}{existing_pythonpath}"
)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kinematic_classifier_sandbox import (
    run_identity_benchmark,
    write_identity_posterior_comparison_artifacts,
    write_identity_posterior_explainer_artifacts,
    write_identity_posterior_failure_artifacts,
    write_identity_posterior_margin_trace_artifacts,
    write_posterior_comparison_artifacts,
    render_posterior_explainer_markdown,
    render_toy_benchmark_markdown,
    run_toy_benchmark,
    write_identity_benchmark_artifacts,
    write_method_survey_artifact,
    write_posterior_explainer_artifacts,
    write_posterior_failure_artifacts,
    write_posterior_margin_trace_artifacts,
    write_toy_benchmark_plot_artifacts,
    write_toy_benchmark_trace_csv,
)


def main() -> int:
    survey_path = write_method_survey_artifact(ROOT / "artifacts")
    identity_result = run_identity_benchmark()
    identity_markdown_path, identity_svg_path, identity_png_path, identity_csv_path = write_identity_benchmark_artifacts(
        ROOT / "artifacts",
        result=identity_result,
    )
    identity_posterior_markdown_path, identity_posterior_svg_path, identity_posterior_png_path = write_identity_posterior_explainer_artifacts(
        ROOT / "artifacts",
        result=identity_result,
    )
    identity_posterior_failure_markdown_path, identity_posterior_failure_svg_path, identity_posterior_failure_png_path = write_identity_posterior_failure_artifacts(
        ROOT / "artifacts",
        result=identity_result,
    )
    identity_posterior_comparison_markdown_path, identity_posterior_comparison_svg_path, identity_posterior_comparison_png_path = write_identity_posterior_comparison_artifacts(
        ROOT / "artifacts",
        result=identity_result,
    )
    identity_posterior_margin_markdown_path, identity_posterior_margin_svg_path, identity_posterior_margin_png_path = write_identity_posterior_margin_trace_artifacts(
        ROOT / "artifacts",
        result=identity_result,
    )
    result = run_toy_benchmark()
    benchmark_path = ROOT / "artifacts" / "toy_1d_benchmark_summary.md"
    benchmark_plot_svg_path, benchmark_plot_png_path = write_toy_benchmark_plot_artifacts(
        ROOT / "artifacts",
        result=result,
    )
    benchmark_trace_csv = write_toy_benchmark_trace_csv(result, ROOT / "artifacts")
    posterior_markdown_path, posterior_svg_path, posterior_png_path = write_posterior_explainer_artifacts(
        ROOT / "artifacts",
        result=result,
    )
    posterior_failure_markdown_path, posterior_failure_svg_path, posterior_failure_png_path = write_posterior_failure_artifacts(
        ROOT / "artifacts",
        result=result,
    )
    posterior_comparison_markdown_path, posterior_comparison_svg_path, posterior_comparison_png_path = write_posterior_comparison_artifacts(
        ROOT / "artifacts",
        result=result,
    )
    posterior_margin_markdown_path, posterior_margin_svg_path, posterior_margin_png_path = write_posterior_margin_trace_artifacts(
        ROOT / "artifacts",
        result=result,
    )
    benchmark_path.write_text(render_toy_benchmark_markdown(result), encoding="utf-8")
    print(survey_path)
    print(identity_markdown_path)
    print(identity_svg_path)
    print(identity_png_path)
    print(identity_csv_path)
    print(identity_posterior_markdown_path)
    print(identity_posterior_svg_path)
    print(identity_posterior_png_path)
    print(identity_posterior_failure_markdown_path)
    print(identity_posterior_failure_svg_path)
    print(identity_posterior_failure_png_path)
    print(identity_posterior_comparison_markdown_path)
    print(identity_posterior_comparison_svg_path)
    print(identity_posterior_comparison_png_path)
    print(identity_posterior_margin_markdown_path)
    print(identity_posterior_margin_svg_path)
    print(identity_posterior_margin_png_path)
    print(benchmark_path)
    print(benchmark_plot_svg_path)
    print(benchmark_plot_png_path)
    print(benchmark_trace_csv)
    print(posterior_markdown_path)
    print(posterior_svg_path)
    print(posterior_png_path)
    print(posterior_failure_markdown_path)
    print(posterior_failure_svg_path)
    print(posterior_failure_png_path)
    print(posterior_comparison_markdown_path)
    print(posterior_comparison_svg_path)
    print(posterior_comparison_png_path)
    print(posterior_margin_markdown_path)
    print(posterior_margin_svg_path)
    print(posterior_margin_png_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
