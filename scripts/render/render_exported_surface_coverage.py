from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
existing_pythonpath = os.environ.get("PYTHONPATH")
os.environ.setdefault("PYTHONPYCACHEPREFIX", str(Path(tempfile.gettempdir()) / "kinematic-classifier-sandbox-pycache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kinematic-classifier-sandbox-mpl"))
os.environ["PYTHONPATH"] = str(SRC) if not existing_pythonpath else f"{SRC}{os.pathsep}{existing_pythonpath}"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kinematic_classifier_sandbox.registry.exported_surface_coverage import (
    write_exported_surface_coverage_artifacts,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python3 scripts/render/render_exported_surface_coverage.py")
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where the exported-surface coverage bundle should be written.",
    )
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="Materialize the selected surfaces into a temporary output directory and classify observed artifact classes.",
    )
    parser.add_argument(
        "--surface-id",
        action="append",
        dest="surface_ids",
        default=None,
        help="Optional surface id to audit. Repeat to audit a subset.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    artifacts = write_exported_surface_coverage_artifacts(
        args.output_dir,
        materialize=args.materialize,
        surface_ids=args.surface_ids,
    )
    print(artifacts.run_dir)
    print(artifacts.report_path)
    print(artifacts.summary_path)
    print(artifacts.coverage_matrix_path)
    print(artifacts.missing_coverage_path)
    print(artifacts.visualization_exemptions_path)
    print(artifacts.rerun_commands_path)
    print(artifacts.category_plot_path)
    print(artifacts.inventory_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
