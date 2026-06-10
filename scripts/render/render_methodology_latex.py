#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
existing_pythonpath = os.environ.get("PYTHONPATH")
os.environ["PYTHONPATH"] = (
    str(SRC) if not existing_pythonpath else f"{SRC}{os.pathsep}{existing_pythonpath}"
)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kinematic_classifier_sandbox.utils.runtime import configure_runtime_environment

configure_runtime_environment()

from kinematic_classifier_sandbox.methodology.latex import write_methodology_latex_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the methodology LaTeX artifact bundle.")
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where the methodology LaTeX bundle should be written.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Write the LaTeX bundle without attempting a PDF build.",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Write the LaTeX bundle without building the PDF.",
    )
    args = parser.parse_args()
    artifacts = write_methodology_latex_artifacts(
        Path(args.output_dir),
        build_pdf=not args.no_pdf,
        artifact_mode="fast" if args.fast else "full",
    )
    print(artifacts.run_dir)
    print(artifacts.source_tex_path)
    print(artifacts.artifact_tex_path)
    if artifacts.pdf_path is not None:
        print(artifacts.pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
