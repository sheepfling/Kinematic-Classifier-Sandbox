#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from _bootstrap import bootstrap_repo

ROOT = bootstrap_repo(configure_runtime=True)


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
