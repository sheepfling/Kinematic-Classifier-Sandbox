#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from _bootstrap import bootstrap_repo

ROOT = bootstrap_repo(configure_runtime=True)



from kinematic_classifier_sandbox.formal_math_visual_registry import (
    write_formal_math_visual_registry_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the formal math visual gallery artifact bundle.")
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where the visual gallery artifact bundle should be written.",
    )
    args = parser.parse_args()
    artifacts = write_formal_math_visual_registry_artifacts(Path(args.output_dir))
    print(artifacts.run_dir)
    print(artifacts.report_path)
    print(artifacts.summary_path)
    print(artifacts.gallery_csv_path)
    print(artifacts.provenance_path)
    print(artifacts.runbook_path)
    print(artifacts.visual_coverage_png_path)
    print(artifacts.assets_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
