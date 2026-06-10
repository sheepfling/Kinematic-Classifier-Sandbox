#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from _bootstrap import bootstrap_repo

ROOT = bootstrap_repo(configure_runtime=True)



from kinematic_classifier_sandbox.formal_math_registry import write_formal_math_registry_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the formal math registry artifact bundle.")
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where the registry artifact bundle should be written.",
    )
    args = parser.parse_args()
    artifacts = write_formal_math_registry_artifacts(Path(args.output_dir))
    print(artifacts.run_dir)
    print(artifacts.report_path)
    print(artifacts.summary_path)
    print(artifacts.function_registry_path)
    print(artifacts.equation_registry_path)
    print(artifacts.crosswalk_path)
    print(artifacts.plot_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
