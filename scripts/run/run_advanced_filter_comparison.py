from __future__ import annotations

import argparse
from pathlib import Path
from _bootstrap import bootstrap_repo

ROOT = bootstrap_repo(configure_runtime=True)



from kinematic_classifier_sandbox.advanced_filters.evaluation import (
    advanced_filter_comparison_surface,
)


def main() -> int:
    parser = argparse.ArgumentParser(prog="python3 scripts/run/run_advanced_filter_comparison.py")
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()
    surface = advanced_filter_comparison_surface()
    artifacts = surface.write_artifacts(args.output_dir)
    for line in surface.describe_artifacts(artifacts):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
