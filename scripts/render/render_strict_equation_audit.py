#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from _bootstrap import bootstrap_repo

ROOT = bootstrap_repo(configure_runtime=True)



from kinematic_classifier_sandbox.registry.strict_equation_audit import write_strict_equation_audit_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the strict formal-math equation audit artifact bundle.")
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where the strict audit bundle should be written.",
    )
    args = parser.parse_args()
    artifacts = write_strict_equation_audit_artifacts(Path(args.output_dir))
    print(artifacts.run_dir)
    print(artifacts.report_path)
    print(artifacts.summary_path)
    print(artifacts.rows_path)
    print(artifacts.status_plot_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
