from __future__ import annotations

import argparse
from pathlib import Path
from _bootstrap import bootstrap_repo

ROOT = bootstrap_repo(configure_runtime=True)



from kinematic_classifier_sandbox.analysis.dimensional_lift_audit import (
    write_dimensional_lift_audit_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 scripts/audit/audit_dimensions.py")
    parser.add_argument("--output-dir", default="artifacts", help="Directory where dimensional audit outputs should be written.")
    args = parser.parse_args(argv)
    artifacts = write_dimensional_lift_audit_artifacts(args.output_dir)
    print(artifacts.run_dir)
    print(artifacts.audit_report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
