from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
existing_pythonpath = os.environ.get("PYTHONPATH")
os.environ["PYTHONPATH"] = str(SRC) if not existing_pythonpath else f"{SRC}{os.pathsep}{existing_pythonpath}"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kinematic_classifier_sandbox import write_dimensional_lift_audit_artifacts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 scripts/audit_dimensions.py")
    parser.add_argument("--output-dir", default="artifacts", help="Directory where dimensional audit outputs should be written.")
    args = parser.parse_args(argv)
    artifacts = write_dimensional_lift_audit_artifacts(args.output_dir)
    print(artifacts.run_dir)
    print(artifacts.audit_report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
