from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/kinematic-classifier-sandbox-mpl")
os.environ.setdefault(
    "PYTHONPYCACHEPREFIX",
    "/Users/rick/LocalStorage/GIT_LOCAL/active/CACHE/kinematic-classifier-sandbox/.pycache",
)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kinematic_classifier_sandbox.meta.human_operability_audit import (  # noqa: E402
    analyze_human_operability_audit,
    write_human_operability_audit_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(prog="python3 scripts/audit/audit_human_operability.py")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--write-artifacts", action="store_true")
    args = parser.parse_args()

    result = analyze_human_operability_audit(output_dir=Path(args.output_dir))
    if args.write_artifacts:
        artifacts = write_human_operability_audit_artifacts(args.output_dir, result=result)
        print(artifacts.run_dir)
        print(artifacts.report_path)
        print(artifacts.summary_path)
    else:
        print(result.report_markdown)
    return 0 if result.summary["hard_fail_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
