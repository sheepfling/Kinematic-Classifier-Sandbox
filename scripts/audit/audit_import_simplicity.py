from __future__ import annotations

import argparse
from pathlib import Path
from _bootstrap import bootstrap_repo

ROOT = bootstrap_repo(configure_runtime=True)




from kinematic_classifier_sandbox.meta.import_simplicity_audit import (  # noqa: E402
    analyze_import_simplicity,
    write_import_simplicity_audit_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--write-artifacts", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    result = analyze_import_simplicity()
    if args.write_artifacts:
        artifacts = write_import_simplicity_audit_artifacts(args.output_dir, result=result)
        print(artifacts.run_dir)
        print(artifacts.report_path)
        print(artifacts.summary_path)
    else:
        print(result.report_markdown)
    return 0 if (not args.strict or result.summary["passes"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
