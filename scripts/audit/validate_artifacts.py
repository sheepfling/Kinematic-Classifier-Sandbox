from __future__ import annotations

import argparse
import json
from _bootstrap import bootstrap_repo

ROOT = bootstrap_repo(configure_runtime=True)

from dataclasses import asdict
from pathlib import Path


from kinematic_classifier_sandbox.showcase.builder import validate_showcase_artifacts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 scripts/audit/validate_artifacts.py")
    parser.add_argument(
        "--showcase-dir",
        default="artifacts/showcase",
        help="Showcase directory to validate.",
    )
    args = parser.parse_args(argv)
    result = validate_showcase_artifacts(args.showcase_dir)
    print(json.dumps(asdict(result), indent=2))
    return 0 if result.overall_status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
