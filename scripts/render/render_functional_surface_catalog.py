from __future__ import annotations

import argparse
from pathlib import Path
from _bootstrap import bootstrap_repo

ROOT = bootstrap_repo(configure_runtime=True)



from kinematic_classifier_sandbox.functional_surface_catalog import (
    write_functional_surface_catalog_artifacts,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python3 scripts/render/render_functional_surface_catalog.py")
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where the functional-surface catalog bundle should be written.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    artifacts = write_functional_surface_catalog_artifacts(args.output_dir)
    print(artifacts.run_dir)
    print(artifacts.report_path)
    print(artifacts.summary_path)
    print(artifacts.catalog_path)
    print(artifacts.plot_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
