from __future__ import annotations

import argparse
from pathlib import Path
from _bootstrap import bootstrap_repo

ROOT = bootstrap_repo(configure_runtime=True)



from kinematic_classifier_sandbox.showcase.builder import build_showcase_artifacts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 scripts/build_gallery.py")
    parser.add_argument("--output-dir", default="artifacts", help="Directory where showcase outputs should be written.")
    args = parser.parse_args(argv)
    artifacts = build_showcase_artifacts(args.output_dir, refresh=False, create_zip=False)
    print(artifacts.plots_dir)
    print(artifacts.reports_dir / "07_visualization_gallery.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
