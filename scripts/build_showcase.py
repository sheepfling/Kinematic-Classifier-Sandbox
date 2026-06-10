from __future__ import annotations

import argparse
from pathlib import Path
from _bootstrap import bootstrap_repo

ROOT = bootstrap_repo(configure_runtime=True)



from kinematic_classifier_sandbox.showcase.builder import build_showcase_artifacts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python3 scripts/build_showcase.py")
    parser.add_argument("--output-dir", default="artifacts", help="Directory where showcase outputs should be written.")
    parser.add_argument("--refresh", action="store_true", help="Refresh source artifacts before building the showcase packet.")
    parser.add_argument("--zip", action="store_true", help="Also write the team packet zip.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    artifacts = build_showcase_artifacts(args.output_dir, refresh=args.refresh, create_zip=args.zip)
    print(artifacts.showcase_dir)
    print(artifacts.index_path)
    print(artifacts.team_packet_dir)
    if artifacts.zip_path is not None:
        print(artifacts.zip_path)
    print(artifacts.validation_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
