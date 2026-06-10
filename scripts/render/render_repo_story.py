from __future__ import annotations

import argparse
from pathlib import Path
from _bootstrap import bootstrap_repo

ROOT = bootstrap_repo(configure_runtime=True)



from kinematic_classifier_sandbox.story.repo_story import write_repo_story_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(prog="python3 scripts/render/render_repo_story.py")
    parser.add_argument("--output-dir", default="artifacts", help="Artifact output root.")
    parser.add_argument("--docs-root", default="docs", help="Docs root to refresh generated story pages.")
    parser.add_argument("--no-showcase", action="store_true", help="Do not refresh showcase/team packet story front doors.")
    args = parser.parse_args()
    artifacts = write_repo_story_artifacts(
        args.output_dir,
        docs_root=args.docs_root,
        write_showcase=not args.no_showcase,
    )
    print(artifacts.run_dir)
    print(artifacts.claim_matrix_path)
    print(artifacts.artifact_manifest_path)
    print(artifacts.status_report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
