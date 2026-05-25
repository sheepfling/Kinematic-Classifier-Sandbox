from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
existing_pythonpath = os.environ.get("PYTHONPATH")
os.environ.setdefault("PYTHONPYCACHEPREFIX", str(Path(tempfile.gettempdir()) / "kinematic-classifier-sandbox-pycache"))
os.environ["PYTHONPATH"] = (
    str(SRC) if not existing_pythonpath else f"{SRC}{os.pathsep}{existing_pythonpath}"
)
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kinematic-classifier-sandbox-mpl"))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kinematic_classifier_sandbox.repo_story import write_repo_story_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(prog="python3 scripts/render_repo_story.py")
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
