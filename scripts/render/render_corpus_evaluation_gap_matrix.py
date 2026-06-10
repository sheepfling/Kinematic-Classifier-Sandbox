from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
existing_pythonpath = os.environ.get("PYTHONPATH")
os.environ.setdefault("PYTHONPYCACHEPREFIX", str(Path(tempfile.gettempdir()) / "kinematic-classifier-sandbox-pycache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kinematic-classifier-sandbox-mpl"))
os.environ["PYTHONPATH"] = str(SRC) if not existing_pythonpath else f"{SRC}{os.pathsep}{existing_pythonpath}"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kinematic_classifier_sandbox.registry.corpus_evaluation_gap_matrix import (
    write_corpus_evaluation_gap_matrix_artifacts,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python3 scripts/render/render_corpus_evaluation_gap_matrix.py")
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where the corpus-evaluation gap-matrix bundle should be written.",
    )
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="Materialize the selected capabilities into a temporary output directory and classify observed artifact classes.",
    )
    parser.add_argument(
        "--capability-id",
        action="append",
        dest="capability_ids",
        default=None,
        help="Optional capability id to audit. Repeat to audit a subset.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    artifacts = write_corpus_evaluation_gap_matrix_artifacts(
        args.output_dir,
        materialize=args.materialize,
        capability_ids=args.capability_ids,
    )
    print(artifacts.run_dir)
    print(artifacts.report_path)
    print(artifacts.summary_path)
    print(artifacts.matrix_path)
    print(artifacts.coherence_issues_path)
    print(artifacts.inventory_path)
    print(artifacts.status_plot_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
