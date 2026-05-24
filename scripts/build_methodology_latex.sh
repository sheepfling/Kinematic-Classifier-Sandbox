#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/private/tmp/kinematic-classifier-pycache}"

KINEMATIC_CLASSIFIER_ROOT="$ROOT" python3 - <<'PY'
import os
from pathlib import Path

from kinematic_classifier_sandbox import write_methodology_latex_artifacts

root = Path(os.environ["KINEMATIC_CLASSIFIER_ROOT"])
artifacts = write_methodology_latex_artifacts(root / "artifacts", build_pdf=True)
print(artifacts.pdf_path or artifacts.artifact_tex_path)
PY
