#!/usr/bin/env python3
import runpy
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "render" / "render_strict_equation_audit.py"
sys.argv[0] = str(SCRIPT)
runpy.run_path(str(SCRIPT), run_name="__main__")
