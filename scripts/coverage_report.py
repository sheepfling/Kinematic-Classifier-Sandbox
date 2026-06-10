from __future__ import annotations

import sys
from pathlib import Path
from _bootstrap import bootstrap_repo

ROOT = bootstrap_repo(configure_runtime=True)



from kinematic_classifier_sandbox.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main(["coverage-report", *sys.argv[1:]]))
