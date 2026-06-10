from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class PackageCliAnalysisCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]

    def test_package_cli_summary_and_clear(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env = os.environ.copy()
            env["KINEMATIC_CLASSIFIER_RUNTIME_ROOT"] = temp_dir
            env["PYTHONPATH"] = "src"

            subprocess.run(
                [
                    "python3",
                    "-c",
                    (
                        "from kinematic_classifier_sandbox.utils.analysis_cache import "
                        "load_or_compute_pickled, stable_cache_key; "
                        "key = stable_cache_key('pkg_cli_demo', {'value': 1}); "
                        "load_or_compute_pickled(namespace='pkg_cli_demo', cache_key=key, compute=lambda: {'value': 1})"
                    ),
                ],
                cwd=self.root,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )

            summary = subprocess.run(
                [
                    "python3",
                    "-m",
                    "kinematic_classifier_sandbox",
                    "analysis-cache",
                    "summary",
                    "--namespace",
                    "pkg_cli_demo",
                    "--json",
                ],
                cwd=self.root,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn('"entry_count": 1', summary.stdout)

            refused = subprocess.run(
                [
                    "python3",
                    "-m",
                    "kinematic_classifier_sandbox",
                    "analysis-cache",
                    "clear",
                    "--namespace",
                    "pkg_cli_demo",
                ],
                cwd=self.root,
                env=env,
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("--yes", refused.stderr or refused.stdout)

            cleared = subprocess.run(
                [
                    "python3",
                    "-m",
                    "kinematic_classifier_sandbox",
                    "analysis-cache",
                    "clear",
                    "--namespace",
                    "pkg_cli_demo",
                    "--yes",
                    "--json",
                ],
                cwd=self.root,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn('"cleared_entry_count": 1', cleared.stdout)


if __name__ == "__main__":
    unittest.main()
