from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class ManageAnalysisCacheScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]

    def test_summary_and_clear_cli(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env = os.environ.copy()
            env["KINEMATIC_CLASSIFIER_RUNTIME_ROOT"] = temp_dir
            env["PYTHONPATH"] = str(self.root / "src")

            populate = subprocess.run(
                [
                    "python3",
                    "-c",
                    (
                        "import sys; sys.path.insert(0, 'src'); "
                        "from kinematic_classifier_sandbox.utils.analysis_cache import "
                        "load_or_compute_pickled, stable_cache_key; "
                        "key = stable_cache_key('script_demo', {'value': 1}); "
                        "load_or_compute_pickled(namespace='script_demo', cache_key=key, compute=lambda: {'value': 1})"
                    ),
                ],
                cwd=self.root,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertEqual(populate.returncode, 0)

            summary = subprocess.run(
                [
                    "python3",
                    "scripts/audit/manage_analysis_cache.py",
                    "summary",
                    "--namespace",
                    "script_demo",
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
                    "scripts/audit/manage_analysis_cache.py",
                    "clear",
                    "--namespace",
                    "script_demo",
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
                    "scripts/audit/manage_analysis_cache.py",
                    "clear",
                    "--namespace",
                    "script_demo",
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
