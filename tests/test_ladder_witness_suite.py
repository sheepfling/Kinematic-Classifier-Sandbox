from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

from kinematic_classifier_sandbox import (
    load_ladder_witness_suite_config,
    write_ladder_witness_suite_artifacts,
)


class LadderWitnessSuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.config_path = cls.root / "experiments" / "ladder_witness_suite" / "ladder_witness_suite.yaml"

    def test_config_declares_sufficiency_and_insufficiency_for_each_rung(self) -> None:
        config = load_ladder_witness_suite_config(self.config_path)
        self.assertEqual(config["suite_id"], "ladder_witness_suite_v1")
        self.assertEqual(len(config["witnesses"]), 13)
        by_rung = defaultdict(lambda: {"sufficient": 0, "insufficient": 0})
        for witness in config["witnesses"]:
            for method, status in witness["expected_result"].items():
                if status in {"sufficient", "insufficient"}:
                    by_rung[method][status] += 1
        expected_rungs = {
            "pointwise",
            "windowed",
            "sequential_bayes",
            "kalman_bank",
            "transition_matrix",
            "imm",
            "particle_filter",
            "rbpf",
        }
        self.assertEqual(set(by_rung), expected_rungs)
        for rung, counts in by_rung.items():
            self.assertGreaterEqual(counts["sufficient"], 1, rung)
            self.assertGreaterEqual(counts["insufficient"], 1, rung)

    def test_writer_emits_schema_manifest_and_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_ladder_witness_suite_artifacts(temp_dir, config_path=self.config_path)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "ladder_witness_suite_v1")
            for path in (
                artifacts.config_path,
                artifacts.schema_path,
                artifacts.manifest_path,
                artifacts.claim_matrix_path,
                artifacts.index_path,
            ):
                self.assertTrue(path.exists(), path)
                self.assertGreater(path.stat().st_size, 0, path)

            manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["suite_id"], "ladder_witness_suite_v1")
            self.assertEqual(manifest["witness_count"], 13)

            script_path = self.root / "scripts" / "render_ladder_witness_suite.py"
            subprocess.run(
                ["python3", str(script_path), "--output-dir", temp_dir, "--config", str(self.config_path)],
                check=True,
                cwd=self.root,
            )

            module_run = subprocess.run(
                [
                    "python3",
                    "-m",
                    "kinematic_classifier_sandbox",
                    "ladder-witness-suite",
                    "--output-dir",
                    temp_dir,
                    "--config",
                    str(self.config_path),
                ],
                check=True,
                cwd=self.root,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": str(self.root / "src")},
            )
            self.assertIn("ladder_witness_suite_v1", module_run.stdout)
            self.assertIn("witness_schema.json", module_run.stdout)


if __name__ == "__main__":
    unittest.main()
