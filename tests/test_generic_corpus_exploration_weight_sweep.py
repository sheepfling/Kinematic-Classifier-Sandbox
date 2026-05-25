from __future__ import annotations

import subprocess
import tempfile
import unittest
import os
from pathlib import Path

from kinematic_classifier_sandbox import (
    analyze_generic_corpus_exploration_weight_sweep,
    load_generic_corpus_exploration_weight_sweep_config,
    write_generic_corpus_exploration_weight_sweep_artifacts,
)


class GenericCorpusExplorationWeightSweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]

    def test_sweep_produces_baseline_and_perturbed_rows(self) -> None:
        result = analyze_generic_corpus_exploration_weight_sweep(seed=7)
        self.assertGreaterEqual(len(result.rows), 7)
        self.assertEqual(result.baseline_variant_id, "baseline")
        self.assertIn("Generic Corpus Exploration Weight Sweep", result.report_markdown)
        self.assertIn("Delta vs Baseline", result.report_markdown)
        self.assertIn("baseline", {variant.variant_id for variant in result.variants})
        self.assertTrue(any(row.coverage_delta_vs_baseline != 0 or row.mean_total_utility_delta_vs_baseline != 0 for row in result.rows if row.variant_id != "baseline"))
        for row in result.rows:
            total = row.weight_validity + row.weight_coverage_novelty + row.weight_boundary + row.weight_stress + row.weight_environment + row.weight_provenance
            self.assertAlmostEqual(total, 1.0, places=6)

    def test_yaml_config_loads_and_round_trips(self) -> None:
        config_path = self.root / "experiments" / "generic_corpus_exploration_weight_sweep" / "generic_corpus_exploration_weight_sweep.yaml"
        config = load_generic_corpus_exploration_weight_sweep_config(config_path)
        self.assertEqual(config.baseline_variant_id, "baseline")
        self.assertEqual(len(config.variants), 7)
        self.assertEqual(config.variants[0].variant_id, "baseline")

    def test_writer_and_cli_emit_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self.root / "experiments" / "generic_corpus_exploration_weight_sweep" / "generic_corpus_exploration_weight_sweep.yaml"
            artifacts = write_generic_corpus_exploration_weight_sweep_artifacts(temp_dir, config_path=config_path)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "generic_corpus_exploration_weight_sweep_v1")
            for path in (
                artifacts.config_path,
                artifacts.report_path,
                artifacts.summary_path,
                artifacts.rows_path,
                artifacts.overlap_matrix_path,
                artifacts.weight_matrix_path,
                artifacts.tradeoff_png_path,
                artifacts.selected_set_png_path,
                artifacts.baseline_manifest_path,
            ):
                self.assertTrue(path.exists(), path)
                self.assertGreater(path.stat().st_size, 0, path)

            script_path = self.root / "scripts" / "render_generic_corpus_exploration_weight_sweep.py"
            subprocess.run(["python3", str(script_path), "--output-dir", temp_dir, "--seed", "7", "--config", str(config_path)], check=True, cwd=self.root)

            module_run = subprocess.run(
                [
                    "python3",
                    "-m",
                    "kinematic_classifier_sandbox",
                    "generic-corpus-exploration-weight-sweep",
                    "--output-dir",
                    temp_dir,
                    "--seed",
                    "7",
                    "--config",
                    str(config_path),
                ],
                check=True,
                cwd=self.root,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": str(self.root / "src")},
            )
            self.assertIn("generic_corpus_exploration_weight_sweep_v1", module_run.stdout)
            self.assertIn("generic_corpus_exploration_weight_sweep_config.yaml", module_run.stdout)


if __name__ == "__main__":
    unittest.main()
