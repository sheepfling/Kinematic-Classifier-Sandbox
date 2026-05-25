from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox import (
    analyze_quality_diversity_corpus,
    write_quality_diversity_corpus_artifacts,
)


class QualityDiversityCorpusTests(unittest.TestCase):
    def test_qd_archive_coverage_and_elites_are_generated(self) -> None:
        result = analyze_quality_diversity_corpus(seed=7, iterations=28)

        self.assertGreater(len(result.archive_cell_rows), 0)
        self.assertGreater(len(result.archive_elite_rows), 0)
        self.assertEqual(len(result.archive_coverage_rows), 28)

        first_coverage = float(result.archive_coverage_rows[0]["coverage_fraction"])
        final_coverage = float(result.archive_coverage_rows[-1]["coverage_fraction"])
        self.assertGreater(final_coverage, first_coverage)

        elite_validities = [float(row["class_validity"]) for row in result.archive_elite_rows]
        self.assertGreaterEqual(min(elite_validities), 0.45)

        target_tiers = {str(row["target_tier"]) for row in result.archive_elite_rows}
        self.assertTrue({"boundary_v1", "adversarial_v1"} & target_tiers)
        self.assertTrue(result.corpus_manifest["improves_feature_excitation_over_baseline"])

    def test_qd_artifacts_are_written(self) -> None:
        result = analyze_quality_diversity_corpus(seed=7, iterations=20)
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_quality_diversity_corpus_artifacts(temp_dir, result=result)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "quality_diversity_corpus")
            self.assertTrue(artifacts.config_path.exists())
            self.assertTrue(artifacts.archive_cells_path.exists())
            self.assertTrue(artifacts.archive_elites_path.exists())
            self.assertTrue(artifacts.archive_coverage_path.exists())
            self.assertTrue(artifacts.manifest_path.exists())
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.archive_coverage_heatmap_path.exists())
            self.assertTrue(artifacts.elite_score_distribution_path.exists())
            self.assertTrue(artifacts.feature_cell_examples_path.exists())

            manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
            self.assertIn("num_archive_cells", manifest)
            report_text = artifacts.report_path.read_text(encoding="utf-8")
            self.assertIn("Quality-Diversity Corpus", report_text)
            self.assertIn("Improves feature excitation over baseline", report_text)


if __name__ == "__main__":
    unittest.main()
