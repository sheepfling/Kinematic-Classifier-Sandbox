from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.corpus.exploration.objective_driven_qd_archive import (
    analyze_objective_driven_qd_archive,
    write_objective_driven_qd_archive_artifacts,
)


class ObjectiveDrivenQdArchiveTests(unittest.TestCase):
    def test_archive_grows_and_separates_failed_coverage(self) -> None:
        result = analyze_objective_driven_qd_archive()
        self.assertGreater(len(result.archive_cell_rows), 0)
        self.assertGreater(len(result.archive_elite_rows), 0)
        self.assertGreater(len(result.lineage_rows), 0)
        self.assertTrue(any(str(row["archive_status"]) == "failed" for row in result.archive_cell_rows))
        first_success = float(result.coverage_rows[0]["successful_coverage_fraction"])
        final_success = float(result.coverage_rows[-1]["successful_coverage_fraction"])
        self.assertGreater(final_success, first_success)
        self.assertIn("failed coverage are tracked separately", result.report_markdown)

    def test_artifacts_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_objective_driven_qd_archive_artifacts(temp_dir)
            self.assertEqual(artifacts.run_dir, Path(temp_dir) / "quality_diversity_corpus_v1")
            self.assertTrue(artifacts.archive_cells_path.exists())
            self.assertTrue(artifacts.archive_elites_path.exists())
            self.assertTrue(artifacts.coverage_path.exists())
            self.assertTrue(artifacts.lineage_path.exists())
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.coverage_plot_path.exists())
            self.assertTrue(artifacts.elite_distribution_path.exists())
            self.assertTrue(artifacts.lineage_plot_path.exists())


if __name__ == "__main__":
    unittest.main()
