from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.workbench.mvp import run_workbench_study
from kinematic_classifier_sandbox.workbench.revision_replay import (
    change_measurement_association,
    correct_measurement,
    diff_revisions,
    ensure_revision_history,
    replay_revision,
    restore_measurement,
    revoke_measurement,
    validate_replay,
)


class RevisionReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.study = cls.root / "experiments" / "common_1d_classifier_study" / "common_experiment_config.yaml"

    def test_revocation_excludes_measurement_from_active_view(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temp_dir:
            run = run_workbench_study(self.study, Path(temp_dir) / "run", trajectories_per_case=2)
            baseline_dir = ensure_revision_history(run.run_dir)
            baseline_rows = self._read_csv_rows(baseline_dir / "active_measurements.csv")
            self.assertGreater(len(baseline_rows), 1)
            first_measurement_id = baseline_rows[0]["measurement_id"]

            revision_id = revoke_measurement(run.run_dir, first_measurement_id, reason="sensor_invalidated")
            revision_dir = replay_revision(run.run_dir, "rev_000", revision_id)

            revised_text = (revision_dir / "active_measurements.csv").read_text(encoding="utf-8")
            self.assertNotIn(first_measurement_id, revised_text)
            delta_text = (revision_dir / "revision_delta.md").read_text(encoding="utf-8")
            self.assertIn("measurement_count_delta: `-1`", delta_text)

    def test_replay_validation_materializes(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temp_dir:
            run = run_workbench_study(self.study, Path(temp_dir) / "run", trajectories_per_case=2)
            baseline_dir = ensure_revision_history(run.run_dir)
            baseline_rows = self._read_csv_rows(baseline_dir / "active_measurements.csv")
            first_measurement_id = baseline_rows[0]["measurement_id"]

            revision_id = revoke_measurement(run.run_dir, first_measurement_id, reason="qa_rejected")
            replay_revision(run.run_dir, "rev_000", revision_id)
            delta = diff_revisions(run.run_dir, "rev_000", revision_id)
            validation = validate_replay(run.run_dir, revision_id)

            self.assertEqual(delta["right_revision_id"], revision_id)
            self.assertIn(first_measurement_id, delta["revoked_measurement_ids"])
            self.assertEqual(validation["validation_status"], "pass")
            self.assertTrue((Path(run.run_dir) / "revisions" / revision_id / "replay_validation.json").exists())

    def test_double_revocation_fails_cleanly(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temp_dir:
            run = run_workbench_study(self.study, Path(temp_dir) / "run", trajectories_per_case=2)
            baseline_dir = ensure_revision_history(run.run_dir)
            baseline_rows = self._read_csv_rows(baseline_dir / "active_measurements.csv")
            first_measurement_id = baseline_rows[0]["measurement_id"]

            revoke_measurement(run.run_dir, first_measurement_id, reason="sensor_invalidated")
            with self.assertRaisesRegex(ValueError, "already revoked"):
                revoke_measurement(run.run_dir, first_measurement_id, reason="sensor_invalidated")

    def test_revoke_restore_roundtrip_recovers_measurement(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temp_dir:
            run = run_workbench_study(self.study, Path(temp_dir) / "run", trajectories_per_case=2)
            baseline_dir = ensure_revision_history(run.run_dir)
            baseline_rows = self._read_csv_rows(baseline_dir / "active_measurements.csv")
            first_measurement_id = baseline_rows[0]["measurement_id"]

            revoke_revision = revoke_measurement(run.run_dir, first_measurement_id, reason="sensor_invalidated")
            replay_revision(run.run_dir, "rev_000", revoke_revision)
            restore_revision = restore_measurement(run.run_dir, first_measurement_id, reason="qa_restore")
            restore_dir = replay_revision(run.run_dir, revoke_revision, restore_revision)

            restored_rows = self._read_csv_rows(restore_dir / "active_measurements.csv")
            restored_ids = {row["measurement_id"] for row in restored_rows}
            self.assertIn(first_measurement_id, restored_ids)

            delta = diff_revisions(run.run_dir, "rev_000", restore_revision)
            self.assertEqual(delta["measurement_count_delta"], 0)
            self.assertEqual(delta["revoked_measurement_ids"], [])
            self.assertEqual(delta["added_measurement_ids"], [])

    def test_correction_replaces_measurement_and_preserves_count(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temp_dir:
            run = run_workbench_study(self.study, Path(temp_dir) / "run", trajectories_per_case=2)
            baseline_dir = ensure_revision_history(run.run_dir)
            baseline_rows = self._read_csv_rows(baseline_dir / "active_measurements.csv")
            first_row = baseline_rows[0]
            first_measurement_id = first_row["measurement_id"]
            corrected_value = float(first_row["measurement_value"]) + 0.5

            correction_revision = correct_measurement(
                run.run_dir,
                first_measurement_id,
                corrected_value=corrected_value,
                reason="operator_correction",
            )
            correction_dir = replay_revision(run.run_dir, "rev_000", correction_revision)
            corrected_rows = self._read_csv_rows(correction_dir / "active_measurements.csv")
            corrected_ids = {row["measurement_id"] for row in corrected_rows}
            corrected_id = f"{first_measurement_id}__corr_{correction_revision}"
            self.assertNotIn(first_measurement_id, corrected_ids)
            self.assertIn(corrected_id, corrected_ids)
            corrected_row = next(row for row in corrected_rows if row["measurement_id"] == corrected_id)
            self.assertEqual(corrected_row["measurement_value"], f"{corrected_value:.12f}")

            delta = diff_revisions(run.run_dir, "rev_000", correction_revision)
            self.assertEqual(delta["measurement_count_delta"], 0)
            self.assertIn(corrected_id, delta["added_measurement_ids"])
            self.assertIn(first_measurement_id, delta["revoked_measurement_ids"])

    def test_association_change_moves_measurement_to_target_slot(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temp_dir:
            run = run_workbench_study(self.study, Path(temp_dir) / "run", trajectories_per_case=2)
            baseline_dir = ensure_revision_history(run.run_dir)
            baseline_rows = self._read_csv_rows(baseline_dir / "active_measurements.csv")
            source_row = baseline_rows[0]
            target_row = next(
                row
                for row in baseline_rows
                if row["trajectory_id"] != source_row["trajectory_id"]
                and row["measurement_time"] == source_row["measurement_time"]
            )

            association_revision = change_measurement_association(
                run.run_dir,
                source_row["measurement_id"],
                target_row["measurement_id"],
                reason="association_fix",
            )
            association_dir = replay_revision(run.run_dir, "rev_000", association_revision)
            associated_rows = self._read_csv_rows(association_dir / "active_measurements.csv")
            source_after = next(row for row in associated_rows if row["measurement_id"] == source_row["measurement_id"])
            associated_ids = {row["measurement_id"] for row in associated_rows}

            self.assertEqual(source_after["trajectory_id"], target_row["trajectory_id"])
            self.assertEqual(source_after["measurement_index"], target_row["measurement_index"])
            self.assertEqual(source_after["measurement_time"], target_row["measurement_time"])
            self.assertEqual(source_after["measurement_value"], source_row["measurement_value"])
            self.assertNotIn(target_row["measurement_id"], associated_ids)

            delta = diff_revisions(run.run_dir, "rev_000", association_revision)
            self.assertEqual(delta["measurement_count_delta"], -1)
            self.assertIn(target_row["measurement_id"], delta["revoked_measurement_ids"])
            self.assertIn(source_row["measurement_id"], delta["mutated_measurement_ids"])

    @staticmethod
    def _read_csv_rows(path: Path) -> list[dict[str, str]]:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
