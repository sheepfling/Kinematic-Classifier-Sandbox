from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.utils.io import read_csv_rows, union_fieldnames, write_csv


class UtilsIoTests(unittest.TestCase):
    def test_union_fieldnames_preserves_order(self) -> None:
        rows = [
            {"a": 1, "b": 2},
            {"b": 3, "c": 4},
        ]
        self.assertEqual(union_fieldnames(rows), ["a", "b", "c"])

    def test_read_csv_rows_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rows.csv"
            write_csv(path, [{"a": "1", "b": "2"}], ["a", "b"])
            self.assertEqual(read_csv_rows(path), [{"a": "1", "b": "2"}])


if __name__ == "__main__":
    unittest.main()
