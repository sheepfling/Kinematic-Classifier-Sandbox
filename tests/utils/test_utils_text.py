from __future__ import annotations

import unittest

from kinematic_classifier_sandbox.utils.text import markdown_table_preview


class UtilsTextTests(unittest.TestCase):
    def test_markdown_table_preview(self) -> None:
        markdown = markdown_table_preview([{"a": 1, "b": 2}, {"a": 3, "b": 4}], ["a", "b"])
        self.assertIn("a", markdown)
        self.assertIn("b", markdown)
        self.assertIn("1", markdown)
        self.assertIn("2", markdown)


if __name__ == "__main__":
    unittest.main()
