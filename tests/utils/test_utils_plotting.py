from __future__ import annotations

import unittest

from kinematic_classifier_sandbox.utils.plotting import figure_to_png_bytes, prepare_matplotlib


class UtilsPlottingTests(unittest.TestCase):
    def test_figure_to_png_bytes(self) -> None:
        plt = plt
        fig, ax = plt.subplots(figsize=(2.0, 1.5))
        ax.plot([0.0, 1.0], [1.0, 0.0])

        payload = figure_to_png_bytes(fig, dpi=120)

        self.assertGreater(len(payload), 1000)
        self.assertEqual(payload[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
