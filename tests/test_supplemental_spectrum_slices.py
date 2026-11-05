"""Protect acquired-row selection when the source map grid changes."""

import unittest

import numpy as np

from scripts.make_supplemental_spectrum_slices import TARGETS_MHZ, select_rows


class SpectrumSliceSelectionTests(unittest.TestCase):
    def test_current_200_point_grid(self):
        drives = np.linspace(0, 20, 200)
        indices = select_rows(drives, TARGETS_MHZ)
        np.testing.assert_array_equal(indices, [30, 50, 99, 149, 199])
        np.testing.assert_allclose(drives[indices], TARGETS_MHZ, atol=0.051)

    def test_rejects_old_targets_outside_current_map(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            select_rows(np.linspace(0, 20, 200), [3, 10, 20, 30, 40])

    def test_rejects_repeated_acquired_rows(self):
        with self.assertRaisesRegex(ValueError, "Distinct"):
            select_rows([0, 10, 20], [1, 2])


if __name__ == "__main__":
    unittest.main()
