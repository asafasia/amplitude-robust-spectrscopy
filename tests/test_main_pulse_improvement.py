"""Checks for the Figure 3 resonance-center RMS convention."""

import unittest

import numpy as np

from scripts.make_main_pulse_improvement import center_stability


class TestCenterStability(unittest.TestCase):
    def test_rms_preserves_constant_offset_instead_of_recentering(self):
        result = center_stability([10, 50, 100], [3000.0, 3000.0, 3000.0])
        self.assertEqual(result["rms_from_zero_hz"], 3000.0)
        self.assertEqual(result["mean_hz"], 3000.0)
        self.assertEqual(result["standard_deviation_hz"], 0.0)

    def test_selection_uses_common_inclusive_range_and_finite_centers(self):
        result = center_stability(
            [0, 9, 10, 50, 100, 101],
            [np.nan, 1e6, -3000.0, np.nan, 4000.0, 1e6],
        )
        self.assertEqual(result["sample_count"], 2)
        self.assertAlmostEqual(result["rms_from_zero_hz"], np.sqrt(12_500_000))
        self.assertEqual(result["mean_hz"], 500.0)
        self.assertEqual(result["standard_deviation_hz"], 3500.0)

    def test_empty_selection_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "No finite resonance centers"):
            center_stability([0, 10], [0, np.nan])
