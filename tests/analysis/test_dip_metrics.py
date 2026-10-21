import unittest

import numpy as np

from echospec.analysis.dip_metrics import dip_metrics


class TestDipMetrics(unittest.TestCase):
    def test_gaussian_center_and_true_half_depth_width(self):
        x = np.linspace(-0.5, 0.5, 201)
        y = 0.5 - 0.2 * np.exp(-0.5 * ((x - 0.03) / 0.05) ** 2)
        result = dip_metrics(x, y)
        self.assertTrue(result["accepted"])
        self.assertAlmostEqual(result["center_hz"], 30000, places=5)
        self.assertLess(abs(result["fwhm_hz"] - 2.354820045 * 0.05 * 1e6), 100)

    def test_sloping_background_is_removed_for_width(self):
        x = np.linspace(-0.5, 0.5, 201)
        y = 0.5 + 0.1 * x - 0.2 * np.exp(-0.5 * ((x - 0.03) / 0.05) ** 2)
        result = dip_metrics(x, y)
        self.assertTrue(result["accepted"])
        self.assertLess(abs(result["fwhm_hz"] - 2.354820045 * 0.05 * 1e6), 100)

    def test_flat_trace_has_no_resonance(self):
        x = np.linspace(-0.5, 0.5, 201)
        self.assertFalse(dip_metrics(x, np.zeros_like(x))["accepted"])
