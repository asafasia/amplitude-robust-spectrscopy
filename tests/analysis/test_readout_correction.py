import unittest

import numpy as np

from echospec.analysis.readout_correction import correct_binary_readout


class ReadoutCorrectionTests(unittest.TestCase):
    def test_recovers_populations_with_row_prepared_convention(self):
        matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
        populations = np.array([0.0, 0.25, 1.0])
        observed = (np.column_stack((1 - populations, populations)) @ matrix)[:, 1]
        corrected, _ = correct_binary_readout(observed, matrix, 2000)
        np.testing.assert_allclose(corrected, populations, atol=1e-15)

    def test_scales_noise_and_contrast_together(self):
        measured = np.array([0.2, 0.5])
        raw_error = np.sqrt(measured * (1 - measured) / 2000)
        corrected, error = correct_binary_readout(
            measured, [[0.9, 0.1], [0.2, 0.8]], 2000
        )
        raw_snr = np.diff(measured)[0] / np.linalg.norm(raw_error)
        corrected_snr = np.diff(corrected)[0] / np.linalg.norm(error)
        self.assertAlmostEqual(raw_snr, corrected_snr)

    def test_retains_out_of_range_estimates(self):
        corrected, _ = correct_binary_readout(
            [0, 1], [[0.9, 0.1], [0.2, 0.8]], 2000
        )
        self.assertLess(corrected[0], 0)
        self.assertGreater(corrected[1], 1)

    def test_rejects_invalid_inputs(self):
        for matrix in ([[0.5, 0.5], [0.5, 0.5]], [[0.9, 0.2], [0.2, 0.8]]):
            with self.assertRaises(ValueError):
                correct_binary_readout([0.2], matrix, 2000)
        with self.assertRaises(ValueError):
            correct_binary_readout([1.1], np.eye(2), 2000)
        with self.assertRaises(ValueError):
            correct_binary_readout([0.2], np.eye(2), 0)


if __name__ == "__main__":
    unittest.main()
