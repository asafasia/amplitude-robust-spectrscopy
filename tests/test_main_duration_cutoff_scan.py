"""Figure 4 duration selection and contrast-boundary regression checks."""

import unittest

import numpy as np

from scripts.make_main_duration_cutoff_scan import (
    select_cutoff_slices,
    select_duration_maps,
)


class TestDurationMaps(unittest.TestCase):
    def test_cutoff_slices_share_amplitude_column_and_preserve_masks(self):
        amps = np.array([0, 49.5, 75])
        width = np.arange(18, dtype=float).reshape(3, 2, 3) + 100_000
        contrast = np.arange(18, dtype=float).reshape(3, 2, 3) / 20
        accepted = np.ones_like(width, dtype=bool)
        accepted[2, 1, 1] = False
        index, slices = select_cutoff_slices(amps, accepted, width, contrast)
        self.assertEqual(index, 1)
        np.testing.assert_array_equal(slices["slice_fwhm_hz"], width[:, :, 1])
        np.testing.assert_array_equal(slices["slice_contrast"], contrast[:, :, 1])
        np.testing.assert_array_equal(slices["slice_accepted"], accepted[:, :, 1])

    def test_exact_duration_preserves_orientation_and_rejected_cells(self):
        lengths = np.array([10, 30, 60])
        width = np.arange(18, dtype=float).reshape(3, 2, 3) + 100_000
        contrast = np.zeros_like(width)
        contrast[1] = [[0.19, 0.2, 0.21], [0.8, 0.5, np.nan]]
        accepted = np.ones_like(width, dtype=bool)
        accepted[1, 1, 1:] = False
        data = select_duration_maps(lengths, accepted, width, contrast)
        np.testing.assert_array_equal(data["fwhm_hz"], width[1])
        np.testing.assert_array_equal(data["contrast"], contrast[1])
        np.testing.assert_array_equal(data["accepted"], accepted[1])
        np.testing.assert_array_equal(
            data["contrast_above_threshold"], [[False, False, True], [True, False, False]]
        )
        data["fwhm_hz"][0, 0] = -1
        self.assertEqual(width[1, 0, 0], 100_006)

    def test_missing_or_duplicate_duration_is_rejected(self):
        values = np.ones((2, 2, 3))
        for lengths in ([10, 60], [30, 30]):
            with self.subTest(lengths=lengths):
                with self.assertRaisesRegex(ValueError, "exactly one saved row"):
                    select_duration_maps(lengths, values > 0, values, values)
