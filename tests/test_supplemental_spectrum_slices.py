"""Protect acquired-row selection when the source map grid changes."""

import unittest

import numpy as np

from scripts.make_supplemental_spectrum_slices import (
    MAPS,
    TARGETS_MHZ,
    measurement_map_context,
    select_rows,
)


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


class SpectrumMapContextTests(unittest.TestCase):
    def setUp(self):
        self.data = {}
        self.maps = {}
        for protocol, name in MAPS.items():
            x = np.array([-0.5, 0.0, 0.5])
            drives = np.array([0.0, 3.0, 5.0, 10.0])
            population = np.arange(12).reshape(4, 3) / 12
            indices = np.array([1, 3])
            prefix = name.lower()
            self.maps.update({
                f"{prefix}_detuning_hz": x * 1e6,
                f"{prefix}_nominal_rabi_mhz": drives,
                f"{prefix}_state": population.copy(),
            })
            self.data.update({
                f"{protocol}_detuning_mhz": x,
                f"{protocol}_amplitude_indices": indices,
                f"{protocol}_rabi_mhz": drives[indices],
                f"{protocol}_experiment_raw": population[indices].copy(),
            })

    def test_preserves_full_map_and_exact_selected_rows(self):
        context = measurement_map_context(self.data, self.maps)
        for protocol in MAPS:
            x, drives, population = context[protocol]
            indices = self.data[f"{protocol}_amplitude_indices"]
            self.assertEqual(population.shape, (4, 3))
            np.testing.assert_array_equal(x, self.data[f"{protocol}_detuning_mhz"])
            np.testing.assert_array_equal(drives[indices], self.data[f"{protocol}_rabi_mhz"])
            np.testing.assert_array_equal(population[indices], self.data[f"{protocol}_experiment_raw"])

    def test_rejects_stale_slice_populations(self):
        self.maps["a3_narrow_no_echo_state"][1, 1] += 0.01
        with self.assertRaisesRegex(ValueError, "does not match"):
            measurement_map_context(self.data, self.maps)

    def test_rejects_changed_frequency_or_drive_axis(self):
        for field in ("detuning_hz", "nominal_rabi_mhz"):
            with self.subTest(field=field):
                maps = {key: value.copy() for key, value in self.maps.items()}
                maps[f"a3_narrow_no_echo_{field}"][1] += 0.01
                with self.assertRaisesRegex(ValueError, "does not match"):
                    measurement_map_context(self.data, maps)

    def test_rejects_transposed_map(self):
        self.maps["a3_narrow_no_echo_state"] = self.maps["a3_narrow_no_echo_state"].T
        with self.assertRaisesRegex(ValueError, "drive-by-detuning"):
            measurement_map_context(self.data, self.maps)


if __name__ == "__main__":
    unittest.main()
