"""Figure 5 must retain the exact measured points and centering of Figure 4."""

import numpy as np
import pytest

from scripts.make_main_ac_stark_shifts import constant_drive_fwhm_khz, measured_centers


def source_arrays():
    source = {"rabi_mhz": np.array([0.0, 10.0, 20.0, 30.0, 60.0, 61.0])}
    for prefix in ("plain", "drag"):
        source[f"{prefix}_display_center_mask"] = np.array([0, 1, 0, 1, 1, 0], bool)
        source[f"{prefix}_display_centers_mhz"] = np.array([9, -0.01, 8, 0, 0.01, 7])
        source[f"{prefix}_display_mean_center_mhz"] = np.array(0.02)
        source[f"{prefix}_centers_mhz"] = source[f"{prefix}_display_centers_mhz"] + 0.02
    return source


def test_measured_curves_preserve_source_rows_and_coordinates():
    source = source_arrays()
    arrays = measured_centers(source)
    for prefix in ("plain", "drag"):
        np.testing.assert_array_equal(arrays[f"{prefix}_rabi_mhz"], [10, 30, 60])
        np.testing.assert_array_equal(arrays[f"{prefix}_source_row_indices"], [1, 3, 4])
        np.testing.assert_array_equal(arrays[f"{prefix}_centers_mhz"], [-0.01, 0, 0.01])
        np.testing.assert_allclose(
            arrays[f"{prefix}_acquisition_centers_mhz"], [0.01, 0.02, 0.03]
        )


def test_rejects_uncentered_source():
    source = source_arrays()
    source["plain_display_centers_mhz"] += 0.02
    with pytest.raises(ValueError, match="mean-centered"):
        measured_centers(source)


def test_constant_linewidth_recovers_coherence_and_saturation_limits():
    t1, t2, gamma = 51.2, 7.31, 43.5
    saturation_one_rabi = 1 / (2 * np.pi * np.sqrt(t1 * t2))
    widths = constant_drive_fwhm_khz(np.array([0, saturation_one_rabi]), t1, t2, gamma)
    np.testing.assert_allclose(widths / gamma, [1, np.sqrt(2)])
    strong_drive = float(constant_drive_fwhm_khz(60, t1, t2, gamma))
    assert strong_drive == pytest.approx(
        gamma * 2 * np.pi * 60 * np.sqrt(t1 * t2), rel=1e-7
    )
