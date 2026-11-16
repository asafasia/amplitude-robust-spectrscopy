"""Protect sample selection and provenance in the saved square-echo export."""

import json

import numpy as np
import pytest

from scripts.make_supplemental_square_echo import import_saved, prepare_arrays


def saved_arrays():
    drive = np.array([0.05, 0.2, 3.05, 10.05, 20.05, 30.0])
    detuning = np.array([-0.5, 0.0, 0.5])
    pe = np.arange(18).reshape(6, 3) / 30
    populations = np.stack([1 - pe, pe, np.zeros_like(pe)], axis=-1)
    response = {
        "amplitude_mhz": drive,
        "detuning_mhz": detuning,
        "dissipative3_square": populations,
        "dissipative3_root": populations[:, ::-1].copy(),
    }
    controls = {
        "amplitude_mhz": np.array([3.05, 10.05]),
        "wide_detuning_mhz": np.array([-10.0, 0.0, 10.0]),
        "wide_square": populations[2:4].copy(),
        "wide_root": populations[2:4, ::-1].copy(),
    }
    return response, controls


def test_export_retains_original_rows_populations_and_wide_slice():
    response, controls = saved_arrays()
    arrays = prepare_arrays(response, controls)
    np.testing.assert_array_equal(arrays["source_amplitude_indices"], [1, 2, 3, 4])
    np.testing.assert_array_equal(arrays["slice_indices"], [1, 2])
    np.testing.assert_array_equal(
        arrays["square_population"], response["dissipative3_square"][1:5]
    )
    np.testing.assert_array_equal(
        arrays["wide_root_population"], controls["wide_root"][1]
    )
    np.testing.assert_allclose(arrays["square_variation"], 2 / 30)


def test_missing_target_drive_is_not_replaced_by_nearest_row():
    response, controls = saved_arrays()
    response["amplitude_mhz"][2] = 3.04
    with pytest.raises(ValueError, match="saved sample at 3.05"):
        prepare_arrays(response, controls)


def test_import_rejects_changed_source_before_export(tmp_path):
    for name in ("configuration", "validation", "additional_validation"):
        (tmp_path / f"{name}.json").write_text("{}")
    (tmp_path / "postprocessing_provenance.json").write_text(
        json.dumps({"response_sha256": "not-the-source-hash"})
    )
    (tmp_path / "responses.npz").write_bytes(b"changed source")
    with pytest.raises(ValueError, match="recorded SHA-256"):
        import_saved(tmp_path)
