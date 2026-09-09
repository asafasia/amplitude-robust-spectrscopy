"""Regression checks for the measured Figure 4 import and export."""

import json

import numpy as np
import pytest

from scripts.make_main_ac_stark_correction_maps import (
    CAMPAIGN,
    RUNS,
    center_display_data,
    load_source,
    validate_arrays,
)


def make_campaign(root):
    campaign = root / CAMPAIGN
    campaign.mkdir(parents=True)
    (campaign / "approved_plan.json").write_text("{}")
    records = []
    population = np.arange(40000, dtype=float).reshape(200, 200) / 40000
    for prefix, label, relative in RUNS:
        run = root / relative
        run.mkdir(parents=True)
        beta = 0.0 if prefix == "plain" else -0.22
        pulse = {
            "pulse_shape": "root_lorentzian",
            "echo": True,
            "lorentzian_length_in_ns": 50000,
            "cutoff": 0.00075,
            "drag_beta": beta,
            "stark_kappa_mhz_inv": 0.0,
            "ac_stark_correction": False,
            "applied_echo_transition_time_ns": 0.0 if prefix == "plain" else 16.0,
            "three_state_discrimination_available": False,
            "lorentzian_peak_amplitude": 1.0,
        }
        (run / "metadata.json").write_text(json.dumps({"pulse": pulse}))
        (run / "parameters.json").write_text(json.dumps({"num_shots": 2000}))
        np.savez(
            run / "sweep.npz",
            qubit=["q6"],
            detuning=np.linspace(-200000, 200000, 200),
            amp_prefactor=np.linspace(0, 1, 200),
        )
        np.savez(run / "results.npz", state=population[None])
        records.append(
            {
                "label": label,
                "drag_beta": beta,
                "rabi_frequency_mhz": np.linspace(0, 61.3, 200).tolist(),
                "center_hz_vs_amplitude": [20000.0] * 200,
                "center_fit_accepted": [False] + [True] * 199,
                "center_rms_hz": 14000.0,
                "weighted_center_hz": 20000.0,
                "spectroscopy_contrast": 0.05,
            }
        )
    (campaign / "records.json").write_text(json.dumps(records))
    return population


def test_import_preserves_orientation_units_masks_and_provenance(tmp_path):
    population = make_campaign(tmp_path)
    arrays, provenance = load_source(tmp_path)
    validate_arrays(arrays)
    np.testing.assert_array_equal(arrays["plain_pe"], population.T)
    np.testing.assert_array_equal(arrays["drag_pe"], population.T)
    np.testing.assert_allclose(arrays["plain_centers_mhz"], 0.02)
    assert not arrays["plain_fit_accepted"][0]
    np.testing.assert_array_equal(arrays["applied_echo_transition_ns"], [0, 16])
    assert len(provenance["source_sha256"]) == 10
    assert all(len(digest) == 64 for digest in provenance["source_sha256"].values())


def test_import_rejects_mismatched_acquisition_settings(tmp_path):
    make_campaign(tmp_path)
    path = tmp_path / RUNS[1][2] / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["pulse"]["stark_kappa_mhz_inv"] = 0.00225
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="stark_kappa"):
        load_source(tmp_path)


def test_rejects_invalid_measured_population(tmp_path):
    make_campaign(tmp_path)
    arrays, _ = load_source(tmp_path)
    arrays["drag_pe"][0, 0] = 1.1
    with pytest.raises(ValueError, match="outside"):
        validate_arrays(arrays)


def test_display_centering_uses_only_accepted_centers_in_visible_range():
    data = {
        "rabi_mhz": np.array([0, 10, 20, 30, 60, 61]),
        "detuning_mhz": np.array([-0.2, 0, 0.2]),
    }
    for prefix, offset in (("plain", 0.0), ("drag", 0.02)):
        data[f"{prefix}_centers_mhz"] = np.array([9, 0.01, 0.02, 8, 0.03, 7]) + offset
        data[f"{prefix}_fit_accepted"] = np.array([1, 1, 1, 0, 1, 1], bool)
    raw_centers = data["plain_centers_mhz"].copy()
    provenance = {}
    center_display_data(data, provenance)
    assert float(data["plain_display_mean_center_mhz"]) == pytest.approx(0.02)
    assert float(data["drag_display_mean_center_mhz"]) == pytest.approx(0.04)
    for prefix, _, _ in RUNS:
        selected = data[f"{prefix}_display_center_mask"]
        assert np.mean(
            data[f"{prefix}_display_centers_mhz"][selected]
        ) == pytest.approx(0)
    np.testing.assert_allclose(data["drag_display_detuning_mhz"], [-0.24, -0.04, 0.16])
    center_display_data(data, provenance)
    np.testing.assert_array_equal(data["plain_centers_mhz"], raw_centers)
    np.testing.assert_allclose(data["drag_display_detuning_mhz"], [-0.24, -0.04, 0.16])
