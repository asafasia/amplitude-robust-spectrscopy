"""Matched-study safeguards and conservative fixed-cutoff selection."""

import base64
import csv
import json

import numpy as np
import pytest

from echospec.analysis.duration_cutoffs import (
    METRIC_KEYS,
    check_matching_configuration,
    load_study,
    pareto_mask,
    robust_cutoff_summary,
)


def test_only_duration_may_change_between_matched_runs():
    common = dict(drag_beta=-.5, power=5, dt_ns=.25, rabi_mhz=[0, 50, 100])
    check_matching_configuration(dict(common, durations_us=[10, 20]), common, 20)
    with pytest.raises(ValueError, match="drag_beta"):
        check_matching_configuration(dict(common, drag_beta=.5, durations_us=[20]), common, 20)
    with pytest.raises(ValueError, match="absent"):
        check_matching_configuration(dict(common, durations_us=[30]), common, 20)


def test_robust_score_uses_worst_amplitude_and_disqualifies_missing_dips():
    metrics = dict(accepted=np.array([[False, True, True], [False, True, False],
                                      [False, True, True]]),
                   fwhm_hz=np.array([[np.nan, 100e3, 200e3], [np.nan, 20e3, 20e3],
                                     [np.nan, 200e3, 200e3]]),
                   contrast_snr=np.array([[np.nan, 10, 8], [np.nan, 30, 30],
                                           [np.nan, 9, 9]]))
    result = robust_cutoff_summary(metrics, [0, 10, 100], [1e-4, .001, .01])
    assert result["cutoff"] == .01
    assert result["minimum_snr_per_mhz"] == 45
    assert result["worst_fwhm_khz"] == 200
    assert not result["all_amplitudes_snr_at_least_10"]


def test_no_complete_cutoff_does_not_receive_a_score():
    metrics = dict(accepted=np.array([[True, False]]), fwhm_hz=np.array([[100e3, 100e3]]),
                   contrast_snr=np.array([[10, np.nan]]))
    assert robust_cutoff_summary(metrics, [10, 100], [.001]) is None
    assert robust_cutoff_summary(metrics, [0, 1], [.001]) is None


def test_pareto_handles_ties_and_invalid_points():
    width = [100, 100, 100, 200, 50, np.nan, 80]
    snr = [10, 10, 9, 12, 6, 20, -1]
    np.testing.assert_array_equal(pareto_mask(width, snr),
                                  [True, True, False, True, True, False, False])


@pytest.fixture
def saved_study(tmp_path):
    """Small physical cube with distinct cutoff/amplitude/detuning values."""
    common = dict(cutoffs=[1e-5, 1.], rabi_mhz=[0., 5., 15., 30., 40., 55., 80., 100.],
                  detuning_mhz=[-.5, 0., .5], drag_beta=-.5)
    manifest = dict(common_configuration=common, durations_us=[10., 20.],
                    reused_durations_us=[], baseline_copy="baseline", new_runs="runs")
    (tmp_path / "dataset_manifest.json").write_text(json.dumps(manifest))
    directory = tmp_path / "runs/L10us"
    directory.mkdir(parents=True)
    pe = np.arange(48).reshape(2, 8, 3) / 100
    population = np.stack([1 - pe, pe, np.zeros_like(pe)], axis=1)
    accepted = np.tile([False, True, True, True, True, True, True, True], (2, 1))
    metrics = {key: np.where(accepted, 1., np.nan) for key in METRIC_KEYS}
    metrics["accepted"] = accepted
    arrays = dict(populations=population, **metrics, cutoffs=common["cutoffs"],
                  rabi_mhz=common["rabi_mhz"], detuning_mhz=common["detuning_mhz"],
                  configuration=json.dumps(dict(common, durations_us=[10., 20.])),
                  duration_us=10.)
    np.savez(directory / "scan.npz", **arrays)
    rows = [dict(duration_us=10., cutoff=c, amplitude_mhz=a,
                  **{key: values[ci, ai] for key, values in metrics.items()})
            for ci, c in enumerate(common["cutoffs"])
            for ai, a in enumerate(common["rabi_mhz"])]
    with (directory / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (directory / "summary.json").write_text(json.dumps(
        dict(diagnostics=[dict(min_density_eigenvalue=0.)])))
    return tmp_path, directory, arrays


def test_partial_study_keeps_missing_lengths_explicit_and_maps_aligned(saved_study):
    root, _, arrays = saved_study
    with pytest.raises(FileNotFoundError, match="L20us"):
        load_study(root)
    study = load_study(root, allow_partial=True)
    assert study["pending"] == [20.]
    assert [c["duration_us"] for c in study["cases"]] == [10.]
    np.testing.assert_array_equal(study["cases"][0]["population_maps"],
                                  arrays["populations"][:, 1].astype("<f4"))
    assert study["cases"][0]["validation"]["accepted_traces"] == 14


def test_browser_population_payload_retains_every_sample_in_documented_order(saved_study):
    from scripts.report_polynomial_durations import payload

    root, _, arrays = saved_study
    exported = payload(load_study(root, allow_partial=True))
    encoded = exported["cases"][0]["population_base64"]
    decoded = np.frombuffer(base64.b64decode(encoded), dtype="<f4").reshape(2, 8, 3)
    np.testing.assert_array_equal(decoded, arrays["populations"][:, 1].astype("<f4"))
    assert len(exported["amps"]) == 8


def test_combining_rejects_plausible_but_reordered_axes(saved_study):
    root, directory, arrays = saved_study
    arrays["cutoffs"] = arrays["cutoffs"][::-1]
    np.savez(directory / "scan.npz", **arrays)
    with pytest.raises(AssertionError):
        load_study(root, allow_partial=True)


def test_combining_rejects_stale_csv_values(saved_study):
    root, directory, _ = saved_study
    path = directory / "metrics.csv"
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    rows[-1]["contrast_snr"] = "999"
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="CSV contrast_snr mismatch"):
        load_study(root, allow_partial=True)


def test_combining_rejects_nonphysical_raw_data(saved_study):
    root, directory, arrays = saved_study
    arrays["populations"][0, 0, 1, 2] += .01
    np.savez(directory / "scan.npz", **arrays)
    with pytest.raises(ValueError, match="Nonphysical"):
        load_study(root, allow_partial=True)
