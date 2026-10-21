"""Guard the full-amplitude spectral objective against sparse-grid regressions."""

import numpy as np

from scripts.optimize_echo_dense import (
    T2_WIDTH_MHZ,
    loss_from_metrics,
    spectrum_metrics,
)


def synthetic(width=T2_WIDTH_MHZ):
    detuning = np.linspace(-.5, .5, 201)
    amplitudes = np.linspace(0., 100., 100)
    pe = np.tile(.65 - .4 / (1 + (2 * detuning / width)**2), (100, 1))
    pe[0] = 0.
    return np.stack([1 - pe, pe, np.zeros_like(pe)]), amplitudes, detuning


def test_zero_drive_does_not_make_operating_range_invalid():
    p, amp, det = synthetic()
    metrics = spectrum_metrics(p, amp, det)
    assert metrics["all_accepted"]
    assert len(metrics["rows"]) == 100
    assert not metrics["rows"][0]["accepted"]
    assert metrics["minimum_snr"] > 10


def test_low_snr_between_old_training_amplitudes_is_penalized():
    p, amp, det = synthetic()
    original = spectrum_metrics(p, amp, det)
    p[1, 42] = .5 + .1 * (p[1, 42] - .5)
    p[0, 42] = 1 - p[1, 42]
    changed = spectrum_metrics(p, amp, det)
    assert changed["minimum_snr"] < 3
    assert loss_from_metrics(changed) > loss_from_metrics(original) + 1e5


def test_extra_dips_are_detected_on_each_side():
    p, amp, det = synthetic()
    original = spectrum_metrics(p, amp, det)
    for center in [-.2, .2]:
        p[1, 62] -= .15 * np.exp(-((det - center) / .02)**2)
    p[0, 62] = 1 - p[1, 62]
    changed = spectrum_metrics(p, amp, det)
    assert original["max_satellite_prominence"] == 0
    assert changed["max_satellite_prominence"] > .13
    assert len(changed["rows"][62]["satellite_detunings_mhz"]) == 2
    assert loss_from_metrics(changed) > loss_from_metrics(original) + 1000


def test_spectral_template_prefers_t2_line_to_broad_line():
    narrow = spectrum_metrics(*synthetic())
    broad = spectrum_metrics(*synthetic(.15))
    assert narrow["mean_shape_error"] < 1e-20
    assert broad["mean_shape_error"] > .02


def test_below_operating_range_still_contributes_to_spectral_error():
    p, amp, det = synthetic()
    original = spectrum_metrics(p, amp, det)
    p[1, 5] = .65 - .4 / (1 + (2 * det / .2)**2)
    p[0, 5] = 1 - p[1, 5]
    changed = spectrum_metrics(p, amp, det)
    assert changed["minimum_snr"] == original["minimum_snr"]
    assert changed["mean_shape_error"] > original["mean_shape_error"]
