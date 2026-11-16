"""Figures must use matching, verified fit arrays and preserve rejection masks."""

import hashlib
import json

import numpy as np
import pytest

from echospec.analysis import cutoff_fit_exports as exports
from echospec.analysis.robust_dip_fit import FIT_SETTINGS


def make_export(tmp_path, monkeypatch, *, reaccept=False):
    monkeypatch.setattr(exports, "ROOT", tmp_path)
    directory = tmp_path / "study/robust_fits"
    directory.mkdir(parents=True)
    source = tmp_path / "source.py"
    source.write_text("simulation and fitter\n")
    hashes = {"source.py": hashlib.sha256(source.read_bytes()).hexdigest()}
    lengths, cutoffs, amplitudes = np.array([20]), np.array([.001, .01]), np.array([10, 50])
    mask = np.array([[[True, True], [True, False]]])
    fitted_mask = mask.copy()
    fitted_mask[0, 1, 1] = reaccept
    shape = mask.shape
    path = directory / "metrics.npz"
    np.savez_compressed(
        path, durations_us=lengths, cutoffs=cutoffs, rabi_mhz=amplitudes,
        accepted=fitted_mask, fwhm_hz=np.full(shape, 100000.),
        contrast=np.full(shape, .2), center_hz=np.zeros(shape),
        relative_rms=np.full(shape, .01), relative_width_error=np.full(shape, .001),
        radius_mhz=np.full(shape, .2), status=np.full(shape, "accepted"),
        configuration=json.dumps(dict(source_sha256=hashes, settings=FIT_SETTINGS)),
    )
    (directory / "provenance.json").write_text(json.dumps(dict(
        complete=True, source_sha256=hashes, settings=FIT_SETTINGS,
        arrays_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )))
    return tmp_path / "study", lengths, cutoffs, amplitudes, mask


def test_verified_fits_preserve_values_and_detect_changed_source(tmp_path, monkeypatch):
    arguments = make_export(tmp_path, monkeypatch)
    arrays, _, hashes = exports.load_cutoff_fits(*arguments)
    np.testing.assert_array_equal(arrays["accepted"], arguments[-1])
    np.testing.assert_array_equal(arrays["fwhm_hz"], np.full((1, 2, 2), 100000.))
    assert "study/robust_fits/metrics.npz" in hashes
    (tmp_path / "source.py").write_text("changed fitter\n")
    with pytest.raises(ValueError, match="source changed"):
        exports.load_cutoff_fits(*arguments)


def test_fit_cannot_reaccept_a_rejected_source_spectrum(tmp_path, monkeypatch):
    arguments = make_export(tmp_path, monkeypatch, reaccept=True)
    with pytest.raises(ValueError, match="reaccepted"):
        exports.load_cutoff_fits(*arguments)


def test_fit_coordinates_must_match_the_figure(tmp_path, monkeypatch):
    source, lengths, cutoffs, amplitudes, mask = make_export(tmp_path, monkeypatch)
    with pytest.raises(AssertionError):
        exports.load_cutoff_fits(source, lengths, cutoffs, amplitudes + 1, mask)
