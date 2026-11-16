"""The logarithmic survey export preserves samples, masks and provenance."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import make_supplemental_extended_cutoff as figure


@pytest.mark.parametrize("durations", ([10, 20, 30, 45, 60], [10, 20, 30, 40, 50]))
def test_log_survey_export_preserves_grid_and_handles_no_robust_cutoff(
    tmp_path, monkeypatch, durations,
):
    figure.plt.switch_backend("Agg")
    monkeypatch.setattr(figure, "ROOT", tmp_path)
    generator = tmp_path / "generator.py"
    generator.write_bytes(Path(figure.__file__).read_bytes())
    monkeypatch.setattr(figure, "__file__", str(generator))
    style = tmp_path / "style.mplstyle"
    style.write_text("")
    monkeypatch.setattr(figure, "STYLE_PATH", style)
    # With a 75 kHz reference and 150 kHz FWHM, Q is half the contrast.
    monkeypatch.setattr(figure, "REFERENCE_FWHM_HZ", 75000.)
    source = tmp_path / "study"
    source.mkdir()
    output = tmp_path / "figures"
    durations = np.array(durations)
    cutoffs, amplitudes = np.geomspace(1e-5, 1, 100), np.geomspace(1, 100, 100)
    accepted = np.ones((5, 100, 100), dtype=bool)
    accepted[-1, :, -1] = False
    widths = np.full((5, 100, 100), 150000.)
    contrast = np.broadcast_to(np.linspace(.1, .4, 100)[:, None], (100, 100))
    config = dict(power=2, amplitude_spacing="log", transition_us=.016,
                  drag_beta=-0.5)
    (source / "dataset_manifest.json").write_text(json.dumps(dict(
        durations_us=durations.tolist(), reused_durations_us=[10, 20, 30],
        baseline_copy="baseline_copy", new_runs="simulations",
    )))
    comparison = source / "comparison_metrics.npz"
    np.savez(comparison, durations_us=durations, cutoffs=cutoffs, rabi_mhz=amplitudes,
             accepted=accepted, fwhm_hz=widths, configuration=json.dumps(config),
             pending_durations_us=[])
    paths = [comparison]
    for index, duration in enumerate(durations):
        folder = "baseline_copy" if duration in (10, 20, 30) else "simulations"
        path = source / f"{folder}/L{duration}us/scan.npz"
        path.parent.mkdir(parents=True)
        np.savez(path, cutoffs=cutoffs, rabi_mhz=amplitudes, accepted=accepted[index],
                 fwhm_hz=widths[index], contrast=contrast)
        paths.append(path)
    (source / "checksums.sha256").write_text("".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(source)}\n"
        for path in paths
    ))
    figure.main(["--input-dir", str(source), "--output-dir", str(output),
                 "--stem", "log_test"])
    with np.load(output / "log_test.npz", allow_pickle=False) as exported:
        np.testing.assert_array_equal(exported["durations_us"], durations)
        np.testing.assert_array_equal(exported["rabi_mhz"], amplitudes)
        np.testing.assert_array_equal(exported["accepted"], accepted)
        np.testing.assert_array_equal(exported["fwhm_hz"], widths)
        np.testing.assert_array_equal(exported["contrast"][0], contrast)
        np.testing.assert_allclose(exported["quality"][0, 0], .05)
        np.testing.assert_allclose(exported["quality"][0, -1], .20)
        assert not exported["quality_above_threshold"][0, 0].any()
        assert exported["quality_above_threshold"][0, -1].all()
        assert np.isnan(exported["quality"][-1, :, -1]).all()
        assert not exported["quality_above_threshold"][-1, :, -1].any()
    provenance = json.loads((output / "log_test_provenance.json").read_text())
    assert provenance["configuration"] == config
    assert provenance["drag_beta_convention"]["source_beta"] == -0.5
    assert provenance["drag_beta_convention"]["paper_beta"] == 0.5
    assert provenance["amplitude_display"]["sampling"] == "logarithmic"
    assert provenance["amplitude_display"]["positive_sample_count"] == 100
    assert provenance["amplitude_display"]["cell_edges"] == "Geometric midpoints"
    assert provenance["quality_outline"]["threshold"] == .10
    assert provenance["color_scales"]["quality"]["vmax"] == .2
    summary = json.loads((output / "log_test_summary.json").read_text())
    assert summary["results"][-1]["cutoff"] is None
    assert (output / "log_test.pdf").stat().st_size > 0
