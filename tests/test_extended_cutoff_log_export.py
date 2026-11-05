"""The logarithmic survey export preserves samples, masks and provenance."""

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts import make_supplemental_extended_cutoff as figure


def test_log_survey_export_preserves_grid_and_handles_no_robust_cutoff(
    tmp_path, monkeypatch,
):
    figure.plt.switch_backend("Agg")
    monkeypatch.setattr(figure, "ROOT", tmp_path)
    generator = tmp_path / "generator.py"
    generator.write_bytes(Path(figure.__file__).read_bytes())
    monkeypatch.setattr(figure, "__file__", str(generator))
    style = tmp_path / "style.mplstyle"
    style.write_text("")
    monkeypatch.setattr(figure, "STYLE_PATH", style)
    source = tmp_path / "study"
    source.mkdir()
    output = tmp_path / "figures"
    durations = np.array([10, 20, 30, 45, 60])
    cutoffs, amplitudes = np.geomspace(1e-5, 1, 100), np.geomspace(1, 100, 100)
    accepted = np.ones((5, 100, 100), dtype=bool)
    accepted[-1, :, -1] = False
    widths = np.full((5, 100, 100), 150000.)
    contrast = np.broadcast_to(np.linspace(.1, .4, 100)[:, None], (100, 100))
    config = dict(power=2, amplitude_spacing="log", transition_us=.016)
    (source / "dataset_manifest.json").write_text(json.dumps(dict(
        reused_durations_us=[], baseline_copy=None, new_runs="simulations",
    )))
    comparison = source / "comparison_metrics.npz"
    np.savez(comparison, durations_us=durations, cutoffs=cutoffs, rabi_mhz=amplitudes,
             accepted=accepted, fwhm_hz=widths, configuration=json.dumps(config),
             pending_durations_us=[])
    paths = [comparison]
    for index, duration in enumerate(durations):
        path = source / f"simulations/L{duration}us/scan.npz"
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
        np.testing.assert_array_equal(exported["rabi_mhz"], amplitudes)
        np.testing.assert_array_equal(exported["accepted"], accepted)
        np.testing.assert_array_equal(exported["fwhm_hz"], widths)
        np.testing.assert_array_equal(exported["contrast"][0], contrast)
    provenance = json.loads((output / "log_test_provenance.json").read_text())
    assert provenance["configuration"] == config
    assert provenance["amplitude_display"]["sampling"] == "logarithmic"
    assert provenance["amplitude_display"]["positive_sample_count"] == 100
    assert provenance["amplitude_display"]["cell_edges"] == "Geometric midpoints"
    summary = json.loads((output / "log_test_summary.json").read_text())
    assert summary["results"][-1]["cutoff"] is None
    assert (output / "log_test.pdf").stat().st_size > 0
