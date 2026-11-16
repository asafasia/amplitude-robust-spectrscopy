"""Check raw-data preservation when paper maps have a rectangular grid."""

import json
from functools import partial

import numpy as np
import pytest

from scripts import make_latest_paper_measurements as paper


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    session = tmp_path / "campaign"
    session.mkdir()
    (session / "waveforms").mkdir()
    axis = np.linspace(0, 100, 3)
    state = np.arange(15, dtype=float).reshape(3, 5) / 20
    items, records = [], []
    for index, name in enumerate(paper.F3):
        raw = session / name
        raw.mkdir()
        np.savez(raw / "results.npz", state=state.T[None, ...])
        np.savez(
            session / f"{name}.npz", state=state,
            amplitude_factor=axis / 100, peak_rabi_mhz=axis,
        )
        np.savez(session / "waveforms" / f"{name}.npz", I=[0, 1], Q=[0, 0])
        items.append(dict(name=name, parameters=dict(
            num_shots=5000, qubits=["q1"], amp_factor_points=3,
            frequency_points=5, lorentzian_length_in_ns=20000,
            cutoff=0.005, reset_type="active", simulate=False,
            echo_transition_time_ns=50,
            drag_beta=0 if index == 0 else 1.2557575757575759,
            edge_smoothing="polynomial" if index == 2 else "none",
            edge_smoothing_power=2,
        )))
        records.append(dict(
            name=name, raw_directory=name, processed_npz=f"{name}.npz",
            status=dict(raw_data_saved=True, interrupted=False),
            fits=[dict(accepted=i > 0, center_hz=i, center_error_hz=1,
                       fwhm_hz=40000, contrast=0.2, r_squared=0.9)
                  for i in range(3)],
        ))
    (session / "plan.json").write_text(json.dumps(dict(
        maps=items, nominal_rabi_hz_per_volt=1e8,
        frequency_reference_hz=4.2e9, reference_pi={},
    )))
    (session / "summary.json").write_text(json.dumps(dict(
        status="complete", records=records,
    )))
    monkeypatch.setattr(
        paper, "save_paper_dataset",
        partial(paper.save_paper_dataset, data_root=tmp_path / "derived"),
    )
    return tmp_path, state


def export(root):
    paper.export_campaign(
        root, "campaign", paper.F3, "example", 5000, "peak_rabi_mhz", (3, 5)
    )


def test_rectangular_map_preserves_data_masks_and_shot_count(campaign):
    root, state = campaign
    export(root)
    with np.load(root / "derived/experimental/example.npz") as saved:
        np.testing.assert_array_equal(saved["e1_no_drag_no_poly_state"], state)
        np.testing.assert_array_equal(
            saved["e1_no_drag_no_poly_accepted"], [False, True, True]
        )
    meta = json.loads((root / "derived/experimental/example.json").read_text())
    assert meta["provenance"]["grid"] == [3, 5]
    assert meta["provenance"]["spectroscopy_shots"] == 225000


def test_processed_map_cannot_replace_different_raw_values(campaign):
    root, state = campaign
    np.savez(
        root / "campaign/E1_no_drag_no_poly/results.npz",
        state=(state + 0.01).T[None, ...],
    )
    with pytest.raises(AssertionError):
        export(root)


def test_figure3_statistics_exclude_out_of_view_outliers_and_rejected_rows(monkeypatch):
    # A campaign-specific linewidth must not replace the common paper reference.
    reference_hz = 1 / (np.pi * 7.6e-6)
    data = {}
    for name in paper.F3:
        prefix = name.lower()
        for field, value in {
            "peak_rabi_mhz": [0, 20, 40, 60, 100],
            "detuning_hz": [-500000, 0, 500000],
            "state": np.full((5, 3), 0.3),
            "accepted": [True, True, name != paper.F3[1], True, True],
            "center_hz": [0, 1000, 2000, 3000, 1e12],
            "center_error_hz": np.ones(5),
            "fwhm_hz": [0, 100000, 200000, 300000, 1e12],
            "contrast": [0, 0.1, 0.2, 0.3, 1],
        }.items():
            data[f"{prefix}_{field}"] = np.asarray(value)
    saved = {}

    def capture(fig, stem, metadata):
        saved.update(metadata)
        for ax in fig.axes[:3]:
            assert ax.get_xlabel() == r"$\Delta/(2\pi)$ (kHz)"
            np.testing.assert_allclose(
                [line.get_xdata()[0] for line in ax.lines],
                np.array([-1, 1]) * reference_hz / 2000,
            )
        center = fig.axes[3]
        assert center.get_ylabel() == r"$\Delta_{\mathrm{res}}/(2\pi)$ (kHz)"
        assert center.get_xlabel() == r"$\Omega_{\mathrm{pk}}/(2\pi)$ (MHz)"
        np.testing.assert_allclose(
            center.lines[0].get_ydata(), [np.nan, 1, 2, 3, np.nan], equal_nan=True
        )
        fwhm = fig.axes[5]
        np.testing.assert_allclose(
            fwhm.lines[0].get_ydata(),
            np.array([np.nan, 100000, 200000, 300000, np.nan]) / reference_hz,
            equal_nan=True,
        )
        fig.canvas.draw()
        np.testing.assert_allclose(fwhm.get_ylim(), [0, 12])
        assert not fwhm.child_axes
        assert fwhm.get_ylabel() == r"$\mathrm{FWHM}/\Gamma_{T_2^*}$"
        assert metadata["fwhm_axes"]["normalization_hz"] == pytest.approx(reference_hz)
        assert len(fwhm.lines) == 4
        paper.plt.close(fig)

    monkeypatch.setattr(paper, "save", capture)
    reference = dict(fwhm_limit_hz=40000, t2_star_s=1 / (np.pi * 40000))
    stats = paper.plot_fig3(data, dict(coherence=dict(before=reference, after=reference)))
    assert saved["common_indices"] == [1, 3]
    assert [s["accepted"] for s in stats] == [3, 2, 3]
    for result in stats:
        assert result["matched_count"] == 2
        assert result["displayed_amplitude_points"] == 4
        assert result["median_fwhm_khz"] == 200
        assert result["median_fwhm_normalized"] == pytest.approx(200000 / reference_hz)
        assert result["median_contrast"] == pytest.approx(0.2)
        assert result["center_mean_khz"] == 2
        assert result["center_rms_about_mean_khz"] == 1


def test_representative_ramsey_uses_mean_time_and_ignores_rejected_fits():
    def reference(tau, accepted=True):
        return dict(t2_star_s=tau, fwhm_limit_hz=1 / (np.pi * tau), accepted=accepted)

    result = paper.representative_ramsey(dict(coherence=dict(
        before=reference(6e-6), after=reference(10e-6), rejected=reference(1e-6, False),
    )))
    assert result["t2_star_s"] == pytest.approx(8e-6)
    assert result["fwhm_limit_hz"] == pytest.approx(1 / (np.pi * 8e-6))
    assert result["source_phases"] == ["before", "after"]
