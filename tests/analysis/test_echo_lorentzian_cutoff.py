from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from echospec.analysis.echo_lorentzian_cutoff import (
    AggregateCampaign,
    RawSweep,
    aggregate_quality_mask,
    analyze_sweep,
    best_amplitude_index,
    plot_cutoff_dashboard,
)

matplotlib.use("Agg")


def gaussian(x, amplitude, center, fwhm, offset=0.5, slope=0.0):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    baseline_x = (x - np.mean(x)) / np.ptp(x)
    return (
        offset
        + slope * baseline_x
        + amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)
    )


def synthetic_sweep() -> RawSweep:
    detuning_hz = np.linspace(-1e6, 1e6, 201)
    widths = [180e3, 260e3, 340e3]
    amplitudes = [0.25, -0.30, 0.22]
    centers = [-80e3, 20e3, 100e3]
    state = np.column_stack(
        [
            gaussian(
                detuning_hz,
                amplitude,
                center,
                width,
                slope=0.03,
            )
            for amplitude, center, width in zip(
                amplitudes, centers, widths, strict=True
            )
        ]
    )
    return RawSweep(
        run_dir=Path("/synthetic/raw/run"),
        parameters={"cutoff": 0.005},
        qubit="q1",
        detuning_hz=detuning_hz,
        amp_prefactor=np.array([0.2, 0.5, 0.8]),
        rabi_mhz=np.array([5.0, 12.5, 20.0]),
        state=state,
        t2_reference_name="t2_ramsey_ns",
        t2_us=25.0,
        t2_limit_hz=1e6 / (np.pi * 25.0),
    )


def test_opx1000_fork_recovers_peak_and_dip_widths():
    analysis = analyze_sweep(synthetic_sweep())

    np.testing.assert_allclose(
        analysis.fwhm_hz,
        [180e3, 260e3, 340e3],
        rtol=0.03,
    )
    assert analysis.valid.tolist() == [True, True, True]
    assert np.all(analysis.fit_amplitude > 0)
    assert np.all(analysis.resolution > 0)


def test_dashboard_and_default_selection_are_usable():
    analysis = analyze_sweep(synthetic_sweep())
    index = best_amplitude_index(analysis)
    figure, axes = plot_cutoff_dashboard(analysis, index)

    assert 0 <= index < 3
    assert len(axes.ravel()) == 4
    assert "OPX1000" in figure._suptitle.get_text()


def test_aggregate_quality_gate_rejects_unresolved_width():
    results = pd.DataFrame(
        {
            "gaussian_center_hz": [0.0, 0.0],
            "fwhm_hz": [100_000.0, 2_000.0],
            "fwhm_t2_units": [2.0, 0.04],
            "fit_abs_amplitude": [0.25, 0.25],
            "fit_r_squared": [0.8, 0.8],
        }
    )
    campaign = AggregateCampaign(
        directory=Path("/synthetic/aggregate"),
        manifest={
            "base_parameters": {
                "frequency_step_in_mhz": 0.005,
                "frequency_span_in_mhz": 5.0,
            }
        },
        results=results,
        cutoffs=(0.01,),
        figures={},
    )

    assert aggregate_quality_mask(campaign).tolist() == [True, False]
