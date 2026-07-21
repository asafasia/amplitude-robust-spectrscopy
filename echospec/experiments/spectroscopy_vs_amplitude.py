from __future__ import annotations
from copy import copy

from matplotlib import pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from tqdm.auto import tqdm

from echospec.utils.units import Units as u
from echospec.utils.parameters import Parameters
from echospec.experiments.core import BaseExperiment
from echospec.simulation.pulses import PulseType
from echospec.experiments.spectroscopy import Spectroscopy
from echospec.simulation.run import Options
# from echospec.plotting.spectroscopy import plot_spectroscopy_2d
from echospec.results.results import ResultsSingleRun, ResultsSpectroscopy1D, ResultsSpectroscopy2D
from numpy.typing import NDArray


@dataclass(slots=True)
class OptionsSpectroscopy2d(Options):
    """Configuration flags for amplitude sweep spectroscopy."""
    plot: bool = True
    plot_fwhm: bool = True
    add_noise: bool = False


class AmplitudeSweepSpectroscopy(BaseExperiment[ResultsSpectroscopy2D]):
    """
    Perform spectroscopy over a sweep of drive amplitudes.

    2D spectroscopy experiment scanning both detuning and amplitude.

    Output shape:
        (amplitude, detuning, observable, time)
    """

    def __init__(
        self,
        amplitudes: NDArray[np.float64],
        detunings: NDArray[np.float64],
        params: Parameters,
        options: Optional[OptionsSpectroscopy2d] = None,
    ) -> None:
        super().__init__(params, options)
        self.options = options or OptionsSpectroscopy2d()

        self.amplitudes = np.asarray(amplitudes, dtype=float)
        self.detunings = np.asarray(detunings, dtype=float)
        self.fwhm_values: list[Optional[float]] = []
        self.snr_values: list[Optional[float]] = []

    # -------------------------------------------------------------------------

    def run(self) -> ResultsSpectroscopy2D:
        """
        Execute spectroscopy for each drive amplitude.

        Returns
        -------
        xr.DataArray | None
            Concatenated spectroscopy results with dimension `amplitude`, or None if empty.
        """
        raw_results = []
        self.fwhm_values = []
        self.snr_values = []

        for amp in tqdm(
            self.amplitudes,
            desc="Amplitude sweep",
            unit="Ω",
            colour="cyan",
        ):
            raw_results.append(self._run_single_amplitude(amp))

        self.results: ResultsSpectroscopy2D | None = ResultsSpectroscopy2D.from_spectroscopy_1d(
            spectroscopies=raw_results,
            amplitudes=self.amplitudes,
        )

        if self.options.plot:
            self.plot()

        if self.options.save:
            self.save()

        return self.results

    # -------------------------------------------------------------------------

    def _run_single_amplitude(self, amplitude: float) -> ResultsSpectroscopy1D:
        """
        Run spectroscopy for a single drive amplitude.
        """
        # Mutate explicitly and locally
        self.params.rabi_frequency = amplitude

        opts = copy(self.options)
        opts.plot = False

        spec = Spectroscopy(
            detunings=self.detunings,
            params=self.params,
            options=opts,
        )

        results = spec.run()

        return results
    # -------------------------------------------------------------------------

    def plot(self) -> None:
        """Plot 2D spectroscopy results."""
        fig = self.results.plot()

    # -------------------------------------------------------------------------
    # Save methods
    # -------------------------------------------------------------------------

    def _get_experiment_name(self) -> str:
        """Get experiment name for saving."""
        return "amplitude_sweep_spectroscopy"

    def _save_results(self, save_dir: Path) -> None:
        """Save amplitude sweep spectroscopy results."""
        # Save xarray DataArray as NetCDF
        self.results.to_netcdf(save_dir / "results.nc")

        # Save FWHM values
        if len(self.fwhm_values) > 0:
            fwhm_data = {
                "amplitudes": self.amplitudes.tolist(),
                "amplitudes_mhz": (self.amplitudes / u.pi2 / u.MHz).tolist(),
                "fwhm_values": [float(f) if f is not None else None for f in self.fwhm_values],
                "fwhm_values_mhz": [
                    float(f / u.pi2 / u.MHz) if f is not None else None
                    for f in self.fwhm_values
                ],
            }
            import json
            with open(save_dir / "fwhm_vs_amplitude.json", "w") as f:
                json.dump(fwhm_data, f, indent=2)


# -----------------------------------------------------------------------------
# Script entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    options = OptionsSpectroscopy2d()
    options.plot = False
    options.save = False
    options.noise = 0.0
    options.with_fwhm = False

    params = Parameters(
        eco_pulse=True,
        pulse_type=PulseType.LORENTZIAN,
        pulse_length=20 * u.us,
        cutoff=0.005
    )

    detunings = np.linspace(
        -0.5 * u.pi2 * u.MHz,
        +0.5* u.pi2 * u.MHz,
        151,
    )

    amplitudes = np.linspace(
        0,
        15 * u.pi2 * u.MHz,
        200,
    )

    sweep = AmplitudeSweepSpectroscopy(
        amplitudes=amplitudes,
        detunings=detunings,
        params=params,
        options=options,
    )

    data = sweep.run()

    np.savez(
        "amplitude_sweep_spectroscopy_results.npz",
        detuning_convention="drive_minus_qubit",
        amplitudes=amplitudes,
        detunings=detunings,
        data=data,
    )
