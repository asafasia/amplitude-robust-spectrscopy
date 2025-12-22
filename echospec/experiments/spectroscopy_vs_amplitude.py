from __future__ import annotations

import numpy as np
import xarray as xr
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from echospec.utils.units import Units as u
from echospec.utils.parameters import Parameters
from echospec.experiments.core import BaseExperiment
from echospec.simulation.pulses import PulseType
from echospec.experiments.spectroscopy import Spectroscopy
from echospec.simulation.utils import z_to_populations
from echospec.simulation.run import Solver, Options
from echospec.plotting.spectroscopy import plot_amplitude_sweep_spectroscopy

from numpy.typing import NDArray


# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class OptionsSpectroscopy2d(Options):
    """Configuration flags for amplitude sweep spectroscopy."""
    plot: bool = True
    plot_fwhm: bool = True


# -----------------------------------------------------------------------------
# Main class
# -----------------------------------------------------------------------------

class AmplitudeSweepSpectroscopy(BaseExperiment[xr.DataArray]):
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

    # -------------------------------------------------------------------------

    def run(self) -> xr.DataArray:
        """
        Execute spectroscopy for each drive amplitude.

        Returns
        -------
        xr.DataArray | None
            Concatenated spectroscopy results with dimension `amplitude`, or None if empty.
        """
        spectra = []
        self.fwhm_values = []

        for amp in tqdm(
            self.amplitudes,
            desc="Amplitude sweep",
            unit="Ω",
            colour="cyan",
        ):
            spectra.append(self._run_single_amplitude(amp))

        self.results = xr.concat(spectra, dim="amplitude")
        self.fwhm_values = np.array(self.fwhm_values)

        if self.options.plot:
            self.plot_final_z()

        if self.options.save:
            # Always plot before saving to ensure figure exists
            if not self.options.plot:
                self.plot_final_z()
            self.save()

        return self.results

    # -------------------------------------------------------------------------

    def _run_single_amplitude(self, amplitude: float) -> xr.DataArray:
        """
        Run spectroscopy for a single drive amplitude.
        """
        # Mutate explicitly and locally
        self.params.rabi_frequency = amplitude

        spec = Spectroscopy(
            detunings=self.detunings,
            params=self.params,
            options=Options(plot=False, with_fwhm=True),
        )

        result = spec.run()

        # Store FWHM from this spectroscopy run
        self.fwhm_values.append(result.fwhm)

        return result.data.expand_dims(amplitude=[amplitude])

    # -------------------------------------------------------------------------

    def plot(self) -> None:
        """Plot 2D spectroscopy results."""
        self.plot_final_z()

    def plot_final_z(self) -> None:
        """
        Plot final-time Z population as a function of detuning and amplitude.
        """
        self._check_results()

        self.current_figure = plot_amplitude_sweep_spectroscopy(
            results=self.results,
            params=self.params,
            fwhm_values=self.fwhm_values,
            amplitudes=self.amplitudes,
            detunings=self.detunings,
            plot_fwhm=self.options.plot_fwhm,
        )

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
    options.plot = True
    options.save = True

    params = Parameters(
        eco_pulse=True,
        pulse_type=PulseType.LORENTZIAN,
        pulse_length=50 * u.us,
        cutoff=1e-4,
    )

    detunings = np.linspace(
        -0.1 * u.pi2 * u.MHz,
        +0.1 * u.pi2 * u.MHz,
        151,
    )

    amplitudes = np.linspace(
        0,
        50 * u.pi2 * u.MHz,
        20,
    )

    sweep = AmplitudeSweepSpectroscopy(
        amplitudes=amplitudes,
        detunings=detunings,
        params=params,
        options=options,
    )

    data = sweep.run()
