from __future__ import annotations

import numpy as np
import xarray as xr
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from qutip import parallel_map

from echospec.experiments.core import BaseExperiment
from echospec.simulation.run import Solver, Options
from echospec.simulation.utils import _non_linear_sweep
from echospec.simulation.pulses import PulseType
from echospec.utils.parameters import Parameters
from echospec.utils.units import Units as u
from echospec.analysis.fwhm import fwhm_gaussian_fit
from echospec.plotting.spectroscopy import plot_spectroscopy


# -----------------------------------------------------------------------------
# Parallel worker
# -----------------------------------------------------------------------------
def _run_single(detuning: float, params: Parameters) -> Optional[np.ndarray]:
    params.detuning = detuning
    return Solver(params).run()


# -----------------------------------------------------------------------------
# Options / results containers
# -----------------------------------------------------------------------------
@dataclass
class OptionsSpectroscopy(Options):
    pass


@dataclass
class ResultsSpectroscopy:
    data: xr.DataArray
    rabi_frequency: float
    fwhm: Optional[float] = None

    def final_z(self) -> xr.DataArray:
        return self.data.sel(observable="z").isel(time=-1)


# -----------------------------------------------------------------------------
# Main class
# -----------------------------------------------------------------------------
class Spectroscopy(BaseExperiment[ResultsSpectroscopy]):
    """
    1D spectroscopy experiment scanning detuning.

    Performs spectroscopy over a range of detunings and optionally
    computes FWHM and plots results.
    """

    def __init__(
        self,
        detunings: np.ndarray,
        params: Parameters,
        options: Optional[OptionsSpectroscopy] = None,
    ) -> None:
        super().__init__(params, options)
        self.options = options or OptionsSpectroscopy()

        self.detunings = _non_linear_sweep(
            detunings,
            self.options.non_linear_sweep,
        )

        self.times = np.linspace(
            -params.pulse_length / 2,
            params.pulse_length / 2,
            1000,
        )

        self.fwhm: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> ResultsSpectroscopy:
        raw = self._execute()
        data = self._to_xarray(raw)

        self.results = ResultsSpectroscopy(
            data=data,
            rabi_frequency=self.params.rabi_frequency,
        )

        if self.options.with_fwhm:
            self._compute_fwhm()

        if self.options.plot:
            self.plot_final_z()

        if self.options.save:
            # Always plot before saving to ensure figure exists
            if not self.options.plot:
                self.plot_final_z()
            self.save()

        return self.results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _execute(self) -> np.ndarray:
        vec = parallel_map(
            _run_single,
            self.detunings,
            task_args=(self.params,),
        )

        results = np.asarray(vec, dtype=float)

        expected_shape = (len(self.detunings), 3, len(self.times))
        if results.shape != expected_shape:
            raise ValueError(
                f"Unexpected results shape {results.shape}, expected {expected_shape}"
            )

        return results

    def _compute_fwhm(self) -> None:
        if self.results is None:
            raise RuntimeError("run() must be called before computing FWHM.")

        z_final = self.results.final_z()

        _, fwhm, snr, _ = fwhm_gaussian_fit(
            x=z_final.detuning.values,
            y=z_final.values,
        )

        self.fwhm = fwhm
        self.results.fwhm = fwhm

    def _to_xarray(self, results: np.ndarray) -> xr.DataArray:
        return xr.DataArray(
            results,
            dims=("detuning", "observable", "time"),
            coords={
                "detuning": self.detunings,
                "observable": ["x", "y", "z"],
                "time": self.times,
            },
            name="expectation_values",
            attrs={
                "pulse_type": str(self.params.pulse_type),
                "eco_pulse": self.params.eco_pulse,
                "pulse_length": self.params.pulse_length,
            },
        )

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot(self) -> None:
        """Plot final Z expectation value vs detuning."""
        self.plot_final_z()

    def plot_final_z(self) -> None:
        """Plot final Z expectation value vs detuning with FWHM markers."""
        self._check_results()

        self.current_figure = plot_spectroscopy(
            z_final=self.results.final_z(),
            params=self.params,
            rabi_frequency=self.params.rabi_frequency,
            fwhm=self.fwhm,
            plot_population=self.options.plot_population,
            with_fwhm=self.options.with_fwhm,
        )

    # ------------------------------------------------------------------
    # Save methods
    # ------------------------------------------------------------------
    def _get_experiment_name(self) -> str:
        """Get experiment name for saving."""
        return "spectroscopy"

    def _save_results(self, save_dir: Path) -> None:
        """Save spectroscopy results."""
        # Save xarray DataArray as NetCDF
        self.results.data.to_netcdf(save_dir / "results.nc")

        # Save FWHM value separately
        if self.fwhm is not None:
            fwhm_data = {
                "fwhm": float(self.fwhm),
                "fwhm_mhz": float(self.fwhm / u.pi2 / u.MHz),
            }
            import json
            with open(save_dir / "fwhm.json", "w") as f:
                json.dump(fwhm_data, f, indent=2)


# -----------------------------------------------------------------------------
# Script entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    options = OptionsSpectroscopy(
        plot=True,
        with_fwhm=True,
        non_linear_sweep=True,
        plot_population=True,
        save=True
    )

    params = Parameters(
        eco_pulse=True,
        cutoff=0.001,
        rabi_frequency=60 * np.pi * u.MHz,
        pulse_type=PulseType.LORENTZIAN,
        pulse_length=40 * u.us,
    )

    span = 0.2 * u.pi2 * u.MHz
    detunings = np.linspace(-span / 2, span / 2, 101)

    Spectroscopy(detunings, params, options).run()
