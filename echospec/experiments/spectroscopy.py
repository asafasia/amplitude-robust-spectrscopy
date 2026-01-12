from __future__ import annotations

from matplotlib import pyplot as plt
import numpy as np
from numpy.typing import NDArray
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
from echospec.results.results import ResultsSingleRun, ResultsSpectroscopy1D


def _run_single(detuning: float, params: Parameters, options: OptionsSpectroscopy) -> ResultsSingleRun:
    params.detuning = detuning
    return Solver(params, options).run()


@dataclass
class OptionsSpectroscopy(Options):
    pass


class Spectroscopy(BaseExperiment[ResultsSpectroscopy1D]):
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

    def run(self) -> ResultsSpectroscopy1D:
        self.results = self._execute()
        if self.options.plot:
            self.plot_final_z()

        if self.options.save:
            # Always plot before saving to ensure figure exists
            if not self.options.plot:
                self.plot_final_z()
            self.save()

        return self.results

    def _execute(self) -> ResultsSpectroscopy1D:
        vec = parallel_map(
            _run_single,
            self.detunings,
            task_args=(self.params, self.options),
        )
        results = ResultsSpectroscopy1D.from_single_runs(
            runs=vec,
            detunings=self.detunings,
            params=self.params,
        )

        if self.options.noise > 0:
            gaussian_noise = self.options.noise * np.random.normal(
                loc=0.0,
                scale=1.0,
                size=results.data.shape
            )
            results.data += gaussian_noise

        return results

    def plot(self) -> None:
        """Plot final Z expectation value vs detuning."""
        self.plot_final_z()

    def plot_final_z(self) -> None:
        if self.results is None:
            raise RuntimeError("No results to plot. Run the experiment first.")
        """Plot final Z expectation value vs detuning with FWHM markers."""
        self._check_results()

        self.current_figure = self.results.plot()

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

    def add_noise(self, data: NDArray[np.floating]) -> NDArray[np.floating]:
        """Add Gaussian noise to the data based on options."""
        if self.options.noise <= 0:
            return data
        gaussian_noise = self.options.noise * np.random.normal(
            loc=0.0,
            scale=1.0,
            size=data.shape
        )
        return self.results.data + gaussian_noise


if __name__ == "__main__":
    options = OptionsSpectroscopy(
        plot=True,
        with_fwhm=True,
        non_linear_sweep=True,
        plot_population=True,
        save=False,
        noise=0.02,
    )

    params = Parameters(
        eco_pulse=False,
        cutoff=0.0005,
        rabi_frequency=60 * np.pi * u.MHz,
        pulse_type=PulseType.LORENTZIAN,
        pulse_length=40 * u.us,
    )

    span = 0.2 * u.pi2 * u.MHz
    detunings = np.linspace(-span / 2, span / 2, 101)

    Spectroscopy(detunings, params, options).run()

    plt.show()
