from __future__ import annotations
from copy import copy

from matplotlib import pyplot as plt
import numpy as np
from dataclasses import dataclass, asdict
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
# %%
if __name__ == "__main__":

    echo = False

    options = OptionsSpectroscopy2d()
    options.plot = True
    options.save = False
    options.noise = 00
    options.with_fwhm = True

    params = Parameters(
        eco_pulse=echo,
        pulse_type=PulseType.LORENTZIAN,
        pulse_length=50 * u.us,
        cutoff=1e-4,
    )

    detunings = np.linspace(
        -0.1 * u.pi2 * u.MHz,
        +0.1 * u.pi2 * u.MHz,
        151,
    )

    amplitudes = np.logspace(
        -3,
        2,
        200,
    ) * 2 * np.pi * u.MHz

    cutoff_vector = np.logspace(
        -2,
        -5,
        200,
    )
    fwhm_matrix = []
    snr_matrix = []

    for i, cutoff in enumerate(cutoff_vector):
        print(f"Cutoff {i+1}/{len(cutoff_vector)}: {cutoff:.1e}")

        params.cutoff = cutoff
        sweep = AmplitudeSweepSpectroscopy(
            amplitudes=amplitudes,
            detunings=detunings,
            params=params,
            options=options,
        )
        data = sweep.run()

        fwhm_vector = data.fwhm_map
        snr_vector = data.snr_map
        fwhm_matrix.append(fwhm_vector)
        snr_matrix.append(snr_vector)
        plt.plot(amplitudes / u.pi2 / u.MHz, fwhm_vector / u.pi2 / u.MHz)

    fwhm_matrix = np.array(fwhm_matrix)
    snr_matrix = np.array(snr_matrix)

    np.savez(
        "spectroscopy_data.npz",
        detunings=detunings,
        amplitudes=amplitudes,
        cutoff_vector=cutoff_vector,
        fwhm_matrix=fwhm_matrix,
        snr_matrix=snr_matrix,
        options=asdict(options),
        params=asdict(params),
    )
    # plt.xscale("log")
    # plt.show()

    # %%

#     # %%

#     fig, axs=plt.subplots(1, 2, figsize=(8, 3.5))
#     mat1=np.array(fwhm_matrix).T / params.T2_limit/2/np.pi

#     mat2=mat1/snr_matrix.T
#     c1=axs[0].pcolormesh(
#         cutoff_vector,
#         amplitudes / u.pi2 / u.MHz,
#         np.array(fwhm_matrix).T / params.T2_limit/2/np.pi,
#         shading="auto",
#         vmin=0,
#         vmax=10,
#         cmap="viridis_r",
#     )
#     axs[0].set_xscale("log")
#     axs[0].set_yscale("log")
#     axs[0].set_xlabel("Cutoff")
#     axs[0].set_ylabel("Drive Amplitude (MHz)")
#     axs[0].set_title("FWHM")
#     fig.colorbar(c1, ax=axs[0], label="FWHM / T2 limit")

#     c2=axs[1].pcolormesh(
#         cutoff_vector,
#         amplitudes / u.pi2 / u.MHz,
#         mat2,
#         shading="auto",
#         vmin=0,
#         vmax=20,
#         cmap="viridis_r",
#     )
#     axs[1].set_xscale("log")
#     axs[1].set_yscale("log")
#     axs[1].set_xlabel("Cutoff")
#     axs[1].set_ylabel("Drive Amplitude (MHz)")
#     axs[1].set_title("FWHM / Signal")
#     fig.colorbar(c2, ax=axs[1], label="FWHM / T2 limit / Signal")

#     fig.suptitle(
#         "Spectroscopy FWHM and SNR vs Amplitude and Cutoff of Lorentzian Pulse")
#     plt.tight_layout()
#     plt.savefig("spectroscopy_fwhm_snr_vs_amplitude_vs_cutoff.png", dpi=300)
#     plt.show()

# # # %%
