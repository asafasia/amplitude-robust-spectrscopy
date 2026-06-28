from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from qutip import basis, mesolve
from scipy.optimize import curve_fit
from tqdm.auto import tqdm

from echospec.experiments.core import BaseExperiment
from echospec.results.results import ResultsSingleRun
from echospec.simulation.hamiltonian import Hamiltonian
from echospec.simulation.operators import N_dim, sx, sy, sz
from echospec.simulation.pulses import (
    PulseArgs,
    PulseType,
    lorentzian_envelope,
)
from echospec.simulation.run import Options
from echospec.utils.parameters import Parameters
from echospec.utils.units import Units as u


def _cosine_model(
    amplitude: NDArray[np.floating],
    contrast: float,
    rabi_scale: float,
    phase: float,
    offset: float,
) -> NDArray[np.floating]:
    return contrast * np.cos(rabi_scale * amplitude + phase) + offset


def lorentzian_area_scale(
    pulse_length: float,
    cutoff: float,
    order: float = 0.5,
    zeroed_pulse: bool = False,
    num_time_points: int = 4001,
) -> float:
    """Numerically integrate the Lorentzian envelope over one pulse."""
    tlist = np.linspace(-pulse_length / 2, pulse_length / 2, num_time_points)
    args = PulseArgs(
        pulse_length=pulse_length,
        cutoff=cutoff,
        order=order,
        zeroed_pulse=zeroed_pulse,
    )
    return float(np.trapezoid(lorentzian_envelope(tlist, args), tlist))


@dataclass(slots=True)
class CosineFit:
    """Cosine fit for final Z versus drive amplitude."""

    contrast: float
    rabi_scale: float
    phase: float
    offset: float
    covariance: NDArray[np.floating]

    @property
    def pi_pulse_amplitude(self) -> float:
        if self.rabi_scale == 0:
            return np.inf
        return float(np.pi / abs(self.rabi_scale))

    def evaluate(self, amplitudes: NDArray[np.floating]) -> NDArray[np.floating]:
        return _cosine_model(
            amplitudes,
            self.contrast,
            self.rabi_scale,
            self.phase,
            self.offset,
        )


def fit_final_z_to_cosine(
    amplitudes: NDArray[np.floating],
    final_z: NDArray[np.floating],
    rabi_scale_guess: Optional[float] = None,
) -> CosineFit:
    """
    Fit final Z to contrast * cos(rabi_scale * amplitude + phase) + offset.

    The fitted ``rabi_scale`` has units of seconds when amplitudes are angular
    frequencies, and gives the effective pulse area per unit drive amplitude.
    """
    amplitudes = np.asarray(amplitudes, dtype=float)
    final_z = np.asarray(final_z, dtype=float)

    if amplitudes.ndim != 1 or final_z.ndim != 1:
        raise ValueError("amplitudes and final_z must be one-dimensional.")
    if len(amplitudes) != len(final_z):
        raise ValueError("amplitudes and final_z must have the same length.")
    if len(amplitudes) < 4:
        raise ValueError("At least four amplitude points are required for a cosine fit.")

    if rabi_scale_guess is None:
        span = np.ptp(amplitudes)
        rabi_scale_guess = 2 * np.pi / span if span > 0 else 1.0

    contrast_guess = 0.5 * (np.nanmax(final_z) - np.nanmin(final_z))
    if contrast_guess == 0:
        contrast_guess = 1.0
    offset_guess = float(np.nanmean(final_z))
    p0 = [contrast_guess, abs(rabi_scale_guess), 0.0, offset_guess]

    popt, pcov = curve_fit(
        _cosine_model,
        amplitudes,
        final_z,
        p0=p0,
        bounds=([-2.0, 0.0, -2 * np.pi, -2.0], [2.0, np.inf, 2 * np.pi, 2.0]),
        maxfev=20_000,
    )

    return CosineFit(
        contrast=float(popt[0]),
        rabi_scale=float(popt[1]),
        phase=float(popt[2]),
        offset=float(popt[3]),
        covariance=pcov,
    )


@dataclass(slots=True)
class ResultsLorentzianRabiAmplitude:
    pulse_lengths: NDArray[np.floating]
    cutoffs: NDArray[np.floating]
    amplitudes: NDArray[np.floating]
    final_z: NDArray[np.floating]
    rabi_scales: NDArray[np.floating]
    pi_pulse_amplitudes: NDArray[np.floating]
    area_scales: NDArray[np.floating]
    fits: list[list[CosineFit]]

    @property
    def populations(self) -> NDArray[np.floating]:
        return (1 - self.final_z) / 2

    def fit_curve(
        self,
        i_length: int,
        i_cutoff: int,
    ) -> NDArray[np.floating]:
        return self.fits[i_length][i_cutoff].evaluate(self.amplitudes)

    def plot(self) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6, 4))
        mesh = ax.pcolormesh(
            self.cutoffs,
            self.pulse_lengths / u.us,
            self.pi_pulse_amplitudes / u.pi2 / u.MHz,
            shading="auto",
        )
        ax.set_xscale("log")
        ax.set_xlabel("Cutoff")
        ax.set_ylabel("Pulse length (us)")
        ax.set_title("Lorentzian pi-pulse amplitude")
        fig.colorbar(mesh, ax=ax, label="Amplitude (MHz)")
        fig.tight_layout()
        return fig


@dataclass(slots=True)
class ResultsLorentzianRabiAmplitudeSweep:
    pulse_length: float
    cutoff: float
    amplitudes: NDArray[np.floating]
    runs: list[ResultsSingleRun]
    fit: CosineFit
    area_scale: float

    @property
    def time(self) -> NDArray[np.floating]:
        return self.runs[0].time

    @property
    def data(self) -> NDArray[np.floating]:
        return np.stack([run.data for run in self.runs], axis=0)

    @property
    def final_state(self) -> NDArray[np.floating]:
        return self.data[:, :, -1]

    @property
    def final_x(self) -> NDArray[np.floating]:
        return self.final_state[:, 0]

    @property
    def final_y(self) -> NDArray[np.floating]:
        return self.final_state[:, 1]

    @property
    def final_z(self) -> NDArray[np.floating]:
        return self.final_state[:, 2]

    @property
    def populations(self) -> NDArray[np.floating]:
        return (1 - self.final_z) / 2

    @property
    def pi_pulse_amplitude(self) -> float:
        return self.fit.pi_pulse_amplitude

    def fit_curve(self) -> NDArray[np.floating]:
        return self.fit.evaluate(self.amplitudes)

    def plot(self) -> plt.Figure:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

        amplitudes_mhz = self.amplitudes / u.pi2 / u.MHz
        axs[0].plot(amplitudes_mhz, self.final_z, "o", label="final Z")
        axs[0].plot(
            amplitudes_mhz,
            self.fit_curve(),
            "-",
            label="cosine fit",
        )
        axs[0].set_xlabel("Lorentzian amplitude (MHz)")
        axs[0].set_ylabel("Final Z")
        axs[0].set_title(
            f"L={self.pulse_length / u.us:.3g} us, cutoff={self.cutoff:.1e}"
        )
        axs[0].legend()

        for amplitude, run in zip(self.amplitudes, self.runs):
            axs[1].plot(
                run.time / u.us,
                run.z(),
                alpha=0.7,
                label=f"{amplitude / u.pi2 / u.MHz:.2g} MHz",
            )
        axs[1].set_xlabel("Time (us)")
        axs[1].set_ylabel("Z")
        axs[1].set_title("State traces by amplitude")
        axs[1].legend(fontsize=7, ncols=2)

        fig.tight_layout()
        return fig


@dataclass(slots=True)
class OptionsLorentzianRabiAmplitude(Options):
    """Options for decay-free Lorentzian Rabi-amplitude calibration."""

    plot: bool = True
    show_progress: bool = True


class LorentzianRabiAmplitudeSweepExperiment(
    BaseExperiment[ResultsLorentzianRabiAmplitudeSweep]
):
    """
    Sweep amplitudes for one Lorentzian pulse length and cutoff.

    This is the calibration-level experiment: pass one ``pulse_length`` and one
    ``cutoff``, inspect the state data versus amplitude, then loop externally if
    you want to scan more pulse shapes.
    """

    def __init__(
        self,
        pulse_length: float,
        cutoff: float,
        amplitudes: NDArray[np.float64],
        params: Parameters,
        options: Optional[OptionsLorentzianRabiAmplitude] = None,
    ) -> None:
        super().__init__(params, options)
        self.options = options or OptionsLorentzianRabiAmplitude()
        self.pulse_length = float(pulse_length)
        self.cutoff = float(cutoff)
        self.amplitudes = np.asarray(amplitudes, dtype=float)

    def run(self) -> ResultsLorentzianRabiAmplitudeSweep:
        runs = []
        for amplitude in tqdm(
            self.amplitudes,
            desc="Lorentzian amplitude sweep",
            unit="amp",
            colour="cyan",
            disable=not self.options.show_progress,
        ):
            params = copy(self.params)
            params.detuning = 0.0
            params.pulse_type = PulseType.LORENTZIAN
            params.pulse_length = self.pulse_length
            params.cutoff = self.cutoff
            params.rabi_frequency = amplitude
            runs.append(self._run_decay_free(params))

        final_z = np.asarray([run.final_z for run in runs], dtype=float)
        area_scale = lorentzian_area_scale(
            pulse_length=self.pulse_length,
            cutoff=self.cutoff,
            order=self.params.order,
            zeroed_pulse=self.params.zeroed_pulse,
            num_time_points=self.options.num_time_points,
        )
        fit = fit_final_z_to_cosine(
            self.amplitudes,
            final_z,
            rabi_scale_guess=area_scale,
        )
        self.results = ResultsLorentzianRabiAmplitudeSweep(
            pulse_length=self.pulse_length,
            cutoff=self.cutoff,
            amplitudes=self.amplitudes,
            runs=runs,
            fit=fit,
            area_scale=area_scale,
        )

        if self.options.plot:
            self.plot()

        if self.options.save:
            self.save()

        return self.results

    def _run_decay_free(self, params: Parameters) -> ResultsSingleRun:
        tlist = np.linspace(
            -params.pulse_length / 2,
            params.pulse_length / 2,
            self.options.num_time_points,
        )
        result = mesolve(
            Hamiltonian(params=params).get_hamiltonian(),
            basis(N_dim, 0),
            tlist,
            c_ops=[],
            e_ops=[sx, sy, sz],
        )
        return ResultsSingleRun(
            data=np.asarray(result.expect, dtype=float),
            time=np.asarray(result.times, dtype=float),
        )

    def plot(self) -> None:
        self._check_results()
        self.current_figure = self.results.plot()

    def _get_experiment_name(self) -> str:
        return "lorentzian_rabi_amplitude_sweep"

    def _save_results(self, save_dir: Path) -> None:
        self._check_results()
        np.savez(
            save_dir / "results.npz",
            pulse_length=self.results.pulse_length,
            cutoff=self.results.cutoff,
            amplitudes=self.results.amplitudes,
            data=self.results.data,
            final_state=self.results.final_state,
            rabi_scale=self.results.fit.rabi_scale,
            pi_pulse_amplitude=self.results.pi_pulse_amplitude,
            area_scale=self.results.area_scale,
        )


class LorentzianRabiAmplitudeExperiment(
    BaseExperiment[ResultsLorentzianRabiAmplitude]
):
    """
    Fit Lorentzian Rabi oscillations versus drive amplitude.

    For each pulse length and cutoff, this experiment sweeps ``rabi_frequency``
    at zero detuning, evolves with no collapse operators (``c_ops=[]``), and
    fits the final Z expectation value to a cosine.
    """

    def __init__(
        self,
        pulse_lengths: NDArray[np.float64],
        cutoffs: NDArray[np.float64],
        amplitudes: NDArray[np.float64],
        params: Parameters,
        options: Optional[OptionsLorentzianRabiAmplitude] = None,
    ) -> None:
        super().__init__(params, options)
        self.options = options or OptionsLorentzianRabiAmplitude()
        self.pulse_lengths = np.asarray(pulse_lengths, dtype=float)
        self.cutoffs = np.asarray(cutoffs, dtype=float)
        self.amplitudes = np.asarray(amplitudes, dtype=float)

    def run(self) -> ResultsLorentzianRabiAmplitude:
        final_z = np.empty(
            (len(self.pulse_lengths), len(self.cutoffs), len(self.amplitudes)),
            dtype=float,
        )
        rabi_scales = np.empty((len(self.pulse_lengths), len(self.cutoffs)))
        pi_pulse_amplitudes = np.empty_like(rabi_scales)
        area_scales = np.empty_like(rabi_scales)
        fits: list[list[CosineFit]] = []

        iterator = tqdm(
            enumerate(self.pulse_lengths),
            total=len(self.pulse_lengths),
            desc="Lorentzian Rabi amplitude",
            unit="length",
            colour="cyan",
        )
        for i_length, pulse_length in iterator:
            row_fits: list[CosineFit] = []
            for i_cutoff, cutoff in enumerate(self.cutoffs):
                z_values = self._run_single_pulse_shape(pulse_length, cutoff)
                final_z[i_length, i_cutoff, :] = z_values

                area_scale = lorentzian_area_scale(
                    pulse_length=pulse_length,
                    cutoff=cutoff,
                    order=self.params.order,
                    zeroed_pulse=self.params.zeroed_pulse,
                    num_time_points=self.options.num_time_points,
                )
                fit = fit_final_z_to_cosine(
                    self.amplitudes,
                    z_values,
                    rabi_scale_guess=area_scale,
                )
                area_scales[i_length, i_cutoff] = area_scale
                rabi_scales[i_length, i_cutoff] = fit.rabi_scale
                pi_pulse_amplitudes[i_length, i_cutoff] = fit.pi_pulse_amplitude
                row_fits.append(fit)
            fits.append(row_fits)

        self.results = ResultsLorentzianRabiAmplitude(
            pulse_lengths=self.pulse_lengths,
            cutoffs=self.cutoffs,
            amplitudes=self.amplitudes,
            final_z=final_z,
            rabi_scales=rabi_scales,
            pi_pulse_amplitudes=pi_pulse_amplitudes,
            area_scales=area_scales,
            fits=fits,
        )

        if self.options.plot:
            self.plot()

        if self.options.save:
            self.save()

        return self.results

    def _run_single_pulse_shape(
        self,
        pulse_length: float,
        cutoff: float,
    ) -> NDArray[np.floating]:
        z_values = []
        for amplitude in self.amplitudes:
            params = copy(self.params)
            params.detuning = 0.0
            params.pulse_type = PulseType.LORENTZIAN
            params.pulse_length = pulse_length
            params.cutoff = cutoff
            params.rabi_frequency = amplitude

            run = self._run_decay_free(params)
            z_values.append(run.final_z)
        return np.asarray(z_values, dtype=float)

    def _run_decay_free(self, params: Parameters) -> ResultsSingleRun:
        tlist = np.linspace(
            -params.pulse_length / 2,
            params.pulse_length / 2,
            self.options.num_time_points,
        )
        result = mesolve(
            Hamiltonian(params=params).get_hamiltonian(),
            basis(N_dim, 0),
            tlist,
            c_ops=[],
            e_ops=[sx, sy, sz],
        )
        return ResultsSingleRun(
            data=np.asarray(result.expect, dtype=float),
            time=np.asarray(result.times, dtype=float),
        )

    def plot(self) -> None:
        self._check_results()
        self.current_figure = self.results.plot()

    def _get_experiment_name(self) -> str:
        return "lorentzian_rabi_amplitude"

    def _save_results(self, save_dir: Path) -> None:
        self._check_results()
        np.savez(
            save_dir / "results.npz",
            pulse_lengths=self.results.pulse_lengths,
            cutoffs=self.results.cutoffs,
            amplitudes=self.results.amplitudes,
            final_z=self.results.final_z,
            rabi_scales=self.results.rabi_scales,
            pi_pulse_amplitudes=self.results.pi_pulse_amplitudes,
            area_scales=self.results.area_scales,
        )


if __name__ == "__main__":
    options = OptionsLorentzianRabiAmplitude(
        plot=True,
        save=False,
        num_time_points=1000,
    )

    params = Parameters(
        eco_pulse=False,
        pulse_type=PulseType.LORENTZIAN,
        order=0.5,
    )

    pulse_length = 50 * u.us
    cutoff = 1e-3
    amplitudes = np.linspace(0, 40.0, 500) * u.pi2 * u.MHz

    experiment = LorentzianRabiAmplitudeSweepExperiment(
        pulse_length=pulse_length,
        cutoff=cutoff,
        amplitudes=amplitudes,
        params=params,
        options=options,
    )
    experiment.run()
    plt.show()
