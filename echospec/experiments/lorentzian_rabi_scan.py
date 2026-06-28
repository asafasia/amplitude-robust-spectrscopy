from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from tqdm.auto import tqdm

from echospec.experiments.core import BaseExperiment
from echospec.experiments.lorentzian_rabi_amplitude import (
    LorentzianRabiAmplitudeSweepExperiment,
    OptionsLorentzianRabiAmplitude,
    ResultsLorentzianRabiAmplitudeSweep,
)
from echospec.simulation.pulses import PulseType
from echospec.utils.parameters import Parameters
from echospec.utils.units import Units as u


def _cell_edges(values: NDArray[np.floating], log: bool = False) -> NDArray[np.floating]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("Grid values must be one-dimensional.")
    if len(values) == 0:
        raise ValueError("Grid values cannot be empty.")
    if len(values) == 1:
        value = values[0]
        if log:
            if value <= 0:
                raise ValueError("Log-scale grid values must be positive.")
            return np.array([value / np.sqrt(10.0), value * np.sqrt(10.0)])
        width = abs(value) * 0.1 if value != 0 else 0.5
        return np.array([value - width, value + width])

    if log:
        if np.any(values <= 0):
            raise ValueError("Log-scale grid values must be positive.")
        transformed = np.log10(values)
        edges = _cell_edges(transformed, log=False)
        return np.power(10.0, edges)

    diffs = np.diff(values)
    if np.any(diffs == 0):
        raise ValueError("Grid values must be unique.")
    mids = values[:-1] + diffs / 2
    first = values[0] - diffs[0] / 2
    last = values[-1] + diffs[-1] / 2
    return np.concatenate([[first], mids, [last]])


@dataclass(slots=True)
class _ScanPointResult:
    i_length: int
    i_cutoff: int
    sweep: ResultsLorentzianRabiAmplitudeSweep


def _run_scan_point(
    i_length: int,
    i_cutoff: int,
    pulse_length: float,
    cutoff: float,
    amplitudes: NDArray[np.floating],
    params: Parameters,
    options: OptionsLorentzianRabiLengthCutoffScan,
) -> _ScanPointResult:
    params = copy(params)
    params.pulse_type = PulseType.LORENTZIAN
    params.eco_pulse = False
    params.pulse_length = pulse_length
    params.cutoff = cutoff

    inner_options = copy(options)
    inner_options.plot = options.plot_inner_sweeps
    inner_options.save = False
    inner_options.show_progress = False

    sweep = LorentzianRabiAmplitudeSweepExperiment(
        pulse_length=pulse_length,
        cutoff=cutoff,
        amplitudes=amplitudes,
        params=params,
        options=inner_options,
    ).run()
    return _ScanPointResult(
        i_length=i_length,
        i_cutoff=i_cutoff,
        sweep=sweep,
    )


@dataclass(slots=True)
class ResultsLorentzianRabiLengthCutoffScan:
    pulse_lengths: NDArray[np.floating]
    cutoffs: NDArray[np.floating]
    amplitudes: NDArray[np.floating]
    rabi_scales: NDArray[np.floating]
    pi_pulse_amplitudes: NDArray[np.floating]
    area_scales: NDArray[np.floating]
    sweep_results: list[list[ResultsLorentzianRabiAmplitudeSweep]]

    def sweep_result(
        self,
        i_length: int,
        i_cutoff: int,
    ) -> ResultsLorentzianRabiAmplitudeSweep:
        return self.sweep_results[i_length][i_cutoff]

    def plot(self) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6, 4))

        cutoff_edges = _cell_edges(self.cutoffs, log=True)
        length_edges_us = _cell_edges(self.pulse_lengths / u.us)
        pi_amp_mhz = np.ma.masked_invalid(
            self.pi_pulse_amplitudes / u.pi2 / u.MHz
        )

        mesh = ax.pcolormesh(
            cutoff_edges,
            length_edges_us,
            pi_amp_mhz,
            shading="auto",
            cmap="viridis_r",
        )
        ax.set_xscale("log")
        ax.set_xlabel("Cutoff")
        ax.set_ylabel("Pulse length (us)")
        ax.set_title("Calculated pi-pulse Rabi amplitude")
        fig.colorbar(mesh, ax=ax, label="Amplitude (MHz)")

        fig.tight_layout()
        return fig


@dataclass(slots=True)
class OptionsLorentzianRabiLengthCutoffScan(OptionsLorentzianRabiAmplitude):
    """Options for scanning Lorentzian Rabi amplitude over length and cutoff."""

    plot: bool = True
    plot_inner_sweeps: bool = False
    max_workers: Optional[int] = None


class LorentzianRabiLengthCutoffScanExperiment(
    BaseExperiment[ResultsLorentzianRabiLengthCutoffScan]
):
    """
    Scan pulse lengths and cutoffs using the single-shape Rabi sweep.

    Each grid point runs ``LorentzianRabiAmplitudeSweepExperiment`` for the
    requested amplitudes, then stores the fitted pi-pulse amplitude and fitted
    Rabi scale.
    """

    def __init__(
        self,
        pulse_lengths: NDArray[np.float64],
        cutoffs: NDArray[np.float64],
        amplitudes: NDArray[np.float64],
        params: Parameters,
        options: Optional[OptionsLorentzianRabiLengthCutoffScan] = None,
    ) -> None:
        super().__init__(params, options)
        self.options = options or OptionsLorentzianRabiLengthCutoffScan()
        self.pulse_lengths = np.asarray(pulse_lengths, dtype=float)
        self.cutoffs = np.asarray(cutoffs, dtype=float)
        self.amplitudes = np.asarray(amplitudes, dtype=float)

    def run(self) -> ResultsLorentzianRabiLengthCutoffScan:
        rabi_scales = np.empty((len(self.pulse_lengths), len(self.cutoffs)))
        pi_pulse_amplitudes = np.empty_like(rabi_scales)
        area_scales = np.empty_like(rabi_scales)
        sweep_results: list[list[ResultsLorentzianRabiAmplitudeSweep | None]] = [
            [None for _ in self.cutoffs] for _ in self.pulse_lengths
        ]

        jobs = [
            (i_length, i_cutoff, pulse_length, cutoff)
            for i_length, pulse_length in enumerate(self.pulse_lengths)
            for i_cutoff, cutoff in enumerate(self.cutoffs)
        ]

        if self.options.max_workers == 1 or len(jobs) == 1:
            iterator = tqdm(
                jobs,
                desc="Length/cutoff Rabi scan",
                unit="point",
                colour="cyan",
                disable=not self.options.show_progress,
            )
            for i_length, i_cutoff, pulse_length, cutoff in iterator:
                point = self._run_sweep(i_length, i_cutoff, pulse_length, cutoff)
                self._store_scan_point(
                    point,
                    rabi_scales,
                    pi_pulse_amplitudes,
                    area_scales,
                    sweep_results,
                )
        else:
            with ProcessPoolExecutor(max_workers=self.options.max_workers) as executor:
                futures = [
                    executor.submit(
                        _run_scan_point,
                        i_length,
                        i_cutoff,
                        pulse_length,
                        cutoff,
                        self.amplitudes,
                        self.params,
                        self.options,
                    )
                    for i_length, i_cutoff, pulse_length, cutoff in jobs
                ]
                iterator = tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Length/cutoff Rabi scan",
                    unit="point",
                    colour="cyan",
                    disable=not self.options.show_progress,
                )
                for future in iterator:
                    point = future.result()
                    self._store_scan_point(
                        point,
                        rabi_scales,
                        pi_pulse_amplitudes,
                        area_scales,
                        sweep_results,
                    )

        self.results = ResultsLorentzianRabiLengthCutoffScan(
            pulse_lengths=self.pulse_lengths,
            cutoffs=self.cutoffs,
            amplitudes=self.amplitudes,
            rabi_scales=rabi_scales,
            pi_pulse_amplitudes=pi_pulse_amplitudes,
            area_scales=area_scales,
            sweep_results=[
                [sweep for sweep in row if sweep is not None]
                for row in sweep_results
            ],
        )

        if self.options.plot:
            self.plot()

        if self.options.save:
            self.save()

        return self.results

    def _run_sweep(
        self,
        i_length: int,
        i_cutoff: int,
        pulse_length: float,
        cutoff: float,
    ) -> _ScanPointResult:
        return _run_scan_point(
            i_length=i_length,
            i_cutoff=i_cutoff,
            pulse_length=pulse_length,
            cutoff=cutoff,
            amplitudes=self.amplitudes,
            params=self.params,
            options=self.options,
        )

    @staticmethod
    def _store_scan_point(
        point: _ScanPointResult,
        rabi_scales: NDArray[np.floating],
        pi_pulse_amplitudes: NDArray[np.floating],
        area_scales: NDArray[np.floating],
        sweep_results: list[list[ResultsLorentzianRabiAmplitudeSweep | None]],
    ) -> None:
        i_length = point.i_length
        i_cutoff = point.i_cutoff
        sweep = point.sweep
        rabi_scales[i_length, i_cutoff] = sweep.fit.rabi_scale
        pi_pulse_amplitudes[i_length, i_cutoff] = sweep.pi_pulse_amplitude
        area_scales[i_length, i_cutoff] = sweep.area_scale
        sweep_results[i_length][i_cutoff] = sweep

    def plot(self) -> None:
        self._check_results()
        self.current_figure = self.results.plot()

    def _get_experiment_name(self) -> str:
        return "lorentzian_rabi_length_cutoff_scan"

    def _save_results(self, save_dir: Path) -> None:
        self._check_results()
        np.savez(
            save_dir / "results.npz",
            pulse_lengths=self.results.pulse_lengths,
            cutoffs=self.results.cutoffs,
            amplitudes=self.results.amplitudes,
            rabi_scales=self.results.rabi_scales,
            pi_pulse_amplitudes=self.results.pi_pulse_amplitudes,
            area_scales=self.results.area_scales,
        )


if __name__ == "__main__":
    options = OptionsLorentzianRabiLengthCutoffScan(
        plot=True,
        save=False,
        num_time_points=1000,
        plot_inner_sweeps=False,
        max_workers=None,
    )
    params = Parameters(
        pulse_type=PulseType.LORENTZIAN,
        eco_pulse=False,
        order=0.5,
    )

    pulse_lengths = np.linspace(10, 100, 10) * u.us
    cutoffs = np.logspace(-5, -1, 10)
    amplitudes = np.linspace(0, 40.0, 400) * u.pi2 * u.MHz

    experiment = LorentzianRabiLengthCutoffScanExperiment(
        pulse_lengths=pulse_lengths,
        cutoffs=cutoffs,
        amplitudes=amplitudes,
        params=params,
        options=options,
    )
    experiment.run()
    plt.show()
