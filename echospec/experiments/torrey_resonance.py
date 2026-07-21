from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from echospec.experiments.core import BaseExperiment
from echospec.figures.paths import FIGURES_DIR
from echospec.simulation.run import Options
from echospec.utils.parameters import Parameters
from echospec.utils.units import Units as u


FIGS_DIR = FIGURES_DIR / "figs"


def torrey_sigma_z(
    detunings: NDArray[np.floating],
    rabi_frequency: float,
    T1: float,
    T2: float,
) -> NDArray[np.floating]:
    """Steady-state Torrey resonance from the analytic expression."""
    detunings = np.asarray(detunings, dtype=float)
    detuning_term = (T2 * detunings) ** 2
    return (1.0 + detuning_term) / (
        1.0 + detuning_term + rabi_frequency**2 * T1 * T2
    )


def torrey_fwhm(rabi_frequency: float, T1: float, T2: float) -> float:
    """Analytic FWHM in angular-frequency units."""
    if T1 <= 0 or T2 <= 0:
        raise ValueError("T1 and T2 must be positive.")
    return float(2.0 * np.sqrt(1.0 + rabi_frequency**2 * T1 * T2) / T2)


def torrey_snr(rabi_frequency: float, T1: float, T2: float) -> float:
    """Peak population signal of the analytic resonance."""
    if T1 <= 0 or T2 <= 0:
        raise ValueError("T1 and T2 must be positive.")
    saturation = rabi_frequency**2 * T1 * T2
    return float(saturation / (2.0 * (1.0 + saturation)))


def _threshold_crossing_x(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    threshold: float,
) -> Optional[float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size:
        raise ValueError("x and y must have the same length.")

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    delta = y - threshold

    exact = np.flatnonzero(np.isclose(delta, 0.0))
    if exact.size:
        return float(x[exact[0]])

    crossings = np.flatnonzero(delta[:-1] * delta[1:] < 0)
    if crossings.size == 0:
        return None

    i = int(crossings[0])
    y0 = y[i]
    y1 = y[i + 1]
    x0 = x[i]
    x1 = x[i + 1]
    return float(x0 + (threshold - y0) * (x1 - x0) / (y1 - y0))


def _threshold_intervals_x(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    threshold: float,
) -> list[tuple[float, float]]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size:
        raise ValueError("x and y must have the same length.")
    if x.size < 2:
        return []

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    above = y >= threshold

    intervals = []
    start: Optional[float] = float(x[0]) if above[0] else None

    for i in range(x.size - 1):
        if above[i] == above[i + 1]:
            continue

        y0 = y[i]
        y1 = y[i + 1]
        x0 = x[i]
        x1 = x[i + 1]
        crossing = float(x0 + (threshold - y0) * (x1 - x0) / (y1 - y0))

        if above[i]:
            intervals.append((float(start), crossing))
            start = None
        else:
            start = crossing

    if start is not None:
        intervals.append((float(start), float(x[-1])))

    return intervals


@dataclass(slots=True)
class ResultsTorreyResonance:
    detunings: NDArray[np.floating]
    sigma_z: NDArray[np.floating]
    params: Parameters
    fwhm: float
    snr: float

    @property
    def populations(self) -> NDArray[np.floating]:
        return (1.0 - self.sigma_z) / 2.0

    def plot(self) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        det_mhz = self.detunings / u.pi2 / u.MHz
        half_width_mhz = self.fwhm / 2.0 / u.pi2 / u.MHz
        fwhm_mhz = self.fwhm / u.pi2 / u.MHz

        ax.plot(
            det_mhz,
            self.sigma_z,
            color="#1f77b4",
            lw=3.0,
            label=r"$\langle\sigma_z\rangle$",
        )
        ax.axvspan(
            -half_width_mhz,
            half_width_mhz,
            color="#f2b84b",
            alpha=0.18,
            zorder=0,
            label=f"FWHM = {fwhm_mhz:.4g} MHz",
        )
        ax.axvline(
            +half_width_mhz,
            color="#c44e52",
            ls="--",
            lw=1.8,
            alpha=0.9,
        )
        ax.axvline(
            -half_width_mhz,
            color="#c44e52",
            ls="--",
            lw=1.8,
            alpha=0.9,
        )
        ax.set_xlabel(r"Drive detuning $(f_d-f_{01})$ (MHz)")
        ax.set_ylabel(r"$\langle\sigma_z\rangle$")
        ax.set_ylim(0.0, 1.04)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.22)
        ax.legend(
            title=f"Contrast = {self.snr:.4g}",
            loc="lower right",
            frameon=True,
            facecolor="white",
            edgecolor="0.85",
            framealpha=0.95,
        )
        fig.tight_layout()
        return fig


@dataclass(slots=True)
class ResultsTorreyFwhmVsRabi:
    rabi_frequencies: NDArray[np.floating]
    fwhm: NDArray[np.floating]
    snr: NDArray[np.floating]
    params: Parameters
    log_scale: bool = False
    plot_product: bool = False
    plot_thresholds: bool = True
    inverse_fwhm_threshold: float = 1e-1
    snr_threshold: float = 0.2
    product_threshold: float = 1e-1

    @property
    def inverse_fwhm_mhz(self) -> NDArray[np.floating]:
        fwhm_mhz = self.fwhm / u.pi2 / u.MHz
        return 1.0 / fwhm_mhz

    @property
    def inverse_t2_limit_mhz(self) -> float:
        t2_limited_fwhm = torrey_fwhm(0.0, self.params.T1, self.params.T2)
        return float(1.0 / (t2_limited_fwhm / u.pi2 / u.MHz))

    @property
    def inverse_fwhm_t2_units(self) -> NDArray[np.floating]:
        return self.inverse_fwhm_mhz / self.inverse_t2_limit_mhz

    @property
    def inverse_fwhm_snr_product(self) -> NDArray[np.floating]:
        return self.inverse_fwhm_t2_units * self.snr

    @property
    def rabi_frequencies_mhz(self) -> NDArray[np.floating]:
        return self.rabi_frequencies / u.pi2 / u.MHz

    @property
    def inverse_fwhm_threshold_rabi_mhz(self) -> Optional[float]:
        return _threshold_crossing_x(
            self.rabi_frequencies_mhz,
            self.inverse_fwhm_t2_units,
            self.inverse_fwhm_threshold,
        )

    @property
    def snr_threshold_rabi_mhz(self) -> Optional[float]:
        return _threshold_crossing_x(
            self.rabi_frequencies_mhz,
            self.snr,
            self.snr_threshold,
        )

    @property
    def product_threshold_rabi_ranges_mhz(self) -> list[tuple[float, float]]:
        return _threshold_intervals_x(
            self.rabi_frequencies_mhz,
            self.inverse_fwhm_snr_product,
            self.product_threshold,
        )

    def plot(
        self,
        *,
        include_snr: bool = True,
        include_product: Optional[bool] = None,
    ) -> plt.Figure:
        if include_product is None:
            include_product = self.plot_product
        if include_product and not include_snr:
            raise ValueError("Product plot requires the contrast axis.")

        fig, ax = plt.subplots(figsize=(8.8, 4.8))
        rabi_mhz = self.rabi_frequencies_mhz

        inverse_fwhm_line = ax.plot(
            rabi_mhz,
            self.inverse_fwhm_t2_units,
            "-",
            color="#1f77b4",
            lw=3.0,
            label="1/FWHM",
        )
        t2_limit_line = ax.axhline(
            1.0,
            color="0.45",
            ls="--",
            lw=1.8,
            label="T2 limit",
        )
        ax.set_xlabel(r"$\Omega_0/2\pi$ (MHz)")
        ax.set_ylabel("1/FWHM (T2-limit units)")
        ax.tick_params(axis="x", labelsize=17)
        ax.tick_params(axis="y", colors="#1f77b4", labelsize=17)
        ax.xaxis.label.set_size(19)
        ax.yaxis.label.set_size(19)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(3e-4, 1.5)
        ax.grid(True, alpha=0.22)

        if include_snr:
            ax_snr = ax.twinx()
            ax_snr.plot(
                rabi_mhz,
                self.snr,
                "--",
                color="#c44e52",
                lw=2.8,
                label="Contrast",
            )
            ax_snr.set_ylabel("Contrast")
            ax_snr.tick_params(axis="y", colors="#c44e52", labelsize=17)
            ax_snr.yaxis.label.set_size(19)
            ax_snr.spines["top"].set_visible(False)
            ax_snr.set_ylim(-0.02, 0.525)

        if include_product:
            ax.plot(
                rabi_mhz,
                self.inverse_fwhm_snr_product,
                "-.",
                color="#2ca02c",
                lw=2.6,
                label="(1/FWHM) × contrast",
            )

        if self.plot_thresholds:
            if include_product:
                for left, right in self.product_threshold_rabi_ranges_mhz:
                    ax.axvspan(left, right, color="#7fc97f", alpha=0.2, zorder=0)

        if self.log_scale:
            positive_rabi = rabi_mhz[rabi_mhz > 0]
            if positive_rabi.size == 0:
                raise ValueError("Log-scale Rabi plot requires positive amplitudes.")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(FixedLocator([1.0, 0.1, 0.01, 0.001]))
            ax.yaxis.set_major_formatter(FixedFormatter(["1", "0.1", "0.01", "0.001"]))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_xlim(float(np.min(positive_rabi)), float(np.max(positive_rabi)))

        fig.tight_layout()
        return fig


@dataclass(slots=True)
class OptionsTorreyResonance(Options):
    plot: bool = True
    save_fig: bool = False
    fig_dir: Optional[Path] = None
    log_scale: bool = False
    plot_product: bool = False
    plot_thresholds: bool = True
    inverse_fwhm_threshold: float = 1e-1
    snr_threshold: float = 0.2
    product_threshold: float = 1e-1


class TorreyResonanceExperiment(BaseExperiment[ResultsTorreyResonance]):
    """
    Plot the analytic steady-state resonance and calculate its FWHM.

    The expression is:
        <sigma_z> = (1 + (T2 Delta)^2)
                    / (1 + (T2 Delta)^2 + Omega^2 T1 T2)
    """

    def __init__(
        self,
        detunings: NDArray[np.floating],
        params: Parameters,
        options: Optional[OptionsTorreyResonance] = None,
    ) -> None:
        super().__init__(params, options)
        self.options = options or OptionsTorreyResonance()
        self.detunings = np.asarray(detunings, dtype=float)

    def run(self) -> ResultsTorreyResonance:
        self.results = ResultsTorreyResonance(
            detunings=self.detunings,
            sigma_z=torrey_sigma_z(
                self.detunings,
                self.params.rabi_frequency,
                self.params.T1,
                self.params.T2,
            ),
            params=copy(self.params),
            fwhm=torrey_fwhm(
                self.params.rabi_frequency,
                self.params.T1,
                self.params.T2,
            ),
            snr=torrey_snr(
                self.params.rabi_frequency,
                self.params.T1,
                self.params.T2,
            ),
        )

        if self.options.plot:
            self.plot()

        if self.options.save_fig:
            if self.current_figure is None:
                self.plot()
            self.figure_path = self._save_fig("torrey_resonance.png")

        if self.options.save:
            self.save()

        return self.results

    def plot(self) -> None:
        self._check_results()
        self.current_figure = self.results.plot()

    def _get_experiment_name(self) -> str:
        return "torrey_resonance"

    def _save_fig(self, filename: str) -> Path:
        fig_dir = Path(self.options.fig_dir) if self.options.fig_dir else FIGS_DIR
        fig_dir.mkdir(parents=True, exist_ok=True)
        path = fig_dir / filename
        self.current_figure.savefig(path, dpi=300, bbox_inches="tight")
        return path

    def _save_results(self, save_dir: Path) -> None:
        self._check_results()
        np.savez(
            save_dir / "results.npz",
            detuning_convention="drive_minus_qubit",
            detunings=self.results.detunings,
            sigma_z=self.results.sigma_z,
            fwhm=self.results.fwhm,
            snr=self.results.snr,
        )


class TorreyFwhmVsRabiExperiment(BaseExperiment[ResultsTorreyFwhmVsRabi]):
    """Calculate and plot analytic FWHM as a function of Rabi amplitude."""

    def __init__(
        self,
        rabi_frequencies: NDArray[np.floating],
        params: Parameters,
        options: Optional[OptionsTorreyResonance] = None,
    ) -> None:
        super().__init__(params, options)
        self.options = options or OptionsTorreyResonance()
        self.rabi_frequencies = np.asarray(rabi_frequencies, dtype=float)

    def run(self) -> ResultsTorreyFwhmVsRabi:
        self.results = ResultsTorreyFwhmVsRabi(
            rabi_frequencies=self.rabi_frequencies,
            fwhm=np.asarray(
                [
                    torrey_fwhm(rabi_frequency, self.params.T1, self.params.T2)
                    for rabi_frequency in self.rabi_frequencies
                ],
                dtype=float,
            ),
            snr=np.asarray(
                [
                    torrey_snr(rabi_frequency, self.params.T1, self.params.T2)
                    for rabi_frequency in self.rabi_frequencies
                ],
                dtype=float,
            ),
            params=copy(self.params),
            log_scale=self.options.log_scale,
            plot_product=self.options.plot_product,
            plot_thresholds=self.options.plot_thresholds,
            inverse_fwhm_threshold=self.options.inverse_fwhm_threshold,
            snr_threshold=self.options.snr_threshold,
            product_threshold=self.options.product_threshold,
        )

        if self.options.plot:
            self.plot()

        if self.options.save_fig:
            if self.current_figure is None:
                self.plot()
            self.figure_path = self._save_fig("torrey_fwhm_vs_rabi.png")

        if self.options.save:
            self.save()

        return self.results

    def plot(self) -> None:
        self._check_results()
        self.current_figure = self.results.plot()

    def _get_experiment_name(self) -> str:
        return "torrey_fwhm_vs_rabi"

    def _save_fig(self, filename: str) -> Path:
        fig_dir = Path(self.options.fig_dir) if self.options.fig_dir else FIGS_DIR
        fig_dir.mkdir(parents=True, exist_ok=True)
        path = fig_dir / filename
        self.current_figure.savefig(path, dpi=300, bbox_inches="tight")
        return path

    def _save_results(self, save_dir: Path) -> None:
        self._check_results()
        np.savez(
            save_dir / "results.npz",
            rabi_frequencies=self.results.rabi_frequencies,
            fwhm=self.results.fwhm,
            snr=self.results.snr,
            log_scale=self.results.log_scale,
            plot_product=self.results.plot_product,
            plot_thresholds=self.results.plot_thresholds,
            inverse_fwhm_threshold=self.results.inverse_fwhm_threshold,
            snr_threshold=self.results.snr_threshold,
            product_threshold=self.results.product_threshold,
        )


if __name__ == "__main__":
    params = Parameters(
        T1=30 * u.us,
        T_dephasing=20 * u.us,
        rabi_frequency=0.1 * u.pi2 * u.MHz,
    )
    options = OptionsTorreyResonance(plot=True, save=False, log_scale=True, plot_product=True)

    span = 2.0 * u.pi2 * u.MHz
    detunings = np.linspace(-span / 2.0, span / 2.0, 501)
    result = TorreyResonanceExperiment(detunings, params, options).run()
    print(f"FWHM: {result.fwhm / u.pi2 / u.MHz:.6g} MHz")

    rabi_frequencies = np.linspace(0.0, 0.5, 251) * u.pi2 * u.MHz
    rabi_frequencies = np.logspace(-3, 1, 251) * u.pi2 * u.MHz
    TorreyFwhmVsRabiExperiment(rabi_frequencies, params, options).run()
    plt.show()
    
