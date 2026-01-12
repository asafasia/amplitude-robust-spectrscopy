"""Plotting utilities for spectroscopy analysis."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional
from numpy.typing import NDArray

from echospec.utils.units import Units as u
from echospec.utils.parameters import Parameters


def plot_spectroscopy(
    populations: NDArray[np.floating],
    detunings: NDArray[np.floating],
    params: Parameters,
    fwhm: Optional[float] = None,
    with_fwhm: bool = False,
) -> Figure:
    """Plot 1D spectroscopy (final Z vs detuning)."""

    det_mhz = detunings / u.pi2 / u.MHz

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        det_mhz,
        populations,
        ".-",
        label=f"Rabi = {params.rabi_frequency / u.pi2 / u.MHz:.2f} MHz",
    )

    if with_fwhm and fwhm is not None:
        half = fwhm / 2 / u.pi2 / u.MHz
        ax.axvline(+half, color="red", ls="-")
        ax.axvline(
            -half,
            color="red",
            ls="-",
            label=f"FWHM = {fwhm / u.pi2 / u.MHz:.3g} MHz",
        )

    t2_half = params.T2_limit / u.MHz / 2
    ax.axvline(
        +t2_half,
        color="gray",
        ls="--",
        label=f"T2 limit = {params.T2_limit / u.MHz:.3g} MHz",
    )
    ax.axvline(-t2_half, color="gray", ls="--")

    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Detuning (MHz)")
    ax.set_ylabel("Final ⟨Z⟩")
    ax.set_title("Spectroscopy")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    return fig


def plot_spectroscopy_2d(
    populations: NDArray[np.floating],          # (n_amp, n_detuning)
    amplitudes: NDArray[np.floating],            # (n_amp,)
    detunings: NDArray[np.floating],             # (n_detuning,)
    params: Parameters,
    fwhm: Optional[NDArray[np.floating]] = None,  # (n_amp,)
    with_fwhm: bool = False,
) -> Figure:
    """
    Plot 2D spectroscopy: final population vs detuning and amplitude.
    """

    fwhm = np.array(fwhm) if fwhm is not None else None
    det_mhz = detunings / u.pi2 / u.MHz
    amp_mhz = amplitudes / u.pi2 / u.MHz

    fig, ax = plt.subplots(figsize=(7, 5))

    pcm = ax.pcolormesh(
        det_mhz,
        amp_mhz,
        populations,
        shading="auto",
    )

    # ---------- FWHM overlay ----------
    if with_fwhm and fwhm is not None:
        fwhm_mhz = fwhm / u.pi2 / u.MHz
        half = fwhm_mhz / 2

        ax.plot(+half, amp_mhz, "r-", lw=2, label="FWHM")
        ax.plot(-half, amp_mhz, "r-", lw=2)
        ax.set_xlim(det_mhz.min(), det_mhz.max())

    # ---------- T2 limit ----------
    t2_half = params.T2_limit / u.MHz / 2
    ax.axvline(
        +t2_half,
        color="gray",
        ls="--",
        label=f"T2 limit = {params.T2_limit / u.MHz:.3g} MHz",
    )
    ax.axvline(-t2_half, color="gray", ls="--")

    # ---------- labels ----------
    ax.set_xlabel("Detuning (MHz)")
    ax.set_ylabel("Drive amplitude (MHz)")
    ax.set_title("Spectroscopy (final population)")

    fig.colorbar(pcm, ax=ax, label="Final population")

    if with_fwhm or params.T2_limit is not None:
        ax.legend()

    fig.tight_layout()
    return fig
