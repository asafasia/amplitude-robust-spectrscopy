"""Plotting utilities for spectroscopy analysis."""

from __future__ import annotations

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from typing import Optional

from echospec.utils.units import Units as u
from echospec.utils.parameters import Parameters
from echospec.simulation.utils import z_to_populations


def plot_spectroscopy(
    z_final: xr.DataArray,
    params: Parameters,
    rabi_frequency: float,
    fwhm: Optional[float] = None,
    plot_population: bool = False,
    with_fwhm: bool = False,
) -> plt.Figure:
    """Plot 1D spectroscopy (final Z vs detuning)."""
    z = z_final
    if plot_population:
        z = z_to_populations(z.values)[1]

    det_mhz = z_final.detuning.values / u.pi2 / u.MHz

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        det_mhz,
        z,
        ".-",
        label=f"Rabi = {rabi_frequency / u.pi2 / u.MHz:.2f} MHz",
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


def plot_amplitude_sweep_spectroscopy(
    results: xr.DataArray,
    params: Parameters,
    fwhm_values: Optional[np.ndarray] = None,
    amplitudes: Optional[np.ndarray] = None,
    detunings: Optional[np.ndarray] = None,
    plot_fwhm: bool = True,
) -> plt.Figure:
    """
    Plot final-time Z population as a function of detuning and amplitude.

    Parameters
    ----------
    results : xr.DataArray
        The spectroscopy results data array with dimensions (amplitude, detuning, observable, time)
    params : Parameters
        Parameters object containing T2_limit and other configuration
    fwhm_values : Optional[np.ndarray]
        Array of FWHM values for each amplitude
    amplitudes : Optional[np.ndarray]
        Array of amplitude values (will be extracted from results if not provided)
    detunings : Optional[np.ndarray]
        Array of detuning values (will be extracted from results if not provided)
    plot_fwhm : bool
        Whether to plot FWHM overlay curves
    """
    # Final-time Z expectation
    z_final = results.sel(observable="z").isel(time=-1)

    # Convert ⟨Z⟩ → populations
    pop_result = z_to_populations(z_final)
    populations = pop_result[1] if isinstance(
        pop_result, (tuple, list)) else pop_result

    # Extract coordinates if not provided
    if amplitudes is None:
        amplitudes = z_final.amplitude.values
    if detunings is None:
        detunings = z_final.detuning.values

    fig, ax = plt.subplots(figsize=(6, 4))

    pcm = ax.pcolormesh(
        z_final.detuning / u.pi2 / u.MHz,
        z_final.amplitude / u.pi2 / u.MHz,
        populations,
        shading="auto",
    )

    # Plot FWHM overlay if enabled
    if plot_fwhm and fwhm_values is not None and len(fwhm_values) > 0:
        fwhm_mhz = np.array([f / u.pi2 / u.MHz if f is not None else np.nan
                             for f in fwhm_values])
        amp_mhz = amplitudes / u.pi2 / u.MHz

        # Plot FWHM as curves (positive and negative)
        ax.plot(fwhm_mhz / 2, amp_mhz, 'r.-', linewidth=2, label='FWHM')
        ax.plot(-fwhm_mhz / 2, amp_mhz, 'r.-', linewidth=2)
        ax.legend()

    # Plot T2 limit
    t2_half = params.T2_limit / u.MHz / 2
    ax.axvline(+t2_half, color="gray", ls="--",
               label=f"T2 limit = {params.T2_limit / u.MHz:.3g} MHz")
    ax.axvline(-t2_half, color="gray", ls="--")
    ax.legend()

    ax.set_xlabel("Detuning [MHz]")
    ax.set_ylabel("Drive amplitude [MHz]")
    ax.set_xlim(detunings.min() / u.pi2 / u.MHz,
                detunings.max() / u.pi2 / u.MHz)
    ax.set_title("Final ⟨Z⟩ Population vs Detuning and Amplitude")
    fig.colorbar(pcm, ax=ax, label="Final population")

    fig.tight_layout()
    return fig
