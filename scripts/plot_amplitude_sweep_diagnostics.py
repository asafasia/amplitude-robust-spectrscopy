from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ars-matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echospec.utils.units import Units as u


DATA_PATH = ROOT / "amplitude_sweep_spectroscopy_results.npz"
OUTPUT_DIR = ROOT / "figures" / "final"
SWEEP_OUTPUT = OUTPUT_DIR / "amplitude_sweep_2d_with_fwhm.png"
CUTS_OUTPUT = OUTPUT_DIR / "amplitude_sweep_fwhm_linecuts.png"


def main() -> None:
    data = np.load(DATA_PATH, allow_pickle=True)
    results = data["data"].item()

    detuning_mhz = results.detunings / u.pi2 / u.MHz
    amplitude_mhz = results.amplitudes / u.pi2 / u.MHz
    populations = results.populations
    fwhm_mhz = results.fwhm_map / u.pi2 / u.MHz
    signal = results.snr_map

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    mesh = ax.pcolormesh(
        detuning_mhz,
        amplitude_mhz,
        populations,
        shading="auto",
        cmap="viridis",
    )
    half = 0.5 * fwhm_mhz
    valid = np.isfinite(half) & (half > 0)
    ax.plot(+half[valid], amplitude_mhz[valid], color="#d62728", lw=2.0, label="FWHM")
    ax.plot(-half[valid], amplitude_mhz[valid], color="#d62728", lw=2.0)
    t2_half = results.spectroscopies[0].params.T2_limit / u.MHz / 2
    ax.axvline(+t2_half, color="white", ls="--", lw=1.4, label="T2 limit")
    ax.axvline(-t2_half, color="white", ls="--", lw=1.4)
    ax.set_xlabel(r"$\Delta/2\pi$ (MHz)")
    ax.set_ylabel(r"$\Omega_0/2\pi$ (MHz)")
    ax.set_title("2D spectroscopy sweep with fitted FWHM")
    ax.legend(loc="upper right")
    fig.colorbar(mesh, ax=ax, label=r"$P_e$")
    fig.tight_layout()
    fig.savefig(SWEEP_OUTPUT)
    plt.close(fig)

    target_amplitudes = np.array([0.3, 0.75, 1.5, 3.0, 6.0, 10.0])
    indices = [int(np.argmin(np.abs(amplitude_mhz - amp))) for amp in target_amplitudes]

    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.2), sharex=True)
    for ax, idx in zip(axes.ravel(), indices):
        amp = amplitude_mhz[idx]
        ax.plot(detuning_mhz, populations[idx], color="#1f77b4", lw=2.0)
        if np.isfinite(fwhm_mhz[idx]) and fwhm_mhz[idx] > 0:
            h = 0.5 * fwhm_mhz[idx]
            ax.axvline(+h, color="#d62728", ls="--", lw=1.5)
            ax.axvline(-h, color="#d62728", ls="--", lw=1.5)
        ax.axvline(+t2_half, color="0.45", ls=":", lw=1.2)
        ax.axvline(-t2_half, color="0.45", ls=":", lw=1.2)
        ax.set_title(
            f"{amp:.2f} MHz, FWHM={fwhm_mhz[idx]:.3g} MHz, Signal={signal[idx]:.3g}"
        )
        ax.grid(True, alpha=0.22)
    for ax in axes[-1]:
        ax.set_xlabel(r"$\Delta/2\pi$ (MHz)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$P_e$")
    fig.tight_layout()
    fig.savefig(CUTS_OUTPUT)
    plt.close(fig)

    print(SWEEP_OUTPUT)
    print(CUTS_OUTPUT)


if __name__ == "__main__":
    main()
