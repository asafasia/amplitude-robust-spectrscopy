from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ars-matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echospec.utils.units import Units as u

DATA_PATH = ROOT / "amplitude_sweep_spectroscopy_results.npz"
OUTPUT_PATH = (
    ROOT / "figures" / "final" / "inverse_fwhm_t2_spec_amplitude_vs_amplitude_log.png"
)
PRODUCT_THRESHOLD = 0.1


def threshold_intervals(
    x: np.ndarray,
    y: np.ndarray,
    threshold: float,
) -> list[tuple[float, float]]:
    order = np.argsort(x)
    x = np.asarray(x, dtype=float)[order]
    y = np.asarray(y, dtype=float)[order]
    above = y >= threshold

    intervals: list[tuple[float, float]] = []
    start = float(x[0]) if above[0] else None

    for i in range(x.size - 1):
        if above[i] == above[i + 1]:
            continue

        crossing = float(
            x[i] + (threshold - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
        )
        if above[i]:
            intervals.append((float(start), crossing))
            start = None
        else:
            start = crossing

    if start is not None:
        intervals.append((float(start), float(x[-1])))

    return intervals


def main() -> None:
    data = np.load(DATA_PATH, allow_pickle=True)
    results = data["data"].item()

    amplitude_mhz = results.amplitudes / u.pi2 / u.MHz
    fwhm_mhz = results.fwhm_map / u.pi2 / u.MHz
    t2_limit_mhz = results.spectroscopies[0].params.T2_limit / u.MHz
    inverse_fwhm_t2_units = 1 / (fwhm_mhz / t2_limit_mhz)
    spec_amplitude = results.snr_map

    mask = (
        np.isfinite(amplitude_mhz)
        & np.isfinite(fwhm_mhz)
        & np.isfinite(inverse_fwhm_t2_units)
        & np.isfinite(spec_amplitude)
        & (amplitude_mhz > 0)
        & (fwhm_mhz > 0)
    )

    amplitude_mhz = amplitude_mhz[mask]
    inverse_fwhm_t2_units = inverse_fwhm_t2_units[mask]
    spec_amplitude = spec_amplitude[mask]
    inverse_fwhm_signal_product = inverse_fwhm_t2_units * spec_amplitude
    product_ranges = threshold_intervals(
        amplitude_mhz,
        inverse_fwhm_signal_product,
        PRODUCT_THRESHOLD,
    )

    fig, ax_inverse_fwhm = plt.subplots(figsize=(8.8, 4.8))

    ax_inverse_fwhm.plot(
        amplitude_mhz,
        inverse_fwhm_t2_units,
        "-",
        color="#1f77b4",
        lw=3.0,
        label="1/FWHM",
    )
    ax_inverse_fwhm.axhline(
        1.0,
        color="0.45",
        ls="--",
        lw=1.8,
        label="T2 limit",
    )
    ax_inverse_fwhm.plot(
        amplitude_mhz,
        inverse_fwhm_signal_product,
        "-.",
        color="#2ca02c",
        lw=2.6,
        label="(1/FWHM) x Signal",
    )
    for left, right in product_ranges:
        ax_inverse_fwhm.axvspan(left, right, color="#7fc97f", alpha=0.2, zorder=0)
    ax_inverse_fwhm.set_xscale("log")
    ax_inverse_fwhm.set_yscale("log")
    ax_inverse_fwhm.set_xlabel(r"$\Omega_0/2\pi$ (MHz)")
    ax_inverse_fwhm.set_ylabel("1/FWHM (T2-limit units)")
    ax_inverse_fwhm.tick_params(axis="x", labelsize=17)
    ax_inverse_fwhm.tick_params(axis="y", colors="#1f77b4", labelsize=17)
    ax_inverse_fwhm.xaxis.label.set_size(19)
    ax_inverse_fwhm.yaxis.label.set_size(19)
    ax_inverse_fwhm.spines["top"].set_visible(False)
    ax_inverse_fwhm.spines["right"].set_visible(False)
    left_axis_max = np.nanmax(
        [np.nanmax(inverse_fwhm_t2_units), np.nanmax(inverse_fwhm_signal_product)]
    )
    ax_inverse_fwhm.set_ylim(
        3e-4,
        max(1.5, float(left_axis_max) * 1.15),
    )
    ax_inverse_fwhm.yaxis.set_major_locator(
        FixedLocator([1.0, 0.1, 0.01, 0.001])
    )
    ax_inverse_fwhm.yaxis.set_major_formatter(
        FixedFormatter(["1", "0.1", "0.01", "0.001"])
    )
    ax_inverse_fwhm.yaxis.set_minor_formatter(NullFormatter())
    ax_inverse_fwhm.grid(True, alpha=0.22)

    ax_spec = ax_inverse_fwhm.twinx()
    ax_spec.plot(
        amplitude_mhz,
        spec_amplitude,
        "--",
        color="#d62728",
        lw=2.8,
        label="Signal",
    )
    ax_spec.set_ylabel("Signal")
    ax_spec.tick_params(axis="y", colors="#d62728", labelsize=17)
    ax_spec.yaxis.label.set_size(19)
    ax_spec.spines["top"].set_visible(False)
    ax_spec.set_ylim(-0.02, max(0.525, float(np.nanmax(spec_amplitude)) * 1.08))

    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH)
    print(OUTPUT_PATH)
    for left, right in product_ranges:
        print(
            f"(1/FWHM) x Signal > {PRODUCT_THRESHOLD}: "
            f"{left:.4g}-{right:.4g} MHz"
        )


if __name__ == "__main__":
    main()
