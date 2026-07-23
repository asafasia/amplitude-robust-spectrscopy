"""Generate the Supplemental Pe comparison for three pulse lengths."""

from __future__ import annotations

# Backend and local-source setup must precede pyplot and echospec imports.
# ruff: noqa: E402, I001

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

from echospec.figures import FigureVariant, apply_figure_style, save_figure


DURATIONS_US = (5, 10, 20)
DATA_DIR = ROOT / "data/generated/accumulated_phase_duration_sweep"
OUTPUT_STEM = "19_accumulated_phase_pe_durations"
COLOR_MIN = 0.0
COLOR_MAX = 0.5


def load_duration(duration_us: int) -> dict[str, np.ndarray]:
    path = DATA_DIR / f"{duration_us}us/results.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}; run make_accumulated_phase_duration_report.py."
        )
    with np.load(path, allow_pickle=False) as data:
        result = {
            "duration_us": np.asarray(data["duration_us"]),
            "cutoff": np.asarray(data["cutoff"]),
            "drag_beta": np.asarray(data["drag_beta"]),
            "kappa": np.asarray(data["selected_kappa_mhz_inv"]),
            "t2_limit_fwhm_mhz": np.asarray(data["t2_limit_fwhm_mhz"]),
            "detuning_mhz": np.asarray(data["detuning_mhz"], dtype=float),
            "rabi_mhz": np.asarray(data["rabi_mhz"], dtype=float),
            "plain_pe": np.asarray(data["plain_pe"], dtype=float),
            "corrected_pe": np.asarray(data["corrected_pe"], dtype=float),
        }

    if float(result["duration_us"]) != duration_us:
        raise ValueError(f"{path} does not contain the requested duration.")
    if float(result["cutoff"]) != 0.001 or float(result["drag_beta"]) != 0.0:
        raise ValueError(f"{path} must use c=0.001 and beta=0.")
    expected_shape = (
        result["rabi_mhz"].size,
        result["detuning_mhz"].size,
    )
    for key in ("plain_pe", "corrected_pe"):
        if result[key].shape != expected_shape:
            raise ValueError(
                f"{key} has shape {result[key].shape}, not {expected_shape}."
            )
    return result


def main() -> None:
    apply_figure_style(FigureVariant.PAPER)
    datasets = [load_duration(duration) for duration in DURATIONS_US]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(6.8, 3.9),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    image = None
    for column, data in enumerate(datasets):
        half_width = 0.5 * float(data["t2_limit_fwhm_mhz"])
        for row, population in enumerate(
            (data["plain_pe"], data["corrected_pe"])
        ):
            axis = axes[row, column]
            image = axis.pcolormesh(
                data["detuning_mhz"],
                data["rabi_mhz"],
                population,
                shading="auto",
                cmap="magma",
                vmin=COLOR_MIN,
                vmax=COLOR_MAX,
                rasterized=True,
            )
            axis.axvline(0.0, color="white", lw=0.7, ls="--")
            axis.axvline(+half_width, color="white", lw=0.65, ls=":")
            axis.axvline(-half_width, color="white", lw=0.65, ls=":")
            axis.set_xlim(-1.0, 1.0)
            axis.set_ylim(0.0, 80.0)

        axes[0, column].set_title(
            rf"$L={float(data['duration_us']):g}~\mu\mathrm{{s}}$"
        )
        axes[1, column].set_xlabel(r"$\Delta/2\pi$ (MHz)")

    axes[0, 0].set_ylabel(
        "Without correction\n" + r"$\Omega_0/2\pi$ (MHz)"
    )
    axes[1, 0].set_ylabel(
        "Accumulated phase\n" + r"$\Omega_0/2\pi$ (MHz)"
    )
    if image is None:
        raise RuntimeError("No population map was rendered.")
    colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.015)
    colorbar.set_label(r"$P_e$")

    save_figure(
        fig,
        OUTPUT_STEM,
        variant=FigureVariant.PAPER,
        formats=("pdf", "png", "svg"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
