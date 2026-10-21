"""Combine the measured maps and compact center/linewidth panels for the Letter.

Uses the documented paper caches; regenerate those with paper-data-core first.
No beta optimization is inferred from the two selected acquisition settings.
"""

from __future__ import annotations

# ruff: noqa: E402
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
STEM = "04_main_drag_summary"


def main():
    paths = [
        ROOT / "paper/data/experimental/04_main_ac_stark_correction_maps.npz",
        ROOT / "paper/data/experimental/04_main_ac_stark_shifts_square.npz",
        ROOT / "paper/data/numerical/04_main_ac_stark_shifts_square.npz",
    ]
    datasets = []
    for path in paths:
        with np.load(path, allow_pickle=False) as source:
            datasets.append({key: source[key] for key in source.files})
    maps, curves, reference = datasets
    gamma_t2_khz = float(reference["reference_t2_fwhm_khz"])
    displayed = {"reference_t2_fwhm_khz": np.asarray(gamma_t2_khz)}
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "axes.linewidth": 0.6,
            "lines.linewidth": 0.7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(3.4, 3.45), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.35, 1])
    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(2)]
    colors = ["#6a1b9a", "#21918c"]
    for i, (prefix, beta, color) in enumerate(
        zip(("plain", "drag"), maps["drag_beta"], colors, strict=True)
    ):
        ax = axes[i]
        mesh = ax.pcolormesh(
            maps[f"{prefix}_display_detuning_mhz"] * 1000,
            maps["rabi_mhz"],
            maps[f"{prefix}_pe"],
            cmap="magma",
            shading="auto",
            rasterized=True,
            vmin=maps["display_population_limits"][0],
            vmax=maps["display_population_limits"][1],
        )
        mask = maps[f"{prefix}_display_center_mask"]
        # Smooth each contiguous accepted segment separately, never across gaps.
        centers = np.full(mask.shape, np.nan)
        indices = np.flatnonzero(mask)
        segments = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
        for segment in segments:
            if segment.size:
                centers[segment] = gaussian_filter1d(
                    maps[f"{prefix}_display_centers_mhz"][segment] * 1000,
                    sigma=1.0,
                    mode="nearest",
                    truncate=3.0,
                )
        displayed[f"{prefix}_centers_khz"] = centers
        displayed[f"{prefix}_rabi_mhz"] = maps["rabi_mhz"]
        ax.plot(
            centers,
            maps["rabi_mhz"],
            color="white",
            lw=0.65,
        )
        ax.set(
            xlim=(-150, 150),
            ylim=(10, 60),
            xticks=[-100, 0, 100],
            xlabel="Centered detuning (kHz)",
        )
        ax.set_title(rf"({chr(97 + i)}) $\beta={beta:g}$", loc="left")
        x = curves[f"{prefix}_rabi_mhz"]
        label = rf"$\beta={beta:g}$"
        axes[2].plot(maps["rabi_mhz"], centers, color=color, label=label)
        widths = np.where(
            curves[f"{prefix}_linewidth_fit_accepted"],
            curves[f"{prefix}_linewidth_fit_fwhm_hz"] / 1000,
            np.nan,
        )
        displayed[f"{prefix}_fwhm_t2_units"] = widths / gamma_t2_khz
        displayed[f"{prefix}_linewidth_rabi_mhz"] = x
        axes[3].plot(x, widths / gamma_t2_khz, color=color, label=label)
    axes[0].set_ylabel(r"$\Omega_{\rm cal}/2\pi$ (MHz)")
    axes[1].tick_params(labelleft=False)
    cb = fig.colorbar(
        mesh,
        ax=axes[:2],
        location="top",
        fraction=0.065,
        pad=0.04,
        aspect=35,
        ticks=[0.4, 0.5, 0.6],
    )
    cb.set_label(r"Measured $P_e$", labelpad=2)
    cb.ax.tick_params(labelsize=6, length=2, pad=1)
    axes[2].set(ylabel="Center variation (kHz)", ylim=(-35, 20), yticks=[-30, 0, 15])
    axes[2].axhline(0, color="0.6", lw=0.5, zorder=0)
    axes[2].legend(
        fontsize=5.5,
        loc="upper left",
        frameon=False,
        handlelength=1.1,
        borderpad=0.1,
        labelspacing=0.2,
    )
    axes[3].plot(
        reference["rabi_mhz"],
        reference["constant_fwhm_khz"] / gamma_t2_khz,
        color="#c65d4b",
        label="Constant drive",
    )
    axes[3].axhline(1.0, color="0.5", ls=":", lw=0.6)
    axes[3].set(
        ylabel=r"FWHM / $\Gamma_{T_2}$",
        yscale="log",
        ylim=(10 / gamma_t2_khz, 1e6 / gamma_t2_khz),
        yticks=[1, 100, 10000],
    )
    axes[3].text(
        0.05,
        0.91,
        "Constant drive",
        color="#c65d4b",
        fontsize=5.5,
        transform=axes[3].transAxes,
    )
    for ax, title in zip(
        axes[2:], ("(c) Center stability", "(d) Linewidth"), strict=True
    ):
        ax.set_title(title, loc="left")
        ax.set(
            xlim=(10, 60), xticks=[10, 30, 60], xlabel=r"$\Omega_{\rm cal}/2\pi$ (MHz)"
        )
        ax.tick_params(length=2, pad=2)
    output = ROOT / "figures/paper" / STEM
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=350)
    plt.close(fig)
    displayed["constant_rabi_mhz"] = reference["rabi_mhz"]
    displayed["constant_fwhm_t2_units"] = reference["constant_fwhm_khz"] / gamma_t2_khz
    np.savez_compressed(output.with_suffix(".npz"), **displayed)
    provenance = {
        "generator": "scripts/make_main_drag_summary.py",
        "reproduction_command": "python scripts/make_main_drag_summary.py",
        "figure_size_inches": [3.4, 3.45],
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
        "panels": {
            "a_b": "Unmodified measured populations; lightly smoothed mean-centered fits",
            "c": "Same smoothed center traces as a/b; MHz converted to kHz",
            "d": "Unsmoothed accepted Gaussian FWHM and Bloch reference divided by Gamma_T2",
        },
        "display_smoothing": {
            "method": "Gaussian filter, sigma=1 amplitude sample, truncate=3, nearest edges",
            "gaps": "Filter contiguous accepted runs independently; retain gaps",
            "scope": "Center traces in a/b/c only; raw fits and reported RMS unchanged",
        },
        "linewidth_normalization_khz": gamma_t2_khz,
        "display_arrays": f"figures/paper/{STEM}.npz",
        "limitations": [
            "Beta and midpoint smoothing both differ between measured runs.",
            "These are selected settings, not beta-optimization sweeps.",
            "No paired experimental and simulated beta sweeps found locally.",
            "The Bloch reference uses the primary qubit coherence calibration.",
        ],
    }
    output.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
