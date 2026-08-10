"""Generate the compact two-panel AC-Stark correction map for the Letter."""

from __future__ import annotations

# Backend and local-source setup must precede pyplot and echospec imports.
# ruff: noqa: E402, I001

import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ars-matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echospec.figures import FigureVariant, apply_figure_style, save_figure
from echospec.paper_data import save_paper_dataset


SOURCE_PATH = (
    ROOT
    / "data/generated/accumulated_phase_duration_sweep/20us/results.npz"
)
CACHE_PATH = ROOT / "figures/paper/04_main_ac_stark_correction_maps.npz"
OUTPUT_STEM = "04_main_ac_stark_correction_maps"
DISPLAY_HALF_WIDTH_MHZ = 0.20
COLOR_MIN = 0.0
COLOR_MAX = 0.5


def latex_macro_float(name: str) -> float:
    """Read a numeric value from the paper's shared coherence parameters."""
    text = (ROOT / "paper/coherence_parameters.tex").read_text()
    match = re.search(rf"\\newcommand\{{\\{name}\}}\{{(-?[0-9.]+)", text)
    if match is None:
        raise ValueError(f"Missing numeric macro {name}")
    return float(match.group(1))


T2_LIMIT_FWHM_MHZ = latex_macro_float("EffectiveCoherenceFwhm") / 1e3
T2_LIMIT_HALF_WIDTH_MHZ = 0.5 * T2_LIMIT_FWHM_MHZ
ANHARMONICITY_MHZ = latex_macro_float("MeasuredAnharmonicity")


def load_map_data() -> dict[str, np.ndarray]:
    """Load the duration-sweep result and preserve a paper-local compact cache."""
    path = SOURCE_PATH if SOURCE_PATH.exists() else CACHE_PATH
    if not path.exists():
        raise FileNotFoundError(
            "Run scripts/make_accumulated_phase_duration_report.py first "
            f"or restore {CACHE_PATH}."
        )

    with np.load(path, allow_pickle=False) as data:
        arrays = {
            "duration_us": np.asarray(data["duration_us"]),
            "cutoff": np.asarray(data["cutoff"]),
            "drag_beta": np.asarray(data["drag_beta"]),
            "anharmonicity_mhz": np.asarray(data["anharmonicity_mhz"]),
            "selected_kappa_mhz_inv": np.asarray(
                data["selected_kappa_mhz_inv"]
            ),
            "detuning_mhz": np.asarray(data["detuning_mhz"], dtype=float),
            "rabi_mhz": np.asarray(data["rabi_mhz"], dtype=float),
            "fit_rabi_mhz": np.asarray(data["fit_rabi_mhz"], dtype=float),
            "plain_pe": np.asarray(data["plain_pe"], dtype=float),
            "corrected_pe": np.asarray(data["corrected_pe"], dtype=float),
            "plain_centers_mhz": np.asarray(
                data["plain_centers_mhz"], dtype=float
            ),
            "corrected_centers_mhz": np.asarray(
                data["corrected_centers_mhz"], dtype=float
            ),
        }

    if float(arrays["duration_us"]) != 20.0:
        raise ValueError("The paper map must use the 20 us simulation.")
    if float(arrays["cutoff"]) != 0.001:
        raise ValueError("The paper map must use cutoff c=0.001.")
    if float(arrays["drag_beta"]) != 0.0:
        raise ValueError("The paper map must use beta=0.")
    if float(arrays["anharmonicity_mhz"]) != ANHARMONICITY_MHZ:
        raise ValueError(
            "The paper map must use the measured q1 anharmonicity."
        )
    expected_shape = (
        arrays["rabi_mhz"].size,
        arrays["detuning_mhz"].size,
    )
    for key in ("plain_pe", "corrected_pe"):
        if arrays[key].shape != expected_shape:
            raise ValueError(f"{key} has shape {arrays[key].shape}, not {expected_shape}.")

    if path == SOURCE_PATH:
        np.savez_compressed(CACHE_PATH, **arrays)
    return arrays


def main() -> None:
    apply_figure_style(FigureVariant.PAPER)
    data = load_map_data()
    paper_data_paths = save_paper_dataset(
        OUTPUT_STEM,
        category="numerical",
        arrays=data,
        provenance={
            "figure_asset": f"figures/paper/{OUTPUT_STEM}.pdf",
            "manuscript_scope": "letter",
            "generator": "scripts/make_main_ac_stark_correction_maps.py",
            "reproduction_command": (
                "PYTHONPATH=. MPLBACKEND=Agg .venv/bin/python "
                "scripts/make_main_ac_stark_correction_maps.py"
            ),
            "source_generator": (
                "scripts/make_accumulated_phase_duration_report.py"
            ),
            "source_cache": (
                "data/generated/accumulated_phase_duration_sweep/20us/"
                "results.npz"
            ),
            "model": "three-level transmon Lindblad/RK4 simulation",
            "detuning_convention": "drive_minus_qubit",
            "population_definition": "P_e",
            "array_dimensions": {
                "plain_pe": ["rabi_mhz", "detuning_mhz"],
                "corrected_pe": ["rabi_mhz", "detuning_mhz"],
                "plain_centers_mhz": ["fit_rabi_mhz"],
                "corrected_centers_mhz": ["fit_rabi_mhz"],
            },
        },
    )
    detuning_mhz = data["detuning_mhz"]
    rabi_mhz = data["rabi_mhz"]
    fit_rabi_mhz = data["fit_rabi_mhz"]
    display = np.abs(detuning_mhz) <= DISPLAY_HALF_WIDTH_MHZ

    fig = plt.figure(figsize=(3.35, 1.72), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 0.045), wspace=0.08)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    color_axis = fig.add_subplot(grid[0, 2])
    image = None

    comparisons = (
        (
            axes[0],
            data["plain_pe"][:, display],
            data["plain_centers_mhz"],
            "(a) Uncorrected",
        ),
        (
            axes[1],
            data["corrected_pe"][:, display],
            data["corrected_centers_mhz"],
            "(b) AC-Stark corrected",
        ),
    )
    for axis, population, centers_mhz, title in comparisons:
        image = axis.pcolormesh(
            detuning_mhz[display],
            rabi_mhz,
            population,
            shading="auto",
            cmap="magma",
            vmin=COLOR_MIN,
            vmax=COLOR_MAX,
            rasterized=True,
        )
        axis.axvline(0.0, color="white", lw=0.55, ls="--", alpha=0.9)
        for bound in (-T2_LIMIT_HALF_WIDTH_MHZ, T2_LIMIT_HALF_WIDTH_MHZ):
            axis.axvline(bound, color="white", lw=0.65, ls=":", alpha=0.95)
        center_line = axis.plot(
            centers_mhz,
            fit_rabi_mhz,
            color="white",
            lw=0.9,
        )[0]
        center_line.set_path_effects(
            [path_effects.Stroke(linewidth=1.6, foreground="black"), path_effects.Normal()]
        )
        axis.set(
            title=title,
            xlabel=r"$\Delta/2\pi$ (MHz)",
            xlim=(-DISPLAY_HALF_WIDTH_MHZ, DISPLAY_HALF_WIDTH_MHZ),
            ylim=(0.0, float(rabi_mhz.max())),
        )

    axes[0].set_ylabel(r"$\Omega_0/2\pi$ (MHz)")
    axes[1].tick_params(labelleft=False)
    if image is None:
        raise RuntimeError("No map was rendered.")
    colorbar = fig.colorbar(image, cax=color_axis)
    colorbar.set_label(r"$P_e$")

    figure_paths = save_figure(
        fig,
        OUTPUT_STEM,
        variant=FigureVariant.PAPER,
        formats=("pdf", "png", "svg"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)
    for path in (*figure_paths, *paper_data_paths):
        print(path)


if __name__ == "__main__":
    main()
