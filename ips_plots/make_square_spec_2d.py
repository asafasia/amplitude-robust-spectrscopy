from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np

from bundle_utils import amp_prefactor_to_rabi_amp_mhz, extract_qubit_variables
from presentation_style import polish_axes, use_presentation_style


use_presentation_style()

PROJECT_DIR = Path.cwd()
DATA_DIR = PROJECT_DIR / "data" / "broad_sqaure"
FIGURES_DIR = PROJECT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_sweeps():
    bundle_paths = sorted(DATA_DIR.glob("*_data_bundle.npz"))
    if len(bundle_paths) < 2:
        raise FileNotFoundError(f"Expected at least two data bundles in {DATA_DIR}")

    sweeps = []
    for path in bundle_paths:
        data = extract_qubit_variables(path)
        frequency_span_mhz = data.parameters.get("frequency_span_in_mhz", np.nan)
        peak_amplitude = data.parameters.get("lorentzian_peak_amplitude", 1.0)
        effective_amp = data.amp_prefactor * peak_amplitude
        rabi_amp_mhz = amp_prefactor_to_rabi_amp_mhz(
            effective_amp,
            data.pi_pulse["amplitude"],
            data.pi_pulse["length_ns"],
        )
        sweeps.append(
            {
                "path": path,
                "data": data,
                "frequency_span_mhz": frequency_span_mhz,
                "peak_amplitude": peak_amplitude,
                "rabi_amp_mhz": rabi_amp_mhz,
            }
        )

    sweeps = sorted(sweeps, key=lambda sweep: sweep["frequency_span_mhz"], reverse=True)
    for index, sweep in enumerate(sweeps):
        sweep["label"] = "Broad square" if index == 0 else "Square zoom" if index == 1 else f"Sweep {index + 1}"
    return sweeps


def set_detuning_ticks(ax, detuning_mhz):
    x_min = float(np.min(detuning_mhz))
    x_max = float(np.max(detuning_mhz))
    if np.isclose(x_min, -50) and np.isclose(x_max, 50):
        ax.set_xticks(np.arange(-50, 51, 10))
    elif np.isclose(x_min, -0.5) and np.isclose(x_max, 0.5):
        ax.set_xticks(np.arange(-0.5, 0.51, 0.25))


def draw_sweep(ax, sweep, vmin, vmax, t2_limit_half_width_mhz=None):
    data = sweep["data"]
    detuning_mhz = data.detuning_hz / 1e6
    mesh = ax.pcolormesh(
        detuning_mhz,
        sweep["rabi_amp_mhz"],
        data.result.T,
        shading="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    ax.text(
        0.03,
        0.95,
        sweep["label"],
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=12,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.35, "edgecolor": "none", "pad": 4},
    )
    if t2_limit_half_width_mhz is not None:
        ax.axvline(-t2_limit_half_width_mhz, color="white", linestyle="--", linewidth=1.1, alpha=0.9, zorder=6)
        ax.axvline(t2_limit_half_width_mhz, color="white", linestyle="--", linewidth=1.1, alpha=0.9, zorder=6)
        ax.axvspan(-t2_limit_half_width_mhz, t2_limit_half_width_mhz, color="white", alpha=0.08, zorder=4)
    ax.set_xlabel("Detuning (MHz)")
    ax.set_ylabel("Effective Rabi amplitude (MHz)")
    ax.set_xlim(float(np.min(detuning_mhz)), float(np.max(detuning_mhz)))
    ax.set_ylim(float(np.min(sweep["rabi_amp_mhz"])), float(np.max(sweep["rabi_amp_mhz"])))
    set_detuning_ticks(ax, detuning_mhz)
    polish_axes(ax)
    return mesh


def save_figure(fig, figure_stem):
    png_path = figure_stem.with_suffix(".png")
    pdf_path = figure_stem.with_suffix(".pdf")
    fig.canvas.draw()
    fig.savefig(png_path, dpi=300, bbox_inches=None, facecolor="white")
    fig.savefig(pdf_path, bbox_inches=None, facecolor="white")
    print(f"Saved PNG: {png_path.resolve()}")
    print(f"Saved PDF: {pdf_path.resolve()}")


def main():
    sweeps = load_sweeps()
    qubit_name = sweeps[-1]["data"].qubit_name
    t2_ramsey_ns = sweeps[-1]["data"].qubit_profile["transmon"]["t2_ramsey_ns"] / 4
    t2_limit_fwhm_mhz = 1000 / (np.pi * t2_ramsey_ns)
    t2_limit_half_width_mhz = t2_limit_fwhm_mhz / 2

    vmin = min(float(np.nanmin(sweep["data"].result)) for sweep in sweeps)
    vmax = max(float(np.nanmax(sweep["data"].result)) for sweep in sweeps)

    fig = plt.figure(figsize=(12.0, 4.8))
    axes = [
        fig.add_axes([0.07, 0.16, 0.39, 0.78]),
        fig.add_axes([0.53, 0.16, 0.39, 0.78]),
    ]
    ax = axes[0]
    axes[1].set_axis_off()
    cax = fig.add_axes([0.955, 0.18, 0.018, 0.74])
    mesh = draw_sweep(ax, sweeps[0], vmin, vmax)
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label("Excitation", fontsize=12)
    cbar.outline.set_visible(False)
    save_figure(fig, FIGURES_DIR / f"00_wide_spec_{qubit_name}_broad_square_broad_only")
    plt.close(fig)

    fig = plt.figure(figsize=(12.0, 4.8))
    axes = [
        fig.add_axes([0.07, 0.16, 0.39, 0.78]),
        fig.add_axes([0.53, 0.16, 0.39, 0.78]),
    ]
    cax = fig.add_axes([0.955, 0.18, 0.018, 0.74])

    mesh = None
    for index, (ax, sweep) in enumerate(zip(axes, sweeps)):
        marker_width = t2_limit_half_width_mhz if index == 1 else None
        mesh = draw_sweep(ax, sweep, vmin, vmax, marker_width)

    big_sweep = sweeps[0]
    zoom_sweep = sweeps[1]
    zoom_x_min = float(np.min(zoom_sweep["data"].detuning_hz / 1e6))
    zoom_x_max = float(np.max(zoom_sweep["data"].detuning_hz / 1e6))
    zoom_y_min = float(np.min(zoom_sweep["rabi_amp_mhz"]))
    zoom_y_max = float(np.max(zoom_sweep["rabi_amp_mhz"]))

    axes[0].add_patch(
        plt.Rectangle(
            (zoom_x_min, zoom_y_min),
            zoom_x_max - zoom_x_min,
            zoom_y_max - zoom_y_min,
            fill=False,
            edgecolor="#d62728",
            linewidth=2.0,
            zorder=6,
        )
    )

    for rect_y, zoom_y in [(zoom_y_min, zoom_y_min), (zoom_y_max, zoom_y_max)]:
        connector = ConnectionPatch(
            xyA=(zoom_x_max, rect_y),
            coordsA=axes[0].transData,
            xyB=(zoom_x_min, zoom_y),
            coordsB=axes[1].transData,
            color="#d62728",
            linewidth=1.4,
            alpha=0.9,
            zorder=7,
            clip_on=False,
        )
        fig.add_artist(connector)

    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label("Excitation", fontsize=12)
    cbar.outline.set_visible(False)
    save_figure(fig, FIGURES_DIR / f"00_wide_spec_{qubit_name}_broad_square_two_sweeps")
    plt.close(fig)

    print(f"T2-limited FWHM: {t2_limit_fwhm_mhz:.5f} MHz")
    for sweep in sweeps:
        print(f"{sweep['label']}: {sweep['path'].resolve()}")


if __name__ == "__main__":
    main()
