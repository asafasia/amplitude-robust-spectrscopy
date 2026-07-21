"""Publication-aware dashboard plotting for a single selected cutoff."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from .aggregate import AggregateCampaign, selected_cutoff_rows
from .analysis import CutoffAnalysis, best_amplitude_index
from .opx1000_fwhm import _gaussian_with_linear_baseline, _robust_linear_baseline


def _selected_gaussian_curve(
    analysis: CutoffAnalysis,
    amplitude_index: int,
) -> np.ndarray:
    raw = analysis.raw
    x = raw.detuning_hz
    y = raw.state[:, amplitude_index]
    center = analysis.center_hz[amplitude_index]
    fwhm = analysis.fwhm_hz[amplitude_index]
    amplitude = float(
        analysis.dataset.gaussian_fit_amplitude.sel(qubit=raw.qubit).values[
            amplitude_index
        ]
    )
    if not np.all(np.isfinite([center, fwhm, amplitude])):
        return np.full_like(x, np.nan, dtype=float)
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    baseline = _robust_linear_baseline(x, y)
    offset = float(np.mean(baseline))
    scale = np.ptp(x)
    slope = float(baseline[-1] - baseline[0]) if scale > 0 else 0.0
    return _gaussian_with_linear_baseline(x, offset, slope, amplitude, center, sigma)


def plot_cutoff_dashboard(
    analysis: CutoffAnalysis,
    amplitude_index: int | None = None,
    *,
    resolution_vmin: float = 0.1,
    resolution_vmax: float = 1.0,
):
    """Plot the raw map, selected OPX fit, FWHM, and resolution metrics."""
    raw = analysis.raw
    if amplitude_index is None:
        amplitude_index = best_amplitude_index(analysis)
    valid = analysis.valid
    detuning_mhz = raw.detuning_hz / 1e6
    fwhm_mhz = analysis.fwhm_hz / 1e6

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2), constrained_layout=True)
    ax_map, ax_trace, ax_fwhm, ax_metric = axes.ravel()

    image = ax_map.pcolormesh(
        raw.rabi_mhz,
        detuning_mhz,
        raw.state,
        shading="auto",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    ax_map.axvline(raw.rabi_mhz[amplitude_index], color="white", ls="--", lw=1.4)
    fig.colorbar(image, ax=ax_map, label="measured excited-state probability")
    ax_map.set(
        title=f"Raw 2D sweep | cutoff = {float(raw.parameters['cutoff']):.4g}",
        xlabel="peak Rabi frequency (MHz)",
        ylabel=r"drive detuning $f_d-f_{01}$ (MHz)",
    )

    y = raw.state[:, amplitude_index]
    fit_curve = _selected_gaussian_curve(analysis, amplitude_index)
    ax_trace.plot(detuning_mhz, y, ".", ms=4, alpha=0.55, label="measured state")
    if np.isfinite(fit_curve).any():
        ax_trace.plot(detuning_mhz, fit_curve, lw=2, label="selected OPX Gaussian")
        center = analysis.center_hz[amplitude_index] / 1e6
        half_width = fwhm_mhz[amplitude_index] / 2
        ax_trace.axvspan(
            center - half_width, center + half_width, color="tab:orange", alpha=0.15
        )
        ax_trace.axvline(center - half_width, color="tab:orange", ls="--")
        ax_trace.axvline(center + half_width, color="tab:orange", ls="--")
    width = fwhm_mhz[amplitude_index]
    width_label = f"{width:.4g} MHz" if np.isfinite(width) else "screened"
    ax_trace.set(
        title=f"OPX1000 fit | Rabi = {raw.rabi_mhz[amplitude_index]:.3g} MHz | FWHM = {width_label}",
        xlabel=r"drive detuning $f_d-f_{01}$ (MHz)",
        ylabel="state",
    )
    ax_trace.legend(fontsize=8)

    norm = LogNorm(vmin=resolution_vmin, vmax=resolution_vmax)
    points = ax_fwhm.scatter(
        raw.rabi_mhz[valid],
        fwhm_mhz[valid],
        c=analysis.resolution[valid],
        s=24,
        cmap="viridis",
        norm=norm,
    )
    ax_fwhm.axhline(raw.t2_limit_hz / 1e6, color="black", ls="--", label="1/(pi T2)")
    ax_fwhm.set_yscale("log")
    ax_fwhm.set(
        title=(
            "FWHM recalculated from raw traces | "
            f"accepted {np.count_nonzero(valid)}/{len(valid)}"
        ),
        xlabel="peak Rabi frequency (MHz)",
        ylabel="Gaussian FWHM (MHz, log)",
    )
    ax_fwhm.legend(fontsize=8)
    fig.colorbar(points, ax=ax_fwhm, label="resolution = [1/(pi T2)] / FWHM")

    ax_metric.plot(
        raw.rabi_mhz[valid], analysis.resolution[valid], ".-", label="resolution"
    )
    ax_metric.plot(
        raw.rabi_mhz[valid],
        analysis.signal_resolution[valid],
        ".-",
        label="fit amplitude x resolution",
    )
    ax_metric.axhline(1, color="black", ls="--", lw=1, label="T2 limit")
    ax_metric.set_yscale("log")
    ax_metric.set_ylim(bottom=0.003)
    ax_metric.set(
        title="Resolution metrics",
        xlabel="peak Rabi frequency (MHz)",
        ylabel="metric (log)",
    )
    ax_metric.legend(fontsize=8)

    fig.suptitle(
        f"{raw.qubit} | T2 reference {raw.t2_us:.3g} us ({raw.t2_reference_name}) | "
        "OPX1000 Gaussian pipeline",
        fontsize=12,
    )
    return fig, axes


def plot_aggregate_cutoff(
    campaign: AggregateCampaign,
    cutoff: float,
    *,
    resolution_vmin: float = 0.1,
    resolution_vmax: float = 1.0,
):
    """Show the retained 2D map and quality-screened stored OPX fit metrics."""
    rows, accepted = selected_cutoff_rows(campaign, cutoff)
    resolution = 1.0 / rows["fwhm_t2_units"]
    signal_resolution = rows["fit_abs_amplitude"] * resolution

    figure = plt.figure(figsize=(14, 11), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(2.3, 1.0))
    ax_map = figure.add_subplot(grid[0, :])
    ax_fwhm = figure.add_subplot(grid[1, 0])
    ax_metric = figure.add_subplot(grid[1, 1])

    ax_map.imshow(plt.imread(campaign.figures[float(cutoff)]))
    ax_map.set_axis_off()
    ax_map.set_title(
        f"Retained measured 2D sweep | cutoff = {cutoff:.4g}",
        fontsize=12,
    )

    finite_width = np.isfinite(rows["fwhm_mhz"]) & (rows["fwhm_mhz"] > 0)
    rejected = finite_width & ~accepted
    ax_fwhm.scatter(
        rows.loc[rejected, "rabi_frequency_mhz"],
        rows.loc[rejected, "fwhm_mhz"],
        marker="x",
        color="0.72",
        s=22,
        label="screened stored fit",
    )
    points = ax_fwhm.scatter(
        rows.loc[accepted, "rabi_frequency_mhz"],
        rows.loc[accepted, "fwhm_mhz"],
        c=resolution[accepted],
        cmap="viridis",
        norm=LogNorm(vmin=resolution_vmin, vmax=resolution_vmax),
        s=32,
        label="accepted stored fit",
    )
    t2_limit_mhz = float(rows["t2_fwhm_limit_hz"].dropna().iloc[0]) / 1e6
    ax_fwhm.axhline(t2_limit_mhz, color="black", ls="--", label="1/(pi T2)")
    ax_fwhm.set_yscale("log")
    ax_fwhm.set(
        title=f"Stored OPX FWHM | accepted {int(accepted.sum())}/{len(rows)}",
        xlabel="Rabi frequency (MHz)",
        ylabel="FWHM (MHz, log)",
    )
    ax_fwhm.legend(fontsize=8)
    figure.colorbar(points, ax=ax_fwhm, label="resolution = [1/(pi T2)] / FWHM")

    ax_metric.plot(
        rows.loc[accepted, "rabi_frequency_mhz"],
        resolution[accepted],
        ".-",
        label="resolution",
    )
    ax_metric.plot(
        rows.loc[accepted, "rabi_frequency_mhz"],
        signal_resolution[accepted],
        ".-",
        label="fit amplitude x resolution",
    )
    ax_metric.axhline(1, color="black", ls="--", lw=1, label="T2 limit")
    ax_metric.set_yscale("log")
    ax_metric.set_ylim(bottom=0.003)
    ax_metric.set(
        title="Quality-screened resolution metrics",
        xlabel="Rabi frequency (MHz)",
        ylabel="metric (log)",
    )
    ax_metric.legend(fontsize=8)
    return figure, (ax_map, ax_fwhm, ax_metric)
