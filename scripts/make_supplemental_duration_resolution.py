"""Render the split-column supplemental S3 from the saved figure arrays.

This leaves the source data and TeX untouched; preview mode also protects S3.
Run with PYTHONPATH=. MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/ars-mpl-cache
.venv/bin/python scripts/make_supplemental_figures.py --only duration_metrics
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from echospec.figures.coherence import REFERENCE_FWHM_HZ
from echospec.figures.supplemental import (
    FIGURE_WIDTH,
    STYLE_PATH,
    apply_supplemental_style,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "figures/paper/03_echo_duration_resolution_comparison.npz"
OUT = ROOT / "figures/paper/supplemental"
STEM = "s03_duration_resolution"
COLORS = ("#0072b2", "#6a1b9a", "#cc79a7", "#d55e00", "#009e73")
MARKERS = ("o", "s", "P", "^", "D")
REFERENCE_COLOR = "0.25"
THRESHOLD = 0.10
METRICS = ("fwhm_t2_units", "contrast", "product")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def threshold_intervals(x: np.ndarray, y: np.ndarray) -> list[list[float]]:
    """Crossings of straight plotted segments in log-x coordinates."""
    selected = np.flatnonzero(np.isfinite(y) & (y >= THRESHOLD))
    if selected.size == 0:
        return []

    def crossing(i: int, j: int) -> float:
        fraction = (THRESHOLD - y[i]) / (y[j] - y[i])
        return float(np.exp(np.log(x[i]) + fraction * np.log(x[j] / x[i])))

    intervals = []
    for group in np.split(selected, np.flatnonzero(np.diff(selected) > 1) + 1):
        first, last = int(group[0]), int(group[-1])
        low, high = float(x[first]), float(x[last])
        if first > 0 and np.isfinite(y[first - 1]):
            low = crossing(first - 1, first)
        if last + 1 < x.size and np.isfinite(y[last + 1]):
            high = crossing(last, last + 1)
        intervals.append([low, high])
    return intervals


def render(output_dir: Path = OUT, stem: str = STEM, *, preview: bool = False) -> None:
    """Write the figure, displayed arrays, and provenance to the chosen directory."""
    protected = [SOURCE, *sorted((ROOT / "paper").rglob("*.tex"))]
    if preview:
        for directory in (
            ROOT / "figures/paper/supplemental",
            ROOT / "paper/figures/generated/supplemental",
        ):
            protected.extend(directory.glob("s03_duration_resolution.*"))
    original_hashes = {path: digest(path) for path in protected}
    with np.load(SOURCE, allow_pickle=False) as saved:
        data = {key: saved[key].copy() for key in saved.files}
    width_scale = float(data["t2_limit_fwhm_mhz"]) / (REFERENCE_FWHM_HZ / 1e6)
    factors = dict(fwhm_t2_units=width_scale, contrast=1.0, product=1 / width_scale)
    durations = data["durations_us"]
    plotted = {"rabi_mhz": data["rabi_mhz"],
               "constant_rabi_mhz": data["constant_rabi_mhz"]}
    intervals = {}
    apply_supplemental_style()
    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.55,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
    })
    fig, axes = plt.subplots(
        3, 2, figsize=(FIGURE_WIDTH, 4.4), sharex="col", sharey="row",
        gridspec_kw={"width_ratios": [1, 3.25]},
    )
    fig.subplots_adjust(left=0.085, right=0.985, top=0.985, bottom=0.115,
                        hspace=0.17, wspace=0.12)
    for row, metric in enumerate(METRICS):
        key = f"constant_{metric}"
        plotted[key] = data[key] * factors[metric]
        axes[row, 0].plot(data["constant_rabi_mhz"] * 1000, plotted[key],
                          color=REFERENCE_COLOR, lw=1.0)
    intervals["constant"] = threshold_intervals(
        data["constant_rabi_mhz"], plotted["constant_product"])

    for duration, color, marker in zip(durations, COLORS, MARKERS, strict=True):
        suffix = f"{int(duration)}us"
        x = data[f"measured_rabi_mhz_{suffix}"]
        plotted[f"measured_rabi_mhz_{suffix}"] = x
        for row, metric in enumerate(METRICS):
            ax = axes[row, 1]
            clean_key, measured_key = (f"{prefix}_{metric}_{suffix}"
                                       for prefix in ("clean", "measured"))
            clean, measured = (data[key] * factors[metric]
                               for key in (clean_key, measured_key))
            plotted[clean_key], plotted[measured_key] = clean, measured
            valid = np.isfinite(measured) & (x > 0)
            ax.plot(x, np.where(valid, measured, np.nan), ls="-", lw=0.45,
                    marker=marker, ms=1.8, color=color, alpha=0.48,
                    mec="none", zorder=2)
            ax.plot(data["rabi_mhz"], clean, color=color, lw=0.95, zorder=3)
        intervals[suffix] = threshold_intervals(
            data["rabi_mhz"], plotted[f"clean_product_{suffix}"])

    limits = ((0, 7.2), (0, 0.82), (0, 0.24))
    ticks = ([0, 2, 4, 6], [0, 0.2, 0.4, 0.6, 0.8], [0, 0.1, 0.2])
    labels = (r"$\mathrm{FWHM}/\Gamma_{T_2^*}$", "Contrast",
              r"$Q=C\,\Gamma_{T_2^*}/\mathrm{FWHM}$")
    for row in range(3):
        for col in range(2):
            ax = axes[row, col]
            ax.set(xscale="log", ylim=limits[row], yticks=ticks[row])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
            if row in (0, 2):
                ax.axhline(1 if row == 0 else THRESHOLD, color="0.5",
                           ls="--", lw=0.55, zorder=0)
            if col == 0:
                ax.set(xlim=(1, 100), ylabel=labels[row])
                ax.xaxis.set_major_locator(FixedLocator([1, 10, 100]))
            else:
                ax.set_xlim(0.1, 20)
                ax.xaxis.set_major_locator(FixedLocator([0.1, 0.3, 1, 3, 10, 20]))
        axes[row, 0].text(0.045, 0.94, f"({chr(97 + row)})",
                          transform=axes[row, 0].transAxes, va="top", fontsize=8)
    axes[2, 0].set_xlabel(r"$\Omega_0/(2\pi)$ (kHz)")
    axes[2, 1].set_xlabel(r"$\Omega_0/(2\pi)$ (MHz)")

    axes[0, 0].text(0.29, 0.94, "Constant drive",
                    transform=axes[0, 0].transAxes, va="top", fontsize=7)
    duration_handles = [
        Line2D([], [], color=color, lw=0.95, marker=marker, ms=2.4, mec="none",
               label=rf"${duration:g}\,\mu$s")
        for duration, color, marker in zip(durations, COLORS, MARKERS, strict=True)
    ]
    axes[0, 1].legend(handles=duration_handles, loc="upper left", ncol=5,
                       fontsize=7, handlelength=1.4, columnspacing=1.0,
                       handletextpad=0.4, borderaxespad=0.6)
    axes[1, 1].legend(handles=[
        Line2D([], [], color="0.25", lw=0.95, label="Simulation"),
        Line2D([], [], color="0.25", ls="-", lw=0.45, marker="o", ms=2.1,
               alpha=0.5, label="Experiment"),
    ], loc="upper left", ncol=2, fontsize=7, handlelength=1.5,
        handletextpad=0.4, columnspacing=1.3, borderaxespad=0.6)

    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(output_dir / f"{stem}.{extension}", dpi=300)
    plt.close(fig)
    arrays_path = output_dir / f"{stem}.npz"
    np.savez_compressed(arrays_path, **plotted)
    metadata = {
        "status": "Preview only" if preview else "Current supplemental Figure S3",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": original_hashes[SOURCE],
        "generator_sha256": digest(Path(__file__)),
        "style_sha256": digest(STYLE_PATH),
        "arrays_sha256": digest(arrays_path),
        "reference_fwhm_hz": REFERENCE_FWHM_HZ,
        "source_width_rescaling": width_scale,
        "axes": {"left_khz": [1, 100], "right_mhz": [0.1, 20],
                 "logarithmic": True, "shared_y_limits_by_row": limits},
        "labels": {"metrics": labels, "contrast_symbol": "C",
                   "drive": r"\Omega_0/(2\pi)"},
        "q_threshold": THRESHOLD,
        "presentation": "Compact publication layout; no titles or interval footer; small in-panel legends; connected translucent measurement markers with thinner lines than simulations",
        "simulated_q_intervals_mhz": intervals,
        "interval_method": "Crossings linearly interpolated in log-amplitude, matching plotted segments",
        "processing": "Saved fitted metrics rescaled to the common linewidth reference; thin lines connect adjacent accepted measurements, leaving gaps at invalid samples; no new fitting, smoothing, or measurement subsampling",
        "reproduce": (
            "PYTHONPATH=. MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/ars-mpl-cache "
            ".venv/bin/python scripts/preview_supplemental_duration_resolution.py"
            if preview else
            "PYTHONPATH=. MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/ars-mpl-cache "
            ".venv/bin/python scripts/make_supplemental_figures.py --only duration_metrics"
        ),
    }
    (output_dir / f"{stem}_provenance.json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    for path, old_digest in original_hashes.items():
        assert digest(path) == old_digest, f"Changed protected file: {path}"
    print(f"Figure saved to {output_dir}")
    print("Source data and manuscript TeX unchanged.")


if __name__ == "__main__":
    render()
