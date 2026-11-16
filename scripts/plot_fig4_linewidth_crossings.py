"""Explain the Figure 4 linewidth jump using saved, refined spectra.

PYTHONPATH=. MPLBACKEND=Agg .venv/bin/python scripts/plot_fig4_linewidth_crossings.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.signal import find_peaks

from echospec.analysis.cutoff_metrics import cutoff_metrics
from echospec.figures.coherence import REFERENCE_FWHM_HZ
from scripts.diagnose_fig4_50us_linewidth import OUT, load_case, profile_details

ROOT = Path(__file__).resolve().parents[1]
STEM = "linewidth_crossings_explained"
COLORS = ("#0072b2", "#d55e00")


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 10,
        "axes.titlesize": 11, "axes.linewidth": 0.7, "axes.grid": False,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
    })
    fig, axes = plt.subplots(
        2, 2, figsize=(10, 6.8), sharex="row", sharey="row",
        gridspec_kw={"height_ratios": [1.25, 1]},
    )
    fig.subplots_adjust(left=0.085, right=0.985, top=0.86, bottom=0.14,
                        wspace=0.14, hspace=0.38)
    fig.suptitle("Why the reported width drops", y=0.98, fontsize=14)
    fig.text(0.5, 0.936,
             r"Actual simulated spectra: $L=50\,\mu$s, "
             r"$\Omega_0/(2\pi)=49.77$ MHz",
             ha="center", fontsize=10, color="0.3")
    arrays, records, hashes = {}, [], {}
    for col, (name, color) in enumerate(zip(("fine_00", "fine_12"), COLORS, strict=True)):
        x, y, config = load_case(name)
        details = profile_details(x, y)
        metrics = cutoff_metrics(x, y)
        half = details["background"] - details["contrast"] / 2
        selected = np.asarray(details["nearest_pair_mhz"])
        crossings = np.asarray(details["crossings_mhz"])
        width = np.diff(selected)[0] * 1e6 / REFERENCE_FWHM_HZ
        np.testing.assert_allclose(width, metrics["fwhm_hz"] / REFERENCE_FWHM_HZ)
        assert len(crossings) == (2 if col == 0 else 6)
        title = "Before: outer crossings" if col == 0 else "After: inner crossings"
        top = axes[0, col]
        top.set_title(title + rf"  ($c={config['pulse']['cutoff']:.5f}$)",
                      loc="left", color=color, pad=10)
        top.plot(x * 1000, y, color=color, lw=2)
        top.plot(x * 1000, half, color="0.3", ls="--", lw=1)
        top.annotate("Half-depth level",
                     xy=(-190, float(np.interp(-.19, x, half))),
                     xytext=(-211, .402), fontsize=9, color="0.3",
                     arrowprops={"arrowstyle": "->", "color": "0.3", "lw": .8})
        top.axvspan(*selected * 1000, color=color, alpha=0.07, zorder=0)
        top.scatter(crossings * 1000, np.interp(crossings, x, half),
                    s=36, facecolors="white", edgecolors="0.55", zorder=5)
        top.scatter(selected * 1000, np.interp(selected, x, half),
                    s=44, color=color, zorder=6)
        arrow_y = .354
        for crossing in selected:
            top.plot([crossing * 1000] * 2,
                     [arrow_y, float(np.interp(crossing, x, half))],
                     ls=":", color=color, lw=0.9)
        top.annotate("", xy=(selected[1] * 1000, arrow_y),
                     xytext=(selected[0] * 1000, arrow_y),
                     arrowprops={"arrowstyle": "<->", "color": color, "lw": 1.5})
        top.text(0, .3495, rf"Reported FWHM / $\Gamma_{{T_2^*}}$ = {width:.2f}",
                 ha="center", va="center", color=color, fontsize=10)
        top.set(xlim=(-220, 220), ylim=(.345, .408),
                xticks=[-200, -100, 0, 100, 200], yticks=[.36, .38, .40])
        if col == 0:
            top.set_ylabel(r"Excited population $P_e$")
        top.set_xlabel(r"Detuning $\Delta/(2\pi)$ (kHz)")

        zoom = axes[1, col]
        residual = (y - half) * 1000
        zoom.plot(x * 1000, residual, color=color, lw=2)
        zoom.axhline(0, color="0.3", ls="--", lw=1)
        zoom.scatter(crossings * 1000, np.zeros(len(crossings)),
                     s=40, facecolors="white", edgecolors="0.55", zorder=5)
        zoom.scatter(selected * 1000, [0, 0], s=48, color=color, zorder=6)
        zoom.set(xlim=(55, 180), ylim=(-2.6, 1.6), xticks=[60, 90, 120, 150, 180],
                 yticks=[-2, -1, 0, 1], xlabel=r"Detuning $\Delta/(2\pi)$ (kHz)")
        zoom.set_title("Close-up of the small bump on the right", loc="left",
                       fontsize=10, pad=9)
        if col == 0:
            zoom.set_ylabel("Signal minus half-depth level\n"
                            r"($10^{-3}$ in $P_e$)")
        peaks, _ = find_peaks(y)
        peak = peaks[(x[peaks] > .05) & (x[peaks] < .12)][0]
        above = "below" if col == 0 else "above"
        zoom.annotate(
            f"Bump stays {above} level" if col == 0 else "Bump crosses above level",
            xy=(x[peak] * 1000, residual[peak]), xytext=(116, 1.05),
            ha="center", va="center", fontsize=9, color=color,
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1},
        )
        crossing = selected[1] * 1000
        text_x = 137 if col == 0 else 85
        zoom.annotate(
            "First crossing:\nfar from center" if col == 0 else
            "New first crossing:\ncloser to center",
            xy=(crossing, 0), xytext=(text_x, -2.05),
            ha="center", va="center", fontsize=9, color=color,
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1},
        )
        for ax in axes[:, col]:
            ax.tick_params(direction="in", top=True, right=True)
        arrays.update({f"{name}_{key}": value for key, value in {
            "detuning_mhz": x, "population": y, "half_depth_level": half,
            "all_crossings_mhz": crossings, "selected_pair_mhz": selected,
        }.items()})
        records.append(dict(case=name, cutoff=config["pulse"]["cutoff"],
                            normalized_width=width, crossings_mhz=crossings.tolist(),
                            selected_pair_mhz=selected.tolist()))
        path = OUT / f"{name}.npz"
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    handles = [
        Line2D([], [], marker="o", color="none", markerfacecolor="0.25",
               markeredgecolor="0.25", label="Filled dots: crossings used for width"),
        Line2D([], [], marker="o", color="none", markerfacecolor="white",
               markeredgecolor="0.55", label="Open dots: other crossings"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.53, .018),
               frameon=False, fontsize=9, ncol=2, handletextpad=.4)
    fig.savefig(OUT / f"{STEM}.png", dpi=220, facecolor="white")
    plt.close(fig)
    np.savez_compressed(OUT / f"{STEM}.npz", **arrays)
    helper = ROOT / "scripts/diagnose_fig4_50us_linewidth.py"
    hashes[str(helper.relative_to(ROOT))] = hashlib.sha256(helper.read_bytes()).hexdigest()
    (OUT / f"{STEM}_provenance.json").write_text(json.dumps(dict(
        generator=str(Path(__file__).resolve().relative_to(ROOT)),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256=hashes, reference_linewidth_hz=REFERENCE_FWHM_HZ,
        cases=records, note="Actual cached spectra. No simulation, smoothing, or curve edits. "
                           "Top row shows excitation probability; bottom row subtracts each "
                           "spectrum's own half-depth level and magnifies the vertical scale.",
    ), indent=2) + "\n")
    print(OUT / f"{STEM}.png")


if __name__ == "__main__":
    main()
