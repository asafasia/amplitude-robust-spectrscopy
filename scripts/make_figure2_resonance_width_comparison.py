"""Standalone position/width comparison from the narrow Figure 2 maps.

The existing dip_metrics convention is also applied to inverted root spectra
for dominant-peak extraction. This is a local feature metric, not a fit to a
unique transition frequency in an oscillatory spectrum. No paper assets change.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from echospec.analysis.dip_metrics import dip_metrics
from echospec.figures import FigureVariant, apply_figure_style

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "figures/paper/readout_preview"
SOURCE = OUTPUT / "02_readout_matrix_preview.npz"
STEM = "02_resonance_fwhm_comparison"
PROTOCOLS = (("root", "Root-Lorentzian"), ("echo", "Echo-root-Lorentzian"))


def main():
    with np.load(SOURCE, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    source_provenance = json.loads(
        (OUTPUT / "02_readout_matrix_preview_provenance.json").read_text()
    )
    coherence_path = ROOT / "paper/coherence_parameters.tex"
    match = re.search(r"\\newcommand\{\\EffectiveTTwo\}\{([\d.]+)\}",
                      coherence_path.read_text())
    if match is None:
        raise ValueError("Effective T2 is missing")
    t2_us = float(match.group(1))
    gamma_khz = 1000 / (np.pi * t2_us)
    rows, series, checks = [], {}, {}
    for protocol, _ in PROTOCOLS:
        sign = -1 if protocol == "root" else 1
        for kind in ("raw", "corrected", "simulation"):
            key = f"narrow_{protocol}_{'simulation' if kind == 'simulation' else 'experiment'}"
            x = arrays[f"{key}_detuning_mhz"]
            drive = arrays[f"{key}_rabi_mhz"]
            suffix = "raw_population" if kind == "raw" else "population"
            population = arrays[f"{key}_{suffix}"]
            metrics = [dip_metrics(x, sign * trace) for trace in population]
            series[(protocol, kind)] = {
                "drive_mhz": drive,
                "center_khz": np.array([r["center_hz"] / 1000 for r in metrics]),
                "fwhm_khz": np.array([r["fwhm_hz"] / 1000 for r in metrics]),
                "accepted": np.array([r["accepted"] for r in metrics]),
            }
            for amplitude, metric in zip(drive, metrics, strict=True):
                rows.append({
                    "protocol": protocol, "series": kind,
                    "drive_mhz": float(amplitude), **metric,
                    "fwhm_over_gamma_t2": metric["fwhm_hz"] / (1000 * gamma_khz),
                    "displayed": bool(metric["accepted"] and amplitude >= 1),
                })
        raw, corrected = series[(protocol, "raw")], series[(protocol, "corrected")]
        np.testing.assert_array_equal(raw["accepted"], corrected["accepted"])
        checks[protocol] = {}
        for field in ("center_khz", "fwhm_khz"):
            np.testing.assert_allclose(raw[field], corrected[field], atol=1e-9, rtol=0)
            checks[protocol][f"maximum_correction_change_{field}"] = float(
                np.nanmax(np.abs(raw[field] - corrected[field]))
            )

    apply_figure_style(FigureVariant.PAPER)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.7,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.6), sharex=True)
    fig.subplots_adjust(left=0.10, right=0.91, bottom=0.20, top=0.84,
                        wspace=0.63, hspace=0.38)
    colors = {"raw": "#2878a0", "corrected": "#d5911c", "simulation": "#b63843"}
    for col, (protocol, title) in enumerate(PROTOCOLS):
        for row, field in enumerate(("center_khz", "fwhm_khz")):
            ax = axes[row, col]
            for kind in ("simulation", "raw", "corrected"):
                data = series[(protocol, kind)]
                index = np.flatnonzero(data["accepted"] & (data["drive_mhz"] >= 1))
                if kind == "corrected":
                    # All corrected points are saved in CSV. Sparse open rings
                    # make coincident raw/corrected results visible without clutter.
                    index = index[::8]
                    ax.plot(data["drive_mhz"][index], data[field][index],
                            linestyle="none", marker="o", ms=4,
                            markerfacecolor="none", markeredgewidth=0.7,
                            color=colors[kind], zorder=4)
                else:
                    ax.plot(data["drive_mhz"][index], data[field][index],
                            linestyle="none", marker="x" if kind == "simulation" else ".",
                            ms=2.8 if kind == "simulation" else 2.3,
                            markeredgewidth=0.65, color=colors[kind],
                            alpha=0.85, zorder=3 if kind == "raw" else 2)
            ax.set_xlim(0, 43)
            ax.set_xticks([0, 10, 20, 30, 40])
            ax.tick_params(direction="in", top=True, right=True, length=3)
            ax.text(0.035, 0.94, f"({chr(97 + row * 2 + col)})",
                    transform=ax.transAxes, va="top", fontsize=8)
            if row == 0:
                ax.set_title(title, pad=9)
                ax.axhline(0, color="0.7", lw=0.5, zorder=0)
                ax.set_ylabel("Dominant peak position (kHz)" if protocol == "root"
                              else "Central dip position (kHz)")
                if protocol == "root":
                    ax.set_ylim(-270, 270)
                    ax.set_yticks([-200, 0, 200])
                else:
                    ax.set_ylim(-60, 60)
                    ax.set_yticks([-50, 0, 50])
            else:
                ax.set_xlabel(r"$\Omega_0/2\pi$ (MHz)")
                ax.set_ylabel("Local FWHM (kHz)")
                ax.set_ylim(0, 520)
                ax.set_yticks([0, 100, 200, 300, 400, 500])
                secondary = ax.secondary_yaxis(
                    "right", functions=(lambda x: x / gamma_khz,
                                        lambda x: x * gamma_khz)
                )
                secondary.set_ylabel(r"FWHM/$\Gamma_{T_2}$", labelpad=4)
                secondary.set_yticks([0, 4, 8, 12])
                secondary.tick_params(direction="in", labelsize=7, length=3)
    handles = [
        Line2D([], [], color=colors["raw"], marker=".", linestyle="none", ms=5,
               label="Experiment: raw"),
        Line2D([], [], color=colors["corrected"], marker="o", markerfacecolor="none",
               linestyle="none", ms=4, markeredgewidth=0.7,
               label="Corrected* (every 8th point)"),
        Line2D([], [], color=colors["simulation"], marker="x", linestyle="none", ms=4,
               markeredgewidth=0.7, label="Simulation"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.98),
               ncol=3, frameon=False, fontsize=7, columnspacing=1.4, handletextpad=0.4)
    fig.text(0.5, 0.082, "Raw and corrected positions/widths coincide; root peak switching is not a unique resonance shift.",
             ha="center", fontsize=6.6, color="0.3")
    fig.text(0.5, 0.047,
             rf"Narrow scans; $\Omega_0/2\pi\geq1$ MHz. $\Gamma_{{T_2}}=1/(\pi T_2)={gamma_khz:.2f}$ kHz; $T_2={t2_us:g}$ $\mu$s.",
             ha="center", fontsize=6.6, color="0.3")
    fig.text(0.5, 0.012, "*Provisional saved-matrix correction; classifier match unverified. Direct extraction; no uncertainty bars.",
             ha="center", fontsize=6.6, color="0.3")
    for extension in ("pdf", "png", "svg"):
        fig.savefig(OUTPUT / f"{STEM}.{extension}", dpi=350)
    plt.close(fig)
    with (OUTPUT / f"{STEM}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provenance = {
        "generator": "scripts/make_figure2_resonance_width_comparison.py",
        "included_in_manuscript": False,
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "source_readout_provenance": source_provenance,
        "qubit": "q6",
        "domain": "narrow scans only: -0.5 to +0.5 MHz",
        "selection": "Accepted dip_metrics extraction; display drive >= 1 MHz. All extracted points retained in CSV.",
        "position_definition": "Quadratically interpolated dominant root maximum / echo minimum inside +/-0.25 MHz; no detuning shift or fit to simulation.",
        "width_definition": "Nearest half-depth crossings after subtracting a linear background fitted to |detuning|>=0.4 MHz, using dip_metrics(-population) for root and dip_metrics(population) for echo.",
        "limitations": [
            "Root spectra have competing oscillatory peaks; largest-peak switches are not a unique physical resonance-frequency shift.",
            "Position uses the raw extremum; width uses the background-subtracted extremum, per the existing dip_metrics convention.",
            "No smoothing or uncertainty estimates. Marker scatter is not a confidence interval. Width is a finite-window local-feature metric.",
            "Experimental and simulation drive grids are kept at their original sampling; no rescaling or alignment.",
            "Readout correction is provisional; positions and widths are invariant to a positive global affine population correction.",
        ],
        "t2_us": t2_us, "gamma_t2_khz": gamma_khz,
        "coherence_source_sha256": hashlib.sha256(coherence_path.read_bytes()).hexdigest(),
        "affine_invariance_checks": checks,
        "counts": {f"{p}_{k}": int(np.sum(s["accepted"] & (s["drive_mhz"] >= 1)))
                   for (p, k), s in series.items()},
    }
    (OUTPUT / f"{STEM}_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"checks": checks, "counts": provenance["counts"]}, indent=2))
    print(OUTPUT / f"{STEM}.pdf")


if __name__ == "__main__":
    main()
