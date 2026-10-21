"""PRL Figure 4: selected slices of the completed polynomial duration/cutoff scan.

Uses the original float64 metrics and acceptance masks, without recomputing,
optimizing, or interpolating simulation samples. The full scan remains in its
self-contained HTML report.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from echospec.figures import FigureVariant, apply_figure_style, save_figure

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/polynomial_duration_cutoff_sweep/n100"
OUT = ROOT / "figures/paper"
STEM = "04_duration_cutoff_scan"
COLORS = ("#0072b2", "#009e73", "#d55e00", "#cc79a7", "#555555")
STYLES = ("-", "--", "-.", ":", (0, (4, 1, 1, 1)))


def main():
    checksums = {}
    for line in (SOURCE / "checksums.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        checksums[name] = digest
    hashes = {}
    for name in ("comparison_metrics.npz", "comparison_summary.json",
                 "dataset_manifest.json", "validation/convergence.json"):
        digest = hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
        assert digest == checksums[name], f"Source checksum changed: {name}"
        hashes[str((SOURCE / name).relative_to(ROOT))] = digest
    with np.load(SOURCE / "comparison_metrics.npz", allow_pickle=False) as data:
        lengths, cutoffs, amps = (data[k].copy() for k in
                                  ("durations_us", "cutoffs", "rabi_mhz"))
        accepted, width = (data[k].copy() for k in ("accepted", "fwhm_hz"))
        config = json.loads(str(data["configuration"]))
        assert data["pending_durations_us"].size == 0
    contrasts = []
    for k, length in enumerate(lengths):
        folder = "baseline_30us" if length == 30 else "simulations"
        name = f"{folder}/L{length:g}us/scan.npz"
        digest = hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
        assert digest == checksums[name], f"Source checksum changed: {name}"
        hashes[str((SOURCE / name).relative_to(ROOT))] = digest
        with np.load(SOURCE / name, allow_pickle=False) as data:
            np.testing.assert_array_equal(data["cutoffs"], cutoffs)
            np.testing.assert_array_equal(data["rabi_mhz"], amps)
            np.testing.assert_array_equal(data["accepted"], accepted[k])
            np.testing.assert_array_equal(data["fwhm_hz"], width[k])
            contrasts.append(data["contrast"].copy())
    contrast = np.stack(contrasts)
    manifest = json.loads((SOURCE / "dataset_manifest.json").read_text())
    summary = json.loads((SOURCE / "comparison_summary.json").read_text())
    convergence = json.loads((SOURCE / "validation/convergence.json").read_text())
    assert config == manifest["common_configuration"]
    np.testing.assert_array_equal(lengths, [10, 20, 30, 45, 60])
    assert accepted.shape == width.shape == contrast.shape == (5, 100, 100)
    assert np.isfinite(width[accepted]).all() and (width[accepted] > 0).all()
    assert np.isfinite(contrast[accepted]).all()
    assert ((contrast[accepted] >= 0) & (contrast[accepted] <= 1)).all()
    ci = int(np.argmin(abs(np.log(cutoffs / 0.001))))
    ai = int(np.argmin(abs(amps - 50)))
    selected = cutoffs <= 0.01
    # Use the report's matched operating point, never a separately optimized
    # cutoff for each length. Validate it against the independently saved summary.
    point = summary["matched_operating_point"]
    assert cutoffs[ci] == point["cutoff"] and amps[ai] == point["amplitude_mhz"]
    for k, result in enumerate(point["results"]):
        assert result["duration_us"] == lengths[k]
        np.testing.assert_allclose(width[k, ci, ai] / 1000, result["fwhm_khz"])
    point = dict(cutoff=float(cutoffs[ci]), amplitude_mhz=float(amps[ai]), results=[
        dict(duration_us=float(length), fwhm_khz=float(width[k, ci, ai] / 1000),
             contrast=float(contrast[k, ci, ai]))
        for k, length in enumerate(lengths)
    ])

    apply_figure_style(FigureVariant.PAPER)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
        "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
        "ytick.labelsize": 6, "axes.linewidth": 0.6, "axes.grid": False,
        "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.bbox": None,
    })
    fig_width, fig_height = 3.38, 3.5
    fig = plt.figure(figsize=(fig_width, fig_height))
    axes = np.empty((2, 2), dtype=object)
    coherence_fwhm_khz = 1000 / (np.pi * config["t2_us"])
    for row, bottom in enumerate((1.98, 0.42)):
        for col, left in enumerate((0.43, 2.22)):
            ax = fig.add_axes([left / fig_width, bottom / fig_height,
                               1.0 / fig_width, 1.06 / fig_height])
            axes[row, col] = ax
            ax.tick_params(direction="in", top=True, right=True, length=2, pad=2)
            ax.set_ylabel("FWHM (kHz)" if col == 0 else "Contrast",
                          labelpad=2)
            ax.set_ylim(0, 450 if col == 0 else 1)
            ax.set_yticks([0, 200, 400] if col == 0 else [0, 0.5, 1])
            ax.text(0.95, 0.94, f"({chr(97 + 2 * row + col)})",
                    ha="right", va="top", transform=ax.transAxes)
            if row == 0:
                ax.set(xscale="log", xlim=(1e-5, 1e-2),
                       xticks=[1e-5, 1e-4, 1e-3, 1e-2], xlabel="Cutoff $c$")
                ax.axvline(cutoffs[ci], color="0.7", lw=0.6, ls=":", zorder=0)
            else:
                ax.set(xlim=(0, 100), xticks=[0, 50, 100],
                       xlabel=r"$\Omega_0/2\pi$ (MHz)")
            ax.xaxis.labelpad = 2
            if col == 0:
                normalized_axis = ax.secondary_yaxis(
                    "right", functions=(lambda value: value / coherence_fwhm_khz,
                                        lambda value: value * coherence_fwhm_khz)
                )
                normalized_axis.set_ylabel(r"FWHM/$\Gamma_{T_2}$",
                                            fontsize=7, labelpad=2)
                normalized_axis.set_yticks([0, 5, 10])
                normalized_axis.tick_params(direction="in", length=2, pad=2, labelsize=6)
                ax.tick_params(right=False)
                ax.axhline(coherence_fwhm_khz,
                           color="0.7", lw=0.6, ls=":", zorder=0)

    rows = []
    for k, length in enumerate(lengths):
        style = dict(color=COLORS[k], ls=STYLES[k], lw=1)
        for col, (values, scale) in enumerate(((width, 0.001), (contrast, 1))):
            cutoff_values = np.where(accepted[k, selected, ai],
                                     values[k, selected, ai] * scale, np.nan)
            drive_values = np.where(accepted[k, ci], values[k, ci] * scale, np.nan)
            axes[0, col].plot(cutoffs[selected], cutoff_values, **style)
            axes[1, col].plot(amps, drive_values, **style)
        for cut_index, amp_index, kind in [
            *((j, ai, "cutoff") for j in np.flatnonzero(selected)),
            *((ci, j, "drive") for j in range(len(amps))),
        ]:
            rows.append(dict(
                slice=kind, duration_us=float(length), cutoff=float(cutoffs[cut_index]),
                amplitude_mhz=float(amps[amp_index]),
                accepted=bool(accepted[k, cut_index, amp_index]),
                fwhm_khz=float(width[k, cut_index, amp_index] / 1000),
                contrast=float(contrast[k, cut_index, amp_index]),
            ))
    fig.text(0.54, 3.24 / fig_height,
             rf"$\Omega_0/2\pi = {amps[ai]:.1f}$ MHz", ha="center")
    fig.text(0.54, 1.68 / fig_height,
             rf"$c = {cutoffs[ci] * 1000:.2f}\times10^{{-3}}$", ha="center")
    handles = [Line2D([], [], color=c, ls=s, lw=1)
               for c, s in zip(COLORS, STYLES, strict=True)]
    fig.legend(handles, [rf"${length:g}\,\mu$s" for length in lengths],
               loc="lower center", bbox_to_anchor=(0.53, 0), ncol=5, frameon=False,
               fontsize=6, handlelength=1.4, columnspacing=0.65, handletextpad=0.35)
    paths = save_figure(fig, STEM, variant=FigureVariant.PAPER,
                        formats=("pdf", "png", "svg"), dpi=350)
    plt.close(fig)
    with (OUT / f"{STEM}_metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        OUT / f"{STEM}.npz", durations_us=lengths, cutoffs=cutoffs, rabi_mhz=amps,
        accepted=accepted, fwhm_hz=width, contrast=contrast,
        selected_cutoff_mask=selected, fixed_cutoff_index=ci, fixed_amplitude_index=ai,
    )
    provenance = dict(
        generator="scripts/make_main_duration_cutoff_scan.py",
        included_in_manuscript=True, source_sha256=hashes,
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_configuration=config, figure_size_inches=[fig_width, fig_height],
        secondary_fwhm_axis=dict(t2_us=config["t2_us"],
                                reference_fwhm_khz=coherence_fwhm_khz,
                                definition="FWHM / [1/(pi*T2)]"),
        selection="Report default: nearest sampled cutoff to 0.001 and drive to 50 MHz; "
                  "show cutoff subset 1e-5 to 1e-2 and all drive samples, all five lengths",
        matched_operating_point=point,
        metric_mask="Only accepted dips drawn; rejected and zero-drive rows are gaps, "
                    "not zero values. FWHM is local, not a guarantee of a unique dip.",
        contrast="Dip depth in excited-state probability relative to the fitted "
                 "linear background at the dip; dimensionless, plotted on [0, 1]",
        validation="Source files verified against report SHA-256 manifest; selected "
                   "point reproduced from summary; no smoothing or metric recomputation",
        representative_convergence={
            "scope": convergence["scope"],
            "maximum_changes": {key: value for key, value in
                                convergence["maximum_changes"].items()
                                if "snr" not in key},
            "time_status_changes": convergence["time_status_changes"],
            "frequency_status_changes": convergence["frequency_status_changes"],
        },
    )
    (OUT / f"{STEM}_provenance.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(point, indent=2))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
