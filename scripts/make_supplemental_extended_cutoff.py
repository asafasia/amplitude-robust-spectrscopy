"""Show full-range linewidth and probability contrast for a supplemental page.

Run with: MPLBACKEND=Agg python scripts/make_supplemental_extended_cutoff.py
Uses archived metrics and masks without rerunning or interpolating simulations.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from echospec.figures.supplemental import STYLE_PATH, apply_supplemental_style

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/polynomial_duration_cutoff_sweep/n100"
OUT = ROOT / "figures/paper"
STEM = "supp_extended_cutoff_duration"
CONTRAST_THRESHOLD = 0.2


def main():
    checksums = dict(
        (name, digest)
        for digest, name in (
            line.split("  ", 1)
            for line in (SOURCE / "checksums.sha256").read_text().splitlines()
        )
    )
    hashes = {}
    for name in ("comparison_metrics.npz", "duration_comparison.pdf"):
        digest = hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
        assert digest == checksums[name], f"Source checksum changed: {name}"
        hashes[str((SOURCE / name).relative_to(ROOT))] = digest
    keys = ("durations_us", "cutoffs", "rabi_mhz", "accepted", "fwhm_hz",
            "configuration")
    with np.load(SOURCE / "comparison_metrics.npz", allow_pickle=False) as source:
        data = {key: source[key].copy() for key in keys}
        assert source["pending_durations_us"].size == 0
    lengths, cutoffs, amps = (data[key] for key in keys[:3])
    accepted = data["accepted"]
    contrasts = []
    for row, length in enumerate(lengths):
        folder = "baseline_30us" if length == 30 else "simulations"
        name = f"{folder}/L{length:g}us/scan.npz"
        digest = hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
        assert digest == checksums[name], f"Source checksum changed: {name}"
        hashes[str((SOURCE / name).relative_to(ROOT))] = digest
        with np.load(SOURCE / name, allow_pickle=False) as source:
            np.testing.assert_array_equal(source["cutoffs"], cutoffs)
            np.testing.assert_array_equal(source["rabi_mhz"], amps)
            np.testing.assert_array_equal(source["accepted"], accepted[row])
            np.testing.assert_array_equal(source["fwhm_hz"], data["fwhm_hz"][row])
            contrasts.append(source["contrast"].copy())
    data["contrast"] = np.stack(contrasts)
    assert ((data["contrast"][accepted] >= 0) & (data["contrast"][accepted] <= 1)).all()
    config = json.loads(str(data["configuration"]))
    np.testing.assert_array_equal(lengths, [10, 20, 30, 45, 60])
    np.testing.assert_allclose(cutoffs, np.geomspace(1e-5, 1, 100))
    np.testing.assert_allclose(amps, np.linspace(0, 100, 100))
    assert accepted.shape == (5, 100, 100)
    reference_hz = 1 / (np.pi * config["t2_us"] * 1e-6)
    # Match the report's cells and width scale; contrast uses probability [0, 1].
    logc = np.log(cutoffs)
    c_edges = np.exp(np.r_[logc[0] - (logc[1] - logc[0]) / 2,
                          (logc[1:] + logc[:-1]) / 2,
                          logc[-1] + (logc[-1] - logc[-2]) / 2])
    a_edges = np.r_[amps[0] - (amps[1] - amps[0]) / 2,
                    (amps[1:] + amps[:-1]) / 2,
                    amps[-1] + (amps[-1] - amps[-2]) / 2]
    # Keep the sampled threshold mask; interpolate only the displayed isoline.
    data["contrast_above_threshold"] = accepted & (data["contrast"] > CONTRAST_THRESHOLD)
    data["contrast_threshold"] = np.array(CONTRAST_THRESHOLD)
    apply_supplemental_style()
    fig, axes = plt.subplots(5, 2, figsize=(7.2, 7.6), sharex=True, sharey=True,
                             layout="constrained")
    scales = {}
    for col, (key, factor, minimum, rounding, cmap, label) in enumerate([
        ("fwhm_hz", 1 / reference_hz, 600_000, 5000, "magma_r",
         r"Local dip FWHM / $\Gamma_{f,T_2}$"),
        ("contrast", 1, 1, 1, "viridis", "Contrast"),
    ]):
        values = data[key]
        assert values.shape == accepted.shape
        assert np.isfinite(values[accepted]).all()
        upper = factor * max(minimum, np.ceil(values[accepted].max() / rounding)
                             * rounding)
        if key == "fwhm_hz":
            upper = 10.0
        scales[key] = {"factor": factor, "vmin": 0, "vmax": upper, "colormap": cmap}
        color = plt.get_cmap(cmap).copy()
        color.set_bad("#e2e7e9")
        for row, length in enumerate(lengths):
            ax = axes[row, col]
            masked = np.ma.masked_where(~accepted[row], values[row] * factor)
            artist = ax.pcolormesh(a_edges, c_edges, masked, cmap=color,
                                   vmin=0, vmax=upper, rasterized=True)
            ax.contour(
                amps, cutoffs,
                np.ma.masked_where(~accepted[row], data["contrast"][row]),
                levels=[CONTRAST_THRESHOLD], colors=["white"], linewidths=0.6,
                corner_mask=False, zorder=3,
            )
            ax.set(yscale="log", ylim=(cutoffs[0], cutoffs[-1]), xlim=(0, 100),
                   xticks=[0, 25, 50, 75, 100], yticks=[1e-5, 1e-3, 1e-1, 1])
            ax.minorticks_off()
            ax.set_title(f"({chr(97 + 2 * row + col)})  "
                         + rf"$L={length:g}\,\mu\mathrm{{s}}$", loc="left", fontsize=9)
            if col == 0:
                ax.set_ylabel("Cutoff $c$")
            if row == 4:
                ax.set_xlabel(r"Nominal drive $\Omega_0/(2\pi)$ (MHz)")
        bar = fig.colorbar(artist, ax=list(axes[:, col]), location="top",
                           fraction=0.035, pad=0.025, aspect=35)
        bar.set_label(label)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{STEM}.{ext}", dpi=300)
    plt.close(fig)
    np.savez_compressed(OUT / f"{STEM}.npz", **data)
    provenance = {
        "style": str(STYLE_PATH.relative_to(ROOT)),
        "style_sha256": hashlib.sha256(STYLE_PATH.read_bytes()).hexdigest(),
        "source_report": str((SOURCE / "report.html").relative_to(ROOT)),
        "source_sha256": hashes,
        "generator": str(Path(__file__).resolve().relative_to(ROOT)),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reproduction": "MPLBACKEND=Agg python scripts/make_supplemental_extended_cutoff.py",
        "change_from_report": "Five duration rows; same samples, masks, and cell boundaries. Replace ideal SNR with saved dip contrast on [0, 1]; show normalized linewidth on [0, 10] with reversed magma colors so low widths are light. Widths above 10 saturate at the darkest color.",
        "contrast_definition": "Dip depth in excited-state probability relative to the fitted linear background at the dip.",
        "contrast_outline": {
            "threshold": CONTRAST_THRESHOLD,
            "rule": "accepted and contrast > threshold",
            "geometry": "Linearly interpolated C=0.2 isoline between accepted samples, with rejected grid cells masked; identical in both columns. Heatmap values are not smoothed.",
            "color": "white",
            "linewidth_pt": 0.6,
            "halo": False,
            "cells_per_duration": data["contrast_above_threshold"].sum(axis=(1, 2)).tolist(),
        },
        "array_order": "duration, cutoff, nominal drive",
        "reference_linewidth_hz": reference_hz,
        "color_scales": scales,
        "accepted_traces_per_duration": accepted.sum(axis=(1, 2)).tolist(),
    }
    (OUT / f"{STEM}_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    # Recompute the supplemental table with the probability-contrast objective.
    drive_mask = amps >= 10
    summaries = []
    for row, length in enumerate(lengths):
        width = data["fwhm_hz"][row][:, drive_mask]
        contrast = data["contrast"][row][:, drive_mask]
        eligible = accepted[row][:, drive_mask].all(axis=1)
        score = np.min(contrast / (width / 1e6), axis=1)
        score[~eligible] = -np.inf
        assert eligible.any()
        index = int(np.argmax(score))
        summaries.append(dict(
            duration_us=float(length), cutoff=float(cutoffs[index]),
            maximum_fwhm_khz=float(width[index].max() / 1000),
            minimum_contrast=float(contrast[index].min()),
            minimum_contrast_per_mhz=float(score[index]),
        ))
    (OUT / f"{STEM}_summary.json").write_text(json.dumps({
        "selection": "Maximize minimum contrast/FWHM over sampled drives >=10 MHz; require accepted dips at every drive. No interpolation.",
        "results": summaries,
    }, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
