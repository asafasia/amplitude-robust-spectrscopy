"""Show linewidth, contrast, and contrast-weighted resolution for the supplement.

Run with: PYTHONPATH=. MPLBACKEND=Agg .venv/bin/python scripts/make_supplemental_extended_cutoff.py
Uses archived metrics and masks without rerunning or interpolating simulations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter

from echospec.analysis.cutoff_fit_exports import load_cutoff_fits
from echospec.figures.coherence import REFERENCE_FWHM_HZ, REFERENCE_T2_STAR_S
from echospec.figures.cutoff_contours import (
    CONTOUR_SMOOTHING,
    smooth_contrast_for_contour,
)
from echospec.figures.drag_convention import polynomial_beta_provenance
from echospec.figures.supplemental import STYLE_PATH, apply_supplemental_style

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/polynomial_duration_cutoff_poly2_log_40_50/n100"
OUT = ROOT / "figures/paper"
STEM = "supp_extended_cutoff_duration"
CONTRAST_THRESHOLD = 0.2
QUALITY_THRESHOLD = 0.10
QUALITY_COLOR = "#d62728"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--stem", default=STEM)
    parser.add_argument("--direct-metrics", action="store_true",
                        help="Render the archived direct half-depth metrics")
    args = parser.parse_args(argv)
    source_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    stem = args.stem
    manifest = json.loads((source_dir / "dataset_manifest.json").read_text())
    checksums = dict(
        (name, digest)
        for digest, name in (
            line.split("  ", 1)
            for line in (source_dir / "checksums.sha256").read_text().splitlines()
        )
    )
    hashes = {}
    for name in ("comparison_metrics.npz",):
        digest = hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
        assert digest == checksums[name], f"Source checksum changed: {name}"
        hashes[str((source_dir / name).relative_to(ROOT))] = digest
    keys = ("durations_us", "cutoffs", "rabi_mhz", "accepted", "fwhm_hz",
            "configuration")
    with np.load(source_dir / "comparison_metrics.npz", allow_pickle=False) as source:
        data = {key: source[key].copy() for key in keys}
        assert source["pending_durations_us"].size == 0
    lengths, cutoffs, amps = (data[key] for key in keys[:3])
    accepted = data["accepted"]
    contrasts = []
    for row, length in enumerate(lengths):
        folder = (manifest["baseline_copy"] if length in manifest["reused_durations_us"]
                  else manifest["new_runs"])
        name = f"{folder}/L{length:g}us/scan.npz"
        digest = hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
        assert digest == checksums[name], f"Source checksum changed: {name}"
        hashes[str((source_dir / name).relative_to(ROOT))] = digest
        with np.load(source_dir / name, allow_pickle=False) as source:
            np.testing.assert_array_equal(source["cutoffs"], cutoffs)
            np.testing.assert_array_equal(source["rabi_mhz"], amps)
            np.testing.assert_array_equal(source["accepted"], accepted[row])
            np.testing.assert_array_equal(source["fwhm_hz"], data["fwhm_hz"][row])
            contrasts.append(source["contrast"].copy())
    data["contrast"] = np.stack(contrasts)
    fit_metadata = None
    use_fits = not args.direct_metrics and (
        source_dir == SOURCE.resolve() or (source_dir / "robust_fits").exists()
    )
    if use_fits:
        fitted, fit_metadata, fit_hashes = load_cutoff_fits(
            source_dir, lengths, cutoffs, amps, accepted,
        )
        for key in ("accepted", "fwhm_hz", "contrast"):
            data[f"direct_{key}"] = data[key].copy()
            data[key] = fitted[key]
        for key in ("relative_rms", "relative_width_error", "radius_mhz", "status"):
            data[f"fit_{key}"] = fitted[key]
        accepted = data["accepted"]
        hashes.update(fit_hashes)
    assert ((data["contrast"][accepted] >= 0) & (data["contrast"][accepted] <= 1)).all()
    np.testing.assert_array_equal(lengths, manifest["durations_us"])
    assert len(lengths) > 0 and np.all(np.diff(lengths) > 0)
    np.testing.assert_allclose(cutoffs, np.geomspace(1e-5, 1, 100))
    config = json.loads(str(data["configuration"]))
    logarithmic = config.get("amplitude_spacing") == "log"
    expected = np.geomspace(1, 100, 100) if logarithmic else np.linspace(0, 100, 100)
    np.testing.assert_allclose(amps, expected)
    assert accepted.shape == (len(lengths), 100, 100)
    reference_hz = REFERENCE_FWHM_HZ
    assert (data["fwhm_hz"][accepted] > 0).all()
    quality = np.full(accepted.shape, np.nan)
    np.divide(data["contrast"] * reference_hz, data["fwhm_hz"],
              out=quality, where=accepted)
    data["quality"] = quality
    data["quality_above_threshold"] = accepted & (quality > QUALITY_THRESHOLD)
    data["quality_threshold"] = np.array(QUALITY_THRESHOLD)
    # Preserve the original cell edges while matching the main Figure 4 display.
    logc = np.log(cutoffs)
    c_edges = np.exp(np.r_[logc[0] - (logc[1] - logc[0]) / 2,
                          (logc[1:] + logc[:-1]) / 2,
                          logc[-1] + (logc[-1] - logc[-2]) / 2])
    centers = np.log(amps) if logarithmic else amps
    a_edges = np.r_[centers[0] - (centers[1] - centers[0]) / 2,
                    (centers[1:] + centers[:-1]) / 2,
                    centers[-1] + (centers[-1] - centers[-2]) / 2]
    if logarithmic:
        a_edges = np.exp(a_edges)
    positive = amps > 0
    first_positive = int(np.flatnonzero(positive)[0])
    # Keep the sampled threshold mask; interpolate only the displayed isoline.
    data["contrast_above_threshold"] = accepted & (data["contrast"] > CONTRAST_THRESHOLD)
    data["contrast_threshold"] = np.array(CONTRAST_THRESHOLD)
    contour_fields = [smooth_contrast_for_contour(c, ok)
                      for c, ok in zip(data["contrast"], accepted, strict=True)]
    apply_supplemental_style()
    fig, axes = plt.subplots(len(lengths), 3, figsize=(7.2, 0.8 + 1.36 * len(lengths)),
                             sharex=True, sharey=True, squeeze=False,
                             layout="constrained")
    scales = {}
    for col, (key, factor, lower, upper, cmap, label, ticks) in enumerate([
        ("fwhm_hz", 1 / reference_hz, 0, 10, "magma",
         r"$\mathrm{FWHM}/\Gamma_{T_2^*}$", [0, 2, 4, 6, 8, 10]),
        ("contrast", 1, 0, 1, "viridis", "Contrast", [0, 0.5, 1]),
        ("quality", 1, 0, 0.2, "cividis",
         r"$Q=C\,\Gamma_{T_2^*}/\mathrm{FWHM}$", [0, 0.1, 0.2]),
    ]):
        values = data[key]
        assert values.shape == accepted.shape
        assert np.isfinite(values[accepted]).all()
        scales[key] = {"factor": factor, "vmin": lower, "vmax": upper, "colormap": cmap}
        color = plt.get_cmap(cmap).copy()
        color.set_bad("#e2e7e9")
        for row, length in enumerate(lengths):
            ax = axes[row, col]
            masked = np.ma.masked_where(~accepted[row], values[row] * factor)
            artist = ax.pcolormesh(
                c_edges, a_edges[first_positive:], masked[:, positive].T,
                cmap=color, vmin=lower, vmax=upper, rasterized=True,
            )
            ax.contour(
                cutoffs, amps[positive],
                contour_fields[row][:, positive].T,
                levels=[CONTRAST_THRESHOLD], colors=["white"], linewidths=0.6,
                corner_mask=False, zorder=3,
            )
            quality_contour = ax.contour(
                cutoffs, amps[positive],
                np.ma.masked_invalid(quality[row])[:, positive].T,
                levels=[QUALITY_THRESHOLD], colors=[QUALITY_COLOR], linewidths=0.8,
                corner_mask=False, zorder=4,
            )
            quality_contour.set_path_effects([
                pe.withStroke(linewidth=1.25, foreground="white"),
            ])
            ax.set(xscale="log", yscale="log", xlim=(1e-5, 1), ylim=(1, 100),
                   xticks=[1e-5, 1e-3, 1e-1, 1], yticks=[1, 10, 100])
            ax.minorticks_off()
            ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.tick_params(direction="in", top=True, right=True, length=2, pad=2,
                           labelleft=col == 0)
            ax.tick_params(axis="y", which="major", length=3)
            ax.tick_params(axis="y", which="minor", direction="in", right=True,
                           length=1.5, width=0.5)
            title = f"({chr(97 + 3 * row + col)})"
            if col == 0:
                title += rf"  $L={length:g}\,\mu\mathrm{{s}}$"
            ax.set_title(title, loc="left", fontsize=9)
            if col == 0:
                ax.set_ylabel(r"$\Omega_0/(2\pi)$ (MHz)")
            if row == len(lengths) - 1:
                ax.set_xlabel("Cutoff $c$")
        bar = fig.colorbar(artist, ax=list(axes[:, col]), location="top",
                           fraction=0.035, pad=0.025, aspect=35, ticks=ticks)
        bar.set_label(label)
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"{stem}.{ext}", dpi=300)
    plt.close(fig)
    arrays_path = output_dir / f"{stem}.npz"
    np.savez_compressed(arrays_path, **data)
    provenance = {
        "style": str(STYLE_PATH.relative_to(ROOT)),
        "style_sha256": hashlib.sha256(STYLE_PATH.read_bytes()).hexdigest(),
        "source_dataset": str(source_dir.relative_to(ROOT)),
        "durations_us": lengths.tolist(),
        "configuration": config,
        "drag_beta_convention": polynomial_beta_provenance(config["drag_beta"]),
        "drag_convention_helper_sha256": hashlib.sha256(
            Path(polynomial_beta_provenance.__code__.co_filename).read_bytes()
        ).hexdigest(),
        "metric_method": fit_metadata["definition"] if use_fits else "Direct finite-window half-depth",
        "fitted_metric_provenance": fit_metadata,
        "source_sha256": hashes,
        "generator": str(Path(__file__).resolve().relative_to(ROOT)),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "arrays_sha256": hashlib.sha256(arrays_path.read_bytes()).hexdigest(),
        "reproduction": ("PYTHONPATH=. MPLBACKEND=Agg .venv/bin/python "
                         "scripts/make_supplemental_extended_cutoff.py "
                         f"--input-dir {source_dir.relative_to(ROOT)} "
                         f"--output-dir {output_dir.relative_to(ROOT)} --stem {stem}"),
        "change_from_report": "Five duration rows from the same saved spectra and cell boundaries. Metric extraction and fit acceptance are specified in metric_method. Match main Figure 4 axis orientation and first two colormaps: logarithmic cutoff on x, logarithmic positive drive on y, normalized linewidth in magma on [0, 10], and probability contrast in viridis on [0, 1]. Contrast-weighted resolution Q uses cividis on [0, 0.2]. Widths above 10 saturate at the lightest color.",
        "contrast_definition": ("Gaussian dip depth above the locally fitted quadratic background"
                                if use_fits else "Dip depth relative to the fitted linear edge background"),
        "contrast_outline": {
            "threshold": CONTRAST_THRESHOLD,
            "rule": "accepted and contrast > threshold",
            "geometry": "C=0.2 isoline of a display-only smoothed contrast field; rejected cells remain masked; identical in all three columns. Heatmap values and quantitative threshold masks are not smoothed.",
            "smoothing": CONTOUR_SMOOTHING,
            "helper_sha256": hashlib.sha256(
                Path(smooth_contrast_for_contour.__code__.co_filename).read_bytes()
            ).hexdigest(),
            "color": "white",
            "linewidth_pt": 0.6,
            "halo": False,
            "cells_per_duration": data["contrast_above_threshold"].sum(axis=(1, 2)).tolist(),
        },
        "quality_outline": {
            "definition": "Q = contrast * reference_linewidth_hz / fwhm_hz",
            "threshold": QUALITY_THRESHOLD,
            "rule": "accepted and Q > threshold",
            "geometry": "Linearly interpolated Q=0.10 isoline between accepted samples, outlining higher-quality regions; rejected cells remain masked; identical in all three columns",
            "color": QUALITY_COLOR,
            "linewidth_pt": 0.8,
            "halo_color": "white",
            "halo_linewidth_pt": 1.25,
            "cells_per_duration": data["quality_above_threshold"].sum(axis=(1, 2)).tolist(),
        },
        "array_order": "duration, cutoff, nominal drive",
        "map_axes": {"x": "cutoff", "y": "nominal drive", "display": "Transposed arrays"},
        "labels": {"linewidth": r"$\mathrm{FWHM}/\Gamma_{T_2^*}$",
                   "contrast": "Contrast", "contrast_symbol": "C",
                   "quality": r"$Q=C\,\Gamma_{T_2^*}/\mathrm{FWHM}$",
                   "drive": r"$\Omega_0/(2\pi)$ (MHz)", "cutoff": "Cutoff $c$"},
        "amplitude_display": {
            "scale": "log", "limits_mhz": [1, 100],
            "positive_sample_count": int(positive.sum()),
            "sampling": "logarithmic" if logarithmic else "linear",
            "zero_drive": ("Not sampled: logarithmic grid starts at 1 MHz" if logarithmic
                           else "Omitted from plot; retained in exported arrays"),
            "cell_edges": ("Geometric midpoints" if logarithmic else
                           "Original linear-amplitude cell edges; no resampling"),
        },
        "reference_linewidth_hz": reference_hz,
        "reference_t2_star_s": REFERENCE_T2_STAR_S,
        "color_scales": scales,
        "accepted_traces_per_duration": accepted.sum(axis=(1, 2)).tolist(),
    }
    (output_dir / f"{stem}_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    # Summarize the fixed-cutoff probability-contrast objective for the text.
    drive_mask = amps >= 10
    summaries = []
    for row, length in enumerate(lengths):
        width = data["fwhm_hz"][row][:, drive_mask]
        contrast = data["contrast"][row][:, drive_mask]
        eligible = accepted[row][:, drive_mask].all(axis=1)
        score = np.min(contrast / (width / 1e6), axis=1)
        score[~eligible] = -np.inf
        if not eligible.any():
            summaries.append(dict(duration_us=float(length), cutoff=None,
                                  status="No cutoff accepted at every drive >=10 MHz"))
            continue
        index = int(np.argmax(score))
        summaries.append(dict(
            duration_us=float(length), cutoff=float(cutoffs[index]),
            maximum_fwhm_khz=float(width[index].max() / 1000),
            minimum_contrast=float(contrast[index].min()),
            minimum_contrast_per_mhz=float(score[index]),
        ))
    (output_dir / f"{stem}_summary.json").write_text(json.dumps({
        "selection": "Maximize minimum contrast/FWHM over sampled drives >=10 MHz; require accepted dips at every drive. No interpolation.",
        "results": summaries,
    }, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
