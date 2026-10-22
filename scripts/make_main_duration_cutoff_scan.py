"""PRL Figure 4: 30 us cutoff-versus-drive maps from the supplemental survey.

Uses the original float64 metrics and acceptance masks, without recomputing,
optimizing, or interpolating simulation samples. The full scan remains in its
self-contained HTML report.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter

from echospec.figures import FigureVariant, apply_figure_style, save_figure

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/polynomial_duration_cutoff_sweep/n100"
OUT = ROOT / "figures/paper"
STEM = "04_duration_cutoff_scan"
CONTRAST_THRESHOLD = 0.2
COLORS = ("#0072b2", "#009e73", "#d55e00", "#aa6e9f", "#555555")
LINESTYLES = ("--", "-.", "-", ":", (0, (4, 1, 1, 1)))


def select_duration_maps(lengths, accepted, width, contrast, duration_us=30.0):
    """Select an exact duration without transposing cutoff-by-amplitude maps."""
    indices = np.flatnonzero(np.asarray(lengths) == duration_us)
    if indices.size != 1:
        raise ValueError("Require exactly one saved row for the requested duration")
    row = int(indices[0])
    data = {key: values[row].copy() for key, values in (
        ("accepted", accepted), ("fwhm_hz", width), ("contrast", contrast)
    )}
    data["contrast_above_threshold"] = data["accepted"] & (
        data["contrast"] > CONTRAST_THRESHOLD
    )
    return data


def select_cutoff_slices(amps, accepted, width, contrast, amplitude_mhz=50.0):
    """Extract the same nearest-amplitude column for every pulse duration."""
    index = int(np.argmin(abs(np.asarray(amps) - amplitude_mhz)))
    return index, {key: values[:, :, index].copy() for key, values in (
        ("slice_accepted", accepted), ("slice_fwhm_hz", width),
        ("slice_contrast", contrast),
    )}


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
    convergence = json.loads((SOURCE / "validation/convergence.json").read_text())
    assert config == manifest["common_configuration"]
    np.testing.assert_array_equal(lengths, [10, 20, 30, 45, 60])
    assert accepted.shape == width.shape == contrast.shape == (5, 100, 100)
    assert np.isfinite(width[accepted]).all() and (width[accepted] > 0).all()
    assert np.isfinite(contrast[accepted]).all()
    assert ((contrast[accepted] >= 0) & (contrast[accepted] <= 1)).all()
    slice_index, slices = select_cutoff_slices(amps, accepted, width, contrast)
    data = select_duration_maps(lengths, accepted, width, contrast)
    supplement = OUT / "supp_extended_cutoff_duration.npz"
    hashes[str(supplement.relative_to(ROOT))] = hashlib.sha256(
        supplement.read_bytes()
    ).hexdigest()
    with np.load(supplement, allow_pickle=False) as saved:
        index = int(np.flatnonzero(saved["durations_us"] == 30)[0])
        for key in ("cutoffs", "rabi_mhz"):
            np.testing.assert_array_equal(saved[key], cutoffs if key == "cutoffs" else amps)
        for key in data:
            np.testing.assert_array_equal(data[key], saved[key][index])
        assert float(saved["contrast_threshold"]) == CONTRAST_THRESHOLD
    accepted = data["accepted"]
    ci = int(np.argmin(abs(cutoffs - 6.579332246575682e-4)))
    drive_mask = amps >= 10
    assert accepted[ci, drive_mask].all()
    region = data["contrast_above_threshold"]
    summaries = dict(
        above_threshold=dict(
            sample_count=int(region.sum()),
            fwhm_khz_range=[float(data["fwhm_hz"][region].min() / 1000),
                            float(data["fwhm_hz"][region].max() / 1000)],
        ),
        fixed_cutoff_example=dict(
            cutoff=float(cutoffs[ci]),
            amplitude_mhz_range=[float(amps[drive_mask].min()), float(amps.max())],
            minimum_contrast=float(data["contrast"][ci, drive_mask].min()),
            maximum_fwhm_khz=float(data["fwhm_hz"][ci, drive_mask].max() / 1000),
        ),
    )

    apply_figure_style(FigureVariant.PAPER)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
        "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
        "ytick.labelsize": 6, "axes.linewidth": 0.6, "axes.grid": False,
        "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.bbox": None,
    })
    slice_amplitude = float(amps[slice_index])
    fig_width, fig_height = 3.38, 3.48
    fig = plt.figure(figsize=(fig_width, fig_height))
    reference_hz = 1 / (np.pi * config["t2_us"] * 1e-6)
    # Use the same geometric/linear cell edges as the supplemental maps.
    logc = np.log(cutoffs)
    c_edges = np.exp(np.r_[logc[0] - (logc[1] - logc[0]) / 2,
                          (logc[1:] + logc[:-1]) / 2,
                          logc[-1] + (logc[-1] - logc[-2]) / 2])
    a_edges = np.r_[amps[0] - (amps[1] - amps[0]) / 2,
                    (amps[1:] + amps[:-1]) / 2,
                    amps[-1] + (amps[-1] - amps[-2]) / 2]
    # Preserve the original cells; omit zero drive from the logarithmic display.
    positive = amps > 0
    first_positive = int(np.flatnonzero(positive)[0])
    scales = {}
    for col, (key, factor, maximum, cmap, label, ticks) in enumerate([
        ("fwhm_hz", 1 / reference_hz, 5, "magma",
         r"FWHM/$\Gamma_{T_2}$", [1, 3, 5]),
        ("contrast", 1, 1, "viridis", "Contrast", [0, 0.5, 1]),
    ]):
        left = 0.42 + 1.54 * col
        ax = fig.add_axes([left / fig_width, 1.66 / fig_height,
                           1.24 / fig_width, 1.25 / fig_height])
        color = plt.get_cmap(cmap).copy()
        color.set_bad("#e2e7e9")
        values = np.ma.masked_where(~accepted, data[key] * factor)
        minimum = 1 if key == "fwhm_hz" else 0
        mesh = ax.pcolormesh(c_edges, a_edges[first_positive:], values[:, positive].T,
                             cmap=color,
                             vmin=minimum, vmax=maximum, rasterized=True)
        ax.contour(
            cutoffs, amps[positive],
            np.ma.masked_where(~accepted, data["contrast"])[:, positive].T,
            levels=[CONTRAST_THRESHOLD], colors=["white"], linewidths=0.6,
            corner_mask=False, zorder=3,
        )
        ax.axhline(
            slice_amplitude, color=COLORS[2], ls="--", lw=1.0, zorder=4,
            path_effects=[pe.withStroke(linewidth=1.6, foreground="white")],
        )
        ax.set(xscale="log", yscale="log", xlim=(1e-5, 1), ylim=(1, 100),
               xticks=[1e-5, 1e-3, 1e-1, 1], yticks=[1, 10, 100],
               xlabel="Cutoff $c$")
        ax.xaxis.labelpad = 2
        ax.minorticks_off()
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(direction="in", top=True, right=True, length=2, pad=2,
                       labelleft=True)
        ax.tick_params(axis="y", which="major", length=3)
        ax.tick_params(axis="y", which="minor", direction="in", right=True,
                       length=1.5, width=0.5)
        if col == 0:
            ax.set_ylabel(r"$\Omega_0/2\pi$ (MHz)", labelpad=2)
        ax.text(0.08, 0.94, f"({chr(97 + col)})", transform=ax.transAxes,
                va="top", color="white",
                path_effects=[pe.withStroke(linewidth=0.65, foreground="0.2")])
        cb_ax = fig.add_axes([left / fig_width, 3.04 / fig_height,
                              1.24 / fig_width, 0.06 / fig_height])
        bar = fig.colorbar(mesh, cax=cb_ax, orientation="horizontal", ticks=ticks)
        bar.ax.xaxis.set_ticks_position("top")
        bar.ax.xaxis.set_label_position("top")
        bar.ax.tick_params(length=2, pad=1, labelsize=6)
        bar.set_label(label, labelpad=3)
        scales[key] = dict(factor=factor, vmin=minimum, vmax=maximum, colormap=cmap)
        slice_height = 0.86
        slice_ax = fig.add_axes([left / fig_width, 0.42 / fig_height,
                                 1.24 / fig_width, slice_height / fig_height])
        for row, length in enumerate(lengths):
            highlighted = length == 30
            slice_values = np.ma.masked_where(
                ~slices["slice_accepted"][row], slices[f"slice_{key}"][row] * factor
            )
            slice_ax.plot(
                cutoffs, slice_values, color=COLORS[row], ls=LINESTYLES[row],
                lw=1.5 if highlighted else 0.75,
                alpha=1.0 if highlighted else 0.8, zorder=5 if highlighted else 2,
            )
        slice_maximum = 15 if key == "fwhm_hz" else maximum
        slice_ticks = [0, 5, 10, 15] if key == "fwhm_hz" else ticks
        slice_ax.set(xscale="log", xlim=(1e-5, 1), ylim=(0, slice_maximum),
                     xticks=[1e-5, 1e-3, 1e-1, 1], yticks=slice_ticks,
                     xlabel="Cutoff $c$", ylabel=label)
        slice_ax.minorticks_off()
        slice_ax.tick_params(direction="in", top=True, right=True, length=2, pad=2)
        slice_ax.xaxis.labelpad = 2
        slice_ax.yaxis.labelpad = 2
        slice_ax.text(0.08, 0.92, f"({chr(99 + col)})", va="top",
                      transform=slice_ax.transAxes)
        if key == "fwhm_hz":
            slice_ax.axhline(1, color="0.5", ls=":", lw=0.65)
            slice_ax.text(0.7, 1.6, r"$1/(\pi T_2)$", ha="right",
                          color="0.35", fontsize=6)
        if key == "contrast":
            slice_ax.axhline(CONTRAST_THRESHOLD, color="0.5", ls=":", lw=0.65)
    handles = [Line2D([], [], color=COLORS[row], ls=LINESTYLES[row],
                      lw=1.5 if length == 30 else 0.75,
                      alpha=1.0 if length == 30 else 0.8)
               for row, length in enumerate(lengths)]
    legend = fig.legend(
        handles, [rf"${length:g}\,\mu$s" for length in lengths],
        loc="lower center", bbox_to_anchor=(0.55, 0), ncol=5, frameon=False,
        fontsize=5.5, handlelength=1.4, columnspacing=0.65, handletextpad=0.35,
    )
    legend.get_texts()[int(np.flatnonzero(lengths == 30)[0])].set_weight("bold")
    paths = save_figure(fig, STEM, variant=FigureVariant.PAPER,
                        formats=("pdf", "png", "svg"), dpi=350)
    plt.close(fig)
    with (OUT / f"{STEM}_metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "duration_us", "cutoff", "amplitude_mhz", "accepted",
            "fwhm_khz", "contrast", "contrast_above_threshold",
        ])
        writer.writeheader()
        for ci, cutoff in enumerate(cutoffs):
            for ai, amp in enumerate(amps):
                writer.writerow(dict(
                    duration_us=30, cutoff=cutoff, amplitude_mhz=amp,
                    accepted=bool(accepted[ci, ai]),
                    fwhm_khz=data["fwhm_hz"][ci, ai] / 1000,
                    contrast=data["contrast"][ci, ai],
                    contrast_above_threshold=bool(data["contrast_above_threshold"][ci, ai]),
                ))
    np.savez_compressed(
        OUT / f"{STEM}.npz", duration_us=30.0, cutoffs=cutoffs, rabi_mhz=amps,
        contrast_threshold=CONTRAST_THRESHOLD, reference_linewidth_hz=reference_hz,
        slice_amplitude_index=slice_index, slice_amplitude_mhz=slice_amplitude,
        slice_durations_us=lengths, **slices, **data,
    )
    provenance = dict(
        generator="scripts/make_main_duration_cutoff_scan.py",
        included_in_manuscript=True, source_sha256=hashes,
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_configuration=config, figure_size_inches=[fig_width, fig_height],
        selection="Exact 30 us row (panels e/f) of the supplemental full-range survey",
        array_order="cutoff, nominal drive", reference_linewidth_hz=reference_hz,
        map_axes=dict(x="cutoff", y="nominal drive", display="Transposed arrays"),
        color_scales=scales, contrast_threshold=CONTRAST_THRESHOLD,
        cutoff_slices=dict(
            panels=["c", "d"], amplitude_index=slice_index,
            amplitude_mhz=slice_amplitude, requested_amplitude_mhz=50,
            selection="Nearest saved amplitude to 50 MHz; no interpolation",
            durations_us=lengths.tolist(),
            values="Same amplitude column for all five durations; rejected samples "
                   "are gaps. The 30 us slices exactly match the displayed maps.",
            highlighted_duration_us=30,
            highlight_style="Thick solid orange, drawn above thinner comparison curves",
            fwhm_axis_limits_normalized=[0, 15],
            contrast_axis_limits=[0, 1],
            map_markers="Orange dashed horizontal lines at the selected amplitude "
                        "in a/b, matching the highlighted 30 us curves",
            amplitude_annotation="Shown in caption only",
            contrast_reference="Dotted horizontal line at C=0.2 in panel d",
        ),
        amplitude_display=dict(
            scale="log", limits_mhz=[1, 100],
            positive_sample_count=int(positive.sum()),
            zero_drive="Omitted from the plot; retained in exported arrays and CSV",
            cell_edges="Original linear-amplitude cell edges; no resampling",
        ),
        white_contours="C=0.2, linearly interpolated between accepted samples with "
                       "corner_mask=False; the band between the curves has C>0.2. "
                       "These are contrast boundaries, not a linewidth threshold.",
        metric_mask="Rejected and unresolved dips are gray, not zero; local FWHM "
                    "does not certify a unique resonance.",
        contrast="Dip depth in excitation probability relative to a fitted linear "
                 "background at the dip; dimensionless and independent of shot count",
        validation="Exact equality to supplemental row and original 30 us scan; "
                   "source SHA-256 checksums verified; no smoothing or recomputation",
        summaries=summaries,
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
    print(json.dumps(summaries, indent=2))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
