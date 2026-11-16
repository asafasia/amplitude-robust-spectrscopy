"""Audit the sharp 50 us linewidth drop without changing manuscript figures.

PYTHONPATH=. MPLBACKEND=Agg .venv/bin/python \
    scripts/diagnose_fig4_50us_linewidth.py --workers 4
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from echospec.analysis.cutoff_metrics import cutoff_metrics
from echospec.figures.coherence import REFERENCE_FWHM_HZ
from echospec.simulation.polynomial_echo import (
    PolynomialEchoPulse,
    simulate_polynomial_echo,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/polynomial_duration_cutoff_poly2_log_40_50/n100"
OUT = ROOT / "outputs/fig4_50us_linewidth_diagnostic"


def run_case(specification):
    name, cutoff, dt_ns, span, intervals, amplitude, config = specification
    path = OUT / f"{name}.npz"
    pulse = PolynomialEchoPulse(
        duration_us=50.0, cutoff=float(cutoff), power=config["power"],
        transition_us=config["transition_us"], drag_beta=config["drag_beta"],
        anharmonicity_mhz=config["anharmonicity_mhz"],
    )
    detuning = np.linspace(-span, span, intervals + 1)
    metadata = dict(
        pulse=vars(pulse), dt_ns=dt_ns, detuning_mhz=detuning.tolist(),
        amplitude_mhz=amplitude, t1_us=config["t1_us"], t2_us=config["t2_us"],
        source_sha256=config["source_sha256"],
    )
    encoded = json.dumps(metadata, sort_keys=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            assert str(saved["configuration"]) == encoded
    else:
        population, diagnostics = simulate_polynomial_echo(
            pulse, [amplitude], detuning, t1_us=config["t1_us"],
            t2_us=config["t2_us"], dt_ns=dt_ns,
        )
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary, detuning_mhz=detuning, populations=population,
            configuration=encoded, diagnostics=json.dumps(diagnostics),
        )
        temporary.replace(path)
    return name


def profile_details(x, y, edge_band=(0.4, 0.5)):
    """Expose every crossing of the published background-relative half depth."""
    edges = (abs(x) >= edge_band[0] - 1e-12) & (abs(x) <= edge_band[1] + 1e-12)
    background = np.polyval(np.polyfit(x[edges], y[edges], 1), x)
    depth = background - y
    central = np.flatnonzero(abs(x) <= 0.25)
    peak = central[np.argmax(depth[central])]
    contrast = depth[peak]
    residual = depth - contrast / 2
    changes = np.flatnonzero(residual[:-1] * residual[1:] < 0)
    crossings = x[changes] - residual[changes] * (
        x[changes + 1] - x[changes]
    ) / (residual[changes + 1] - residual[changes])
    left, right = crossings[crossings < x[peak]], crossings[crossings > x[peak]]
    nearest = [float(left[-1]), float(right[0])] if left.size and right.size else []
    outer = [float(left[0]), float(right[-1])] if left.size and right.size else []
    minima, properties = find_peaks(-y, prominence=0.0001)
    return dict(
        background=background, normalized_depth=depth / contrast,
        crossings_mhz=crossings.tolist(), nearest_pair_mhz=nearest,
        outer_pair_mhz=outer, contrast=float(contrast),
        depth_peak_mhz=float(x[peak]), minima_mhz=x[minima].tolist(),
        minima_prominence=properties["prominences"].tolist(),
    )


def load_case(name):
    with np.load(OUT / f"{name}.npz", allow_pickle=False) as saved:
        return (saved["detuning_mhz"].copy(), saved["populations"][1, 0].copy(),
                json.loads(str(saved["configuration"])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((SOURCE / "dataset_manifest.json").read_text())["common_configuration"]
    for name, digest in config["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    scan_path = SOURCE / "simulations/L50us/scan.npz"
    with np.load(scan_path, allow_pickle=False) as saved:
        amplitudes = saved["rabi_mhz"]
        ai = int(np.argmin(abs(amplitudes - 50)))
        amplitude = float(amplitudes[ai])
        original_cutoffs = saved["cutoffs"].copy()
        original_width = saved["fwhm_hz"][:, ai].copy()
        original_contrast = saved["contrast"][:, ai].copy()
        accepted = saved["accepted"][:, ai].copy()
        original_x = saved["detuning_mhz"].copy()
        original_y = saved["populations"][:, 1, ai].copy()
    width_changes = np.where(accepted[:-1] & accepted[1:], np.diff(original_width), np.nan)
    before = int(np.nanargmin(width_changes))
    cutoffs = np.r_[
        np.geomspace(original_cutoffs[before], original_cutoffs[before + 1], 13),
        original_cutoffs[before + 2],
    ]
    specifications = [
        (f"local_{i:02d}", c, 0.25, 0.5, 1000, amplitude, config)
        for i, c in enumerate(cutoffs)
    ]
    refined_indices = [0, 6, 9, 10, 12, 13]
    specifications += [
        (f"fine_{i:02d}", cutoffs[i], 0.125, 0.5, 2000, amplitude, config)
        for i in refined_indices
    ]
    specifications += [
        (f"wide_{i:02d}", cutoffs[i], 0.125, 1.5, 3000, amplitude, config)
        for i in (0, 12)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for future in as_completed([executor.submit(run_case, item) for item in specifications]):
            print("Completed", future.result(), flush=True)

    local_rows = []
    for i, cutoff in enumerate(cutoffs):
        x, y, _ = load_case(f"local_{i:02d}")
        metrics = cutoff_metrics(x, y)
        details = profile_details(x, y)
        assert np.isclose(metrics["fwhm_hz"], np.diff(details["nearest_pair_mhz"])[0] * 1e6)
        local_rows.append(dict(
            cutoff=float(cutoff), fwhm_normalized=metrics["fwhm_hz"] / REFERENCE_FWHM_HZ,
            center_hz=metrics["center_hz"], contrast=metrics["contrast"],
            crossing_count=len(details["crossings_mhz"]),
            crossings_mhz=details["crossings_mhz"],
            outer_width_normalized=float(np.diff(details["outer_pair_mhz"])[0] * 1e6 / REFERENCE_FWHM_HZ),
        ))
    refinements = []
    for i in refined_indices:
        x, y, _ = load_case(f"local_{i:02d}")
        fx, fy, _ = load_case(f"fine_{i:02d}")
        np.testing.assert_allclose(fx[::2], x, rtol=0, atol=1e-15)
        coarse, finer_time, finer_both = (
            cutoff_metrics(x, y), cutoff_metrics(x, fy[::2]), cutoff_metrics(fx, fy)
        )
        refinements.append(dict(
            cutoff=float(cutoffs[i]),
            max_population_time_change=float(abs(y - fy[::2]).max()),
            time_width_change_normalized=abs(coarse["fwhm_hz"] - finer_time["fwhm_hz"]) / REFERENCE_FWHM_HZ,
            frequency_width_change_normalized=abs(finer_time["fwhm_hz"] - finer_both["fwhm_hz"]) / REFERENCE_FWHM_HZ,
            finer_fwhm_normalized=finer_both["fwhm_hz"] / REFERENCE_FWHM_HZ,
            crossing_count=len(profile_details(fx, fy)["crossings_mhz"]),
        ))
    background_checks = []
    for i in (0, 12):
        x, y, _ = load_case(f"wide_{i:02d}")
        for band in ((0.4, 0.5), (0.6, 0.8), (1.0, 1.5)):
            detail = profile_details(x, y, band)
            background_checks.append(dict(
                cutoff=float(cutoffs[i]), edge_band_mhz=band,
                contrast=detail["contrast"], crossings_mhz=detail["crossings_mhz"],
                fwhm_normalized=float(np.diff(detail["nearest_pair_mhz"])[0] * 1e6 / REFERENCE_FWHM_HZ)
                if detail["nearest_pair_mhz"] else None,
            ))

    plt.rcParams.update({"font.size": 9, "axes.grid": False, "savefig.bbox": "tight"})
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), layout="constrained")
    region = slice(before - 3, before + 4)
    axes[0].plot(original_cutoffs[region] * 1000, original_width[region] / REFERENCE_FWHM_HZ,
                 "o--", color="0.6", ms=4, label="Original 5 kHz grid")
    axes[0].plot(cutoffs * 1000, [row["fwhm_normalized"] for row in local_rows],
                 ".-", color="black", label="Dense cutoff scan, 1 kHz grid")
    axes[0].plot(cutoffs * 1000, [row["outer_width_normalized"] for row in local_rows],
                 "--", color="#009e73", label="Outermost half-depth span")
    axes[0].set(xlabel=r"Cutoff $c$ ($\times10^{-3}$)", ylabel=r"Width / $\Gamma_{T_2^*}$",
                title="(a) Reported width changes branch")
    axes[0].set_ylim(2, 9.2)
    axes[0].legend(fontsize=7, frameon=False, loc="upper right")
    colors = ("#0072b2", "#d55e00")
    for i, color in zip((0, 12), colors, strict=True):
        x, y, _ = load_case(f"fine_{i:02d}")
        details = profile_details(x, y)
        label = rf"$c={cutoffs[i]:.5f}$"
        axes[1].plot(x * 1000, y, color=color, label=label)
        axes[1].plot(x * 1000, details["background"] - details["contrast"] / 2,
                     ":", color=color, lw=1)
        axes[2].plot(x * 1000, details["normalized_depth"], color=color, label=label)
        cross = np.asarray(details["crossings_mhz"])
        axes[2].scatter(cross * 1000, np.full(cross.size, 0.5), color=color, s=20, zorder=5)
    axes[1].set(xlabel=r"Detuning $\Delta/(2\pi)$ (kHz)", ylabel=r"Excited population $P_e$",
                xlim=(-240, 240), title="(b) Raw spectra and half-depth levels")
    axes[1].legend(fontsize=8, frameon=False)
    axes[2].axhline(0.5, color="0.5", ls=":", lw=1)
    axes[2].set(xlabel=r"Detuning $\Delta/(2\pi)$ (kHz)", ylabel="Dip depth / central depth",
                xlim=(-200, 200), ylim=(0, 1.05), title="(c) Two crossings become six")
    for ax in axes:
        ax.tick_params(direction="in", top=True, right=True)
    fig.savefig(OUT / "linewidth_drop_diagnostic.png", dpi=220)
    plt.close(fig)

    original_rows = []
    for i in (before, before + 1):
        detail = profile_details(original_x, original_y[i])
        original_rows.append(dict(
            cutoff=float(original_cutoffs[i]), fwhm_normalized=float(original_width[i] / REFERENCE_FWHM_HZ),
            contrast=float(original_contrast[i]), crossings_mhz=detail["crossings_mhz"],
        ))
    summary = dict(
        amplitude_mhz=amplitude, duration_us=50, original_indices=[before, before + 1],
        reference_linewidth_hz=REFERENCE_FWHM_HZ, original=original_rows,
        local_scan=local_rows, refinements=refinements, background_checks=background_checks,
        source_dataset=str(SOURCE.relative_to(ROOT)), source_sha256=config["source_sha256"],
        original_scan_sha256=hashlib.sha256(scan_path.read_bytes()).hexdigest(),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interpretation="Nearest background-relative half-depth crossings change from the outer trough to the central dip when internal shoulders cross half depth. No parametric line-shape fit is used. The outermost half-depth span is shown to diagnose the crossing switch, not as a replacement manuscript metric.",
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (OUT / "local_scan.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(local_rows[0]))
        writer.writeheader()
        writer.writerows(local_rows)
    print(json.dumps(dict(original=original_rows, refinements=refinements,
                         background_checks=background_checks), indent=2), flush=True)


if __name__ == "__main__":
    main()
