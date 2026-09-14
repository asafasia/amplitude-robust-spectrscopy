"""Render the September q6 pulse-length comparison and its paper data package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from echospec.figures.style import apply_figure_style, save_figure  # noqa: E402
from echospec.paper_data import save_paper_dataset  # noqa: E402

STEM = "12_dense_pulse_length"
CAMPAIGN = Path("pulse_length_spectroscopy/overnight_dense_20260909")
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#7D5BA6", "#8C6D31", "#303030"]


def export_source(source: Path) -> None:
    campaign = source / CAMPAIGN
    measured_path = campaign / "measured_analysis/analysis.json"
    simulated_path = campaign / "simulation_comparison/comparison.json"
    measured = json.loads(measured_path.read_text(encoding="utf-8"))
    simulated = json.loads(simulated_path.read_text(encoding="utf-8"))
    manifest = measured["manifest"]
    device = manifest["device"]
    validation = json.loads((campaign / "validation.json").read_text())
    assert all(row["metrics"]["drag_beta"] == 0 for row in validation)
    assert not manifest["ac_stark_correction"]
    assert [d["length"] for d in measured["datasets"]] == [
        d["length"] for d in simulated["results"]
    ]
    assert len(measured["datasets"]) == 10
    amps = np.array([d["amps"] for d in measured["datasets"]])
    factor = 1 / (device["x180_amplitude_v"] * 2 * device["x180_length_ns"] * 1e-3)
    source_files = [
        measured_path,
        simulated_path,
        campaign / "status.json",
        campaign / "validation.json",
        campaign / "build_analysis_report.py",
        campaign / "simulation_comparison/analyze_comparison.py",
        campaign / "simulation_comparison/run_simulations.py",
        campaign / "simulation_comparison/qutrit_fast.cpp",
        campaign / "simulation_comparison/validation.json",
    ]
    provenance = {
        "generator": "scripts/make_dense_pulse_length_comparison.py",
        "scope": "Supplemental: dense pulse-length dependence, q6, September 9-10 2026",
        "source_root": "OPX1000_DATA_DIR (default sibling opx1000-codes/data)",
        "source_files": [
            {
                "path": f.relative_to(source).as_posix(),
                "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
            }
            for f in source_files
        ],
        "raw_run_directories": [
            str(r["run_directory"]).replace("\\", "/").split("/data/", 1)[1]
            for r in manifest["runs"]
        ],
        "device": device,
        "acquisition": {
            k: manifest[k]
            for k in [
                "qubit",
                "echo",
                "pulse_shape",
                "cutoff",
                "num_shots",
                "frequency_span_mhz",
                "frequency_points",
                "amplitude_points",
                "min_amplitude_v",
                "max_amplitude_v",
                "ac_stark_correction",
                "stark_kappa_mhz_inv",
            ]
        },
        "pulse_template_mode": "template length equals pulse length; no duration stretching",
        "drag_beta": 0.0,
        "kappa_note": "Stored 0.005 MHz^-1 is inactive; no AC-Stark compensation applied.",
        "rabi_conversion": {
            "mhz_per_v": factor,
            "definition": "f_R=Omega/(2*pi)=A/(2*A_pi*t_pi)",
        },
        "coherence_reference_khz": measured["reference_khz"],
        "coherence_note": "1/(pi*T2_star) from this q6 campaign; not the paper's q1 normalization",
        "simulation_detuning_convention": simulated["detuning_convention"],
        "simulation_validation": simulated["validation"],
        "linewidth_method": "9-point order-2 Savitzky-Golay; linear baseline from outer 10% edges; strongest residual peak/dip; nearest half-height crossings",
        "experimental_acceptance": "SNR >= 5; both crossings; FWHM >= 3 bins and <= 80% scan",
        "simulation_acceptance": "contrast >= 0.001; both crossings; FWHM 15.038-1600 kHz; no SNR gate",
        "pairing": "same pulse length and original amplitude index; intersection of acceptance masks",
        "display": "lengths <=25 us; accepted experimental points and simulated curves; no new fit or alignment",
    }
    for category, datasets in [
        ("experimental", measured["datasets"]),
        ("numerical", simulated["results"]),
    ]:
        arrays = {
            "pulse_length_us": np.array([d["length"] for d in datasets]),
            "amplitude_v": amps,
            "rabi_frequency_mhz": amps * factor,
        }
        for key in ["fwhm_khz", "contrast", "left_khz", "right_khz"]:
            arrays[key] = np.array(
                [
                    [f[key] if f[key] is not None else np.nan for f in d["fits"]]
                    for d in datasets
                ]
            )
        arrays["accepted"] = np.array(
            [[f["accepted"] for f in d["fits"]] for d in datasets], dtype=bool
        )
        arrays["feature_is_peak"] = np.array(
            [[f["polarity"] == "peak" for f in d["fits"]] for d in datasets]
        )
        if category == "experimental":
            arrays["snr"] = np.array(
                [
                    [f["snr"] if f["snr"] is not None else np.nan for f in d["fits"]]
                    for d in datasets
                ]
            )
        save_paper_dataset(
            STEM, category=category, arrays=arrays, provenance=provenance
        )


def load_package(category: str):
    folder = ROOT / "paper/data" / category
    with np.load(folder / f"{STEM}.npz", allow_pickle=False) as cache:
        arrays = {k: cache[k] for k in cache.files}
    provenance = json.loads((folder / f"{STEM}.json").read_text())["provenance"]
    return arrays, provenance


def generate(source: Path | None = None) -> None:
    if source is not None:
        export_source(source.resolve())
    exp, provenance = load_package("experimental")
    sim, sim_provenance = load_package("numerical")
    assert np.array_equal(exp["pulse_length_us"], sim["pulse_length_us"])
    assert np.array_equal(exp["rabi_frequency_mhz"], sim["rabi_frequency_mhz"])
    reference = provenance["coherence_reference_khz"]
    assert np.isclose(reference, 1 / (np.pi * provenance["device"]["t2_star_s"]) / 1000)
    indices = np.flatnonzero(exp["pulse_length_us"] <= 25)
    assert np.array_equal(exp["pulse_length_us"][indices], [1, 2, 5, 10, 15, 20, 25])
    apply_figure_style("paper")
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )
    fig, ax = plt.subplots(figsize=(7, 4.7), layout="constrained")
    summary = []
    for i, color in zip(indices, COLORS, strict=True):
        x = exp["rabi_frequency_mhz"][i]
        ew = np.where(exp["accepted"][i], exp["fwhm_khz"][i], np.nan)
        sw = np.where(sim["accepted"][i], sim["fwhm_khz"][i], np.nan)
        ax.plot(x, sw, color=color, ls="--", lw=1)
        ax.scatter(x, ew, color=color, s=7, alpha=0.7, edgecolors="none")
        paired = exp["accepted"][i] & sim["accepted"][i]
        summary.append(
            {
                "length_us": float(exp["pulse_length_us"][i]),
                "pairs": int(paired.sum()),
                "linewidth_rmse_khz": float(
                    np.sqrt(np.mean((ew[paired] - sw[paired]) ** 2))
                ),
                "median_exp_over_sim": float(np.median(ew[paired] / sw[paired])),
            }
        )
    handles = [
        Line2D([], [], color=c, lw=1.5, label=rf"${exp['pulse_length_us'][i]:g}\,\mu$s")
        for i, c in zip(indices, COLORS, strict=True)
    ]
    legend = ax.legend(
        handles=handles,
        title="Pulse / template length",
        loc="upper center",
        ncol=7,
        frameon=False,
        columnspacing=1,
        handlelength=1.6,
    )
    ax.add_artist(legend)
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                ls="none",
                color=".4",
                markersize=3,
                label="Experiment",
            ),
            Line2D([], [], ls="--", color=".4", lw=1, label="Simulation"),
        ],
        loc="lower right",
        frameon=False,
    )
    ax.set(
        xscale="log",
        yscale="log",
        xlim=(exp["rabi_frequency_mhz"].min(), exp["rabi_frequency_mhz"].max()),
        ylim=(10, 2000),
        xlabel=r"Nominal peak Rabi frequency $f_R$ (MHz)",
        ylabel="FWHM (kHz)",
    )
    ax.set_xticks([1, 2, 5, 10, 20, 40])
    ax.set_yticks([10, 20, 50, 100, 200, 500, 1000, 2000])
    fmt = FuncFormatter(lambda value, _: f"{value:g}")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    right = ax.secondary_yaxis(
        "right", functions=(lambda w: w / reference, lambda r: r * reference)
    )
    right.set_ylabel(r"FWHM / $\Gamma_{T_2^*,q6}$")
    right.set_yticks([0.2, 0.5, 1, 2, 5, 10, 20])
    right.yaxis.set_major_formatter(fmt)
    right.yaxis.set_minor_formatter(NullFormatter())
    ax.axhline(reference, color=".4", ls=":", lw=1.1)
    ax.annotate(
        rf"$\Gamma_{{T_2^*,q6}}={reference:.1f}$ kHz",
        xy=(0.98, reference),
        xycoords=ax.get_yaxis_transform(),
        xytext=(0, 5),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
    )
    ax.grid(alpha=0.18, which="major")
    fig.canvas.draw()
    assert np.allclose(right.get_ylim(), np.array(ax.get_ylim()) / reference)
    save_figure(
        fig, "12_pulse_length_linewidth", variant="paper", formats=("pdf", "png", "svg")
    )
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.2, 4.4), layout="constrained")
    total = 0
    for i, color in zip(indices, COLORS, strict=True):
        pair = exp["accepted"][i] & sim["accepted"][i]
        total += int(pair.sum())
        ax.scatter(
            sim["fwhm_khz"][i, pair],
            exp["fwhm_khz"][i, pair],
            s=8,
            alpha=0.65,
            color=color,
            edgecolors="none",
            label=rf"${exp['pulse_length_us'][i]:g}\,\mu$s ($n={pair.sum()}$)",
        )
    ax.plot([15, 1600], [15, 1600], ls="--", color=".4", lw=1)
    ax.set(
        xscale="log",
        yscale="log",
        xlim=(15, 1600),
        ylim=(15, 1600),
        xlabel="Simulated FWHM (kHz)",
        ylabel="Experimental FWHM (kHz)",
    )
    ax.set_aspect("equal", adjustable="box")
    ticks = [20, 50, 100, 200, 500, 1000]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.legend(loc="upper left", title=f"Paired linewidths (N={total})", frameon=False)
    ax.grid(alpha=0.18)
    assert total == 1278
    save_figure(
        fig,
        "12_pulse_length_paired_linewidths",
        variant="paper",
        formats=("pdf", "png", "svg"),
    )
    plt.close(fig)
    # Refresh each documented data/provenance pair together with the figures.
    for category, arrays, meta in [
        ("experimental", exp, provenance),
        ("numerical", sim, sim_provenance),
    ]:
        save_paper_dataset(STEM, category=category, arrays=arrays, provenance=meta)
    (ROOT / "figures/paper/12_pulse_length_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-data-dir",
        type=Path,
        help="opx1000-codes/data root; reimport saved analyses",
    )
    args = parser.parse_args()
    source = args.source_data_dir
    if source is None and os.environ.get("OPX1000_DATA_DIR"):
        source = Path(os.environ["OPX1000_DATA_DIR"])
    if source is None and not (ROOT / f"paper/data/experimental/{STEM}.npz").exists():
        source = ROOT.parent / "opx1000-codes/data"
    generate(source)


if __name__ == "__main__":
    main()
