"""Compare existing matched simulations for the polynomial echo pulse note."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ars-matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from echospec.analysis.spectroscopy_quality import spectroscopy_quality  # noqa: E402

OUT = ROOT / "figures/paper/polynomial_echo_note"
SOURCES = {
    "hard": "outputs/polynomial_power_30us_scan/c0.001_hard.npz",
    "polynomial5": "outputs/optimized_echo_tail/polynomial5_full.npz",
    "c2": "outputs/optimized_echo_tail/c2_full.npz",
}
LABELS = ["Hard cutoff", "Polynomial p = 5 + cutoff subtraction", "C² subtraction"]
COLORS = ["#586579", "#176eaa", "#d47823"]
STYLES = ["-", "--", "-."]
COMMON = dict(duration_us=30., anharmonicity_mhz=-216., drag_beta=-.5,
              echo=True, echo_transition_us=.016, order=.5,
              stark_kappa_mhz_inv=0., t1_us=51.24,
              t_phi_us=7.871480508563624)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    amps, det = np.linspace(0, 100, 100), np.linspace(-.5, .5, 201)
    rows, summaries, configs, populations = [], {}, {}, {}
    for family, source in SOURCES.items():
        with np.load(ROOT / source) as data:
            cfg = json.loads(str(data["config"]))
            for key, value in COMMON.items():
                assert cfg[key] == value, (family, key, cfg[key], value)
            assert cfg.get("cutoff", cfg.get("cutoff_core")) == .001
            dt = (cfg["dt_ns"] if "dt_ns" in cfg else
                  cfg["duration_us"] * 1000 / (2 * cfg["num_steps_per_half"]))
            assert dt == .5
            if family == "hard":
                assert cfg["edge_window_kind"] == "none"
            else:
                assert cfg["family"] == family and cfg["parameters"] is None
                np.testing.assert_allclose(cfg["sigma_us"], 15 / np.sqrt(.001**-2 - 1))
            np.testing.assert_array_equal(data["rabi_mhz"], amps)
            np.testing.assert_array_equal(data["detuning_mhz"], det)
            p = data["populations"]
        assert p.shape == (3, 100, 201) and np.isfinite(p).all()
        np.testing.assert_allclose(p.sum(axis=0), 1., atol=1e-8)
        assert p.min() >= -1e-10 and p.max() <= 1 + 1e-10
        configs[family], populations[family] = cfg, p
        local = [dict(family=family, amplitude_mhz=float(a),
                      **spectroscopy_quality(det, y))
                 for a, y in zip(amps, p[1], strict=True)]
        assert not local[0]["accepted"]
        active = [r for r in local if r["amplitude_mhz"] >= 10]
        assert all(r["accepted"] for r in active)
        summaries[family] = {
            "operating_range_mhz": [float(amps[10]), 100.],
            "minimum_snr": min(r["contrast_snr"] for r in active),
            "maximum_fwhm_khz": max(r["fwhm_hz"] / 1000 for r in active),
            "at_100_mhz": local[-1],
        }
        rows.extend(local)
    with (OUT / "tradeoff_metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(OUT / "tradeoff_populations.npz", rabi_mhz=amps,
                        detuning_mhz=det, **populations)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": .18, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), layout="constrained")
    for family, label, color, style in zip(
            SOURCES, LABELS, COLORS, STYLES, strict=True):
        local = [r for r in rows if r["family"] == family]
        axes[0].plot(amps, [r["fwhm_hz"] / 1000 for r in local],
                     color=color, ls=style, lw=2.3, label=label)
        axes[1].plot(amps, [r["contrast_snr"] for r in local],
                     color=color, ls=style, lw=2.3)
        for ax, key, scale in [(axes[0], "fwhm_hz", .001),
                                (axes[1], "contrast_snr", 1)]:
            value = local[-1][key] * scale
            ax.plot(100, value, "o", color=color, ms=4)
            ax.annotate(f"{value:.1f}", (100, value), xytext=(5, 0),
                        textcoords="offset points", color=color, va="center",
                        fontsize=10)
    t2_width = 1000 / (np.pi * 7.31)
    axes[0].axhline(t2_width, color="#70439b", lw=1.3, ls=":")
    axes[0].text(33, t2_width + 9, "T₂ reference: 43.5 kHz", color="#70439b",
                 fontsize=10)
    axes[1].axhline(10, color="#70439b", lw=1.3, ls=":")
    axes[1].text(48, 10.25, "SNR target = 10", color="#70439b", fontsize=10)
    axes[0].set(title="a   Linewidth — lower is better", ylabel="Dip FWHM (kHz)",
                ylim=(0, 550))
    axes[1].set(title="b   Ideal contrast SNR — higher is better",
                ylabel="SNR (1,000 total shots)", ylim=(0, 20))
    for ax in axes:
        ax.set(xlim=(0, 111), xlabel="Nominal drive amplitude A (MHz)",
               xticks=np.arange(0, 101, 20))
        ax.axvspan(0, 10, color="#b6bdc7", alpha=.14)
    fig.suptitle("Linewidth and visibility tradeoff for the three pulse shapes\n"
                 "30 µs · cutoff 0.001 · 16 ns echo transition · β = −0.5",
                 fontsize=15)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center",
               ncol=3, fontsize=10, frameon=False)
    fig.savefig(OUT / "tradeoff.png", dpi=180)
    fig.savefig(OUT / "tradeoff.pdf")
    plt.close(fig)
    provenance = {
        "common_parameters": COMMON, "t2_us": 7.31, "cutoff": .001,
        "dt_ns": .5, "amplitude_points": 100, "detuning_points": 201,
        "simulation_status": "Reused matched existing three-level simulation maps",
        "source_configs": configs,
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                          for p in [*SOURCES.values(),
                                    "scripts/plot_polynomial_echo_tradeoff.py",
                                    "echospec/analysis/spectroscopy_quality.py",
                                    "echospec/analysis/dip_metrics.py"]},
        "snr": "500 shots at center; 250 at each +/-0.45 MHz shoulder; ideal Bernoulli",
        "fwhm": "Nearest half-depth crossings; linear background at |detuning| >=0.4 MHz",
        "zero_amplitude": "No dip; width and SNR are undefined and omitted",
        "reproduce": "python scripts/plot_polynomial_echo_tradeoff.py",
        "summaries": summaries,
    }
    (OUT / "tradeoff_provenance.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
