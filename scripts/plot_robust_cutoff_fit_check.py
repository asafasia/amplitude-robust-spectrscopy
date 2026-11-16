"""Compare direct crossing widths with robust fits to cached 50 us spectra.

PYTHONPATH=. MPLBACKEND=Agg .venv/bin/python scripts/plot_robust_cutoff_fit_check.py
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from echospec.analysis.robust_dip_fit import evaluate_dip_fit, robust_dip_fit
from echospec.figures.coherence import REFERENCE_FWHM_HZ
from scripts.diagnose_fig4_50us_linewidth import OUT, load_case

ROOT = Path(__file__).resolve().parents[1]
STEM = "robust_fit_comparison"


def main():
    rows = list(csv.DictReader((OUT / "local_scan.csv").open()))
    cutoffs = np.array([float(row["cutoff"]) for row in rows])
    direct = np.array([float(row["fwhm_normalized"]) for row in rows])
    fitted, hashes = [], {}
    for index, cutoff in enumerate(cutoffs):
        name = f"local_{index:02d}"
        x, y, config = load_case(name)
        np.testing.assert_allclose(config["pulse"]["cutoff"], cutoff)
        fit = robust_dip_fit(x, y)
        assert fit["accepted"], (name, fit["status"])
        fitted.append(fit["fwhm_hz"] / REFERENCE_FWHM_HZ)
        path = OUT / f"{name}.npz"
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": .8,
        "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 7,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
    })
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), layout="constrained")
    ax = axes[0]
    ax.plot(cutoffs * 1e3, direct, "o--", color=".65", ms=3,
            label="Old crossing rule")
    ax.plot(cutoffs * 1e3, fitted, "o-", color="#0072b2", ms=3,
            label="Local Gaussian fit")
    ax.set(title="(a) Central width across the drop", ylim=(0, 8.5),
           xlabel=r"Cutoff $c$ ($\times 10^{-3}$)",
           ylabel=r"FWHM / $\Gamma_{T_2^*}$")
    ax.legend(frameon=False, loc="upper right")
    arrays = dict(cutoff=cutoffs, direct_normalized_width=direct,
                  fitted_normalized_width=np.array(fitted))
    records = []
    cases = zip(("fine_00", "fine_12"), ("#0072b2", "#d55e00"), strict=True)
    for index, (name, color) in enumerate(cases, start=1):
        x, y, config = load_case(name)
        fit = robust_dip_fit(x, y)
        assert fit["accepted"], (name, fit["status"])
        ax = axes[index]
        prediction = evaluate_dip_fit(x, fit)
        v = (x - fit["reference_mhz"]) / fit["radius_mhz"]
        b0, b1, b2 = fit["parameters"][3:]
        background = fit["y_offset"] + fit["y_scale"] * (b0 + b1 * v + b2 * v**2)
        width = fit["fwhm_hz"] / REFERENCE_FWHM_HZ
        ax.plot(x * 1000, y, color=".6", lw=1.8, label="Simulated spectrum")
        ax.plot(x * 1000, prediction, "--", color=color, lw=1.8,
                label="Gaussian + background")
        ax.plot(x * 1000, background, ":", color=color, lw=1,
                label="Local background")
        label = "(b) Before" if index == 1 else "(c) After"
        ax.set(title=label + rf": $c={config['pulse']['cutoff']:.5f}$",
               xlim=(-140, 140), ylim=(.355, .407),
               xlabel=r"Detuning $\Delta/(2\pi)$ (kHz)", ylabel=r"$P_e$")
        ax.text(0, .360, rf"FWHM / $\Gamma_{{T_2^*}}$ = {width:.2f}",
                ha="center", color=color, fontsize=9)
        ax.legend(frameon=False, loc="upper right", handlelength=1.8)
        arrays.update({f"{name}_{key}": value for key, value in {
            "detuning_mhz": x, "population": y, "fitted_population": prediction,
            "local_background": background,
        }.items()})
        records.append(dict(case=name, cutoff=config["pulse"]["cutoff"], fit=fit))
        path = OUT / f"{name}.npz"
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()

    fig.savefig(OUT / f"{STEM}.png", dpi=220, facecolor="white")
    plt.close(fig)
    np.savez_compressed(OUT / f"{STEM}.npz", **arrays)
    for path in (OUT / "local_scan.csv", Path(__file__).resolve(),
                 ROOT / "echospec/analysis/robust_dip_fit.py",
                 ROOT / "echospec/analysis/dip_metrics.py"):
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / f"{STEM}_provenance.json").write_text(json.dumps(dict(
        generator=str(Path(__file__).resolve().relative_to(ROOT)),
        source_sha256=hashes, reference_linewidth_hz=REFERENCE_FWHM_HZ,
        cases=records, note="Fits are independent. No smoothing across cutoffs. "
                           "Widths and depths refer to the Gaussian central component.",
    ), indent=2) + "\n")
    print(OUT / f"{STEM}.png")


if __name__ == "__main__":
    main()
