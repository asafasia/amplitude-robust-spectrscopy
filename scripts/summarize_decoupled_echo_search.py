"""Summarize the saved finite pulse searches without additional simulations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/decoupled_echo_50mhz_20261101"
CAMPAIGNS = ("initial", "core_phase", "strict_local", "preserve_original",
             "preserve_original_local")


def summarize(output):
    records, campaigns, hashes = [], {}, {}
    for campaign in CAMPAIGNS:
        directory = output / campaign
        paths = sorted((directory / "evaluations").glob("*.json"))
        if not paths:
            continue
        rows = [json.loads(path.read_text()) for path in paths]
        config = directory / "configuration.json"
        hashes[str(config.relative_to(output))] = hashlib.sha256(config.read_bytes()).hexdigest()
        best = min(rows, key=lambda row: row["loss"])
        campaigns[campaign] = dict(
            saved_evaluations=len(rows), best_quality=best["quality"],
            accepted_under_campaign_constraints=sum(r["quality"]["feasible"] for r in rows),
            contrast_floor_scope=("50 MHz only" if campaign.startswith("preserve")
                                  else "5--60 MHz"),
        )
        records.extend(rows)
    data = np.array([[r["quality"].get(key, np.nan) for key in
                      ("width50_khz", "min_contrast", "max_width_regression_khz")]
                     for r in records], dtype=float)
    valid = np.isfinite(data).all(axis=1) & (data[:, 0] < 400)
    data = data[valid]
    contrast_pass = data[:, 1] > .4
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for ax, column in zip(axes, (1, 2), strict=True):
        ax.scatter(data[:, column], data[:, 0], s=9, color="#81919d", alpha=.14,
                   edgecolors="none", rasterized=True, label="Evaluated pulse shapes")
        ax.scatter(data[contrast_pass, column], data[contrast_pass, 0], s=11,
                   color="#c76b22", alpha=.45, edgecolors="none", rasterized=True,
                   label="Contrast > 0.4 at all training amplitudes")
        ax.axhline(1000 / (np.pi * 7.31), color="#111111", ls="--", lw=1,
                   label="T₂ linewidth reference")
        ax.set(ylabel="FWHM at 50 MHz (kHz)", ylim=(35, 250))
    axes[0].axvline(.4, color="#c76b22", ls=":", lw=1)
    axes[0].set(xlim=(0, .6), xlabel="Minimum contrast over trained 5–60 MHz points")
    axes[1].axvline(.25, color="#28746a", ls=":", lw=1)
    axes[1].axvspan(-50, .25, color="#28746a", alpha=.07)
    axes[1].set(xlim=(-20, 80), xlabel="Worst linewidth regression vs original (kHz)")
    axes[0].legend(loc="upper left", fontsize=8)
    result_path = output / "results.json"
    if result_path.exists():
        results = json.loads(result_path.read_text())
        baseline = {r["rabi_mhz"]: r for r in results["cases"]["baseline"]["rows"]}
        for name, case in results["cases"].items():
            finite = [r for r in case["rows"] if r["accepted"]]
            if not finite:
                continue
            contrast = min(r["contrast"] for r in finite)
            regression = max((r["fwhm_hz"] - baseline[r["rabi_mhz"]]["fwhm_hz"]) / 1000
                             for r in finite)
            width = case["at50"]["fwhm_hz"] / 1000
            for ax, x in zip(axes, (contrast, regression), strict=True):
                if ax.get_xlim()[0] <= x <= ax.get_xlim()[1]:
                    ax.scatter(x, width, marker="D", s=40, color="#163c59", zorder=5)
                    ax.annotate(name.replace("_", " "), (x, width), xytext=(5, -11),
                                textcoords="offset points", fontsize=8, color="#163c59")
        hashes["results.json"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
    fig.suptitle("Finite search: contrast and linewidth tradeoffs", fontsize=14)
    fig.supxlabel("Training points: 5, 7.5, 10, 15, 20, 30, 40, 50, 60 MHz. "
                  "Diamonds: dense-grid validated trials. This is not a no-go proof.", fontsize=9)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"search_tradeoffs.{extension}", dpi=180)
    plt.close(fig)
    summary = dict(campaigns=campaigns, saved_evaluations=len(records),
                   finite_search_only=True, input_sha256=hashes,
                   generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output / "search_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    summarize(parser.parse_args().output)
