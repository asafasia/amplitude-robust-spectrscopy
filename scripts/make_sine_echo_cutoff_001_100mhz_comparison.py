"""Repeat the 16 ns / 100 amplitude negative-beta sweep at cutoff 0.001 and nominal amplitudes up to 100 MHz.

Run: .venv/bin/python scripts/make_sine_echo_cutoff_001_100mhz_comparison.py
Use --plot-only to rebuild the report from saved arrays.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from echospec.analysis.dip_metrics import dip_metrics  # noqa: E402
from scripts import make_sine_echo_beta_comparison as comparison  # noqa: E402

OUT = ROOT / "outputs/sine_echo_cutoff_001_100mhz_comparison"
BETAS = (0.0, -0.5, -1.0)


def configure():
    comparison.OUT = OUT
    comparison.BETAS = BETAS
    comparison.RABI_MAX_MHZ = 100.0
    comparison.CONFIG = {**comparison.CONFIG, "cutoff": 0.001}


def run_case(beta):
    configure()
    comparison.run_case(beta)


def refine_detuning():
    path = OUT / "detuning_refinement.json"
    if path.exists():
        return
    data = dict(np.load(OUT / "beta-0.5_narrow.npz"))
    metrics = [dip_metrics(data["detuning_mhz"], y) for y in data["populations"][1]]
    centers = np.array([m["center_hz"] for m in metrics])
    indices = sorted(set([int(np.nanargmax(abs(centers))), 46, 99]))
    detuning = np.linspace(-0.5, 0.5, 401)
    populations = comparison.simulate(-0.5, detuning, data["rabi_mhz"][indices])
    rows = []
    for j, i in enumerate(indices):
        fine = dip_metrics(detuning, populations[1, j])
        rows.append(
            dict(
                rabi_mhz=float(data["rabi_mhz"][i]),
                coarse=metrics[i],
                fine=fine,
                center_change_hz=fine["center_hz"] - metrics[i]["center_hz"],
                width_change_hz=fine["fwhm_hz"] - metrics[i]["fwhm_hz"],
            )
        )
    np.savez_compressed(
        OUT / "beta-0.5_detuning_refinement.npz",
        detuning_mhz=detuning,
        rabi_mhz=data["rabi_mhz"][indices],
        populations=populations,
    )
    path.write_text(json.dumps(rows, indent=2))
    print("Detuning refinement complete", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    configure()
    if not args.plot_only:
        pending = [
            b
            for b in BETAS
            if not all(
                (OUT / f"beta{b:g}_{s}.npz").exists()
                for s in ("narrow", "broad", "convergence")
            )
        ]
        with ProcessPoolExecutor(max_workers=3) as pool:
            list(pool.map(run_case, pending))
        refine_detuning()
    comparison.plots()
    path = OUT / "detuning_refinement.json"
    if path.exists():
        rows = json.loads(path.read_text())
        provenance = OUT / "provenance.json"
        metadata = json.loads(provenance.read_text())
        metadata["detuning_grid_refinement"] = rows
        metadata["parameter_change"] = (
            "Cutoff 0.001; nominal amplitude range extended to 0–100 MHz with 100 points; all beta cases recomputed."
        )
        provenance.write_text(json.dumps(metadata, indent=2))
        center = max(abs(r["center_change_hz"]) for r in rows)
        width = max(abs(r["width_change_hz"]) for r in rows)
        report = OUT / "report.html"
        report.write_text(
            report.read_text()
            + f"<h2>Detuning-grid check</h2><p>Halving spacing from 5 to 2.5 kHz at selected β=−0.5 amplitudes changed the center by at most {center:.3f} Hz and width by {width:.1f} Hz.</p>"
        )


if __name__ == "__main__":
    main()
