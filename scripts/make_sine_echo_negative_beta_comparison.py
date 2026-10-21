"""Compare beta=0,-0.5,-1 with the established 16 ns / 100 amplitude protocol.

Run: .venv/bin/python scripts/make_sine_echo_negative_beta_comparison.py
Reuses the verified beta=0 arrays from the positive-beta comparison when present.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import make_sine_echo_beta_comparison as comparison  # noqa: E402

OUT = ROOT / "outputs/sine_echo_negative_beta_comparison"
BETAS = (0.0, -0.5, -1.0)


def configure():
    comparison.OUT = OUT
    comparison.BETAS = BETAS


def run_case(beta):
    configure()
    comparison.run_case(beta)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    original = ROOT / "outputs/sine_echo_beta_comparison"
    for suffix in ("narrow", "broad", "convergence"):
        source = original / f"beta0_{suffix}.npz"
        target = OUT / source.name
        if source.exists() and not target.exists():
            shutil.copy2(source, target)
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
        with ProcessPoolExecutor(max_workers=2) as pool:
            list(pool.map(run_case, pending))
    comparison.plots()
    refinement = OUT / "detuning_refinement.json"
    if refinement.exists():
        rows = json.loads(refinement.read_text())
        center_error = max(abs(row["center_change_hz"]) for row in rows)
        width_error = max(abs(row["width_change_hz"]) for row in rows)
        provenance_path = OUT / "provenance.json"
        provenance = json.loads(provenance_path.read_text())
        provenance["beta_zero_reused_from"] = str(original.relative_to(ROOT))
        provenance["detuning_grid_refinement"] = rows
        provenance_path.write_text(json.dumps(provenance, indent=2))
        report = OUT / "report.html"
        report.write_text(
            report.read_text()
            + f"<h2>Detuning-grid refinement</h2><p>For β=−0.5, halving the detuning step from 5 to 2.5 kHz at three selected amplitudes changed the center by at most {center_error:.3f} Hz and the half-depth linewidth by {width_error:.1f} Hz. "
            "<a href='detuning_refinement.json'>Refinement data</a></p>"
        )


if __name__ == "__main__":
    main()
