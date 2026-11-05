"""Add population maps of the selected 50 MHz echo shapes to the saved report.

Uses the validated arrays without interpolation, normalization, or new dynamics.
Run this after scripts/report_50mhz_echo.py to reproduce the report addition.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/single_echo_50mhz_20261101"
STEM = "spectroscopy_maps_2d"
CASES = ("baseline", "retuned", "phase")
LABELS = ("Original polynomial, p = 5", "Retuned polynomial", "Symmetric phase shaping")
BEGIN = "<!-- BEGIN 50MHZ 2D MAPS -->"
END = "<!-- END 50MHZ 2D MAPS -->"


def main():
    source = OUT / "validated_spectra.npz"
    results_path = OUT / "results.json"
    results = json.loads(results_path.read_text())
    with np.load(source) as data:
        detuning = data["detuning_mhz"].copy()
        amplitudes = data["rabi_mhz"].copy()
        populations = {name: data[name][1].copy() for name in CASES}
    np.testing.assert_array_equal(amplitudes, results["metadata"]["validation_amplitudes_mhz"])
    assert np.all(np.diff(detuning) > 0) and np.all(np.diff(amplitudes) > 0)
    for name, population in populations.items():
        assert population.shape == (len(amplitudes), len(detuning)), name
        assert np.all(np.isfinite(population)), name
        assert population.min() >= 0 and population.max() <= 1, name

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8), layout="constrained")
    for column, (name, label) in enumerate(zip(CASES, LABELS, strict=True)):
        at50 = results["cases"][name]["at50"]
        for row in range(2):
            ax = axes[row, column]
            mesh = ax.pcolormesh(detuning * 1000, amplitudes, populations[name],
                                 shading="nearest", cmap="viridis", vmin=0, vmax=1,
                                 rasterized=True)
            ax.axhline(50, color="white", lw=1.2, ls=(0, (5, 3)), alpha=.85)
            ax.set_xlabel("Detuning (kHz)")
            ax.set_ylabel("Actual peak Rabi rate (MHz)" if column == 0 else "")
            if row == 0:
                ax.set(xlim=(-500, 500), ylim=(10, 100), yticks=[10, 25, 40, 50, 60, 75, 100])
                for amplitude in (40, 60):
                    ax.axhline(amplitude, color="white", lw=.9, ls=":", alpha=.8)
                ax.set_title(f"{label}\n50 MHz: {at50['fwhm_hz']/1000:.1f} kHz, C = {at50['contrast']:.3f}",
                             fontsize=11, pad=10)
            else:
                ax.set(xlim=(-150, 150), ylim=(40, 60), yticks=[40, 45, 50, 55, 60])
                ax.set_title("Operating-band detail", fontsize=10, pad=8)
    colorbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=.92, pad=.015)
    colorbar.set_label("Excited-state probability  Pₑ")
    fig.suptitle("Single-echo spectroscopy maps · common population scale", fontsize=16)
    fig.supxlabel("Dashed: 50 MHz. Dotted: 40–60 MHz band. Simulated grid cells; no interpolation.",
                  fontsize=10)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{STEM}.{suffix}", dpi=180)
    plt.close(fig)

    provenance = dict(
        generator=str(Path(__file__).relative_to(ROOT)),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (source, results_path)},
        source_population_order="level, actual peak Rabi rate, detuning",
        displayed_population_level=1, cases=list(CASES),
        detuning_mhz=detuning.tolist(), actual_peak_rabi_mhz=amplitudes.tolist(),
        population_limits=[0, 1], interpolation="none; nearest cell shading",
        top_view=dict(detuning_khz=[-500, 500], actual_peak_rabi_mhz=[10, 100]),
        bottom_view=dict(detuning_khz=[-150, 150], actual_peak_rabi_mhz=[40, 60]),
        pulse_parameters={name: results["cases"][name]["pulse"] for name in CASES},
        outputs=[f"{STEM}.{suffix}" for suffix in ("png", "pdf", "svg")],
        simulations_rerun=False,
    )
    (OUT / f"{STEM}_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    block = f"""{BEGIN}
<h2>Two-dimensional spectroscopy maps</h2>
<p>The retuned polynomial and symmetric phase-shaped echo are shown alongside the original p=5 control. Each column keeps its pulse shape fixed while sweeping the actual peak Rabi rate. The top row spans 10–100 MHz and ±500 kHz detuning; the bottom row zooms into the 40–60 MHz operating band and ±150 kHz detuning. All six panels share the same 0–1 excited-state population color scale. Dashed lines mark 50 MHz; dotted lines on the full maps mark 40 and 60 MHz.</p>
<img src="{STEM}.png" alt="Two-dimensional population maps for the original polynomial, retuned polynomial, and symmetric phase-shaped echo, with operating-band zooms">
<p>These are the saved three-level simulation values, displayed without interpolation. Detuning spacing is 2.5 kHz; amplitude spacing is 2.5 MHz inside 40–60 MHz and 5 MHz elsewhere. This addition does not introduce new simulations or extend the validation range.</p>
<p><a href="{STEM}.pdf">PDF</a> · <a href="{STEM}.svg">SVG</a> · <a href="{STEM}_provenance.json">Map provenance</a> · <a href="validated_spectra.npz">Underlying arrays</a></p>
<p>Reproduce this addition after generating the base report: <code>.venv/bin/python scripts/plot_50mhz_echo_maps.py</code>.</p>
{END}
"""
    report_path = OUT / "report.html"
    report = report_path.read_text()
    if BEGIN in report:
        report, count = re.subn(re.escape(BEGIN) + r".*?" + re.escape(END),
                               lambda _: block.rstrip(), report, flags=re.DOTALL)
        assert count == 1
    else:
        anchor = "<h2>Why the polynomial does not remove every source of broadening</h2>"
        assert report.count(anchor) == 1
        report = report.replace(anchor, block + anchor)
    report_path.write_text(report)
    print(f"Added maps: {OUT / (STEM + '.png')}")
    print(f"Updated report: {report_path}")


if __name__ == "__main__":
    main()
