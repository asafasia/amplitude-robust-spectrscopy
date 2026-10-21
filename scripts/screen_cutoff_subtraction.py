"""Small hypothesis screen; three amplitudes, not a full robustness scan.

Test subtracting the endpoint value before tapering and an analytic C2
subtraction. Uses the same native qutrit equations and DRAG convention as the
30 us polynomial study. Does not change the shared pulse implementation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ars-matplotlib-cache")

from echospec.simulation.qutrit import _finalize, sine_echo_sign  # noqa: E402
from echospec.simulation.qutrit_native import integrate_native  # noqa: E402
from scripts import make_polynomial_power_scan as previous  # noqa: E402

OUT = ROOT / "outputs/cutoff_subtraction_screen"
AMPS = previous.base.AMPS[[50, 74, 99]]
DETUNING = previous.base.DETUNING
KINDS = ["polynomial3", "polynomial8", "floor", "floor_p1", "floor_p3",
         "subtract_c2"]


def envelope(t, c, kind):
    """Return positive base envelope and exact physical-time derivative."""
    if abs(t) > 15:
        return 0., 0.
    sigma = 15 / np.sqrt(c**-2-1)
    f = (1+(t/sigma)**2)**-.5
    df = -t/sigma**2 * (1+(t/sigma)**2)**-1.5
    y, dy = max(0., 1-(t/15)**2), -2*t/15**2
    if kind == "subtract_c2":
        a = 1-c*c
        correction = c*(1+a*y/2+3*a*a*y*y/8)
        derivative = c*(a/2+3*a*a*y/4)*dy
        norm = 1-c*(1+a/2+3*a*a/8)
        return (f-correction)/norm, (df-derivative)/norm
    power = {"polynomial3": 3, "polynomial8": 8, "floor": 0,
             "floor_p1": 1, "floor_p3": 3}[kind]
    if kind.startswith("floor"):
        f, df = (f-c)/(1-c), df/(1-c)
    if power:
        return f*y**power, df*y**power + f*power*y**(power-1)*dy
    return f, df


def drive(t, c, kind):
    f, df = envelope(t, c, kind)
    sign, ds = sine_echo_sign(t, .016)
    i = f*sign
    q = .5*(df*sign+f*ds)/(2*np.pi*216)
    return i-1j*q


def simulate(c, kind, steps=30000):
    det, amp = np.meshgrid(2*np.pi*DETUNING, 2*np.pi*AMPS)
    state = np.zeros((6, *det.shape), dtype=complex)
    state[0] = 1
    phase = np.zeros_like(det)
    for start, stop in [(-15., 0.), (0., 15.)]:
        state, phase = integrate_native(
            state, coordinate_start=start, coordinate_stop=stop,
            num_steps=steps, detuning=det, rabi=amp, anharmonicity=-216*2*np.pi,
            inv_t1=1/51.24, inv_t_phi=1/7.871480508563624,
            drive_scale=lambda t: drive(t, c, kind), time_jacobian=lambda _: 1.,
            stark_kappa_mhz_inv=0., stark_correction_mode="detuning",
            accumulated_phase=phase, phase_time_origin=-15.)
    result = _finalize(state)
    return np.array([result.ground, result.excited, result.second_excited])


def run(c, kind):
    populations = simulate(c, kind)
    if kind.startswith("polynomial"):
        p = float(kind.removeprefix("polynomial"))
        with np.load(previous.OUT / f"{previous.name(c, p)}.npz") as old:
            error = float(abs(populations-old["populations"][:, [50, 74, 99]]).max())
            assert error < 1e-10, error
    np.savez_compressed(OUT/f"c{c:g}_{kind}.npz", populations=populations,
                        rabi_mhz=AMPS, detuning_mhz=DETUNING)
    rows = [dict(cutoff=c, shape=kind, amplitude_mhz=float(a),
                 **previous.base.spectroscopy_quality(DETUNING, populations[1, j]),
                 max_final_pf=float(populations[2, j].max()))
            for j, a in enumerate(AMPS)]
    print(c, kind, "at 100 MHz", rows[-1]["fwhm_hz"]/1000,
          rows[-1]["contrast_snr"], flush=True)
    return rows


def write_report(rows, controls):
    labels = dict(polynomial3="Original cubic", polynomial8="Original p = 8",
                  floor="Subtract cutoff only", floor_p1="Subtract + linear window",
                  floor_p3="Subtract + cubic window", subtract_c2="C2 endpoint subtraction")
    table = "".join(
        f"<tr><td>{labels[r['shape']]}</td><td>{r['fwhm_hz']/1000:.2f}</td>"
        f"<td>{r['contrast_snr']:.3f}</td><td>{r['shoulder_contrast']:.3f}</td></tr>"
        for r in rows if r["cutoff"] == .001 and r["amplitude_mhz"] == 100)
    worst_dt = max(r["max_fwhm_change_hz"] for r in controls)
    body = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cutoff subtraction: an exploratory screen</title>
<style>body{{font:17px/1.6 system-ui;max-width:1200px;margin:auto;padding:30px;
color:#172636;background:#fafbfd}}img{{width:100%}}table{{border-collapse:collapse}}
td,th{{padding:9px 18px;border-bottom:1px solid #ddd;text-align:left}}
code{{background:#eaf0f5;padding:3px}}a{{color:#176da1}}</style>
<h1>Subtracting the cutoff before shaping</h1>
<p>Exploratory screen at <b>50.505, 74.747, and 100 MHz</b> only. This is not a
full 100-amplitude robustness test. The same 30 µs, 16 ns sine echo inversion,
β = −0.5, three-level model, T₁ = 51.24 µs, T₂ = 7.31 µs and α = −216 MHz apply.</p>
<p>Best new candidate at cutoff 0.001: cancel the endpoint value and its first
two derivatives by subtracting a smooth background. This improves width and
visibility relative to the cubic at the three checked amplitudes. It does not
reach the 43.54 kHz coherence reference.</p>
<h2>At cutoff 0.001 and amplitude 100 MHz</h2>
<table><tr><th>Shape</th><th>FWHM (kHz)</th><th>Ideal SNR</th><th>Pe contrast</th></tr>
{table}</table>
<h2>Exact pulse construction</h2>
<p>Let f(t) = [1 + (t/σ)²]<sup>−1/2</sup>, σ = 15/√(c<sup>−2</sup>−1) µs,
x = 2t/L, y = 1 − x², and a = 1 − c². All shapes have support |t| ≤ L/2.</p>
<p>The direct floor subtraction is <code>g(t) = [f(t) − c]/(1 − c)</code>.
Its value reaches zero, but its slope does not. Multiplication by y or y³
gives the mild and cubic window controls.</p>
<p>The C2 construction is</p>
<p><code>g(t) = [f(t) − c(1 + ay/2 + 3a²y²/8)] /
[1 − c(1 + a/2 + 3a²/8)]</code>.</p>
<p>This subtracts the quadratic Taylor expansion in y of
f = c(1 − ay)<sup>−1/2</sup> at each endpoint. Consequently g, g′ and g″
vanish there. The denominator restores g(0) = 1. The waveform is I = A s(t)g(t),
with Q = −β I′/(2π|α|). There is no signed-pulse peak or area normalization.
The cutoff still determines σ; zero endpoints do not remove core-width dependence.</p>
<img src="screen.png" alt="Linewidth and SNR for six pulse shapes at three amplitudes">
<p>Curves join three sampled points only. FWHM uses the same linear background
at |detuning| ≥ 0.4 MHz in the ±0.5 MHz sweep. SNR uses ideal binary Pe readout,
500 center shots and 250 at each ±0.45 MHz shoulder. It is not 10% population contrast.</p>
<h2>Scope and validation</h2>
<p>18 maps, each 3 states × 3 amplitudes × 201 detunings. Analytic derivatives
were checked numerically. Original p = 3 and p = 8 populations match the previous
study within 10⁻¹⁰. Halving dt from 0.5 to 0.25 ns for the new floor, floor_p1,
and C2 cases at cutoff 0.001 changes widths by at most {worst_dt:.2f} Hz.
This screen does not establish performance between the three amplitudes,
or independence from background choice, and contains no hardware validation.</p>
<p><a href="results.json">Parameters, all metrics, checks and source hashes</a> ·
<a href="metrics.csv">Metrics CSV</a> · <a href="screen.pdf">Figure PDF</a></p>
<p>Reproduce with <code>.venv/bin/python scripts/screen_cutoff_subtraction.py</code>.</p>
</html>"""
    (OUT/"report.html").write_text(body)


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    # Deterministic shape/derivative checks independently of the integrator.
    for c in previous.base.CUTOFFS:
        for kind in KINDS:
            assert abs(envelope(0, c, kind)[0]-1) < 1e-12
            assert abs(envelope(15, c, kind)[0]) < 1e-12
            for t in [.003, .3, 4., 14.]:
                numeric = (envelope(t+1e-6, c, kind)[0]
                           - envelope(t-1e-6, c, kind)[0])/2e-6
                np.testing.assert_allclose(envelope(t, c, kind)[1], numeric,
                                           atol=1e-8, rtol=1e-6)
            if kind != "floor":
                assert abs(envelope(15, c, kind)[1]) < 1e-12
    rows = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(run, c, kind) for c in previous.base.CUTOFFS
                for kind in KINDS]
        for job in as_completed(jobs):
            rows.extend(job.result())
    rows.sort(key=lambda r: (r["cutoff"], KINDS.index(r["shape"]), r["amplitude_mhz"]))
    with (OUT/"metrics.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    # Refine the new C2 and minimally tapered hypotheses at the lowest cutoff.
    controls = []
    for kind in ["floor", "floor_p1", "subtract_c2"]:
        fine = simulate(.001, kind, steps=60000)
        with np.load(OUT/f"c0.001_{kind}.npz") as old:
            error = float(abs(fine-old["populations"]).max())
            old_metrics = [previous.base.spectroscopy_quality(DETUNING, p)
                           for p in old["populations"][1]]
        fine_metrics = [previous.base.spectroscopy_quality(DETUNING, p) for p in fine[1]]
        width_error = max(abs(a["fwhm_hz"]-b["fwhm_hz"])
                          for a, b in zip(old_metrics, fine_metrics, strict=True))
        assert error < 1e-4 and width_error < 1000
        controls.append(dict(shape=kind, cutoff=.001, max_population_change=error,
                             max_fwhm_change_hz=width_error))
    model_paths = [ROOT/p for p in previous.MODEL_FILES] + [Path(__file__)]
    model_paths += [ROOT/"echospec/analysis/spectroscopy_quality.py",
                    ROOT/"echospec/analysis/dip_metrics.py"]
    (OUT/"results.json").write_text(json.dumps(dict(
        scope="Exploratory three-amplitude screen, not a full robustness scan",
        common=previous.COMMON, drag_beta=-.5, amplitudes_mhz=AMPS.tolist(),
        detuning_mhz=DETUNING.tolist(), cutoffs=previous.base.CUTOFFS,
        dt_ns=.5, shapes=KINDS, rows=rows, convergence=controls,
        baseline_agreement="p3 and p8 agree with prior grids to <1e-10",
        snr="Same ideal 1000-shot allocation and +/-0.45 MHz shoulders as prior study",
        normalization="Floor and C2 subtraction preserve positive base envelope f(0)=1; "
                      "no signed-echo peak or area renormalization",
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in model_paths}), indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    colors = plt.colormaps["tab10"](np.arange(len(KINDS)))
    for col, c in enumerate(previous.base.CUTOFFS):
        for kind, color in zip(KINDS, colors, strict=True):
            group = [r for r in rows if r["cutoff"] == c and r["shape"] == kind]
            axes[0, col].plot(AMPS, [r["fwhm_hz"]/1000 for r in group], "o-",
                              label=kind, color=color)
            axes[1, col].plot(AMPS, [r["contrast_snr"] for r in group], "o-", color=color)
        axes[0, col].set(title=f"Base cutoff {c:g}", ylabel="FWHM (kHz)")
        axes[0, col].axhline(1000/(np.pi*7.31), color="grey", ls=":")
        axes[1, col].set(ylabel="Ideal SNR · 1000 total shots", xlabel="Amplitude (MHz)")
        axes[1, col].axhline(10, color="grey", ls=":")
        for ax in axes[:, col]:
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Subtracting the cutoff before shaping · exploratory 3-amplitude screen\n"
                 "30 µs, β = −0.5, 16 ns echo inversion; same previous width definition")
    fig.savefig(OUT/"screen.png", dpi=170)
    fig.savefig(OUT/"screen.pdf")
    plt.close(fig)
    write_report(rows, controls)


if __name__ == "__main__":
    main()
