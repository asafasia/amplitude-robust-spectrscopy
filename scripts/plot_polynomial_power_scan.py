"""Render the power scan as scientific figures and a standalone HTML report."""

from __future__ import annotations

import csv
import html
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
from scripts import make_polynomial_power_scan as scan  # noqa: E402

OUT = scan.OUT


def label(power):
    return "Hard edges" if power is None else f"p = {power:g}"


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=170)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


def main():
    summaries = json.loads((OUT / "summary.json").read_text())
    rows = list(csv.DictReader((OUT / "metrics.csv").open()))
    powers = sorted({s["power"] for s in summaries if s["power"] is not None})
    colors = {None: "#7b8493", **dict(zip(
        powers, plt.colormaps["viridis"](np.linspace(.08, .9, len(powers))), strict=True))}
    groups = {s["case"]: [r for r in rows if r["case"] == s["case"]] for s in summaries}

    def values(case, field):
        return np.array([float(r[field]) for r in groups[case]])

    def ordered(c):
        return sorted([s for s in summaries if s["cutoff"] == c],
                      key=lambda s: s["power"] or 0)

    best = {c: max(ordered(c), key=lambda s: s["minimum_snr_per_mhz"])
            for c in scan.base.CUTOFFS}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained", sharex=True)
    for col, c in enumerate(scan.base.CUTOFFS):
        for s in ordered(c):
            p = s["power"]
            for row, field, scale in [(0, "fwhm_hz", .001), (1, "contrast_snr", 1)]:
                axes[row, col].plot(scan.base.AMPS, values(s["case"], field)*scale,
                                    color=colors[p], label=label(p), lw=1.8)
        axes[0, col].set(title=f"Base cutoff {c:g}", ylim=(0, 550))
        axes[0, col].axhline(1000/(np.pi*7.31), color="black", ls=":", lw=.9)
        axes[1, col].axhline(10, color="black", ls=":", lw=.9)
        axes[1, col].set(xlabel="Nominal amplitude (MHz)", ylim=(0, None))
        for ax in axes[:, col]:
            ax.axvspan(50, 100, color="#68a3c7", alpha=.08)
            ax.grid(alpha=.15)
            ax.set_xlim(0, 100)
    axes[0, 0].set_ylabel("Central dip FWHM (kHz)")
    axes[1, 0].set_ylabel("Ideal contrast SNR · 1000 total shots")
    axes[0, 0].legend(ncol=2, fontsize=8)
    fig.suptitle("Changing the polynomial exponent · 30 µs · β = −0.5 · 16 ns echo flip\n"
                 "100 amplitudes per curve; shading = selection range; "
                 "43.5 kHz dotted line = CW T₂ reference, not an echo floor")
    save(fig, "linewidth_snr")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for c, color in zip(scan.base.CUTOFFS, ["#1b7898", "#d77435", "#8654a2"], strict=True):
        group = [s for s in ordered(c) if s["power"] is not None]
        for ax, key in zip(axes, ["worst_fwhm_khz", "minimum_snr",
                                  "minimum_snr_per_mhz"], strict=True):
            ax.plot([s["power"] for s in group], [s[key] for s in group],
                    "o-", color=color, label=f"cutoff {c:g}")
            ax.set_xlabel("Polynomial exponent p")
            ax.grid(alpha=.15)
    for ax, title in zip(axes, ["Worst FWHM (kHz) ↓", "Minimum ideal SNR ↑",
                               "Minimum SNR / FWHM (MHz⁻¹) ↑"], strict=True):
        ax.set_title(title)
    axes[0].legend()
    fig.suptitle("Tradeoff across every sampled amplitude from 50.505 to 100 MHz")
    save(fig, "power_tradeoff")

    shoulder_figure = ""
    shoulder_file = OUT / "shoulder_checks.json"
    if shoulder_file.exists():
        shoulder_rows = json.loads(shoulder_file.read_text())
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
        subset = [r for r in shoulder_rows
                  if r["cutoff"] == .001 and r["amplitude_mhz"] == 100]
        for power in sorted({r["power"] for r in subset}):
            group = sorted([r for r in subset if r["power"] == power],
                           key=lambda r: r["shoulder_mhz"])
            for ax, field in zip(axes, ["fwhm_khz", "snr"], strict=True):
                ax.plot([r["shoulder_mhz"] for r in group],
                        [r[field] for r in group], "o-", color=colors[power],
                        label=label(power))
                ax.set_xlabel("Symmetric shoulder position ±Δ (MHz)")
                ax.axvline(.45, ls=":", lw=1, color="grey")
                ax.grid(alpha=.15)
        axes[0].set_ylabel("Half-depth width (kHz)")
        axes[1].set_ylabel("Ideal contrast SNR · 1000 total shots")
        axes[0].legend()
        fig.suptitle("Background choice matters · cutoff 0.001 · 100 MHz\n"
                     "Wider ±2 MHz check; dotted line = primary SNR shoulders")
        save(fig, "shoulder_sensitivity")
        shoulder_figure = (
            '<figure><img src="shoulder_sensitivity.png" alt="Shoulder sensitivity">'
            '<figcaption>These dips are not flat-background Lorentzians. '
            'At ±1 MHz shoulders, p = 8 is narrower than p = 3 '
            '(199.94 vs 212.10 kHz), while retaining higher SNR '
            '(12.61 vs 9.41). The primary table consistently uses the previous '
            '±0.5 MHz analysis window. '
            '<a href="shoulder_sensitivity.pdf">PDF</a></figcaption></figure>')

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
    t = np.linspace(-15, 15, 10001)
    sigma = 15/np.sqrt(.001**-2-1)
    f = (1+(t/sigma)**2)**-.5
    sign = -np.sin(np.pi*np.clip(t, -.008, .008)/.016)
    for p in [None, *powers]:
        w = np.ones_like(t) if p is None else np.maximum(0, 1-(t/15)**2)**p
        axes[0].plot(t, w, label=label(p), color=colors[p])
        axes[1].plot(t, abs(f*sign*w)*100, color=colors[p])
    axes[0].set(xlabel="Time from midpoint (µs)", ylabel="Window w(t)", ylim=(0, 1.03))
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].set(xlabel="Time from midpoint (µs)", ylabel="|I| (MHz)",
                yscale="log", ylim=(1e-5, 110), title="Cutoff 0.001 · nominal A = 100 MHz")
    for ax in axes:
        ax.grid(alpha=.15)
    fig.suptitle("Same 30 µs support and same core; larger p suppresses more tail area")
    save(fig, "windows")

    for c in scan.base.CUTOFFS:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
        for s in ordered(c):
            p = s["power"]
            for ax, field, factor in [(axes[0, 0], "center_hz", .001),
                                       (axes[0, 1], "shoulder_contrast", 1),
                                       (axes[0, 2], "max_final_pf", 100)]:
                ax.plot(scan.base.AMPS, values(s["case"], field)*factor,
                        color=colors[p], label=label(p))
            with np.load(OUT / f"{s['case']}.npz") as d:
                for state, ax in enumerate(axes[1]):
                    ax.plot(d["detuning_mhz"]*1000,
                            d["populations"][state, -1]*(100 if state == 2 else 1),
                            color=colors[p], label=label(p))
        for ax, title in zip(axes[0], ["Resonance center (kHz)", "Pe shoulder contrast",
                                       "Maximum final Pf (%)"], strict=True):
            ax.set(title=title, xlabel="Nominal amplitude (MHz)")
        axes[0, 0].legend(ncol=2, fontsize=8)
        for ax, title in zip(axes[1], ["Final Pg", "Final Pe", "Final Pf (%)"], strict=True):
            ax.set(title=f"{title} at 100 MHz", xlabel="Detuning (kHz)")
        for ax in axes.flat:
            ax.grid(alpha=.15)
        fig.suptitle(f"Three populations and resonance diagnostics · cutoff {c:g} · 30 µs")
        save(fig, f"details_c{c:g}")

        chosen_p = best[c]["power"]
        if chosen_p == 3 or chosen_p is None:
            chosen_p = max(powers)
        selected = list(dict.fromkeys([None, 3., 8., chosen_p]))
        fig, axes = plt.subplots(3, len(selected), figsize=(17, 10), layout="constrained",
                                 sharex=True, sharey=True)
        maps = [None]*3
        pf_max = max(np.load(OUT/f"{scan.name(c, p)}.npz")["populations"][2].max()*100
                     for p in selected)
        for col, p in enumerate(selected):
            with np.load(OUT/f"{scan.name(c, p)}.npz") as d:
                for state in range(3):
                    maps[state] = axes[state, col].pcolormesh(
                        d["detuning_mhz"]*1000, d["rabi_mhz"],
                        d["populations"][state]*(100 if state == 2 else 1),
                        shading="auto", vmin=0, vmax=pf_max if state == 2 else 1,
                        cmap="magma" if state == 2 else "viridis")
                    axes[state, col].set_title(f"{label(p)} · " + ["Pg", "Pe", "Pf (%)"][state])
            axes[2, col].set_xlabel("Detuning (kHz)")
        for state in range(3):
            axes[state, 0].set_ylabel("Nominal amplitude (MHz)")
            fig.colorbar(maps[state], ax=list(axes[state]), shrink=.8)
        fig.suptitle(f"Final populations · 100 amplitude points · cutoff {c:g}\n"
                     "Common color scales within each state; Pf is final, not transient leakage")
        save(fig, f"populations_c{c:g}")

    table = ""
    for s in summaries:
        m = s["at_100_mhz"]
        table += (f"<tr><td>{s['cutoff']:g}</td><td>{label(s['power'])}</td>"
                  f"<td>{m['fwhm_hz']/1000:.2f}</td><td>{m['contrast_snr']:.2f}</td>"
                  f"<td>{m['center_hz']/1000:+.3f}</td>"
                  f"<td>{s['worst_fwhm_khz']:.2f}</td><td>{s['minimum_snr']:.2f}</td>"
                  f"<td>{s['minimum_snr_per_mhz']:.2f}</td>"
                  f"<td>{s['points_passing_goal']}/50</td>"
                  f"<td>{'yes' if s['pareto'] else ''}</td></tr>")
    conclusions = []
    for c in scan.base.CUTOFFS:
        s = best[c]
        p3 = next(x for x in ordered(c) if x["power"] == 3)
        edge = " (at the tested upper boundary)" if s["power"] == max(powers) else ""
        conclusions.append(
            f"<li>Cutoff {c:g}: best sampled minimum SNR/width score is "
            f"<b>{label(s['power'])}</b>{edge}. Worst width "
            f"{s['worst_fwhm_khz']:.1f} kHz and minimum ideal SNR "
            f"{s['minimum_snr']:.2f}; the original p = 3 gives "
            f"{p3['worst_fwhm_khz']:.1f} kHz and {p3['minimum_snr']:.2f}.</li>")
    validation_file = OUT / "validation.json"
    validation = json.loads(validation_file.read_text()) if validation_file.exists() else []
    validation_text = "Convergence controls have not been run."
    if validation:
        dt = max(r["time_max_fwhm_hz_change"] for r in validation)
        grid = max(r["grid_max_fwhm_hz_change"] for r in validation)
        validation_text = (f"{len(validation)} selected cases checked at 50.505, 74.747, "
                           f"100 MHz: halving dt to 0.25 ns changes widths by at most "
                           f"{dt:.2f} Hz; halving detuning spacing to 2.5 kHz adds at "
                           f"most {grid:.2f} Hz. "
                           "See validation.json for equal unsigned-I-area controls.")
        precision_notes = [r for r in validation
                           if not r.get("primary_population_tolerance_passed", True)]
        for r in precision_notes:
            extra = r.get("extra_refinement")
            validation_text += (
                f" Case {r['case']} has a maximum population change of "
                f"{r['max_population_difference']:.2g} between 0.5 and 0.25 ns, "
                "exceeding the stricter 1e−4 population threshold.")
            if extra:
                validation_text += (
                    " Refining from 0.25 to 0.125 ns reduces that change to "
                    f"{extra['max_population_difference']:.2g} (a factor of "
                    f"{extra['population_error_reduction_factor']:.1f}); the "
                    f"additional width change is {extra['max_fwhm_hz_change']:.2f} Hz. "
                    "Primary maps retain 0.5 ns sampling.")
    area_text = ""
    area_example = next((v for v in validation
                         if v["cutoff"] == .001 and v["power"] == 8), None)
    if area_example:
        control = area_example["equal_unsigned_I_area_controls"][-1]
        area_text = (
            "<p>At cutoff 0.001, p = 8 at nominal 100 MHz has the same unsigned I "
            f"area as p = 3 at {control['p3_amplitude_mhz']:.2f} MHz. On the refined "
            f"grid these give {control['selected_fwhm_khz']:.2f} vs "
            f"{control['p3_fwhm_khz']:.2f} kHz, and ideal SNR "
            f"{control['selected_snr']:.2f} vs {control['p3_snr']:.2f}. "
            "The width–visibility tradeoff survives area matching. "
            "Smooth endpoints alone do not guarantee a narrower dip.</p>")
    def figure(name, caption):
        return (f'<figure><img src="{name}.png" alt="{html.escape(caption)}">'
                f'<figcaption>{caption} · <a href="{name}.pdf">PDF</a>'
                '</figcaption></figure>')

    body = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Polynomial window exponent · 30 µs echo</title>
<style>body{{font:16px/1.55 system-ui;color:#172636;background:#f6f8fb;
max-width:1400px;margin:auto;padding:28px}}h1,h2{{line-height:1.2}}
.card,figure{{background:white;padding:22px;border-radius:12px;margin:22px 0}}
img{{width:100%;height:auto}}figcaption{{color:#566477;font-size:14px}}
table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:8px;
border-bottom:1px solid #ddd;text-align:right}}th{{background:#eaf0f7}}
.scroll{{overflow:auto}}a{{color:#146aa2}}code{{background:#eaf0f7;padding:2px 5px}}</style>
<h1>Change the exponent, preserve smooth endpoints</h1>
<p>Three-level Lindblad simulation · 30 µs · β = −0.5 · 16 ns sine echo transition
· 100 amplitudes from 0 to 100 MHz.</p>
<div class="card"><p>The screenshot shows the existing cubic window. This scan tests
<b>w<sub>p</sub>(t) = [1 − (2t/L)²]<sup>p</sup></b> inside |t| &lt; L/2 and zero outside,
with p = {', '.join(f'{p:g}' for p in powers)}. Each tested window has zero value,
first derivative, and second derivative at the endpoints.</p>
<ul>{''.join(conclusions)}</ul><p>“Best” maximizes the minimum ideal SNR/FWHM across
the 50 sampled amplitudes from 50.505–100 MHz. It is a heuristic sharpness score,
not a frequency uncertainty or a global optimum over possible window functions.
The table separately reports width and SNR, so a tradeoff is visible.</p>
<p>At cutoff 0.001, p = 8 is a useful tested compromise: all 50 high-amplitude
points satisfy the illustrative targets width ≤ 200 kHz and ideal SNR ≥ 10.
The cubic window keeps a narrower dip under the previous ±0.5 MHz linewidth
definition. Larger powers improve visibility while broadening that measured width.
The targets are examples, not experimental requirements.</p></div>
{figure('linewidth_snr', 'Linewidth and ideal SNR versus amplitude, with all tested exponents')}
{figure('power_tradeoff', 'High-amplitude worst cases, rather than a favorable single amplitude')}
{figure('windows', 'Window shapes and unsigned in-phase envelope; no area or peak renormalization')}
{shoulder_figure}
<div class="card"><h2>Numbers</h2><div class="scroll"><table><thead><tr>
<th>Cutoff</th><th>Window</th><th>FWHM at 100<br>kHz</th><th>SNR at 100</th>
<th>Center at 100<br>kHz</th><th>Worst FWHM<br>50–100 MHz, kHz</th>
<th>Min SNR<br>50–100 MHz</th><th>Min SNR/width<br>MHz⁻¹</th>
<th>Width ≤ 200 kHz<br>SNR ≥ 10</th><th>Pareto</th>
</tr></thead><tbody>{table}</tbody></table></div></div>
"""
    for c in scan.base.CUTOFFS:
        body += f"<h2>Cutoff {c:g}</h2>"
        body += figure(f"details_c{c:g}", "Resonance shift, contrast, final Pf, and all three state populations at 100 MHz")
        body += figure(f"populations_c{c:g}", "Hard edges, original cubic, p = 8, and best score: full population maps")
    body += f"""<div class="card"><h2>Model and interpretation</h2>
<p>The base envelope is [1+(t/σ)²]<sup>−1/2</sup>, with
σ = 15 µs / √(cutoff<sup>−2</sup>−1). T₁ = 51.24 µs, T₂ = 7.31 µs,
α = −216 MHz, no added Stark correction. I = A f(t)s(t)w(t) and
Q = −β(dI/dt)/(2π|α|). DRAG differentiates the complete windowed envelope.
The 30 µs duration, nominal amplitude and base σ remain fixed.</p>
<p>FWHM is measured at half depth relative to a linear background from |detuning|
≥ 0.4 MHz. Ideal SNR assumes perfect binary Pe readout with 500 shots at the
center and 250 at each of ±0.45 MHz. Readout errors, drift, technical noise,
and the cost of locating the center are omitted. Zero amplitude has no dip.
The 43.5 kHz CW value 1/(πT₂) is a reference, not a lower bound on an echo notch.</p>
<p>Changing p changes the effective duration and unsigned pulse area. The base
cutoff still sets σ: this modification does not eliminate cutoff dependence.
Equal-area controls rescale the p = 3 amplitude and are a separate diagnostic.</p>
{area_text}
<h2>Validation</h2><p>{validation_text}</p>
<p>All main arrays have shape 3 × 100 × 201. The original hard and p = 3 cases
were rerun and compared numerically with the previous study.
The main solver uses 0.5 ns RK4 steps.</p>
<p><a href="metrics.csv">All metrics CSV</a> · <a href="summary.json">Summary</a> ·
<a href="provenance.json">Parameters and reproducible commands</a> ·
<a href="array_checks.json">Array and baseline checks</a> ·
<a href="validation.json">Convergence and area controls</a> ·
<a href="shoulder_checks.json">Shoulder sensitivity</a> ·
<a href="waveform_metrics.json">Pulse areas</a> ·
<a href="software_checks.json">Software checks</a></p>
<p>54 focused regression tests passed, including product-rule DRAG and native/NumPy
agreement. Scoped Ruff checks passed. The full suite is blocked at collection by
missing PyTorch in the test environment; repository-wide lint reports 692 existing
findings outside the changed files.</p>
<details><summary>Raw populations and axes</summary><ul>
"""
    body += "".join(f'<li><a href="{s["case"]}.npz">{s["case"]}.npz</a></li>'
                    for s in summaries)
    body += "</ul></details></div></html>"
    (OUT / "report.html").write_text(body)
    print(OUT / "report.html")


if __name__ == "__main__":
    main()
