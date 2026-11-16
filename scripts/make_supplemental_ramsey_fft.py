"""Generate the analytic finite-delay Ramsey reference and actual cosine FFTs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from echospec.analysis.ramsey_linewidth import (
    KINDS,
    cosine_transform,
    linewidth_mhz,
    normalized_spectrum,
)
from echospec.figures.coherence import REFERENCE_T2_STAR_S
from echospec.figures.supplemental import FIGURE_WIDTH, apply_supplemental_style
from echospec.paper_data import save_paper_dataset

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "figures/paper/supplemental"
T2_US = REFERENCE_T2_STAR_S * 1e6
REFERENCE_MHZ = 1 / (np.pi * T2_US)
DT_US = 0.01
FRINGE_MHZ = 1.0
NFFT = 2**18
COLORS = ("#126aa2", "#e18828", "#8b4a9c")
LABELS = (r"Absorption: $\mathrm{Re}\,G$", r"Power: $|G|^2$",
          r"Magnitude: $|G|$")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_arrays():
    time = np.arange(round(40 / DT_US)) * DT_US
    trace = np.exp(-time / T2_US) * np.cos(2 * np.pi * FRINGE_MHZ * time)
    frequency = np.fft.rfftfreq(NFFT, DT_US)
    selected = abs(frequency - FRINGE_MHZ) <= 0.15
    frequency = frequency[selected]
    transforms = []
    fft_errors = []
    for duration in (20.0, 40.0):
        n = round(duration / DT_US)
        spectrum = DT_US * np.fft.rfft(trace[:n], n=NFFT)[selected]
        # Include the negative-frequency image in this independent integral check.
        exact = cosine_transform(frequency, duration, T2_US, FRINGE_MHZ)
        error = float(np.max(abs(spectrum - exact)) / np.max(abs(exact)))
        if error > 0.002:
            raise RuntimeError(f"FFT discretization error too large: {error}")
        fft_errors.append(error)
        transforms.append(spectrum)
    duration_grid = np.linspace(5, 60, 441)
    table_duration = np.array([5, 10, 20, 30, 40, np.inf])
    offset = np.linspace(-0.15, 0.15, 1601)
    arrays = {
        "t2_us": np.asarray(T2_US),
        "reference_fwhm_mhz": np.asarray(REFERENCE_MHZ),
        "time_us": time,
        "ramsey_trace": trace,
        "fft_duration_us": np.array([20.0, 40.0]),
        "fft_frequency_mhz": frequency,
        "cosine_fft": np.array(transforms),
        "line_offset_mhz": offset,
        "normalized_20us_spectra": np.array([
            normalized_spectrum(offset, 20, T2_US, kind) for kind in KINDS
        ]),
        "duration_us": duration_grid,
        "fwhm_mhz": np.array([
            [linewidth_mhz(t, T2_US, kind) for t in duration_grid] for kind in KINDS
        ]),
        "table_duration_us": table_duration,
        "table_fwhm_mhz": np.array([
            [linewidth_mhz(t, T2_US, kind) for t in table_duration] for kind in KINDS
        ]),
    }
    return arrays, fft_errors


def render(arrays, echo_stat):
    apply_supplemental_style()
    fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH, 5.5))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.94,
                        hspace=0.49, wspace=0.30)
    ax = axes[0, 0]
    t = arrays["time_us"]
    ax.plot(t, arrays["ramsey_trace"], color=COLORS[0], linewidth=0.7)
    for sign in (-1, 1):
        ax.plot(t, sign * np.exp(-t / T2_US), color="0.45", linestyle="--",
                linewidth=0.8)
    ax.axvline(20, color=COLORS[1], linestyle=":")
    ax.text(20.7, 0.64, r"20 $\mu$s cutoff", fontsize=8, color="#b56810")
    ax.set(title=r"(a) Ramsey signal, $T_2^*=7.6\,\mu$s", xlim=(0, 40),
           ylim=(-1.05, 1.05), xlabel=r"Free-evolution delay $t$ ($\mu$s)",
           ylabel=r"$e^{-t/T_2^*}\cos(2\pi f_Rt)$")
    ax = axes[0, 1]
    offset = (arrays["fft_frequency_mhz"] - FRINGE_MHZ) * 1000
    for duration, spectrum, color in zip(
        arrays["fft_duration_us"], arrays["cosine_fft"], COLORS[1::-1], strict=True
    ):
        ax.plot(offset, spectrum.real / spectrum.real.max(), color=color,
                label=rf"Cosine FFT, {duration:g} $\mu$s")
    ax.plot(offset, normalized_spectrum(offset / 1000, np.inf, T2_US),
            color="0.25", linestyle="--", label=r"Infinite record")
    ax.set(title="(b) Phase-corrected real FFT", xlim=(-150, 150), ylim=(-0.03, 1.05),
           xlabel=r"Frequency offset $f-f_R$ (kHz)", ylabel="Normalized spectrum")
    ax.legend(fontsize=7, loc="upper right")
    ax = axes[1, 0]
    for spectrum, color, label in zip(
        arrays["normalized_20us_spectra"], COLORS, LABELS, strict=True
    ):
        ax.plot(arrays["line_offset_mhz"] * 1000, spectrum, color=color, label=label)
    ax.axhline(0.5, color="0.7", linestyle=":", linewidth=0.8)
    ax.set(title=r"(c) Spectral convention, $T=20\,\mu$s", xlim=(-150, 150),
           ylim=(-0.03, 1.05), xlabel=r"Frequency offset $f-f_R$ (kHz)",
           ylabel="Normalized isolated line")
    ax.legend(fontsize=7, loc="upper right")
    ax = axes[1, 1]
    for width, color in zip(arrays["fwhm_mhz"], COLORS, strict=True):
        ax.plot(arrays["duration_us"], width / REFERENCE_MHZ, color=color)
    ax.axhline(1, color="0.5", linestyle="--", linewidth=0.8)
    ax.axhline(np.sqrt(3), color="0.5", linestyle=":", linewidth=0.8)
    ax.text(47, 1.10, r"$\Gamma_{T_2^*}$", color="0.35", fontsize=8)
    ax.text(47, 1.85, r"$\sqrt{3}\,\Gamma_{T_2^*}$", color="0.35", fontsize=8)
    ax.plot(20, echo_stat["median_fwhm_normalized"], "D", color="#b83735",
            label=r"Measured echo median (20 $\mu$s)", markersize=4)
    ax.set(title="(d) Linewidth versus delay span", xlim=(5, 60), ylim=(0.75, 6.2),
           xlabel=r"Maximum Ramsey delay $T$ ($\mu$s)",
           ylabel=r"FWHM / $\Gamma_{T_2^*}$")
    ax.legend(fontsize=6.8, loc="upper right")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(OUTPUT / f"ramsey_fft_linewidth.{extension}", dpi=300)
    plt.close(fig)


def main():
    arrays, fft_errors = make_arrays()
    model_sources = [Path(__file__), ROOT / "echospec/analysis/ramsey_linewidth.py",
                     ROOT / "echospec/figures/coherence.py"]
    metadata = {
        "scope": "Supplemental S11 / Fig. S10: ideal exponential Ramsey reference",
        "t2_us": T2_US, "fringe_mhz": FRINGE_MHZ, "phase_rad": 0.0,
        "baseline": 0.0, "amplitude": 1.0, "noise": "none",
        "dt_us": DT_US, "nfft": NFFT, "spectrum_kind_order": list(KINDS),
        "sampling": "t = j*dt, j=0,...,N-1; T=N*dt; rectangular window",
        "fft_relative_integral_error": fft_errors,
        "linewidth": "First half-height crossings of exact isolated finite-duration line",
        "normalization": "Each displayed spectrum normalized to its own central peak",
        "padding_note": "Zero padding interpolates the transform, not new information",
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in model_sources},
        "reproduce": "PYTHONPATH=. MPLBACKEND=Agg .venv/bin/python scripts/make_supplemental_ramsey_fft.py",
    }
    paths = save_paper_dataset("10_ramsey_fft_linewidth", category="numerical",
                               arrays=arrays, provenance=metadata)
    echo_source = ROOT / "figures/paper/03_latest_q1_pulse_improvement_provenance.json"
    echo_stat = next(row for row in json.loads(echo_source.read_text())["statistics"]
                     if row["name"] == "E3_drag_poly")
    render(arrays, echo_stat)
    figure_metadata = {
        "numerical_dataset_sha256": {p.name: sha256(p) for p in paths},
        "experimental_reference_source": str(echo_source.relative_to(ROOT)),
        "experimental_reference_sha256": sha256(echo_source),
        "experimental_reference_statistic": echo_stat,
        "comparison_scope": "Common T2 reference and 20 us duration; not equal total acquisition time",
    }
    (OUTPUT / "ramsey_fft_linewidth_provenance.json").write_text(
        json.dumps(figure_metadata, indent=2, allow_nan=False) + "\n"
    )
    rows = [r"\begin{tabular}{rccc}", r"\hline\hline",
            r"$T$ ($\mu\mathrm{s}$) & $\operatorname{Re}G_T$ & $|G_T|^2$ & $|G_T|$ \\",
            r"\hline"]
    for duration, widths in zip(arrays["table_duration_us"],
                               arrays["table_fwhm_mhz"].T, strict=True):
        label = r"$\infty$" if np.isinf(duration) else f"{duration:g}"
        rows.append(label + " & " + " & ".join(f"{x * 1000:.2f}" for x in widths)
                    + r" \\")
    rows += [r"\hline\hline", r"\end{tabular}"]
    (ROOT / "paper/ramsey_linewidth_table.tex").write_text(
        "% Generated by scripts/make_supplemental_ramsey_fft.py\n"
        + "\n".join(rows) + "\n"
    )
    print("Reference FWHM (kHz):", REFERENCE_MHZ * 1000)
    print("FFT versus exact cosine integral errors:", fft_errors)
    print("Table: duration (us); absorption, power, magnitude FWHM (kHz)")
    for t, widths in zip(arrays["table_duration_us"], arrays["table_fwhm_mhz"].T,
                         strict=True):
        print(t, widths * 1000)


if __name__ == "__main__":
    main()
