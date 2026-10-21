"""Check DRAG dependence and shoulder-choice sensitivity of a selected window."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ars-matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from echospec.analysis.spectroscopy_quality import spectroscopy_quality  # noqa: E402
from scripts.make_echo_edge_ramp_comparison import (  # noqa: E402
    AMPS,
    COMMON,
    WINDOWS,
    add_experiment_arguments,
    case_name,
    configure_experiment,
    run_case,
)


def shoulder_metrics(x, y, shoulder):
    central = np.flatnonzero(abs(x) <= .25)
    index = central[np.argmin(y[central])]
    pc = y[index]
    pl, pr = np.interp([-shoulder, shoulder], x, y)
    contrast = (pl+pr)/2 - pc
    half = pc + contrast/2
    li = np.flatnonzero(y[:index] >= half)[-1]
    ri = np.flatnonzero(y[index+1:] >= half)[0] + index + 1
    left = x[li] + (half-y[li])/(y[li+1]-y[li])*(x[li+1]-x[li])
    right = x[ri-1] + (half-y[ri-1])/(y[ri]-y[ri-1])*(x[ri]-x[ri-1])
    variance = (2*pc*(1-pc)+pl*(1-pl)+pr*(1-pr))/1000
    return dict(shoulder_mhz=shoulder, fwhm_khz=float((right-left)*1000),
                contrast=float(contrast), snr=float(contrast/np.sqrt(variance)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", type=float, default=.001)
    parser.add_argument("--window", default="polynomial")
    parser.add_argument("--workers", type=int, default=2)
    add_experiment_arguments(parser)
    args = parser.parse_args()
    OUT = configure_experiment(args.duration_us, args.output_dir)
    c, w = args.cutoff, args.window
    _, kind, ramp = next(row for row in WINDOWS if row[0] == w)
    indices = [50, 74, 99]
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=configure_experiment,
                             initargs=(args.duration_us, OUT)) as pool:
        jobs = []
        for beta in [0., -1.]:
            for label, k, r in [(w, kind, ramp), ("hard", "none", 0.)]:
                jobs.append(pool.submit(run_case, c, label, k, r, beta,
                                        amplitudes=AMPS[indices], suffix="_beta_control"))
        for label, k, r in [(w, kind, ramp), ("hard", "none", 0.)]:
            jobs.append(pool.submit(run_case, c, label, k, r,
                                    amplitudes=AMPS[[50, 99]],
                                    detuning=np.linspace(-2, 2, 801), suffix="_wide"))
        for job in as_completed(jobs):
            print(job.result(), flush=True)
    beta_rows = []
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained", sharey=True)
    for ax, beta in zip(axes, [0., -.5, -1.], strict=True):
        for label, color in [("hard", "#536173"), (w, "#b34c8a")]:
            suffix = "" if beta == -.5 else "_beta_control"
            with np.load(OUT/f"{case_name(c, label, beta)}{suffix}.npz") as data:
                idx = indices if beta == -.5 else range(3)
                for j in idx:
                    metrics = spectroscopy_quality(data["detuning_mhz"], data["populations"][1, j])
                    beta_rows.append(dict(beta=beta, window=label,
                                          amplitude_mhz=float(data["rabi_mhz"][j]), **metrics))
                ax.plot(data["detuning_mhz"]*1000, data["populations"][1, -1],
                        label=label, color=color)
        ax.set(title=f"β = {beta:g}", xlabel="Detuning (kHz)", ylim=(0, 1))
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Final Pe")
    axes[0].legend()
    fig.suptitle(f"DRAG control: 100 MHz, base cutoff {c:g}, {w}, "
                 f"{COMMON['duration_us']:g} µs pulse")
    fig.savefig(OUT/"beta_controls.png", dpi=170)
    fig.savefig(OUT/"beta_controls.pdf")
    plt.close(fig)
    (OUT/"beta_controls.json").write_text(json.dumps(beta_rows, indent=2))
    wide_rows = []
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout="constrained", sharex=True)
    for label, color in [("hard", "#536173"), (w, "#b34c8a")]:
        with np.load(OUT/f"{case_name(c, label)}_wide.npz") as data:
            x, pop = data["detuning_mhz"], data["populations"]
            for row, amp in enumerate(data["rabi_mhz"]):
                for shoulder in [.3, .45, .6, .8, 1.]:
                    wide_rows.append(dict(window=label, amplitude_mhz=float(amp),
                                          **shoulder_metrics(x, pop[1, row], shoulder)))
                for state in range(3):
                    axes[row, state].plot(x, pop[state, row]*(100 if state == 2 else 1),
                                          label=label, color=color)
                    axes[row, state].set(title=f"A = {amp:.1f} MHz · " + ["Pg", "Pe", "Pf (%)"][state],
                                          xlabel="Detuning (MHz)")
                    axes[row, state].axvspan(-.5, .5, alpha=.06, color="black")
                    axes[row, state].grid(alpha=.15)
    axes[0, 0].legend()
    fig.suptitle(f"Wider frequency check · β = −0.5 · base cutoff {c:g} · "
                 f"{COMMON['duration_us']:g} µs pulse\n"
                 "Shading marks the primary ±0.5 MHz measurement window")
    fig.savefig(OUT/"wide_frequency_check.png", dpi=170)
    fig.savefig(OUT/"wide_frequency_check.pdf")
    plt.close(fig)
    (OUT/"shoulder_sensitivity.json").write_text(json.dumps(wide_rows, indent=2))


if __name__ == "__main__":
    main()
