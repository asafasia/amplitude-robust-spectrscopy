"""Map the simple polynomial echo versus cutoff, amplitude, and duration.

Start: python scripts/scan_polynomial_cutoffs.py --cutoff-count 10 --workers 2
Full:  python scripts/scan_polynomial_cutoffs.py --cutoff-count 50 --workers 2
No hardware is used. See paper/notes/POLYNOMIAL_ECHO_PULSE.md for definitions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ars-matplotlib-cache")

from echospec.analysis.cutoff_metrics import cutoff_metrics  # noqa: E402
from echospec.simulation.polynomial_echo import (  # noqa: E402
    PolynomialEchoPulse,
    integration_segments,
    simulate_polynomial_echo,
)

SOURCE_FILES = [
    "scripts/scan_polynomial_cutoffs.py",
    "echospec/simulation/polynomial_echo.py",
    "echospec/simulation/qutrit_native.py",
    "echospec/simulation/qutrit_rk4.cpp",
    "echospec/analysis/cutoff_metrics.py",
    "echospec/analysis/spectroscopy_quality.py",
    "echospec/analysis/dip_metrics.py",
]


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--cutoff-count", type=int, default=10,
                        help="Logarithmic samples, including both endpoints; use 50 next")
    result.add_argument("--cutoff-min", type=float, default=1e-5)
    result.add_argument("--cutoff-max", type=float, default=1.0)
    result.add_argument("--durations-us", type=float, nargs="+", default=[30.0])
    result.add_argument("--power", type=float, default=5.0)
    result.add_argument("--drag-beta", type=float, default=-0.5)
    result.add_argument("--transition-ns", type=float, default=16.0)
    result.add_argument("--amplitude-min-mhz", type=float, default=0.0)
    result.add_argument("--amplitude-max-mhz", type=float, default=100.0)
    result.add_argument("--amplitude-points", type=int, default=100)
    result.add_argument("--detuning-span-mhz", type=float, default=0.5,
                        help="Half-span; metrics always use the central +/-0.5 MHz")
    result.add_argument("--detuning-points", type=int, default=201)
    result.add_argument("--t1-us", type=float, default=51.24)
    result.add_argument("--t2-us", type=float, default=7.31)
    result.add_argument("--anharmonicity-mhz", type=float, default=-216.0)
    result.add_argument("--dt-ns", type=float, default=0.25,
                        help="Maximum RK4 step; use 0.125 for convergence checks")
    result.add_argument("--shots", type=int, default=1000)
    result.add_argument("--workers", type=int, default=1)
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--metrics-only", action="store_true",
                        help="Re-export matching cached simulations without integrating")
    result.add_argument("--dry-run", action="store_true",
                        help="Print the complete grid and integration plan, without writes")
    return result


def settings_from_args(args):
    numeric = [v for v in vars(args).values() if isinstance(v, (int, float))]
    if not all(np.isfinite(v) for v in numeric):
        raise ValueError("Numeric settings must be finite")
    if args.cutoff_count < 2 or not 0 < args.cutoff_min < args.cutoff_max <= 1:
        raise ValueError("Require >=2 cutoffs and 0 < cutoff-min < cutoff-max <= 1")
    if args.amplitude_points < 2 or not 0 <= args.amplitude_min_mhz < args.amplitude_max_mhz:
        raise ValueError("Require >=2 amplitudes and 0 <= amplitude-min < amplitude-max")
    if args.detuning_span_mhz < 0.5 or args.detuning_points < 21:
        raise ValueError("Require a half-span >=0.5 MHz and >=21 detuning points")
    if args.detuning_points % 2 != 1:
        raise ValueError("Use an odd number of detuning points to include zero")
    if args.shots < 4 or args.shots % 4 or args.workers < 1:
        raise ValueError("shots must be a positive multiple of 4; workers must be >=1")
    if not 0 < args.t2_us <= 2 * args.t1_us:
        raise ValueError("Require 0 < T2 <= 2*T1")
    if len(set(args.durations_us)) != len(args.durations_us):
        raise ValueError("Durations must be unique")
    cutoffs = np.geomspace(args.cutoff_min, args.cutoff_max, args.cutoff_count)
    pulses = [PolynomialEchoPulse(
        duration_us=length, cutoff=float(c), power=args.power,
        transition_us=args.transition_ns / 1000, drag_beta=args.drag_beta,
        anharmonicity_mhz=args.anharmonicity_mhz,
    ) for length in args.durations_us for c in cutoffs]
    for pulse in pulses:
        integration_segments(pulse, args.dt_ns)
    return dict(
        schema_version=1,
        cutoffs=cutoffs.tolist(), durations_us=args.durations_us,
        rabi_mhz=np.linspace(args.amplitude_min_mhz, args.amplitude_max_mhz,
                             args.amplitude_points).tolist(),
        detuning_mhz=np.linspace(-args.detuning_span_mhz, args.detuning_span_mhz,
                                 args.detuning_points).tolist(),
        power=args.power, transition_us=args.transition_ns / 1000,
        drag_beta=args.drag_beta, anharmonicity_mhz=args.anharmonicity_mhz,
        t1_us=args.t1_us, t2_us=args.t2_us, dt_ns=args.dt_ns, shots=args.shots,
        model="three-level Duffing Lindblad, native RK4, zero added Stark correction",
        amplitude_convention="Scale A before echo smoothing; no peak/area renormalization",
        cutoff_one="Continuous normalized-subtraction limit g=(1-(2t/L)^2)^(p+1)",
        source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                       for p in SOURCE_FILES},
    )


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False) + "\n")


def run_case(path, config, pulse_values, metrics_only=False):
    encoded = json.dumps(dict(config=config, pulse=pulse_values), sort_keys=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["configuration"]) != encoded:
                raise ValueError(f"Cache differs from this run: {path}")
        return f"Cached {path.parent.name}/{path.name}"
    if metrics_only:
        raise FileNotFoundError(f"Missing simulation for --metrics-only: {path}")
    start = time.perf_counter()
    population, diagnostics = simulate_polynomial_echo(
        PolynomialEchoPulse(**pulse_values), config["rabi_mhz"], config["detuning_mhz"],
        t1_us=config["t1_us"], t2_us=config["t2_us"], dt_ns=config["dt_ns"],
    )
    # An interrupted write never leaves a file that looks like a complete case.
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary, populations=population, configuration=encoded,
        diagnostics=json.dumps(diagnostics, sort_keys=True),
        rabi_mhz=config["rabi_mhz"], detuning_mhz=config["detuning_mhz"],
    )
    temporary.replace(path)
    return f"Completed c={pulse_values['cutoff']:.6g}, L={pulse_values['duration_us']:g} us in {time.perf_counter() - start:.1f}s"


def export_duration(directory, config, duration):
    rows, cubes, diagnostics = [], [], []
    for index, c in enumerate(config["cutoffs"]):
        with np.load(directory / f"cutoff_{index:03d}.npz", allow_pickle=False) as data:
            populations = data["populations"]
            diagnostics.append(json.loads(str(data["diagnostics"])))
        cubes.append(populations)
        for j, amplitude in enumerate(config["rabi_mhz"]):
            row = dict(
                duration_us=duration, cutoff=c, amplitude_mhz=amplitude,
                **cutoff_metrics(config["detuning_mhz"], populations[1, j],
                                 shots=config["shots"]),
                max_final_pf=float(populations[2, j].max()),
            )
            rows.append(row)
    with (directory / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    maps = {}
    for key in rows[0]:
        if key not in {"duration_us", "cutoff", "amplitude_mhz", "status"}:
            maps[key] = np.array([r[key] for r in rows]).reshape(
                len(config["cutoffs"]), len(config["rabi_mhz"])
            )
    cube = np.array(cubes)
    np.savez_compressed(
        directory / "scan.npz", populations=cube,
        cutoffs=config["cutoffs"], rabi_mhz=config["rabi_mhz"],
        detuning_mhz=config["detuning_mhz"], **maps,
        configuration=json.dumps(config, sort_keys=True), duration_us=duration,
    )
    summary = []
    for cutoff in config["cutoffs"]:
        group = [r for r in rows if r["cutoff"] == cutoff]
        active = [r for r in group if r["amplitude_mhz"] >= 10]
        valid = [r for r in active if r["accepted"]]
        complete = bool(active) and len(valid) == len(active)
        summary.append(dict(
            cutoff=cutoff, sampled_points_10mhz_up=len(active),
            accepted_points_10mhz_up=len(valid), all_active_points_accepted=complete,
            worst_fwhm_khz=max(r["fwhm_hz"] / 1000 for r in valid) if complete else None,
            minimum_snr=min(r["contrast_snr"] for r in valid) if complete else None,
            minimum_snr_per_mhz=min(r["snr_per_mhz"] for r in valid) if complete else None,
            best_sampled_tradeoff=max(valid, key=lambda r: r["snr_per_mhz"], default=None),
            status_counts={s: sum(r["status"] == s for r in group)
                           for s in sorted({r["status"] for r in group})},
        ))
    write_json(directory / "summary.json", dict(
        duration_us=duration, cutoffs=summary, diagnostics=diagnostics,
        caveat="Local finite-window dip metrics; multiple minima can remain. SNR is ideal, not measured.",
    ))
    plot_results(directory, config, duration, cube, maps)
    return dict(duration_us=duration, total_traces=len(rows),
                accepted_traces=sum(r["accepted"] for r in rows))


def plot_results(directory, config, duration, populations, maps):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    cutoffs = np.array(config["cutoffs"])
    amplitudes = np.array(config["rabi_mhz"])
    detuning = np.array(config["detuning_mhz"])
    valid = maps["accepted"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})

    def save(fig, name):
        for extension in ("png", "pdf"):
            fig.savefig(directory / f"{name}.{extension}", dpi=170)
        plt.close(fig)

    # Paginate the 50-cutoff run instead of shrinking every map into one page.
    for page, start in enumerate(range(0, len(cutoffs), 10), 1):
        stop = min(start + 10, len(cutoffs))
        columns = min(5, stop - start)
        nrows = int(np.ceil((stop - start) / columns))
        fig, axes = plt.subplots(nrows, columns, figsize=(3.35 * columns, 3.1 * nrows),
                                 squeeze=False, layout="constrained", sharex=True, sharey=True)
        for k, ax in enumerate(axes.flat):
            index = start + k
            if index >= stop:
                ax.set_visible(False)
                continue
            im = ax.pcolormesh(detuning * 1000, amplitudes, populations[index, 1],
                               shading="nearest", cmap="viridis", vmin=0, vmax=1)
            ax.set_title(f"cutoff = {cutoffs[index]:.4g}")
            ax.set_xlabel("Detuning (kHz)")
            if k % columns == 0:
                ax.set_ylabel("Drive scale A (MHz)")
        fig.colorbar(im, ax=list(axes.flat), label="Final excited population $P_e$", shrink=0.86)
        fig.suptitle(f"Polynomial echo • L = {duration:g} µs • p = {config['power']:g} • β = {config['drag_beta']:g}")
        save(fig, f"spectroscopy_maps_{page:02d}")

    fields = [
        ("fwhm_hz", 1e-3, "Dip FWHM (kHz)", "magma"),
        ("contrast_snr", 1, "Ideal contrast SNR", "viridis"),
        ("snr_per_mhz", 1, "SNR / FWHM (MHz⁻¹)", "viridis"),
        ("shoulder_contrast", 1, "Shoulder − center contrast", "viridis"),
        ("center_hz", 1e-3, "Dip center (kHz)", "coolwarm"),
        ("max_final_pf", 1, "Maximum final $P_f$ over detuning", "magma"),
    ]
    log_centers = np.log(cutoffs)
    log_edges = np.r_[log_centers[0] - (log_centers[1] - log_centers[0]) / 2,
                      (log_centers[1:] + log_centers[:-1]) / 2,
                      log_centers[-1] + (log_centers[-1] - log_centers[-2]) / 2]
    amplitude_edges = np.r_[amplitudes[0] - (amplitudes[1] - amplitudes[0]) / 2,
                            (amplitudes[1:] + amplitudes[:-1]) / 2,
                            amplitudes[-1] + (amplitudes[-1] - amplitudes[-2]) / 2]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for ax, (key, scale, label, cmap) in zip(axes.flat, fields, strict=True):
        data = maps[key] * scale
        if key != "max_final_pf":
            data = np.where(valid, data, np.nan)
        color = plt.get_cmap(cmap).copy()
        color.set_bad("#e1e4e8")
        im = ax.pcolormesh(amplitude_edges, np.exp(log_edges), np.ma.masked_invalid(data),
                           shading="flat", cmap=color)
        ax.set_yscale("log")
        ax.set_ylim(cutoffs[0], cutoffs[-1])
        ax.set_xlim(amplitudes[0], amplitudes[-1])
        ax.set(xlabel="Drive scale A (MHz)", ylabel="Cutoff", title=label)
        fig.colorbar(im, ax=ax)
    fig.suptitle(f"L = {duration:g} µs • {config['shots']} ideal shots • gray = unresolved/rejected dip")
    save(fig, "metric_maps")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    norm = LogNorm(cutoffs[0], cutoffs[-1])
    cmap = plt.get_cmap("viridis")
    for j, cutoff in enumerate(cutoffs):
        color = cmap(norm(cutoff))
        for ax, key, factor in [(axes[0, 0], "fwhm_hz", 1e-3),
                                (axes[0, 1], "contrast_snr", 1),
                                (axes[1, 0], "snr_per_mhz", 1)]:
            ax.plot(amplitudes, np.where(valid[j], maps[key][j] * factor, np.nan),
                    color=color, lw=1.2)
        axes[1, 1].scatter(maps["fwhm_hz"][j, valid[j]] / 1000,
                           maps["contrast_snr"][j, valid[j]], color=color, s=13, alpha=0.7)
    axes[0, 0].axhline(1000 / (np.pi * config["t2_us"]), color="gray", ls="--", lw=1,
                       label="1/(π T₂) reference")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].axhline(10, color="gray", ls="--", lw=1)
    for ax, label in zip([axes[0, 0], axes[0, 1], axes[1, 0]],
                         ["Dip FWHM (kHz)", "Ideal contrast SNR", "SNR / FWHM (MHz⁻¹)"], strict=True):
        ax.set(xlabel="Drive scale A (MHz)", ylabel=label)
        ax.grid(alpha=0.2)
    axes[1, 1].set(xlabel="Dip FWHM (kHz)", ylabel="Ideal contrast SNR",
                   title="Sampled linewidth / SNR tradeoff")
    axes[1, 1].grid(alpha=0.2)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=list(axes.flat), label="Cutoff")
    fig.suptitle(f"Polynomial echo • L = {duration:g} µs • accepted local dips only")
    save(fig, "tradeoffs")


def main(argv=None):
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    try:
        config = settings_from_args(args)
    except ValueError as error:
        argument_parser.error(str(error))
    output = (args.output_dir or ROOT / "outputs/polynomial_cutoff_sweep" /
              f"n{args.cutoff_count}").resolve()
    jobs = []
    for length in config["durations_us"]:
        directory = output / f"L{length:.12g}us"
        for index, cutoff in enumerate(config["cutoffs"]):
            pulse = PolynomialEchoPulse(
                duration_us=length, cutoff=cutoff, power=config["power"],
                transition_us=config["transition_us"], drag_beta=config["drag_beta"],
                anharmonicity_mhz=config["anharmonicity_mhz"],
            )
            jobs.append((directory / f"cutoff_{index:03d}.npz", config,
                         asdict(pulse), args.metrics_only))
    if args.dry_run:
        print(json.dumps(dict(output_dir=str(output), **config,
                              cases=len(jobs), points_per_case=len(config["rabi_mhz"]) * len(config["detuning_mhz"])), indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "configuration.json"
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError(f"Settings or source changed; choose a fresh --output-dir (existing: {output})")
    write_json(manifest, config)
    for path, _, _, _ in jobs:
        path.parent.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} pulse maps; {len(config['rabi_mhz'])} amplitudes × {len(config['detuning_mhz'])} detunings each → {output}", flush=True)
    if args.workers == 1:
        for job in jobs:
            print(run_case(*job), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_case, *job) for job in jobs]
            for future in as_completed(futures):
                print(future.result(), flush=True)
    summaries = [export_duration(output / f"L{length:.12g}us", config, length)
                 for length in config["durations_us"]]
    write_json(output / "provenance.json", dict(
        created_utc=datetime.now(timezone.utc).isoformat(), configuration=config,
        invocation=[sys.executable, str(Path(__file__).resolve()),
                    *(sys.argv[1:] if argv is None else argv)],
        simulations_only=True, results=summaries,
        array_layout="populations[cutoff, level=(g,e,f), amplitude, detuning]; metrics[cutoff, amplitude]",
        fwhm="Nearest half-depth crossings after linear background subtraction using |detuning|>=0.4 MHz in +/-0.5 MHz; central search +/-0.25 MHz; >=6 grid intervals required.",
        snr="Ideal independent binary Pe readout: N/2 center, N/4 each at +/-0.45 MHz. Contrast / sqrt((2*pc*(1-pc)+pl*(1-pl)+pr*(1-pr))/N). No technical noise or center-search cost.",
        caveats=["Not hardware validation or a calibrated experimental SNR.",
                 "At fixed duration, cutoff changes core width; A is not the signed I peak.",
                 "Widths are local, finite-window metrics; inspect maps and prominent_minima for satellites.",
                 "Time/detuning refinement is needed before comparing close optima."],
    ))
    print(f"Saved maps, tradeoffs, metrics.csv, scan.npz and provenance in {output}", flush=True)


if __name__ == "__main__":
    main()
