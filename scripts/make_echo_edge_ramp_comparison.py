"""Reproduce fixed-core endpoint-window comparisons (100 amplitudes to 100 MHz).

Run: .venv/bin/python scripts/make_echo_edge_ramp_comparison.py
All outputs are simulations. SNR assumes ideal binary Pe readout and 1000 shots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ars-matplotlib-cache")

from echospec.analysis.spectroscopy_quality import spectroscopy_quality  # noqa: E402
from echospec.simulation.edge_window import edge_window  # noqa: E402
from echospec.simulation.qutrit import simulate_qutrit_map, sine_echo_sign  # noqa: E402

OUT = ROOT / "outputs/echo_edge_ramp_comparison"
COMMON = dict(duration_us=20., t1_us=51.24, t_phi_us=7.871480508563624,
              anharmonicity_mhz=-216., order=.5, echo=True,
              echo_transition_shape="sine", echo_transition_us=.016,
              stark_kappa_mhz_inv=0.)
WINDOWS = [("hard", "none", 0.), ("ramp1", "quintic", 1.),
           ("ramp3", "quintic", 3.), ("ramp6", "quintic", 6.),
           ("polynomial", "polynomial", 0.)]
CUTOFFS = [.001, .003, .005]
AMPS = np.linspace(0., 100., 100)
DETUNING = np.linspace(-.5, .5, 201)


def add_experiment_arguments(parser):
    parser.add_argument("--duration-us", type=float, default=20.)
    parser.add_argument("--output-dir", type=Path)


def configure_experiment(duration_us=20., output_dir=None):
    """Configure this process; call also as the spawned worker initializer."""
    global OUT
    if not np.isfinite(duration_us) or duration_us < 12:
        raise ValueError("Duration must be at least 12 us for the 6 us edge ramps")
    COMMON["duration_us"] = float(duration_us)
    default = ("echo_edge_ramp_comparison" if duration_us == 20 else
               f"echo_edge_ramp_{duration_us:g}us_comparison")
    OUT = (ROOT / "outputs" / default if output_dir is None
           else Path(output_dir).resolve())
    return OUT


def integration_steps(dt_ns=.5):
    """Preserve the physical timestep when duration changes."""
    return int(np.ceil(COMMON["duration_us"] * 1000 / (2 * dt_ns)))


def case_name(cutoff, window, beta=-.5):
    return f"c{cutoff:g}_{window}_beta{beta:g}"


def waveform(time_us, cutoff, window, ramp, beta=-.5):
    duration = COMMON["duration_us"]
    sigma = duration / 2 / np.sqrt(cutoff**-2 - 1)
    f = (1 + (time_us/sigma)**2)**-.5
    df = -time_us / sigma**2 * (1 + (time_us/sigma)**2)**-1.5
    s, ds = sine_echo_sign(time_us, .016)
    w, dw = edge_window(time_us, duration, kind=window, ramp_us=ramp)
    i = f*s*w
    q = -beta * ((df*s + f*ds)*w + f*s*dw) / (2*np.pi*216)
    return i, q


def run_case(cutoff, window_id, kind, ramp, beta=-.5, *,
             amplitudes=None, detuning=None, steps=None, suffix="", backend="native"):
    steps = integration_steps() if steps is None else steps
    amps = AMPS if amplitudes is None else np.asarray(amplitudes)
    det = DETUNING if detuning is None else np.asarray(detuning)
    config = dict(**COMMON, cutoff=cutoff, drag_beta=beta,
                  edge_window_kind=kind, edge_ramp_us=ramp,
                  num_steps_per_half=steps)
    key = case_name(cutoff, window_id, beta) + suffix
    path = OUT / f"{key}.npz"
    fingerprint = json.dumps(config, sort_keys=True)
    if path.exists():
        with np.load(path) as old:
            if (str(old["config"]) == fingerprint
                    and np.array_equal(old["rabi_mhz"], amps)
                    and np.array_equal(old["detuning_mhz"], det)):
                return key
        raise ValueError(f"Cache parameter mismatch: {path}")
    start = time.perf_counter()
    reference = ROOT / "outputs/sine_echo_cutoff_001_100mhz_comparison" / f"beta{beta:g}_narrow.npz"
    if (COMMON["duration_us"] == 20 and cutoff == .001
            and kind == "none" and steps == 20000
            and np.array_equal(amps, AMPS) and np.array_equal(det, DETUNING)
            and reference.exists()):
        with np.load(reference) as previous:
            np.testing.assert_array_equal(previous["rabi_mhz"], amps)
            np.testing.assert_array_equal(previous["detuning_mhz"], det)
            population = previous["populations"].copy()
        print(f"{key}: reused matching prior baseline", flush=True)
        backend = "numpy_reused_reference"
    else:
        chunks = []
        for offset in range(0, len(amps), 25):
            result = simulate_qutrit_map(
                **config, rabi_mhz=amps[offset:offset+25], detuning_mhz=det,
                integration_backend=backend)
            chunks.append(np.array([result.ground, result.excited,
                                    result.second_excited]))
            print(f"{key}: {min(offset+25, len(amps))}/{len(amps)} amplitudes; "
                  f"{time.perf_counter()-start:.1f} s", flush=True)
        population = np.concatenate(chunks, axis=1)
    np.savez_compressed(path, populations=population, rabi_mhz=amps,
                        detuning_mhz=det, config=fingerprint,
                        integration_backend=backend)
    return key


def export_metrics():
    import csv

    rows = []
    for cutoff in CUTOFFS:
        for label, _, _ in WINDOWS:
            name = case_name(cutoff, label)
            path = OUT / f"{name}.npz"
            if not path.exists():
                continue
            with np.load(path) as data:
                for index, amp in enumerate(data["rabi_mhz"]):
                    metrics = spectroscopy_quality(data["detuning_mhz"],
                                                   data["populations"][1, index])
                    rows.append(dict(case=name, cutoff=cutoff, window=label,
                                     beta=-.5, amplitude_mhz=float(amp), **metrics,
                                     max_final_pf=float(data["populations"][2, index].max())))
    if rows:
        with (OUT / "metrics.csv").open("w") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--backend", choices=["numpy", "native"], default="native")
    parser.add_argument("--metrics-only", action="store_true")
    add_experiment_arguments(parser)
    args = parser.parse_args()
    configure_experiment(args.duration_us, args.output_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.metrics_only:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=configure_experiment,
                                 initargs=(args.duration_us, OUT)) as pool:
            jobs = [pool.submit(run_case, c, name, kind, ramp, backend=args.backend)
                    for c in CUTOFFS for name, kind, ramp in WINDOWS]
            for job in as_completed(jobs):
                print("Finished", job.result(), flush=True)
    rows = export_metrics()
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in [Path(__file__), ROOT/"echospec/simulation/qutrit.py",
                           ROOT/"echospec/simulation/edge_window.py",
                           ROOT/"echospec/simulation/qutrit_native.py",
                           ROOT/"echospec/simulation/qutrit_rk4.cpp",
                           ROOT/"echospec/analysis/spectroscopy_quality.py",
                           ROOT/"echospec/analysis/dip_metrics.py",
                           ROOT/"scripts/plot_echo_edge_ramp_comparison.py",
                           ROOT/"scripts/validate_echo_edge_ramps.py",
                           ROOT/"scripts/check_echo_edge_controls.py"]}
    import shlex

    arguments = f" --duration-us {args.duration_us:g} --output-dir {shlex.quote(str(OUT))}"
    commands = [
        ".venv/bin/python scripts/make_echo_edge_ramp_comparison.py --workers 2",
        ".venv/bin/python scripts/plot_echo_edge_ramp_comparison.py",
        ".venv/bin/python scripts/validate_echo_edge_ramps.py 0.001:ramp6 0.001:polynomial 0.003:polynomial 0.005:polynomial --compare-backends",
        ".venv/bin/python scripts/check_echo_edge_controls.py --cutoff 0.001 --window polynomial",
        ".venv/bin/python scripts/plot_echo_edge_ramp_comparison.py",
    ]
    (OUT / "provenance.json").write_text(json.dumps(dict(
        parameters=COMMON, primary_beta=-.5, cutoffs=CUTOFFS, windows=WINDOWS,
        amplitudes_mhz=AMPS.tolist(), detuning_mhz=DETUNING.tolist(),
        dt_ns=COMMON["duration_us"]*1000/(2*integration_steps()),
        num_steps_per_half=integration_steps(),
        base_sigma_ns={str(c): COMMON["duration_us"]*500/np.sqrt(c**-2-1)
                       for c in CUTOFFS},
        metrics_rows=len(rows), source_sha256=hashes,
        python_version=sys.version, numpy_version=np.__version__,
        reproduce_commands=[command + arguments for command in commands],
        illustrative_operating_targets=dict(maximum_fwhm_khz=200,
                                             minimum_ideal_snr=10,
                                             amplitude_range_mhz=[50, 100]),
        integrators="Identical RK4 equations: NumPy for original cases, optional fused C++ CPU kernel for remaining cases. Per-file integration_backend records new runs; absent means the original NumPy solver.",
        snr="Ideal binary Pe readout: 500 shots at center, 250 at each of +/-0.45 MHz. No technical noise or assignment errors.",
        normalization="Same underlying width, same duration and nominal amplitude within each cutoff. No normalization after windowing.",
        caveat="Three-level rotating-wave Lindblad simulation; final Pf is not peak transient leakage. FWHM uses a finite-window linear background, not a Gaussian fit.",
    ), indent=2))


if __name__ == "__main__":
    main()
