"""Test longer polynomial-windowed echo pulses, with matched native dynamics.

Run --stage screen, inspect summary_screen.json, then --stage full --cases ...
and --stage refine --cases ... . No shared pulse implementation is modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echospec.simulation.qutrit import _finalize, sine_echo_sign  # noqa: E402
from echospec.simulation.qutrit_native import integrate_native  # noqa: E402
from scripts import make_polynomial_power_scan as previous  # noqa: E402

OUT = ROOT / "outputs/duration_polynomial_scan"
SCREEN_INDICES = [50, 74, 99]
DURATIONS = [30., 45., 60., 90., 120.]
POWERS = [3., 8., 16., 32., 64., 128.]
SIGMA_REFERENCE = 15 / np.sqrt(.001**-2 - 1)
T2_US = 7.31


def specification(duration, power, family="raw", scaling="cutoff"):
    cutoff = (.001 if scaling == "cutoff" else
              1 / np.sqrt(1 + (duration / (2 * SIGMA_REFERENCE))**2))
    return dict(duration_us=duration, power=power, family=family,
                scaling=scaling, cutoff=cutoff,
                sigma_us=duration / 2 / np.sqrt(cutoff**-2 - 1))


def name(spec):
    return (f"L{spec['duration_us']:g}_p{spec['power']:g}_"
            f"{spec['family']}_{spec['scaling']}")


def screen_cases():
    primary = [specification(length, power, family)
               for length in DURATIONS for power in POWERS
               for family in ["raw", "subtracted"]]
    control = [specification(length, power, scaling="core")
               for length in [60., 120.] for power in POWERS]
    # Resolve the SNR = 10 boundary near the shortest, most visible pulses.
    local = [specification(length, power, family="subtracted")
             for length, powers in [(30., [4., 5., 6.]), (35., [6., 8., 12.]),
                                    (40., [8., 12., 16.]), (45., [24.])]
             for power in powers]
    local.append(specification(30., 7.))
    return primary + control + local


def envelope(t, spec):
    """Unsigned envelope and exact derivative; physical time is in us."""
    half, sigma = spec["duration_us"] / 2, spec["sigma_us"]
    if abs(t) >= half:
        return 0., 0.
    f = (1 + (t / sigma)**2)**-.5
    df = -t / sigma**2 * (1 + (t / sigma)**2)**-1.5
    if spec["family"] == "subtracted":
        cutoff = spec["cutoff"]
        f, df = (f - cutoff) / (1 - cutoff), df / (1 - cutoff)
    y, dy = 1 - (t / half)**2, -2 * t / half**2
    power = spec["power"]
    return f * y**power, df * y**power + f * power * y**(power - 1) * dy


def drive(t, spec):
    f, df = envelope(t, spec)
    sign, ds = sine_echo_sign(t, .016)
    i = f * sign
    q = .5 * (df * sign + f * ds) / (2 * np.pi * 216)
    return i - 1j * q


def simulate(spec, amplitudes, detuning, dt_ns):
    half = spec["duration_us"] / 2
    steps = int(np.ceil(half * 1000 / dt_ns))
    det, amp = np.meshgrid(2 * np.pi * detuning, 2 * np.pi * amplitudes)
    state = np.zeros((6, *det.shape), dtype=complex)
    state[0] = 1
    phase = np.zeros_like(det)
    for start, stop in [(-half, 0.), (0., half)]:
        state, phase = integrate_native(
            state, coordinate_start=start, coordinate_stop=stop,
            num_steps=steps, detuning=det, rabi=amp, anharmonicity=-216 * 2 * np.pi,
            inv_t1=1 / 51.24, inv_t_phi=1 / 7.871480508563624,
            drive_scale=lambda t: drive(t, spec), time_jacobian=lambda _: 1.,
            stark_kappa_mhz_inv=0., stark_correction_mode="detuning",
            accumulated_phase=phase, phase_time_origin=-half)
    result = _finalize(state)
    return np.array([result.ground, result.excited, result.second_excited])


def computation_hashes():
    hashes = previous.hashes()
    for fn in [specification, envelope, drive, simulate]:
        hashes[fn.__name__] = hashlib.sha256(
            inspect.getsource(fn).encode()).hexdigest()
    return hashes


def run_case(spec, stage):
    amps = (previous.base.AMPS if stage == "full" else
            previous.base.AMPS[SCREEN_INDICES])
    det = (np.linspace(-.5, .5, 401) if stage == "refine" else
           previous.base.DETUNING)
    dt_ns = .25 if stage == "refine" else .5
    config = dict(previous.COMMON, **{
        "duration_us": spec["duration_us"], "drag_beta": -.5,
        "pulse": spec, "dt_ns": dt_ns, "stage": stage,
        "computation_sha256": computation_hashes()})
    encoded = json.dumps(config, sort_keys=True)
    key = name(spec) + "_" + stage
    path = OUT / f"{key}.npz"
    if path.exists():
        with np.load(path) as data:
            if (str(data["config"]) != encoded or
                    not np.array_equal(data["rabi_mhz"], amps) or
                    not np.array_equal(data["detuning_mhz"], det)):
                raise ValueError(f"Cache does not match current calculation: {path}")
        return key
    start = time.perf_counter()
    chunks = [simulate(spec, amps[offset:offset + 25], det, dt_ns)
              for offset in range(0, len(amps), 25)]
    populations = np.concatenate(chunks, axis=1)
    np.savez_compressed(path, populations=populations, rabi_mhz=amps,
                        detuning_mhz=det, config=encoded)
    print(f"{key}: {len(amps)} amps in {time.perf_counter() - start:.1f}s",
          flush=True)
    return key


def metrics(path):
    with np.load(path) as data:
        config = json.loads(str(data["config"]))
        assert config["computation_sha256"] == computation_hashes()
        spec = config["pulse"]
        populations = data["populations"]
        assert np.isfinite(populations).all()
        assert populations.min() >= 0 and populations.max() <= 1
        assert abs(populations.sum(axis=0) - 1).max() < 1e-10
        return [dict(case=name(spec), **spec, amplitude_mhz=float(amp),
                     **previous.base.spectroscopy_quality(
                         data["detuning_mhz"], populations[1, i]),
                     max_final_pf=float(populations[2, i].max()))
                for i, amp in enumerate(data["rabi_mhz"])]


def clean(value):
    if isinstance(value, dict):
        return {key: clean(val) for key, val in value.items()}
    if isinstance(value, list):
        return [clean(val) for val in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def summarize(rows):
    high = [r for r in rows if r["amplitude_mhz"] >= 50]
    valid = all(r["accepted"] and np.isfinite(r["contrast_snr"]) for r in high)
    spec = {k: rows[0][k] for k in specification(30., 3.)}
    return dict(
        case=rows[0]["case"], **spec, points=len(high), all_accepted=valid,
        worst_fwhm_khz=max(r["fwhm_hz"] / 1000 for r in high) if valid else None,
        minimum_snr=min(r["contrast_snr"] for r in high) if valid else None,
        minimum_contrast=min(r["shoulder_contrast"] for r in high) if valid else None,
        max_abs_center_khz=max(abs(r["center_hz"]) / 1000 for r in high)
        if valid else None,
        max_final_pf=max(r["max_final_pf"] for r in high),
        near_t2_and_snr10_points=sum(
            r["accepted"] and r["fwhm_hz"] <= 50000 and r["contrast_snr"] >= 10
            for r in high),
        at_100_mhz=rows[-1])


def export(stage):
    all_rows, summary = [], []
    for path in sorted(OUT.glob(f"*_{stage}.npz")):
        rows = metrics(path)
        all_rows.extend(rows)
        summary.append(summarize(rows))
    with (OUT / f"metrics_{stage}.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=all_rows[0])
        writer.writeheader()
        writer.writerows(all_rows)
    (OUT / f"summary_{stage}.json").write_text(
        json.dumps(clean(summary), indent=2, allow_nan=False))
    return summary


def check_envelopes():
    for spec in screen_cases():
        half = spec["duration_us"] / 2
        assert envelope(0., spec) == (1., 0.)
        assert envelope(half, spec) == (0., 0.)
        grid = np.linspace(-half, half, 2001)
        values = np.array([envelope(t, spec)[0] for t in grid])
        assert values.min() >= 0 and values.max() <= 1
        np.testing.assert_allclose(values, values[::-1], atol=1e-13)
        for t in [.003, .3, half / 2, .99 * half]:
            numeric = (envelope(t + 1e-6, spec)[0] -
                       envelope(t - 1e-6, spec)[0]) / 2e-6
            np.testing.assert_allclose(envelope(t, spec)[1], numeric,
                                       atol=1e-8, rtol=1e-6)


def checks(stage):
    results = []
    if stage == "screen":
        for power in [3., 8., 16.]:
            path = OUT / f"L30_p{power:g}_raw_cutoff_screen.npz"
            with np.load(path) as new, np.load(
                    previous.OUT / f"c0.001_p{power:g}.npz") as old:
                error = float(abs(new["populations"] -
                                  old["populations"][:, SCREEN_INDICES]).max())
                assert error < 1e-10
                results.append(dict(case=path.stem, prior_agreement=error))
    if stage == "full":
        for path in sorted(OUT.glob("*_full.npz")):
            stem = path.stem.removesuffix("_full")
            with np.load(path) as full, np.load(
                    OUT / f"{stem}_screen.npz") as screen:
                error = float(abs(full["populations"][:, SCREEN_INDICES] -
                                  screen["populations"]).max())
                assert error < 1e-10
                results.append(dict(case=stem, full_screen_agreement=error))
    if stage == "refine":
        for path in sorted(OUT.glob("*_refine.npz")):
            stem = path.stem.removesuffix("_refine")
            with np.load(path) as fine, np.load(
                    OUT / f"{stem}_screen.npz") as coarse:
                error = float(abs(fine["populations"][:, :, ::2] -
                                  coarse["populations"]).max())
                coarse_metrics = metrics(OUT / f"{stem}_screen.npz")
                fine_metrics = metrics(path)
                sampled_metrics = [previous.base.spectroscopy_quality(
                    fine["detuning_mhz"][::2], pe[::2])
                    for pe in fine["populations"][1]]
                width_dt = max(abs(a["fwhm_hz"] - b["fwhm_hz"])
                               for a, b in zip(coarse_metrics, sampled_metrics,
                                               strict=True))
                width_grid = max(abs(a["fwhm_hz"] - b["fwhm_hz"])
                                 for a, b in zip(sampled_metrics, fine_metrics,
                                                 strict=True))
                snr_change = max(abs(a["contrast_snr"] - b["contrast_snr"])
                                 for a, b in zip(coarse_metrics, fine_metrics,
                                                 strict=True))
                assert error < 1e-4 and width_dt < 1000 and width_grid < 1000
                results.append(dict(case=stem, max_population_dt_change=error,
                                    max_width_dt_change_hz=width_dt,
                                    max_width_grid_change_hz=width_grid,
                                    max_snr_change=snr_change))
    (OUT / f"checks_{stage}.json").write_text(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["screen", "full", "refine"],
                        default="screen")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    cases = screen_cases()
    if args.cases:
        chosen = set(args.cases)
        cases = [s for s in cases if name(s) in chosen]
        if {name(s) for s in cases} != chosen:
            parser.error("Unknown case; choose a case from summary_screen.json")
    elif args.stage != "screen":
        parser.error("Specify selected --cases for full and refine stages")
    OUT.mkdir(parents=True, exist_ok=True)
    check_envelopes()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run_case, spec, args.stage) for spec in cases]
        for job in as_completed(jobs):
            job.result()
    summary = export(args.stage)
    checks(args.stage)
    provenance = dict(
        parameters=dict(previous.COMMON, duration_us="scanned", drag_beta=-.5),
        t2_us=T2_US, coherence_reference_khz=1000 / (np.pi * T2_US),
        screen_cases=screen_cases(), screen_amplitude_indices=SCREEN_INDICES,
        full_amplitude_points=100, amplitude_range_mhz=[0, 100],
        detuning_range_mhz=[-.5, .5], detuning_step_khz=5,
        dt_ns=.5, refined_dt_ns=.25, refined_detuning_step_khz=2.5,
        normalization="Unsigned base envelope equals 1 at midpoint before sine "
                      "echo; no signed-pulse peak or area normalization.",
        subtraction="(f-c)/(1-c), then multiply by polynomial window",
        fixed_core_sigma_us=SIGMA_REFERENCE,
        target="FWHM <= 50 kHz and ideal SNR >= 10 at every high-amplitude point",
        snr="1000 ideal Bernoulli shots: 500 center, 250 each +/-0.45 MHz",
        width="Nearest half-depth crossings relative to linear background "
              "at |detuning| >= 0.4 MHz; finite-window-dependent metric",
        computation_sha256=computation_hashes(), python_version=sys.version,
        numpy_version=np.__version__)
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps([{k: s[k] for k in ["case", "worst_fwhm_khz", "minimum_snr",
                                        "near_t2_and_snr10_points"]}
                      for s in summary], indent=2), flush=True)


if __name__ == "__main__":
    main()
