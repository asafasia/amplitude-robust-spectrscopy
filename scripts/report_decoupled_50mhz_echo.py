"""Validate and report independently shaped echo candidates without refitting.

The selected-candidate manifest contains ``candidates`` with ``name``,
``pulse_type`` (decoupled/phase/polynomial), and serialized ``pulse`` fields.
The ``validate`` stage computes and caches numerical evidence. ``render`` only
uses the saved results and arrays; it never launches simulations. Previous
study files and manuscript artifacts are read-only inputs to this script.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echospec.analysis.dip_metrics import dip_metrics  # noqa: E402
from echospec.simulation.decoupled_echo import DecoupledEchoPulse  # noqa: E402
from echospec.simulation.polynomial_echo import (  # noqa: E402
    PolynomialEchoPulse,
    simulate_polynomial_echo,
)
from echospec.simulation.selective_echo import (  # noqa: E402
    PhaseEchoPulse,
    simulate_echo_split,
)
from scripts.optimize_50mhz_echo import metrics, peak_scale  # noqa: E402

OUT = ROOT / "outputs/decoupled_echo_50mhz_20261101"
PREVIOUS = ROOT / "outputs/single_echo_50mhz_20261101/results.json"
MODEL_FILES = [ROOT / "echospec/simulation" / name for name in (
    "decoupled_echo.py", "polynomial_echo.py", "selective_echo.py",
    "qutrit_native.py", "qutrit_rk4.cpp")]
T1, T2 = 51.24, 7.31


def clean(value):
    """Write standards-compliant JSON while retaining rejected metrics as null."""
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def save_json(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SampledPulse:
    def __init__(self, base, time, values, filter_sigma_ns):
        self.base, self.time, self.values = base, time, values
        self.identity = dict(type="filtered", base_type=type(base).__name__,
                             base=asdict(base), filter_sigma_ns=filter_sigma_ns,
                             time_sha256=hashlib.sha256(time.tobytes()).hexdigest(),
                             values_sha256=hashlib.sha256(values.tobytes()).hexdigest())

    def __getattr__(self, name):
        return getattr(self.base, name)

    def drive(self, time):
        return np.interp(time, self.time, self.values, left=0, right=0)


def cached_native(output, pulse, amplitudes, detuning, dt_ns=.25):
    identity = dict(pulse=pulse.identity if isinstance(pulse, SampledPulse) else
                    dict(type=type(pulse).__name__, fields=asdict(pulse)),
                    amplitudes=np.asarray(amplitudes).tolist(),
                    detuning=np.asarray(detuning).tolist(), dt_ns=dt_ns,
                    t1_us=T1, t2_us=T2, sources={p.name: sha(p) for p in MODEL_FILES})
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = output / "evaluations" / f"validate_native_{key}.npz"
    if path.exists():
        with np.load(path) as saved:
            return saved["populations"].copy(), json.loads(str(saved["diagnostics"]))
    populations, diagnostics = simulate_polynomial_echo(
        pulse, amplitudes, detuning, t1_us=T1, t2_us=T2, dt_ns=dt_ns)
    np.savez_compressed(path, populations=populations,
                        diagnostics=json.dumps(diagnostics), identity=json.dumps(identity))
    return populations, diagnostics


def cached_split(output, pulse, amplitude, detuning, levels, refinement):
    identity = dict(pulse=asdict(pulse), type=type(pulse).__name__, amplitude=amplitude,
                    detuning=np.asarray(detuning).tolist(), levels=levels,
                    refinement=refinement, t1_us=T1, t2_us=T2,
                    sources={p.name: sha(p) for p in MODEL_FILES})
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = output / "evaluations" / f"validate_split_{key}.npz"
    if path.exists():
        with np.load(path) as saved:
            return saved["populations"].copy()
    populations = simulate_echo_split(pulse, amplitude, detuning, levels=levels,
                                      refinement=refinement, t1_us=T1, t2_us=T2)
    np.savez_compressed(path, populations=populations, identity=json.dumps(identity))
    return populations


def local_shoulder_metrics(detuning, probability):
    """Conservative nearest-shoulder depth, separate from the primary background.

    A shoulder is the nearest local maximum with prominence >=0.001 on each
    side. If none exists, use that side's maximum within the supplied window.
    The lower shoulder sets depth; this is not a fitted or detrended metric.
    """
    x, y = np.asarray(detuning), np.asarray(probability)
    central = np.flatnonzero(abs(x) <= .25)
    center = central[np.argmin(y[central])]
    if center == 0 or center == len(x) - 1:
        return dict(accepted=False, contrast=np.nan, fwhm_hz=np.nan)
    peaks, _ = find_peaks(y, prominence=.001)
    left_peaks, right_peaks = peaks[peaks < center], peaks[peaks > center]
    left = int(left_peaks[-1]) if len(left_peaks) else int(np.argmax(y[:center]))
    right = int(right_peaks[0]) if len(right_peaks) else center + 1 + int(np.argmax(y[center+1:]))
    contrast = min(y[left], y[right]) - y[center]
    half = y[center] + contrast / 2
    lc = np.flatnonzero(y[:center] >= half)
    rc = np.flatnonzero(y[center+1:] >= half) + center + 1
    width = np.nan
    if contrast > .001 and lc.size and rc.size:
        li, ri = lc[-1], rc[0]
        xl = x[li] + (half-y[li]) / (y[li+1]-y[li]) * (x[li+1]-x[li])
        xr = x[ri-1] + (half-y[ri-1]) / (y[ri]-y[ri-1]) * (x[ri]-x[ri-1])
        width = (xr-xl) * 1e6
    return dict(accepted=bool(np.isfinite(width)), contrast=float(contrast),
                fwhm_hz=float(width), center_mhz=float(x[center]),
                left_shoulder_mhz=float(x[left]), right_shoulder_mhz=float(x[right]))


def satellite_metrics(detuning, probability):
    indices, properties = find_peaks(-probability, prominence=0)
    others = np.ones(len(indices), dtype=bool)
    if len(indices):
        others[np.argmin(abs(detuning[indices]))] = False
    prominence = properties["prominences"][others]
    locations = detuning[indices][others]
    active = prominence >= .01
    return dict(max_prominence=float(prominence.max(initial=0)),
                detuning_mhz=locations[active].tolist(), prominence=prominence[active].tolist())


def load_cases(manifest):
    saved = json.loads(PREVIOUS.read_text())
    cases = {
        "baseline": dict(label="Original polynomial p=5", control=True,
                         pulse=PolynomialEchoPulse(**saved["cases"]["baseline"]["pulse"])),
        "previous_phase": dict(label="Previous phase-shaped echo", control=True,
                               pulse=PhaseEchoPulse(**saved["cases"]["phase"]["pulse"])),
    }
    classes = {"decoupled": DecoupledEchoPulse, "phase": PhaseEchoPulse,
               "polynomial": PolynomialEchoPulse}
    for record in manifest["candidates"]:
        name = record["name"]
        if not re.fullmatch(r"[a-z0-9_]+", name) or name in cases:
            raise ValueError(f"Invalid or duplicate candidate name: {name}")
        cases[name] = dict(label=record.get("label", name.replace("_", " ").capitalize()),
                           control=False, pulse=classes[record["pulse_type"]](**record["pulse"]),
                           selection_record=record)
    return cases


def assess_constraints(rows, baseline_rows, constraints):
    minimum = float(constraints.get("amplitude_min_mhz", 5))
    maximum = float(constraints.get("amplitude_max_mhz", 60))
    floor = float(constraints.get("contrast_min", .4))
    width_tolerance = float(constraints.get("no_regression_width_khz_tolerance", .25))
    contrast_tolerance = float(constraints.get("no_regression_contrast_tolerance", .001))
    reference = {r["rabi_mhz"]: r for r in baseline_rows}
    active = [r for r in rows if minimum <= r["rabi_mhz"] <= maximum]
    accepted = all(r["accepted"] for r in active)
    contrast_fail = [r["rabi_mhz"] for r in active if not r["contrast"] > floor]
    width_fail, regression_contrast = [], []
    for row in active:
        old = reference[row["rabi_mhz"]]
        if old["accepted"] and (not row["accepted"] or
                                row["fwhm_hz"] > old["fwhm_hz"] + 1000 * width_tolerance):
            width_fail.append(row["rabi_mhz"])
        if np.isfinite(old["contrast"]) and row["contrast"] < old["contrast"] - contrast_tolerance:
            regression_contrast.append(row["rabi_mhz"])
    satellite_cap = float(constraints.get("max_satellite_prominence", .01))
    center_cap = 1000 * float(constraints.get("max_center_khz", 2))
    satellites = [r["rabi_mhz"] for r in active if r["satellite_prominence"] > satellite_cap]
    centers = [r["rabi_mhz"] for r in active if abs(r["center_hz"]) > center_cap]
    leakage_cap = float(constraints.get("max_final_leakage", .005))
    leakage = [r["rabi_mhz"] for r in active if r["final_leakage_max"] > leakage_cap]
    preserved = accepted and not any((width_fail, regression_contrast, satellites, centers, leakage))
    operating_contrast = next(r["contrast"] for r in rows if r["rabi_mhz"] == 50.)
    return dict(
        feasible=bool(preserved and not contrast_fail),
        preserve_original=bool(preserved and operating_contrast > floor),
        preserve_original_definition="C>0.4 at50 MHz, with all no-regression, center, satellite, and leakage criteria across the band",
        all_dips_accepted=accepted, contrast_failure_amplitudes_mhz=contrast_fail,
        linewidth_regression_amplitudes_mhz=width_fail,
        contrast_regression_amplitudes_mhz=regression_contrast,
        satellite_failure_amplitudes_mhz=satellites, center_failure_amplitudes_mhz=centers,
        leakage_failure_amplitudes_mhz=leakage,
        min_contrast=min((r["contrast"] for r in active if np.isfinite(r["contrast"])), default=np.nan),
        max_fwhm_khz=max((r["fwhm_hz"] / 1000 for r in active
                          if np.isfinite(r["fwhm_hz"])), default=np.nan),
        max_satellite_prominence=max(r["satellite_prominence"] for r in active),
        max_final_leakage=max(r["final_leakage_max"] for r in active),
    )


def validate(args):
    output = args.output
    (output / "evaluations").mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.candidates.read_text())
    constraints = dict(amplitude_min_mhz=5., amplitude_max_mhz=60., contrast_min=.4,
                       no_regression_width_khz_tolerance=.25,
                       no_regression_contrast_tolerance=.001,
                       max_center_khz=2., max_satellite_prominence=.01,
                       max_final_leakage=.005)
    constraints.update(manifest.get("constraints", {}))
    cases = load_cases(manifest)
    lo, hi = constraints["amplitude_min_mhz"], constraints["amplitude_max_mhz"]
    amplitudes = np.unique(np.r_[np.arange(lo, args.map_max_mhz + 1e-9, args.map_step_mhz),
                                np.arange(lo, hi + 1e-9, args.robustness_step_mhz), 50., hi])
    detuning = np.linspace(-.5, .5, 401)
    fine_detuning = np.linspace(-.5, .5, 801)
    check_detuning = np.linspace(-.5, .5, 201)
    wide_detuning = np.linspace(-2, 2, 801)
    wide_amplitudes = np.unique(np.r_[lo, 10., 20., 40., 50., hi])
    wide_amplitudes = wide_amplitudes[(wide_amplitudes >= lo) & (wide_amplitudes <= hi)]
    arrays = dict(detuning_mhz=detuning, rabi_mhz=amplitudes,
                  fine_detuning_mhz=fine_detuning, check_detuning_mhz=check_detuning,
                  wide_detuning_mhz=wide_detuning, wide_rabi_mhz=wide_amplitudes)
    results = {}
    for name, case in cases.items():
        pulse = case["pulse"]
        normalization = peak_scale(pulse)
        core_window = min(pulse.duration_us / 2, max(.05, 32 * pulse.sigma_us))
        waveform_time = np.unique(np.r_[np.linspace(-pulse.duration_us/2, pulse.duration_us/2, 6001),
                                        np.linspace(-core_window, core_window, 4001)])
        arrays[f"{name}_waveform_time_us"] = waveform_time
        arrays[f"{name}_waveform_50mhz"] = 50 * pulse.drive(waveform_time) / normalization
        if hasattr(pulse, "phase"):
            phase_profile = pulse.phase(waveform_time)[0]
        elif isinstance(pulse, PhaseEchoPulse):
            phase_profile = pulse.phase_rad * np.exp(-(waveform_time / pulse.phase_scale_us)**2)
        else:
            phase_profile = np.zeros_like(waveform_time)
        arrays[f"{name}_phase_rad"] = phase_profile
        populations, diagnostics = cached_native(output, pulse, amplitudes / normalization, detuning)
        rows = metrics(populations, amplitudes, detuning)
        for row, population in zip(rows, populations[1], strict=True):
            row["local_shoulders"] = local_shoulder_metrics(detuning, population)
        index50 = int(np.flatnonzero(amplitudes == 50)[0])
        fine_time, _ = cached_native(output, pulse, [50 / normalization], detuning, dt_ns=.125)
        fine_grid, _ = cached_native(output, pulse, [50 / normalization], fine_detuning, dt_ns=.125)
        metric_fine = dip_metrics(fine_detuning, fine_grid[1, 0])
        metric_time = dip_metrics(detuning, fine_time[1, 0])
        verification = dict(
            native_diagnostics=diagnostics,
            native_dt_halving_max_population=float(abs(populations[:, index50] - fine_time[:, 0]).max()),
            native_dt_halving_fwhm_change_khz=abs(rows[index50]["fwhm_hz"] - metric_time["fwhm_hz"]) / 1000,
            detuning_halving_fwhm_change_khz=abs(metric_time["fwhm_hz"] - metric_fine["fwhm_hz"]) / 1000,
            fine_grid_50mhz=metric_fine,
        )
        wide, wide_diag = cached_native(output, pulse, wide_amplitudes / normalization, wide_detuning)
        wide_rows = [dict(rabi_mhz=float(amp), **satellite_metrics(wide_detuning, pe),
                          max_final_leakage=float(pf.max())) for amp, pe, pf in
                     zip(wide_amplitudes, wide[1], wide[2], strict=True)]
        arrays[name] = populations
        arrays[f"{name}_fine50"] = fine_grid[:, 0]
        arrays[f"{name}_wide"] = wide
        verification["wide_native_diagnostics"] = wide_diag
        if not case["control"]:
            native, _ = cached_native(output, pulse, [50 / normalization], check_detuning, dt_ns=.125)
            split = {}
            for levels in (3, 4, 5):
                split[levels] = cached_split(output, pulse, 50 / normalization,
                                             check_detuning, levels, 2)
                arrays[f"{name}_levels{levels}"] = split[levels]
            split_fine = cached_split(output, pulse, 50 / normalization, check_detuning, 4, 4)
            verification.update(
                split3_vs_native_max_pe=float(abs(split[3][1] - native[1, 0]).max()),
                levels3_vs4_max_pe=float(abs(split[3][1] - split[4][1]).max()),
                levels4_vs5_max_pe=float(abs(split[4][1] - split[5][1]).max()),
                split4_dt_refinement_max_pe=float(abs(split[4][1] - split_fine[1]).max()),
                levels4_max_final_leakage=float(split_fine[2:].sum(axis=0).max()),
                levels5_max_final_leakage=float(split[5][2:].sum(axis=0).max()),
                levels4_metrics=dip_metrics(check_detuning, split_fine[1]),
                levels5_metrics=dip_metrics(check_detuning, split[5][1]),
            )
            arrays[f"{name}_levels4_refined"] = split_fine
        bandwidth = []
        if not case["control"] and not args.skip_bandwidth:
            sample_dt = .000125
            time = np.arange(-pulse.duration_us/2, pulse.duration_us/2 + sample_dt/2, sample_dt)
            raw = pulse.drive(time) / normalization
            for sigma_ns in (1., 2., 4.):
                wave = gaussian_filter1d(raw, sigma_ns / 1000 / sample_dt, mode="constant", cval=0)
                retained = float(abs(wave).max())
                filtered = SampledPulse(pulse, time, wave / retained, sigma_ns)
                filtered_pop, filtered_diag = cached_native(output, filtered, wide_amplitudes,
                                                             detuning, dt_ns=.125)
                bandwidth.append(dict(gaussian_sigma_ns=sigma_ns,
                                      amplitude_3db_mhz=np.sqrt(np.log(2)) / (2*np.pi*sigma_ns/1000),
                                      postfilter_peak_retention=retained,
                                      renormalized_to_target_peak=True,
                                      rows=metrics(filtered_pop, wide_amplitudes, detuning),
                                      diagnostics=filtered_diag))
                arrays[f"{name}_filter{int(sigma_ns)}ns"] = filtered_pop
        results[name] = dict(label=case["label"], control=case["control"], pulse_type=type(pulse).__name__,
                             pulse=asdict(pulse), unit_waveform_peak=normalization,
                             scale_for_actual_50mhz=50 / normalization, rows=rows,
                             at50=rows[index50], validation=verification, broad_spectrum=wide_rows,
                             bandwidth=bandwidth, selection_record=case.get("selection_record"))
        save_json(output / f"{name}_validation.json", results[name])
        print(json.dumps(clean(dict(case=name, at50=rows[index50], validation=verification))), flush=True)
    for result in results.values():
        result["constraints"] = assess_constraints(result["rows"], results["baseline"]["rows"], constraints)
    reference = 1000 / (np.pi*T2)
    feasible = [name for name, result in results.items() if not result["control"] and
                result["constraints"]["feasible"]]
    preserved = [name for name, result in results.items() if not result["control"] and
                 result["constraints"]["preserve_original"]]
    source_files = MODEL_FILES + [Path(__file__), ROOT / "scripts/optimize_50mhz_echo.py",
                                 ROOT / "echospec/analysis/dip_metrics.py"]
    metadata = dict(t1_us=T1, t2_us=T2, reference_fwhm_khz=reference,
                    nominal_operating_peak_mhz=50., constraints=constraints,
                    actual_peak_rabi_grid_mhz=amplitudes.tolist(), detuning_step_khz=2.5,
                    amplitude_definition="maximum absolute complete I-iQ waveform; includes derivative correction",
                    contrast_definition="linear-background dip depth, fitted at |detuning|>=0.4 MHz within +-0.5 MHz",
                    local_shoulder_definition=local_shoulder_metrics.__doc__,
                    feasible_new_candidates=feasible,
                    preserve_original_candidates=preserved,
                    target_reached=any(results[name]["validation"]["fine_grid_50mhz"]["fwhm_hz"] <=
                                       reference * 1000 for name in feasible),
                    preserve_original_target_reached=any(results[name]["validation"]["fine_grid_50mhz"]["fwhm_hz"] <=
                                                         reference * 1000 for name in preserved),
                    input_sha256={str(args.candidates): sha(args.candidates), str(PREVIOUS): sha(PREVIOUS)},
                    source_sha256={str(path.relative_to(ROOT)): sha(path) for path in source_files},
                    hardware_validation=False)
    np.savez_compressed(output / "validated_spectra.npz", **arrays)
    save_json(output / "results.json", dict(metadata=metadata, cases=results))
    return cases


def render(output):
    results = json.loads((output / "results.json").read_text())
    cases = results["cases"]
    with np.load(output / "validated_spectra.npz") as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    x, amplitudes = arrays["detuning_mhz"], arrays["rabi_mhz"]
    names = list(cases)

    def format_value(value, digits=2, scale=1):
        return "unresolved" if value is None else f"{value / scale:.{digits}f}"

    lo, hi = [results["metadata"]["constraints"][key]
              for key in ("amplitude_min_mhz", "amplitude_max_mhz")]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    def make_maps(plot_names, stem, title):
        fig, axes = plt.subplots(2, len(plot_names), figsize=(4.2*len(plot_names)+.8, 8),
                                 squeeze=False, layout="constrained")
        for column, name in enumerate(plot_names):
            for row in range(2):
                ax = axes[row, column]
                mesh = ax.pcolormesh(x*1000, amplitudes, arrays[name][1], shading="nearest",
                                     cmap="viridis", vmin=0, vmax=1, rasterized=True)
                ax.axhline(50, color="white", ls="--", lw=1, alpha=.85)
                ax.set(xlabel="Detuning (kHz)", ylabel="Actual peak Rabi rate (MHz)" if column == 0 else "")
                if row == 0:
                    m = cases[name]["at50"]
                    ax.set(xlim=(-500, 500), ylim=(lo, hi),
                           title=f"{cases[name]['label']}\n50 MHz: {format_value(m['fwhm_hz'], 1, 1000)} kHz, C={format_value(m['contrast'], 3)}")
                else:
                    ax.set(xlim=(-250, 250), ylim=(lo, min(20, hi)), title="Low-drive detail")
        fig.colorbar(mesh, ax=axes.ravel().tolist(), pad=.015, shrink=.92).set_label("Excited-state probability Pₑ")
        fig.suptitle(title, fontsize=15)
        fig.supxlabel("Shared population scale; no interpolation. Dashed line: 50 MHz.", fontsize=10)
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(output / f"{stem}.{suffix}", dpi=180)
        plt.close(fig)

    make_maps(names, "maps_2d", "Independent core, tails, edge taper, and phase · single-echo maps")
    preserving_names = ["baseline"] + [name for name, case in cases.items()
                                         if not case["control"] and case["constraints"].get("preserve_original", False)]
    make_maps(preserving_names, "maps_preserving_start",
              "Original pulse and candidates preserving the low-drive start")

    palette = ["#777777", "#c35a2b", "#136f9c", "#37856a", "#8759b4"]
    colors = [palette[index % len(palette)] for index in range(len(names))]
    waveform_html = ""
    if all(f"{name}_waveform_time_us" in arrays for name in names):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained")
        for color, name in zip(colors, names, strict=True):
            time, wave = arrays[f"{name}_waveform_time_us"], arrays[f"{name}_waveform_50mhz"]
            kwargs = dict(color=color, label=cases[name]["label"])
            axes[0].plot(time*1000, abs(wave), **kwargs)
            axes[1].plot(time, abs(wave), **kwargs)
            axes[2].plot(time, arrays[f"{name}_phase_rad"], **kwargs)
        axes[0].set(xlim=(-50, 50), xlabel="Time (ns)", ylabel="|I−iQ| (MHz)", title="Core and echo reversal")
        axes[1].set(xlabel="Time (µs)", ylabel="|I−iQ| (MHz)", title="Core, tails, and edge closure",
                    yscale="log", ylim=(1e-4, 60))
        axes[1].set_xscale("symlog", linthresh=.01)
        axes[2].set(xlabel="Time (µs)", ylabel="Phase before DRAG (rad)", title="Even phase profile")
        axes[2].legend(fontsize=8)
        fig.savefig(output / "waveforms.png", dpi=180)
        fig.savefig(output / "waveforms.pdf")
        plt.close(fig)
        waveform_html = '<img src="waveforms.png" alt="Saved core, tail, and phase waveforms at50 MHz">'
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    index50 = int(np.flatnonzero(amplitudes == 50)[0])
    for color, name in zip(colors, names, strict=False):
        case = cases[name]
        kwargs = dict(color=color, label=case["label"])
        axes[0, 0].plot(x*1000, arrays[name][1, index50], **kwargs)
        axes[0, 1].plot(amplitudes, [row["fwhm_hz"]/1000 if row["fwhm_hz"] is not None else np.nan
                                   for row in case["rows"]], **kwargs)
        axes[1, 0].plot(amplitudes, [row["contrast"] for row in case["rows"]], **kwargs)
        axes[1, 1].plot(amplitudes, np.array([row["final_leakage_max"] for row in case["rows"]])*100, **kwargs)
    axes[0, 0].set(xlim=(-250, 250), xlabel="Detuning (kHz)", ylabel="Pₑ", title="Actual 50 MHz peak")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].axhline(results["metadata"]["reference_fwhm_khz"], color="black", ls="--")
    axes[0, 1].set(xlim=(lo, hi), xlabel="Actual peak Rabi rate (MHz)", ylabel="FWHM (kHz)")
    axes[1, 0].axhline(.4, color="black", ls="--")
    axes[1, 0].set(xlim=(lo, hi), xlabel="Actual peak Rabi rate (MHz)", ylabel="Absolute dip contrast")
    axes[1, 1].set(xlim=(lo, hi), xlabel="Actual peak Rabi rate (MHz)", ylabel="Maximum final P₂ (%)")
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    wide_index = int(np.flatnonzero(arrays["wide_rabi_mhz"] == 50)[0])
    for color, name in zip(colors, names, strict=False):
        ax.plot(arrays["wide_detuning_mhz"], arrays[f"{name}_wide"][1, wide_index],
                color=color, label=cases[name]["label"])
    ax.set(xlabel="Detuning (MHz)", ylabel="Pₑ", title="Broad scan at an actual 50 MHz peak")
    ax.legend(fontsize=9)
    fig.savefig(output / "broad_spectra.png", dpi=180)
    plt.close(fig)

    table = "".join(
        f"<tr><td>{html.escape(case['label'])}</td><td>{format_value(case['at50']['fwhm_hz'], 2, 1000)}</td>"
        f"<td>{format_value(case['at50']['contrast'], 3)}</td>"
        f"<td>{format_value(case['at50'].get('local_shoulders', {}).get('contrast'), 3)}</td>"
        f"<td>{format_value(case['constraints']['min_contrast'], 3)}</td>"
        f"<td>{format_value(case['constraints']['max_fwhm_khz'])}</td><td>{'PASS' if case['constraints']['feasible'] else 'FAIL'}</td>"
        f"<td>{'PASS' if case['constraints'].get('preserve_original', False) else 'FAIL'}</td></tr>"
        for case in cases.values())
    feasible = results["metadata"]["feasible_new_candidates"]
    conclusion = ("New candidates passed all specified constraints on the validation grid: " +
                  ", ".join(html.escape(cases[name]["label"]) for name in feasible) + ".") if feasible else (
        "No new candidate passed all specified constraints on the validation grid. The displayed candidates must not be interpreted as meeting the full requested target.")
    preserved = results["metadata"].get("preserve_original_candidates", [])
    if preserved:
        conclusion += (" The separate preserve-original condition passed for " +
                       ", ".join(html.escape(cases[name]["label"]) for name in preserved) +
                       ": C>0.4 at 50 MHz with no regression over the amplitude band. This does not require C>0.4 everywhere.")
    high_contrast_note = ""
    high_contrast = cases.get("preserve_high_contrast")
    if high_contrast:
        passing = [row["rabi_mhz"] for row in high_contrast["rows"]
                   if lo <= row["rabi_mhz"] <= hi and row["contrast"] is not None and row["contrast"] > .4]
        failures = high_contrast["constraints"]["contrast_failure_amplitudes_mhz"]
        if passing and not any(amp >= min(passing) for amp in failures):
            failed_text = ", ".join(f"{amp:g}" for amp in failures)
            high_contrast_note = (f"<p>The higher-contrast candidate has C&gt;0.4 at every validated amplitude from {min(passing):g} to {hi:g} MHz. "
                                  f"It falls below that absolute floor at {failed_text} MHz, while still improving on the original pulse there. "
                                  "Both preserve-original verdicts cover the full requested amplitude band; they are not restricted to high drive.</p>")
    ref = results["metadata"]["reference_fwhm_khz"]
    parameters = html.escape(json.dumps({name: case["pulse"] for name, case in cases.items()}, indent=2))
    constraints_text = html.escape(json.dumps(results["metadata"]["constraints"], indent=2))
    verification = html.escape(json.dumps({name: case["validation"] for name, case in cases.items()
                                           if not case["control"]}, indent=2))
    theory_link = ('<p><a href="theory_note.md">Theory and interpretation note</a></p>'
                   if (output / "theory_note.md").exists() else "")
    implementation_link = ('<p><a href="implementation_validation.json">Implementation checks: focused tests and repository-wide exceptions</a></p>'
                           if (output / "implementation_validation.json").exists() else "")
    search_links = ""
    if (output / "search_tradeoffs.png").exists():
        search_links += '<p><a href="search_tradeoffs.png">Search tradeoffs figure</a></p>'
    if (output / "search_summary.json").exists():
        search_links += '<p><a href="search_summary.json">Search summary and evaluated campaigns</a></p>'
    bandwidth_html = ""
    bandwidth_rows = ""
    for case in cases.values():
        if not case.get("bandwidth"):
            continue
        values = [case["at50"]["contrast"]]
        values += [next(row["contrast"] for row in check["rows"] if row["rabi_mhz"] == 50.)
                   for check in case["bandwidth"]]
        bandwidth_rows += (f"<tr><td>{html.escape(case['label'])}</td>" +
                           "".join(f"<td>{format_value(value, 4)}</td>" for value in values) + "</tr>")
    if bandwidth_rows:
        bandwidth_html = f"""<h2>Illustrative bandwidth sensitivity at 50 MHz</h2>
<p>Absolute contrast after Gaussian convolution of the complete waveform, with the actual peak restored to 50 MHz. The 1, 2, and 4 ns kernel widths correspond to approximately 133, 66, and 33 MHz amplitude-response −3 dB bandwidths. These are test transfer functions, not measured hardware responses. Filtered linewidths and sampled amplitude scans are saved in results.json.</p>
<table><tr><th>Candidate</th><th>Unfiltered C</th><th>1 ns C</th><th>2 ns C</th><th>4 ns C</th></tr>{bandwidth_rows}</table>"""
    ablation_html = ""
    if (output / "ablation.json").exists():
        ablation = json.loads((output / "ablation.json").read_text())
        labels = dict(selected="Selected narrow candidate", phase_off="Phase turned off",
                       unit_tail_gain="Tail gain set to 1", taper_over_whole_half="Taper over the whole half-pulse")
        ablation_rows = ""
        for name, case in ablation["cases"].items():
            low = next(row for row in case["rows"] if row["rabi_mhz"] == 5.)
            nominal = next(row for row in case["rows"] if row["rabi_mhz"] == 50.)
            ablation_rows += (f"<tr><td>{html.escape(labels.get(name, name))}</td>"
                              f"<td>{format_value(low['fwhm_hz'], 2, 1000)}</td><td>{format_value(low['contrast'], 4)}</td>"
                              f"<td>{format_value(nominal['fwhm_hz'], 2, 1000)}</td><td>{format_value(nominal['contrast'], 4)}</td></tr>")
        ablation_html = f"""<h2>What each part contributes</h2>
<p>These one-at-a-time changes start from the narrower candidate, renormalize the complete waveform to the same actual peak, and are <b>not reoptimized</b>. The table uses the coarse search grid and a 0.5 ns maximum integration step. It is a local comparison, not a proof of global optimality or a unique causal decomposition.</p>
<table><tr><th>Variation</th><th>FWHM at 5 MHz (kHz)</th><th>C at 5 MHz</th><th>FWHM at 50 MHz (kHz)</th><th>C at 50 MHz</th></tr>{ablation_rows}</table>
<p>Turning off the compact phase barely changes the 50 MHz linewidth but broadens the 5 MHz line; the phase improves low-drive selectivity at some contrast cost. Extending the taper over the whole half-pulse broadens the 50 MHz line substantially. Setting tail gain to one loses nominal contrast. The useful candidates retain a core near 15 ns, close to the original pulse, rather than the previous phase pulse's approximately 1.68 ns core. This avoids its low-drive collapse while allowing the tails and edge closure to be adjusted separately.</p>
<p><a href="ablation.json">Ablation metrics and provenance</a> · <a href="ablation.npz">Ablation population arrays</a></p>"""
    report = f"""<!doctype html><meta charset="utf-8"><title>Decoupled echo optimization</title>
<style>body{{font:17px/1.55 system-ui;max-width:1280px;margin:32px auto;padding:20px;color:#172b3a}}h1,h2{{line-height:1.2}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #ccd5dc}}pre{{font-size:13px;background:#f2f5f7;padding:16px;overflow:auto}}.callout{{padding:18px;background:#edf4f8;border-left:4px solid #136f9c}}a{{color:#136f9c}}</style>
<h1>Independent core, tails, edge taper, and phase</h1><p>Local numerical study. One echo reversal; no free-evolution interval. All operating-point comparisons have an actual maximum |I−iQ| of 50 MHz.</p>
<div class="callout">{conclusion} The homogeneous coherence reference is {ref:.2f} kHz, from T₂={T2} µs. Target reached under all stated constraints: <b>{results['metadata']['target_reached']}</b>.</div>
<p>Robustness is assessed over {lo:g}–{hi:g} MHz, including the low-drive start. The original p=5 pulse and previous phase-shaped pulse are controls, not automatic feasible solutions. Primary contrast is the absolute depth below a linear background fitted at 0.4–0.5 MHz from resonance; local-shoulder and broad-spectrum metrics are reported separately in the data.</p>
<table><tr><th>Shape</th><th>FWHM at 50 (kHz)</th><th>C at 50</th><th>Local-shoulder C at 50</th><th>Minimum C in band</th><th>Maximum FWHM in band (kHz)</th><th>Strict full band</th><th>Preserve original + C₅₀&gt;0.4</th></tr>{table}</table>
<p>The strict verdict requires C&gt;0.4 across the full amplitude band. The separate preserve-original verdict requires C&gt;0.4 only at 50 MHz while retaining every amplitude's baseline contrast and linewidth, plus the same center, satellite, and leakage checks. The minimum contrast in the band is shown explicitly for both interpretations.</p>
{high_contrast_note}
<img src="maps_preserving_start.png" alt="Original pulse and candidates passing the preserve-original verdict"><p><a href="maps_preserving_start.pdf">PDF of candidates preserving the start</a>. This compact comparison includes only new candidates passing the preserve-original verdict on the full validation grid, alongside the original control. It does not imply C&gt;0.4 at every amplitude.</p>
<details><summary>All candidates and controls, including rejected trials</summary>
<img src="maps_2d.png" alt="Population maps and low-drive zooms"><p><a href="maps_2d.pdf">PDF maps</a> · <a href="maps_2d.svg">SVG maps</a>. Every panel uses the same 0–1 probability scale, with no interpolation. Each shape is held fixed as its amplitude is swept.</p>
</details>
<img src="comparison.png" alt="Spectral slices, linewidth, contrast, and final leakage"><img src="broad_spectra.png" alt="Broad-spectrum comparison">
<h2>What is independently controlled</h2><p>The new positive envelope multiplies a root-Lorentzian core by a separately weighted tail, a generalized Lorentzian tail decay, and a local quintic endpoint taper. Core width, tail weight/decay, and edge duration are distinct parameters. Two even Gaussian phase components preserve odd time-reversed control before the complete complex-waveform DRAG correction. The ideal resonant two-level cancellation does not imply constant linewidth or contrast in the dissipative multilevel model.</p>{waveform_html}
{theory_link}
{search_links}
{ablation_html}
<h2>Validation boundary</h2><p>The model uses T₁={T1} µs, T₂={T2} µs, α/2π=−216 MHz. New candidates receive time-step and frequency-grid checks at 50 MHz plus independent three-/four-/five-level propagation. The main maps are three-level simulations. Local shoulder contrast can reveal a misleading broad background; the ±2 MHz scans inspect additional dips. Gaussian bandwidth checks, when present, convolve the entire I/Q waveform and then restore its actual peak. They are illustrative transfer functions, not measurements of the hardware. No hardware experiment is run.</p>
{implementation_link}
{bandwidth_html}
<h2>Exact constraints</h2><pre>{constraints_text}</pre><p>Linewidth/contrast regression checks compare each amplitude with the original pulse at that same amplitude. Tolerances are numerical allowances, not improvements to the target. Any unresolved dip fails acceptance; summary minima and maxima include only defined metrics. See the per-amplitude failure lists in results.json. A PASS describes these spectral constraints on the sampled grid; convergence, multilevel, local-shoulder, and broad-spectrum audits remain separate diagnostics.</p>
<h2>Exact pulse parameters</h2><pre>{parameters}</pre><h2>Independent numerical checks</h2><pre>{verification}</pre>
<p><a href="results.json">All metrics, failure lists, bandwidth checks, and provenance</a> · <a href="validated_spectra.npz">Saved population arrays</a> · <a href="comparison.pdf">Comparison PDF</a></p>
<p>Reproduce with <code>.venv/bin/python scripts/report_decoupled_50mhz_echo.py --candidates PATH --stage all</code>. Use <code>--stage render</code> to regenerate figures and this report from saved data without rerunning simulations.</p>
"""
    (output / "report.html").write_text(report)
    save_json(output / "render_provenance.json", dict(generator=str(Path(__file__).relative_to(ROOT)),
                                                      generator_sha256=sha(Path(__file__)),
                                                      inputs={name: sha(output / name) for name in
                                                              ("results.json", "validated_spectra.npz", "ablation.json",
                                                               "theory_note.md", "search_summary.json", "search_tradeoffs.png",
                                                               "implementation_validation.json")
                                                              if (output / name).exists()},
                                                      interpolation="none", population_limits=[0, 1]))
    for link in re.findall(r'(?:href|src)="([^"]+)"', report):
        if not link.startswith("https://") and not (output / link).exists():
            raise RuntimeError(f"Missing report target: {link}")
    print("Report:", output / "report.html", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--stage", choices=("validate", "render", "all"), default="all")
    parser.add_argument("--map-step-mhz", type=float, default=2.5)
    parser.add_argument("--map-max-mhz", type=float, default=60)
    parser.add_argument("--robustness-step-mhz", type=float, default=1)
    parser.add_argument("--skip-bandwidth", action="store_true")
    args = parser.parse_args()
    if args.stage != "render":
        if args.candidates is None:
            parser.error("--candidates is required for numerical validation")
        validate(args)
    if args.stage != "validate":
        render(args.output)


if __name__ == "__main__":
    main()
