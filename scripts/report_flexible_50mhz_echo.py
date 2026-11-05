"""Validate flexible single-echo candidates and render a self-contained report.

The candidate manifest uses ``candidates`` records with name, pulse_type, and
pulse fields. User acceptance requires C>0.4 at 50 MHz and no linewidth or
contrast regression against the original pulse over the held-out 5--60 MHz
grid. Center, satellite, and leakage limits are separate quality guardrails.
Historical outputs are read-only; exactly matching cached control simulations
are copied explicitly into the new output directory and recorded in a ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import inspect
import json
import re
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echospec.analysis.dip_metrics import dip_metrics  # noqa: E402
from echospec.simulation.decoupled_echo import DecoupledEchoPulse  # noqa: E402
from echospec.simulation.flexible_echo import FlexibleEchoPulse  # noqa: E402
from echospec.simulation.polynomial_echo import (  # noqa: E402
    PolynomialEchoPulse,
    simulate_polynomial_echo,
)
from echospec.simulation.selective_echo import simulate_echo_split  # noqa: E402
from scripts.optimize_50mhz_echo import metrics, peak_scale  # noqa: E402
from scripts.report_decoupled_50mhz_echo import (  # noqa: E402
    SampledPulse,
    clean,
    local_shoulder_metrics,
    satellite_metrics,
    save_json,
    sha,
)

OUT = ROOT / "outputs/flexible_echo_50mhz_20261101"
OLD = ROOT / "outputs/decoupled_echo_50mhz_20261101"
MODEL_FILES = [ROOT / "echospec/simulation" / name for name in (
    "decoupled_echo.py", "polynomial_echo.py", "selective_echo.py",
    "qutrit_native.py", "qutrit_rk4.cpp")]
T1, T2 = 51.24, 7.31


def model_sources(pulse):
    files = MODEL_FILES.copy()
    base = pulse.base if isinstance(pulse, SampledPulse) else pulse
    if isinstance(base, FlexibleEchoPulse):
        files.append(ROOT / "echospec/simulation/flexible_echo.py")
    return {path.name: sha(path) for path in files}


class Cache:
    """Exact identity checks and explicit read-only import of historical caches."""

    def __init__(self, output):
        self.output = output
        self.directory = output / "evaluations"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.ledger_path = output / "cache_imports.json"
        self.imports = json.loads(self.ledger_path.read_text()) if self.ledger_path.exists() else []
        if not self.ledger_path.exists():
            save_json(self.ledger_path, self.imports)

    def locate(self, kind, identity, allow_old=False):
        identity_text = json.dumps(identity, sort_keys=True)
        key = hashlib.sha256(identity_text.encode()).hexdigest()
        path = self.directory / f"validate_{kind}_{key}.npz"
        canonical = json.loads(identity_text)
        if not path.exists() and allow_old:
            previous = OLD / "evaluations" / path.name
            if previous.exists():
                with np.load(previous) as saved:
                    if json.loads(str(saved["identity"])) != canonical:
                        raise RuntimeError(f"Historical cache identity mismatch: {previous}")
                shutil.copy2(previous, path)
                self.imports.append(dict(source=str(previous), destination=str(path),
                                         sha256=sha(path), exact_identity_match=True))
                save_json(self.ledger_path, self.imports)
        if path.exists():
            with np.load(path) as saved:
                if json.loads(str(saved["identity"])) != canonical:
                    raise RuntimeError(f"Cache identity mismatch: {path}")
        return path

    def native(self, pulse, amplitudes, detuning, dt_ns=.25, allow_old=False):
        identity = dict(pulse=pulse.identity if isinstance(pulse, SampledPulse) else
                        dict(type=type(pulse).__name__, fields=asdict(pulse)),
                        amplitudes=np.asarray(amplitudes).tolist(), detuning=np.asarray(detuning).tolist(),
                        dt_ns=dt_ns, t1_us=T1, t2_us=T2, sources=model_sources(pulse))
        path = self.locate("native", identity, allow_old)
        if path.exists():
            with np.load(path) as saved:
                return saved["populations"].copy(), json.loads(str(saved["diagnostics"])), path.name
        populations, diagnostics = simulate_polynomial_echo(
            pulse, amplitudes, detuning, dt_ns=dt_ns, t1_us=T1, t2_us=T2)
        np.savez_compressed(path, populations=populations, diagnostics=json.dumps(diagnostics),
                            identity=json.dumps(identity))
        return populations, diagnostics, path.name

    def split(self, pulse, amplitude, detuning, levels, refinement, allow_old=False):
        identity = dict(pulse=asdict(pulse), type=type(pulse).__name__, amplitude=amplitude,
                        detuning=np.asarray(detuning).tolist(), levels=levels, refinement=refinement,
                        t1_us=T1, t2_us=T2, sources=model_sources(pulse))
        path = self.locate("split", identity, allow_old)
        if path.exists():
            with np.load(path) as saved:
                return saved["populations"].copy()
        populations = simulate_echo_split(pulse, amplitude, detuning, levels=levels,
                                          refinement=refinement, t1_us=T1, t2_us=T2)
        np.savez_compressed(path, populations=populations, identity=json.dumps(identity))
        return populations


def control_map(cache, pulse, normalization, amplitudes, detuning, old_amplitudes):
    """Reuse an exactly matched old grid and propagate only additional amplitudes."""
    if not all(amp in amplitudes for amp in old_amplitudes):
        pop, diag, path = cache.native(pulse, amplitudes / normalization, detuning, allow_old=True)
        return pop, diag, [path]
    missing = amplitudes[~np.isin(amplitudes, old_amplitudes)]
    parts = [cache.native(pulse, old_amplitudes / normalization, detuning, allow_old=True)]
    grids = [old_amplitudes]
    if len(missing):
        parts.append(cache.native(pulse, missing / normalization, detuning))
        grids.append(missing)
    populations = np.empty((3, len(amplitudes), len(detuning)))
    for grid, (population, _, _) in zip(grids, parts, strict=True):
        populations[:, np.searchsorted(amplitudes, grid)] = population
    diagnostics = {}
    for key in parts[0][1]:
        values = [part[1][key] for part in parts]
        diagnostics[key] = min(values) if key.startswith("min_") else max(values)
    diagnostics["merged_exact_amplitude_chunks"] = len(parts)
    return populations, diagnostics, [part[2] for part in parts]


def raw_drive(pulse, time):
    if hasattr(pulse, "raw_drive_and_derivative"):
        return pulse.raw_drive_and_derivative(time)[0]
    envelope = pulse.envelope(time)[0]
    sign = np.where(abs(time) < pulse.transition_us / 2,
                    -np.sin(np.pi * time / pulse.transition_us), np.where(time < 0, 1., -1.))
    return envelope * sign


def coherent_grid(pulse, refinement):
    half = pulse.duration_us / 2
    core_end = min(.1, half)
    positive = list(np.linspace(0, core_end, int(np.ceil(core_end / .0005 * refinement)) + 1))
    while positive[-1] < half:
        current = positive[-1]
        positive.append(min(half, current + min(.02, .01 * current) / refinement))
    joins = np.linspace(0, pulse.transition_us / 2, 32 * refinement + 1)
    positive = np.unique(np.r_[positive, joins, half - getattr(pulse, "edge_width_us", 0)])
    positive = positive[(positive >= 0) & (positive <= half)]
    return np.r_[-positive[:0:-1], positive]


def coherent_zero(pulse, amplitudes, normalization, refinement=4):
    """Exact SU(2) step propagators, midpoint sampled, with no dissipation.

    Both raw pre-DRAG and completed complex waveforms are propagated. A phase
    winding center or odd phase can break raw time antisymmetry; completed
    derivative correction can independently alter cancellation.
    """
    grid = coherent_grid(pulse, refinement)
    middle, steps = (grid[1:] + grid[:-1]) / 2, np.diff(grid)
    results = {}
    for name, drive in (("raw", raw_drive(pulse, middle)), ("complete", pulse.drive(middle))):
        ground = np.ones(len(amplitudes), complex)
        excited = np.zeros(len(amplitudes), complex)
        for value, step in zip(drive, steps, strict=True):
            angle = np.pi * amplitudes / normalization * abs(value) * step
            cosine, sine = np.cos(angle), np.sin(angle)
            axis = value / abs(value) if abs(value) else 1.
            ground, excited = (cosine * ground - 1j * sine * axis * excited,
                               cosine * excited - 1j * sine * axis.conjugate() * ground)
        results[name] = abs(excited)**2
        results[f"{name}_identity_infidelity"] = np.maximum(0., 1 - ground.real**2)
    return results


def coherent_cached(cache, pulse, amplitudes, normalization, refinement):
    source = "".join(inspect.getsource(fn) for fn in (raw_drive, coherent_grid, coherent_zero))
    identity = dict(pulse=asdict(pulse), type=type(pulse).__name__, amplitudes=amplitudes.tolist(),
                    normalization=normalization, refinement=refinement,
                    sources=model_sources(pulse), algorithm_sha256=hashlib.sha256(source.encode()).hexdigest())
    path = cache.locate("coherent", identity)
    if path.exists():
        with np.load(path) as saved:
            return {name: saved[name].copy() for name in
                    ("raw", "complete", "raw_identity_infidelity", "complete_identity_infidelity")}
    data = coherent_zero(pulse, amplitudes, normalization, refinement)
    np.savez_compressed(path, **data, identity=json.dumps(identity))
    return data


def waveform_audit(pulse, normalization):
    half = pulse.duration_us / 2
    feature = min(pulse.sigma_us, pulse.transition_us) / 200
    radius = np.geomspace(feature, half, 6001)
    core = min(.2, half)
    time = np.unique(np.r_[-radius[::-1], radius, np.linspace(-half, half, 10001),
                           np.linspace(-core, core, 8001)])
    time = time[np.r_[True, np.diff(time) > 1e-11]]
    wave = pulse.drive(time) * 50 / normalization
    raw = raw_drive(pulse, time) * 50 / normalization
    if hasattr(pulse, "phase"):
        phase, derivative = pulse.phase(time)
    else:
        phase, derivative = np.zeros_like(time), np.zeros_like(time)
    active = abs(wave) >= .01 * abs(wave).max()
    chirp = np.imag(np.conj(wave) * np.gradient(wave, time)) / np.maximum(abs(wave)**2, 1e-300) / (2*np.pi)
    center = complex(pulse.drive(0.)) * 50 / normalization
    raw_center = complex(raw_drive(pulse, np.asarray(0.))) * 50 / normalization
    audit = dict(actual_sampled_peak_mhz=float(abs(wave).max()),
                  negative_time_peak_mhz=float(abs(wave[time < 0]).max()),
                  positive_time_peak_mhz=float(abs(wave[time > 0]).max()),
                  i_min_mhz=float(wave.real.min()), i_max_mhz=float(wave.real.max()),
                  q_min_mhz=float((-wave.imag).min()), q_max_mhz=float((-wave.imag).max()),
                  raw_time_antisymmetry_relative_error=float(abs(raw + raw_drive(pulse, -time) * 50 / normalization).max() / max(abs(raw).max(), 1e-300)),
                  completed_time_antisymmetry_relative_error=float(abs(wave + pulse.drive(-time) * 50 / normalization).max() / abs(wave).max()),
                  programmed_chirp_min_mhz=float(derivative.min() / (2*np.pi)),
                  programmed_chirp_max_mhz=float(derivative.max() / (2*np.pi)),
                  completed_chirp_min_mhz_above_1percent_peak=float(chirp[active].min()),
                  completed_chirp_max_mhz_above_1percent_peak=float(chirp[active].max()),
                  rotation_mix=float(getattr(pulse, "rotation_mix", 0)),
                  odd_phase_coefficients=list(getattr(pulse, "odd_phase_coefficients", ())),
                  center_i_mhz=float(center.real), center_q_mhz=float(-center.imag),
                  completed_center_magnitude_mhz=float(abs(center)),
                  completed_center_fraction_of_peak=float(abs(center) / 50),
                  raw_center_magnitude_mhz=float(abs(raw_center)),
                  raw_center_fraction_of_completed_peak=float(abs(raw_center) / 50))
    return audit, dict(time_us=time, complete_50mhz=wave, raw_50mhz=raw,
                       phase_rad=phase, phase_chirp_mhz=derivative / (2*np.pi), complete_chirp_mhz=chirp)


def fixed_offset_depth(detuning, probability, offset_mhz=.2):
    """Unfitted depth from fixed ±offset probabilities to the central raw minimum.

    The center is the sampled minimum within ±0.25 MHz. The conservative depth
    uses the lower of P(-offset) and P(+offset), with linear interpolation only
    when an offset is between samples. This is separate from primary contrast.
    """
    x, pe = np.asarray(detuning), np.asarray(probability)
    central = np.flatnonzero(abs(x) <= .25)
    index = central[np.argmin(pe[central])]
    left, right = float(np.interp(-offset_mhz, x, pe)), float(np.interp(offset_mhz, x, pe))
    return dict(offset_mhz=offset_mhz, center_detuning_mhz=float(x[index]),
                probability_at_center=float(pe[index]), probability_at_negative_offset=left,
                probability_at_positive_offset=right, conservative_depth=float(min(left, right) - pe[index]),
                mean_offset_depth=float((left + right) / 2 - pe[index]))


def assess(rows, baseline, settings, fine50=None):
    reference = {row["rabi_mhz"]: row for row in baseline}
    width_tolerance = settings["no_regression_width_khz_tolerance"] * 1000
    contrast_tolerance = settings["no_regression_contrast_tolerance"]
    active = [r for r in rows if settings["amplitude_min_mhz"] <= r["rabi_mhz"] <= settings["amplitude_max_mhz"]]
    invalid = [r["rabi_mhz"] for r in active if not r["accepted"]]
    width_fail = [r["rabi_mhz"] for r in active if not r["accepted"] or
                  r["fwhm_hz"] > reference[r["rabi_mhz"]]["fwhm_hz"] + width_tolerance]
    contrast_fail = [r["rabi_mhz"] for r in active if not np.isfinite(r["contrast"]) or
                     r["contrast"] < reference[r["rabi_mhz"]]["contrast"] - contrast_tolerance]
    at50 = fine50 if fine50 is not None else next(r for r in rows if r["rabi_mhz"] == 50.)
    user_pass = not any((invalid, width_fail, contrast_fail)) and at50["accepted"] and at50["contrast"] > settings["contrast_min_at50"]
    centers = [r["rabi_mhz"] for r in active if abs(r["center_hz"]) > settings["max_center_khz"] * 1000]
    satellites = [r["rabi_mhz"] for r in active if r["satellite_prominence"] > settings["max_satellite_prominence"]]
    leakage = [r["rabi_mhz"] for r in active if r["final_leakage_max"] > settings["max_final_leakage"]]
    guardrails = not any((centers, satellites, leakage, invalid))
    return dict(user_constraints_pass=bool(user_pass), quality_guardrails_pass=guardrails,
                accepted=bool(user_pass and guardrails), contrast_at50_pass=bool(at50["contrast"] > settings["contrast_min_at50"]),
                invalid_amplitudes_mhz=invalid, linewidth_regression_amplitudes_mhz=width_fail,
                contrast_regression_amplitudes_mhz=contrast_fail, center_failure_amplitudes_mhz=centers,
                satellite_failure_amplitudes_mhz=satellites, leakage_failure_amplitudes_mhz=leakage,
                minimum_band_contrast=min((r["contrast"] for r in active if np.isfinite(r["contrast"])), default=np.nan),
                worst_width_change_khz=max((r["fwhm_hz"] - reference[r["rabi_mhz"]]["fwhm_hz"]) / 1000 for r in active),
                worst_contrast_change=min(r["contrast"] - reference[r["rabi_mhz"]]["contrast"] for r in active),
                maximum_final_leakage=max(r["final_leakage_max"] for r in active))


def load_cases(manifest):
    old = json.loads((OLD / "results.json").read_text())
    cases = dict(baseline=dict(label="Original polynomial p=5", control=True,
                               pulse=PolynomialEchoPulse(**old["cases"]["baseline"]["pulse"])),
                  prior_narrow=dict(label="Prior narrow pulse", control=True,
                                     pulse=DecoupledEchoPulse(**old["cases"]["preserve_original"]["pulse"])))
    classes = dict(flexible=FlexibleEchoPulse, decoupled=DecoupledEchoPulse, polynomial=PolynomialEchoPulse)
    for record in manifest["candidates"]:
        name = record["name"]
        if not re.fullmatch("[a-z0-9_]+", name) or name in cases:
            raise ValueError(f"Invalid or duplicate candidate name: {name}")
        cases[name] = dict(label=record.get("label", name.replace("_", " ")), control=False,
                           pulse=classes[record["pulse_type"]](**record["pulse"]), selection_record=record)
    return cases, np.asarray(old["metadata"]["actual_peak_rabi_grid_mhz"])


def validate(args):
    output = args.output
    cache = Cache(output)
    manifest_text = args.candidates.read_text()
    manifest_sha256 = hashlib.sha256(manifest_text.encode()).hexdigest()
    manifest = json.loads(manifest_text)
    manifest_snapshot = output / f"candidate_manifest_{manifest_sha256[:16]}.json"
    if not manifest_snapshot.exists():
        manifest_snapshot.write_text(manifest_text)
    settings = dict(amplitude_min_mhz=5., amplitude_max_mhz=60., contrast_min_at50=.4,
                    no_regression_width_khz_tolerance=.05, no_regression_contrast_tolerance=.0001,
                    max_center_khz=2., max_satellite_prominence=.01, max_final_leakage=.005)
    settings.update(manifest.get("constraints", {}))
    cases, old_amplitudes = load_cases(manifest)
    amplitudes = np.unique(np.r_[np.arange(settings["amplitude_min_mhz"], settings["amplitude_max_mhz"] + 1e-9,
                                          args.amplitude_step_mhz), 50., settings["amplitude_max_mhz"]])
    detuning, fine_detuning, check_detuning = np.linspace(-.5, .5, 401), np.linspace(-.5, .5, 801), np.linspace(-.5, .5, 201)
    wide_detuning, wide_amplitudes = np.linspace(-2, 2, 801), np.array([5., 10., 20., 40., 50., 60.])
    arrays = dict(rabi_mhz=amplitudes, detuning_mhz=detuning, fine_detuning_mhz=fine_detuning,
                  check_detuning_mhz=check_detuning, wide_detuning_mhz=wide_detuning, wide_rabi_mhz=wide_amplitudes)
    results = {}
    for name, case in cases.items():
        pulse, control = case["pulse"], case["control"]
        normalization = peak_scale(pulse)
        print(f"{name}: amplitude map", flush=True)
        if control:
            populations, diagnostics, components = control_map(cache, pulse, normalization, amplitudes, detuning, old_amplitudes)
        else:
            populations, diagnostics, component = cache.native(pulse, amplitudes / normalization, detuning)
            components = [component]
        rows = metrics(populations, amplitudes, detuning)
        for row, pe in zip(rows, populations[1], strict=True):
            row["local_shoulders"] = local_shoulder_metrics(detuning, pe)
            row["fixed_offset_depth"] = fixed_offset_depth(detuning, pe)
        index50 = int(np.flatnonzero(amplitudes == 50)[0])
        fine_time, _, _ = cache.native(pulse, [50 / normalization], detuning, dt_ns=.125, allow_old=control)
        fine, _, _ = cache.native(pulse, [50 / normalization], fine_detuning, dt_ns=.125, allow_old=control)
        time_metric, fine_metric = dip_metrics(detuning, fine_time[1, 0]), dip_metrics(fine_detuning, fine[1, 0])
        fine_metric["local_shoulders"] = local_shoulder_metrics(fine_detuning, fine[1, 0])
        fine_metric["fixed_offset_depth"] = fixed_offset_depth(fine_detuning, fine[1, 0])
        wide, wide_diag, _ = cache.native(pulse, wide_amplitudes / normalization, wide_detuning, allow_old=control)
        verification = dict(native_diagnostics=diagnostics, native_map_cache_components=components,
                            native_dt_halving_max_population=float(abs(populations[:, index50] - fine_time[:, 0]).max()),
                            native_dt_halving_fwhm_change_khz=abs(rows[index50]["fwhm_hz"] - time_metric["fwhm_hz"]) / 1000,
                            detuning_halving_fwhm_change_khz=abs(time_metric["fwhm_hz"] - fine_metric["fwhm_hz"]) / 1000,
                            fine_grid_50mhz=fine_metric, wide_native_diagnostics=wide_diag)
        arrays[name], arrays[f"{name}_fine50"], arrays[f"{name}_wide"] = populations, fine[:, 0], wide
        broad = [dict(rabi_mhz=float(amp), **satellite_metrics(wide_detuning, pe), max_final_leakage=float(pf.max()))
                 for amp, pe, pf in zip(wide_amplitudes, wide[1], wide[2], strict=True)]
        print(f"{name}: coherent cancellation and waveform audits", flush=True)
        audit, waveform = waveform_audit(pulse, normalization)
        for key, value in waveform.items():
            arrays[f"{name}_{key}"] = value
        coherent = coherent_cached(cache, pulse, amplitudes, normalization, 4)
        coherent_fine = coherent_cached(cache, pulse, amplitudes, normalization, 8)
        for key, value in coherent_fine.items():
            arrays[f"{name}_coherent_{key}_zero"] = value
        verification["coherent_refinement_max_population"] = {key: float(abs(coherent[key] - coherent_fine[key]).max()) for key in ("raw", "complete")}
        verification["coherent_refinement_max_identity_infidelity"] = {key: float(abs(coherent[f"{key}_identity_infidelity"] - coherent_fine[f"{key}_identity_infidelity"]).max()) for key in ("raw", "complete")}
        verification["coherent_max_on_resonance_population"] = {key: float(coherent_fine[key].max()) for key in ("raw", "complete")}
        verification["coherent_max_identity_infidelity"] = {key: float(coherent_fine[f"{key}_identity_infidelity"].max()) for key in ("raw", "complete")}
        if not control or name == "prior_narrow":
            print(f"{name}: independent multilevel check", flush=True)
            native, _, _ = cache.native(pulse, [50 / normalization], check_detuning, dt_ns=.125, allow_old=control)
            split = {level: cache.split(pulse, 50 / normalization, check_detuning, level, 2, allow_old=control)
                     for level in (3, 4, 5)}
            refined = cache.split(pulse, 50 / normalization, check_detuning, 4, 4, allow_old=control)
            verification.update(split3_vs_native_max_pe=float(abs(split[3][1] - native[1, 0]).max()),
                                levels3_vs4_max_pe=float(abs(split[3][1] - split[4][1]).max()),
                                levels4_vs5_max_pe=float(abs(split[4][1] - split[5][1]).max()),
                                split4_dt_refinement_max_pe=float(abs(split[4][1] - refined[1]).max()),
                                levels4_metrics=dip_metrics(check_detuning, refined[1]),
                                levels5_metrics=dip_metrics(check_detuning, split[5][1]),
                                levels4_max_final_leakage=float(refined[2:].sum(axis=0).max()),
                                levels5_max_final_leakage=float(split[5][2:].sum(axis=0).max()))
            for level, population in split.items():
                arrays[f"{name}_levels{level}"] = population
            arrays[f"{name}_levels4_refined"] = refined
        bandwidth = []
        if not control and not args.skip_bandwidth:
            print(f"{name}: bandwidth checks", flush=True)
            sample_dt = .000125
            time = np.arange(-pulse.duration_us/2, pulse.duration_us/2 + sample_dt/2, sample_dt)
            raw = pulse.drive(time) / normalization
            for sigma in (1., 2., 4.):
                wave = gaussian_filter1d(raw, sigma / 1000 / sample_dt, mode="constant", cval=0)
                retained = float(abs(wave).max())
                filtered = SampledPulse(pulse, time, wave / retained, sigma)
                population, diag, _ = cache.native(filtered, wide_amplitudes, detuning, dt_ns=.125)
                bandwidth.append(dict(gaussian_sigma_ns=sigma, amplitude_3db_mhz=np.sqrt(np.log(2)) / (2*np.pi*sigma/1000),
                                      postfilter_peak_retention=retained, renormalized_to_target_peak=True,
                                      rows=metrics(population, wide_amplitudes, detuning), diagnostics=diag))
                arrays[f"{name}_filter{int(sigma)}ns"] = population
        results[name] = dict(label=case["label"], control=control, pulse_type=type(pulse).__name__, pulse=asdict(pulse),
                             pulse_sha256=hashlib.sha256(json.dumps(asdict(pulse), sort_keys=True).encode()).hexdigest(),
                             unit_waveform_peak=normalization, scale_for_actual_50mhz=50 / normalization,
                             rows=rows, at50=rows[index50], validation=verification, waveform_audit=audit,
                             broad_spectrum=broad, bandwidth=bandwidth, selection_record=case.get("selection_record"))
        save_json(output / f"{name}_validation.json", results[name])
        print(json.dumps(clean(dict(case=name, fine50=fine_metric, waveform_audit=audit))), flush=True)
    for case in results.values():
        case["constraints"] = assess(case["rows"], results["baseline"]["rows"], settings, case["validation"]["fine_grid_50mhz"])
    accepted = [name for name, case in results.items() if not case["control"] and case["constraints"]["accepted"]]
    sources = MODEL_FILES + [ROOT / "echospec/simulation/flexible_echo.py", Path(__file__),
                            ROOT / "scripts/report_decoupled_50mhz_echo.py", ROOT / "scripts/optimize_50mhz_echo.py",
                            ROOT / "echospec/analysis/dip_metrics.py"]
    metadata = dict(t1_us=T1, t2_us=T2, reference_fwhm_khz=1000 / (np.pi*T2), constraints=settings,
                    actual_peak_rabi_grid_mhz=amplitudes.tolist(), detuning_step_khz=2.5,
                    user_constraint_scope="C>0.4 at 50 MHz only; no linewidth or contrast regression vs original across 5--60 MHz",
                    guardrail_scope="Center <=2 kHz, satellite prominence <=0.01, final P2 <=0.005; separate from user constraints",
                    accepted_new_candidates=accepted, hardware_validation=False,
                    contrast_definition="Absolute dip depth below a linear background at |detuning|>=0.4MHz within +-0.5MHz",
                    fixed_offset_depth_definition=fixed_offset_depth.__doc__,
                    amplitude_definition="Maximum absolute completed I-iQ waveform, with derivative correction",
                    historical_cache_imports=cache.imports,
                    candidate_manifest_snapshot=str(manifest_snapshot),
                    input_sha256={str(args.candidates): manifest_sha256, str(OLD / "results.json"): sha(OLD / "results.json")},
                    source_sha256={str(path.relative_to(ROOT)): sha(path) for path in sources})
    save_json(output / "results.json", dict(metadata=metadata, cases=results))
    np.savez_compressed(output / "validated_spectra.npz", **arrays)


def render(output):
    results = json.loads((output / "results.json").read_text())
    cases, metadata = results["cases"], results["metadata"]
    for name, case in cases.items():
        if (name not in {"baseline", "prior_narrow"}
                and not case["constraints"]["accepted"]
                and "rejected" not in case["label"].lower()):
            case["label"] += " — rejected"
    panel_labels = {name: case["label"].replace(" — rejected", "\n(rejected)")
                    for name, case in cases.items()}
    with np.load(output / "validated_spectra.npz") as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    names = list(cases)
    amplitudes, x = arrays["rabi_mhz"], arrays["detuning_mhz"]
    lo, hi = metadata["constraints"]["amplitude_min_mhz"], metadata["constraints"]["amplitude_max_mhz"]
    colors = {name: plt.get_cmap("tab10")(index) for index, name in enumerate(names)}
    colors["baseline"], colors["prior_narrow"] = "#777777", "#136f9c"
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    diagnostics50 = {name: dict(local=local_shoulder_metrics(arrays["fine_detuning_mhz"], arrays[f"{name}_fine50"][1]),
                                fixed=fixed_offset_depth(arrays["fine_detuning_mhz"], arrays[f"{name}_fine50"][1])) for name in names}

    def fmt(value, scale=1, digits=3):
        return "unresolved" if value is None or not np.isfinite(value) else f"{value / scale:.{digits}f}"

    def maps(selected, filename, title):
        fig, axes = plt.subplots(2, len(selected), figsize=(4.2*len(selected)+.8, 8), squeeze=False, layout="constrained")
        for column, name in enumerate(selected):
            for row in range(2):
                ax = axes[row, column]
                mesh = ax.pcolormesh(x*1000, amplitudes, arrays[name][1], cmap="viridis", vmin=0, vmax=1, shading="nearest", rasterized=True)
                ax.axhline(50, color="white", ls="--", lw=1)
                ax.set(xlabel="Detuning (kHz)", ylabel="Actual peak Rabi rate (MHz)" if column == 0 else "",
                       xlim=(-500, 500) if row == 0 else (-250, 250), ylim=(lo, hi) if row == 0 else (lo, 20))
                fine = cases[name]["validation"]["fine_grid_50mhz"]
                ax.set_title(f"{panel_labels[name]}\n50 MHz: {fmt(fine['fwhm_hz'], 1000, 1)} kHz, primary C={fmt(fine['contrast'])}\nLocal-shoulder C={fmt(diagnostics50[name]['local']['contrast'])}" if row == 0 else "Low-drive detail")
        fig.colorbar(mesh, ax=axes.ravel().tolist(), pad=.015).set_label("Excited-state probability Pₑ")
        fig.suptitle(title, fontsize=15)
        fig.supxlabel("Same population scale, no interpolation. Each waveform is fixed while amplitude varies.")
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(output / f"{filename}.{suffix}", dpi=180)
        plt.close(fig)

    maps(names, "maps_all", "Flexible echo: all selected candidates and controls")
    preferred = ["baseline"] + [name for name in metadata["accepted_new_candidates"]
                                 if cases[name].get("selection_record", {}).get("preferred", True)]
    maps(preferred, "maps_preferred", "Candidates preserving the original amplitude response")
    zero = int(np.argmin(abs(x)))
    index50 = int(np.flatnonzero(amplitudes == 50)[0])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for name, case in cases.items():
        kw = dict(color=colors[name], label=case["label"])
        axes[0, 0].plot(x*1000, arrays[name][1, index50], **kw)
        axes[0, 1].plot(amplitudes, [np.nan if row["fwhm_hz"] is None else row["fwhm_hz"] / 1000 for row in case["rows"]], **kw)
        axes[1, 0].plot(amplitudes, [row["contrast"] for row in case["rows"]], **kw)
        axes[1, 1].plot(amplitudes, [100*row["final_leakage_max"] for row in case["rows"]], **kw)
    axes[0, 0].set(xlim=(-250, 250), xlabel="Detuning (kHz)", ylabel="Pₑ", title="Actual 50 MHz peak")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(xlim=(lo, hi), xlabel="Actual peak Rabi rate (MHz)", ylabel="FWHM (kHz)")
    axes[0, 1].axhline(metadata["reference_fwhm_khz"], color="black", ls="--")
    axes[1, 0].set(xlim=(lo, hi), xlabel="Actual peak Rabi rate (MHz)", ylabel="Absolute dip contrast")
    axes[1, 0].plot([50], [.4], marker="+", color="black", markersize=14, label="Required only at 50 MHz")
    axes[1, 1].set(xlim=(lo, hi), xlabel="Actual peak Rabi rate (MHz)", ylabel="Maximum final P₂ (%)")
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.pdf")
    plt.close(fig)

    fig, axes_grid = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    axes = axes_grid.ravel()
    for name, case in cases.items():
        time, wave = arrays[f"{name}_time_us"], arrays[f"{name}_complete_50mhz"]
        kw = dict(color=colors[name], label=case["label"])
        axes[0].plot(time*1000, wave.real, **kw)
        axes[0].plot(time*1000, -wave.imag, color=colors[name], ls="--", alpha=.6)
        axes[1].plot(time, abs(wave), **kw)
        axes[2].plot(time, arrays[f"{name}_phase_rad"], **kw)
        axes[3].plot(time*1000, arrays[f"{name}_phase_chirp_mhz"], **kw)
    axes[0].set(xlim=(-200, 200), xlabel="Time (ns)", ylabel="I, Q (MHz)", title="Center: I solid, Q dashed")
    axes[1].set(xlabel="Time (µs)", ylabel="|I−iQ| (MHz)", yscale="log", ylim=(1e-4, 60), title="Core, tails, and edge closure")
    axes[1].set_xscale("symlog", linthresh=.01)
    axes[2].set(xlabel="Time (µs)", ylabel="Programmed phase φ (rad)", title="Phase shaping of core and tails")
    axes[2].set_xscale("symlog", linthresh=.03)
    axes[3].set(xlim=(-500, 500), xlabel="Time (ns)", ylabel="Programmed phase chirp (MHz)", title="φ′/(2π), excluding center rotation")
    axes[3].legend(fontsize=8)
    fig.savefig(output / "waveforms.png", dpi=180)
    fig.savefig(output / "waveforms.pdf")
    plt.close(fig)

    columns = min(3, len(names))
    fig, axes = plt.subplots(int(np.ceil(len(names) / columns)), columns, figsize=(4.3*columns, 4.3*np.ceil(len(names)/columns)), squeeze=False, layout="constrained")
    for ax, name in zip(axes.ravel(), names, strict=False):
        time, wave = arrays[f"{name}_time_us"], arrays[f"{name}_complete_50mhz"]
        window = max(.05, cases[name]["pulse"]["transition_us"])
        selected = abs(time) <= window
        circle = np.exp(1j * np.linspace(0, 2*np.pi, 401)) * 50
        ax.plot(circle.real, circle.imag, color=".75", ls=":")
        ax.plot(wave[selected].real, -wave[selected].imag, color=colors[name])
        midpoint = int(np.argmin(abs(time)))
        ax.scatter([wave[midpoint].real], [-wave[midpoint].imag], marker="x", color="black", zorder=3)
        first, last = np.flatnonzero(selected)[[0, -1]]
        ax.scatter(wave[[first, last]].real, -wave[[first, last]].imag, marker="o", s=24, color=["#269357", "#b74435"])
        audit = cases[name]["waveform_audit"]
        center_fraction = np.hypot(audit["center_i_mhz"], audit["center_q_mhz"]) / 50
        ax.set(xlabel="I (MHz)", ylabel="Q (MHz)", xlim=(-53, 53), ylim=(-53, 53), aspect="equal",
               title=f"{panel_labels[name]}\nη={audit['rotation_mix']:.3g}; ±{window*1000:.0f} ns\nFull |u(0)|/peak={center_fraction:.3f}")
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    fig.suptitle("Completed I/Q trajectory near the echo center · black × marks t=0")
    fig.savefig(output / "center_iq_trajectory.png", dpi=180)
    fig.savefig(output / "center_iq_trajectory.pdf")
    plt.close(fig)

    fig, axes_grid = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    axes = axes_grid.ravel()
    for name, case in cases.items():
        kw = dict(color=colors[name], label=case["label"])
        axes[0].plot(amplitudes, arrays[name][1, :, zero], **kw)
        axes[1].plot(amplitudes, arrays[f"{name}_coherent_raw_zero"], **kw)
        axes[2].plot(amplitudes, arrays[f"{name}_coherent_complete_zero"], **kw)
        axes[3].plot(amplitudes, arrays[f"{name}_coherent_raw_identity_infidelity_zero"], **kw)
        axes[3].plot(amplitudes, arrays[f"{name}_coherent_complete_identity_infidelity_zero"], color=colors[name], ls="--")
    for ax, title in zip(axes[:3], ("Dissipative three-level model", "Coherent two-level: before DRAG", "Coherent two-level: complete waveform"), strict=True):
        ax.set(xlabel="Actual peak Rabi rate (MHz)", ylabel="On-resonance Pₑ", xlim=(lo, hi), title=title)
    for axis, kind in ((axes[1], "raw"), (axes[2], "complete")):
        axis.set_ylim(0, max(.01, 1.05 * max(arrays[f"{name}_coherent_{kind}_zero"].max() for name in names)))
    axes[3].set(xlabel="Actual peak Rabi rate (MHz)", ylabel="1−|Tr(U)|²/4", xlim=(lo, hi),
                title="Identity infidelity: raw solid, complete dashed", ylim=(0, max(.01, axes[3].get_ylim()[1])))
    axes[0].legend(fontsize=8)
    fig.savefig(output / "on_resonance.png", dpi=180)
    fig.savefig(output / "on_resonance.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5), layout="constrained")
    wide50 = int(np.flatnonzero(arrays["wide_rabi_mhz"] == 50)[0])
    for name in names:
        ax.plot(arrays["wide_detuning_mhz"], arrays[f"{name}_wide"][1, wide50], color=colors[name], label=cases[name]["label"])
    ax.set(xlabel="Detuning (MHz)", ylabel="Pₑ", title="Broad spectrum at an actual 50 MHz peak")
    ax.legend(fontsize=8)
    fig.savefig(output / "broad_spectra.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for ax, amplitude in zip(axes, (5., 10.), strict=True):
        index = int(np.flatnonzero(arrays["wide_rabi_mhz"] == amplitude)[0])
        for name in names:
            ax.plot(arrays["wide_detuning_mhz"], arrays[f"{name}_wide"][1, index], color=colors[name], label=cases[name]["label"])
        ax.axvspan(-.5, .5, color="gray", alpha=.08)
        ax.set(xlabel="Detuning (MHz)", ylabel="Pₑ", ylim=(0, 1), title=f"Broad scan at {amplitude:g} MHz peak")
    axes[0].legend(fontsize=8)
    fig.savefig(output / "broad_low_drive.png", dpi=180)
    fig.savefig(output / "broad_low_drive.pdf")
    plt.close(fig)

    table = ""
    bandwidth = ""
    broad_table = ""
    for name, case in cases.items():
        fine, quality = case["validation"]["fine_grid_50mhz"], case["constraints"]
        table += (f"<tr><td>{html.escape(case['label'])}</td><td>{fmt(fine['fwhm_hz'], 1000, 2)}</td><td>{fmt(fine['contrast'], digits=4)}</td>"
                  f"<td><b>{fmt(diagnostics50[name]['local']['contrast'], digits=4)}</b></td>"
                  f"<td><b>{fmt(diagnostics50[name]['fixed']['conservative_depth'], digits=4)}</b></td><td>{fmt(quality['minimum_band_contrast'], digits=4)}</td>"
                  f"<td>{'PASS' if quality['user_constraints_pass'] else 'FAIL'}</td><td>{'PASS' if quality['quality_guardrails_pass'] else 'FAIL'}</td></tr>")
        if case["bandwidth"]:
            values = [case["at50"]["contrast"]] + [next(row["contrast"] for row in item["rows"] if row["rabi_mhz"] == 50) for item in case["bandwidth"]]
            bandwidth += f"<tr><td>{html.escape(case['label'])}</td>" + "".join(f"<td>{fmt(value, digits=4)}</td>" for value in values) + "</tr>"
        broad_worst = max(case["broad_spectrum"], key=lambda row: row["max_prominence"])
        central_worst = max(row["satellite_prominence"] for row in case["rows"])
        broad_table += (f"<tr><td>{html.escape(case['label'])}</td><td>{fmt(central_worst, digits=4)}</td>"
                        f"<td><b>{fmt(broad_worst['max_prominence'], digits=4)}</b></td><td>{broad_worst['rabi_mhz']:g}</td></tr>")
    accepted_labels = ", ".join(html.escape(cases[name]["label"]) for name in metadata["accepted_new_candidates"])
    outcome = f"Accepted new candidates on the held-out grid: {accepted_labels}." if accepted_labels else "No new candidate passed both user constraints and quality guardrails on the held-out grid."
    optional_names = ("theory_note.md", "search_summary.json", "search_tradeoffs.png", "implementation_validation.json", "ablation.json", "cleaner_local_amplitude_refinement.json", "phase_local_refinement.json")
    optional = " · ".join(f'<a href="{name}">{name.replace("_", " ")}</a>' for name in optional_names if (output / name).exists())
    local_refinement_html = ""
    local_refinement_source = output / "cleaner_local_amplitude_refinement.json"
    if local_refinement_source.exists():
        local_refinement = json.loads(local_refinement_source.read_text())
        matching = [case for case in cases.values()
                    if case["pulse_sha256"] == local_refinement["pulse_sha256"]]
        if matching:
            quality = local_refinement["quality"]
            width = quality["minimum_linewidth_improvement"]
            contrast = quality["minimum_contrast_improvement"]
            passed = "passed" if quality["true_no_width_or_contrast_regression"] else "failed"
            local_refinement_html = f"""<p><b>Focused amplitude refinement:</b> {html.escape(matching[0]['label'])} also {passed} an exact zero-regression comparison at {len(local_refinement['amplitudes_mhz'])} samples with 0.05 MHz spacing in 5–5.5 and 14.5–15.5 MHz. The smallest linewidth improvement was {width['linewidth_improvement_khz']:.6f} kHz at {width['rabi_mhz']:g} MHz; the smallest contrast improvement was {contrast['contrast_improvement']:.7f} at {contrast['rabi_mhz']:g} MHz. Some samples overlap the main amplitude grid. This is sampled verification, not a proof for every point in the continuous interval. <a href="cleaner_local_amplitude_refinement.json">Metrics, exact grids, and provenance</a>.</p>"""
    phase_refinement_html = ""
    phase_refinement_source = output / "phase_local_refinement.json"
    if phase_refinement_source.exists():
        phase_refinement = json.loads(phase_refinement_source.read_text())
        matching = [case for case in cases.values()
                    if case["pulse_sha256"] == phase_refinement["pulse_sha256"]]
        if matching:
            quality = phase_refinement["quality"]
            width = quality["minimum_linewidth_improvement"]
            contrast = quality["minimum_contrast_improvement"]
            passed = "passed" if quality["true_no_width_or_contrast_regression"] else "failed"
            phase_refinement_html = f"""<p><b>The narrow phase-shaped candidate has a very small contrast margin.</b> {html.escape(matching[0]['label'])} {passed} a finer-frequency check at peak amplitudes {', '.join(f'{amplitude:g}' for amplitude in phase_refinement['amplitudes_mhz'])} MHz, using {len(phase_refinement['detuning_mhz'])} detuning points and a {phase_refinement['dt_ns']:g} ns maximum time step. Its smallest contrast improvement over the original was only <b>{contrast['contrast_improvement']:.7f}</b> at {contrast['rabi_mhz']:g} MHz. The smallest linewidth improvement in this spot check was {width['linewidth_improvement_khz']:.6f} kHz at {width['rabi_mhz']:g} MHz. These are sampled numerical results, not a guarantee between samples or against model uncertainty. <a href="phase_local_refinement.json">Spot-check metrics and provenance</a> · <a href="phase_local_check/phase_local_refinement.npz">Population arrays</a>.</p>"""
    ablation_html = ""
    ablation_source = output / "cleaner_ablation/ablation.json"
    if ablation_source.exists():
        ablation = json.loads(ablation_source.read_text())
        labels = dict(as_is="Repaired candidate", amplitude_terms_zero="Extra amplitude terms zero",
                       extra_phase_terms_zero="Extra even-phase terms zero", transition_16ns="Transition fixed at 16 ns")
        ablation_rows = ""
        for name, case in ablation["cases"].items():
            low = next(row for row in case["rows"] if row["rabi_mhz"] == 5.)
            nominal = next(row for row in case["rows"] if row["rabi_mhz"] == 50.)
            ablation_rows += (f"<tr><td>{html.escape(labels.get(name, name))}</td><td>{fmt(nominal['fwhm_hz'], 1000, 3)}</td>"
                              f"<td>{fmt(nominal['contrast'], digits=5)}</td><td>{fmt(low['contrast'], digits=5)}</td></tr>")
        transition_ns = ablation["cases"]["as_is"]["pulse"]["transition_us"] * 1000
        ablation_html = f"""<h2>Why optimizing only the 50 MHz slice is misleading</h2>
<p>These are fixed one-at-a-time changes to the repaired candidate, not reoptimized alternatives. Each completed waveform is renormalized to the same actual peak. Removing the extra even-phase terms retains the inherited phase profile. Values use the saved 2.5 kHz detuning grid with a {ablation['dt_ns']:g} ns maximum time step.</p>
<table><tr><th>Variant</th><th>FWHM at 50 MHz (kHz)</th><th>C at 50 MHz</th><th>C at 5 MHz</th></tr>{ablation_rows}</table>
<p>In this local comparison, the extra phase and the approximately {transition_ns:.1f} ns transition mainly protect low-drive contrast. Shortening the transition improves the nominal 50 MHz slice but loses contrast at the start of the amplitude range. Removing the extra amplitude terms loses nominal contrast. This does not establish necessity or global optimality; it explains the observed tradeoff for this fixed pulse.</p>
<p><a href="cleaner_ablation/ablation.json">Ablation metrics and provenance</a> · <a href="cleaner_ablation/ablation.npz">Ablation population arrays</a></p>"""
    parameters = html.escape(json.dumps({name: case["pulse"] for name, case in cases.items()}, indent=2))
    audits = html.escape(json.dumps({name: dict(waveform=case["waveform_audit"], validation=case["validation"], constraints=case["constraints"]) for name, case in cases.items()}, indent=2))
    report = f"""<!doctype html><meta charset="utf-8"><title>Flexible echo at 50 MHz</title>
<style>body{{font:17px/1.55 system-ui;max-width:1360px;margin:32px auto;padding:20px;color:#172b3a}}h1,h2{{line-height:1.2}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:9px;border-bottom:1px solid #ccd5dc}}pre{{font-size:13px;background:#f2f5f7;padding:16px;overflow:auto}}.callout{{padding:18px;background:#edf4f8;border-left:4px solid #136f9c}}a{{color:#136f9c}}</style>
<h1>Flexible pulse and phase shaping at 50 MHz</h1>
<div class="callout">{outcome} The homogeneous T₂ reference is {metadata['reference_fwhm_khz']:.2f} kHz, using T₂={T2} µs. All results are local simulations.</div>
<p>The contrast floor C&gt;0.4 applies <b>at 50 MHz only</b>. Across {lo:g}–{hi:g} MHz, linewidth and contrast must preserve the original p=5 response. The held-out amplitude grid has {len(amplitudes)} points with {amplitudes[1]-amplitudes[0]:g} MHz spacing. User constraints are shown separately from center, satellite, and leakage guardrails. Minimum contrast across the band is descriptive, not an extra full-band floor.</p>
<p><b>Primary contrast and local notch depth are different measurements.</b> Primary C uses the fitted background at |δ|=0.4–0.5 MHz. A broad pedestal can raise this number while the isolated central notch remains shallower than 0.4. The bold local-shoulder column and fixed-offset depth expose this distinction; acceptance under the historical primary metric does not certify an isolated 0.4-depth notch.</p>
<table><tr><th>Shape</th><th>Primary FWHM₅₀ (kHz)</th><th>Primary C₅₀</th><th><b>Local-shoulder depth₅₀</b></th><th><b>Fixed ±0.2 MHz depth₅₀</b></th><th>Minimum band C</th><th>User constraints</th><th>Guardrails within ±0.5 MHz</th></tr>{table}</table>
<p>The fixed-offset depth is min[P(−0.2 MHz),P(+0.2 MHz)]−P(center), where center is the raw minimum within ±0.25 MHz. It uses no background fit. Local-shoulder depth uses the lower nearest shoulder; if a side has no local maximum, its highest sampled probability within ±0.5 MHz is used. Both are diagnostics, with no change to the primary baseline comparison.</p>
{phase_refinement_html}
<img src="maps_preferred.png" alt="Controls and accepted flexible candidates"><p><a href="maps_preferred.pdf">Preferred maps PDF</a>. Color limits are 0–1 in every panel; no interpolation. Each waveform remains fixed throughout its amplitude sweep.</p>
<details><summary>All selected candidates, including rejected rotation examples</summary><img src="maps_all.png" alt="All candidate maps"><a href="maps_all.pdf">All maps PDF</a></details>
<img src="comparison.png" alt="Spectra, linewidths, contrast, and final leakage"><img src="waveforms.png" alt="Actual I/Q waveforms and programmed phase chirp">
<h2>What happens when the sign reversal rotates through I/Q</h2>
<p>The center parameter η introduces a continuous quadrature component during the echo reversal. It can keep the raw drive away from zero. With η=0 and no odd phase component, the pre-DRAG control is time antisymmetric and cancels exactly in the resonant coherent two-level model. A nonzero η or odd phase removes that guarantee; cancellation must be evaluated rather than assumed. The completed I/Q trajectory includes the full derivative correction. Green/red dots mark the beginning/end of the plotted center interval; the black cross marks t=0.</p>
<p><b>η=0 does not imply a zero physical I/Q center.</b> The derivative correction can already give a nonzero completed u(0). Each trajectory reports the actual completed |u(0)| divided by its 50 MHz peak; the raw and completed center amplitudes are also saved separately in the waveform audit.</p>
<img src="center_iq_trajectory.png" alt="Completed I/Q center trajectories"><img src="on_resonance.png" alt="On-resonance response and coherent cancellation audits">
<p>The coherent audit uses independent exact two-level step propagators with midpoint sampling, no T₁/T₂ loss, and both pre-DRAG and completed control. Its refinement error is saved below. Identity infidelity also detects residual phase rotations that an excited-state population alone misses. Dissipative on-resonance population also includes relaxation, dephasing, and multilevel dynamics. The programmed chirp plot excludes the center rotation; signed I/Q ranges, positive/negative-time peaks, and total-waveform chirp above 1% of peak are recorded in the waveform audit.</p>
{ablation_html}
<h2>Broad spectrum and illustrative bandwidth</h2><img src="broad_spectra.png" alt="Broad detuning scan at 50 MHz"><img src="broad_low_drive.png" alt="Broad scans of all controls and candidates at 5 and 10 MHz">
<p>The shaded region in the low-drive scans is the primary ±0.5 MHz window. A pulse can pass guardrails there and still introduce additional spectral features farther away. The wider audit is separate and must not be described as globally clean just because the central window passes.</p>
<table><tr><th>Shape</th><th>Maximum secondary-dip prominence within ±0.5 MHz</th><th>Maximum within ±2 MHz</th><th>Peak amplitude at broad maximum (MHz)</th></tr>{broad_table}</table>
<p>The broad maximum is evaluated at peak amplitudes 5, 10, 20, 40, 50, and 60 MHz; the central-window maximum uses the full {len(amplitudes)}-point amplitude grid.</p>
<p>Gaussian convolution widths 1/2/4 ns correspond to approximately 133/66/33 MHz amplitude-response −3 dB bandwidths. Each complete filtered I/Q waveform is renormalized to the same actual 50 MHz peak. These are illustrative transfer functions, not hardware measurements. Filtered widths and sampled amplitude sweeps are saved with the contrast results.</p>
<table><tr><th>Candidate</th><th>Unfiltered C₅₀</th><th>1 ns C₅₀</th><th>2 ns C₅₀</th><th>4 ns C₅₀</th></tr>{bandwidth}</table>
<h2>Validation and provenance</h2><p>Maps use the native three-level Lindblad model with T₁={T1} µs, T₂={T2} µs and α/2π=−216 MHz. New candidates receive timestep and frequency-grid refinement at 50 MHz, independent 3/4/5-level checks, final leakage checks, local-shoulder contrast, and ±2 MHz scans. Numerical convergence diagnostics remain separate from the spectral acceptance rules. No hardware experiment was run. Historical controls are imported only when raw-cache identities match exactly; their source files remain unchanged.</p>
{local_refinement_html}
<p>{optional}</p><p><a href="results.json">All metrics and provenance</a> · <a href="validated_spectra.npz">Population and waveform arrays</a> · <a href="cache_imports.json">Exact historical cache imports</a> · <a href="comparison.pdf">Comparison PDF</a> · <a href="center_iq_trajectory.pdf">I/Q trajectory PDF</a></p>
<h2>Exact constraints</h2><pre>{html.escape(json.dumps(metadata['constraints'], indent=2))}</pre>
<h2>Exact pulse parameters</h2><pre>{parameters}</pre><details><summary>Waveform, cancellation, multilevel, and numerical audits</summary><pre>{audits}</pre></details>
<p>Reproduce with <code>.venv/bin/python scripts/report_flexible_50mhz_echo.py --candidates PATH</code>. Use <code>--stage render</code> to regenerate artifacts from saved data.</p>"""
    (output / "report.html").write_text(report)
    inputs = ("results.json", "validated_spectra.npz", "cache_imports.json",
              "cleaner_ablation/ablation.json", "cleaner_ablation/ablation.npz",
              "phase_local_check/phase_local_refinement.npz") + optional_names
    save_json(output / "render_provenance.json", dict(generator=str(Path(__file__).relative_to(ROOT)), generator_sha256=sha(Path(__file__)),
                                                      inputs={name: sha(output / name) for name in inputs if (output / name).exists()},
                                                      interpolation="none", population_limits=[0, 1]))
    for link in re.findall(r'(?:href|src)="([^"]+)"', report):
        if not (output / link).is_file():
            raise RuntimeError(f"Missing report link: {link}")
    print("Report:", output / "report.html", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--stage", choices=("validate", "render", "all"), default="all")
    parser.add_argument("--amplitude-step-mhz", type=float, default=.5)
    parser.add_argument("--skip-bandwidth", action="store_true")
    args = parser.parse_args()
    if args.stage != "render":
        if args.candidates is None:
            parser.error("--candidates is required for validation")
        validate(args)
    if args.stage != "validate":
        render(args.output)


if __name__ == "__main__":
    main()
