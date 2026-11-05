"""Band-by-band constrained descent for a single polynomial echo (two levels).

Reproduce: MPLCONFIGDIR=/tmp/echo-mpl .venv/bin/python scripts/staged_echo_descent.py
The pulse is shared across all amplitudes. Baseline guardrails cover 5--60 MHz
at every stage; the objective alternates a new band and all bands seen so far.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.signal import find_peaks
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echospec.analysis.dip_metrics import dip_metrics  # noqa: E402
from echospec.simulation.bloch_echo import simulate_bloch_echo  # noqa: E402
from echospec.simulation.polynomial_echo import PolynomialEchoPulse  # noqa: E402
from scripts.optimize_50mhz_echo import peak_scale  # noqa: E402
from scripts.report_decoupled_50mhz_echo import (  # noqa: E402
    local_shoulder_metrics,
    save_json,
    sha,
)

DEFAULT_OUT = ROOT / "outputs/staged_bloch_echo_20261101"
LOW = np.array([8.0, -5.0, 1.0, np.log10(0.002)])
HIGH = np.array([70.0, np.log10(0.02), 16.0, np.log10(0.5)])
SPAN = HIGH - LOW
SOURCES = [
    Path(__file__),
    ROOT / "echospec/simulation/bloch_echo.py",
    ROOT / "echospec/simulation/bloch_rk4.cpp",
    ROOT / "echospec/simulation/polynomial_echo.py",
    ROOT / "echospec/analysis/dip_metrics.py",
    ROOT / "scripts/optimize_50mhz_echo.py",
    ROOT / "scripts/report_decoupled_50mhz_echo.py",
]


def pulse_from(z, beta=0.0):
    p = LOW + np.asarray(z) * SPAN
    return PolynomialEchoPulse(
        duration_us=p[0],
        cutoff=10 ** p[1],
        power=p[2],
        transition_us=10 ** p[3],
        drag_beta=beta,
    )


def encode(pulse):
    p = np.array(
        [
            pulse.duration_us,
            np.log10(pulse.cutoff),
            pulse.power,
            np.log10(pulse.transition_us),
        ]
    )
    return (p - LOW) / SPAN


def rows_from(pop, amps, det):
    rows = []
    for amp, pe in zip(amps, pop[1], strict=True):
        row = dict(rabi_mhz=float(amp), **dip_metrics(det, pe))
        peaks, props = find_peaks(-pe, prominence=0)
        other = np.ones(len(peaks), bool)
        if len(peaks):
            other[np.argmin(abs(det[peaks]))] = False
        row["satellite_prominence"] = float(props["prominences"][other].max(initial=0))
        rows.append(row)
    return rows


class Engine:
    def __init__(self, output, beta=0.0, extra_amplitudes=()):
        self.output, self.beta = output, beta
        self.directory = output / "evaluations"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.amps = np.unique(
            np.r_[np.arange(5.0, 60.01, 2.5), 6.0, 9.0, extra_amplitudes]
        )
        self.det = np.linspace(-0.5, 0.5, 201)
        self.lock, self.records, self.count = threading.Lock(), {}, 0
        self.source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in SOURCES}
        self.identity = dict(
            amps=self.amps.tolist(),
            det=self.det.tolist(),
            beta=beta,
            dt_ns=2.0,
            t1_us=51.24,
            t2_us=7.31,
            sources=self.source_hashes,
            parameter_names=[
                "duration_us",
                "log10_cutoff",
                "power",
                "log10_transition_us",
            ],
            bounds=np.stack((LOW, HIGH), axis=1).tolist(),
        )
        self.config_hash = hashlib.sha256(
            json.dumps(self.identity, sort_keys=True).encode()
        ).hexdigest()
        save_json(output / "configuration.json", self.identity)
        snapshot = output / "source_snapshot"
        snapshot.mkdir(exist_ok=True)
        for path in SOURCES:
            (snapshot / (path.name + ".txt")).write_bytes(path.read_bytes())
        save_json(
            snapshot / "manifest.json",
            {
                str(path.relative_to(ROOT)): dict(
                    file=path.name + ".txt",
                    sha256=self.source_hashes[str(path.relative_to(ROOT))],
                )
                for path in SOURCES
            },
        )
        self.baseline = self.evaluate(encode(PolynomialEchoPulse(drag_beta=beta)))
        if not self.baseline["valid"]:
            raise RuntimeError(
                "The reference failed its simulation or dip check; cannot optimize against it."
            )
        self.bw = np.array([r["fwhm_hz"] / 1000 for r in self.baseline["rows"]])
        self.bc = np.array([r["contrast"] for r in self.baseline["rows"]])
        self.i50 = int(np.flatnonzero(self.amps == 50)[0])

    def evaluate(self, z):
        z = np.clip(np.asarray(z, float), 0, 1)
        key = hashlib.sha256(z.tobytes() + self.config_hash.encode()).hexdigest()
        with self.lock:
            if key in self.records:
                return self.records[key]
        path = self.directory / f"{key}.npz"
        if path.exists():
            with np.load(path) as saved:
                record = json.loads(str(saved["record"]))
            assert record["configuration_sha256"] == self.config_hash
        else:
            pulse = pulse_from(z, self.beta)
            try:
                norm = peak_scale(pulse)
                pop, diag = simulate_bloch_echo(pulse, self.amps / norm, self.det)
                rows = rows_from(pop, self.amps, self.det)
                record = dict(
                    valid=all(r["accepted"] for r in rows),
                    pulse=asdict(pulse),
                    normalized=z.tolist(),
                    rows=rows,
                    diagnostics=diag,
                    peak_scale=norm,
                    key=key,
                    configuration_sha256=self.config_hash,
                )
                pe = pop[1]
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                record = dict(
                    valid=False,
                    error=str(exc),
                    pulse=asdict(pulse),
                    normalized=z.tolist(),
                    key=key,
                    configuration_sha256=self.config_hash,
                )
                pe = np.empty(0)
            np.savez_compressed(path, pe=pe, record=json.dumps(record))
            with self.lock:
                self.count += 1
        with self.lock:
            self.records[key] = record
        return record

    def constraints(self, r, seen_hi=60, band_floor=0.0):
        n = len(self.amps)
        if not r["valid"]:
            return np.full(4 * n + 1, -100.0)
        rows = r["rows"]
        w, c = (
            np.array([x["fwhm_hz"] / 1000 for x in rows]),
            np.array([x["contrast"] for x in rows]),
        )
        target_c = self.bc + 0.001
        seen = (self.amps >= 10) & (self.amps <= seen_hi)
        target_c[seen] = np.maximum(target_c[seen], band_floor)
        return np.r_[
            (self.bw - w - 0.3) / 10,
            (c - target_c) / 0.02,
            (c[self.i50] - 0.403) / 0.02,
            [(0.009 - x["satellite_prominence"]) / 0.01 for x in rows],
            [(1.8 - abs(x["center_hz"]) / 1000) / 2 for x in rows],
        ]

    def objective(self, r, lo, hi):
        if not r["valid"]:
            return 1e4
        mask = (self.amps >= lo) & (self.amps <= hi)
        ratios = (
            np.array([x["fwhm_hz"] / 1000 for x in r["rows"]])[mask] / self.bw[mask]
        )
        c = np.array([x["contrast"] for x in r["rows"]])[mask]
        smooth_worst = 0.03 * (logsumexp(ratios / 0.03) - np.log(mask.sum()))
        return float(0.7 * ratios.mean() + 0.3 * smooth_worst - 0.03 * c.mean())

    def quality(self, r, band_floor=0.0, seen_hi=60.0):
        if not r["valid"]:
            return dict(accepted=False)
        w, c = (
            np.array([x["fwhm_hz"] / 1000 for x in r["rows"]]),
            np.array([x["contrast"] for x in r["rows"]]),
        )
        user = np.all(w <= self.bw) and np.all(c >= self.bc) and c[self.i50] > 0.4
        user &= np.all(c[(self.amps >= 10) & (self.amps <= seen_hi)] >= band_floor)
        guards = all(
            x["satellite_prominence"] <= 0.01 and abs(x["center_hz"]) <= 2000
            for x in r["rows"]
        )
        return dict(
            accepted=bool(user and guards),
            user_constraints_pass=bool(user),
            width50_khz=float(w[self.i50]),
            contrast50=float(c[self.i50]),
            minimum_width_gain_khz=float((self.bw - w).min()),
            minimum_contrast_gain=float((c - self.bc).min()),
            minimum_contrast_10_to_60=float(c[self.amps >= 10].min()),
        )


def optimize_stage(
    engine,
    initial,
    label,
    lo,
    hi,
    seen_hi,
    iterations,
    workers,
    band_floor=0.0,
    fixed_power=None,
):
    seen, gradients = {}, {}
    free = [0, 1, 2, 3] if fixed_power is None else [0, 1, 3]
    template = np.array(initial, float)
    if fixed_power is not None:
        template[2] = (fixed_power - LOW[2]) / SPAN[2]

    def expand(x):
        z = template.copy()
        z[free] = x
        return z

    def fg(x):
        r = engine.evaluate(expand(x))
        seen[r["key"]] = r
        return engine.objective(r, lo, hi), engine.constraints(r, seen_hi, band_floor)

    with ThreadPoolExecutor(max_workers=workers) as pool:

        def derivative(x):
            key = np.asarray(x).tobytes()
            if key not in gradients:
                f, g = fg(x)
                points, steps = [], []
                for j in range(len(free)):
                    step = 1e-4 if x[j] < 1 - 1e-4 else -1e-4
                    point = x.copy()
                    point[j] += step
                    points.append(point)
                    steps.append(step)
                values = list(pool.map(fg, points))
                gradients[key] = (
                    np.array(
                        [(v[0] - f) / h for v, h in zip(values, steps, strict=True)]
                    ),
                    np.column_stack(
                        [(v[1] - g) / h for v, h in zip(values, steps, strict=True)]
                    ),
                )
            return gradients[key]

        result = minimize(
            lambda x: fg(x)[0],
            template[free],
            method="SLSQP",
            jac=lambda x: derivative(x)[0],
            bounds=[(0, 1)] * len(free),
            constraints={
                "type": "ineq",
                "fun": lambda x: fg(x)[1],
                "jac": lambda x: derivative(x)[1],
            },
            options=dict(maxiter=iterations, ftol=2e-6),
        )
        fg(result.x)
    # A failed last step must not replace an earlier feasible pulse.
    feasible = [
        r for r in seen.values() if engine.quality(r, band_floor, seen_hi)["accepted"]
    ]
    if feasible:
        selected = min(feasible, key=lambda r: engine.objective(r, lo, hi))
    else:
        selected = min(
            seen.values(),
            key=lambda r: (
                engine.objective(r, lo, hi)
                + 100 * np.maximum(-engine.constraints(r, seen_hi, band_floor), 0).sum()
            ),
        )
    entry = dict(
        label=label,
        objective_band_mhz=[lo, hi],
        seen_up_to_mhz=seen_hi,
        optimizer_success=bool(result.success),
        optimizer_status=str(result.message),
        evaluations_in_stage=len(seen),
        selected=selected,
        quality=engine.quality(selected, band_floor, seen_hi),
    )
    print(
        json.dumps(
            dict(
                stage=label,
                status=str(result.message),
                evaluations=len(seen),
                quality=entry["quality"],
                pulse=selected["pulse"],
            )
        ),
        flush=True,
    )
    return entry


def train(args):
    engine = Engine(args.output, args.drag_beta, args.extra_amplitudes)
    initial = np.array(engine.baseline["normalized"])
    if args.seed_json:
        seed = json.loads(args.seed_json.read_text())
        initial = encode(PolynomialEchoPulse(**seed["pulse"]))
    histories = {}
    for mode in ("curriculum", "joint"):
        history, z = [], initial.copy()
        stages = [("focus_10_20", 10.0, 20.0, 20.0)]
        for lo in (20.0, 30.0, 40.0, 50.0):
            stages.extend(
                [
                    (f"focus_{lo:g}_{lo + 10:g}", lo, lo + 10, lo + 10),
                    (f"joint_10_{lo + 10:g}", 10.0, lo + 10, lo + 10),
                ]
            )
        if mode == "joint":
            stages = [("joint_from_start", 10.0, 60.0, 60.0)]
        for label, lo, hi, seen_hi in stages:
            entry = optimize_stage(
                engine,
                z,
                label,
                lo,
                hi,
                seen_hi,
                args.iterations if mode == "curriculum" else args.joint_iterations,
                args.workers,
                args.band_contrast_floor,
            )
            history.append(entry)
            z = np.array(entry["selected"]["normalized"])
            save_json(args.output / f"{mode}_history.json", history)
        histories[mode] = history
    last = histories["curriculum"][-1]["selected"]
    integer_entries = []
    power = last["pulse"]["power"]
    for order in sorted(
        set((max(1, int(np.floor(power))), min(16, int(np.ceil(power)))))
    ):
        entry = optimize_stage(
            engine,
            last["normalized"],
            f"integer_p{order}",
            10.0,
            60.0,
            60.0,
            args.iterations,
            args.workers,
            args.band_contrast_floor,
            fixed_power=order,
        )
        integer_entries.append(entry)
    save_json(args.output / "integer_orders.json", integer_entries)
    selections = {mode: h[-1]["selected"] for mode, h in histories.items()}
    for entry in integer_entries:
        selections[entry["label"]] = entry["selected"]
    save_json(args.output / "selected.json", selections)
    save_json(
        args.output / "training_summary.json",
        dict(
            new_simulations=engine.count,
            evaluated_records=len(engine.records),
            algorithm="SLSQP with finite-difference gradients; not an adjoint method",
            continuation="New-band objective alternates with cumulative objective; hard guards always cover 5--60 MHz.",
            no_forgetting_scope="Earlier bands are replayed, but may trade improvements while remaining above the ORIGINAL baseline. Their intermediate best values are not frozen.",
            band_contrast_floor=args.band_contrast_floor,
            model="Two levels; no leakage or multilevel Stark shift validation; beta configurable, default zero.",
            comparison="Single joint-start reference, not an equal-budget or global-optimality benchmark.",
        ),
    )


def validate(args):
    selections = json.loads((args.output / "selected.json").read_text())
    amps, det = np.arange(5.0, 60.01, 0.5), np.linspace(-0.5, 0.5, 401)
    cases = {
        "baseline": dict(pulse=asdict(PolynomialEchoPulse(drag_beta=args.drag_beta))),
        **selections,
    }
    directory = args.output / "validation_cache"
    directory.mkdir(exist_ok=True)
    arrays, results = dict(rabi_mhz=amps, detuning_mhz=det), {}
    for name, record in cases.items():
        pulse = PolynomialEchoPulse(**record["pulse"])
        norm = peak_scale(pulse)
        chunks = [amps[i : i + 16] for i in range(0, len(amps), 16)]

        def cached(grid, x, dt, pulse=pulse, norm=norm):
            identity = dict(
                pulse=asdict(pulse),
                amp=grid.tolist(),
                det=x.tolist(),
                dt_ns=dt,
                t1_us=51.24,
                t2_us=7.31,
                sources={str(p.relative_to(ROOT)): sha(p) for p in SOURCES},
            )
            key = hashlib.sha256(
                json.dumps(identity, sort_keys=True).encode()
            ).hexdigest()
            path = directory / f"{key}.npz"
            if path.exists():
                with np.load(path) as saved:
                    assert json.loads(str(saved["identity"])) == identity
                    return saved["populations"].copy(), json.loads(
                        str(saved["diagnostics"])
                    )
            pop, diag = simulate_bloch_echo(pulse, grid / norm, x, max_dt_ns=dt)
            np.savez_compressed(
                path,
                populations=pop,
                diagnostics=json.dumps(diag),
                identity=json.dumps(identity),
            )
            return pop, diag

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            parts = list(pool.map(lambda grid: cached(grid, det, 1.0), chunks))
        population = np.concatenate([p[0] for p in parts], axis=1)
        rows = rows_from(population, amps, det)
        fine_det = np.linspace(-0.5, 0.5, 801)
        fine, diagnostics = cached(np.array([50.0]), fine_det, 0.5)
        fine50 = rows_from(fine, np.array([50.0]), fine_det)[0]
        same_frequency, _ = cached(np.array([50.0]), det, 0.5)
        checks = dict(
            max_timestep_population_change=float(
                abs(same_frequency[:, 0] - population[:, 90]).max()
            ),
            fine50=fine50,
            local50=local_shoulder_metrics(fine_det, fine[1, 0]),
            fine_diagnostics=diagnostics,
            map_diagnostics=[p[1] for p in parts],
        )
        result = dict(
            pulse=record["pulse"], rows=rows, validation=checks, peak_scale=norm
        )
        if name != "baseline":
            baseline = results["baseline"]["rows"]
            wg = np.array(
                [
                    (b["fwhm_hz"] - r["fwhm_hz"]) / 1000
                    for b, r in zip(baseline, rows, strict=True)
                ]
            )
            cg = np.array(
                [
                    r["contrast"] - b["contrast"]
                    for b, r in zip(baseline, rows, strict=True)
                ]
            )
            user = np.all(wg >= 0) and np.all(cg >= 0) and fine50["contrast"] > 0.4
            user &= all(
                r["contrast"] >= args.band_contrast_floor
                for r in rows
                if r["rabi_mhz"] >= 10
            )
            guards = all(
                r["accepted"]
                and r["satellite_prominence"] <= 0.01
                and abs(r["center_hz"]) <= 2000
                for r in rows
            )
            result["constraints"] = dict(
                accepted=bool(user and guards),
                user_pass=bool(user),
                minimum_width_gain_khz=float(wg.min()),
                width_min_at=float(amps[wg.argmin()]),
                minimum_contrast_gain=float(cg.min()),
                contrast_min_at=float(amps[cg.argmin()]),
                width_regression_amplitudes=amps[wg < 0].tolist(),
                contrast_regression_amplitudes=amps[cg < 0].tolist(),
            )
        results[name] = result
        arrays[name], arrays[f"{name}_fine50"] = population, fine[:, 0]
        print(
            "VALIDATED",
            name,
            json.dumps(dict(fine50=fine50, constraints=result.get("constraints"))),
            flush=True,
        )
        save_json(
            args.output / "results.json",
            dict(
                levels=2,
                drag_beta=args.drag_beta,
                band_contrast_floor=args.band_contrast_floor,
                cases=results,
            ),
        )
    arrays["fine_detuning_mhz"] = fine_det
    np.savez_compressed(args.output / "spectra.npz", **arrays)


def audit(args):
    data = json.loads((args.output / "results.json").read_text())
    candidates = {
        n: c
        for n, c in data["cases"].items()
        if n != "baseline" and c["constraints"]["accepted"]
    }
    if not candidates:
        save_json(
            args.output / "dense_audit.json",
            dict(accepted=False, reason="No feasible candidate"),
        )
        return
    best = min(
        candidates, key=lambda n: candidates[n]["validation"]["fine50"]["fwhm_hz"]
    )
    amps, det = np.round(np.linspace(5.0, 60.0, 551), 8), np.linspace(-0.5, 0.5, 401)
    parts, rows = {}, {}
    for name in ("baseline", best):
        pulse = PolynomialEchoPulse(**data["cases"][name]["pulse"])
        norm = peak_scale(pulse)

        def simulate(chunk, pulse=pulse, norm=norm):
            return simulate_bloch_echo(pulse, chunk / norm, det, max_dt_ns=1.0)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            blocks = list(
                pool.map(simulate, [amps[i : i + 24] for i in range(0, len(amps), 24)])
            )
        pop = np.concatenate([b[0] for b in blocks], axis=1)
        parts[name] = pop
        rows[name] = rows_from(pop, amps, det)
        print("DENSE AUDIT", name, "551 amplitude points completed", flush=True)
    wg = np.array(
        [
            (b["fwhm_hz"] - c["fwhm_hz"]) / 1000
            for b, c in zip(rows["baseline"], rows[best], strict=True)
        ]
    )
    cg = np.array(
        [
            c["contrast"] - b["contrast"]
            for b, c in zip(rows["baseline"], rows[best], strict=True)
        ]
    )
    nominal = candidates[best]["validation"]["fine50"]
    spectral_guards = all(
        r["accepted"]
        and r["satellite_prominence"] <= 0.01
        and abs(r["center_hz"]) <= 2000
        for r in rows[best]
    )
    floors = all(
        r["contrast"] >= args.band_contrast_floor
        for r in rows[best]
        if r["rabi_mhz"] >= 10
    )
    result = dict(
        accepted=bool(
            wg.min() >= 0
            and cg.min() >= 0
            and nominal["contrast"] > 0.4
            and spectral_guards
            and floors
        ),
        preferred=best,
        preferred_pulse=candidates[best]["pulse"],
        results_sha256=sha(args.output / "results.json"),
        amplitude_points=len(amps),
        amplitude_step_mhz=0.1,
        detuning_points=len(det),
        minimum_width_gain_khz=float(wg.min()),
        width_min_at_mhz=float(amps[wg.argmin()]),
        minimum_contrast_gain=float(cg.min()),
        contrast_min_at_mhz=float(amps[cg.argmin()]),
        width_regression_amplitudes=amps[wg < 0].tolist(),
        contrast_regression_amplitudes=amps[cg < 0].tolist(),
        rows=rows,
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in SOURCES},
        limitation="Discrete-grid validation; not a proof over the continuous interval.",
    )
    save_json(args.output / "dense_audit.json", result)
    np.savez_compressed(
        args.output / "dense_audit.npz", rabi_mhz=amps, detuning_mhz=det, **parts
    )
    print(
        "AUDIT RESULT",
        json.dumps(
            {k: v for k, v in result.items() if k not in ("rows", "source_sha256")}
        ),
        flush=True,
    )


def render(args):
    data = json.loads((args.output / "results.json").read_text())
    selected_records = json.loads((args.output / "selected.json").read_text())
    if (
        data["drag_beta"] != args.drag_beta
        or data.get("band_contrast_floor", 0.0) != args.band_contrast_floor
        or any(
            n not in data["cases"] or r["pulse"] != data["cases"][n]["pulse"]
            for n, r in selected_records.items()
        )
    ):
        raise ValueError(
            "Saved validation does not match this search; run --stage validate."
        )
    with np.load(args.output / "spectra.npz") as saved:
        arrays = {k: saved[k].copy() for k in saved.files}
    cases = data["cases"]
    names = list(cases)
    audit_file = args.output / "dense_audit.json"
    audit_data = json.loads(audit_file.read_text()) if audit_file.exists() else None
    if (
        audit_data
        and audit_data.get("preferred")
        and audit_data.get("results_sha256") != sha(args.output / "results.json")
    ):
        raise ValueError(
            "Dense audit does not match this validation; run --stage audit."
        )
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(
        1,
        len(names),
        figsize=(3.4 * len(names), 4.5),
        layout="constrained",
        squeeze=False,
    )
    for ax, name in zip(axes[0], names, strict=True):
        c = cases[name]
        f = c["validation"]["fine50"]
        mesh = ax.pcolormesh(
            arrays["detuning_mhz"] * 1000,
            arrays["rabi_mhz"],
            arrays[name][1],
            vmin=0,
            vmax=1,
            shading="nearest",
            cmap="viridis",
        )
        ax.axhline(50, color="white", ls="--", lw=1)
        status = (
            "reference"
            if name == "baseline"
            else ("passes" if c["constraints"]["accepted"] else "fails")
        )
        ax.set(
            xlabel="Detuning (kHz)",
            ylabel="Actual peak Rabi (MHz)",
            title=f"{name} — {status}\n{f['fwhm_hz'] / 1000:.2f} kHz; C={f['contrast']:.4f}",
        )
    fig.colorbar(mesh, ax=axes.ravel().tolist(), label="Pₑ")
    fig.suptitle("Two-level prototype: one pulse shape per panel")
    fig.savefig(args.output / "maps.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for name, c in cases.items():
        axes[0].plot(
            arrays["rabi_mhz"], [r["fwhm_hz"] / 1000 for r in c["rows"]], label=name
        )
        axes[1].plot(arrays["rabi_mhz"], [r["contrast"] for r in c["rows"]], label=name)
    axes[0].axhline(1000 / (np.pi * 7.31), color="gray", ls=":", label="T₂ reference")
    axes[1].axhline(0.4, color="gray", ls=":")
    axes[0].set(xlabel="Actual peak Rabi (MHz)", ylabel="FWHM (kHz)")
    axes[1].set(xlabel="Actual peak Rabi (MHz)", ylabel="Primary contrast")
    axes[0].legend(fontsize=8)
    fig.savefig(args.output / "comparison.png", dpi=180)
    plt.close(fig)
    history = json.loads((args.output / "curriculum_history.json").read_text())
    stages = []
    for h in history:
        p, q = h["selected"]["pulse"], h["quality"]
        stages.append(
            f"<tr><td>{h['label']}</td><td>{p['duration_us']:.3f}</td><td>{p['cutoff']:.7g}</td><td>{p['power']:.4f}</td><td>{1000 * p['transition_us']:.3f}</td><td>{q.get('width50_khz', 0):.3f}</td><td>{q.get('contrast50', 0):.5f}</td><td>{q['accepted']}</td><td>{h['optimizer_status']}</td></tr>"
        )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    amp = np.array([r["rabi_mhz"] for r in history[0]["selected"]["rows"]])
    widths = np.array(
        [[r["fwhm_hz"] / 1000 for r in h["selected"]["rows"]] for h in history]
    )
    contrasts = np.array(
        [[r["contrast"] for r in h["selected"]["rows"]] for h in history]
    )
    for ax, values, title in zip(
        axes, (widths, contrasts), ("FWHM (kHz)", "Contrast"), strict=True
    ):
        image = ax.pcolormesh(amp, np.arange(len(history)), values, shading="nearest")
        ax.set_yticks(range(len(history)), [h["label"] for h in history])
        ax.set(xlabel="Actual peak Rabi (MHz)", title=title)
        fig.colorbar(image, ax=ax)
    fig.savefig(args.output / "stages.png", dpi=180)
    plt.close(fig)
    import html

    selected_name = audit_data.get("preferred") if audit_data else None
    preferred_section = ""
    if selected_name:
        selected = cases[selected_name]
        preferred = selected["validation"]["fine50"]
        baseline = cases["baseline"]["validation"]["fine50"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), layout="constrained")
        for name in ("baseline", selected_name):
            pulse = PolynomialEchoPulse(**cases[name]["pulse"])
            for ax, span in zip(axes, (16.0, 0.1), strict=True):
                time = np.linspace(-span, span, 12001)
                ax.plot(
                    time if span > 1 else time * 1000,
                    (50 * pulse.drive(time) / peak_scale(pulse)).real,
                    label=name,
                )
        axes[0].set(
            xlabel="Time (µs)",
            ylabel="Signed Rabi drive (MHz)",
            title="Full pulse and tails",
            ylim=(-3, 3),
        )
        axes[1].set(
            xlabel="Time (ns)",
            ylabel="Signed Rabi drive (MHz)",
            title="Central reversal; actual peak 50 MHz",
        )
        axes[0].legend()
        fig.savefig(args.output / "waveforms.png", dpi=180)
        plt.close(fig)
        with np.load(args.output / "dense_audit.npz") as dense:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
            for ax, name in zip(axes[:2], ("baseline", selected_name), strict=True):
                mesh = ax.pcolormesh(
                    dense["detuning_mhz"] * 1000,
                    dense["rabi_mhz"],
                    dense[name][1],
                    vmin=0,
                    vmax=1,
                    cmap="viridis",
                    shading="nearest",
                )
                ax.axhline(50, color="white", ls="--", lw=1)
                ax.set(xlabel="Detuning (kHz)", ylabel="Peak Rabi (MHz)", title=name)
            fig.colorbar(mesh, ax=axes[:2], label="Pₑ")
            for name in ("baseline", selected_name):
                axes[2].plot(
                    arrays["fine_detuning_mhz"] * 1000,
                    arrays[f"{name}_fine50"][1],
                    label=name,
                )
            axes[2].set(
                xlim=(-250, 250),
                xlabel="Detuning (kHz)",
                ylabel="Pₑ",
                title="50 MHz slice",
            )
            axes[2].legend(fontsize=9)
            fig.savefig(args.output / "preferred_maps.png", dpi=180)
            plt.close(fig)
        preferred_section = f"""<h2>Preferred candidate: {selected_name}</h2><p>50 MHz FWHM: <b>{preferred["fwhm_hz"] / 1000:.3f} kHz</b> (original {baseline["fwhm_hz"] / 1000:.3f} kHz). Contrast: <b>{preferred["contrast"]:.5f}</b> (original {baseline["contrast"]:.5f}). This is {preferred["fwhm_hz"] / 1000 / (1000 / (np.pi * 7.31)):.2f} times the conventional T₂ linewidth reference, 43.54 kHz. That reference is not a proven bound for this finite-duration echo.</p>
<p>Dense 551-point audit passed: <b>{audit_data["accepted"]}</b>. Smallest linewidth gain: {audit_data["minimum_width_gain_khz"]:.3f} kHz. Smallest contrast gain: {audit_data["minimum_contrast_gain"]:.6f}, at {audit_data["contrast_min_at_mhz"]:g} MHz. <a href="dense_audit.json">Audit metrics</a>.</p><img src="preferred_maps.png"><img src="waveforms.png"><p>The low-power contrast constraint and the lower order boundary p=1 are tight in this run. Simple parameter retuning gives a modest improvement. This does not establish a global optimum or a limit for more flexible pulse/phase families.</p>"""
    page = f"""<!doctype html><meta charset="utf-8"><title>Staged two-level echo optimization</title>
<style>body{{font:16px system-ui;max-width:1250px;margin:30px auto;padding:0 20px}}img{{width:100%}}td,th{{padding:7px;border-bottom:1px solid #ddd}}table{{border-collapse:collapse}}pre{{white-space:pre-wrap}}</style>
<h1>Staged gradient optimization: two-level prototype</h1>
<p>Start with 10–20 MHz, focus on 20–30 MHz, consolidate 10–30 MHz, then repeat through 50–60 MHz. A single waveform is carried forward. Each stage adjusts duration, cutoff, continuous polynomial exponent, and reversal time. Nearby integer orders are checked with the remaining parameters reoptimized.</p>
<p>SLSQP uses finite-difference gradients with log-scaled cutoff/reversal variables. All stages guard the original linewidth and contrast on 5–60 MHz, including the 5–10 MHz protection band. C₅₀ must exceed 0.4. Earlier bands return to the cumulative objective; their best intermediate values are not frozen. This permits trade-offs above the original baseline.</p>
<p><b>This is a two-level model, with DRAG beta={args.drag_beta:g}, T₁=51.24 µs and T₂=7.31 µs.</b> The original reference is recomputed in this same model. No multilevel leakage or hardware validation is claimed. The original three-level result is not substituted for this baseline.</p>
{preferred_section}
<img src="maps.png"><img src="comparison.png"><h2>Stage history</h2><img src="stages.png">
<table><tr><th>Stage</th><th>Length (µs)</th><th>Cutoff</th><th>Exponent</th><th>Reversal (ns)</th><th>FWHM₅₀ (kHz)</th><th>C₅₀</th><th>Training pass</th><th>Solver status</th></tr>{"".join(stages)}</table>
<p>Final maps directly simulate 111 amplitudes and 401 detunings, with finer time/frequency checks at 50 MHz. A grid test is not a continuous-interval proof. The joint-from-start control is not an equal-budget benchmark.</p>
<p>Solver iteration limits are shown above; a feasible selected candidate is not a convergence certificate. The local shoulder contrast is also reported below, independently of the fixed-window background contrast used for optimization.</p>
<h2>Run the script</h2><pre>MPLCONFIGDIR=/tmp/echo-mpl .venv/bin/python scripts/staged_echo_descent.py --output outputs/my_staged_run
# Longer stages:
.venv/bin/python scripts/staged_echo_descent.py --output outputs/my_longer_run --iterations 80 --joint-iterations 200
# Optional stronger requirement: contrast ≥0.4 at every optimized amplitude:
.venv/bin/python scripts/staged_echo_descent.py --output outputs/my_high_contrast_run --band-contrast-floor 0.4</pre>
<p>Bounds: length 8–70 µs, cutoff 10⁻⁵–0.02, exponent 1–16, reversal 2–500 ns. Polynomial exponent is relaxed to a real number for descent; integer orders receive separate refinements. The fixed pulse is normalized to its actual complex-drive peak at each parameter evaluation. A nominal 50 MHz pulse is scaled from 5 to 60 MHz for the robustness ensemble.</p>
<p><a href="results.json">Final metrics and failed constraints</a> · <a href="selected.json">Selected parameters</a> · <a href="training_summary.json">Algorithm and evaluation counts</a> · <a href="spectra.npz">Population arrays</a> · <a href="configuration.json">Training settings</a></p>
<details><summary>Parameters and convergence diagnostics</summary><pre>{html.escape(json.dumps({n: dict(pulse=c["pulse"], validation=c["validation"], constraints=c.get("constraints")) for n, c in cases.items()}, indent=2))}</pre></details>"""
    (args.output / "report.html").write_text(page)
    input_names = ["results.json", "spectra.npz", "curriculum_history.json"]
    output_names = ["report.html", "maps.png", "comparison.png", "stages.png"]
    if audit_data:
        input_names.append("dense_audit.json")
    if selected_name:
        input_names.append("dense_audit.npz")
        output_names.extend(("preferred_maps.png", "waveforms.png"))
    save_json(
        args.output / "render_provenance.json",
        dict(
            generator_sha256=sha(Path(__file__)),
            inputs={n: sha(args.output / n) for n in input_names},
            outputs={n: sha(args.output / n) for n in output_names},
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--stage",
        choices=("all", "train", "validate", "audit", "render"),
        default="all",
    )
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--joint-iterations", type=int, default=100)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--drag-beta", type=float, default=0.0)
    parser.add_argument("--band-contrast-floor", type=float, default=0.0)
    parser.add_argument("--extra-amplitudes", nargs="*", type=float, default=[])
    parser.add_argument("--seed-json", type=Path)
    args = parser.parse_args()
    if args.iterations < 1 or args.joint_iterations < 1 or args.workers < 1:
        parser.error("Iteration counts and workers must be positive")
    if not 0 <= args.band_contrast_floor < 1:
        parser.error("Contrast floor must be between zero (inclusive) and one")
    if not all(5 <= a <= 60 for a in args.extra_amplitudes):
        parser.error("Extra training amplitudes must lie in 5--60 MHz")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage not in ("all", "train"):
        run = json.loads((args.output / "run.json").read_text())
        args.drag_beta, args.band_contrast_floor = (
            run["drag_beta"],
            run["band_contrast_floor"],
        )
    if args.stage in ("all", "train"):
        save_json(
            args.output / "run.json",
            {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        )
        train(args)
    if args.stage in ("all", "validate"):
        validate(args)
    if args.stage in ("all", "validate", "audit"):
        audit(args)
    if args.stage in ("all", "validate", "audit", "render"):
        render(args)


if __name__ == "__main__":
    main()
