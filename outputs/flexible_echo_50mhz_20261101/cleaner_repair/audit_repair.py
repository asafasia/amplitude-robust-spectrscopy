"""Bounded cleaner-pulse repair: six stated variants, at most two scale probes."""
import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from echospec.simulation.flexible_echo import FlexibleEchoPulse
from echospec.simulation.polynomial_echo import simulate_polynomial_echo
from scripts.optimize_50mhz_echo import metrics, peak_scale
from scripts.optimize_decoupled_50mhz_echo import clean
from scripts.optimize_flexible_50mhz_echo import assess, encode

BASE = ROOT / 'outputs/flexible_echo_50mhz_20261101'
OUT = BASE / 'cleaner_repair'
DET = np.linspace(-.5, .5, 401)
AMPS = np.array([5., 15., 50.])


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, record):
    path.write_text(json.dumps(clean(record), indent=2, allow_nan=False) + '\n')


def compare(rows, baseline):
    return [dict(rabi_mhz=r['rabi_mhz'],
                 width_change_khz=(r['fwhm_hz']-b['fwhm_hz'])/1000,
                 contrast_change=r['contrast']-b['contrast'])
            for r, b in zip(rows, baseline, strict=True)]


def run_variant(label, pulse, baseline, phase_delta=0., scale_delta=0.):
    norm = peak_scale(pulse)
    pop, diagnostics = simulate_polynomial_echo(pulse, AMPS/norm, DET, dt_ns=.25)
    rows = metrics(pop, AMPS, DET)
    changes = compare(rows, baseline)
    passes = (all(r['accepted'] for r in rows)
              and all(changes[i]['width_change_khz'] <= -.05
                      and changes[i]['contrast_change'] >= 0 for i in (0, 1))
              and rows[2]['contrast'] >= .419)
    return dict(label=label, pulse=asdict(pulse), peak_scale=norm, rows=rows,
                changes=changes, pilot_margin_pass=passes, diagnostics=diagnostics,
                phase_delta=phase_delta, scale_delta_us=scale_delta)


def pilot():
    with np.load(OUT/'reference_maps.npz') as ref:
        indices = [int(np.flatnonzero(ref['rabi_mhz'] == amp)[0]) for amp in AMPS]
        baseline = metrics(ref['baseline'][:, indices, :], AMPS, DET)
        previous_rows = metrics(ref['cleaner_smooth'][:, indices, :], AMPS, DET)
    previous = json.loads((BASE/'cleaner_preselection.json').read_text())['candidates'][0]
    best = json.loads((BASE/'cleaner_shape/best.json').read_text())
    base_pulse = FlexibleEchoPulse(**best['pulse'])
    changes = compare(previous_rows, baseline)
    records = [dict(label='preselection_cached', pulse=previous['pulse'],
                    peak_scale=previous['peak_scale'], rows=previous_rows,
                    changes=changes, pilot_margin_pass=False, reused_reference=True)]
    variants = [('final_best', base_pulse, 0., 0.)]
    variants += [(f'phase_{delta:+.3f}', replace(base_pulse, phase_rad=base_pulse.phase_rad+delta),
                  delta, 0.) for delta in (-.002, .002, -.005, .005)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_variant, name, pulse, baseline, dp, ds)
                   for name, pulse, dp, ds in variants]
        for future in futures:
            record = future.result()
            records.append(record)
            print(json.dumps({key: record[key] for key in ('label', 'pilot_margin_pass', 'changes')}), flush=True)
    if not any(r['pilot_margin_pass'] for r in records):
        # Two fixed scale perturbations only; no optimizer or adaptive loop.
        variants = [(f'scale_{delta:+.4f}',
                     replace(base_pulse, phase_scale_us=base_pulse.phase_scale_us+delta),
                     0., delta) for delta in (-.0002, .0002)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_variant, name, pulse, baseline, dp, ds)
                       for name, pulse, dp, ds in variants]
            for future in futures:
                record = future.result()
                records.append(record)
                print(json.dumps({key: record[key] for key in ('label', 'pilot_margin_pass', 'changes')}), flush=True)
    passing = [r for r in records if r['pilot_margin_pass']]
    if passing:
        selected = min(passing, key=lambda r: (abs(r['phase_delta']) + abs(r['scale_delta_us'])/.01,
                                              r['rows'][2]['fwhm_hz']))
    else:
        selected = min(records, key=lambda r: max(z['width_change_khz'] for z in r['changes'][:2]))
    save(OUT/'pilot.json', dict(records=records, selected=selected, baseline=baseline,
                              amplitudes_mhz=AMPS.tolist(), detuning_mhz=DET.tolist(), dt_ns=.25,
                              native_probe_count=len(records)-1, total_variants=len(records),
                              script_sha256=digest(Path(__file__)),
                              reference_provenance=json.loads((OUT/'reference_provenance.json').read_text())))
    save(OUT/'selected_probe.json', selected)
    print('SELECTED', json.dumps(selected), flush=True)


def dense():
    selected = json.loads((OUT/'selected_probe.json').read_text())
    pulse = FlexibleEchoPulse(**selected['pulse'])
    with np.load(OUT/'reference_maps.npz') as ref:
        amplitudes = np.array(ref['rabi_mhz'])
        baseline = metrics(ref['baseline'], amplitudes, DET)
    norm = peak_scale(pulse)
    pop, diagnostics = simulate_polynomial_echo(pulse, amplitudes/norm, DET, dt_ns=.25)
    rows = metrics(pop, amplitudes, DET)
    changes = compare(rows, baseline)
    quality, inequalities = assess(rows, baseline, contrast_target=.419, satellite_target=.001)
    quality['true_no_width_or_contrast_regression'] = all(
        z['width_change_khz'] <= 0 and z['contrast_change'] >= 0 for z in changes)
    record = dict(name='cleaner_repaired', label='Cleaner smooth echo, repaired margins',
                  pulse_type='flexible', pulse=asdict(pulse), parameters=encode(asdict(pulse)).tolist(),
                  pulse_sha256=hashlib.sha256(json.dumps(asdict(pulse), sort_keys=True).encode()).hexdigest(),
                  peak_scale=norm, rows=rows, baseline=baseline, quality=quality,
                  inequalities=inequalities.tolist(), loss=quality['loss'], diagnostics=diagnostics,
                  regressions=changes, selection=selected, amplitudes_mhz=amplitudes.tolist(),
                  detuning_mhz=DET.tolist(), dt_ns=.25, t1_us=51.24, t2_us=7.31,
                  provenance=dict(script_sha256=digest(Path(__file__)),
                                  pilot_sha256=digest(OUT/'pilot.json'),
                                  reference_sha256=digest(OUT/'reference_maps.npz'),
                                  reference_provenance=json.loads((OUT/'reference_provenance.json').read_text())))
    save(BASE/'cleaner_repaired.json', record)
    np.savez_compressed(OUT/'selected_dense_map.npz', rabi_mhz=amplitudes,
                        detuning_mhz=DET, populations=pop)
    print('DENSE', json.dumps(dict(label=selected['label'], pulse_sha256=record['pulse_sha256'],
                                 quality=quality, worst_width=max(changes,key=lambda r:r['width_change_khz']),
                                 worst_contrast=min(changes,key=lambda r:r['contrast_change']))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['pilot', 'dense'])
    args = parser.parse_args()
    pilot() if args.stage == 'pilot' else dense()
