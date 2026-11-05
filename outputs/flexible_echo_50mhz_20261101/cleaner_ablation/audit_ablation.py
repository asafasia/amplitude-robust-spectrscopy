"""Fixed ablations of one repaired pulse; no variant is reoptimized."""
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from echospec.simulation.flexible_echo import FlexibleEchoPulse
from echospec.simulation.polynomial_echo import simulate_polynomial_echo
from scripts.optimize_50mhz_echo import metrics,peak_scale
from scripts.optimize_decoupled_50mhz_echo import clean

BASE=ROOT/'outputs/flexible_echo_50mhz_20261101'
OUT=BASE/'cleaner_ablation'
source=BASE/'cleaner_repaired.json'
repaired=json.loads(source.read_text())
pulse=FlexibleEchoPulse(**repaired['pulse'])
amplitudes=np.array([5.,15.,50.,60.])
detuning=np.linspace(-.5,.5,401)
with np.load(BASE/'cleaner_repair/selected_dense_map.npz') as saved:
 indices=[int(np.flatnonzero(saved['rabi_mhz']==amp)[0]) for amp in amplitudes]
 original=saved['populations'][:,indices,:].copy()
 np.testing.assert_array_equal(saved['detuning_mhz'],detuning)

variants=[('amplitude_terms_zero',replace(pulse,amp_coefficients=(0.,)*4)),
          ('extra_phase_terms_zero',replace(pulse,phase_coefficients=(0.,)*5)),
          ('transition_16ns',replace(pulse,transition_us=.016))]

def run(item):
 name,shape=item
 normalization=peak_scale(shape)
 populations,diagnostics=simulate_polynomial_echo(shape,amplitudes/normalization,detuning,dt_ns=.25)
 return name,dict(pulse=asdict(shape),peak_scale=normalization,
                  rows=metrics(populations,amplitudes,detuning),diagnostics=diagnostics),populations

records={'as_is':dict(pulse=asdict(pulse),peak_scale=repaired['peak_scale'],
                      rows=metrics(original,amplitudes,detuning),reused_dense_map=True)}
arrays={'rabi_mhz':amplitudes,'detuning_mhz':detuning,'as_is':original}
with ThreadPoolExecutor(max_workers=2) as pool:
 for name,record,populations in pool.map(run,variants):
  records[name]=record
  arrays[name]=populations
for name,record in records.items():
 record['changes_vs_repaired']=[dict(rabi_mhz=r['rabi_mhz'],
     fwhm_change_khz=(r['fwhm_hz']-b['fwhm_hz'])/1000,
     contrast_change=r['contrast']-b['contrast'])
     for r,b in zip(record['rows'],records['as_is']['rows'],strict=True)]
result=dict(cases=records,amplitudes_mhz=amplitudes.tolist(),detuning_mhz=detuning.tolist(),
            dt_ns=.25,t1_us=51.24,t2_us=7.31,native_calls=3,
            actual_peak_normalization_per_variant=True,
            reference_pulse_sha256=repaired['pulse_sha256'],
            source_record=str(source.relative_to(ROOT)),
            source_record_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            model_sources=repaired['provenance']['model_sources'],
            note='Fixed ablations around one selected pulse, independently renormalized to the same actual peaks. Variants are not reoptimized; differences demonstrate coupled sensitivity, not necessity or global optimality of any component.')
(OUT/'ablation.json').write_text(json.dumps(clean(result),indent=2,allow_nan=False)+'\n')
np.savez_compressed(OUT/'ablation.npz',**arrays)
for name,record in records.items():
 print(json.dumps(dict(name=name,peak_scale=record['peak_scale'],
    rows=[dict(amplitude=r['rabi_mhz'],fwhm_khz=r['fwhm_hz']/1000,contrast=r['contrast'],accepted=r['accepted']) for r in record['rows']]),allow_nan=False),flush=True)
