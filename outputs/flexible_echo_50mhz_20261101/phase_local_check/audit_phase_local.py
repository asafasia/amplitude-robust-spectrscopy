"""Two-call verification of one fixed phase pulse on refined amplitude/frequency grids."""
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from echospec.simulation.flexible_echo import FlexibleEchoPulse
from echospec.simulation.polynomial_echo import PolynomialEchoPulse,simulate_polynomial_echo
from scripts.optimize_50mhz_echo import metrics,peak_scale
from scripts.optimize_decoupled_50mhz_echo import clean

OUT=ROOT/'outputs/flexible_echo_50mhz_20261101'
source=OUT/'repaired_flexible_phase.json'
record=json.loads(source.read_text())
pulse=FlexibleEchoPulse(**record['pulse'])
original=PolynomialEchoPulse()
amplitudes=np.array([7.25,7.5,7.75,16.75,17.,17.25,50.])
detuning=np.linspace(-.5,.5,801)

def run(shape):
 normalization=peak_scale(shape)
 populations,diagnostics=simulate_polynomial_echo(shape,amplitudes/normalization,detuning,dt_ns=.25)
 return metrics(populations,amplitudes,detuning),diagnostics,normalization,populations

with ThreadPoolExecutor(max_workers=2) as pool:
 candidate_future=pool.submit(run,pulse)
 original_future=pool.submit(run,original)
 rows,diagnostics,normalization,populations=candidate_future.result()
 baseline,baseline_diagnostics,baseline_normalization,baseline_populations=original_future.result()
margins=[dict(rabi_mhz=r['rabi_mhz'],linewidth_improvement_khz=(b['fwhm_hz']-r['fwhm_hz'])/1000,
              contrast_improvement=r['contrast']-b['contrast'])
         for r,b in zip(rows,baseline,strict=True)]
min_width=min(margins,key=lambda r:r['linewidth_improvement_khz'])
min_contrast=min(margins,key=lambda r:r['contrast_improvement'])
quality=dict(all_rows_accepted=all(r['accepted'] for r in rows+baseline),
             true_no_width_or_contrast_regression=all(
                 r['linewidth_improvement_khz']>=0 and r['contrast_improvement']>=0 for r in margins),
             minimum_linewidth_improvement=min_width,minimum_contrast_improvement=min_contrast,
             width_regression_amplitudes_mhz=[r['rabi_mhz'] for r in margins if r['linewidth_improvement_khz']<0],
             contrast_regression_amplitudes_mhz=[r['rabi_mhz'] for r in margins if r['contrast_improvement']<0],
             contrast_at50=rows[-1]['contrast'],width50_khz=rows[-1]['fwhm_hz']/1000,
             max_center_khz=max(abs(r['center_hz']) for r in rows)/1000,
             max_satellite=max(r['satellite_prominence'] for r in rows))
result=dict(pulse=asdict(pulse),pulse_sha256=hashlib.sha256(json.dumps(asdict(pulse),sort_keys=True).encode()).hexdigest(),
            baseline_pulse=asdict(original),amplitudes_mhz=amplitudes.tolist(),detuning_mhz=detuning.tolist(),
            dt_ns=.25,t1_us=51.24,t2_us=7.31,peak_scale=normalization,baseline_peak_scale=baseline_normalization,
            rows=rows,baseline=baseline,margins=margins,quality=quality,
            diagnostics=diagnostics,baseline_diagnostics=baseline_diagnostics,native_calls=2,
            source_record=str(source.relative_to(ROOT)),source_record_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),model_sources=record['model_sources'],
            note='Fixed-pulse sampled verification with finer amplitudes near the smallest prior margins and1.25kHz detuning steps. No optimization or extra probes; this is not a continuous-interval proof.')
(OUT/'phase_local_refinement.json').write_text(json.dumps(clean(result),indent=2,allow_nan=False)+'\n')
np.savez_compressed(OUT/'phase_local_check/phase_local_refinement.npz',rabi_mhz=amplitudes,detuning_mhz=detuning,
                    populations=populations,baseline=baseline_populations)
print(json.dumps(dict(pulse_sha256=result['pulse_sha256'],quality=quality,margins=margins)),flush=True)
