"""Targeted sampled robustness verification; no pulse modification or search."""
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
source=OUT/'cleaner_repaired.json'
record=json.loads(source.read_text())
pulse=FlexibleEchoPulse(**record['pulse'])
original=PolynomialEchoPulse()
amplitudes=np.unique(np.round(np.r_[np.arange(5.,5.5001,.05),np.arange(14.5,15.5001,.05)],10))
detuning=np.linspace(-.5,.5,401)

def run(shape):
 normalization=peak_scale(shape)
 populations,diagnostics=simulate_polynomial_echo(shape,amplitudes/normalization,detuning,dt_ns=.25)
 return metrics(populations,amplitudes,detuning),diagnostics,normalization

with ThreadPoolExecutor(max_workers=2) as pool:
 candidate_future=pool.submit(run,pulse)
 original_future=pool.submit(run,original)
 rows,diagnostics,normalization=candidate_future.result()
 baseline,baseline_diagnostics,baseline_normalization=original_future.result()
margins=[dict(rabi_mhz=r['rabi_mhz'],linewidth_improvement_khz=(b['fwhm_hz']-r['fwhm_hz'])/1000,
              contrast_improvement=r['contrast']-b['contrast'])
         for r,b in zip(rows,baseline,strict=True)]
min_width=min(margins,key=lambda r:r['linewidth_improvement_khz'])
min_contrast=min(margins,key=lambda r:r['contrast_improvement'])
quality=dict(all_rows_accepted=all(r['accepted'] for r in rows+baseline),
             true_no_width_or_contrast_regression=all(
                 r['linewidth_improvement_khz']>=0 and r['contrast_improvement']>=0 for r in margins),
             minimum_linewidth_improvement=min_width,minimum_contrast_improvement=min_contrast)
result=dict(pulse=asdict(pulse),pulse_sha256=record['pulse_sha256'],baseline_pulse=asdict(original),
            amplitudes_mhz=amplitudes.tolist(),detuning_mhz=detuning.tolist(),dt_ns=.25,
            t1_us=51.24,t2_us=7.31,peak_scale=normalization,baseline_peak_scale=baseline_normalization,
            rows=rows,baseline=baseline,margins=margins,quality=quality,
            diagnostics=diagnostics,baseline_diagnostics=baseline_diagnostics,native_calls=2,
            source_record=str(source.relative_to(ROOT)),source_record_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            model_sources=record['provenance']['model_sources'],
            note='Targeted0.05MHz sampled verification near the smallest margins from the prior0.5MHz sweep. This does not prove robustness at every point of the continuous interval; no optimization or extra probes were performed.')
(OUT/'cleaner_local_amplitude_refinement.json').write_text(json.dumps(clean(result),indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(amplitude_count=len(amplitudes),quality=quality)),flush=True)
