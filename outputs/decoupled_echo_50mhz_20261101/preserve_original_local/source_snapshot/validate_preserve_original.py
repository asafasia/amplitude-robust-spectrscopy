import json,sys,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
ROOT=Path('/Users/asafsolonnikov/Developer/amplitude-robust-spectroscopy')
sys.path.insert(0,str(ROOT))
from scripts.optimize_decoupled_50mhz_echo import DET,constraints,clean
from scripts.optimize_50mhz_echo import metrics,peak_scale
from echospec.simulation.polynomial_echo import PolynomialEchoPulse,simulate_polynomial_echo
from echospec.simulation.decoupled_echo import DecoupledEchoPulse
out=ROOT/'outputs/decoupled_echo_50mhz_20261101/preserve_original_local'
record=json.loads((out/'candidates.json').read_text())['candidates'][0]
pulse=DecoupledEchoPulse(**record['pulse'])
amps=np.arange(5.,61.)
def simulate(shape):
 norm=peak_scale(shape)
 pop,diag=simulate_polynomial_echo(shape,amps/norm,DET,dt_ns=.5)
 return metrics(pop,amps,DET),diag,norm
with ThreadPoolExecutor(max_workers=2) as pool:
 candidate=pool.submit(simulate,pulse)
 original=pool.submit(simulate,PolynomialEchoPulse())
 rows,diagnostics,norm=candidate.result()
 baseline,base_diag,base_norm=original.result()
regressions=[dict(rabi_mhz=r['rabi_mhz'],contrast_difference=r['contrast']-b['contrast'],width_difference_khz=(r['fwhm_hz']-b['fwhm_hz'])/1000) for r,b in zip(rows,baseline)]
result=dict(pulse=record['pulse'],peak_scale=norm,amplitudes_mhz=amps.tolist(),detuning_mhz=DET.tolist(),dt_ns=.5,t1_us=51.24,t2_us=7.31,rows=rows,baseline=baseline,regressions=regressions,quality=constraints(rows,baseline,True),strict_quality=constraints(rows,baseline,False),amplitudes_below_contrast_04=[r['rabi_mhz'] for r in rows if r['contrast']<.4],diagnostics=diagnostics,baseline_diagnostics=base_diag,note='Held-out amplitude pilot using same native RK4 solver and detuning grid, not independent solver validation; contrast0.4 floor applies only at50MHz.',candidate_file='candidates.json',candidate_file_sha256=hashlib.sha256((out/'candidates.json').read_bytes()).hexdigest(),driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(out/'heldout_validation_pilot.json').write_text(json.dumps(clean(result),indent=2,allow_nan=False)+'\n')
(out/'source_snapshot'/'validate_preserve_original.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(dict(quality=result['quality'],strict_quality=result['strict_quality'],below04=result['amplitudes_below_contrast_04'],row50=next(r for r in rows if r['rabi_mhz']==50.))))
