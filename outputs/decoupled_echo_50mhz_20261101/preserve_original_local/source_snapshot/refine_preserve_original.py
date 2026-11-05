import json,sys,hashlib,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
from scipy.optimize import minimize
ROOT=Path('/Users/asafsolonnikov/Developer/amplitude-robust-spectroscopy')
sys.path.insert(0,str(ROOT))
from scripts.optimize_decoupled_50mhz_echo import AMPS,DET,BOUNDS,encode,make_pulse,constraints,clean
from scripts.optimize_50mhz_echo import metrics,peak_scale
from echospec.simulation.polynomial_echo import simulate_polynomial_echo
from echospec.simulation.decoupled_echo import DecoupledEchoPulse
base=ROOT/'outputs/decoupled_echo_50mhz_20261101'
out=base/'preserve_original_local'
(out/'evaluations').mkdir(parents=True,exist_ok=True)
config=json.loads((base/'preserve_original/configuration.json').read_text())
config.update(campaign='preserve_original_local',method='bounded Nelder-Mead from best preserve_original differential evolution candidate',maxfev=200,constraints='C50>0.4; preserve baseline contrast and linewidth at all sampled5--60MHz amplitudes',contrast_floor_scope='50 MHz only')
config['parent_de_configuration_sha256']=hashlib.sha256((base/'preserve_original/configuration.json').read_bytes()).hexdigest()
for key in ('population','iterations','seed'):
 config.pop(key,None)
config['source_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in config['source_sha256']}
config['local_driver_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
snapshot=out/'source_snapshot'
snapshot.mkdir(exist_ok=True)
for name in config['source_sha256']:
 (snapshot/Path(name).name).write_bytes((ROOT/name).read_bytes())
(snapshot/'refine_preserve_original.py').write_bytes(Path(__file__).read_bytes())
(out/'configuration.json').write_text(json.dumps(config,indent=2)+'\n')
config_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
baseline=config['baseline']
records=[]

def objective(x):
 pulse=make_pulse(x)
 norm=peak_scale(pulse)
 pop,diag=simulate_polynomial_echo(pulse,AMPS/norm,DET,dt_ns=.5)
 rows=metrics(pop,AMPS,DET)
 quality=constraints(rows,baseline,True)
 loss=.7*quality['width50_khz']+.3*quality['max_width_khz']+quality['violation']
 r=dict(parameters=x.tolist(),pulse=asdict(pulse),peak_scale=norm,rows=rows,quality=quality,loss=loss,diagnostics=diag,configuration_sha256=config_hash)
 key=hashlib.sha256(np.asarray(x,dtype=float).tobytes()).hexdigest()
 (out/'evaluations'/f'{key}.json').write_text(json.dumps(clean(r),indent=2,allow_nan=False)+'\n')
 records.append(r)
 if len(records)%10==0 or quality['feasible']:
  best=min(records,key=lambda r:r['loss'])
  print(json.dumps(dict(n=len(records),loss=best['loss'],quality=best['quality'])),flush=True)
 return loss

start=json.loads((base/'preserve_original/best.json').read_text())
x0=encode(DecoupledEchoPulse(**start['pulse']))
lower,upper=np.array(BOUNDS).T
x0=np.clip(x0,lower,upper)
steps=np.array([.75,.012,-.012,-.05,.025,-.5,-.03,.1,.05])
simplex=np.vstack([x0]+[np.clip(x0+np.eye(9)[i]*steps[i],lower,upper) for i in range(9)])
result=minimize(objective,x0,method='Nelder-Mead',bounds=BOUNDS,options=dict(maxfev=200,initial_simplex=simplex,xatol=.001,fatol=.01,adaptive=True))
records.sort(key=lambda r:r['loss'])
feasible=sorted([r for r in records if r['quality']['feasible']],key=lambda r:.7*r['quality']['width50_khz']+.3*r['quality']['max_width_khz'])
(out/'best.json').write_text(json.dumps(clean(records[0]),indent=2,allow_nan=False)+'\n')
(out/'candidates.json').write_text(json.dumps(clean(dict(candidates=feasible[:8] if feasible else records[:8],feasible_count=len(feasible),evaluations=len(records),optimizer_loss=float(result.fun),termination=str(result.message))),indent=2,allow_nan=False)+'\n')
print('DONE',len(records),'feasible',len(feasible),'best',records[0]['quality'],flush=True)
