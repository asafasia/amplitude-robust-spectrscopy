"""Publish the already propagated repair map under the report's exact identity."""
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from echospec.simulation.flexible_echo import FlexibleEchoPulse
from scripts.report_flexible_50mhz_echo import Cache,model_sources
BASE=ROOT/'outputs/flexible_echo_50mhz_20261101'
record_path=BASE/'cleaner_repaired.json'
record=json.loads(record_path.read_text())
pulse=FlexibleEchoPulse(**record['pulse'])
with np.load(BASE/'cleaner_repair/selected_dense_map.npz') as saved:
 populations=saved['populations'].copy()
 amplitudes=saved['rabi_mhz']/record['peak_scale']
 detuning=saved['detuning_mhz'].copy()
identity=dict(pulse=dict(type=type(pulse).__name__,fields=asdict(pulse)),
              amplitudes=amplitudes.tolist(),detuning=detuning.tolist(),dt_ns=.25,
              t1_us=51.24,t2_us=7.31,sources=model_sources(pulse))
path=Cache(BASE).locate('native',identity)
if not path.exists():
 np.savez_compressed(path,populations=populations,diagnostics=json.dumps(record['diagnostics']),identity=json.dumps(identity))
with np.load(path) as check:
 np.testing.assert_array_equal(check['populations'],populations)
 assert json.loads(str(check['identity']))==json.loads(json.dumps(identity))
record['provenance']['native_report_cache']=str(path.relative_to(ROOT))
record['provenance']['native_report_cache_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
record['provenance']['model_sources']=identity['sources']
record['provenance']['cache_publisher_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
record_path.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
print(path)
