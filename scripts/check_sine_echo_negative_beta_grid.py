"""Halve detuning spacing for selected beta=-0.5 traces after the main sweep."""

# ruff: noqa: E402
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import json

import numpy as np

from echospec.analysis.dip_metrics import dip_metrics
from scripts.make_sine_echo_beta_comparison import simulate

out = ROOT / "outputs/sine_echo_negative_beta_comparison"
d = dict(np.load(out / "beta-0.5_narrow.npz"))
fits = [dip_metrics(d["detuning_mhz"], y) for y in d["populations"][1]]
centers = np.array([f["center_hz"] for f in fits])
indices = sorted(set([int(np.nanargmax(abs(centers))), 46, 99]))
x = np.linspace(-0.5, 0.5, 401)
result = simulate(-0.5, x, d["rabi_mhz"][indices])
rows = []
for j, i in enumerate(indices):
    fine = dip_metrics(x, result[1, j])
    rows.append(
        dict(
            rabi_mhz=float(d["rabi_mhz"][i]),
            coarse=fits[i],
            fine=fine,
            center_change_hz=float(fine["center_hz"] - fits[i]["center_hz"]),
            width_change_hz=float(fine["fwhm_hz"] - fits[i]["fwhm_hz"]),
        )
    )
np.savez_compressed(
    out / "beta-0.5_detuning_refinement.npz",
    detuning_mhz=x,
    rabi_mhz=d["rabi_mhz"][indices],
    populations=result,
)
(out / "detuning_refinement.json").write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
