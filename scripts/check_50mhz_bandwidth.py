"""Illustrative Gaussian I/Q bandwidth check; not a measured hardware model."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echospec.simulation.polynomial_echo import simulate_polynomial_echo  # noqa: E402
from scripts.optimize_50mhz_echo import (  # noqa: E402
    OUT,
    make_pulse,
    metrics,
    peak_scale,
)


class SampledPulse:
    def __init__(self, base, time, drive):
        self.base, self.time, self.samples = base, time, drive

    def __getattr__(self, name):
        return getattr(self.base, name)

    def drive(self, time):
        return np.interp(time, self.time, self.samples, left=0, right=0)


def main():
    records = [json.loads(p.read_text()) for p in
               (OUT / "peak/evaluations").glob("phase_*.json")]
    eligible = [r for r in records if all(
        m["accepted"] and m["contrast"] >= .4 and m["satellite_prominence"] <= .01
        and abs(m["center_hz"]) <= 2000 for m in r["rows"])]
    selected = min(eligible, key=lambda r: max(m["fwhm_hz"] for m in r["rows"]))
    pulse = make_pulse(selected["parameters"], "phase")
    dt = .000125
    time = np.arange(-pulse.duration_us / 2, pulse.duration_us / 2 + dt / 2, dt)
    raw = pulse.drive(time) / peak_scale(pulse)
    x = np.linspace(-.5, .5, 401)
    amps = np.array([40., 50., 60.])
    rows = []
    for sigma_ns in (0., 1., 2., 4.):
        wave = raw if sigma_ns == 0 else gaussian_filter1d(raw, sigma_ns / 1000 / dt,
                                                          mode="constant", cval=0)
        scale = abs(wave).max()
        shaped = SampledPulse(pulse, time, wave / scale)
        pop, diag = simulate_polynomial_echo(shaped, amps, x, dt_ns=.125)
        row = dict(gaussian_sigma_ns=sigma_ns,
                   amplitude_3db_bandwidth_mhz=None if sigma_ns == 0 else
                   np.sqrt(np.log(2)) / (2 * np.pi * sigma_ns / 1000),
                   postfilter_peak_retention=float(scale),
                   renormalized_to_actual_peak=True,
                   metrics=metrics(pop, amps, x), diagnostics=diag)
        rows.append(row)
        print(json.dumps(row), flush=True)
    (OUT / "bandwidth_check.json").write_text(json.dumps(dict(
        model="Gaussian convolution of complete complex waveform before peak renormalization",
        caveat="Illustrative transfer functions, not a calibration of the experiment",
        pulse=selected["pulse"], rows=rows), indent=2))


if __name__ == "__main__":
    main()
