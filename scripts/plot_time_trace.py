"""
Plot time traces from amplitude sweep spectroscopy results.

Usage examples:

- Plot Z vs time at amplitude 20 MHz and detuning +5 MHz:
  python scripts/plot_time_trace.py /path/to/experiment_dir --amp-mhz 20 --det-mhz 5 --obs z

- Plot all observables vs time selecting nearest values in raw units (rad/s):
  python scripts/plot_time_trace.py /path/to/experiment_dir --amp 1.0e8 --det -3.0e6 --obs all

- Save figure instead of showing it:
  python scripts/plot_time_trace.py /path/to/experiment_dir --amp-mhz 30 --det-mhz 0 --output trace.png

Notes:
- The script expects a NetCDF file named 'results.nc' inside the provided directory.
- The DataArray variable name is 'expectation_values' and has dims: (amplitude, detuning, observable, time).
"""

from __future__ import annotations
from echospec.simulation.utils import z_to_populations
from echospec.utils.units import Units as u

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Ensure project root is on sys.path when running from scripts/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot time trace from amplitude-sweep results")
    p.add_argument(
        "results_path",
        type=str,
        help="Path to experiment directory containing results.nc or the results.nc file itself",
    )
    p.add_argument(
        "--file",
        type=str,
        default="results.nc",
        help="Override results filename if providing a directory (default: results.nc)",
    )
    # Selection in MHz (common convenient units)
    p.add_argument("--amp-mhz", type=float,
                   help="Drive amplitude in MHz for selection")
    p.add_argument("--det-mhz", type=float,
                   help="Detuning in MHz for selection")
    # Selection in raw rad/s
    p.add_argument("--amp", type=float,
                   help="Drive amplitude in rad/s for selection")
    p.add_argument("--det", type=float, help="Detuning in rad/s for selection")
    p.add_argument(
        "--obs",
        type=str,
        default="z",
        choices=["x", "y", "z", "all"],
        help="Observable to plot (default: z). Use 'all' to plot x,y,z",
    )
    p.add_argument(
        "--population",
        action="store_true",
        help="Convert Z to population (only applies when obs=z)",
    )
    p.add_argument(
        "--output",
        type=str,
        help="Output PNG path to save figure. If omitted, shows the plot.",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List available amplitudes and detunings and exit",
    )
    return p.parse_args(argv)


def load_dataarray(path: Path) -> xr.DataArray:
    ds = xr.open_dataset(path)
    # variable name from saved DataArray
    if "expectation_values" in ds:
        da = ds["expectation_values"]
    else:
        # fallback: first variable
        var = list(ds.data_vars)[0]
        da = ds[var]
    return da


def select_point(
    da: xr.DataArray,
    amp_mhz: Optional[float],
    det_mhz: Optional[float],
    amp: Optional[float],
    det: Optional[float],
) -> xr.DataArray:
    """Select nearest point by amplitude and detuning."""
    # Build target values in raw units (rad/s)
    target_amp = None
    target_det = None
    if amp_mhz is not None:
        target_amp = amp_mhz * u.pi2 * u.MHz
    if det_mhz is not None:
        target_det = det_mhz * u.pi2 * u.MHz
    if amp is not None:
        target_amp = amp
    if det is not None:
        target_det = det

    # If not provided, choose center
    if target_amp is None:
        amps = da.coords["amplitude"].values
        target_amp = float(amps[len(amps) // 2])
    if target_det is None:
        dets = da.coords["detuning"].values
        target_det = float(dets[len(dets) // 2])

    # Nearest selection using xarray
    sel = da.sel(amplitude=target_amp, detuning=target_det, method="nearest")
    return sel


def plot_time_trace(sel: xr.DataArray, obs: str, population: bool, output: Optional[Path]) -> None:
    time_us = sel.coords["time"].values / u.us

    fig, ax = plt.subplots(figsize=(8, 5))

    if obs == "all":
        # Expectation values indexed by 'observable'
        x = sel.sel(observable="x").values
        y = sel.sel(observable="y").values
        z = sel.sel(observable="z").values
        ax.plot(time_us, x, label="⟨X⟩")
        ax.plot(time_us, y, label="⟨Y⟩")
        ax.plot(time_us, z, label="⟨Z⟩")
    elif obs == "z" and population:
        z = sel.sel(observable="z")
        pop_result = z_to_populations(z.values)
        populations = pop_result[1] if isinstance(
            pop_result, (tuple, list)) else pop_result
        ax.plot(time_us, populations, label="Population")
    else:
        vals = sel.sel(observable=obs).values
        label_map = {"x": "⟨X⟩", "y": "⟨Y⟩", "z": "⟨Z⟩"}
        ax.plot(time_us, vals, label=label_map.get(obs, obs))

    amp = float(sel.coords["amplitude"].values)
    det = float(sel.coords["detuning"].values)
    ax.set_title(
        f"Time trace at amplitude={amp / u.pi2 / u.MHz:.3f} MHz, detuning={det / u.pi2 / u.MHz:.3f} MHz"
    )
    ax.set_xlabel("Time [µs]")
    ax.set_ylabel("Expectation value")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    # Resolve results path
    rp = Path(args.results_path)
    results_file = rp if rp.is_file() else rp / args.file

    if not results_file.exists():
        print(f"Error: results file not found: {results_file}")
        return 1

    da = load_dataarray(results_file)

    if args.list:
        amps = da.coords["amplitude"].values / u.pi2 / u.MHz
        dets = da.coords["detuning"].values / u.pi2 / u.MHz
        print("Available amplitudes [MHz] (first 10 shown):", np.round(
            amps, 6)[:10])
        print("Available detunings [MHz] (first 10 shown):", np.round(
            dets, 6)[:10])
        return 0

    sel = select_point(da, args.amp_mhz, args.det_mhz, args.amp, args.det)

    output = Path(args.output) if args.output else None
    plot_time_trace(sel, args.obs, args.population, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
