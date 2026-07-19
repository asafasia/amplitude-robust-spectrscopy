"""Build the clean, raw-data-backed echo-Lorentzian cutoff explorer notebook."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "paper" / "15_echo_lorentzian_cutoff_explorer.ipynb"


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.12"},
}

nb["cells"] = [
    markdown(
        r"""
# Echo-Lorentzian cutoff explorer

This notebook explores **raw OPX1000 2D measurements** and recalculates the
FWHM for every amplitude at the selected cutoff. It does not read the
precomputed `*_fit_results.csv` FWHM columns.

The newer aggregate folders in `cutoff_amp_fwhm_map/` and
`echo_lorentzian_cutoff_sweep/` retain CSV summaries and rendered images, but
not the numerical 2D state arrays needed for a fresh fit. Consequently, the
interactive catalog below includes only matched raw-backed campaigns under
`calibrations/.../echo_lorentzian/`.
"""
    ),
    code(
        r"""
from collections import defaultdict
from functools import lru_cache
import json
import os
from pathlib import Path

import ipywidgets as widgets
from IPython.display import Markdown, clear_output, display
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

DATA_DIR = Path(os.environ.get(
    "OPX1000_DATA_DIR",
    "/Users/asafsolonnikov/Developer/data_opx1000",
)).expanduser()
RAW_ROOT = DATA_DIR / "calibrations"

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "font.size": 10,
})
"""
    ),
    markdown(
        r"""
## Discover matched raw campaigns

A campaign contains runs with identical sweep settings and different cutoffs.
When a cutoff was repeated, the latest run is used. Campaigns are separated by
measurement date so that profile changes are not silently mixed.
"""
    ),
    code(
        r"""
SETTING_FIELDS = (
    "lorentzian_length_in_ns",
    "lorentzian_peak_amplitude",
    "frequency_span_in_mhz",
    "frequency_step_in_mhz",
    "num_shots",
    "min_amp_factor",
    "max_amp_factor",
    "amp_factor_step",
)


def discover_campaigns(raw_root=RAW_ROOT):
    grouped = defaultdict(list)
    for parameter_file in sorted(raw_root.glob("*/echo_lorentzian/*/parameters.json")):
        run_dir = parameter_file.parent
        if not (run_dir / "sweep.npz").exists() or not (run_dir / "results.npz").exists():
            continue
        parameters = json.loads(parameter_file.read_text())
        if not parameters.get("echo", False):
            continue
        date = run_dir.parents[1].name
        key = (date,) + tuple(parameters.get(name) for name in SETTING_FIELDS)
        grouped[key].append((float(parameters["cutoff"]), run_dir, parameters))

    campaigns = []
    for key, entries in grouped.items():
        latest_by_cutoff = {}
        for cutoff, run_dir, parameters in entries:
            latest_by_cutoff[cutoff] = (run_dir, parameters)
        if len(latest_by_cutoff) < 2:
            continue
        parameters = next(iter(latest_by_cutoff.values()))[1]
        cutoffs = sorted(latest_by_cutoff)
        label = (
            f"{key[0]} | {len(cutoffs)} cutoffs | "
            f"{parameters['lorentzian_length_in_ns']/1000:g} us | "
            f"peak {parameters['lorentzian_peak_amplitude']:g} | "
            f"span {parameters['frequency_span_in_mhz']:g} MHz | "
            f"{parameters['num_shots']} shots"
        )
        campaigns.append({
            "label": label,
            "date": key[0],
            "cutoffs": cutoffs,
            "runs": {c: latest_by_cutoff[c][0] for c in cutoffs},
            "parameters": parameters,
        })
    return sorted(campaigns, key=lambda item: (item["date"], len(item["cutoffs"])), reverse=True)


CAMPAIGNS = discover_campaigns()
if not CAMPAIGNS:
    raise FileNotFoundError(f"No matched raw echo-Lorentzian campaigns below {RAW_ROOT}")

print(f"Found {len(CAMPAIGNS)} matched raw-backed campaigns; newest is {CAMPAIGNS[0]['date']}.")
"""
    ),
    markdown(
        r"""
## FWHM calculation used here

For each measured amplitude trace, the notebook smooths only along frequency,
determines whether the central feature is a peak or dip, estimates the edge
baseline, and linearly interpolates the two nearest half-depth crossings.
The diagnostic panel exposes all of these quantities for the selected trace.
"""
    ),
    code(
        r"""
def nearest_crossing(x, values, start, direction):
    i = start
    while 0 <= i + direction < len(x):
        j = i + direction
        if values[i] == 0:
            return float(x[i])
        if values[i] * values[j] <= 0:
            if values[j] == values[i]:
                return float(x[i])
            fraction = -values[i] / (values[j] - values[i])
            return float(x[i] + fraction * (x[j] - x[i]))
        i = j
    return np.nan


def half_depth_fwhm(detuning_mhz, state, smooth_sigma_points=2):
    # Calculate one trace's central-feature FWHM directly from measured state.
    x = np.asarray(detuning_mhz, dtype=float)
    y = np.asarray(state, dtype=float)
    y_smooth = gaussian_filter1d(y, smooth_sigma_points) if smooth_sigma_points else y.copy()

    edge_count = max(2, len(x) // 10)
    baseline = float(np.median(np.r_[y_smooth[:edge_count], y_smooth[-edge_count:]]))
    center_index = int(np.argmin(np.abs(x)))
    is_dip = y_smooth[center_index] < baseline
    feature_index = int(np.argmin(y_smooth) if is_dip else np.argmax(y_smooth))
    feature_value = float(y_smooth[feature_index])
    signal = abs(feature_value - baseline)
    half_level = 0.5 * (feature_value + baseline)
    crossings = y_smooth - half_level
    left = nearest_crossing(x, crossings, feature_index, -1)
    right = nearest_crossing(x, crossings, feature_index, +1)
    fwhm = right - left if np.isfinite(left) and np.isfinite(right) else np.nan

    return {
        "fwhm_mhz": float(fwhm),
        "signal": float(signal),
        "baseline": baseline,
        "half_level": half_level,
        "left_mhz": left,
        "right_mhz": right,
        "feature_index": feature_index,
        "is_dip": is_dip,
        "smoothed": y_smooth,
    }


def load_raw_run(run_dir):
    run_dir = Path(run_dir)
    parameters = json.loads((run_dir / "parameters.json").read_text())
    with np.load(run_dir / "sweep.npz") as sweep, np.load(run_dir / "results.npz") as results:
        qubit = str(np.ravel(sweep["qubit"])[0])
        detuning_mhz = np.asarray(sweep["detuning"], dtype=float) / 1e6
        amp_prefactor = np.asarray(sweep["amp_prefactor"], dtype=float)
        state = np.squeeze(np.asarray(results["state"], dtype=float))
    if state.shape != (len(detuning_mhz), len(amp_prefactor)):
        if state.T.shape == (len(detuning_mhz), len(amp_prefactor)):
            state = state.T
        else:
            raise ValueError(f"Unexpected state shape {state.shape} in {run_dir}")

    qubits = json.loads((run_dir / "profile" / "qubits.json").read_text())["qubits"]
    pulses = json.loads((run_dir / "profile" / "pulses.json").read_text())["pulses"]
    x180_name = qubits[qubit]["operations"]["x180"]
    x180 = pulses[qubit][x180_name]
    pi_rabi_mhz = 1000.0 / (2.0 * float(x180["length_ns"]))
    drive_scale = float(parameters["lorentzian_peak_amplitude"]) / float(x180["amplitude"])
    rabi_mhz = amp_prefactor * drive_scale * pi_rabi_mhz

    transmon = qubits[qubit]["transmon"]
    t2_name = "t2_echo_ns" if transmon.get("t2_echo_ns") else "t2_ramsey_ns"
    t2_us = float(transmon[t2_name]) / 1000.0
    t2_limit_mhz = 1.0 / (np.pi * t2_us)
    return {
        "run_dir": run_dir,
        "parameters": parameters,
        "qubit": qubit,
        "detuning_mhz": detuning_mhz,
        "amp_prefactor": amp_prefactor,
        "rabi_mhz": rabi_mhz,
        "state": state,
        "t2_name": t2_name,
        "t2_us": t2_us,
        "t2_limit_mhz": t2_limit_mhz,
    }


@lru_cache(maxsize=64)
def analyze_run(run_dir_string, smooth_sigma_points=2):
    raw = load_raw_run(run_dir_string)
    details = [
        half_depth_fwhm(raw["detuning_mhz"], raw["state"][:, index], smooth_sigma_points)
        for index in range(raw["state"].shape[1])
    ]
    fwhm = np.array([item["fwhm_mhz"] for item in details])
    signal = np.array([item["signal"] for item in details])
    frequency_bin = float(np.median(np.diff(raw["detuning_mhz"])))
    valid = (
        np.isfinite(fwhm)
        & (signal >= 0.05)
        & (fwhm >= 2 * frequency_bin)
        & (fwhm <= 0.5 * np.ptp(raw["detuning_mhz"]))
    )
    resolution = np.divide(raw["t2_limit_mhz"], fwhm, out=np.full_like(fwhm, np.nan), where=fwhm > 0)
    raw.update({
        "details": details,
        "fwhm_mhz": fwhm,
        "signal": signal,
        "valid": valid,
        "resolution": resolution,
        "signal_resolution": signal * resolution,
        "smooth_sigma_points": smooth_sigma_points,
    })
    return raw
"""
    ),
    markdown(
        r"""
## Immediately visible example

Move the **cutoff slider** to switch raw 2D sweeps. The amplitude slider selects
the trace whose half-depth construction is shown. Smoothing is in measured
frequency bins; changing it recalculates every FWHM in the notebook.

Color normalization for resolution is logarithmic with `vmin=0.1` and
`vmax=1`. Here, resolution is `[1/(pi T2)] / FWHM`, so values at or below one
mean `FWHM >=` the T2-limited linewidth. `signal x resolution` is also plotted
on a logarithmic axis.
"""
    ),
    code(
        r"""
def plot_dashboard(result, amplitude_index):
    x = result["detuning_mhz"]
    rabi = result["rabi_mhz"]
    state = result["state"]
    detail = result["details"][amplitude_index]
    valid = result["valid"]
    resolution = result["resolution"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2), constrained_layout=True)
    ax_map, ax_trace, ax_fwhm, ax_metric = axes.ravel()

    image = ax_map.pcolormesh(rabi, x, state, shading="auto", cmap="viridis", vmin=0, vmax=1)
    ax_map.axvline(rabi[amplitude_index], color="white", lw=1.5, ls="--")
    fig.colorbar(image, ax=ax_map, label="measured excited-state probability")
    ax_map.set(
        title=f"Raw 2D sweep | cutoff = {result['parameters']['cutoff']:.4g}",
        xlabel="peak Rabi frequency (MHz)",
        ylabel="detuning (MHz)",
    )

    y = state[:, amplitude_index]
    ax_trace.plot(x, y, ".", ms=3.5, alpha=0.55, label="raw state")
    ax_trace.plot(x, detail["smoothed"], lw=2, label="frequency-smoothed")
    ax_trace.axhline(detail["baseline"], color="0.35", ls=":", label="edge baseline")
    ax_trace.axhline(detail["half_level"], color="tab:orange", ls="--", label="half depth")
    if np.isfinite(detail["left_mhz"]) and np.isfinite(detail["right_mhz"]):
        ax_trace.axvspan(detail["left_mhz"], detail["right_mhz"], color="tab:orange", alpha=0.15)
        ax_trace.axvline(detail["left_mhz"], color="tab:orange", ls="--")
        ax_trace.axvline(detail["right_mhz"], color="tab:orange", ls="--")
    width_text = f"{detail['fwhm_mhz']:.4g} MHz" if np.isfinite(detail["fwhm_mhz"]) else "no two crossings"
    ax_trace.set(
        title=f"Auditable trace | Rabi = {rabi[amplitude_index]:.3g} MHz | FWHM = {width_text}",
        xlabel="detuning (MHz)",
        ylabel="state",
    )
    ax_trace.legend(fontsize=8, ncol=2)

    ax_fwhm.plot(rabi[valid], result["fwhm_mhz"][valid], color="0.3", lw=1, alpha=0.5)
    points = ax_fwhm.scatter(
        rabi[valid], result["fwhm_mhz"][valid], c=resolution[valid], s=22,
        cmap="viridis", norm=LogNorm(vmin=0.1, vmax=1), label="accepted crossings",
    )
    ax_fwhm.scatter(rabi[~valid], result["fwhm_mhz"][~valid], marker="x", s=16, color="0.7", label="screened")
    ax_fwhm.axhline(result["t2_limit_mhz"], color="black", ls="--", label="1/(pi T2)")
    ax_fwhm.set_yscale("log")
    ax_fwhm.set(
        title="FWHM recalculated from each raw trace",
        xlabel="peak Rabi frequency (MHz)",
        ylabel="FWHM (MHz, log scale)",
    )
    ax_fwhm.legend(fontsize=8)
    fig.colorbar(points, ax=ax_fwhm, label="resolution = [1/(pi T2)] / FWHM")

    ax_metric.plot(rabi[valid], resolution[valid], ".-", ms=4, label="resolution")
    ax_metric.plot(rabi[valid], result["signal_resolution"][valid], ".-", ms=4, label="signal x resolution")
    ax_metric.axhline(1, color="black", ls="--", lw=1, label="T2 limit")
    ax_metric.set_yscale("log")
    ax_metric.set_ylim(bottom=0.03)
    ax_metric.set(
        title="Resolution metrics",
        xlabel="peak Rabi frequency (MHz)",
        ylabel="metric (log scale)",
    )
    ax_metric.legend(fontsize=8)

    fig.suptitle(
        f"{result['qubit']} | T2 reference: {result['t2_us']:.3g} us ({result['t2_name']}) | "
        f"smoothing: {result['smooth_sigma_points']} bins",
        fontsize=12,
    )
    plt.show()


campaign_widget = widgets.Dropdown(
    options=[(campaign["label"], index) for index, campaign in enumerate(CAMPAIGNS)],
    value=0,
    description="Campaign:",
    layout=widgets.Layout(width="920px"),
    style={"description_width": "90px"},
)
cutoff_widget = widgets.SelectionSlider(
    description="Cutoff:", continuous_update=False,
    layout=widgets.Layout(width="920px"), style={"description_width": "90px"},
)
amplitude_widget = widgets.SelectionSlider(
    description="Rabi (MHz):", continuous_update=False,
    layout=widgets.Layout(width="920px"), style={"description_width": "90px"},
)
smoothing_widget = widgets.IntSlider(
    value=2, min=0, max=6, step=1, description="Smoothing:", continuous_update=False,
    layout=widgets.Layout(width="500px"), style={"description_width": "90px"},
)
controls = widgets.VBox([campaign_widget, cutoff_widget, amplitude_widget, smoothing_widget])
updating = False


def selected_campaign():
    return CAMPAIGNS[campaign_widget.value]


def update_cutoff_options():
    campaign = selected_campaign()
    cutoff_widget.options = [(f"{value:.4g}", value) for value in campaign["cutoffs"]]
    cutoff_widget.value = campaign["cutoffs"][len(campaign["cutoffs"]) // 2]


def update_amplitude_options():
    run_dir = selected_campaign()["runs"][float(cutoff_widget.value)]
    result = analyze_run(str(run_dir), int(smoothing_widget.value))
    options = [(f"{value:.4g}", index) for index, value in enumerate(result["rabi_mhz"])]
    amplitude_widget.options = options
    amplitude_widget.value = len(options) // 2


def render(change=None):
    global updating
    if updating:
        return
    clear_output(wait=True)
    display(controls)
    campaign = selected_campaign()
    run_dir = campaign["runs"][float(cutoff_widget.value)]
    result = analyze_run(str(run_dir), int(smoothing_widget.value))
    plot_dashboard(result, int(amplitude_widget.value))
    accepted = int(result["valid"].sum())
    display(Markdown(
        f"**Fresh calculation:** {accepted}/{len(result['valid'])} amplitude traces pass the visible "
        f"signal/width screen.  \\n**Raw source:** `{result['run_dir']}`"
    ))


def campaign_changed(change):
    global updating
    if change.get("name") != "value":
        return
    updating = True
    update_cutoff_options()
    update_amplitude_options()
    updating = False
    render()


def cutoff_changed(change):
    global updating
    if updating or change.get("name") != "value":
        return
    updating = True
    update_amplitude_options()
    updating = False
    render()


campaign_widget.observe(campaign_changed, names="value")
cutoff_widget.observe(cutoff_changed, names="value")
amplitude_widget.observe(render, names="value")
smoothing_widget.observe(render, names="value")

updating = True
update_cutoff_options()
update_amplitude_options()
updating = False
render()
"""
    ),
    markdown(
        r"""
## Raw-backed campaign catalog

This table is discovery output, not a hand-maintained list. It makes the
available measurement dates, settings, and cutoff coverage explicit.
"""
    ),
    code(
        r"""
pd.DataFrame([
    {
        "campaign": campaign["label"],
        "cutoffs": ", ".join(f"{value:.4g}" for value in campaign["cutoffs"]),
        "raw runs": len(campaign["runs"]),
    }
    for campaign in CAMPAIGNS
])
"""
    ),
]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUTPUT)
print(OUTPUT)
