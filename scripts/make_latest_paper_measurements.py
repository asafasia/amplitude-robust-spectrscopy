"""Import completed q1 maps and render the measured main-text Figures 2 and 3.

No instrument connection, fitting or simulation is performed. Run with
--refresh-source to validate/import raw acquisitions; otherwise use the portable
paper datasets once present. --source-data-dir (or OPX1000_DATA_DIR) may point to
the paper_q1 campaign data directory on another machine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from echospec.figures.coherence import (  # noqa: E402
    REFERENCE_FWHM_HZ,
    REFERENCE_T2_STAR_S,
)
from echospec.paper_data import save_paper_dataset  # noqa: E402

GENERATOR = "scripts/make_latest_paper_measurements.py"
FIG2_SESSION = "20261031_002217_547409_optimized_5000/figures2-3"
FIG3_SESSION = "20261031_191600_figure3_50ns_poly2_avg5000_amp300/maps"
F2 = ["A1_broad_no_echo", "A2_broad_echo", "A3_narrow_no_echo", "A4_narrow_echo"]
F3 = ["E1_no_drag_no_poly", "E2_drag_no_poly", "E3_drag_poly"]
COLORS = ["#287bb4", "#d97916", "#138568"]
LABELS = ["Echo", "+ DRAG", "+ polynomial"]
COLOR_LIMITS = (0.0, 0.8)
FIG3_DISPLAY_MAX_RABI_MHZ = 60.0
OUT = ROOT / "figures/paper"
DATA = ROOT / "paper/data"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_campaign(source_root, session_name, names, stem, averages, axis, grid):
    session = source_root / session_name
    amplitude_points, frequency_points = grid
    summary, plan = (
        read_json(session / "summary.json"),
        read_json(session / "plan.json"),
    )
    records = {r["name"]: r for r in summary["records"]}
    items = {r["name"]: r for r in plan["maps"]}
    arrays, selected = {}, {}
    for name in names:
        record, item = records[name], items[name]
        p, status = item["parameters"], record["status"]
        assert status["raw_data_saved"] and not status["interrupted"], name
        assert p["num_shots"] == averages and p["qubits"] == ["q1"]
        assert p["amp_factor_points"] == amplitude_points
        assert p["frequency_points"] == frequency_points
        assert p["lorentzian_length_in_ns"] == 20000 and p["cutoff"] == 0.005
        assert p["reset_type"] == "active" and not p["simulate"]
        processed = session / record["processed_npz"]
        raw = session / record["raw_directory"] / "results.npz"
        with np.load(processed) as data:
            local = {k: data[k].copy() for k in data.files}
        with np.load(raw) as data:
            np.testing.assert_array_equal(local["state"], data["state"][0].T)
        assert local["state"].shape == tuple(grid)
        assert np.isfinite(local["state"]).all()
        wavepath = session / "waveforms" / f"{name}.npz"
        with np.load(wavepath) as wave:
            peak = np.max(np.abs(wave["I"] + 1j * wave["Q"]))
        peak_rabi = (
            abs(local["amplitude_factor"])
            * peak
            * plan["nominal_rabi_hz_per_volt"]
            / 1e6
        )
        local["waveform_peak_rabi_mhz"] = peak_rabi
        max_rabi = 20 if names == F2 else 100
        np.testing.assert_allclose(
            local[axis], np.linspace(0, max_rabi, amplitude_points), atol=1e-8
        )
        if names == F3:
            np.testing.assert_allclose(local[axis], peak_rabi, rtol=1e-12)
            assert p["echo_transition_time_ns"] == 50
            assert p["drag_beta"] == (0 if name == F3[0] else 1.2557575757575759)
            assert p["edge_smoothing"] == ("polynomial" if name == F3[2] else "none")
            if name == F3[2]:
                assert p["edge_smoothing_power"] == 2
        prefix = name.lower()
        for key, value in local.items():
            arrays[f"{prefix}_{key}"] = value
        if "fits" in record:
            arrays[f"{prefix}_accepted"] = np.array(
                [row["accepted"] for row in record["fits"]]
            )
            for key in [
                "center_hz",
                "center_error_hz",
                "fwhm_hz",
                "contrast",
                "r_squared",
            ]:
                arrays[f"{prefix}_{key}"] = np.array(
                    [
                        row.get(key) if row.get(key) is not None else np.nan
                        for row in record["fits"]
                    ]
                )
        selected[name] = dict(
            parameters=p,
            source_processed=f"{session_name}/{Path(record['processed_npz']).as_posix()}",
            processed_sha256=sha(processed),
            source_raw=f"{session_name}/{Path(record['raw_directory']).as_posix()}/results.npz",
            raw_sha256=sha(raw),
            waveform_sha256=sha(wavepath),
            raw_state_exactly_matches=True,
            raw_saved=True,
            interrupted=False,
            color_limits=[float(local["state"].min()), float(local["state"].max())],
        )
    provenance = dict(
        generator=GENERATOR,
        manuscript_scope=f"Main-text Figure {2 if names == F2 else 3}",
        source_session=session_name,
        source_root_convention="Relative to q1 campaign data root",
        source_session_status=summary["status"],
        selection="Only individually completed maps with saved, uninterrupted raw data",
        source_plan_sha256=sha(session / "plan.json"),
        source_summary_sha256=sha(session / "summary.json"),
        frequency_reference_hz=plan["frequency_reference_hz"],
        rabi_axis=axis,
        nominal_rabi_hz_per_volt=plan["nominal_rabi_hz_per_volt"],
        reference_pi=plan["reference_pi"],
        maps=selected,
        grid=list(grid),
        averages=averages,
        spectroscopy_shots=len(names) * amplitude_points * frequency_points * averages,
        coherence={
            phase: read_json(session / f"coherence_{phase}.json")
            for phase in ["before", "after"]
            if (session / f"coherence_{phase}.json").exists()
        },
        calibrations={
            name: r["fit"]
            for name, r in records.items()
            if name.startswith(("before_", "after_")) and "fit" in r
        },
        processing="Raw populations unchanged; saved Gaussian fit values and quality masks retained",
    )
    save_paper_dataset(
        stem, category="experimental", arrays=arrays, provenance=provenance
    )


def load_export(stem, category="experimental"):
    with np.load(DATA / category / f"{stem}.npz") as data:
        arrays = {k: data[k].copy() for k in data.files}
    return arrays, read_json(DATA / category / f"{stem}.json")["provenance"]


def series(data, name, field):
    return data[f"{name.lower()}_{field}"]


def save(fig, stem, metadata):
    OUT.mkdir(exist_ok=True)
    for extension in ["pdf", "png", "svg"]:
        fig.savefig(OUT / f"{stem}.{extension}", dpi=220)
    plt.close(fig)
    (OUT / f"{stem}_provenance.json").write_text(
        json.dumps(dict(generator=GENERATOR, **metadata), indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def reference_simulations():
    stem = "02_latest_q1_reference_simulations"
    if not (DATA / "numerical" / f"{stem}.npz").exists():
        source = OUT / "02_central_spectroscopy_experiment_simulation_variant.npz"
        meta = read_json(source.with_name(source.stem + "_provenance.json"))
        with np.load(source) as old:
            arrays = {k: old[k].copy() for k in old.files if "_simulation_" in k}
        save_paper_dataset(
            stem,
            category="numerical",
            arrays=arrays,
            provenance=dict(
                generator=GENERATOR,
                source=str(source.relative_to(ROOT)),
                source_sha256=sha(source),
                source_provenance=meta,
                processing="Existing simulations preserved without recalculation; display cropped to measured domains",
            ),
        )
    return load_export(stem, "numerical")


def representative_ramsey(meta):
    """Summarize accepted Ramsey measurements without changing source values."""
    references = {
        phase: ref for phase, ref in meta["coherence"].items()
        if ref.get("accepted", True)
    }
    assert references, "No accepted Ramsey coherence reference"
    times = np.array([ref["t2_star_s"] for ref in references.values()])
    assert np.isfinite(times).all() and (times > 0).all()
    for ref in references.values():
        np.testing.assert_allclose(
            ref["fwhm_limit_hz"], 1 / (np.pi * ref["t2_star_s"])
        )
    tau = float(np.mean(times))
    return dict(
        method="Arithmetic mean of accepted Ramsey T2* measurements",
        t2_star_s=tau,
        fwhm_limit_hz=1 / (np.pi * tau),
        source_phases=list(references),
        source_t2_star_s=times.tolist(),
    )


def plot_fig2(data, meta, with_simulations):
    ramsey = representative_ramsey(meta)
    reference_half_width_mhz = REFERENCE_FWHM_HZ / 2e6
    t1_s = float(np.mean([
        fit["t1"] * 1e-9 for name, fit in meta["calibrations"].items()
        if name.endswith("_t1") and fit.get("success", False)
    ]))
    assert np.isfinite(t1_s) and t1_s > 0
    square_hwhm_slope = np.sqrt(t1_s / REFERENCE_T2_STAR_S)
    rows = 2 if with_simulations else 1
    fig = plt.figure(figsize=(7.1, 3.35 if rows == 2 else 2.0))
    blocks = fig.add_gridspec(
        1, 2, left=0.075, right=0.91, bottom=0.14, top=0.86, wspace=0.18
    )
    block_grids = [
        block.subgridspec(rows, 2, wspace=0.15, hspace=0.10) for block in blocks
    ]
    axes = np.array([
        [fig.add_subplot(block_grids[col // 2][row, col % 2]) for col in range(4)]
        for row in range(rows)
    ])
    simulation = reference_simulations()[0] if with_simulations else {}
    columns = [
        (F2[0], "broad", "root"),
        (F2[2], "narrow", "root"),
        (F2[1], "broad", "echo"),
        (F2[3], "narrow", "echo"),
    ]
    for start, title in [(0, "Root-Lorentzian"), (2, "Echo-root-Lorentzian")]:
        left = axes[0, start].get_position().x0
        right = axes[0, start + 1].get_position().x1
        fig.text((left + right) / 2, 0.95, title, ha="center", fontsize=8)
    row_arrays = []
    for row in range(rows):
        displayed = []
        for name, domain, protocol in columns:
            if row == 0:
                x = series(data, name, "detuning_hz") / 1e6
                y = series(data, name, "nominal_rabi_mhz")
                z = series(data, name, "state")
            else:
                base = f"{domain}_{protocol}_simulation_"
                x, y, z = (
                    simulation[base + k]
                    for k in ["detuning_mhz", "rabi_mhz", "population"]
                )
            span = 20 if domain == "broad" else 0.5
            mask = np.ix_((y >= 0) & (y <= 20), (x >= -span) & (x <= span))
            displayed.append((x, y, z, span, z[mask]))
        lo = min(a[4].min() for a in displayed)
        hi = max(a[4].max() for a in displayed)
        row_arrays.append([float(lo), float(hi)])
        for col, (x, y, z, span, _) in enumerate(displayed):
            ax = axes[row, col]
            mesh = ax.pcolormesh(
                x, y, z, shading="auto", cmap="magma",
                vmin=COLOR_LIMITS[0], vmax=COLOR_LIMITS[1], rasterized=True
            )
            ax.set(
                xlim=(-span, span),
                ylim=(0, 20),
                xticks=[-15, 0, 15] if span > 1 else [-0.4, 0, 0.4],
                yticks=[0, 10, 20],
            )
            ax.tick_params(labelleft=col == 0, labelbottom=row == rows - 1)
            if row == rows - 1:
                ax.set_xlabel(r"$\Delta/(2\pi)$ (MHz)", labelpad=2)
            if col == 0:
                ax.set_ylabel(
                    ("Experiment" if row == 0 else "Reference simulation")
                    + "\n"
                    + r"$\Omega_0/(2\pi)$ (MHz)",
                    labelpad=2,
                )
            if row == 0:
                ax.set_title(
                    "Broad detuning" if span > 1 else "Narrow detuning",
                    fontsize=6.5,
                    pad=5,
                )
            if span > 1:
                ax.axvline(-0.5, color="white", lw=0.5, ls=":")
                ax.axvline(0.5, color="white", lw=0.5, ls=":")
                reference_rabi_mhz = np.array([0, min(20, span / square_hwhm_slope)])
                for sign in [-1, 1]:
                    ax.plot(
                        sign * square_hwhm_slope * reference_rabi_mhz,
                        reference_rabi_mhz, color="red", lw=0.4, alpha=0.5,
                    )
            else:
                for sign in [-1, 1]:
                    ax.axvline(
                        sign * reference_half_width_mhz,
                        color="white", lw=0.5, ls=(0, (2, 6)),
                    )
            # Letter each pulse-type block: experiment above reference simulation.
            letter = (
                "abcdefgh"[(col // 2) * 4 + row * 2 + col % 2]
                if rows == 2
                else "abcd"[col]
            )
            ax.text(
                0.05,
                0.93,
                f"({letter})",
                color="white",
                va="top",
                transform=ax.transAxes,
                bbox=dict(facecolor="black", alpha=0.25, edgecolor="none", pad=1),
            )
    bottom = axes[-1, 0].get_position().y0
    top = axes[0, 0].get_position().y1
    cax = fig.add_axes([0.925, bottom, 0.012, top - bottom])
    bar = fig.colorbar(mesh, cax=cax, ticks=np.linspace(*COLOR_LIMITS, 5))
    bar.ax.set_title(r"$P_e$", fontsize=7, pad=4)
    save(
        fig,
        "02_latest_q1_spectroscopy",
        dict(
            experimental_dataset="paper/data/experimental/02_latest_q1_spectroscopy.npz",
            existing_simulations_retained=with_simulations,
            representative_ramsey=ramsey,
            square_pulse_broadening_reference=dict(
                panels=["a", "c", "e", "g"] if with_simulations else ["a", "c"],
                definition="Strong-drive Bloch HWHM asymptotes: detuning = +/- f_R * sqrt(T1/T2_star)",
                t1_s=t1_s,
                t1_source="Mean accepted Figure 2 T1 fits; saved fit t1 is in ns",
                t2_star_s=REFERENCE_T2_STAR_S,
                hwhm_slope_mhz_per_mhz=float(square_hwhm_slope),
                origin=[0, 0],
                color="red",
                linewidth_pt=0.4,
                alpha=0.5,
            ),
            narrow_panel_reference=dict(
                t2_star_s=REFERENCE_T2_STAR_S,
                source="User-specified common display reference",
                definition="Lines at +/- 1/(2*pi*T2_star), enclosing the FWHM",
                vertical_lines_mhz=[-reference_half_width_mhz, reference_half_width_mhz],
                rows="experiment and reference simulation",
            ),
            column_order=[name for name, _, _ in columns],
            layout="Root-Lorentzian left, echo-root-Lorentzian right; broad then narrow in each block",
            spacing=dict(between_blocks_wspace=0.18, within_blocks_wspace=0.15),
            row_data_extrema=row_arrays,
            color_limits=list(COLOR_LIMITS),
            colorbar_count=1,
            display_limits=dict(
                detuning_mhz=[[-20, 20], [-0.5, 0.5]], nominal_rabi_mhz=[0, 20]
            ),
        ),
    )


def plot_fig3(data, meta):
    fig = plt.figure(figsize=(7.1, 4.5))
    grid = fig.add_gridspec(
        2, 3, left=0.075, right=0.89, bottom=0.09, top=0.94,
        wspace=0.34, hspace=0.33, height_ratios=[0.96, 1],
    )
    maps = [fig.add_subplot(grid[0, col]) for col in range(3)]
    center, contrast, fwhm = [fig.add_subplot(grid[1, col]) for col in range(3)]
    ramsey = representative_ramsey(meta)
    ramsey_fwhm_hz = REFERENCE_FWHM_HZ
    for i, (name, ax) in enumerate(zip(F3, maps, strict=True)):
        x = series(data, name, "detuning_hz") / 1000
        rabi, state = series(data, name, "peak_rabi_mhz"), series(data, name, "state")
        mesh = ax.pcolormesh(
            x,
            rabi,
            state,
            shading="auto",
            cmap="magma",
            vmin=COLOR_LIMITS[0],
            vmax=COLOR_LIMITS[1],
            rasterized=True,
        )
        ax.set(
            xlim=(-500, 500), ylim=(0, FIG3_DISPLAY_MAX_RABI_MHZ),
            xticks=[-500, 0, 500], yticks=[0, 20, 40, 60],
            xlabel=r"$\Delta/(2\pi)$ (kHz)",
        )
        ax.tick_params(labelleft=i == 0)
        if i == 0:
            ax.set_ylabel(r"$\Omega_{\mathrm{pk}}/(2\pi)$ (MHz)", labelpad=2)
        for sign in [-1, 1]:
            ax.axvline(
                sign * ramsey_fwhm_hz / 2000,
                color="white", lw=0.5, ls=(0, (2, 6)),
            )
        ax.text(
            0.06,
            0.94,
            f"({chr(97 + i)})",
            transform=ax.transAxes,
            color="white",
            va="top",
        )
        ax.set_title(
            ["No DRAG, no polynomial", "DRAG only", "Polynomial 2 + DRAG"][i],
            color=COLORS[i], pad=4,
        )
    top = maps[0].get_position()
    cax = fig.add_axes([0.91, top.y0, 0.012, top.height])
    fig.colorbar(
        mesh, cax=cax, ticks=np.linspace(*COLOR_LIMITS, 5)
    )
    cax.set_title(r"$P_e$", pad=4)
    masks = [
        series(data, n, "accepted")
        & (series(data, n, "peak_rabi_mhz") > 0)
        & (series(data, n, "peak_rabi_mhz") <= FIG3_DISPLAY_MAX_RABI_MHZ)
        for n in F3
    ]
    common = np.logical_and.reduce(masks)
    assert common.any(), "No commonly accepted fits in the displayed amplitude range"
    statistics = []
    center_display_exclusions = {F3[2]: [2]}
    for i, (name, mask) in enumerate(zip(F3, masks, strict=True)):
        rabi = series(data, name, "peak_rabi_mhz")
        center_mask = mask.copy()
        center_mask[center_display_exclusions.get(name, [])] = False
        fields = ["center_hz", "fwhm_hz", "contrast"]
        for ax, field, scale in zip(
            [center, fwhm, contrast], fields, [1e-3, 1 / ramsey_fwhm_hz, 1], strict=True
        ):
            ax.plot(
                rabi,
                np.where(
                    center_mask if field == "center_hz" else mask,
                    series(data, name, field) * scale, np.nan,
                ),
                color=COLORS[i],
                lw=0.65,
                marker=".",
                ms=1.6,
                label=LABELS[i],
            )
        center.errorbar(
            rabi[center_mask],
            series(data, name, "center_hz")[center_mask] / 1000,
            yerr=series(data, name, "center_error_hz")[center_mask] / 1000,
            fmt="none",
            ecolor=COLORS[i],
            elinewidth=0.35,
            alpha=0.35,
        )
        centers = series(data, name, "center_hz")[common]
        statistics.append(
            dict(
                name=name,
                accepted=int(mask.sum()),
                displayed_amplitude_points=int(
                    np.count_nonzero((rabi >= 0) & (rabi <= FIG3_DISPLAY_MAX_RABI_MHZ))
                ),
                matched_count=int(common.sum()),
                median_fwhm_khz=float(
                    np.median(series(data, name, "fwhm_hz")[common]) / 1000
                ),
                median_fwhm_normalized=float(
                    np.median(series(data, name, "fwhm_hz")[common]) / ramsey_fwhm_hz
                ),
                median_contrast=float(
                    np.median(series(data, name, "contrast")[common])
                ),
                center_mean_khz=float(np.mean(centers) / 1000),
                center_rms_about_mean_khz=float(np.std(centers) / 1000),
            )
        )
    center.set(ylim=(-100, 100), yticks=[-100, -50, 0, 50, 100])
    center.set_ylabel(r"$\Delta_{\mathrm{res}}/(2\pi)$ (kHz)", labelpad=2)
    center.axhline(0, color="0.6", lw=0.4, zorder=0)
    ramsey_half_width_khz = ramsey_fwhm_hz / 2000
    for sign in [-1, 1]:
        ramsey_center_line = center.axhline(
            sign * ramsey_half_width_khz, color="0.35", ls="--", lw=0.6, zorder=1
        )
    center.add_artist(center.legend(
        loc="lower left", frameon=False, fontsize=5.7, handlelength=1
    ))
    center.legend(
        [ramsey_center_line],
        [r"$\pm\Gamma_{T_2^*}/2$"],
        loc="upper left", frameon=False, fontsize=5.4, handlelength=1.3,
    )
    fwhm.set(
        ylabel=r"$\mathrm{FWHM}/\Gamma_{T_2^*}$",
        yticks=[0, 1, 2, 4, 6, 8, 10, 12], ylim=(0, 12),
    )
    contrast.set(ylabel="Contrast", ylim=(0, 0.5), yticks=[0, 0.2, 0.4, 0.5])
    fwhm.axhline(1, color="0.4", ls="--", lw=0.55)
    for letter, ax in zip("def", [center, contrast, fwhm], strict=True):
        ax.set(
            xlim=(0, FIG3_DISPLAY_MAX_RABI_MHZ),
            xticks=[0, 20, 40, 60],
            xlabel=r"$\Omega_{\mathrm{pk}}/(2\pi)$ (MHz)",
        )
        ax.grid(alpha=0.2)
        ax.text(0.02, 1.07, f"({letter})", transform=ax.transAxes)
    save(
        fig,
        "03_latest_q1_pulse_improvement",
        dict(
            experimental_dataset="paper/data/experimental/03_latest_q1_pulse_improvement.npz",
            color_limits=list(COLOR_LIMITS),
            colormap="magma",
            colorbar_count=1,
            panel_order=[*F3, "resonance_position", "contrast", "fwhm"],
            layout=dict(
                figure_size_inches=[7.1, 4.5],
                row_height_ratios=[0.96, 1],
                row_spacing=0.33,
            ),
            display_amplitude_limits_mhz=[0, FIG3_DISPLAY_MAX_RABI_MHZ],
            acquisition_amplitude_limits_mhz=[0, 100],
            statistics=statistics,
            common_indices=np.flatnonzero(common).tolist(),
            fit_method="Saved Gaussian dip plus linear baseline; saved acceptance masks retained; statistics and curves restricted to 0 < peak Rabi <= 60 MHz",
            center_rms_definition="Unweighted standard deviation about the arithmetic mean on common accepted rows",
            center_errorbars="One covariance standard error",
            center_display_exclusions=dict(
                indices=center_display_exclusions,
                reason="User-requested omission of the first high green point in panel d",
                scope="Center curve and error bar only; source arrays and statistics unchanged",
            ),
            pulse_legend_panels=["d"],
            center_display_ylim_khz=[-100, 100],
            contrast_display_ylim=[0, 0.5],
            fwhm_axes=dict(
                left="FWHM / Gamma_T2_star",
                right=None,
                normalization="Common paper reference: Gamma_T2_star = 1 / (pi * T2_star)",
                t2_star_s=REFERENCE_T2_STAR_S,
                normalization_hz=ramsey_fwhm_hz,
                left_ylim=[0, 12],
            ),
            center_reference=dict(
                calibration="common_paper_reference",
                t2_star_s=REFERENCE_T2_STAR_S,
                definition="Lines at +/- 1/(2*pi*Ramsey_T2_star), enclosing the FWHM",
                horizontal_lines_khz=[-ramsey_half_width_khz, ramsey_half_width_khz],
                legend_panel="d",
                legend_location="upper left",
                inline_annotation=False,
            ),
            map_reference_vertical_lines_khz=[-ramsey_half_width_khz, ramsey_half_width_khz],
            representative_ramsey=ramsey,
            coherence=meta["coherence"],
        ),
    )
    return statistics


def write_parameters(f2, f3, stats):
    values = {
        "PaperTTwoStar": f"{REFERENCE_T2_STAR_S * 1e6:.1f}",
        "PaperCoherenceFwhm": f"{REFERENCE_FWHM_HZ / 1000:.1f}",
    }
    for prefix, meta in [
        ("LatestTwo", f2),
        ("LatestThree", f3),
    ]:
        c, cal = representative_ramsey(meta), meta["calibrations"]
        for suffix, kind, field, scale in [
            ("TOne", "_t1", "t1", 0.001),
            ("TTwoEcho", "_echo", "T2_echo", 1e6),
        ]:
            measurements = [
                fit[field] * scale for name, fit in cal.items()
                if name.endswith(kind) and fit.get("success", False)
            ]
            assert measurements
            values[prefix + suffix] = f"{np.mean(measurements):.1f}"
        values[prefix + "TTwoStar"] = f"{c['t2_star_s'] * 1e6:.1f}"
        values[prefix + "Limit"] = f"{c['fwhm_limit_hz'] / 1000:.1f}"
    for label, s in zip(["Echo", "Drag", "Poly"], stats, strict=True):
        values[f"Latest{label}Accepted"] = str(s["accepted"])
        values[f"Latest{label}Width"] = f"{round(s['median_fwhm_khz'], -1):.0f}"
        values[f"Latest{label}WidthNormalized"] = f"{s['median_fwhm_normalized']:.1f}"
        values[f"Latest{label}Contrast"] = f"{s['median_contrast']:.2f}"
        values[f"Latest{label}CenterRms"] = f"{s['center_rms_about_mean_khz']:.0f}"
    values["LatestCommonCount"] = str(stats[0]["matched_count"])
    values["LatestThreeDisplayedAmplitudePoints"] = str(stats[0]["displayed_amplitude_points"])
    values["LatestThreeDisplayMaxRabi"] = f"{FIG3_DISPLAY_MAX_RABI_MHZ:g}"
    parameters = f3["maps"][F3[0]]["parameters"]
    values["LatestThreeAmplitudePoints"] = str(parameters["amp_factor_points"])
    values["LatestThreeFrequencyPoints"] = str(parameters["frequency_points"])
    values["LatestThreeAverages"] = str(parameters["num_shots"])
    values["LatestThreeDriveGHz"] = f"{f3['frequency_reference_hz'] / 1e9:.9f}"
    values["LatestPolyLimitRatio"] = (
        f"{stats[2]['median_fwhm_khz'] * 1000 / REFERENCE_FWHM_HZ:.0f}"
    )
    (ROOT / "paper/latest_results_parameters.tex").write_text(
        "% Generated from completed experimental datasets by "
        + GENERATOR
        + "\n"
        + "\n".join(
            "\\newcommand{\\" + name + "}{" + value + "}"
            for name, value in values.items()
        )
        + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-data-dir",
        type=Path,
        default=Path(
            os.environ.get("OPX1000_DATA_DIR", ROOT / "experiments/paper_q1/data")
        ),
    )
    parser.add_argument("--refresh-source", action="store_true")
    parser.add_argument("--experimental-only", action="store_true")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--figure2-only", action="store_true")
    selection.add_argument("--figure3-only", action="store_true")
    args = parser.parse_args()
    for session, names, stem, averages, axis, grid in [
        (FIG2_SESSION, F2, "02_latest_q1_spectroscopy", 5000, "nominal_rabi_mhz", (200, 200)),
        (FIG3_SESSION, F3, "03_latest_q1_pulse_improvement", 5000, "peak_rabi_mhz", (300, 200)),
    ]:
        if args.figure3_only and names == F2:
            continue
        if args.figure2_only and names == F3:
            continue
        if args.refresh_source or not (DATA / "experimental" / f"{stem}.npz").exists():
            export_campaign(
                args.source_data_dir, session, names, stem, averages, axis, grid
            )
    f2, m2 = load_export("02_latest_q1_spectroscopy")
    if not args.figure2_only:
        f3, m3 = load_export("03_latest_q1_pulse_improvement")
        if m3["source_session"] != FIG3_SESSION:
            raise ValueError("Figure 3 cache is from an earlier run; use --refresh-source")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "axes.linewidth": 0.55,
            "xtick.major.pad": 2,
            "ytick.major.pad": 2,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": None,
        }
    )
    if not args.figure3_only:
        plot_fig2(f2, m2, not args.experimental_only)
    if args.figure2_only:
        return
    stats = plot_fig3(f3, m3)
    write_parameters(m2, m3, stats)
    print(
        json.dumps(
            dict(
                figure2_shots=m2["spectroscopy_shots"],
                figure3_shots=m3["spectroscopy_shots"],
                figure3_statistics=stats,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
