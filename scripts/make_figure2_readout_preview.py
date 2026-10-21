"""Provisional matrix-corrected Figure 2; never overwrite manuscript assets.

Uses the matrix for the qubit identified by each measurement's sweep. Original
IQ shots and the matching
threshold-based fidelity matrices are unavailable locally, so this is a
sensitivity preview, not a validated readout correction for publication.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from make_main_central_spectroscopy_experiment import DATA_ROOT, RUNS, load_measurement

from echospec.analysis.readout_correction import correct_binary_readout
from echospec.figures import FigureVariant, apply_figure_style

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "figures/paper/readout_preview"
SOURCE = ROOT / "figures/paper/02_central_spectroscopy_experiment_simulation_variant"
STEM = "02_readout_matrix_preview"
DOMAINS = ("broad", "narrow")
PROTOCOLS = (("root", "Root-Lorentzian"), ("echo", "Echo-root-Lorentzian"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    original = json.loads(Path(f"{SOURCE}_provenance.json").read_text())
    provenance = {
        "generator": "scripts/make_figure2_readout_preview.py",
        "status": "PROVISIONAL: snapshot matrix, classifier match unverified",
        "included_in_manuscript": False,
        "primary_color_limits": [0, 0.6],
        "diagnostic_color_limits": [0, 0.8],
        "source_arrays_sha256": sha256(SOURCE.with_suffix(".npz")),
        "source_figure_provenance": original,
        "matrix_convention": "rows prepared g,e; columns assigned g,e",
        "correction": "p_e = (q_e - M[0,1]) / (M[1,1] - M[0,1])",
        "uncertainty": "sqrt(q_e*(1-q_e)/N)/abs(M[1,1]-M[0,1]); fixed matrix only",
        "snr": "An affine correction scales signal contrast and noise equally; statistical SNR is unchanged for a fixed matrix. Calibration uncertainty is not available.",
        "limitations": [
            "The sweep metadata identifies q6. The older figure loader docstring calls this q1; that description is stale. Matrices are selected by the recorded sweep qubit, not by that docstring.",
            "Original matching IQ shots were not found in the local paper data or sibling data repository.",
            "Available acquisition code uses a threshold classifier; available IQ analysis saves a nearest-center confusion matrix separately from its threshold fidelity matrix. Exact historical classifier matching is unverified.",
            "Saved centers imply a different midpoint from the acquisition threshold. Do not publish as a validated correction without the original threshold fidelity calibration.",
            "Calibration may contain state-preparation errors; drift and f-state assignment are not resolved by this two-state matrix.",
            "Negative estimates are retained in NPZ; the displayed color scale saturates below 0 and above 0.6.",
            "Narrow-run metadata explicitly identifies results.npz as unmitigated. Referenced results_mitigated.npz files are not in the local export.",
        ],
        "runs": {},
    }
    for domain in DOMAINS:
        for protocol, _ in PROTOCOLS:
            key = f"{domain}_{protocol}_experiment"
            run = DATA_ROOT / RUNS[(domain, protocol)]
            m = load_measurement(RUNS[(domain, protocol)])
            np.testing.assert_array_equal(arrays[f"{key}_population"], m.excited)
            profile = json.loads((run / "profile/qubits.json").read_text())
            readout = profile["qubits"][m.qubit]["readout"]
            matrix = np.asarray(readout["confusion_matrix"])
            raw = m.excited
            corrected, stderr = correct_binary_readout(raw, matrix, m.effective_shots)
            arrays[f"{key}_raw_population"] = raw
            arrays[f"{key}_population"] = corrected
            arrays[f"{key}_conditional_stderr"] = stderr
            arrays[f"{key}_matrix"] = matrix
            contrast = matrix[1, 1] - matrix[0, 1]
            reconstructed = matrix[0, 1] + contrast * corrected
            np.testing.assert_allclose(reconstructed, raw, atol=1e-15, rtol=0)
            centers = np.asarray(readout["gef_centers"])
            provenance["runs"][f"{domain}_{protocol}"] = {
                "source": str(run.relative_to(ROOT)),
                "qubit": m.qubit,
                "results_sha256": sha256(run / "results.npz"),
                "profile_sha256": sha256(run / "profile/qubits.json"),
                "measurement_timestamp": json.loads((run / "metadata.json").read_text())["timestamp"],
                "shots": m.effective_shots,
                "matrix": matrix.tolist(),
                "contrast_gain": float(1 / contrast),
                "mean_assignment_fidelity": float(np.trace(matrix) / 2),
                "acquisition_threshold": readout["threshold"],
                "saved_center_midpoint_I": float(centers[:, 0].mean()),
                "raw_range": [float(raw.min()), float(raw.max())],
                "corrected_range": [float(corrected.min()), float(corrected.max())],
                "fraction_below_zero": float(np.mean(corrected < 0)),
                "fraction_above_one": float(np.mean(corrected > 1)),
                "fraction_above_color_max": float(np.mean(corrected > 0.6)),
            }

    apply_figure_style(FigureVariant.PAPER)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.7,
        "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.bbox": None,
    })
    width, height, panel = 7.1, 3.8, 1.4
    fig = plt.figure(figsize=(width, height * 0.9))
    lefts, bottom = (0.53, 3.77), 0.35
    # T2 obtained from the same T1/Tphi relation as the existing paper model.
    settings = original["simulation_settings"]
    t2 = 1 / (1 / (2 * settings["t1_us"]) + 1 / settings["t_phi_us"])
    slope = np.sqrt(settings["t1_us"] / t2)
    for group, domain in enumerate(DOMAINS):
        for row, kind in enumerate(("experiment", "simulation")):
            for col, (protocol, title) in enumerate(PROTOCOLS):
                key = f"{domain}_{protocol}_{kind}"
                x, y, z = (arrays[f"{key}_{suffix}"] for suffix in (
                    "detuning_mhz", "rabi_mhz", "population"
                ))
                ax = fig.add_axes([
                    (lefts[group] + col * panel) / width,
                    (bottom + (1 - row) * panel) / height,
                    panel / width, panel / height,
                ])
                mesh = ax.pcolormesh(x, y, z, shading="auto", cmap="magma",
                                     vmin=0, vmax=0.6, rasterized=True)
                span = 25 if domain == "broad" else 0.5
                if domain == "broad":
                    ref_y = np.linspace(0, min(y[-1], span / slope), 200)
                    for sign in (-1, 1):
                        ax.plot(sign * slope * ref_y, ref_y, color="#ee3333", lw=0.45)
                ax.set(xlim=(-span, span), ylim=(0, y[-1]), yticks=[0, 20, 40],
                       xticks=[-20, 0, 20] if domain == "broad" else [-0.3, 0, 0.3])
                ax.tick_params(direction="in", top=True, right=True,
                               labelbottom=row == 1, labelleft=col == 0,
                               length=2.5, pad=2)
                if row == 1:
                    ax.set_xlabel(r"$\Delta/2\pi$ (MHz)", labelpad=2)
                if col == 0:
                    ax.set_ylabel(r"$\Omega_0/2\pi$ (MHz)", labelpad=2)
                text = ax.text(0.04, 0.92, f"({chr(97 + group * 4 + row * 2 + col)})",
                               color="white", transform=ax.transAxes, va="top", fontsize=8)
                text.set_path_effects([pe.withStroke(linewidth=1, foreground="black")])
                if row == 0:
                    ax.set_title(title, fontsize=7, pad=4)
        fig.text((lefts[group] + panel) / width, 3.53 / height,
                 "Broad spectrum" if domain == "broad" else "Narrow spectrum",
                 ha="center", fontsize=10)
    for row, label in enumerate(("Experiment (corrected*)", "Simulation")):
        fig.text(0.012, (bottom + (1.5 - row) * panel) / height,
                 label, rotation=90, va="center", fontsize=8, color="0.35",
                 fontstyle="italic")
    cb_ax = fig.add_axes([0.38, 3.48 / height, 0.22, 0.07 / height])
    cb = fig.colorbar(mesh, cax=cb_ax, orientation="horizontal", ticks=[0, 0.3, 0.6])
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.tick_params(labelsize=7, length=2, pad=1)
    fig.text(0.615, 3.515 / height, r"$P_e$", va="center", fontsize=8)
    # Add a footer margin while preserving the original physical panel sizes.
    old_height = height * 0.9
    footer_margin = 0.18
    new_height = old_height + footer_margin
    fig.set_size_inches(width, new_height)
    for ax in fig.axes:
        pos = ax.get_position()
        ax.set_position([pos.x0, (pos.y0 * old_height + footer_margin) / new_height,
                         pos.width, pos.height * old_height / new_height])
    for text in fig.texts:
        x, y = text.get_position()
        text.set_position((x, (y * old_height + footer_margin) / new_height))
    fig.text(0.5, 0.008,
             "*Provisional: saved matrix; acquisition-classifier match unverified",
             ha="center", fontsize=6, color="0.35")
    for extension in ("pdf", "png", "svg"):
        fig.savefig(OUTPUT / f"{STEM}.{extension}", dpi=350)
    # Separate diagnostic only; the primary preview retains the requested 0--0.6.
    for ax in fig.axes[:8]:
        ax.collections[0].set_clim(0, 0.8)
    cb.set_ticks([0, 0.4, 0.8])
    fig.savefig(OUTPUT / f"{STEM}_scale_0p8.png", dpi=350)
    plt.close(fig)
    np.savez_compressed(OUTPUT / f"{STEM}.npz", **arrays)
    (OUTPUT / f"{STEM}_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance["runs"], indent=2))
    print(OUTPUT / f"{STEM}.pdf")


if __name__ == "__main__":
    main()
