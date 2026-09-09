"""Figure 5: measured Figure 4 centers and the constant-drive simulation."""

from __future__ import annotations

# Backend and local-source setup precede package imports.
# ruff: noqa: E402, I001
import hashlib
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from echospec.figures import FigureVariant, apply_figure_style, save_figure
from echospec.paper_data import save_paper_dataset
from echospec.analysis.calibration_gaussian import fit_calibration_dip

# Dense linear sampling plus extra points near zero resolves the steep onset
# of the constant-drive linewidth on the logarithmic axis.
RABI_MHZ = np.unique(
    np.concatenate((np.linspace(0.0, 60.0, 1201), np.geomspace(1e-5, 1.0, 201)))
)
# Preserve the original q1 three-level constant-drive reference.
ANHARMONICITY_MHZ = -216.0
SOURCE_PATH = (
    PROJECT_ROOT / "paper/data/experimental/04_main_ac_stark_correction_maps.npz"
)
OUTPUT_STEM = "04_main_ac_stark_shifts_square"
CONSTANT_COLOR = "#C65D4B"
ECHO_SERIES = (
    ("plain", r"$\beta=0$ (measured)", "#6a1b9a"),
    ("drag", r"$\beta=-0.22$ (measured)", "#21918C"),
)


def coherence_reference() -> tuple[float, float, float]:
    """Read the paper's q1 reference: T1 and T2 in us, linewidth in kHz."""
    text = (PROJECT_ROOT / "paper/coherence_parameters.tex").read_text()
    values = []
    for name in ("MeasuredTOne", "EffectiveTTwo", "EffectiveCoherenceFwhm"):
        match = re.search(rf"\\newcommand\{{\\{name}\}}\{{([0-9.]+)\}}", text)
        if match is None:
            raise ValueError(f"Missing coherence parameter {name}")
        values.append(float(match.group(1)))
    return tuple(values)


def constant_drive_fwhm_khz(rabi_mhz, t1_us, t2_us, t2_fwhm_khz):
    """Steady-state Bloch linewidth, Gamma_T2*sqrt(1+Omega^2*T1*T2)."""
    saturation = (2 * np.pi * np.asarray(rabi_mhz)) ** 2 * t1_us * t2_us
    return t2_fwhm_khz * np.sqrt(1 + saturation)


def dressed_resonance_center_mhz(rabi_mhz: np.ndarray) -> np.ndarray:
    """Return the exact dressed g-e center of the three-level Hamiltonian."""
    local_detuning = np.linspace(-30.0, 15.0, 9001)
    centers = np.zeros_like(rabi_mhz, dtype=float)
    for index, omega in enumerate(rabi_mhz):
        hamiltonian = np.zeros((local_detuning.size, 3, 3), dtype=float)
        hamiltonian[:, 1, 1] = -local_detuning
        hamiltonian[:, 2, 2] = -2.0 * local_detuning + ANHARMONICITY_MHZ
        hamiltonian[:, 0, 1] = hamiltonian[:, 1, 0] = omega / 2.0
        hamiltonian[:, 1, 2] = hamiltonian[:, 2, 1] = omega / np.sqrt(2.0)
        eigenvalues = np.linalg.eigvalsh(hamiltonian)
        centers[index] = local_detuning[
            np.argmin(eigenvalues[:, 2] - eigenvalues[:, 1])
        ]
    return centers


def measured_centers(source: dict) -> dict:
    """Select the exact accepted and mean-centered points shown in Figure 4."""
    arrays = {}
    for prefix, _, _ in ECHO_SERIES:
        mask = source[f"{prefix}_display_center_mask"]
        rabi = source["rabi_mhz"][mask]
        centers = source[f"{prefix}_display_centers_mhz"][mask]
        if rabi.size < 2 or not np.all(np.isfinite(centers)):
            raise ValueError(f"Invalid measured {prefix} centers")
        if np.any((rabi < 10) | (rabi > 60)):
            raise ValueError("Measured centers must be within 10--60 MHz")
        if not np.isclose(np.mean(centers), 0, atol=1e-12):
            raise ValueError("Regenerate Figure 4 to obtain mean-centered data")
        arrays[f"{prefix}_rabi_mhz"] = rabi
        arrays[f"{prefix}_centers_mhz"] = centers
        arrays[f"{prefix}_acquisition_centers_mhz"] = source[f"{prefix}_centers_mhz"][
            mask
        ]
        arrays[f"{prefix}_subtracted_mean_mhz"] = source[
            f"{prefix}_display_mean_center_mhz"
        ]
        arrays[f"{prefix}_source_row_indices"] = np.flatnonzero(mask)
    return arrays


def main() -> None:
    with np.load(SOURCE_PATH, allow_pickle=False) as source:
        measured = measured_centers(source)
        for prefix, _, _ in ECHO_SERIES:
            rows = measured[f"{prefix}_source_row_indices"]
            fits = [
                fit_calibration_dip(
                    source["detuning_mhz"] * 1e6, source[f"{prefix}_pe"][row]
                )
                for row in rows
            ]
            for key in fits[0]:
                measured[f"{prefix}_linewidth_fit_{key}"] = np.asarray(
                    [fit[key] for fit in fits]
                )
            accepted = measured[f"{prefix}_linewidth_fit_accepted"]
            # Widths must come from the same estimator as the published centers.
            np.testing.assert_allclose(
                measured[f"{prefix}_linewidth_fit_center_hz"][accepted],
                1e6 * measured[f"{prefix}_acquisition_centers_mhz"][accepted],
                atol=1.0,
                rtol=0,
            )
    source_metadata = json.loads(SOURCE_PATH.with_suffix(".json").read_text())
    common_provenance = {
        "figure_asset": f"figures/paper/{OUTPUT_STEM}.pdf",
        "manuscript_scope": "letter",
        "figure_number": 5,
        "center_inset_ylim_khz": [-16.0, 16.0],
        "generator": "scripts/make_main_ac_stark_shifts.py",
        "reproduction_command": "python scripts/make_main_ac_stark_shifts.py",
    }
    measured_paths = save_paper_dataset(
        OUTPUT_STEM,
        category="experimental",
        arrays=measured,
        provenance={
            **common_provenance,
            "source_dataset": SOURCE_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "source_sha256": hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(),
            "source_campaign": source_metadata["provenance"]["source_campaign"],
            "selection": "Exact Figure 4 display_center_mask; all accepted points at 10--60 MHz",
            "centering": "Exact Figure 4 display_centers_mhz; each run's arithmetic mean removed",
            "center_processing": "Original Figure 4 centers retained without refitting",
            "linewidth_processing": "Refit original measured spectra with echospec.analysis.calibration_gaussian.fit_calibration_dip, matching the campaign estimator; centers verified within 1 Hz",
            "linewidth_estimator": "Offset minus Gaussian dip; trim 40 edge points, Gaussian smoothing sigma=1 sample, bounded curve_fit maxfev=800; FWHM=2*sqrt(2*ln(2))*sigma",
            "linewidth_acceptance": "Figure 4 selected rows, finite positive center error and depth, R squared >= 0.1; no linewidth-based exclusion",
            "uncertainties": "Exported linewidth covariance errors are nominal errors of the smoothed fit; no uncertainty bands plotted",
        },
    )
    dressed_center = dressed_resonance_center_mhz(RABI_MHZ)
    t1_us, t2_us, t2_fwhm_khz = coherence_reference()
    constant_fwhm = constant_drive_fwhm_khz(RABI_MHZ, t1_us, t2_us, t2_fwhm_khz)
    numerical_paths = save_paper_dataset(
        OUTPUT_STEM,
        category="numerical",
        arrays={
            "rabi_mhz": RABI_MHZ,
            "constant_center_mhz": dressed_center,
            "anharmonicity_mhz": np.asarray(ANHARMONICITY_MHZ),
            "constant_fwhm_khz": constant_fwhm,
            "constant_fwhm_t2_units": constant_fwhm / t2_fwhm_khz,
            "reference_t1_us": np.asarray(t1_us),
            "reference_t2_us": np.asarray(t2_us),
            "reference_t2_fwhm_khz": np.asarray(t2_fwhm_khz),
        },
        provenance={
            **common_provenance,
            "model": "Original three-level constant-drive Hamiltonian; center minimizes upper dressed eigenvalue gap",
            "detuning_search_mhz": [-30.0, 15.0],
            "detuning_search_points": 9001,
            "reference": "Unchanged q1 anharmonicity -216 MHz; illustrative reference, not a q6 fit",
            "centering": "No empirical mean subtraction; detuning relative to bare transition",
            "linewidth_model": "Steady-state two-level Bloch power broadening: Gamma=Gamma_T2*sqrt(1+(2*pi*rabi_MHz)^2*T1_us*T2_us)",
            "coherence_source": "paper/coherence_parameters.tex; existing q1 reference, not a q6 coherence measurement",
            "linewidth_axis": "Logarithmic kHz on the left; FWHM/Gamma_T2 on the right",
            "rabi_sampling": "Union of 1201 linear points over 0--60 MHz and 201 logarithmic points over 1e-5--1 MHz; both red curves evaluated directly on this grid",
        },
    )
    apply_figure_style(FigureVariant.PAPER)
    figure, (axis, width_axis) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(3.35, 4.4),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.35, 1.0]},
    )
    axis.set_title("(a)", loc="left")
    axis.plot(
        RABI_MHZ,
        dressed_center,
        "-",
        color=CONSTANT_COLOR,
        label="Constant drive (simulation)",
    )
    for prefix, label, color in ECHO_SERIES:
        axis.plot(
            measured[f"{prefix}_rabi_mhz"],
            measured[f"{prefix}_centers_mhz"],
            ".-",
            color=color,
            ms=1.8,
            lw=0.6,
            label=label,
        )
    axis.axhline(0, color="0.5", lw=0.6)
    axis.set(
        ylabel="Resonance-center shift (MHz)",
        xlim=(0, 60),
        xticks=[0, 10, 20, 30, 40, 50, 60],
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=5.0, loc="upper left")

    # Symmetric display zoom; preserve all points in the data and main panel.
    inset = axis.inset_axes([0.35, 0.27, 0.62, 0.40])
    for prefix, _, color in ECHO_SERIES:
        inset.plot(
            measured[f"{prefix}_rabi_mhz"],
            1e3 * measured[f"{prefix}_centers_mhz"],
            ".-",
            color=color,
            ms=1.3,
            lw=0.6,
        )
    inset.axhline(0, color="0.5", lw=0.6)
    inset.set(
        xlim=(10, 60),
        ylim=(-16, 16),
        xticks=[10, 20, 30, 40, 50, 60],
        yticks=[-16, -8, 0, 8, 16],
    )
    inset.set_ylabel("Center variation (kHz)", fontsize=5.2)
    inset.set_title("Measured resonance centers", fontsize=5.5)
    inset.tick_params(labelsize=4.8)
    inset.grid(alpha=0.2)
    width_axis.plot(
        RABI_MHZ,
        constant_fwhm,
        "-",
        color=CONSTANT_COLOR,
        lw=0.8,
        label="Constant drive (model)",
    )
    for prefix, label, color in ECHO_SERIES:
        widths = np.where(
            measured[f"{prefix}_linewidth_fit_accepted"],
            measured[f"{prefix}_linewidth_fit_fwhm_hz"] / 1e3,
            np.nan,
        )
        width_axis.plot(
            measured[f"{prefix}_rabi_mhz"],
            widths,
            ".-",
            color=color,
            ms=1.8,
            lw=0.6,
            label=label,
        )
    width_axis.set(
        title="",
        xlabel=r"$\Omega_{\mathrm{cal}}/2\pi$ (MHz)",
        ylabel="FWHM (kHz)",
        xlim=(0, 60),
        yscale="log",
        ylim=(10, 1e6),
    )
    width_axis.axhline(t2_fwhm_khz, color="0.5", ls=":", lw=0.6)
    secondary = width_axis.secondary_yaxis(
        "right",
        functions=(
            lambda width: width / t2_fwhm_khz,
            lambda units: units * t2_fwhm_khz,
        ),
    )
    secondary.set_ylabel(r"FWHM / $\Gamma_{T_2}$")
    width_axis.set_title("(b)", loc="left")
    width_axis.grid(alpha=0.2)
    width_axis.legend(fontsize=5.5, loc="best")
    paths = save_figure(
        figure,
        OUTPUT_STEM,
        variant=FigureVariant.PAPER,
        formats=("pdf", "png", "svg"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(figure)
    for path in paths:
        if path.suffix == ".svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines())
                + "\n"
            )
    for path in (*paths, *measured_paths, *numerical_paths):
        print(path)


if __name__ == "__main__":
    main()
