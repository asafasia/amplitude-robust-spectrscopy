"""Build an offline spectrum and fit explorer for the current cutoff survey.

PYTHONPATH=. MPLBACKEND=Agg .venv/bin/python scripts/report_cutoff_fits.py
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from echospec.analysis.cutoff_fit_exports import load_cutoff_fits
from echospec.analysis.duration_cutoffs import duration_directory
from echospec.figures.coherence import REFERENCE_FWHM_HZ, REFERENCE_T2_STAR_S
from echospec.figures.cutoff_contours import (
    CONTOUR_SMOOTHING,
    smooth_contrast_for_contour,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/polynomial_duration_cutoff_poly2_log_40_50/n100"
TEMPLATE = ROOT / "scripts/templates/cutoff_fit_report.html"
FIT_FIELDS = (
    "parameters", "accepted", "fwhm_hz", "contrast", "center_hz", "relative_rms",
    "relative_width_error", "radius_mhz", "reference_mhz", "y_offset", "y_scale",
)


def pack_arrays(arrays):
    """Pack aligned typed arrays losslessly, then gzip/base64 for offline HTML."""
    chunks, layout, offset = [], {}, 0
    for key, value in arrays.items():
        array = np.ascontiguousarray(value)
        if array.dtype not in (np.dtype("<f4"), np.dtype("<f8"), np.dtype("uint8")):
            raise ValueError(f"Unsupported browser dtype: {array.dtype}")
        padding = (-offset) % 8
        chunks.append(b"\0" * padding)
        offset += padding
        layout[key] = dict(offset=offset, shape=list(array.shape),
                           count=array.size, dtype=array.dtype.str)
        chunks.append(array.tobytes())
        offset += array.nbytes
    raw = b"".join(chunks)
    return dict(layout=layout, byte_length=len(raw),
                payload=base64.b64encode(gzip.compress(raw, mtime=0)).decode("ascii"))


def load_plotly_bundle():
    """Prefer installed Plotly; reuse the licensed offline bundle if unavailable."""
    try:
        from plotly.offline import get_plotlyjs
        return get_plotlyjs()
    except ImportError:
        old_report = ROOT / "outputs/polynomial_duration_cutoff_sweep/n100/report.html"
        text = old_report.read_text()
        for match in re.finditer(r"<script>(.*?)</script>", text, re.DOTALL):
            if re.match(r"/\*\*\s*\* plotly\.js v", match.group(1)):
                return match.group(1)
        raise RuntimeError("Install Plotly in .venv to bundle its offline runtime") from None


def contrast_boundary(cutoffs, amps, contrast, accepted, *, smooth=True):
    """Use the same masked contour construction as the manuscript figures."""
    fig, ax = plt.subplots()
    field = (smooth_contrast_for_contour(contrast, accepted) if smooth
             else np.ma.masked_where(~accepted, contrast))
    result = ax.contour(cutoffs, amps, field.T,
                        levels=[.2], corner_mask=False)
    paths = [path.tolist() for path in result.allsegs[0] if len(path) >= 2]
    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = args.input_dir.resolve()
    destination = args.output or source / "fit_explorer.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = source / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    common = manifest["common_configuration"]
    lengths = np.asarray(manifest["durations_us"])
    cutoffs, amps = np.asarray(common["cutoffs"]), np.asarray(common["rabi_mhz"])
    detuning = np.asarray(common["detuning_mhz"])
    scans, source_hashes = [], {}
    for length in lengths:
        path = duration_directory(source, manifest, length) / "scan.npz"
        with np.load(path, allow_pickle=False) as saved:
            for key, grid in (("cutoffs", cutoffs), ("rabi_mhz", amps),
                              ("detuning_mhz", detuning)):
                np.testing.assert_array_equal(saved[key], grid)
            scans.append({"population": saved["populations"][:, 1].copy(),
                          **{key: saved[key].copy()
                             for key in ("accepted", "contrast", "fwhm_hz")}})
        source_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    original_accepted = np.stack([scan["accepted"] for scan in scans])
    _, fit_provenance, fit_hashes = load_cutoff_fits(
        source, lengths, cutoffs, amps, original_accepted)
    source_hashes.update(fit_hashes)
    with np.load(source / "robust_fits/metrics.npz", allow_pickle=False) as saved:
        fits = {key: saved[key].copy() for key in (*FIT_FIELDS, "status")}
    statuses = sorted(set(fits["status"].ravel()))
    status_ids = {status: index for index, status in enumerate(statuses)}
    cases, display_errors = [], []
    for row, (length, scan) in enumerate(zip(lengths, scans, strict=True)):
        population = scan["population"].astype("<f4")
        if population.shape != (len(cutoffs), len(amps), len(detuning)):
            raise ValueError("Unexpected spectrum grid")
        display_errors.append(float(np.max(abs(population - scan["population"]))))
        arrays = {"population": population}
        arrays.update({key: fits[key][row].astype("uint8" if key == "accepted" else "<f8")
                       for key in FIT_FIELDS})
        arrays.update(
            status=np.array([status_ids[s] for s in fits["status"][row].ravel()],
                            dtype="uint8").reshape(len(cutoffs), len(amps)),
            direct_fwhm_hz=scan["fwhm_hz"].astype("<f8"),
            direct_contrast=scan["contrast"].astype("<f8"),
        )
        cases.append(dict(
            duration=float(length), **pack_arrays(arrays),
            boundary=contrast_boundary(cutoffs, amps, fits["contrast"][row],
                                       fits["accepted"][row]),
            original_boundary=contrast_boundary(cutoffs, amps, fits["contrast"][row],
                                                fits["accepted"][row], smooth=False),
        ))
    li = int(np.flatnonzero(lengths == 20)[0])
    ai = int(np.argmin(abs(np.log(amps / 12))))
    valid = np.flatnonzero(fits["accepted"][li, :, ai])
    peak = int(np.nanargmax(fits["contrast"][li, :, ai]))
    candidates = valid[valid > peak]
    ci = int(candidates[np.argmin(abs(fits["contrast"][li, candidates, ai] - .2))])
    payload = dict(
        durations=lengths.tolist(), cutoffs=cutoffs.tolist(), amps=amps.tolist(),
        detuning=detuning.tolist(), cases=cases, statuses=statuses,
        reference_hz=REFERENCE_FWHM_HZ, reference_t2_us=REFERENCE_T2_STAR_S * 1e6,
        default=dict(li=li, ci=ci, ai=ai), settings=fit_provenance["settings"],
        model=dict(t1_us=common["t1_us"], t2_us=common["t2_us"],
                   power=common["power"], beta=common["drag_beta"],
                   transition_ns=common["transition_us"] * 1000),
    )
    runtime = load_plotly_bundle()
    text = TEMPLATE.read_text().replace("/*__PLOTLY_JS__*/", runtime)
    text = text.replace("__STUDY_DATA__", json.dumps(payload, separators=(",", ":"),
                                                        allow_nan=False))
    temporary = destination.with_suffix(".tmp.html")
    temporary.write_text(text)
    temporary.replace(destination)
    for path in (Path(__file__).resolve(), TEMPLATE, manifest_path,
                 ROOT / "echospec/analysis/cutoff_fit_exports.py",
                 ROOT / "echospec/analysis/robust_dip_fit.py",
                 ROOT / "echospec/figures/coherence.py",
                 ROOT / "echospec/figures/cutoff_contours.py"):
        source_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    provenance = dict(
        source_sha256=source_hashes, html_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        plotly_sha256=hashlib.sha256(runtime.encode()).hexdigest(),
        durations_us=lengths.tolist(), spectra=len(lengths) * len(cutoffs) * len(amps),
        samples_per_spectrum=len(detuning), reference_linewidth_hz=REFERENCE_FWHM_HZ,
        population_display_precision="float32; original float64 spectra unchanged",
        maximum_display_population_error=max(display_errors), fit_parameters_precision="float64",
        fit_definition=fit_provenance["definition"], default_indices=payload["default"],
        contour_smoothing=CONTOUR_SMOOTHING,
        note="All saved spectra included. Rejected candidates are labeled and excluded from "
             "reported metrics. No new simulations, fitting, or metric interpolation. "
             "Only the displayed white boundary is smoothed; the original can be toggled.",
    )
    destination.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Created {destination} ({destination.stat().st_size / 1e6:.1f} MB)")
    print(f"Embedded {provenance['spectra']} spectra; max display error "
          f"{provenance['maximum_display_population_error']:.3g}")


if __name__ == "__main__":
    main()
