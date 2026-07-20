"""Build the July 12 cutoff-campaign explorer notebook."""

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "echo_lorentzian_cutoff_explorer.ipynb"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


notebook = nbf.v4.new_notebook()
notebook["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.12"},
}
notebook["cells"] = [
    markdown(
        """
# Echo-Lorentzian cutoff explorer

This notebook is fixed to the completed experimental campaign
`cutoff_amp_fwhm_map/20260712_171809_single_cutoff_full`: 20 cutoff values,
25 amplitude points per cutoff, 5 MHz detuning span, 5 kHz frequency step, and
60 shots.

The campaign retained its rendered measured-state maps and OPX-generated fit
tables, but not numerical `state` arrays. Therefore this notebook displays the
retained 2D measurements and independently quality-screens the stored OPX FWHM
fits; it does not claim to refit pixels from PNG images.
"""
    ),
    code(
        """
%matplotlib inline

from pathlib import Path
import sys

for candidate in (Path.cwd(), *Path.cwd().parents):
    if (candidate / "echospec").is_dir():
        REPO_ROOT = candidate
        break
else:
    raise FileNotFoundError("Could not locate the repository root containing echospec/.")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from echospec.analysis.echo_lorentzian_cutoff import (
    AggregateCutoffExplorer,
    aggregate_quality_mask,
    load_aggregate_campaign,
    plot_aggregate_cutoff,
)

campaign = load_aggregate_campaign()
accepted = aggregate_quality_mask(campaign)
print(f"Loaded {len(campaign.cutoffs)} cutoffs and {len(campaign.results)} stored fit rows.")
print(f"Quality gate accepts {int(accepted.sum())}/{len(accepted)} rows.")
"""
    ),
    markdown(
        """
## Immediately visible example

The middle cutoff is shown initially. Grey crosses are stored fits rejected by
the quality gate. Accepted fits require `R² >= 0.60`, fit amplitude from `0.05`
to `1`, at least two measured frequency bins of width, a bounded center, and a
FWHM below half the sweep span.

Resolution color uses logarithmic normalization with `vmin=0.1` and `vmax=1`.
Resolution and fit-amplitude-times-resolution use logarithmic axes.
"""
    ),
    code(
        """
cutoff = campaign.cutoffs[len(campaign.cutoffs) // 2]
figure, axes = plot_aggregate_cutoff(
    campaign,
    cutoff,
    resolution_vmin=0.1,
    resolution_vmax=1.0,
)
figure
"""
    ),
    markdown(
        """
## Live 20-point cutoff sweeper

Run this cell in JupyterLab or VS Code with the repository Python kernel. A
static notebook preview cannot execute Python widget callbacks and may display
the widget's plain-text fallback instead of the slider.
"""
    ),
    code(
        """
explorer = AggregateCutoffExplorer(campaign)
explorer.display()
"""
    ),
    markdown(
        """
## Provenance and limitation

- Campaign: `data_opx1000/cutoff_amp_fwhm_map/20260712_171809_single_cutoff_full`.
- Manifest reports `complete=true`, 20/20 completed runs, and no failures.
- The stored FWHM fields were produced by the OPX1000 shaped-pulse Gaussian
  analysis. The source-controlled fork remains available in
  `echospec.analysis.echo_lorentzian_cutoff.opx1000_fwhm` for raw-backed runs.
- This campaign contains no NPZ/HDF5/numerical state arrays or raw-data IDs, so
  an independent FWHM refit is impossible without recovering the original raw
  acquisition records.
"""
    ),
]

nbf.write(notebook, OUTPUT)
print(OUTPUT)
