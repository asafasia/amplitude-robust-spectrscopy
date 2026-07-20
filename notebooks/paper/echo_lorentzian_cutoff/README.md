# Echo-Lorentzian cutoff explorer

This folder contains the focused July 12 aggregate-campaign explorer notebook
and its reproducible builder. Analysis code lives in
`echospec.analysis.echo_lorentzian_cutoff` so the notebook remains short and
reviewable.

Build and execute from the repository root:

```bash
.venv/bin/python notebooks/paper/echo_lorentzian_cutoff/build_notebook.py
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 \
  notebooks/paper/echo_lorentzian_cutoff/echo_lorentzian_cutoff_explorer.ipynb
```

The notebook targets
`cutoff_amp_fwhm_map/20260712_171809_single_cutoff_full`. For live widgets, open
the notebook in JupyterLab or VS Code and run all cells.
The widgets require a running Python kernel; static previewers show only their
plain-text fallback representation.
