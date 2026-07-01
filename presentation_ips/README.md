# IPS Presentation Figure Notebooks

Use this folder for notebooks that produce IPS presentation figures:
larger fonts, simpler labels, fewer annotations, and slide-friendly aspect
ratios.

Start each notebook with:

```python
from echospec.figures import FigureVariant, apply_figure_style, save_figure

VARIANT = FigureVariant.PRESENTATION_IPS
apply_figure_style(VARIANT)
```

Save outputs with:

```python
save_figure(fig, "figure_name", variant=VARIANT)
```

Outputs go to `figures/presentation_ips/`.

Current IPS notebooks:

- `00_template.ipynb` - blank starting point for new slide figures.
- `01_t1_t2_decay.ipynb` - first-slide T1/T2 decay figure.
- `01_spectroscopy_comparison.ipynb` - spectroscopy comparison starter.
- `02_2d_spectroscopy.ipynb` - 2D spectroscopy starter.
- `03_pulses.ipynb` - pulse profile starter.
