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
