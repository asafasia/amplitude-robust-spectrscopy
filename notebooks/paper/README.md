# Paper Figure Notebooks

Use this folder for notebooks that produce paper-oriented figures:
compact panels, small fonts, full labels, and publication-ready exports.

Start each notebook with:

```python
from echospec.figures import FigureVariant, apply_figure_style, save_figure

VARIANT = FigureVariant.PAPER
apply_figure_style(VARIANT)
```

Save outputs with:

```python
save_figure(fig, "figure_name", variant=VARIANT)
```

Outputs go to `figures/paper/`.
