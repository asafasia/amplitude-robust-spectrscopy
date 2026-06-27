# Figure Workflow

Use paired notebooks for related figures, and keep the scientific logic in
Python modules.

## Folder contract

```text
notebooks/paper/
notebooks/presentation_ips/
figures/paper/
figures/presentation_ips/
styles/paper.mplstyle
styles/presentation_ips.mplstyle
echospec/figures/
```

## Notebook pattern

Start each paper notebook with:

```python
from echospec.figures import FigureVariant, apply_figure_style, save_figure

VARIANT = FigureVariant.PAPER
apply_figure_style(VARIANT)
```

Start each IPS presentation notebook with:

```python
from echospec.figures import FigureVariant, apply_figure_style, save_figure

VARIANT = FigureVariant.PRESENTATION_IPS
apply_figure_style(VARIANT)
```

Save figures with:

```python
save_figure(fig, "figure_name", variant=VARIANT)
```

## Parallel development rule

When a paper figure and presentation figure are related, give them matching
notebook names:

```text
notebooks/paper/01_spectroscopy_comparison.ipynb
notebooks/presentation_ips/01_spectroscopy_comparison.ipynb
```

The paper notebook should keep full scientific detail. The presentation notebook
should keep the same data and message, but use fewer labels, fewer overlays,
larger fonts, and simpler annotations.
