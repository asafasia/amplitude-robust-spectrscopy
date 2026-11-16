"""Present legacy polynomial-pulse results in the manuscript DRAG convention.

This conversion is for labels and exported figure metadata only. The archived
polynomial backend uses Q=-beta_source*dI/dt/abs(alpha); its pulses and saved
simulation arrays retain that convention.
"""

import numpy as np

PAPER_DRAG_CONVENTION = (
    "I-iQ; Q=-beta*dI/dt/alpha; alpha=omega12-omega01<0 in angular-frequency units"
)
SOURCE_DRAG_CONVENTION = "I-iQ; Q=-beta_source*dI/dt/abs(alpha)"


def polynomial_beta_to_paper(beta_source):
    """Convert the legacy negative-anharmonicity parameter without changing Q."""
    beta = -np.asarray(beta_source, dtype=float)
    return np.where(beta == 0, 0.0, beta)


def polynomial_beta_provenance(beta_source):
    """Record both conventions for an unchanged polynomial simulation setting."""
    return {
        "paper_definition": PAPER_DRAG_CONVENTION,
        "paper_beta": float(polynomial_beta_to_paper(beta_source)),
        "source_definition": SOURCE_DRAG_CONVENTION,
        "source_beta": float(beta_source),
        "conversion": "beta=-beta_source; waveform and populations unchanged",
    }
