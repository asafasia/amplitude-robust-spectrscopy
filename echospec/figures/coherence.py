"""Common linewidth reference for manuscript figures, independent of source fits."""

from math import pi

REFERENCE_T2_STAR_S = 7.6e-6
REFERENCE_FWHM_HZ = 1 / (pi * REFERENCE_T2_STAR_S)
