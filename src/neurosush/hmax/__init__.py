"""HMAX building blocks (Serre et al. 2007): Gabor bank, S1, C1, S2, C2 and an invariance probe."""

from .c1 import c1_bands_from_spec, c1_pool
from .c2 import c2_global_max
from .gabor import (
    SERRE2007_BANDS,
    SERRE2007_GAMMA,
    SERRE2007_THETAS,
    GaborBandSpec,
    gabor_bank,
    serre2007_gabor_bank,
)
from .inhibition import spatial_lateral_inhibition
from .probe import bar_image, invariance_curve, normalized_correlation, shift_image
from .s1 import s1_response
from .s2 import s2_rbf, sample_c1_patches
from .wta import select_conv_winners, simplified_stdp, stdp_convergence

__all__ = [
    "SERRE2007_BANDS",
    "SERRE2007_GAMMA",
    "SERRE2007_THETAS",
    "GaborBandSpec",
    "bar_image",
    "c1_bands_from_spec",
    "c1_pool",
    "c2_global_max",
    "gabor_bank",
    "invariance_curve",
    "normalized_correlation",
    "s1_response",
    "s2_rbf",
    "sample_c1_patches",
    "select_conv_winners",
    "serre2007_gabor_bank",
    "shift_image",
    "simplified_stdp",
    "spatial_lateral_inhibition",
    "stdp_convergence",
]
