"""SR²-CSD: Spatially Regularized Super-Resolved Spherical Deconvolution."""

from .core.fit_qp import fit_qp
from .model import SR2CSDModel

__all__ = (
    "fit_qp", 
    "SR2CSDModel",
)
