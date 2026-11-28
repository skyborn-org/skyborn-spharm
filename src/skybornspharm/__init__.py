"""
spharm - Spherical harmonic transforms for atmospheric/oceanic modeling

This module provides Python interfaces to a Pure-Rust SPHEREPACK-compatible
engine for spherical harmonic transforms on the sphere. It is particularly
useful for atmospheric and oceanic modeling applications.

Main classes:
    Spharmt: Main interface for spherical harmonic transforms
    SpharmTransform: Modern C-order alias for Spharmt (defaults to layout="c")
    ReducedGaussianSpharmt: Interface for packed reduced Gaussian transforms
    ReducedGaussianTransform: Modern C-order alias for ReducedGaussianSpharmt

Example:
    >>> from skybornspharm import SpharmTransform
    >>> sht = SpharmTransform(nlon=144, nlat=73)
    >>> spec = sht.grdtospec(data)  # Grid to spectral transform
    >>> data_back = sht.spectogrd(spec)  # Spectral to grid transform
"""

from .reduced_gaussian import ReducedGaussianSpharmt, ReducedGaussianTransform
from .spherical_harmonics import (
    SpherepackError,
    Spharmt,
    SpharmTransform,
    SpheremackError,
    ValidationError,
    gaussian_lats_wts,
    getgeodesicpts,
    getspecindx,
    invlap,
    legendre,
    regrid,
    specintrp,
)

__all__ = [
    "__version__",
    "Spharmt",
    "SpharmTransform",
    "SpherepackError",
    "SpheremackError",
    "ValidationError",
    "regrid",
    "invlap",
    "gaussian_lats_wts",
    "getspecindx",
    "getgeodesicpts",
    "legendre",
    "specintrp",
    "ReducedGaussianSpharmt",
    "ReducedGaussianTransform",
]


__version__ = "0.2.0"
__author__ = "Qianye Su"
__license__ = "BSD-3-Clause"
