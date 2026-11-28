"""
Introduction
============

This module provides a Python interface to a Pure-Rust SPHEREPACK-compatible
spherical harmonic transform engine (`skybornspharm._spharm_rs`) oriented toward
working with atmospheric and oceanic general circulation model (GCM) data.

Usage
=====

>>> from skybornspharm import Spharmt, SpharmTransform
>>> x = Spharmt(144, 72, rsphere=8e6, gridtype="gaussian", legfunc="computed")
>>> t = SpharmTransform(144, 73)  # defaults to layout="c"

Class methods
=============
 - grdtospec: grid to spectral transform (spherical harmonic analysis).
 - spectogrd: spectral to grid transform (spherical harmonic synthesis).
 - getuv: compute u and v winds from spectral coefficients of vorticity
   and divergence.
 - getvrtdivspec: get spectral coefficients of vorticity and divergence
   from u and v winds.
 - getgrad: compute the vector gradient given spectral coefficients.
 - getpsichi: compute streamfunction and velocity potential from winds.
 - invlap / invlapspec: apply the inverse spherical Laplacian to spectra.
 - specsmooth: isotropic spectral smoothing.

Functions
=========
 - regrid: spectral re-gridding, with optional spectral smoothing and/or
   truncation.
 - invlap: standalone inverse spherical Laplacian on spectral coefficients.
 - gaussian_lats_wts: compute gaussian latitudes and weights.
 - getspecindx: compute indices of zonal wavenumber and degree
   for complex spherical harmonic coefficients.
 - legendre: compute associated legendre functions.
 - getgeodesicpts: computes the points on the surface of the sphere
   corresponding to a twenty-sided (icosahedral) geodesic.
 - specintrp: spectral interpolation to an arbitrary point on the sphere.
"""

from __future__ import annotations

import math
import sys
from typing import Any, Callable, Literal, Optional, Tuple, Union

__all__ = [
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
]

import numpy as np
from numpy.typing import NDArray

from . import _spharm_rs

# Type aliases for better readability
GridType = Literal["regular", "gaussian"]
LegendreFunc = Literal["stored", "computed"]
PrecisionMode = Literal["auto", "single", "float32", "double", "float64"]
LayoutMode = Literal["legacy", "c", "auto"]
LatOrderMode = Literal["north_to_south", "south_to_north", "N2S", "S2N", "auto"]
FloatArray = NDArray[np.floating]
ComplexArray = NDArray[np.complexfloating]
IntArray = NDArray[np.integer]

# Read-only instance variables that cannot be rebound or unbound.
_private_vars = (
    "nlon",
    "nlat",
    "gridtype",
    "legfunc",
    "rsphere",
    "precision",
    "layout",
    "lat_order",
    "south_to_north",
)

# Constants for better code clarity
DEFAULT_EARTH_RADIUS = 6.3712e6
MIN_NLON = 4
MIN_NLAT = 3
DEGREES_TO_RADIANS = math.pi / 180.0
RADIANS_TO_DEGREES = 180.0 / math.pi
_DTYPE_F32 = np.dtype(np.float32)
_DTYPE_F64 = np.dtype(np.float64)
_DTYPE_C64 = np.dtype(np.complex64)
_DTYPE_C128 = np.dtype(np.complex128)

PRECISION_ALIASES = {
    "auto": "auto",
    "single": "single",
    "float32": "single",
    "double": "double",
    "float64": "double",
}
LAYOUT_ALIASES = {
    "legacy": "legacy",
    "c": "c",
    "auto": "auto",
}
LAT_ORDER_ALIASES = {
    "north_to_south": "north_to_south",
    "n2s": "north_to_south",
    "south_to_north": "south_to_north",
    "s2n": "south_to_north",
    "auto": "auto",
}


class SpherepackError(Exception):
    """Custom exception for SPHEREPACK / spherical harmonic backend errors."""

    pass


SpheremackError = SpherepackError


class ValidationError(ValueError):
    """Custom exception for input validation errors."""

    pass


def _validate_precision(precision: str) -> str:
    """Validate and normalize the requested public-output precision mode."""
    normalized = PRECISION_ALIASES.get(precision) if type(precision) is str else None
    if normalized is None:
        normalized = PRECISION_ALIASES.get(str(precision).lower())
    if normalized is None:
        raise ValidationError(
            'precision must be "auto", "single", "float32", "double", or "float64", '
            f'got "{precision}"'
        )
    return normalized


def _validate_layout(layout: str) -> str:
    """Validate and normalize the array layout mode."""
    normalized = LAYOUT_ALIASES.get(layout) if type(layout) is str else None
    if normalized is None:
        normalized = LAYOUT_ALIASES.get(str(layout).lower())
    if normalized is None:
        raise ValidationError(
            f'layout must be "legacy", "c", or "auto", got "{layout}"'
        )
    return normalized


def _validate_lat_order(lat_order: str) -> str:
    """Validate and normalize the latitude ordering mode."""
    normalized = LAT_ORDER_ALIASES.get(lat_order) if type(lat_order) is str else None
    if normalized is None:
        normalized = LAT_ORDER_ALIASES.get(str(lat_order).lower())
    if normalized is None:
        raise ValidationError(
            'lat_order must be "north_to_south", "south_to_north", '
            f'"N2S", "S2N", or "auto", got "{lat_order}"'
        )
    return normalized


def _resolve_lat_order(
    nlat: int, lat_order: str, lat: Optional[FloatArray]
) -> Tuple[str, bool]:
    """Validate optional latitude coordinates and resolve south-to-north flag."""
    normalized_order = _validate_lat_order(lat_order)
    if lat is None:
        return normalized_order, normalized_order == "south_to_north"

    lat_arr = np.asarray(lat, dtype=np.float64)
    if lat_arr.ndim != 1 or lat_arr.size != nlat:
        raise ValidationError(
            f"lat must be rank 1 with length {nlat}, got shape {lat_arr.shape}"
        )
    if not np.isfinite(lat_arr).all():
        raise ValidationError("lat must contain only finite values")

    diffs = np.diff(lat_arr)
    is_increasing = bool(np.all(diffs > 0.0))
    is_decreasing = bool(np.all(diffs < 0.0))
    if not (is_increasing or is_decreasing):
        raise ValidationError("lat must be strictly monotonic")

    detected_s2n = bool(lat_arr[0] < lat_arr[-1])
    if normalized_order == "north_to_south" and detected_s2n:
        raise ValidationError(
            'lat increases south-to-north but lat_order="north_to_south" was specified'
        )
    if normalized_order == "south_to_north" and not detected_s2n:
        raise ValidationError(
            'lat decreases north-to-south but lat_order="south_to_north" was specified'
        )
    return normalized_order, detected_s2n


def _input_public_real_dtype(array: np.ndarray) -> np.dtype:
    """Map an input array dtype to the matching public real dtype."""
    arr = array if type(array) is np.ndarray else np.asarray(array)
    dtype = arr.dtype
    if dtype == _DTYPE_F32 or dtype == _DTYPE_C64:
        return _DTYPE_F32
    return _DTYPE_F64


def _resolve_public_real_dtype(precision: str, *arrays: np.ndarray) -> np.dtype:
    """Resolve the public real dtype given precision policy and input arrays."""
    if precision == "single":
        return _DTYPE_F32
    if precision == "double":
        return _DTYPE_F64

    for array in arrays:
        arr = array if type(array) is np.ndarray else np.asarray(array)
        dtype = arr.dtype
        if dtype != _DTYPE_F32 and dtype != _DTYPE_C64:
            return _DTYPE_F64
    return _DTYPE_F32


def _resolve_public_complex_dtype(precision: str, *arrays: np.ndarray) -> np.dtype:
    """Resolve the public complex dtype given precision policy and input arrays."""
    return (
        _DTYPE_C128
        if _resolve_public_real_dtype(precision, *arrays) == _DTYPE_F64
        else _DTYPE_C64
    )


def _infer_ntrunc_from_spec_size(
    size: int, max_ntrunc: int, check_bounds: bool = True
) -> Optional[int]:
    """Infer triangular truncation ntrunc from a 1D spectral length, or None."""
    if size < 1:
        return None
    ntrunc = int(-1.5 + 0.5 * math.sqrt(9.0 - 8.0 * (1.0 - size)))
    if (ntrunc + 1) * (ntrunc + 2) // 2 != size:
        return None
    if check_bounds and ntrunc > max_ntrunc:
        return None
    return ntrunc


def _resolve_spectral_layout(
    data: np.ndarray, layout: str, max_ntrunc: int, operation_name: str
) -> Tuple[str, int, Tuple[int, ...]]:
    """Resolve spectral axis position ('c' vs 'legacy'), ntrunc, and batch shape."""
    if data.ndim < 1:
        raise ValidationError(
            f"{operation_name} needs at least a rank 1 array, got rank {data.ndim}"
        )

    if data.ndim == 1:
        size = data.shape[0]
        ntrunc_unbounded = _infer_ntrunc_from_spec_size(
            size, max_ntrunc, check_bounds=False
        )
        if ntrunc_unbounded is None:
            raise ValidationError(
                f"{operation_name} got invalid triangular spectral size {size}"
            )
        if ntrunc_unbounded > max_ntrunc:
            raise ValidationError(
                f"ntrunc too large - can be max of {max_ntrunc}, got {ntrunc_unbounded}"
            )
        return "c", ntrunc_unbounded, ()

    if layout == "c":
        size = data.shape[-1]
        ntrunc_unbounded = _infer_ntrunc_from_spec_size(
            size, max_ntrunc, check_bounds=False
        )
        if ntrunc_unbounded is None:
            raise ValidationError(
                f"{operation_name} got invalid triangular spectral size {size}"
            )
        if ntrunc_unbounded > max_ntrunc:
            raise ValidationError(
                f"ntrunc too large - can be max of {max_ntrunc}, got {ntrunc_unbounded}"
            )
        return "c", ntrunc_unbounded, tuple(data.shape[:-1])

    if layout == "legacy":
        size = data.shape[0]
        ntrunc_unbounded = _infer_ntrunc_from_spec_size(
            size, max_ntrunc, check_bounds=False
        )
        if ntrunc_unbounded is None:
            raise ValidationError(
                f"{operation_name} got invalid triangular spectral size {size}"
            )
        if ntrunc_unbounded > max_ntrunc:
            raise ValidationError(
                f"ntrunc too large - can be max of {max_ntrunc}, got {ntrunc_unbounded}"
            )
        return "legacy", ntrunc_unbounded, tuple(data.shape[1:])

    # layout == "auto"
    t_last = _infer_ntrunc_from_spec_size(data.shape[-1], max_ntrunc)
    t_first = _infer_ntrunc_from_spec_size(data.shape[0], max_ntrunc)

    if t_last is not None and t_first is None:
        return "c", t_last, tuple(data.shape[:-1])
    if t_first is not None and t_last is None:
        return "legacy", t_first, tuple(data.shape[1:])
    if t_first is not None and t_last is not None:
        if data.flags.f_contiguous and not data.flags.c_contiguous:
            return "legacy", t_first, tuple(data.shape[1:])
        if t_first == max_ntrunc and t_last != max_ntrunc:
            return "legacy", t_first, tuple(data.shape[1:])
        if t_last == max_ntrunc and t_first != max_ntrunc:
            return "c", t_last, tuple(data.shape[:-1])
        if data.shape[-1] >= data.shape[0]:
            return "c", t_last, tuple(data.shape[:-1])
        return "legacy", t_first, tuple(data.shape[1:])

    # Neither end is valid within bounds; provide informative error message.
    for size in (data.shape[0], data.shape[-1]):
        unbounded = _infer_ntrunc_from_spec_size(size, max_ntrunc, check_bounds=False)
        if unbounded is not None and unbounded > max_ntrunc:
            raise ValidationError(
                f"ntrunc too large - can be max of {max_ntrunc}, got {unbounded}"
            )
    raise ValidationError(
        f"{operation_name} got invalid triangular spectral shape {data.shape}"
    )


def _call_backend(func: Callable[..., Any], *args: Any, operation_name: str, **kwargs: Any) -> Any:
    """Invoke a Rust backend function, wrapping unexpected errors as SpheremackError."""
    try:
        return func(*args, **kwargs)
    except (SpheremackError, ValidationError):
        raise
    except Exception as exc:
        raise SpheremackError(f"{operation_name} failed: {exc}") from exc


class Spharmt:
    """
    Spherical harmonic transform class backed by the Pure-Rust `_spharm_rs` engine.

    Attributes:
        nlat: Number of latitudes (read-only after initialization)
        nlon: Number of longitudes (read-only after initialization)
        rsphere: Sphere radius in meters (read-only after initialization)
        gridtype: Grid type - 'regular' or 'gaussian' (read-only after initialization)
        legfunc: Legendre function handling - 'stored' or 'computed' (read-only after initialization)
        precision: Public output precision mode - 'auto', 'single', or 'double'
        layout: Array layout mode - 'legacy', 'c', or 'auto'
        lat_order: Latitude ordering mode - 'north_to_south', 'south_to_north', or 'auto'
        south_to_north: Resolved boolean indicating south-to-north input/output latitude order
    """

    def __init__(
        self,
        nlon: int,
        nlat: int,
        rsphere: float = DEFAULT_EARTH_RADIUS,
        gridtype: GridType = "regular",
        legfunc: LegendreFunc = "stored",
        precision: PrecisionMode = "auto",
        layout: LayoutMode = "auto",
        lat_order: LatOrderMode = "auto",
        lat: Optional[FloatArray] = None,
    ) -> None:
        """
        Create a Spharmt class instance.

        Args:
            nlon: Number of longitudes (must be >= 4)
            nlat: Number of latitudes (must be >= 3)
            rsphere: Sphere radius in meters (default: Earth radius)
            gridtype: Grid type - 'regular' or 'gaussian'
            legfunc: Legendre function handling - 'stored' or 'computed'
            precision: Public output precision mode. 'auto' preserves input
                float32/float64 precision natively, 'single' returns float32/complex64,
                and 'double' returns float64/complex128.
            layout: Array axis ordering - 'legacy' for (lat, lon, ...),
                'c' for (..., lat, lon), or 'auto' to infer from shape.
            lat_order: Latitude coordinate direction - 'north_to_south'/'N2S',
                'south_to_north'/'S2N', or 'auto'.
            lat: Optional 1D latitude array of length nlat used to auto-detect
                south-to-north ordering without reversing the caller's array.

        Raises:
            ValidationError: If input parameters are invalid
            SpheremackError: If Rust engine initialization fails
        """
        self._validate_inputs(nlon, nlat, rsphere, gridtype, legfunc)

        self.nlon = int(nlon)
        self.nlat = int(nlat)
        self.rsphere = float(rsphere)
        self.gridtype = gridtype
        self.legfunc = legfunc
        self.precision = self._validate_precision(precision)
        self.layout = _validate_layout(layout)
        self.lat_order, self.south_to_north = _resolve_lat_order(
            self.nlat, lat_order, lat
        )

        self._core = _call_backend(
            _spharm_rs.RustSpharmCore,
            self.nlon,
            self.nlat,
            self.rsphere,
            self.gridtype,
            self.legfunc,
            operation_name="RustSpharmCore initialization",
        )

    def _validate_inputs(
        self, nlon: int, nlat: int, rsphere: float, gridtype: str, legfunc: str
    ) -> None:
        """Validate all constructor parameters with descriptive error messages."""
        validators = [
            (rsphere > 0.0, f"rsphere must be positive, got {rsphere}"),
            (nlon >= MIN_NLON, f"nlon must be >= {MIN_NLON}, got {nlon}"),
            (nlat >= MIN_NLAT, f"nlat must be >= {MIN_NLAT}, got {nlat}"),
            (
                gridtype in ("regular", "gaussian"),
                f'gridtype must be "regular" or "gaussian", got "{gridtype}"',
            ),
            (
                legfunc in ("stored", "computed"),
                f'legfunc must be "stored" or "computed", got "{legfunc}"',
            ),
        ]

        for condition, error_msg in validators:
            if not condition:
                raise ValidationError(error_msg)

    @staticmethod
    def _validate_precision(precision: str) -> str:
        """Validate the requested public-output precision mode."""
        return _validate_precision(precision)

    @staticmethod
    def _input_public_real_dtype(array: np.ndarray) -> np.dtype:
        """Map an input array dtype to the matching public real dtype."""
        return _input_public_real_dtype(array)

    def _public_real_dtype(self, *arrays: np.ndarray) -> np.dtype:
        """Resolve the public real output dtype for this instance."""
        return _resolve_public_real_dtype(getattr(self, "precision", "auto"), *arrays)

    def _public_complex_dtype(self, *arrays: np.ndarray) -> np.dtype:
        """Resolve the public complex output dtype for this instance."""
        return _resolve_public_complex_dtype(getattr(self, "precision", "auto"), *arrays)

    def __setattr__(self, key: str, val: Any) -> None:
        """Prevent modification of read-only instance variables."""
        if key in self.__dict__ and key in _private_vars:
            raise AttributeError(f"Attempt to rebind read-only instance variable {key}")
        self.__dict__[key] = val

    def __delattr__(self, key: str) -> None:
        """Prevent deletion of read-only instance variables."""
        if key in self.__dict__ and key in _private_vars:
            raise AttributeError(f"Attempt to unbind read-only instance variable {key}")
        del self.__dict__[key]

    def __repr__(self) -> str:
        """Return string representation of Spharmt instance."""
        return (
            f"{type(self).__name__}({self.nlon:d}, {self.nlat:d}, "
            f"{self.rsphere:e}, '{self.gridtype}', '{self.legfunc}')"
        )

    def _resolve_grid_layout(
        self, data: np.ndarray, operation_name: str
    ) -> Tuple[str, Tuple[int, ...]]:
        """Validate grid rank/dimensions and resolve 'c' vs 'legacy' layout."""
        if data.ndim < 2:
            raise ValidationError(
                f"{operation_name} needs at least a rank 2 array, got rank {data.ndim}"
            )

        layout = getattr(self, "layout", "auto")
        if layout == "c":
            if data.shape[-2:] != (self.nlat, self.nlon):
                raise ValidationError(
                    f"{operation_name} needs trailing grid shape ({self.nlat}, {self.nlon}), "
                    f"got {data.shape}"
                )
            return "c", data.shape[:-2]

        if layout == "legacy":
            if data.shape[:2] != (self.nlat, self.nlon):
                raise ValidationError(
                    f"{operation_name} needs an array of size {self.nlat} by {self.nlon}, "
                    f"got {data.shape[0]} by {data.shape[1]}"
                )
            return "legacy", data.shape[2:]

        # layout == "auto"
        if data.ndim == 2:
            if data.shape != (self.nlat, self.nlon):
                raise ValidationError(
                    f"{operation_name} needs an array of size {self.nlat} by {self.nlon}, "
                    f"got {data.shape[0]} by {data.shape[1]}"
                )
            return "c", ()

        is_c = data.shape[-2:] == (self.nlat, self.nlon)
        is_legacy = data.shape[:2] == (self.nlat, self.nlon)
        if is_c and not is_legacy:
            return "c", data.shape[:-2]
        if is_legacy and not is_c:
            return "legacy", data.shape[2:]
        if is_c and is_legacy:
            if data.flags.f_contiguous and not data.flags.c_contiguous:
                return "legacy", data.shape[2:]
            return "c", data.shape[:-2]

        raise ValidationError(
            f"{operation_name} needs an array of size {self.nlat} by {self.nlon}, "
            f"got {data.shape}"
        )

    def _prepare_grid_input(
        self,
        data: FloatArray,
        operation_name: str,
        target_dtype: Optional[np.dtype] = None,
    ) -> Tuple[str, int, FloatArray, Tuple[int, ...]]:
        """Normalize grid input to C-contiguous (nt, nlat, nlon) without extra copies."""
        arr = data if type(data) is np.ndarray else np.asarray(data)
        resolved_layout, batch_shape = self._resolve_grid_layout(arr, operation_name)
        compute_dtype = (
            self._public_real_dtype(arr)
            if target_dtype is None
            else (target_dtype if type(target_dtype) is np.dtype else np.dtype(target_dtype))
        )
        nt = 1 if not batch_shape else math.prod(batch_shape)

        if resolved_layout == "c":
            reshaped = arr.reshape(nt, self.nlat, self.nlon)
            if reshaped.dtype == compute_dtype and reshaped.flags.c_contiguous:
                c_3d = reshaped
            else:
                c_3d = np.ascontiguousarray(reshaped, dtype=compute_dtype)
        else:
            reshaped = arr.reshape(self.nlat, self.nlon, nt)
            c_3d = np.ascontiguousarray(
                np.moveaxis(reshaped, -1, 0), dtype=compute_dtype
            )

        return resolved_layout, nt, c_3d, batch_shape

    def _prepare_spectral_input(
        self,
        data: ComplexArray,
        operation_name: str,
        target_dtype: Optional[np.dtype] = None,
    ) -> Tuple[str, int, int, ComplexArray, Tuple[int, ...]]:
        """Normalize spectral input to C-contiguous (nt, nmdim)."""
        arr = data if type(data) is np.ndarray else np.asarray(data)
        resolved_layout, ntrunc, batch_shape = _resolve_spectral_layout(
            arr, getattr(self, "layout", "auto"), self.nlat - 1, operation_name
        )
        compute_dtype = (
            self._public_complex_dtype(arr)
            if target_dtype is None
            else (target_dtype if type(target_dtype) is np.dtype else np.dtype(target_dtype))
        )
        nt = 1 if not batch_shape else math.prod(batch_shape)

        if arr.ndim == 1:
            reshaped = arr.reshape(1, arr.shape[0])
            c_2d = (
                reshaped
                if reshaped.dtype == compute_dtype and reshaped.flags.c_contiguous
                else np.ascontiguousarray(reshaped, dtype=compute_dtype)
            )
        elif resolved_layout == "c":
            reshaped = arr.reshape(nt, arr.shape[-1])
            c_2d = (
                reshaped
                if reshaped.dtype == compute_dtype and reshaped.flags.c_contiguous
                else np.ascontiguousarray(reshaped, dtype=compute_dtype)
            )
        else:
            c_2d = np.ascontiguousarray(
                np.moveaxis(arr.reshape(arr.shape[0], nt), 0, -1),
                dtype=compute_dtype,
            )

        return resolved_layout, nt, ntrunc, c_2d, batch_shape

    def _prepare_paired_grid_inputs(
        self, ugrid: FloatArray, vgrid: FloatArray, operation_name: str
    ) -> Tuple[str, int, FloatArray, FloatArray, Tuple[int, ...]]:
        """Validate and normalize a pair of grid arrays with shared compute dtype."""
        u_arr = np.asarray(ugrid)
        v_arr = np.asarray(vgrid)
        if u_arr.shape != v_arr.shape:
            raise ValidationError("ugrid and vgrid must have the same shape")

        compute_dtype = self._public_real_dtype(u_arr, v_arr)
        layout_u, nt_u, u_3d, batch_u = self._prepare_grid_input(
            u_arr, operation_name, target_dtype=compute_dtype
        )
        layout_v, nt_v, v_3d, batch_v = self._prepare_grid_input(
            v_arr, operation_name, target_dtype=compute_dtype
        )
        if layout_u != layout_v or nt_u != nt_v or batch_u != batch_v:
            raise ValidationError("ugrid and vgrid must have consistent dimensions")
        return layout_u, nt_u, u_3d, v_3d, batch_u

    def _prepare_paired_spectral_inputs(
        self, spec_a: ComplexArray, spec_b: ComplexArray, operation_name: str
    ) -> Tuple[str, int, int, ComplexArray, ComplexArray, Tuple[int, ...]]:
        """Validate and normalize a pair of spectral arrays with shared compute dtype."""
        a_arr = np.asarray(spec_a)
        b_arr = np.asarray(spec_b)
        if a_arr.shape != b_arr.shape:
            raise ValidationError("paired spectra must have the same shape")

        compute_dtype = self._public_complex_dtype(a_arr, b_arr)
        layout_a, nt_a, ntrunc_a, a_2d, batch_a = self._prepare_spectral_input(
            a_arr, operation_name, target_dtype=compute_dtype
        )
        layout_b, nt_b, ntrunc_b, b_2d, batch_b = self._prepare_spectral_input(
            b_arr, operation_name, target_dtype=compute_dtype
        )
        if (
            layout_a != layout_b
            or nt_a != nt_b
            or ntrunc_a != ntrunc_b
            or batch_a != batch_b
        ):
            raise ValidationError("paired spectra must have consistent dimensions")
        return layout_a, nt_a, ntrunc_a, a_2d, b_2d, batch_a

    def _restore_grid_output(
        self,
        grid_3d: FloatArray,
        resolved_layout: str,
        batch_shape: Tuple[int, ...],
        output_dtype: Optional[np.dtype] = None,
    ) -> FloatArray:
        """Restore (nt, nlat, nlon) Rust output to caller's shape and dtype."""
        array = np.asarray(grid_3d)
        if output_dtype is not None and array.dtype != np.dtype(output_dtype):
            array = array.astype(np.dtype(output_dtype), copy=False)
        if not batch_shape:
            return array[0]
        if resolved_layout == "c":
            return array.reshape(batch_shape + (self.nlat, self.nlon))
        return np.moveaxis(array, 0, -1).reshape((self.nlat, self.nlon) + batch_shape)

    def _restore_spectral_output(
        self,
        spec_2d: ComplexArray,
        resolved_layout: str,
        batch_shape: Tuple[int, ...],
        output_dtype: Optional[np.dtype] = None,
    ) -> ComplexArray:
        """Restore (nt, nmdim) Rust output to caller's shape and dtype."""
        array = np.asarray(spec_2d)
        if output_dtype is not None and array.dtype != np.dtype(output_dtype):
            array = array.astype(np.dtype(output_dtype), copy=False)
        if not batch_shape:
            return array[0]
        if resolved_layout == "c":
            return array.reshape(batch_shape + (array.shape[1],))
        return np.moveaxis(array, 0, -1).reshape((array.shape[1],) + batch_shape)

    def _validate_ntrunc(self, ntrunc: Optional[int], max_allowed: int) -> int:
        """Validate and return appropriate ntrunc value."""
        if ntrunc is None:
            return self.nlat - 1
        ntrunc = int(ntrunc)
        if ntrunc < 0 or ntrunc > max_allowed:
            raise ValidationError(
                f"ntrunc must be between 0 and {max_allowed}, got {ntrunc}"
            )
        return ntrunc

    # Legacy internal validation/restoration compatibility helpers
    def _validate_grid_data(
        self, data: FloatArray, operation_name: str
    ) -> Tuple[int, FloatArray, Tuple[int, ...]]:
        """Validate grid data and return flattened C-order (nt, nlat, nlon) array."""
        _, nt, c_3d, batch_shape = self._prepare_grid_input(data, operation_name)
        return nt, c_3d, batch_shape

    def _validate_spectral_data(
        self, data: ComplexArray, operation_name: str
    ) -> Tuple[int, int, ComplexArray, Tuple[int, ...]]:
        """Validate spectral data and return flattened C-order (nt, nmdim) array."""
        _, nt, ntrunc, c_2d, batch_shape = self._prepare_spectral_input(
            data, operation_name
        )
        return nt, ntrunc, c_2d, batch_shape

    def _restore_grid_shape(
        self,
        datagrid: FloatArray,
        extra_shape: Tuple[int, ...],
        output_dtype: Optional[np.dtype] = None,
    ) -> FloatArray:
        """Restore flattened grid fields according to instance layout."""
        mode = "c" if getattr(self, "layout", "auto") == "c" else "legacy"
        return self._restore_grid_output(datagrid, mode, extra_shape, output_dtype)

    def _restore_spectral_shape(
        self,
        dataspec: ComplexArray,
        extra_shape: Tuple[int, ...],
        output_dtype: Optional[np.dtype] = None,
    ) -> ComplexArray:
        """Restore flattened spectral fields according to instance layout."""
        mode = "c" if getattr(self, "layout", "auto") == "c" else "legacy"
        return self._restore_spectral_output(dataspec, mode, extra_shape, output_dtype)

    def _spectogrd_pair(
        self, spec_a: ComplexArray, spec_b: ComplexArray
    ) -> Tuple[FloatArray, FloatArray]:
        """Synthesize two scalar spectra together through one batched call."""
        resolved_layout, nt, _, a_2d, b_2d, batch_shape = (
            self._prepare_paired_spectral_inputs(spec_a, spec_b, "_spectogrd_pair")
        )
        if nt == 0:
            real_dtype = np.dtype(
                np.float64 if a_2d.dtype == np.complex128 else np.float32
            )
            empty_3d = np.empty((0, self.nlat, self.nlon), dtype=real_dtype)
            return (
                self._restore_grid_output(empty_3d, resolved_layout, batch_shape),
                self._restore_grid_output(empty_3d.copy(), resolved_layout, batch_shape),
            )

        stacked = np.concatenate((a_2d, b_2d), axis=0)
        grids = _call_backend(
            self._core.spectogrd,
            stacked,
            south_to_north=self.south_to_north,
            operation_name="spectogrd_pair transform",
        )
        return (
            self._restore_grid_output(grids[:nt], resolved_layout, batch_shape),
            self._restore_grid_output(grids[nt:], resolved_layout, batch_shape),
        )

    def _invlapspec(
        self, dataspec: ComplexArray, operation_name: str = "_invlapspec"
    ) -> ComplexArray:
        """Apply the inverse Laplacian to a scalar spectrum and preserve shape."""
        resolved_layout, nt, _, spec_2d, batch_shape = self._prepare_spectral_input(
            dataspec, operation_name
        )
        if nt == 0:
            return self._restore_spectral_output(spec_2d, resolved_layout, batch_shape)
        invlap_2d = _call_backend(
            _spharm_rs.invlap,
            spec_2d,
            self.rsphere,
            operation_name="invlap",
        )
        return self._restore_spectral_output(invlap_2d, resolved_layout, batch_shape)

    def invlapspec(self, dataspec: ComplexArray) -> ComplexArray:
        """Apply the inverse spherical Laplacian to spectral coefficients."""
        return self._invlapspec(dataspec, "invlapspec")

    def invlap(self, dataspec: ComplexArray) -> ComplexArray:
        """Apply the inverse spherical Laplacian to spectral coefficients."""
        return self._invlapspec(dataspec, "invlap")

    def _getuv_spec_component(
        self, dataspec: ComplexArray, component: str, operation_name: str
    ) -> Tuple[FloatArray, FloatArray]:
        """Synthesize wind from a single vorticity or divergence spectral field."""
        if component not in ("div", "vrt"):
            raise ValidationError(
                f'{operation_name} invalid component "{component}"; expected "div" or "vrt"'
            )

        resolved_layout, nt, _, spec_2d, batch_shape = self._prepare_spectral_input(
            dataspec, operation_name
        )
        if nt == 0:
            real_dtype = np.dtype(
                np.float64 if spec_2d.dtype == np.complex128 else np.float32
            )
            empty_3d = np.empty((0, self.nlat, self.nlon), dtype=real_dtype)
            return (
                self._restore_grid_output(empty_3d, resolved_layout, batch_shape),
                self._restore_grid_output(empty_3d.copy(), resolved_layout, batch_shape),
            )

        zeros = np.zeros_like(spec_2d)
        if component == "div":
            u_3d, v_3d = _call_backend(
                self._core.getuv,
                zeros,
                spec_2d,
                south_to_north=self.south_to_north,
                operation_name=f"{operation_name} divergence synthesis",
            )
        else:
            u_3d, v_3d = _call_backend(
                self._core.getuv,
                spec_2d,
                zeros,
                south_to_north=self.south_to_north,
                operation_name=f"{operation_name} vorticity synthesis",
            )
        return (
            self._restore_grid_output(u_3d, resolved_layout, batch_shape),
            self._restore_grid_output(v_3d, resolved_layout, batch_shape),
        )

    def grdtospec(
        self, datagrid: FloatArray, ntrunc: Optional[int] = None
    ) -> ComplexArray:
        """
        Grid to spectral transform (spherical harmonic analysis).

        Args:
            datagrid: Grid data with shape (..., nlat, nlon) or (nlat, nlon, *extra_dims)
            ntrunc: Optional spectral truncation limit (default: nlat - 1)

        Returns:
            Complex spherical harmonic coefficients preserving native precision and layout
        """
        resolved_layout, nt, grid_3d, batch_shape = self._prepare_grid_input(
            datagrid, "grdtospec"
        )
        ntrunc = self._validate_ntrunc(ntrunc, self.nlat - 1)
        if nt == 0:
            nmdim = (ntrunc + 1) * (ntrunc + 2) // 2
            cdtype = np.dtype(
                np.complex128 if grid_3d.dtype == np.float64 else np.complex64
            )
            return self._restore_spectral_output(
                np.empty((0, nmdim), dtype=cdtype), resolved_layout, batch_shape
            )

        spec_2d = _call_backend(
            self._core.grdtospec,
            grid_3d,
            ntrunc=ntrunc,
            south_to_north=self.south_to_north,
            operation_name="grdtospec transform",
        )
        return self._restore_spectral_output(spec_2d, resolved_layout, batch_shape)

    def spectogrd(self, dataspec: ComplexArray) -> FloatArray:
        """
        Spectral to grid transform (spherical harmonic synthesis).

        Args:
            dataspec: Complex spectral coefficients

        Returns:
            Grid data preserving native precision and layout
        """
        resolved_layout, nt, _, spec_2d, batch_shape = self._prepare_spectral_input(
            dataspec, "spectogrd"
        )
        if nt == 0:
            rdtype = np.dtype(
                np.float64 if spec_2d.dtype == np.complex128 else np.float32
            )
            return self._restore_grid_output(
                np.empty((0, self.nlat, self.nlon), dtype=rdtype),
                resolved_layout,
                batch_shape,
            )

        grid_3d = _call_backend(
            self._core.spectogrd,
            spec_2d,
            south_to_north=self.south_to_north,
            operation_name="spectogrd transform",
        )
        return self._restore_grid_output(grid_3d, resolved_layout, batch_shape)

    def getvrtdivspec(
        self, ugrid: FloatArray, vgrid: FloatArray, ntrunc: Optional[int] = None
    ) -> Tuple[ComplexArray, ComplexArray]:
        """Compute spectral coefficients of vorticity and divergence from vector wind."""
        resolved_layout, nt, u_3d, v_3d, batch_shape = self._prepare_paired_grid_inputs(
            ugrid, vgrid, "getvrtdivspec"
        )
        ntrunc = self._validate_ntrunc(ntrunc, self.nlat - 1)
        if nt == 0:
            nmdim = (ntrunc + 1) * (ntrunc + 2) // 2
            cdtype = np.dtype(
                np.complex128 if u_3d.dtype == np.float64 else np.complex64
            )
            empty = np.empty((0, nmdim), dtype=cdtype)
            return (
                self._restore_spectral_output(empty, resolved_layout, batch_shape),
                self._restore_spectral_output(empty.copy(), resolved_layout, batch_shape),
            )

        vrt_2d, div_2d = _call_backend(
            self._core.getvrtdivspec,
            u_3d,
            v_3d,
            ntrunc=ntrunc,
            south_to_north=self.south_to_north,
            operation_name="getvrtdivspec transform",
        )
        return (
            self._restore_spectral_output(vrt_2d, resolved_layout, batch_shape),
            self._restore_spectral_output(div_2d, resolved_layout, batch_shape),
        )

    def getvrtspec(
        self, ugrid: FloatArray, vgrid: FloatArray, ntrunc: Optional[int] = None
    ) -> ComplexArray:
        """Compute only vorticity spectral coefficients from vector wind."""
        resolved_layout, nt, u_3d, v_3d, batch_shape = self._prepare_paired_grid_inputs(
            ugrid, vgrid, "getvrtspec"
        )
        ntrunc = self._validate_ntrunc(ntrunc, self.nlat - 1)
        if nt == 0:
            nmdim = (ntrunc + 1) * (ntrunc + 2) // 2
            cdtype = np.dtype(
                np.complex128 if u_3d.dtype == np.float64 else np.complex64
            )
            return self._restore_spectral_output(
                np.empty((0, nmdim), dtype=cdtype), resolved_layout, batch_shape
            )

        vrt_2d = _call_backend(
            self._core.getvrtspec,
            u_3d,
            v_3d,
            ntrunc=ntrunc,
            south_to_north=self.south_to_north,
            operation_name="getvrtspec transform",
        )
        return self._restore_spectral_output(vrt_2d, resolved_layout, batch_shape)

    def getdivspec(
        self, ugrid: FloatArray, vgrid: FloatArray, ntrunc: Optional[int] = None
    ) -> ComplexArray:
        """Compute only divergence spectral coefficients from vector wind."""
        resolved_layout, nt, u_3d, v_3d, batch_shape = self._prepare_paired_grid_inputs(
            ugrid, vgrid, "getdivspec"
        )
        ntrunc = self._validate_ntrunc(ntrunc, self.nlat - 1)
        if nt == 0:
            nmdim = (ntrunc + 1) * (ntrunc + 2) // 2
            cdtype = np.dtype(
                np.complex128 if u_3d.dtype == np.float64 else np.complex64
            )
            return self._restore_spectral_output(
                np.empty((0, nmdim), dtype=cdtype), resolved_layout, batch_shape
            )

        div_2d = _call_backend(
            self._core.getdivspec,
            u_3d,
            v_3d,
            ntrunc=ntrunc,
            south_to_north=self.south_to_north,
            operation_name="getdivspec transform",
        )
        return self._restore_spectral_output(div_2d, resolved_layout, batch_shape)

    def getuv(
        self, vrtspec: ComplexArray, divspec: ComplexArray
    ) -> Tuple[FloatArray, FloatArray]:
        """Compute vector wind from spectral coefficients of vorticity and divergence."""
        if np.shape(vrtspec) != np.shape(divspec):
            raise ValidationError("vrtspec and divspec must have the same shape")
        resolved_layout, nt, _, vrt_2d, div_2d, batch_shape = (
            self._prepare_paired_spectral_inputs(vrtspec, divspec, "getuv")
        )
        if nt == 0:
            rdtype = np.dtype(
                np.float64 if vrt_2d.dtype == np.complex128 else np.float32
            )
            empty_3d = np.empty((0, self.nlat, self.nlon), dtype=rdtype)
            return (
                self._restore_grid_output(empty_3d, resolved_layout, batch_shape),
                self._restore_grid_output(empty_3d.copy(), resolved_layout, batch_shape),
            )

        u_3d, v_3d = _call_backend(
            self._core.getuv,
            vrt_2d,
            div_2d,
            south_to_north=self.south_to_north,
            operation_name="getuv transform",
        )
        return (
            self._restore_grid_output(u_3d, resolved_layout, batch_shape),
            self._restore_grid_output(v_3d, resolved_layout, batch_shape),
        )

    def getpsichi_component(
        self,
        ugrid: FloatArray,
        vgrid: FloatArray,
        ntrunc: Optional[int],
        component: str,
        operation_name: str,
    ) -> FloatArray:
        """Compute one streamfunction or velocity-potential grid component."""
        fieldspec = self.getpsichi_spec_component(
            ugrid, vgrid, ntrunc, component, operation_name
        )
        return self.spectogrd(fieldspec)

    def getpsichi_spec_component(
        self,
        ugrid: FloatArray,
        vgrid: FloatArray,
        ntrunc: Optional[int],
        component: str,
        operation_name: str,
    ) -> ComplexArray:
        """Compute one streamfunction or velocity-potential spectrum."""
        if component not in ("psi", "chi"):
            raise ValidationError(
                f'{operation_name} invalid component "{component}"; expected "psi" or "chi"'
            )
        source_spec = (
            self.getvrtspec(ugrid, vgrid, ntrunc)
            if component == "psi"
            else self.getdivspec(ugrid, vgrid, ntrunc)
        )
        return self._invlapspec(source_spec, operation_name)

    def getpsispec(
        self, ugrid: FloatArray, vgrid: FloatArray, ntrunc: Optional[int] = None
    ) -> ComplexArray:
        """Compute streamfunction spectral coefficients without gridding."""
        return self.getpsichi_spec_component(
            ugrid, vgrid, ntrunc, component="psi", operation_name="getpsispec"
        )

    def getchispec(
        self, ugrid: FloatArray, vgrid: FloatArray, ntrunc: Optional[int] = None
    ) -> ComplexArray:
        """Compute velocity-potential spectral coefficients without gridding."""
        return self.getpsichi_spec_component(
            ugrid, vgrid, ntrunc, component="chi", operation_name="getchispec"
        )

    def getpsichispec(
        self, ugrid: FloatArray, vgrid: FloatArray, ntrunc: Optional[int] = None
    ) -> Tuple[ComplexArray, ComplexArray]:
        """Compute streamfunction and velocity-potential spectra."""
        vrtspec, divspec = self.getvrtdivspec(ugrid, vgrid, ntrunc)
        if np.shape(vrtspec) != np.shape(divspec):
            raise ValidationError(
                "vorticity and divergence spectra have inconsistent dimensions"
            )
        return (
            self._invlapspec(vrtspec, "getpsichispec"),
            self._invlapspec(divspec, "getpsichispec"),
        )

    def getpsi(
        self, ugrid: FloatArray, vgrid: FloatArray, ntrunc: Optional[int] = None
    ) -> FloatArray:
        """Compute streamfunction from vector wind without materializing chi."""
        return self.getpsichi_component(
            ugrid, vgrid, ntrunc, component="psi", operation_name="getpsi"
        )

    def getchi(
        self, ugrid: FloatArray, vgrid: FloatArray, ntrunc: Optional[int] = None
    ) -> FloatArray:
        """Compute velocity potential from vector wind without materializing psi."""
        return self.getpsichi_component(
            ugrid, vgrid, ntrunc, component="chi", operation_name="getchi"
        )

    def getpsichi(
        self, ugrid: FloatArray, vgrid: FloatArray, ntrunc: Optional[int] = None
    ) -> Tuple[FloatArray, FloatArray]:
        """Compute streamfunction and velocity potential from vector wind."""
        resolved_layout, nt, u_3d, v_3d, batch_shape = self._prepare_paired_grid_inputs(
            ugrid, vgrid, "getpsichi"
        )
        ntrunc = self._validate_ntrunc(ntrunc, self.nlat - 1)
        if nt == 0:
            empty = np.empty((0, self.nlat, self.nlon), dtype=u_3d.dtype)
            return (
                self._restore_grid_output(empty, resolved_layout, batch_shape),
                self._restore_grid_output(empty.copy(), resolved_layout, batch_shape),
            )

        psi_3d, chi_3d = _call_backend(
            self._core.getpsichi,
            u_3d,
            v_3d,
            ntrunc=ntrunc,
            south_to_north=self.south_to_north,
            operation_name="getpsichi transform",
        )
        return (
            self._restore_grid_output(psi_3d, resolved_layout, batch_shape),
            self._restore_grid_output(chi_3d, resolved_layout, batch_shape),
        )

    def getgrad(self, chispec: ComplexArray) -> Tuple[FloatArray, FloatArray]:
        """Compute vector gradient from spectral coefficients."""
        resolved_layout, nt, _, spec_2d, batch_shape = self._prepare_spectral_input(
            chispec, "getgrad"
        )
        if nt == 0:
            rdtype = np.dtype(
                np.float64 if spec_2d.dtype == np.complex128 else np.float32
            )
            empty_3d = np.empty((0, self.nlat, self.nlon), dtype=rdtype)
            return (
                self._restore_grid_output(empty_3d, resolved_layout, batch_shape),
                self._restore_grid_output(empty_3d.copy(), resolved_layout, batch_shape),
            )

        u_3d, v_3d = _call_backend(
            self._core.getgrad,
            spec_2d,
            south_to_north=self.south_to_north,
            operation_name="getgrad transform",
        )
        return (
            self._restore_grid_output(u_3d, resolved_layout, batch_shape),
            self._restore_grid_output(v_3d, resolved_layout, batch_shape),
        )

    def specsmooth(self, datagrid: FloatArray, smooth: FloatArray) -> FloatArray:
        """Apply isotropic spectral smoothing to grid data."""
        resolved_layout, nt, grid_3d, batch_shape = self._prepare_grid_input(
            datagrid, "specsmooth"
        )
        smooth_arr = np.asarray(smooth)
        if smooth_arr.ndim != 1 or smooth_arr.shape[0] != self.nlat:
            raise ValidationError(
                f"smooth must be rank 1 with size {self.nlat}, got shape {smooth_arr.shape}"
            )
        if nt == 0:
            return self._restore_grid_output(grid_3d, resolved_layout, batch_shape)

        spec_2d = _call_backend(
            self._core.grdtospec,
            grid_3d,
            ntrunc=self.nlat - 1,
            south_to_north=self.south_to_north,
            operation_name="specsmooth grdtospec",
        )
        smooth_vec = np.ascontiguousarray(smooth_arr, dtype=grid_3d.dtype)
        smoothed_2d = _call_backend(
            _spharm_rs.multsmoothfact,
            spec_2d,
            smooth_vec,
            operation_name="specsmooth multsmoothfact",
        )
        out_3d = _call_backend(
            self._core.spectogrd,
            smoothed_2d,
            south_to_north=self.south_to_north,
            operation_name="specsmooth spectogrd",
        )
        return self._restore_grid_output(out_3d, resolved_layout, batch_shape)


class SpharmTransform(Spharmt):
    """Modern C-order spherical harmonic transform class (defaults to ``layout="c"``)."""

    def __init__(
        self,
        nlon: int,
        nlat: int,
        rsphere: float = DEFAULT_EARTH_RADIUS,
        gridtype: GridType = "regular",
        legfunc: LegendreFunc = "stored",
        precision: PrecisionMode = "auto",
        layout: LayoutMode = "c",
        lat_order: LatOrderMode = "auto",
        lat: Optional[FloatArray] = None,
    ) -> None:
        super().__init__(
            nlon=nlon,
            nlat=nlat,
            rsphere=rsphere,
            gridtype=gridtype,
            legfunc=legfunc,
            precision=precision,
            layout=layout,
            lat_order=lat_order,
            lat=lat,
        )


# Standalone functions


def regrid(
    grdin: Spharmt,
    grdout: Spharmt,
    datagrid: FloatArray,
    ntrunc: Optional[int] = None,
    smooth: Optional[FloatArray] = None,
) -> FloatArray:
    """Regrid data using spectral interpolation with optional smoothing and truncation."""
    datagrid_arr = np.asarray(datagrid)

    if datagrid_arr.ndim < 2:
        raise ValidationError(
            f"regrid needs at least a rank 2 array, got rank {datagrid_arr.ndim}"
        )

    # Validate horizontal shape according to grdin.layout
    grdin._resolve_grid_layout(datagrid_arr, "regrid")

    if smooth is not None:
        smooth_arr = np.asarray(smooth)
        if smooth_arr.ndim != 1 or smooth_arr.shape[0] != grdout.nlat:
            raise ValidationError("smooth must be rank 1 with size grdout.nlat")
    else:
        smooth_arr = None

    if ntrunc is None:
        ntrunc = min(grdout.nlat - 1, grdin.nlat - 1)

    dataspec = grdin.grdtospec(datagrid_arr, ntrunc)

    if smooth_arr is not None:
        resolved_layout, nt, _, spec_2d, batch_shape = grdin._prepare_spectral_input(
            dataspec, "regrid"
        )
        if nt > 0:
            smooth_dtype = np.float64 if spec_2d.dtype == np.complex128 else np.float32
            smoothed_2d = _call_backend(
                _spharm_rs.multsmoothfact,
                spec_2d,
                np.ascontiguousarray(smooth_arr, dtype=smooth_dtype),
                operation_name="regrid multsmoothfact",
            )
            dataspec = grdin._restore_spectral_output(
                smoothed_2d, resolved_layout, batch_shape
            )

    return grdout.spectogrd(dataspec)


def invlap(
    dataspec: ComplexArray,
    rsphere: float = DEFAULT_EARTH_RADIUS,
    layout: LayoutMode = "auto",
) -> ComplexArray:
    """Apply the inverse spherical Laplacian to triangular spectral coefficients.

    Scales degree-``n`` modes by ``-rsphere**2 / (n * (n + 1))`` and zeroes ``n = 0``.
    Supports 1D ``(nmdim,)``, C-order ``(..., nmdim)``, and legacy ``(nmdim, ...)``
    arrays while preserving ``complex64``/``complex128`` precision and input layout.
    """
    try:
        rsphere_val = float(rsphere)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"rsphere must be a positive number, got {rsphere!r}") from exc
    if not math.isfinite(rsphere_val) or rsphere_val <= 0.0:
        raise ValidationError(f"rsphere must be positive, got {rsphere}")

    norm_layout = _validate_layout(layout)
    arr = dataspec if type(dataspec) is np.ndarray else np.asarray(dataspec)
    resolved_layout, _, batch_shape = _resolve_spectral_layout(
        arr, norm_layout, sys.maxsize, "invlap"
    )
    compute_dtype = _resolve_public_complex_dtype("auto", arr)
    nt = 1 if not batch_shape else math.prod(batch_shape)

    if arr.ndim == 1:
        reshaped = arr.reshape(1, arr.shape[0])
        c_2d = (
            reshaped
            if reshaped.dtype == compute_dtype and reshaped.flags.c_contiguous
            else np.ascontiguousarray(reshaped, dtype=compute_dtype)
        )
    elif resolved_layout == "c":
        reshaped = arr.reshape(nt, arr.shape[-1])
        c_2d = (
            reshaped
            if reshaped.dtype == compute_dtype and reshaped.flags.c_contiguous
            else np.ascontiguousarray(reshaped, dtype=compute_dtype)
        )
    else:
        c_2d = np.ascontiguousarray(
            np.moveaxis(arr.reshape(arr.shape[0], nt), 0, -1),
            dtype=compute_dtype,
        )

    if nt == 0:
        out_2d = c_2d.copy()
    else:
        out_2d = np.asarray(
            _call_backend(
                _spharm_rs.invlap,
                c_2d,
                rsphere_val,
                operation_name="invlap",
            )
        )

    if not batch_shape:
        return out_2d[0]
    if resolved_layout == "c":
        return out_2d.reshape(batch_shape + (out_2d.shape[1],))
    return np.moveaxis(out_2d, 0, -1).reshape((out_2d.shape[1],) + batch_shape)


def gaussian_lats_wts(nlat: int) -> Tuple[FloatArray, FloatArray]:
    """Compute Gaussian latitudes (degrees north, N2S) and quadrature weights."""
    try:
        lats, wts = _spharm_rs.gaussian_lats_wts(int(nlat))
    except Exception as exc:
        raise SpheremackError(f"Failed to compute Gaussian latitudes: {exc}") from exc
    return np.asarray(lats, dtype=np.float64), np.asarray(wts, dtype=np.float64)


def getspecindx(ntrunc: int) -> Tuple[IntArray, IntArray]:
    """Compute indices of zonal wavenumber and degree for spherical harmonic coefficients."""
    if ntrunc < 0:
        raise ValidationError(f"ntrunc must be non-negative, got {ntrunc}")

    nmdim = (ntrunc + 1) * (ntrunc + 2) // 2
    indxm = np.empty(nmdim, dtype=np.intp)
    indxn = np.empty(nmdim, dtype=np.intp)
    pos = 0
    for m in range(ntrunc + 1):
        cnt = ntrunc + 1 - m
        indxm[pos : pos + cnt] = m
        indxn[pos : pos + cnt] = np.arange(m, ntrunc + 1, dtype=np.intp)
        pos += cnt

    return indxm, indxn


def getgeodesicpts(m: int) -> Tuple[FloatArray, FloatArray]:
    """Compute lat/lon values (degrees) for icosahedral geodesic points."""
    if m < 2:
        raise ValidationError(f"m must be >= 2, got {m}")

    try:
        lats, lons = _spharm_rs.ihgeod(int(m))
    except Exception as exc:
        raise SpheremackError(f"Failed to compute geodesic points: {exc}") from exc

    return np.asarray(lats, dtype=np.float32), np.asarray(lons, dtype=np.float32)


def legendre(lat: float, ntrunc: int) -> FloatArray:
    """Calculate associated Legendre functions for triangular truncation."""
    if not -90.0 <= lat <= 90.0:
        raise ValidationError(f"lat must be between -90 and 90 degrees, got {lat}")

    if ntrunc < 0:
        raise ValidationError(f"ntrunc must be non-negative, got {ntrunc}")

    try:
        return np.asarray(_spharm_rs.getlegfunc(float(lat), int(ntrunc)))
    except Exception as exc:
        raise SpheremackError(f"Failed to compute Legendre functions: {exc}") from exc


def specintrp(
    lon: float, dataspec: ComplexArray, legfuncs: FloatArray
) -> Union[float, complex]:
    """Spectral interpolation to arbitrary point on sphere."""
    spec_arr = np.asarray(dataspec)
    leg_arr = np.asarray(legfuncs)

    ntrunc1 = int(-1.5 + 0.5 * math.sqrt(9.0 - 8.0 * (1.0 - spec_arr.shape[0])))
    ntrunc2 = int(-1.5 + 0.5 * math.sqrt(9.0 - 8.0 * (1.0 - leg_arr.shape[0])))

    if ntrunc1 != ntrunc2:
        raise ValidationError(
            f"dataspec and legfuncs have inconsistent spectral truncations: {ntrunc1} vs {ntrunc2}"
        )

    use_double = (
        spec_arr.dtype == np.complex128
        or leg_arr.dtype == np.float64
        or spec_arr.dtype not in (np.complex64, np.complex128)
    )
    spec_dtype = np.complex128 if use_double else np.complex64
    leg_dtype = np.float64 if use_double else np.float32

    try:
        return _spharm_rs.specintrp(
            float(lon) * DEGREES_TO_RADIANS,
            np.ascontiguousarray(spec_arr, dtype=spec_dtype),
            np.ascontiguousarray(leg_arr, dtype=leg_dtype),
        )
    except Exception as exc:
        raise SpheremackError(f"Spectral interpolation failed: {exc}") from exc
