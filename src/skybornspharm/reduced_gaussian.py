"""Packed reduced-Gaussian spherical harmonic transforms."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from . import _spharm_rs
from .spherical_harmonics import (
    DEFAULT_EARTH_RADIUS,
    ComplexArray,
    FloatArray,
    SpheremackError,
    ValidationError,
    _call_backend,
    _resolve_lat_order,
    _resolve_public_complex_dtype,
    _resolve_public_real_dtype,
    _resolve_spectral_layout,
    _validate_layout,
    _validate_precision,
)

IntArray = NDArray[np.integer]

__all__ = [
    "ReducedGaussianSpharmt",
    "ReducedGaussianTransform",
]

_private_vars = (
    "pl",
    "nlat",
    "npoints",
    "rsphere",
    "gridtype",
    "legfunc",
    "precision",
    "layout",
    "lat_order",
    "south_to_north",
)


class ReducedGaussianSpharmt:
    """
    Spherical harmonic transforms for packed reduced Gaussian grids.

    Backed by Pure-Rust `_spharm_rs.RustReducedSpharmCore`, supporting both legacy
    ``(sum(pl), *extra)`` and C-order ``(*lead, sum(pl))`` layouts, native
    ``float32``/``float64`` precision, and zero-copy south-to-north latitude
    ordering.
    """

    def __init__(
        self,
        pl: IntArray,
        rsphere: float = DEFAULT_EARTH_RADIUS,
        legfunc: str = "stored",
        precision: str = "auto",
        layout: str = "auto",
        lat_order: str = "auto",
        lat: Optional[FloatArray] = None,
    ) -> None:
        self.pl = self._validate_pl(pl)
        self.nlat = int(self.pl.size)
        self.npoints = int(self.pl.sum())
        self.rsphere = float(rsphere)
        self.gridtype = "reduced_gaussian"
        self.legfunc = self._validate_legfunc(legfunc)
        self.precision = self._validate_precision(precision)
        self.layout = _validate_layout(layout)
        self.lat_order, self.south_to_north = _resolve_lat_order(
            self.nlat, lat_order, lat
        )

        if self.rsphere <= 0.0:
            raise ValidationError(f"rsphere must be positive, got {rsphere}")

        self._core = _call_backend(
            _spharm_rs.RustReducedSpharmCore,
            self.pl,
            self.rsphere,
            self.legfunc,
            operation_name="RustReducedSpharmCore initialization",
        )

    @staticmethod
    def _validate_pl(pl: IntArray) -> IntArray:
        arr = np.asarray(pl, dtype=np.int32)
        if arr.ndim != 1:
            raise ValidationError(f"pl must be rank 1, got shape {arr.shape}")
        if arr.size < 3:
            raise ValidationError(f"pl must contain at least 3 rows, got {arr.size}")
        if np.any(arr < 4):
            raise ValidationError("each reduced Gaussian row must contain >= 4 points")
        return np.ascontiguousarray(arr)

    @staticmethod
    def _validate_legfunc(legfunc: str) -> str:
        if legfunc not in ("stored", "computed"):
            raise ValidationError(
                f'legfunc must be "stored" or "computed", got "{legfunc}"'
            )
        return legfunc

    @staticmethod
    def _validate_precision(precision: str) -> str:
        return _validate_precision(precision)

    def _public_real_dtype(self, *arrays: np.ndarray) -> np.dtype:
        return _resolve_public_real_dtype(self.precision, *arrays)

    def _public_complex_dtype(self, *arrays: np.ndarray) -> np.dtype:
        return _resolve_public_complex_dtype(self.precision, *arrays)

    def __setattr__(self, key: str, val) -> None:
        if key in self.__dict__ and key in _private_vars:
            raise AttributeError(f"Attempt to rebind read-only instance variable {key}")
        self.__dict__[key] = val

    def __delattr__(self, key: str) -> None:
        if key in self.__dict__ and key in _private_vars:
            raise AttributeError(f"Attempt to unbind read-only instance variable {key}")
        del self.__dict__[key]

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(nlat={self.nlat:d}, "
            f"npoints={self.npoints:d}, rsphere={self.rsphere:e}, "
            f"precision='{self.precision}')"
        )

    def close(self) -> None:
        """Compatibility no-op for callers that manage transform lifetimes."""

    def _validate_ntrunc(self, ntrunc: Optional[int]) -> int:
        if ntrunc is None:
            return self.nlat - 1
        ntrunc = int(ntrunc)
        if ntrunc < 0 or ntrunc > self.nlat - 1:
            raise ValidationError(
                f"ntrunc must be between 0 and {self.nlat - 1}, got {ntrunc}"
            )
        return ntrunc

    def _resolve_grid_layout(
        self, data: np.ndarray, operation_name: str
    ) -> Tuple[str, Tuple[int, ...]]:
        if data.ndim < 1:
            raise ValidationError(
                f"{operation_name} needs at least a rank 1 packed array, "
                f"got rank {data.ndim}"
            )

        if data.ndim == 1:
            if data.shape[0] != self.npoints:
                raise ValidationError(
                    f"{operation_name} needs a packed first dimension of "
                    f"{self.npoints}, got {data.shape[0]}"
                )
            return "c", ()

        if self.layout == "c":
            if data.shape[-1] != self.npoints:
                raise ValidationError(
                    f"{operation_name} needs a packed trailing dimension of "
                    f"{self.npoints}, got {data.shape[-1]}"
                )
            return "c", data.shape[:-1]

        if self.layout == "legacy":
            if data.shape[0] != self.npoints:
                raise ValidationError(
                    f"{operation_name} needs a packed first dimension of "
                    f"{self.npoints}, got {data.shape[0]}"
                )
            return "legacy", data.shape[1:]

        is_c = data.shape[-1] == self.npoints
        is_legacy = data.shape[0] == self.npoints
        if is_c and not is_legacy:
            return "c", data.shape[:-1]
        if is_legacy and not is_c:
            return "legacy", data.shape[1:]
        if is_c and is_legacy:
            if data.flags.f_contiguous and not data.flags.c_contiguous:
                return "legacy", data.shape[1:]
            return "c", data.shape[:-1]

        raise ValidationError(
            f"{operation_name} needs a packed dimension of {self.npoints}, "
            f"got shape {data.shape}"
        )

    def _prepare_grid_input(
        self,
        data: FloatArray,
        operation_name: str,
        target_dtype: Optional[np.dtype] = None,
    ) -> Tuple[str, int, FloatArray, Tuple[int, ...]]:
        arr = data if type(data) is np.ndarray else np.asarray(data)
        resolved_layout, batch_shape = self._resolve_grid_layout(arr, operation_name)
        compute_dtype = (
            self._public_real_dtype(arr)
            if target_dtype is None
            else (target_dtype if type(target_dtype) is np.dtype else np.dtype(target_dtype))
        )
        nt = 1 if not batch_shape else math.prod(batch_shape)

        if arr.ndim == 1:
            reshaped = arr.reshape(1, self.npoints)
            c_2d = (
                reshaped
                if reshaped.dtype == compute_dtype and reshaped.flags.c_contiguous
                else np.ascontiguousarray(reshaped, dtype=compute_dtype)
            )
        elif resolved_layout == "c":
            reshaped = arr.reshape(nt, self.npoints)
            c_2d = (
                reshaped
                if reshaped.dtype == compute_dtype and reshaped.flags.c_contiguous
                else np.ascontiguousarray(reshaped, dtype=compute_dtype)
            )
        else:
            c_2d = np.ascontiguousarray(
                np.moveaxis(arr.reshape(self.npoints, nt), 0, -1),
                dtype=compute_dtype,
            )

        return resolved_layout, nt, c_2d, batch_shape

    def _prepare_spectral_input(
        self,
        data: ComplexArray,
        operation_name: str,
        target_dtype: Optional[np.dtype] = None,
    ) -> Tuple[str, int, int, ComplexArray, Tuple[int, ...]]:
        arr = data if type(data) is np.ndarray else np.asarray(data)
        if arr.ndim < 1:
            raise ValidationError(
                f"{operation_name} needs at least a rank 1 spectrum, got rank {arr.ndim}"
            )
        resolved_layout, ntrunc, batch_shape = _resolve_spectral_layout(
            arr, self.layout, self.nlat - 1, operation_name
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
        u_arr = np.asarray(ugrid)
        v_arr = np.asarray(vgrid)
        if u_arr.shape != v_arr.shape:
            raise ValidationError("ugrid and vgrid must have the same shape")

        compute_dtype = self._public_real_dtype(u_arr, v_arr)
        layout_u, nt_u, u_2d, batch_u = self._prepare_grid_input(
            u_arr, operation_name, target_dtype=compute_dtype
        )
        layout_v, nt_v, v_2d, batch_v = self._prepare_grid_input(
            v_arr, operation_name, target_dtype=compute_dtype
        )
        if layout_u != layout_v or nt_u != nt_v or batch_u != batch_v:
            raise ValidationError("ugrid and vgrid must have consistent dimensions")
        return layout_u, nt_u, u_2d, v_2d, batch_u

    def _prepare_paired_spectral_inputs(
        self, spec_a: ComplexArray, spec_b: ComplexArray, operation_name: str
    ) -> Tuple[str, int, int, ComplexArray, ComplexArray, Tuple[int, ...]]:
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
        grid_2d: FloatArray,
        resolved_layout: str,
        batch_shape: Tuple[int, ...],
        output_dtype: Optional[np.dtype] = None,
    ) -> FloatArray:
        array = np.asarray(grid_2d)
        if output_dtype is not None and array.dtype != np.dtype(output_dtype):
            array = array.astype(np.dtype(output_dtype), copy=False)
        if not batch_shape:
            return array[0]
        if resolved_layout == "c":
            return array.reshape(batch_shape + (self.npoints,))
        return np.moveaxis(array, 0, -1).reshape((self.npoints,) + batch_shape)

    def _restore_spectral_output(
        self,
        spec_2d: ComplexArray,
        resolved_layout: str,
        batch_shape: Tuple[int, ...],
        output_dtype: Optional[np.dtype] = None,
    ) -> ComplexArray:
        array = np.asarray(spec_2d)
        if output_dtype is not None and array.dtype != np.dtype(output_dtype):
            array = array.astype(np.dtype(output_dtype), copy=False)
        if not batch_shape:
            return array[0]
        if resolved_layout == "c":
            return array.reshape(batch_shape + (array.shape[1],))
        return np.moveaxis(array, 0, -1).reshape((array.shape[1],) + batch_shape)

    # Legacy internal adapters used by Skyborn's ReducedVectorWind (expects (npoints, nt) <-> (nmdim, nt))
    def _validate_grid_data(
        self, data: FloatArray, operation_name: str
    ) -> Tuple[int, FloatArray, Tuple[int, ...]]:
        _, nt, c_2d, batch_shape = self._prepare_grid_input(data, operation_name)
        return nt, c_2d, batch_shape

    def _validate_spectral_data(
        self, data: ComplexArray, operation_name: str
    ) -> Tuple[int, int, ComplexArray, Tuple[int, ...]]:
        _, nt, ntrunc, c_2d, batch_shape = self._prepare_spectral_input(
            data, operation_name
        )
        return nt, ntrunc, c_2d, batch_shape

    def _validate_paired_grid_data(
        self, ugrid: FloatArray, vgrid: FloatArray, operation_name: str
    ) -> Tuple[int, FloatArray, FloatArray, Tuple[int, ...]]:
        if np.shape(ugrid) != np.shape(vgrid):
            raise ValidationError("ugrid and vgrid must have the same shape")
        nt, u_norm, extra_shape = self._validate_grid_data(ugrid, operation_name)
        _, v_norm, v_extra_shape = self._validate_grid_data(vgrid, operation_name)
        if extra_shape != v_extra_shape:
            raise ValidationError("ugrid and vgrid must have consistent dimensions")
        return nt, u_norm, v_norm, extra_shape

    def _validate_paired_spectral_data(
        self, spec_a: ComplexArray, spec_b: ComplexArray, operation_name: str
    ) -> Tuple[int, int, ComplexArray, ComplexArray, Tuple[int, ...]]:
        if np.shape(spec_a) != np.shape(spec_b):
            raise ValidationError("paired spectra must have the same shape")
        nt_a, ntrunc_a, a_norm, extra_shape = self._validate_spectral_data(
            spec_a, operation_name
        )
        nt_b, ntrunc_b, b_norm, b_extra_shape = self._validate_spectral_data(
            spec_b, operation_name
        )
        if nt_a != nt_b or ntrunc_a != ntrunc_b or extra_shape != b_extra_shape:
            raise ValidationError("paired spectra must have consistent dimensions")
        return nt_a, ntrunc_a, a_norm, b_norm, extra_shape

    def _restore_spectral_shape(
        self,
        dataspec: ComplexArray,
        extra_shape: Tuple[int, ...],
        output_dtype: Optional[np.dtype] = None,
    ) -> ComplexArray:
        arr = np.asarray(dataspec)
        if not extra_shape:
            result = arr[:, 0]
        else:
            result = arr.reshape((arr.shape[0],) + extra_shape)
        if output_dtype is None:
            if self.precision == "double":
                output_dtype = np.complex128
            elif self.precision == "single":
                output_dtype = np.complex64
        if output_dtype is not None:
            result = result.astype(np.dtype(output_dtype), copy=False)
        return result

    def _restore_grid_shape(
        self,
        datagrid: FloatArray,
        extra_shape: Tuple[int, ...],
        output_dtype: Optional[np.dtype] = None,
    ) -> FloatArray:
        arr = np.asarray(datagrid)
        if not extra_shape:
            result = arr[:, 0]
        else:
            result = arr.reshape((self.npoints,) + extra_shape)
        if output_dtype is None:
            if self.precision == "double":
                output_dtype = np.float64
            elif self.precision == "single":
                output_dtype = np.float32
        if output_dtype is not None:
            result = result.astype(np.dtype(output_dtype), copy=False)
        return result

    def _analyze_scalar(
        self, datagrid: FloatArray, ntrunc: Optional[int] = None
    ) -> ComplexArray:
        ntrunc = self._validate_ntrunc(ntrunc)
        c_2d = np.ascontiguousarray(np.asarray(datagrid).T)
        spec_2d = _call_backend(
            self._core.grdtospec,
            c_2d,
            ntrunc=ntrunc,
            south_to_north=self.south_to_north,
            operation_name="reduced Gaussian grdtospec",
        )
        return spec_2d.T

    def _synthesize_scalar(
        self, dataspec: ComplexArray, ntrunc: Optional[int] = None
    ) -> FloatArray:
        self._validate_ntrunc(ntrunc)
        c_2d = np.ascontiguousarray(np.asarray(dataspec).T)
        grid_2d = _call_backend(
            self._core.spectogrd,
            c_2d,
            south_to_north=self.south_to_north,
            operation_name="reduced Gaussian spectogrd",
        )
        return grid_2d.T

    def _synthesize_scalar_pair(
        self,
        dataspec_a: ComplexArray,
        dataspec_b: ComplexArray,
        ntrunc: Optional[int] = None,
    ) -> Tuple[FloatArray, FloatArray]:
        self._validate_ntrunc(ntrunc)
        a_2d = np.ascontiguousarray(np.asarray(dataspec_a).T)
        b_2d = np.ascontiguousarray(np.asarray(dataspec_b).T)
        grid_a, grid_b = _call_backend(
            self._core.spectogrd_pair,
            a_2d,
            b_2d,
            south_to_north=self.south_to_north,
            operation_name="reduced Gaussian paired spectogrd",
        )
        return grid_a.T, grid_b.T

    def _analyze_wind(
        self,
        ugrid: FloatArray,
        vgrid: FloatArray,
        ntrunc: Optional[int] = None,
    ) -> Tuple[ComplexArray, ComplexArray]:
        ntrunc = self._validate_ntrunc(ntrunc)
        u_2d = np.ascontiguousarray(np.asarray(ugrid).T)
        v_2d = np.ascontiguousarray(np.asarray(vgrid).T)
        vrt, div = _call_backend(
            self._core.getvrtdivspec,
            u_2d,
            v_2d,
            ntrunc=ntrunc,
            south_to_north=self.south_to_north,
            operation_name="reduced Gaussian vector analysis",
        )
        return vrt.T, div.T

    def _analyze_vorticity(
        self,
        ugrid: FloatArray,
        vgrid: FloatArray,
        ntrunc: Optional[int] = None,
    ) -> ComplexArray:
        ntrunc = self._validate_ntrunc(ntrunc)
        u_2d = np.ascontiguousarray(np.asarray(ugrid).T)
        v_2d = np.ascontiguousarray(np.asarray(vgrid).T)
        vrt = _call_backend(
            self._core.getvrtspec,
            u_2d,
            v_2d,
            ntrunc=ntrunc,
            south_to_north=self.south_to_north,
            operation_name="reduced Gaussian vorticity analysis",
        )
        return vrt.T

    def _analyze_divergence(
        self,
        ugrid: FloatArray,
        vgrid: FloatArray,
        ntrunc: Optional[int] = None,
    ) -> ComplexArray:
        ntrunc = self._validate_ntrunc(ntrunc)
        u_2d = np.ascontiguousarray(np.asarray(ugrid).T)
        v_2d = np.ascontiguousarray(np.asarray(vgrid).T)
        div = _call_backend(
            self._core.getdivspec,
            u_2d,
            v_2d,
            ntrunc=ntrunc,
            south_to_north=self.south_to_north,
            operation_name="reduced Gaussian divergence analysis",
        )
        return div.T

    def _synthesize_gradient(
        self, dataspec: ComplexArray, ntrunc: Optional[int] = None
    ) -> Tuple[FloatArray, FloatArray]:
        self._validate_ntrunc(ntrunc)
        c_2d = np.ascontiguousarray(np.asarray(dataspec).T)
        ugrad, vgrad = _call_backend(
            self._core.getgrad,
            c_2d,
            south_to_north=self.south_to_north,
            operation_name="reduced Gaussian gradient synthesis",
        )
        return ugrad.T, vgrad.T

    def _synthesize_gradient_pair(
        self,
        dataspec_a: ComplexArray,
        dataspec_b: ComplexArray,
        ntrunc: Optional[int] = None,
    ) -> Tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        self._validate_ntrunc(ntrunc)
        a_2d = np.ascontiguousarray(np.asarray(dataspec_a).T)
        b_2d = np.ascontiguousarray(np.asarray(dataspec_b).T)
        au, av, bu, bv = _call_backend(
            self._core.getgrad_pair,
            a_2d,
            b_2d,
            south_to_north=self.south_to_north,
            operation_name="reduced Gaussian paired gradient synthesis",
        )
        return au.T, av.T, bu.T, bv.T

    def _invlap(self, dataspec: ComplexArray) -> ComplexArray:
        c_2d = np.ascontiguousarray(np.asarray(dataspec).T)
        inv = _call_backend(
            _spharm_rs.invlap,
            c_2d,
            self.rsphere,
            operation_name="reduced Gaussian invlap",
        )
        return inv.T

    # Public API
    def grdtospec(
        self, datagrid: FloatArray, ntrunc: Optional[int] = None
    ) -> ComplexArray:
        """Analyze packed reduced Gaussian scalar data into `spharm` spectra."""
        resolved_layout, nt, grid_2d, batch_shape = self._prepare_grid_input(
            datagrid, "grdtospec"
        )
        ntrunc = self._validate_ntrunc(ntrunc)
        if nt == 0:
            nmdim = (ntrunc + 1) * (ntrunc + 2) // 2
            cdtype = np.dtype(
                np.complex128 if grid_2d.dtype == np.float64 else np.complex64
            )
            return self._restore_spectral_output(
                np.empty((0, nmdim), dtype=cdtype), resolved_layout, batch_shape
            )

        spec_2d = _call_backend(
            self._core.grdtospec,
            grid_2d,
            ntrunc=ntrunc,
            south_to_north=self.south_to_north,
            operation_name="reduced Gaussian grdtospec",
        )
        return self._restore_spectral_output(spec_2d, resolved_layout, batch_shape)

    def spectogrd(self, dataspec: ComplexArray) -> FloatArray:
        """Synthesize `spharm` spectra onto the packed reduced Gaussian grid."""
        resolved_layout, nt, _, spec_2d, batch_shape = self._prepare_spectral_input(
            dataspec, "spectogrd"
        )
        if nt == 0:
            rdtype = np.dtype(
                np.float64 if spec_2d.dtype == np.complex128 else np.float32
            )
            return self._restore_grid_output(
                np.empty((0, self.npoints), dtype=rdtype),
                resolved_layout,
                batch_shape,
            )

        grid_2d = _call_backend(
            self._core.spectogrd,
            spec_2d,
            south_to_north=self.south_to_north,
            operation_name="reduced Gaussian spectogrd",
        )
        return self._restore_grid_output(grid_2d, resolved_layout, batch_shape)

    def _spectogrd_pair(
        self, spec_a: ComplexArray, spec_b: ComplexArray
    ) -> Tuple[FloatArray, FloatArray]:
        """Synthesize two scalar spectra together through one batched call."""
        resolved_layout, nt, _, a_2d, b_2d, batch_shape = (
            self._prepare_paired_spectral_inputs(spec_a, spec_b, "_spectogrd_pair")
        )
        if nt == 0:
            rdtype = np.dtype(
                np.float64 if a_2d.dtype == np.complex128 else np.float32
            )
            empty = np.empty((0, self.npoints), dtype=rdtype)
            return (
                self._restore_grid_output(empty, resolved_layout, batch_shape),
                self._restore_grid_output(empty.copy(), resolved_layout, batch_shape),
            )

        grid_a, grid_b = _call_backend(
            self._core.spectogrd_pair,
            a_2d,
            b_2d,
            south_to_north=self.south_to_north,
            operation_name="reduced Gaussian paired spectogrd",
        )
        return (
            self._restore_grid_output(grid_a, resolved_layout, batch_shape),
            self._restore_grid_output(grid_b, resolved_layout, batch_shape),
        )

    def getvrtdivspec(
        self, ugrid: FloatArray, vgrid: FloatArray, ntrunc: Optional[int] = None
    ) -> Tuple[ComplexArray, ComplexArray]:
        """Compute vorticity and divergence spectra from packed reduced-grid winds."""
        resolved_layout, nt, u_2d, v_2d, batch_shape = self._prepare_paired_grid_inputs(
            ugrid, vgrid, "getvrtdivspec"
        )
        ntrunc = self._validate_ntrunc(ntrunc)
        if nt == 0:
            nmdim = (ntrunc + 1) * (ntrunc + 2) // 2
            cdtype = np.dtype(
                np.complex128 if u_2d.dtype == np.float64 else np.complex64
            )
            empty = np.empty((0, nmdim), dtype=cdtype)
            return (
                self._restore_spectral_output(empty, resolved_layout, batch_shape),
                self._restore_spectral_output(empty.copy(), resolved_layout, batch_shape),
            )

        vrt_2d, div_2d = _call_backend(
            self._core.getvrtdivspec,
            u_2d,
            v_2d,
            ntrunc=ntrunc,
            south_to_north=self.south_to_north,
            operation_name="reduced Gaussian vector analysis",
        )
        return (
            self._restore_spectral_output(vrt_2d, resolved_layout, batch_shape),
            self._restore_spectral_output(div_2d, resolved_layout, batch_shape),
        )

    def getvrtspec(
        self, ugrid: FloatArray, vgrid: FloatArray, ntrunc: Optional[int] = None
    ) -> ComplexArray:
        """Compute only vorticity spectral coefficients from packed winds."""
        resolved_layout, nt, u_2d, v_2d, batch_shape = self._prepare_paired_grid_inputs(
            ugrid, vgrid, "getvrtspec"
        )
        ntrunc = self._validate_ntrunc(ntrunc)
        if nt == 0:
            nmdim = (ntrunc + 1) * (ntrunc + 2) // 2
            cdtype = np.dtype(
                np.complex128 if u_2d.dtype == np.float64 else np.complex64
            )
            return self._restore_spectral_output(
                np.empty((0, nmdim), dtype=cdtype), resolved_layout, batch_shape
            )

        vrt_2d = _call_backend(
            self._core.getvrtspec,
            u_2d,
            v_2d,
            ntrunc=ntrunc,
            south_to_north=self.south_to_north,
            operation_name="reduced Gaussian vorticity analysis",
        )
        return self._restore_spectral_output(vrt_2d, resolved_layout, batch_shape)

    def getdivspec(
        self, ugrid: FloatArray, vgrid: FloatArray, ntrunc: Optional[int] = None
    ) -> ComplexArray:
        """Compute only divergence spectral coefficients from packed winds."""
        resolved_layout, nt, u_2d, v_2d, batch_shape = self._prepare_paired_grid_inputs(
            ugrid, vgrid, "getdivspec"
        )
        ntrunc = self._validate_ntrunc(ntrunc)
        if nt == 0:
            nmdim = (ntrunc + 1) * (ntrunc + 2) // 2
            cdtype = np.dtype(
                np.complex128 if u_2d.dtype == np.float64 else np.complex64
            )
            return self._restore_spectral_output(
                np.empty((0, nmdim), dtype=cdtype), resolved_layout, batch_shape
            )

        div_2d = _call_backend(
            self._core.getdivspec,
            u_2d,
            v_2d,
            ntrunc=ntrunc,
            south_to_north=self.south_to_north,
            operation_name="reduced Gaussian divergence analysis",
        )
        return self._restore_spectral_output(div_2d, resolved_layout, batch_shape)

    def getuv(
        self, vrtspec: ComplexArray, divspec: ComplexArray
    ) -> Tuple[FloatArray, FloatArray]:
        """Synthesize packed reduced-grid winds from vorticity/divergence spectra."""
        resolved_layout, nt, _, vrt_2d, div_2d, batch_shape = (
            self._prepare_paired_spectral_inputs(vrtspec, divspec, "getuv")
        )
        if nt == 0:
            rdtype = np.dtype(
                np.float64 if vrt_2d.dtype == np.complex128 else np.float32
            )
            empty = np.empty((0, self.npoints), dtype=rdtype)
            return (
                self._restore_grid_output(empty, resolved_layout, batch_shape),
                self._restore_grid_output(empty.copy(), resolved_layout, batch_shape),
            )

        u_2d, v_2d = _call_backend(
            self._core.getuv,
            vrt_2d,
            div_2d,
            south_to_north=self.south_to_north,
            operation_name="reduced Gaussian getuv",
        )
        return (
            self._restore_grid_output(u_2d, resolved_layout, batch_shape),
            self._restore_grid_output(v_2d, resolved_layout, batch_shape),
        )

    def getgrad(self, chispec: ComplexArray) -> Tuple[FloatArray, FloatArray]:
        """Compute the packed reduced-grid vector gradient of a scalar spectrum."""
        resolved_layout, nt, _, spec_2d, batch_shape = self._prepare_spectral_input(
            chispec, "getgrad"
        )
        if nt == 0:
            rdtype = np.dtype(
                np.float64 if spec_2d.dtype == np.complex128 else np.float32
            )
            empty = np.empty((0, self.npoints), dtype=rdtype)
            return (
                self._restore_grid_output(empty, resolved_layout, batch_shape),
                self._restore_grid_output(empty.copy(), resolved_layout, batch_shape),
            )

        ugrad_2d, vgrad_2d = _call_backend(
            self._core.getgrad,
            spec_2d,
            south_to_north=self.south_to_north,
            operation_name="reduced Gaussian gradient synthesis",
        )
        return (
            self._restore_grid_output(ugrad_2d, resolved_layout, batch_shape),
            self._restore_grid_output(vgrad_2d, resolved_layout, batch_shape),
        )

    def getgrad_pair(
        self, spec_a: ComplexArray, spec_b: ComplexArray
    ) -> Tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """Compute packed reduced-grid gradients for two scalar spectra at once."""
        resolved_layout, nt, _, a_2d, b_2d, batch_shape = (
            self._prepare_paired_spectral_inputs(spec_a, spec_b, "getgrad_pair")
        )
        if nt == 0:
            rdtype = np.dtype(
                np.float64 if a_2d.dtype == np.complex128 else np.float32
            )
            empty = np.empty((0, self.npoints), dtype=rdtype)
            return (
                self._restore_grid_output(empty, resolved_layout, batch_shape),
                self._restore_grid_output(empty.copy(), resolved_layout, batch_shape),
                self._restore_grid_output(empty.copy(), resolved_layout, batch_shape),
                self._restore_grid_output(empty.copy(), resolved_layout, batch_shape),
            )

        au, av, bu, bv = _call_backend(
            self._core.getgrad_pair,
            a_2d,
            b_2d,
            south_to_north=self.south_to_north,
            operation_name="reduced Gaussian paired gradient synthesis",
        )
        return (
            self._restore_grid_output(au, resolved_layout, batch_shape),
            self._restore_grid_output(av, resolved_layout, batch_shape),
            self._restore_grid_output(bu, resolved_layout, batch_shape),
            self._restore_grid_output(bv, resolved_layout, batch_shape),
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
            operation_name="reduced Gaussian invlap",
        )
        return self._restore_spectral_output(invlap_2d, resolved_layout, batch_shape)

    def invlapspec(self, dataspec: ComplexArray) -> ComplexArray:
        """Apply the inverse spherical Laplacian to spectral coefficients."""
        return self._invlapspec(dataspec, "invlapspec")

    def invlap(self, dataspec: ComplexArray) -> ComplexArray:
        """Apply the inverse spherical Laplacian to spectral coefficients."""
        return self._invlapspec(dataspec, "invlap")

    def getpsispec(
        self, ugrid: FloatArray, vgrid: FloatArray, ntrunc: Optional[int] = None
    ) -> ComplexArray:
        """Compute streamfunction spectral coefficients from packed winds."""
        return self._invlapspec(self.getvrtspec(ugrid, vgrid, ntrunc), "getpsispec")

    def getchispec(
        self, ugrid: FloatArray, vgrid: FloatArray, ntrunc: Optional[int] = None
    ) -> ComplexArray:
        """Compute velocity-potential spectral coefficients from packed winds."""
        return self._invlapspec(self.getdivspec(ugrid, vgrid, ntrunc), "getchispec")

    def getpsichispec(
        self, ugrid: FloatArray, vgrid: FloatArray, ntrunc: Optional[int] = None
    ) -> Tuple[ComplexArray, ComplexArray]:
        """Compute streamfunction and velocity-potential spectra from packed winds."""
        vrtspec, divspec = self.getvrtdivspec(ugrid, vgrid, ntrunc)
        return (
            self._invlapspec(vrtspec, "getpsichispec"),
            self._invlapspec(divspec, "getpsichispec"),
        )

    def getpsi(
        self, ugrid: FloatArray, vgrid: FloatArray, ntrunc: Optional[int] = None
    ) -> FloatArray:
        """Compute streamfunction from packed winds without materializing chi."""
        return self.spectogrd(self.getpsispec(ugrid, vgrid, ntrunc))

    def getchi(
        self, ugrid: FloatArray, vgrid: FloatArray, ntrunc: Optional[int] = None
    ) -> FloatArray:
        """Compute velocity potential from packed winds without materializing psi."""
        return self.spectogrd(self.getchispec(ugrid, vgrid, ntrunc))

    def getpsichi(
        self, ugrid: FloatArray, vgrid: FloatArray, ntrunc: Optional[int] = None
    ) -> Tuple[FloatArray, FloatArray]:
        """Compute streamfunction and velocity potential from packed winds."""
        psispec, chispec = self.getpsichispec(ugrid, vgrid, ntrunc)
        return self._spectogrd_pair(psispec, chispec)

    def specsmooth(self, datagrid: FloatArray, smooth: FloatArray) -> FloatArray:
        """Apply latitude-dependent spectral smoothing to a packed scalar field."""
        resolved_layout, nt, grid_2d, batch_shape = self._prepare_grid_input(
            datagrid, "specsmooth"
        )
        smooth_arr = np.asarray(smooth)
        if smooth_arr.ndim != 1 or smooth_arr.shape[0] != self.nlat:
            raise ValidationError(
                f"smooth must be rank 1 with size {self.nlat}, got shape {smooth_arr.shape}"
            )
        if nt == 0:
            return self._restore_grid_output(grid_2d, resolved_layout, batch_shape)

        spec_2d = _call_backend(
            self._core.grdtospec,
            grid_2d,
            ntrunc=self.nlat - 1,
            south_to_north=self.south_to_north,
            operation_name="reduced Gaussian specsmooth grdtospec",
        )
        smoothed_2d = _call_backend(
            _spharm_rs.multsmoothfact,
            spec_2d,
            np.ascontiguousarray(smooth_arr, dtype=grid_2d.dtype),
            operation_name="reduced Gaussian specsmooth multsmoothfact",
        )
        out_2d = _call_backend(
            self._core.spectogrd,
            smoothed_2d,
            south_to_north=self.south_to_north,
            operation_name="reduced Gaussian specsmooth spectogrd",
        )
        return self._restore_grid_output(out_2d, resolved_layout, batch_shape)


class ReducedGaussianTransform(ReducedGaussianSpharmt):
    """Modern C-order reduced-Gaussian transform class (defaults to ``layout="c"``)."""

    def __init__(
        self,
        pl: IntArray,
        rsphere: float = DEFAULT_EARTH_RADIUS,
        legfunc: str = "stored",
        precision: str = "auto",
        layout: str = "c",
        lat_order: str = "auto",
        lat: Optional[FloatArray] = None,
    ) -> None:
        super().__init__(
            pl=pl,
            rsphere=rsphere,
            legfunc=legfunc,
            precision=precision,
            layout=layout,
            lat_order=lat_order,
            lat=lat,
        )
