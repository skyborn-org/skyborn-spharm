//! PyO3 bindings for the Rust spherical-harmonic core of skybornspharm (Python extension module `skybornspharm._spharm_rs`).
//!
//! Exposed items:
//! - classes `RustSpharmCore` (full latitude-longitude grid, regular or Gaussian) and `RustReducedSpharmCore` (reduced Gaussian grid);
//! - functions `gaussian_lats_wts`, `getlegfunc`, `ihgeod`, `specintrp`, `lap`, `invlap`, `multsmoothfact`.
//!
//! Conventions shared by all entry points:
//! - Spectral coefficients use SPHEREPACK triangular truncation: one complex array of length `nmdim = (ntrunc + 1) * (ntrunc + 2) / 2` per field, `m` outer and `n = m..=ntrunc` inner, so `(m, n)` is at `m * (2 * ntrunc - m + 3) / 2 + (n - m)` (0-based). Fortran: `nm = nmstrt + n - m + 1`, `nmstrt = nmstrt + ntrunc - m + 2` (see `spectral.rs`). `ntrunc` is recovered from `nmdim` when only the spectral array is given.
//! - Batches are C-order with the field index first: grids `(nt, nlat, nlon)` (or `(nt, npoints)` for reduced grids) and spectra `(nt, nmdim)`. The spectral layout equals the Fortran `dataspec(nmdim, nt)` memory layout.
//! - Grid rows are ordered north to south by default (SPHEREPACK colatitude order). `south_to_north=True` declares that input rows are ordered south to north and/or requests output rows in that order.
//! - Precision follows the numpy dtype: `float32`/`complex64` or `float64`/`complex128`; outputs have the matching dtype. When `u` and `v` have different float dtypes the `float32` operand is promoted to `float64`.
//! - Errors: `ValueError` for invalid shapes, values or truncation (errors that Fortran returns through `ierror`), `TypeError` for unsupported dtypes or dimensionality.
//! - Large problems release the GIL (`py.detach`) once the work exceeds a size threshold.
//! - Vector quantities: vorticity and divergence spectra are scaled by the sphere radius `rsphere`. In the Fortran packing routines (`onedtotwod_vrtdiv.f`, `twodtooned_vrtdiv.f`) the vector harmonic coefficients `(br, bi)` (divergence part) and `(cr, ci)` (vorticity part) relate to the spectra by `div = -sqrt(n * (n + 1)) / rsphere * scale * (br + i*bi)` and `vrt = +sqrt(n * (n + 1)) / rsphere * scale * (cr + i*ci)` with `scale = 0.5` (0-based `n`; Fortran writes `sqrt(float(n)*float(n-1))` with 1-based `n`).
//!
//! Difference from Fortran: the original SPHEREPACK interface (and the f2py layer `_spherepack.pyf`, `spherepack_bindings.f90`, `reduced_bindings.f90`) is organised around separate `*i` initialisation routines that fill `wsave` arrays, `work`/`lwork` scratch arguments, `ierror` return codes and Fortran-order `(nlat, nlon, nt)` arrays. Here the engines build their tables lazily and cache them, no work arrays are exposed, and errors are Python exceptions.
//!
//! Derived from NCAR SPHEREPACK 3.2 (J. Adams and P. Swarztrauber), see LICENSE.spherepack.
pub mod engine;
pub mod fft;
pub mod precision;
pub mod reduced;
pub mod spectral;

use std::f64::consts::FRAC_PI_2;

use num_complex::Complex;
use numpy::ndarray::{s, Array1, Array2, Array3, ArrayView2, ArrayView3};
use numpy::{
    Complex32, Complex64, IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1,
    PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use spharm_math::GridType;

use crate::engine::{compute_tile_size, FullGridEngine, TypedTables, VectorMode};
use crate::reduced::{ReducedEngine, TypedReducedBasis};
use crate::spectral::{
    apply_invlap_batch, apply_lap_batch, apply_multsmoothfact_batch, eval_specintrp,
    nmdim_from_ntrunc, ntrunc_from_nmdim,
};

/// Run `f`, releasing the Python GIL around it when `detach` is true.
///
/// Callers pass `detach = true` only for large problems, because releasing and re-acquiring the GIL costs more than small transforms.
///
/// No direct Fortran counterpart.
#[inline]
fn maybe_detach<F, R>(py: Python<'_>, detach: bool, f: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    if detach {
        py.detach(f)
    } else {
        f()
    }
}

/// Reinterpret a `numpy::Complex32` slice as `num_complex::Complex<f32>` without copying (identical `(re, im)` memory layout).
///
/// No direct Fortran counterpart.
fn slice_c32<'a>(arr: &'a [Complex32]) -> &'a [Complex<f32>] {
    unsafe { std::slice::from_raw_parts(arr.as_ptr() as *const Complex<f32>, arr.len()) }
}

/// Mutable counterpart of `slice_c32`.
fn slice_c32_mut<'a>(arr: &'a mut [Complex32]) -> &'a mut [Complex<f32>] {
    unsafe { std::slice::from_raw_parts_mut(arr.as_mut_ptr() as *mut Complex<f32>, arr.len()) }
}

/// Reinterpret a `numpy::Complex64` slice as `num_complex::Complex<f64>` without copying (identical `(re, im)` memory layout).
fn slice_c64<'a>(arr: &'a [Complex64]) -> &'a [Complex<f64>] {
    unsafe { std::slice::from_raw_parts(arr.as_ptr() as *const Complex<f64>, arr.len()) }
}

/// Mutable counterpart of `slice_c64`.
fn slice_c64_mut<'a>(arr: &'a mut [Complex64]) -> &'a mut [Complex<f64>] {
    unsafe { std::slice::from_raw_parts_mut(arr.as_mut_ptr() as *mut Complex<f64>, arr.len()) }
}

/// Convert an engine error message into a Python `ValueError`. Replaces the Fortran `ierror` return code.
fn to_py_err(msg: String) -> PyErr {
    PyValueError::new_err(msg)
}

/// Check that the sphere radius is positive and finite; otherwise raise `ValueError`.
///
/// Fortran: none (the Fortran routines simply divide by `rsphere`).
#[inline]
fn validate_rsphere(rsphere: f64) -> PyResult<f64> {
    if !(rsphere > 0.0) || !rsphere.is_finite() {
        return Err(PyValueError::new_err(format!(
            "rsphere must be positive and finite, got {rsphere}"
        )));
    }
    Ok(rsphere)
}

/// Resolve the optional `ntrunc` argument: `None` selects the maximum truncation `max_trunc` (callers pass `nlat - 1`); otherwise it must satisfy `0 <= ntrunc <= max_trunc`, else `ValueError`.
///
/// Fortran: the `ntrunc` argument of `twodtooned.f` / `getlegfunc.f`; the SPHEREPACK limit is `ntrunc <= nlat - 1`.
#[inline]
fn resolve_ntrunc(ntrunc: Option<isize>, max_trunc: usize) -> PyResult<usize> {
    match ntrunc {
        None => Ok(max_trunc),
        Some(t) if t < 0 || (t as usize) > max_trunc => Err(PyValueError::new_err(format!(
            "ntrunc must be between 0 and {max_trunc}, got {t}"
        ))),
        Some(t) => Ok(t as usize),
    }
}

/// Validate a 1-D `(nmdim,)` or 2-D `(nt, nmdim)` spectral shape and return `(nt, nmdim, ntrunc, is_1d)`; a 1-D input is treated as `nt = 1`. Raises `ValueError` for other ranks or an `nmdim` that is not `(T + 1) * (T + 2) / 2`.
///
/// Fortran: `ntrunc = -1.5 + 0.5*sqrt(9.-8.*(1.-float(nmdim)))` in `lap.f`, `invlap.f`, `multsmoothfact.f`.
fn validate_spec_dyn_shape(shape: &[usize], op_name: &str) -> PyResult<(usize, usize, usize, bool)> {
    match shape.len() {
        1 => {
            let nmdim = shape[0];
            let ntrunc = ntrunc_from_nmdim(nmdim).map_err(to_py_err)?;
            Ok((1, nmdim, ntrunc, true))
        }
        2 => {
            let nt = shape[0];
            let nmdim = shape[1];
            let ntrunc = ntrunc_from_nmdim(nmdim).map_err(to_py_err)?;
            Ok((nt, nmdim, ntrunc, false))
        }
        nd => Err(PyValueError::new_err(format!(
            "{op_name} expects a 1D (nmdim,) or 2D (nt, nmdim) spectral array, got {nd}D"
        ))),
    }
}

/// Check a grid shape is `(nt, nlat, nlon)` for this engine and return `nt`; otherwise `ValueError`. Fortran: the `g(nlat, nlon, nt)` array declaration (here C-order with `nt` first).
fn validate_grid3_shape(shape: &[usize], nlat: usize, nlon: usize, op: &str) -> PyResult<usize> {
    if shape.len() != 3 || shape[1] != nlat || shape[2] != nlon {
        return Err(PyValueError::new_err(format!(
            "{op} expects grid shape (nt, {nlat}, {nlon}), got {:?}",
            shape
        )));
    }
    Ok(shape[0])
}

/// Check a reduced-grid shape is `(nt, npoints)` where `npoints = sum(pl)` and return `nt`; otherwise `ValueError`. Fortran: `ngptot` argument of the `reduced_gaussian_*` routines.
fn validate_reduced_grid2(shape: &[usize], npoints: usize, op: &str) -> PyResult<usize> {
    if shape.len() != 2 || shape[1] != npoints {
        return Err(PyValueError::new_err(format!(
            "{op} expects reduced grid shape (nt, {npoints}), got {:?}",
            shape
        )));
    }
    Ok(shape[0])
}

/// Check a spectral array is 2-D `(nt, nmdim)` with `ntrunc` derived from `nmdim` not exceeding `max_trunc` (`nlat - 1`); returns `(nt, nmdim, ntrunc)`. Raises `ValueError` otherwise.
///
/// Fortran: the `ntrunc` limit checked by the analysis/synthesis routines (`ierror`).
fn validate_spec2_shape(
    shape: &[usize],
    max_trunc: usize,
    op: &str,
) -> PyResult<(usize, usize, usize)> {
    if shape.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "{op} expects 2D spectral array (nt, nmdim), got {:?}",
            shape
        )));
    }
    let nt = shape[0];
    let nmdim = shape[1];
    let ntrunc = ntrunc_from_nmdim(nmdim).map_err(to_py_err)?;
    if ntrunc > max_trunc {
        return Err(PyValueError::new_err(format!(
            "ntrunc too large - can be max of {max_trunc}, got {ntrunc}"
        )));
    }
    Ok((nt, nmdim, ntrunc))
}

/// Borrow a numpy `complex64` 2-D array as an `ndarray` view of `Complex<f32>`; zero-copy when contiguous, otherwise copy into `own` first.
fn view2_c32<'a>(
    arr: &'a PyReadonlyArray2<'_, Complex32>,
    own: &'a mut Vec<Complex<f32>>,
) -> ArrayView2<'a, Complex<f32>> {
    let v = arr.as_array();
    let (r, c) = (v.shape()[0], v.shape()[1]);
    if let Some(s) = v.as_slice() {
        let sl = unsafe { std::slice::from_raw_parts(s.as_ptr() as *const Complex<f32>, s.len()) };
        ArrayView2::from_shape((r, c), sl).unwrap()
    } else {
        *own = v.iter().map(|x| Complex::new(x.re, x.im)).collect();
        ArrayView2::from_shape((r, c), own).unwrap()
    }
}

/// Borrow a numpy `complex128` 2-D array as an `ndarray` view of `Complex<f64>`; zero-copy when contiguous, otherwise copy into `own` first.
fn view2_c64<'a>(
    arr: &'a PyReadonlyArray2<'_, Complex64>,
    own: &'a mut Vec<Complex<f64>>,
) -> ArrayView2<'a, Complex<f64>> {
    let v = arr.as_array();
    let (r, c) = (v.shape()[0], v.shape()[1]);
    if let Some(s) = v.as_slice() {
        let sl = unsafe { std::slice::from_raw_parts(s.as_ptr() as *const Complex<f64>, s.len()) };
        ArrayView2::from_shape((r, c), sl).unwrap()
    } else {
        *own = v.iter().map(|x| Complex::new(x.re, x.im)).collect();
        ArrayView2::from_shape((r, c), own).unwrap()
    }
}

/// Move a `nt * nmdim` complex buffer into a new numpy array of shape `(nt, nmdim)` without copying.
fn to_py_spec2<'py, T: crate::precision::ShtFloat>(
    py: Python<'py>,
    nt: usize,
    nmdim: usize,
    buf: Vec<Complex<T>>,
) -> Py<PyAny>
where
    Complex<T>: numpy::Element,
{
    Array2::from_shape_vec((nt, nmdim), buf)
        .expect("shape matches len")
        .into_pyarray(py)
        .into_any()
        .unbind()
}

/// Move a `nt * nlat * nlon` buffer into a new numpy array of shape `(nt, nlat, nlon)` without copying.
fn to_py_grid3<'py, T: numpy::Element>(
    py: Python<'py>,
    nt: usize,
    nlat: usize,
    nlon: usize,
    buf: Vec<T>,
) -> Py<PyAny> {
    Array3::from_shape_vec((nt, nlat, nlon), buf)
        .expect("shape matches len")
        .into_pyarray(py)
        .into_any()
        .unbind()
}

/// Move a `nt * npoints` buffer into a new numpy array of shape `(nt, npoints)` without copying.
fn to_py_grid2<'py, T: numpy::Element>(
    py: Python<'py>,
    nt: usize,
    npoints: usize,
    buf: Vec<T>,
) -> Py<PyAny> {
    Array2::from_shape_vec((nt, npoints), buf)
        .expect("shape matches len")
        .into_pyarray(py)
        .into_any()
        .unbind()
}

/// Gaussian latitudes and quadrature weights for `nlat` points between the poles.
///
/// Parameters: `nlat` (int, >= 1) number of Gaussian latitudes.
///
/// Returns `(lats, wts)`, two `float64` arrays of length `nlat`: `lats` are latitudes in degrees north, descending from north to south (`90 - colatitude`); `wts` are the Gauss-Legendre weights in the same order (they sum to 2).
///
/// Raises `ValueError` if `nlat < 1` (Fortran: `ierror = 1` if `nlat.le.0`).
///
/// Algorithm: Gauss-Legendre points and weights by the Fourier-Newton method of P. N. Swarztrauber, "Computing the points and weights for Gauss-Legendre quadrature", SIAM J. Sci. Comput. 24 (2002) pp. 945-954. Original Fortran comment: "gauss points and weights are computed using the fourier-newton method ... This routine is faster and more accurate than older program with the same name"; the colatitudes lie in the open interval (0, pi) and nlat = 1, 2 are computed analytically.
///
/// Fortran: gaqd.f / gaqd
///
/// Difference from Fortran: `gaqd(nlat, theta, wts, w, lwork, ierror)` returns colatitudes `theta` in increasing radians and has dummy `w`/`lwork` arguments; here the colatitudes are converted to latitudes in degrees (descending) and the computation lives in the `spharm_math` crate.
#[pyfunction]
fn gaussian_lats_wts<'py>(
    py: Python<'py>,
    nlat: isize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    if nlat < 1 {
        return Err(PyValueError::new_err(format!(
            "nlat must be >= 1, got {nlat}"
        )));
    }
    let (lats, wts) = py
        .detach(|| spharm_math::gaussian_lats_wts(nlat as usize))
        .map_err(to_py_err)?;
    Ok((
        Array1::from_vec(lats).into_pyarray(py),
        Array1::from_vec(wts).into_pyarray(py),
    ))
}

/// Normalised associated Legendre functions for all `(m, n)` with `0 <= m <= n <= ntrunc` at one latitude.
///
/// Parameters: `lat_deg` (float) latitude in degrees, `-90 <= lat_deg <= 90`; `ntrunc` (int, >= 0) triangular truncation.
///
/// Returns a `float64` array of length `nmdim = (ntrunc + 1) * (ntrunc + 2) / 2` in the packed order `m` outer, `n` inner (index `m * (2 * ntrunc - m + 3) / 2 + (n - m)`), with the SPHEREPACK normalisation (`P_0^0 = sqrt(1/2)`). The array can be passed as `legfuncs` to `specintrp`.
///
/// Raises `ValueError` if `lat_deg` is outside `[-90, 90]` or not finite, or if `ntrunc < 0`.
///
/// Algorithm: colatitude `theta = pi/2 - (pi/180) * lat` (clamped to `[0, pi]`), then the O(ntrunc^2) Bonnet (m = 0, 1) and Schwarztrauber (m >= 2) recurrences in `spharm_math::legendre_pbar`.
///
/// Fortran: getlegfunc.f / getlegfunc
///
/// Difference from Fortran: the original calls the single-precision `alfk` and `lfpt` for every `(m, n)`, which costs O(ntrunc^3) and underflows to zero for high degrees (n >= 149); the fortran_ref header marks this as a known historical bug, fixed in `getlegfunc.f90`. The Rust version works in `float64` throughout and returns `float64`.
#[pyfunction]
fn getlegfunc<'py>(
    py: Python<'py>,
    lat_deg: f64,
    ntrunc: isize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if !(-90.0..=90.0).contains(&lat_deg) || !lat_deg.is_finite() {
        return Err(PyValueError::new_err(format!(
            "lat must be between -90 and 90 degrees, got {lat_deg}"
        )));
    }
    if ntrunc < 0 {
        return Err(PyValueError::new_err(format!(
            "ntrunc must be non-negative, got {ntrunc}"
        )));
    }
    let colat_rad = (FRAC_PI_2 - lat_deg.to_radians()).clamp(0.0, std::f64::consts::PI);
    let pbar = py
        .detach(|| spharm_math::legendre_pbar(ntrunc as usize, colat_rad))
        .map_err(to_py_err)?;
    Ok(Array1::from_vec(pbar).into_pyarray(py))
}

/// Points of an icosahedral geodesic grid on the sphere.
///
/// Parameters: `m` (int, >= 2) number of points on the edge of a single geodesic triangle. The total number of points is `10 * (m - 1)**2 + 2`, the number of triangles `20 * (m - 1)**2` and the number of edges `30 * (m - 1)**2`.
///
/// Returns `(lats, lons)`, two `float32` arrays with the latitudes and longitudes in degrees of the grid points (see `spharm_math::icosahedral_geodesic` for the output ordering).
///
/// Raises `ValueError` if `m < 2`.
///
/// Algorithm (from the Fortran header): the geodesic points are generated on the unfolded icosahedron (north pole, two rings of vertices at colatitudes `theta1 = acos(beta/(1 - beta))` and `theta2 = pi - theta1` with `beta = cos(2*pi/5)`, south pole), five longitudinal sectors `k = 1..5` of `dphi = 0.4*pi`, each triangle edge divided into `m - 1` equal steps, and every point is then projected back onto the unit sphere.
///
/// Fortran: ihgeod.f / ihgeod
///
/// Difference from Fortran: `ihgeod(m, idp, jdp, x, y, z)` fills Cartesian arrays `x(idp, jdp, 5)`, `y`, `z` with `idp = 2*m - 1`, `jdp = m` indexed by the unfolded-icosahedron indices `(i, j, k)`; here the points are returned as flat latitude and longitude arrays.
#[pyfunction]
fn ihgeod<'py>(
    py: Python<'py>,
    m: isize,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>)> {
    if m < 2 {
        return Err(PyValueError::new_err(format!("m must be >= 2, got {m}")));
    }
    let (lats, lons) = py
        .detach(|| spharm_math::icosahedral_geodesic(m as usize))
        .map_err(to_py_err)?;
    Ok((
        Array1::from_vec(lats).into_pyarray(py),
        Array1::from_vec(lons).into_pyarray(py),
    ))
}

/// Interpolate a spectrally truncated function to a single point.
///
/// Parameters: `rlon_rad` (float) longitude of the point in radians (must be finite); `dataspec` (1-D complex64 or complex128 array, length `nmdim`) packed spectral coefficients; `legfuncs` (1-D float32 or float64 array, same length) associated Legendre functions at the point's latitude, in the same packed order (see `getlegfunc`).
///
/// Returns a Python `float`: the function value at the point (computed in `float64` regardless of the input dtypes).
///
/// Raises `ValueError` if `rlon_rad` is not finite, if the two lengths differ, or if the length is not `(T + 1) * (T + 2) / 2`; `TypeError` if `dataspec` or `legfuncs` has an unsupported dtype or is not 1-D.
///
/// Formula: `f = Re(S_0) + sum_{m=1..ntrunc} 2 * (Re(S_m) cos(m lon) - Im(S_m) sin(m lon))` with `S_m = sum_{n=m..ntrunc} dataspec(m, n) * legfuncs(m, n)`. Fortran comment: "given spectral coeffs of a function (datnm) truncated at T(ntrunc), longitude of pt in radians (rlon), and associated legendre polynomials for that pt (pnm), return interpolated function at that pt (ob)."
///
/// Fortran: specintrp.f / specintrp
///
/// Difference from Fortran: `ntrunc` is inferred from the array length and there is no `scrm` scratch argument; the kernel is `spectral::eval_specintrp`.
#[pyfunction]
fn specintrp<'py>(
    py: Python<'py>,
    rlon_rad: f64,
    dataspec: &Bound<'py, PyAny>,
    legfuncs: &Bound<'py, PyAny>,
) -> PyResult<f64> {
    if !rlon_rad.is_finite() {
        return Err(PyValueError::new_err(format!(
            "rlon_rad must be finite, got {rlon_rad}"
        )));
    }

    enum SpecBorrow<'a> {
        C32(&'a [Complex<f32>]),
        C64(&'a [Complex<f64>]),
    }
    enum LegBorrow<'a> {
        F32(&'a [f32]),
        F64(&'a [f64]),
    }

    // Accept complex64 or complex128 for dataspec: try complex64 first, then complex128.
    let sp32 = dataspec.extract::<PyReadonlyArray1<'py, Complex32>>().ok();
    let sp64 = if sp32.is_none() {
        dataspec.extract::<PyReadonlyArray1<'py, Complex64>>().ok()
    } else {
        None
    };
    if sp32.is_none() && sp64.is_none() {
        return Err(PyTypeError::new_err(
            "dataspec must be a 1D complex64 or complex128 numpy array",
        ));
    }

    // Same for legfuncs: float32 first, then float64.
    let lg32 = legfuncs.extract::<PyReadonlyArray1<'py, f32>>().ok();
    let lg64 = if lg32.is_none() {
        legfuncs.extract::<PyReadonlyArray1<'py, f64>>().ok()
    } else {
        None
    };
    if lg32.is_none() && lg64.is_none() {
        return Err(PyTypeError::new_err(
            "legfuncs must be a 1D float32 or float64 numpy array",
        ));
    }

    // Borrow each input as a contiguous slice when possible, otherwise copy into an owned Vec (non-contiguous numpy view).
    let sp64_own: Vec<Complex<f64>>;
    let sp32_own: Vec<Complex<f32>>;
    let (len_s, sp) = if let Some(ref a) = sp32 {
        let len = a.len();
        let b = if let Ok(sl) = a.as_slice() {
            SpecBorrow::C32(slice_c32(sl))
        } else {
            sp32_own = a.as_array().iter().map(|c| Complex::new(c.re, c.im)).collect();
            SpecBorrow::C32(&sp32_own)
        };
        (len, b)
    } else {
        let a = sp64.as_ref().unwrap();
        let len = a.len();
        let b = if let Ok(sl) = a.as_slice() {
            SpecBorrow::C64(slice_c64(sl))
        } else {
            sp64_own = a.as_array().iter().map(|c| Complex::new(c.re, c.im)).collect();
            SpecBorrow::C64(&sp64_own)
        };
        (len, b)
    };

    let lg64_own: Vec<f64>;
    let lg32_own: Vec<f32>;
    let (len_l, lg) = if let Some(ref a) = lg32 {
        let len = a.len();
        let b = if let Ok(sl) = a.as_slice() {
            LegBorrow::F32(sl)
        } else {
            lg32_own = a.as_array().iter().copied().collect();
            LegBorrow::F32(&lg32_own)
        };
        (len, b)
    } else {
        let a = lg64.as_ref().unwrap();
        let len = a.len();
        let b = if let Ok(sl) = a.as_slice() {
            LegBorrow::F64(sl)
        } else {
            lg64_own = a.as_array().iter().copied().collect();
            LegBorrow::F64(&lg64_own)
        };
        (len, b)
    };

    // Fortran: pnm and datnm both have (ntrunc+1)*(ntrunc+2)/2 elements.
    if len_s != len_l {
        return Err(PyValueError::new_err(format!(
            "dataspec length ({len_s}) must match legfuncs length ({len_l})"
        )));
    }
    let ntrunc = ntrunc_from_nmdim(len_s).map_err(to_py_err)?;
    // Release the GIL only for large inputs.
    let detach = len_s >= 65_536;

    // Monomorphised dispatch over (spectral dtype, Legendre dtype); the kernel always accumulates in f64.
    let out = match (sp, lg) {
        (SpecBorrow::C32(s), LegBorrow::F32(l)) => {
            maybe_detach(py, detach, || eval_specintrp(rlon_rad, s, l, ntrunc))
        }
        (SpecBorrow::C32(s), LegBorrow::F64(l)) => {
            maybe_detach(py, detach, || eval_specintrp(rlon_rad, s, l, ntrunc))
        }
        (SpecBorrow::C64(s), LegBorrow::F32(l)) => {
            maybe_detach(py, detach, || eval_specintrp(rlon_rad, s, l, ntrunc))
        }
        (SpecBorrow::C64(s), LegBorrow::F64(l)) => {
            maybe_detach(py, detach, || eval_specintrp(rlon_rad, s, l, ntrunc))
        }
    };
    Ok(out)
}

/// Shared implementation of `lap` and `invlap`: dispatch on dtype and rank, then apply the Laplacian (`inverse == false`) or inverse Laplacian (`inverse == true`) factor per degree.
///
/// Accepts a 1-D `(nmdim,)` or 2-D `(nt, nmdim)` complex64/complex128 array; the result has the same shape and dtype. `op_name` is only used in error messages. Raises `ValueError` for an invalid `rsphere`, an `nmdim` that is not triangular, or a rank other than 1 or 2; `TypeError` for a non-complex dtype.
///
/// Small problems (fewer than 262144 complex elements) write straight into an uninitialised numpy output while holding the GIL; larger ones compute into a Vec with the GIL released.
///
/// Fortran: lap.f / lap and invlap.f / invlap
fn run_lap_like<'py>(
    py: Python<'py>,
    dataspec: &Bound<'py, PyAny>,
    rsphere: f64,
    inverse: bool,
    op_name: &str,
) -> PyResult<Py<PyAny>> {
    let rsphere = validate_rsphere(rsphere)?;
    // complex64, shape (nmdim,)
    if let Ok(arr32_1d) = dataspec.extract::<PyReadonlyArray1<'py, Complex32>>() {
        let nmdim = arr32_1d.len();
        let ntrunc = ntrunc_from_nmdim(nmdim).map_err(to_py_err)?;
        let own;
        let src: &[Complex<f32>] = if let Ok(sl) = arr32_1d.as_slice() {
            slice_c32(sl)
        } else {
            own = arr32_1d
                .as_array()
                .iter()
                .map(|c| Complex::new(c.re, c.im))
                .collect::<Vec<_>>();
            &own
        };
        if nmdim < 262_144 {
            let out_arr = unsafe { PyArray1::<Complex32>::new(py, [nmdim], false) };
            {
                let dst = slice_c32_mut(unsafe { out_arr.as_slice_mut().unwrap() });
                if inverse {
                    apply_invlap_batch(src, dst, 1, nmdim, ntrunc, rsphere);
                } else {
                    apply_lap_batch(src, dst, 1, nmdim, ntrunc, rsphere);
                }
            }
            return Ok(out_arr.into_any().unbind());
        }
        let mut dst = vec![Complex::new(0.0_f32, 0.0_f32); nmdim];
        py.detach(|| {
            if inverse {
                apply_invlap_batch(src, &mut dst, 1, nmdim, ntrunc, rsphere);
            } else {
                apply_lap_batch(src, &mut dst, 1, nmdim, ntrunc, rsphere);
            }
        });
        return Ok(Array1::from_vec(dst).into_pyarray(py).into_any().unbind());
    }
    // complex64, shape (nt, nmdim)
    if let Ok(arr32_2d) = dataspec.extract::<PyReadonlyArray2<'py, Complex32>>() {
        let shape = arr32_2d.shape();
        let nt = shape[0];
        let nmdim = shape[1];
        let ntrunc = ntrunc_from_nmdim(nmdim).map_err(to_py_err)?;
        let own;
        let src: &[Complex<f32>] = if let Ok(sl) = arr32_2d.as_slice() {
            slice_c32(sl)
        } else {
            own = arr32_2d
                .as_array()
                .iter()
                .map(|c| Complex::new(c.re, c.im))
                .collect::<Vec<_>>();
            &own
        };
        if nt * nmdim < 262_144 {
            let out_arr = unsafe { PyArray2::<Complex32>::new(py, [nt, nmdim], false) };
            {
                let dst = slice_c32_mut(unsafe { out_arr.as_slice_mut().unwrap() });
                if inverse {
                    apply_invlap_batch(src, dst, nt, nmdim, ntrunc, rsphere);
                } else {
                    apply_lap_batch(src, dst, nt, nmdim, ntrunc, rsphere);
                }
            }
            return Ok(out_arr.into_any().unbind());
        }
        let mut dst = vec![Complex::new(0.0_f32, 0.0_f32); nt * nmdim];
        py.detach(|| {
            if inverse {
                apply_invlap_batch(src, &mut dst, nt, nmdim, ntrunc, rsphere);
            } else {
                apply_lap_batch(src, &mut dst, nt, nmdim, ntrunc, rsphere);
            }
        });
        return Ok(to_py_spec2(py, nt, nmdim, dst));
    }
    // complex128, shape (nmdim,)
    if let Ok(arr64_1d) = dataspec.extract::<PyReadonlyArray1<'py, Complex64>>() {
        let nmdim = arr64_1d.len();
        let ntrunc = ntrunc_from_nmdim(nmdim).map_err(to_py_err)?;
        let own;
        let src: &[Complex<f64>] = if let Ok(sl) = arr64_1d.as_slice() {
            slice_c64(sl)
        } else {
            own = arr64_1d
                .as_array()
                .iter()
                .map(|c| Complex::new(c.re, c.im))
                .collect::<Vec<_>>();
            &own
        };
        if nmdim < 262_144 {
            let out_arr = unsafe { PyArray1::<Complex64>::new(py, [nmdim], false) };
            {
                let dst = slice_c64_mut(unsafe { out_arr.as_slice_mut().unwrap() });
                if inverse {
                    apply_invlap_batch(src, dst, 1, nmdim, ntrunc, rsphere);
                } else {
                    apply_lap_batch(src, dst, 1, nmdim, ntrunc, rsphere);
                }
            }
            return Ok(out_arr.into_any().unbind());
        }
        let mut dst = vec![Complex::new(0.0_f64, 0.0_f64); nmdim];
        py.detach(|| {
            if inverse {
                apply_invlap_batch(src, &mut dst, 1, nmdim, ntrunc, rsphere);
            } else {
                apply_lap_batch(src, &mut dst, 1, nmdim, ntrunc, rsphere);
            }
        });
        return Ok(Array1::from_vec(dst).into_pyarray(py).into_any().unbind());
    }
    // complex128, shape (nt, nmdim)
    if let Ok(arr64_2d) = dataspec.extract::<PyReadonlyArray2<'py, Complex64>>() {
        let shape = arr64_2d.shape();
        let nt = shape[0];
        let nmdim = shape[1];
        let ntrunc = ntrunc_from_nmdim(nmdim).map_err(to_py_err)?;
        let own;
        let src: &[Complex<f64>] = if let Ok(sl) = arr64_2d.as_slice() {
            slice_c64(sl)
        } else {
            own = arr64_2d
                .as_array()
                .iter()
                .map(|c| Complex::new(c.re, c.im))
                .collect::<Vec<_>>();
            &own
        };
        if nt * nmdim < 262_144 {
            let out_arr = unsafe { PyArray2::<Complex64>::new(py, [nt, nmdim], false) };
            {
                let dst = slice_c64_mut(unsafe { out_arr.as_slice_mut().unwrap() });
                if inverse {
                    apply_invlap_batch(src, dst, nt, nmdim, ntrunc, rsphere);
                } else {
                    apply_lap_batch(src, dst, nt, nmdim, ntrunc, rsphere);
                }
            }
            return Ok(out_arr.into_any().unbind());
        }
        let mut dst = vec![Complex::new(0.0_f64, 0.0_f64); nt * nmdim];
        py.detach(|| {
            if inverse {
                apply_invlap_batch(src, &mut dst, nt, nmdim, ntrunc, rsphere);
            } else {
                apply_lap_batch(src, &mut dst, nt, nmdim, ntrunc, rsphere);
            }
        });
        return Ok(to_py_spec2(py, nt, nmdim, dst));
    }
    // No supported dtype and rank matched: report a rank error if ndim is not 1 or 2, otherwise a dtype error.
    let nd = dataspec.getattr("ndim").and_then(|v| v.extract::<usize>()).ok();
    if let Some(d) = nd {
        if d != 1 && d != 2 {
            return Err(PyValueError::new_err(format!(
                "{op_name} expects a 1D (nmdim,) or 2D (nt, nmdim) spectral array, got {d}D"
            )));
        }
    }
    Err(PyTypeError::new_err(format!(
        "{op_name} dataspec must be a 1D or 2D complex64 or complex128 numpy array"
    )))
}

/// Spherical Laplacian of fields given by their spectral coefficients.
///
/// Parameters: `dataspec` (complex64 or complex128, shape `(nmdim,)` or `(nt, nmdim)`) packed triangular spectral coefficients; `rsphere` (float, > 0) sphere radius.
///
/// Returns an array of the same shape and dtype with every coefficient `(m, n)` multiplied by `-n * (n + 1) / rsphere**2` (0-based degree `n`).
///
/// Raises `ValueError` if `rsphere` is not positive and finite, `nmdim` is not `(T + 1) * (T + 2) / 2`, or the rank is not 1 or 2; `TypeError` for a non-complex dtype.
///
/// Fortran: lap.f / lap. Original expression `-(float(n)*float(n-1)/rsphere**2)*dataspec(nm,i)` with 1-based `n`, over `do m=1,ntrunc+1 / do n=m,ntrunc+1`.
///
/// Difference from Fortran: 0-based indexing, `ntrunc` inferred from `nmdim` as in Fortran, but invalid sizes raise instead of being truncated.
#[pyfunction]
fn lap<'py>(
    py: Python<'py>,
    dataspec: &Bound<'py, PyAny>,
    rsphere: f64,
) -> PyResult<Py<PyAny>> {
    run_lap_like(py, dataspec, rsphere, false, "lap")
}

/// Inverse spherical Laplacian of fields given by their spectral coefficients.
///
/// Parameters: `dataspec` (complex64 or complex128, shape `(nmdim,)` or `(nt, nmdim)`); `rsphere` (float, > 0) sphere radius.
///
/// Returns an array of the same shape and dtype with every coefficient `(m, n)` multiplied by `-rsphere**2 / (n * (n + 1))` for `n >= 1`; the `n = 0` coefficient (the global mean, only `(0, 0)`) is set to zero.
///
/// Raises as `lap`.
///
/// Fortran: invlap.f / invlap. Original expression `-(rsphere**2/(float(n)*float(n-1)))*dataspec(nm,i)`, skipping `n = 1` (1-based) for `m = 1` and finishing with `dataspec_ilap(1,i) = (0.,0.)`.
#[pyfunction]
fn invlap<'py>(
    py: Python<'py>,
    dataspec: &Bound<'py, PyAny>,
    rsphere: f64,
) -> PyResult<Py<PyAny>> {
    run_lap_like(py, dataspec, rsphere, true, "invlap")
}

/// Multiply spectral coefficients by a degree-dependent smoothing (filter) factor.
///
/// Parameters: `dataspec` (complex64 or complex128, shape `(nmdim,)` or `(nt, nmdim)`) packed spectral coefficients; `smooth` (1-D float32 or float64 array) filter indexed by total wavenumber `n`, with `len(smooth) >= ntrunc + 1` (extra entries are ignored).
///
/// Returns an array of the same shape and dtype as `dataspec` with coefficient `(m, n)` multiplied by `smooth[n]`; the dtype of `smooth` does not change the output dtype.
///
/// Raises `ValueError` if `smooth` is shorter than `ntrunc + 1`, `nmdim` is not `(T + 1) * (T + 2) / 2`, or the rank is not 1 or 2; `TypeError` for unsupported dtypes (`smooth` must be real, `dataspec` complex).
///
/// Fortran: multsmoothfact.f / multsmoothfact, `dataspec_smooth(nm,i) = dataspec(nm,i)*smooth(n)` with `smooth(nlat)` and 1-based `n`.
///
/// Difference from Fortran: `nlat` is not an argument; only `ntrunc + 1` factors are needed.
#[pyfunction]
fn multsmoothfact<'py>(
    py: Python<'py>,
    dataspec: &Bound<'py, PyAny>,
    smooth: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    enum SmoothBorrow<'a> {
        F32(&'a [f32]),
        F64(&'a [f64]),
    }
    // smooth may be float32 or float64; it is borrowed (or copied if non-contiguous) and converted per element to the coefficient precision inside the kernel.
    let sm32 = smooth.extract::<PyReadonlyArray1<'py, f32>>().ok();
    let sm64 = if sm32.is_none() {
        smooth.extract::<PyReadonlyArray1<'py, f64>>().ok()
    } else {
        None
    };
    if sm32.is_none() && sm64.is_none() {
        return Err(PyTypeError::new_err(
            "smooth must be a 1D float32 or float64 numpy array",
        ));
    }

    let sm64_own: Vec<f64>;
    let sm32_own: Vec<f32>;
    let (smooth_len, sm) = if let Some(ref a) = sm32 {
        let len = a.len();
        let b = if let Ok(sl) = a.as_slice() {
            SmoothBorrow::F32(sl)
        } else {
            sm32_own = a.as_array().iter().copied().collect();
            SmoothBorrow::F32(&sm32_own)
        };
        (len, b)
    } else {
        let a = sm64.as_ref().unwrap();
        let len = a.len();
        let b = if let Ok(sl) = a.as_slice() {
            SmoothBorrow::F64(sl)
        } else {
            sm64_own = a.as_array().iter().copied().collect();
            SmoothBorrow::F64(&sm64_own)
        };
        (len, b)
    };

    // complex64, shape (nmdim,)
    if let Ok(arr32_1d) = dataspec.extract::<PyReadonlyArray1<'py, Complex32>>() {
        let nmdim = arr32_1d.len();
        let ntrunc = ntrunc_from_nmdim(nmdim).map_err(to_py_err)?;
        if smooth_len <= ntrunc {
            return Err(PyValueError::new_err(format!(
                "smooth length ({smooth_len}) must be >= ntrunc + 1 ({})",
                ntrunc + 1
            )));
        }
        let own;
        let src: &[Complex<f32>] = if let Ok(sl) = arr32_1d.as_slice() {
            slice_c32(sl)
        } else {
            own = arr32_1d
                .as_array()
                .iter()
                .map(|c| Complex::new(c.re, c.im))
                .collect::<Vec<_>>();
            &own
        };
        if nmdim < 262_144 {
            let out_arr = unsafe { PyArray1::<Complex32>::new(py, [nmdim], false) };
            {
                let dst = slice_c32_mut(unsafe { out_arr.as_slice_mut().unwrap() });
                match &sm {
                    SmoothBorrow::F32(s) => apply_multsmoothfact_batch(src, s, dst, 1, nmdim, ntrunc),
                    SmoothBorrow::F64(s) => apply_multsmoothfact_batch(src, s, dst, 1, nmdim, ntrunc),
                }
            }
            return Ok(out_arr.into_any().unbind());
        }
        let mut dst = vec![Complex::new(0.0_f32, 0.0_f32); nmdim];
        py.detach(|| match &sm {
            SmoothBorrow::F32(s) => apply_multsmoothfact_batch(src, s, &mut dst, 1, nmdim, ntrunc),
            SmoothBorrow::F64(s) => apply_multsmoothfact_batch(src, s, &mut dst, 1, nmdim, ntrunc),
        });
        return Ok(Array1::from_vec(dst).into_pyarray(py).into_any().unbind());
    }
    // complex64, shape (nt, nmdim)
    if let Ok(arr32_2d) = dataspec.extract::<PyReadonlyArray2<'py, Complex32>>() {
        let shape = arr32_2d.shape();
        let nt = shape[0];
        let nmdim = shape[1];
        let ntrunc = ntrunc_from_nmdim(nmdim).map_err(to_py_err)?;
        if smooth_len <= ntrunc {
            return Err(PyValueError::new_err(format!(
                "smooth length ({smooth_len}) must be >= ntrunc + 1 ({})",
                ntrunc + 1
            )));
        }
        let own;
        let src: &[Complex<f32>] = if let Ok(sl) = arr32_2d.as_slice() {
            slice_c32(sl)
        } else {
            own = arr32_2d
                .as_array()
                .iter()
                .map(|c| Complex::new(c.re, c.im))
                .collect::<Vec<_>>();
            &own
        };
        if nt * nmdim < 262_144 {
            let out_arr = unsafe { PyArray2::<Complex32>::new(py, [nt, nmdim], false) };
            {
                let dst = slice_c32_mut(unsafe { out_arr.as_slice_mut().unwrap() });
                match &sm {
                    SmoothBorrow::F32(s) => apply_multsmoothfact_batch(src, s, dst, nt, nmdim, ntrunc),
                    SmoothBorrow::F64(s) => apply_multsmoothfact_batch(src, s, dst, nt, nmdim, ntrunc),
                }
            }
            return Ok(out_arr.into_any().unbind());
        }
        let mut dst = vec![Complex::new(0.0_f32, 0.0_f32); nt * nmdim];
        py.detach(|| match &sm {
            SmoothBorrow::F32(s) => apply_multsmoothfact_batch(src, s, &mut dst, nt, nmdim, ntrunc),
            SmoothBorrow::F64(s) => apply_multsmoothfact_batch(src, s, &mut dst, nt, nmdim, ntrunc),
        });
        return Ok(to_py_spec2(py, nt, nmdim, dst));
    }
    // complex128, shape (nmdim,)
    if let Ok(arr64_1d) = dataspec.extract::<PyReadonlyArray1<'py, Complex64>>() {
        let nmdim = arr64_1d.len();
        let ntrunc = ntrunc_from_nmdim(nmdim).map_err(to_py_err)?;
        if smooth_len <= ntrunc {
            return Err(PyValueError::new_err(format!(
                "smooth length ({smooth_len}) must be >= ntrunc + 1 ({})",
                ntrunc + 1
            )));
        }
        let own;
        let src: &[Complex<f64>] = if let Ok(sl) = arr64_1d.as_slice() {
            slice_c64(sl)
        } else {
            own = arr64_1d
                .as_array()
                .iter()
                .map(|c| Complex::new(c.re, c.im))
                .collect::<Vec<_>>();
            &own
        };
        if nmdim < 262_144 {
            let out_arr = unsafe { PyArray1::<Complex64>::new(py, [nmdim], false) };
            {
                let dst = slice_c64_mut(unsafe { out_arr.as_slice_mut().unwrap() });
                match &sm {
                    SmoothBorrow::F32(s) => apply_multsmoothfact_batch(src, s, dst, 1, nmdim, ntrunc),
                    SmoothBorrow::F64(s) => apply_multsmoothfact_batch(src, s, dst, 1, nmdim, ntrunc),
                }
            }
            return Ok(out_arr.into_any().unbind());
        }
        let mut dst = vec![Complex::new(0.0_f64, 0.0_f64); nmdim];
        py.detach(|| match &sm {
            SmoothBorrow::F32(s) => apply_multsmoothfact_batch(src, s, &mut dst, 1, nmdim, ntrunc),
            SmoothBorrow::F64(s) => apply_multsmoothfact_batch(src, s, &mut dst, 1, nmdim, ntrunc),
        });
        return Ok(Array1::from_vec(dst).into_pyarray(py).into_any().unbind());
    }
    // complex128, shape (nt, nmdim)
    if let Ok(arr64_2d) = dataspec.extract::<PyReadonlyArray2<'py, Complex64>>() {
        let shape = arr64_2d.shape();
        let nt = shape[0];
        let nmdim = shape[1];
        let ntrunc = ntrunc_from_nmdim(nmdim).map_err(to_py_err)?;
        if smooth_len <= ntrunc {
            return Err(PyValueError::new_err(format!(
                "smooth length ({smooth_len}) must be >= ntrunc + 1 ({})",
                ntrunc + 1
            )));
        }
        let own;
        let src: &[Complex<f64>] = if let Ok(sl) = arr64_2d.as_slice() {
            slice_c64(sl)
        } else {
            own = arr64_2d
                .as_array()
                .iter()
                .map(|c| Complex::new(c.re, c.im))
                .collect::<Vec<_>>();
            &own
        };
        if nt * nmdim < 262_144 {
            let out_arr = unsafe { PyArray2::<Complex64>::new(py, [nt, nmdim], false) };
            {
                let dst = slice_c64_mut(unsafe { out_arr.as_slice_mut().unwrap() });
                match &sm {
                    SmoothBorrow::F32(s) => apply_multsmoothfact_batch(src, s, dst, nt, nmdim, ntrunc),
                    SmoothBorrow::F64(s) => apply_multsmoothfact_batch(src, s, dst, nt, nmdim, ntrunc),
                }
            }
            return Ok(out_arr.into_any().unbind());
        }
        let mut dst = vec![Complex::new(0.0_f64, 0.0_f64); nt * nmdim];
        py.detach(|| match &sm {
            SmoothBorrow::F32(s) => apply_multsmoothfact_batch(src, s, &mut dst, nt, nmdim, ntrunc),
            SmoothBorrow::F64(s) => apply_multsmoothfact_batch(src, s, &mut dst, nt, nmdim, ntrunc),
        });
        return Ok(to_py_spec2(py, nt, nmdim, dst));
    }

    // No supported dtype and rank matched: report a rank error if ndim is not 1 or 2, otherwise a dtype error.
    let nd = dataspec.getattr("ndim").and_then(|v| v.extract::<usize>()).ok();
    if let Some(d) = nd {
        if d != 1 && d != 2 {
            return Err(PyValueError::new_err(format!(
                "multsmoothfact expects a 1D (nmdim,) or 2D (nt, nmdim) spectral array, got {d}D"
            )));
        }
    }
    Err(PyTypeError::new_err(
        "multsmoothfact dataspec must be a 1D or 2D complex64 or complex128 numpy array",
    ))
}

/// Spherical-harmonic transform engine for a full latitude-longitude grid (regular or Gaussian).
///
/// Created with `RustSpharmCore(nlon, nlat, rsphere=6371200.0, gridtype="regular", legfunc="stored")`. Grid fields are `(nt, nlat, nlon)` arrays (float32 or float64), spectral fields are `(nt, nmdim)` complex arrays in the triangular packing described in the module docs. Methods: `grdtospec`, `spectogrd`, `getvrtdivspec`, `getvrtspec`, `getdivspec`, `getuv`, `getpsichi`, `getgrad`; read-only attributes `nlon`, `nlat`, `rsphere`, `gridtype`, `legfunc`.
///
/// Fortran: scalar analysis/synthesis `shaec`/`shaes`/`shagc`/`shags` and `shsec`/`shses`/`shsgc`/`shsgs`; vector analysis/synthesis `vhaec`/`vhaes`/`vhagc`/`vhags` and `vhsec`/`vhses`/`vhsgc`/`vhsgs`; spectral packing `onedtotwod*.f`, `twodtooned*.f`.
///
/// Difference from Fortran: one engine covers every grid/Legendre variant (SPHEREPACK has a separate routine pair and `*i` initialiser per variant), tables are built lazily on the first call per dtype and reused, longitude transforms use `rustfft`/`realfft` instead of `hrfft.f`, batches are C-order `(nt, ...)`, fields are processed in tiles with rayon parallelism, and there are no `wsave`/`work` arrays.
#[pyclass(frozen, module = "skybornspharm._spharm_rs")]
pub struct RustSpharmCore {
    engine: FullGridEngine,
}

// Internal helpers (not exposed to Python): typed workers and dtype dispatch.
impl RustSpharmCore {
    /// Typed worker for `grdtospec`: validate the `(nt, nlat, nlon)` shape, allocate the `(nt, nmdim)` output and run the engine analysis, releasing the GIL for large problems.
    ///
    /// Fortran: `shaec`/`shaes`/`shagc`/`shags` followed by `twodtooned.f` / `twodtooned` packing. No direct Fortran counterpart for the batch/dtype handling.
    fn run_grdtospec<'py, T: TypedTables + numpy::Element>(
        &self,
        py: Python<'py>,
        grid: PyReadonlyArray3<'py, T>,
        ntrunc: usize,
        south_to_north: bool,
    ) -> PyResult<Py<PyAny>>
    where
        Complex<T>: numpy::Element,
    {
        let view = grid.as_array();
        let nt = validate_grid3_shape(view.shape(), self.engine.nlat, self.engine.nlon, "grdtospec")?;
        let nmdim = nmdim_from_ntrunc(ntrunc);
        let mut out = vec![T::complex_zero(); nt * nmdim];
        let detach = nt * self.engine.nlat * self.engine.nlon >= engine::PAR_WORK_THRESHOLD;
        maybe_detach(py, detach, || {
            self.engine.grdtospec(&view, &mut out, ntrunc, south_to_north)
        });
        Ok(to_py_spec2(py, nt, nmdim, out))
    }

    /// Typed worker for `getvrtdivspec`, `getvrtspec` and `getdivspec` on the full grid: validate that `u` and `v` are `(nt, nlat, nlon)` and equal in shape, then compute vorticity and/or divergence spectra according to `mode` (`VrtDiv` returns a `(vrt, div)` tuple, `VrtOnly` and `DivOnly` return a single array).
    ///
    /// Fortran: `vhaec`/`vhaes`/`vhagc`/`vhags` followed by `twodtooned_vrtdiv.f`, which scales by `sqrt(n * (n + 1)) / rsphere`. No direct Fortran counterpart for the mode selection and batch handling.
    fn run_vec_analysis<'py, T: TypedTables + numpy::Element>(
        &self,
        py: Python<'py>,
        u: ArrayView3<'_, T>,
        v: ArrayView3<'_, T>,
        ntrunc: usize,
        south_to_north: bool,
        mode: VectorMode,
        op_name: &str,
    ) -> PyResult<Py<PyAny>>
    where
        Complex<T>: numpy::Element,
    {
        let nt = validate_grid3_shape(u.shape(), self.engine.nlat, self.engine.nlon, op_name)?;
        if u.shape() != v.shape() {
            return Err(PyValueError::new_err(format!(
                "ugrid shape {:?} must match vgrid shape {:?}",
                u.shape(),
                v.shape()
            )));
        }
        let nmdim = nmdim_from_ntrunc(ntrunc);
        let detach = nt * self.engine.nlat * self.engine.nlon >= engine::PAR_WORK_THRESHOLD;
        match mode {
            VectorMode::VrtDiv => {
                let mut vrt = vec![T::complex_zero(); nt * nmdim];
                let mut div = vec![T::complex_zero(); nt * nmdim];
                maybe_detach(py, detach, || {
                    self.engine.getvrtdivspec(
                        &u,
                        &v,
                        Some(&mut vrt),
                        Some(&mut div),
                        ntrunc,
                        south_to_north,
                        VectorMode::VrtDiv,
                    );
                });
                let pv = to_py_spec2(py, nt, nmdim, vrt);
                let pd = to_py_spec2(py, nt, nmdim, div);
                Ok((pv, pd).into_pyobject(py)?.into_any().unbind())
            }
            VectorMode::VrtOnly => {
                let mut vrt = vec![T::complex_zero(); nt * nmdim];
                maybe_detach(py, detach, || {
                    self.engine.getvrtdivspec(
                        &u,
                        &v,
                        Some(&mut vrt),
                        None,
                        ntrunc,
                        south_to_north,
                        VectorMode::VrtOnly,
                    );
                });
                Ok(to_py_spec2(py, nt, nmdim, vrt))
            }
            VectorMode::DivOnly => {
                let mut div = vec![T::complex_zero(); nt * nmdim];
                maybe_detach(py, detach, || {
                    self.engine.getvrtdivspec(
                        &u,
                        &v,
                        None,
                        Some(&mut div),
                        ntrunc,
                        south_to_north,
                        VectorMode::DivOnly,
                    );
                });
                Ok(to_py_spec2(py, nt, nmdim, div))
            }
        }
    }

    /// Typed worker for `getpsichi`: from `u`, `v` of shape `(nt, nlat, nlon)` compute vorticity and divergence spectra, apply the inverse Laplacian with `rsphere`, and synthesise the streamfunction `psi` and velocity potential `chi` grids (`psi = invlap(vrt)`, `chi = invlap(div)`).
    ///
    /// The `nt` fields are processed in tiles (`compute_tile_size`) so the four spectral work buffers stay within a memory budget.
    ///
    /// No direct Fortran counterpart (a composition of `vhaec`-type analysis, `invlap.f` and `shsec`-type synthesis).
    fn run_getpsichi<'py, T: TypedTables + numpy::Element>(
        &self,
        py: Python<'py>,
        u: ArrayView3<'_, T>,
        v: ArrayView3<'_, T>,
        ntrunc: usize,
        south_to_north: bool,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let nt = validate_grid3_shape(u.shape(), self.engine.nlat, self.engine.nlon, "getpsichi")?;
        if u.shape() != v.shape() {
            return Err(PyValueError::new_err(format!(
                "ugrid shape {:?} must match vgrid shape {:?}",
                u.shape(),
                v.shape()
            )));
        }
        let nmdim = nmdim_from_ntrunc(ntrunc);
        let rsphere = self.engine.rsphere;
        let nlat = self.engine.nlat;
        let nlon = self.engine.nlon;
        let plane = nlat * nlon;
        let mut psi = vec![T::ZERO; nt * plane];
        let mut chi = vec![T::ZERO; nt * plane];

        if nt > 0 {
            maybe_detach(py, nt * plane >= engine::PAR_WORK_THRESHOLD, || {
                // Four spectral work buffers per field (vrt, div, psispec, chispec); choose how many fields to process per tile.
                let per_field_bytes = 4 * nmdim * std::mem::size_of::<Complex<T>>();
                let tile_max = compute_tile_size(per_field_bytes, nt);
                let mut vrt = vec![T::complex_zero(); tile_max * nmdim];
                let mut div = vec![T::complex_zero(); tile_max * nmdim];
                let mut psispec = vec![T::complex_zero(); tile_max * nmdim];
                let mut chispec = vec![T::complex_zero(); tile_max * nmdim];

                for start in (0..nt).step_by(tile_max) {
                    let t = (nt - start).min(tile_max);
                    // Per tile: (1) vorticity and divergence spectra of (u, v); (2) inverse Laplacian with rsphere of each (psi = invlap(vrt), chi = invlap(div), invlap.f); (3) synthesise psi and chi on the grid.
                    let u_sub = u.slice(s![start..start + t, .., ..]);
                    let v_sub = v.slice(s![start..start + t, .., ..]);
                    self.engine.getvrtdivspec(
                        &u_sub,
                        &v_sub,
                        Some(&mut vrt[..t * nmdim]),
                        Some(&mut div[..t * nmdim]),
                        ntrunc,
                        south_to_north,
                        VectorMode::VrtDiv,
                    );
                    apply_invlap_batch(
                        &vrt[..t * nmdim],
                        &mut psispec[..t * nmdim],
                        t,
                        nmdim,
                        ntrunc,
                        rsphere,
                    );
                    apply_invlap_batch(
                        &div[..t * nmdim],
                        &mut chispec[..t * nmdim],
                        t,
                        nmdim,
                        ntrunc,
                        rsphere,
                    );
                    let psi_v = ArrayView2::from_shape((t, nmdim), &psispec[..t * nmdim]).unwrap();
                    let chi_v = ArrayView2::from_shape((t, nmdim), &chispec[..t * nmdim]).unwrap();
                    self.engine.spectogrd(
                        &psi_v,
                        ntrunc,
                        &mut psi[start * plane..(start + t) * plane],
                        south_to_north,
                    );
                    self.engine.spectogrd(
                        &chi_v,
                        ntrunc,
                        &mut chi[start * plane..(start + t) * plane],
                        south_to_north,
                    );
                }
            });
        }

        Ok((
            to_py_grid3(py, nt, nlat, nlon, psi),
            to_py_grid3(py, nt, nlat, nlon, chi),
        ))
    }

    /// Dtype dispatcher for the full-grid vector analysis methods: resolve `ntrunc` (`None` means `nlat - 1`), extract `ugrid` and `vgrid` as float32 or float64 3-D arrays and call `run_vec_analysis`.
    ///
    /// Both float64 or both float32 run in that precision; mixed dtypes promote the float32 array to float64. Raises `TypeError` (using `op_name`) if either argument is not a 3-D float32/float64 array.
    ///
    /// No direct Fortran counterpart.
    fn dispatch_vec_analysis<'py>(
        &self,
        py: Python<'py>,
        ugrid: &Bound<'py, PyAny>,
        vgrid: &Bound<'py, PyAny>,
        ntrunc: Option<isize>,
        south_to_north: bool,
        mode: VectorMode,
        op_name: &str,
    ) -> PyResult<Py<PyAny>> {
        let ntrunc = resolve_ntrunc(ntrunc, self.engine.nlat - 1)?;
        let u32 = ugrid.extract::<PyReadonlyArray3<'py, f32>>().ok();
        let v32 = vgrid.extract::<PyReadonlyArray3<'py, f32>>().ok();
        let u64 = if u32.is_none() {
            ugrid.extract::<PyReadonlyArray3<'py, f64>>().ok()
        } else {
            None
        };
        let v64 = if v32.is_none() {
            vgrid.extract::<PyReadonlyArray3<'py, f64>>().ok()
        } else {
            None
        };
        // Same dtype for u and v runs natively; mixed dtypes are handled by the two promoting arms below.
        match (u64, u32, v64, v32) {
            (Some(u), None, Some(v), None) => self.run_vec_analysis(
                py,
                u.as_array(),
                v.as_array(),
                ntrunc,
                south_to_north,
                mode,
                op_name,
            ),
            (None, Some(u), None, Some(v)) => self.run_vec_analysis(
                py,
                u.as_array(),
                v.as_array(),
                ntrunc,
                south_to_north,
                mode,
                op_name,
            ),
            // u is float64, v is float32: v is promoted to float64.
            (Some(u), None, None, Some(v)) => {
                let vo = v.as_array().mapv(|x| x as f64);
                self.run_vec_analysis(
                    py,
                    u.as_array(),
                    vo.view(),
                    ntrunc,
                    south_to_north,
                    mode,
                    op_name,
                )
            }
            // u is float32, v is float64: u is promoted to float64.
            (None, Some(u), Some(v), None) => {
                let uo = u.as_array().mapv(|x| x as f64);
                self.run_vec_analysis(
                    py,
                    uo.view(),
                    v.as_array(),
                    ntrunc,
                    south_to_north,
                    mode,
                    op_name,
                )
            }
            _ => Err(PyTypeError::new_err(format!(
                "{op_name} expects 3D (nt, nlat, nlon) float32 or float64 arrays"
            ))),
        }
    }
}

#[pymethods]
// Python-visible methods.
impl RustSpharmCore {
    /// Create a full-grid transform engine.
    ///
    /// Parameters: `nlon` (int, >= 4) number of longitudes; `nlat` (int, >= 3) number of latitudes; `rsphere` (float, > 0, default 6371200.0) sphere radius in metres, used by the vorticity/divergence/gradient scalings and by the Laplacian-type operators; `gridtype` (`"regular"` for equally spaced latitudes or `"gaussian"`); `legfunc` (`"stored"` or `"computed"`).
    ///
    /// Raises `ValueError` for an invalid size, radius, `gridtype` or `legfunc`.
    ///
    /// Fortran: the `*i` initialisation routines (`shaeci`, `shaesi`, `shagci`, `shagsi`, `shseci`, ..., `vhaeci`, ...), which fill `wsave` for a given `nlat`, `nlon`. Here nothing is precomputed at construction: tables are built lazily on first use.
    ///
    /// Difference from Fortran: `legfunc` mirrors the SPHEREPACK choice between recomputing ("c" variants, e.g. `shaec`) and storing ("s" variants, e.g. `shaes`) the associated Legendre functions; in this module the value is validated and exposed through the `legfunc` getter, but it does not select a different code path.
    #[new]
    #[pyo3(signature = (nlon, nlat, rsphere=6371200.0, gridtype="regular", legfunc="stored"))]
    fn new(
        nlon: usize,
        nlat: usize,
        rsphere: f64,
        gridtype: &str,
        legfunc: &str,
    ) -> PyResult<Self> {
        let gt = GridType::parse(gridtype).map_err(to_py_err)?;
        let engine = FullGridEngine::new(nlon, nlat, rsphere, gt, legfunc.to_string())
            .map_err(to_py_err)?;
        Ok(Self { engine })
    }

    /// Number of longitudes of the grid (int).
    #[getter]
    fn nlon(&self) -> usize {
        self.engine.nlon
    }

    /// Number of latitudes of the grid (int). The maximum spectral truncation is `nlat - 1`.
    #[getter]
    fn nlat(&self) -> usize {
        self.engine.nlat
    }

    /// Sphere radius used by the engine (float).
    #[getter]
    fn rsphere(&self) -> f64 {
        self.engine.rsphere
    }

    /// Grid type as a string: `"regular"` (equally spaced latitudes) or `"gaussian"`.
    #[getter]
    fn gridtype(&self) -> &'static str {
        match self.engine.gridtype {
            GridType::Regular => "regular",
            GridType::Gaussian => "gaussian",
        }
    }

    /// Legendre function mode string given at construction (`"stored"` or `"computed"`).
    #[getter]
    fn legfunc(&self) -> &str {
        &self.engine.legfunc
    }

    /// Spectral analysis of scalar grids: grid values to packed spherical-harmonic coefficients.
    ///
    /// Parameters: `datagrid` (float32 or float64 array of shape `(nt, nlat, nlon)`), latitude rows north to south unless `south_to_north=True`; `ntrunc` (int or None, default None meaning `nlat - 1`) triangular truncation, `0 <= ntrunc <= nlat - 1`; `south_to_north` (bool, default False) set True when the latitude rows run from south to north.
    ///
    /// Returns a complex array of shape `(nt, nmdim)` with `nmdim = (ntrunc + 1) * (ntrunc + 2) / 2` (complex64 for float32 input, complex128 for float64), packed as `m` outer, `n = m..=ntrunc` inner.
    ///
    /// Raises `ValueError` for a wrong grid shape or an out-of-range `ntrunc`, `TypeError` if `datagrid` is not a 3-D float32/float64 array.
    ///
    /// Algorithm: longitude FFT of each latitude row, then quadrature in latitude (weights appropriate to the grid type) against the normalised associated Legendre functions, with the north/south symmetric (parity-split) halves handled separately.
    ///
    /// Fortran: shaec.f / shaec, shaes.f / shaes, shagc.f / shagc, shags.f / shags (selected by `gridtype` and `legfunc`), followed by `twodtooned.f` / `twodtooned` to pack the `(a, b)` coefficient arrays into complex `dataspec` (`scale = 0.5`).
    ///
    /// Difference from Fortran: Fortran returns real coefficient arrays `a(nlat,nlat,nt)` and `b(nlat,nlat,nt)` indexed by `(m, n)`; here the result is already the packed complex array.
    #[pyo3(signature = (datagrid, ntrunc=None, south_to_north=false))]
    fn grdtospec<'py>(
        &self,
        py: Python<'py>,
        datagrid: &Bound<'py, PyAny>,
        ntrunc: Option<isize>,
        south_to_north: bool,
    ) -> PyResult<Py<PyAny>> {
        let ntrunc = resolve_ntrunc(ntrunc, self.engine.nlat - 1)?;
        if let Ok(a32) = datagrid.extract::<PyReadonlyArray3<'py, f32>>() {
            return self.run_grdtospec(py, a32, ntrunc, south_to_north);
        }
        if let Ok(a64) = datagrid.extract::<PyReadonlyArray3<'py, f64>>() {
            return self.run_grdtospec(py, a64, ntrunc, south_to_north);
        }
        Err(PyTypeError::new_err(
            "grdtospec expects a 3D (nt, nlat, nlon) float32 or float64 numpy array",
        ))
    }

    /// Spectral synthesis of scalar fields: packed spherical-harmonic coefficients to grid values.
    ///
    /// Parameters: `dataspec` (complex64 or complex128 array of shape `(nt, nmdim)`, with `ntrunc` inferred from `nmdim` and required to be `<= nlat - 1`); `south_to_north` (bool, default False) request latitude rows ordered south to north.
    ///
    /// Returns a real array of shape `(nt, nlat, nlon)` (float32 for complex64 input, float64 for complex128).
    ///
    /// Raises `ValueError` if `dataspec` is not 2-D, `nmdim` is not triangular, or `ntrunc > nlat - 1`; `TypeError` for a dtype other than complex64/complex128.
    ///
    /// Fortran: shsec.f / shsec, shses.f / shses, shsgc.f / shsgc, shsgs.f / shsgs, after `onedtotwod.f` / `onedtotwod` unpacking.
    ///
    /// Difference from Fortran: the coefficients are consumed in packed complex form (Fortran unpacks to `a`, `b` with `scale = 0.5`), and the inverse longitude FFT uses `realfft` rather than `hrfft.f`.
    #[pyo3(signature = (dataspec, south_to_north=false))]
    fn spectogrd<'py>(
        &self,
        py: Python<'py>,
        dataspec: &Bound<'py, PyAny>,
        south_to_north: bool,
    ) -> PyResult<Py<PyAny>> {
        if let Ok(s32) = dataspec.extract::<PyReadonlyArray2<'py, Complex32>>() {
            let (nt, _nmdim, ntrunc) =
                validate_spec2_shape(s32.shape(), self.engine.nlat - 1, "spectogrd")?;
            let mut own = Vec::new();
            let vc = view2_c32(&s32, &mut own);
            let mut out = vec![0.0_f32; nt * self.engine.nlat * self.engine.nlon];
            maybe_detach(
                py,
                nt * self.engine.nlat * self.engine.nlon >= engine::PAR_WORK_THRESHOLD,
                || self.engine.spectogrd(&vc, ntrunc, &mut out, south_to_north),
            );
            return Ok(to_py_grid3(py, nt, self.engine.nlat, self.engine.nlon, out));
        }
        if let Ok(s64) = dataspec.extract::<PyReadonlyArray2<'py, Complex64>>() {
            let (nt, _nmdim, ntrunc) =
                validate_spec2_shape(s64.shape(), self.engine.nlat - 1, "spectogrd")?;
            let mut own = Vec::new();
            let vc = view2_c64(&s64, &mut own);
            let mut out = vec![0.0_f64; nt * self.engine.nlat * self.engine.nlon];
            maybe_detach(
                py,
                nt * self.engine.nlat * self.engine.nlon >= engine::PAR_WORK_THRESHOLD,
                || self.engine.spectogrd(&vc, ntrunc, &mut out, south_to_north),
            );
            return Ok(to_py_grid3(py, nt, self.engine.nlat, self.engine.nlon, out));
        }
        Err(PyTypeError::new_err(
            "spectogrd expects a 2D (nt, nmdim) complex64 or complex128 numpy array",
        ))
    }

    /// Spectral vorticity and divergence of a horizontal wind field given on the grid.
    ///
    /// Parameters: `ugrid`, `vgrid` (float32 or float64 arrays, both `(nt, nlat, nlon)`; zonal and meridional wind components); `ntrunc` (int or None, default `nlat - 1`); `south_to_north` (bool, default False).
    ///
    /// Returns `(vrtspec, divspec)`, two complex arrays of shape `(nt, nmdim)` in the packed triangular order. Mixed float32/float64 inputs are promoted to float64.
    ///
    /// Raises `ValueError` for shape mismatch or an invalid `ntrunc`, `TypeError` for a non-float or non-3-D input.
    ///
    /// Algorithm: vector spherical-harmonic analysis of `(u, v)` gives coefficients `(br, bi)` (divergence part) and `(cr, ci)` (vorticity part); these are converted with the factor `sqrt(n * (n + 1)) / rsphere`: `div = -sqrt(n*(n+1))/rsphere * scale * (br + i bi)` and `vrt = +sqrt(n*(n+1))/rsphere * scale * (cr + i ci)`, `scale = 0.5` (0-based `n`).
    ///
    /// Fortran: vhaec.f / vhaec (or vhaes, vhagc, vhags) followed by `twodtooned_vrtdiv.f` / `twodtooned_vrtdiv`.
    ///
    /// Difference from Fortran: the vector analysis and the `rsphere` scaling are one step, the output is the packed complex spectra rather than `br, bi, cr, ci` arrays.
    #[pyo3(signature = (ugrid, vgrid, ntrunc=None, south_to_north=false))]
    fn getvrtdivspec<'py>(
        &self,
        py: Python<'py>,
        ugrid: &Bound<'py, PyAny>,
        vgrid: &Bound<'py, PyAny>,
        ntrunc: Option<isize>,
        south_to_north: bool,
    ) -> PyResult<Py<PyAny>> {
        self.dispatch_vec_analysis(
            py,
            ugrid,
            vgrid,
            ntrunc,
            south_to_north,
            VectorMode::VrtDiv,
            "getvrtdivspec",
        )
    }

    /// Spectral vorticity only. Same arguments, shapes, dtype rules and errors as `getvrtdivspec`, but returns a single complex array `vrtspec` of shape `(nt, nmdim)`: `vrt = +sqrt(n*(n+1))/rsphere * scale * (cr + i ci)`, `scale = 0.5`.
    ///
    /// Fortran: vhaec.f / vhaec (or vhaes, vhagc, vhags) plus the vorticity half of `twodtooned_vrtdiv.f` / `twodtooned_vrtdiv` (the f2py layer exposes this as `twodtooned_vrt`).
    #[pyo3(signature = (ugrid, vgrid, ntrunc=None, south_to_north=false))]
    fn getvrtspec<'py>(
        &self,
        py: Python<'py>,
        ugrid: &Bound<'py, PyAny>,
        vgrid: &Bound<'py, PyAny>,
        ntrunc: Option<isize>,
        south_to_north: bool,
    ) -> PyResult<Py<PyAny>> {
        self.dispatch_vec_analysis(
            py,
            ugrid,
            vgrid,
            ntrunc,
            south_to_north,
            VectorMode::VrtOnly,
            "getvrtspec",
        )
    }

    /// Spectral divergence only. Same arguments, shapes, dtype rules and errors as `getvrtdivspec`, but returns a single complex array `divspec` of shape `(nt, nmdim)`: `div = -sqrt(n*(n+1))/rsphere * scale * (br + i bi)`, `scale = 0.5`.
    ///
    /// Fortran: vhaec.f / vhaec (or vhaes, vhagc, vhags) plus the divergence half of `twodtooned_vrtdiv.f` / `twodtooned_vrtdiv` (the f2py layer exposes this as `twodtooned_div`).
    #[pyo3(signature = (ugrid, vgrid, ntrunc=None, south_to_north=false))]
    fn getdivspec<'py>(
        &self,
        py: Python<'py>,
        ugrid: &Bound<'py, PyAny>,
        vgrid: &Bound<'py, PyAny>,
        ntrunc: Option<isize>,
        south_to_north: bool,
    ) -> PyResult<Py<PyAny>> {
        self.dispatch_vec_analysis(
            py,
            ugrid,
            vgrid,
            ntrunc,
            south_to_north,
            VectorMode::DivOnly,
            "getdivspec",
        )
    }

    /// Wind components on the grid from spectral vorticity and divergence.
    ///
    /// Parameters: `vrtspec`, `divspec` (complex64 or complex128 arrays of identical shape `(nt, nmdim)`, same dtype); `south_to_north` (bool, default False).
    ///
    /// Returns `(u, v)`, two real arrays of shape `(nt, nlat, nlon)` (zonal and meridional wind components).
    ///
    /// Raises `ValueError` if the shapes differ, `nmdim` is not triangular or `ntrunc > nlat - 1`; `TypeError` for other dtypes.
    ///
    /// Algorithm: the spectra are converted to vector harmonic coefficients with the radius scaling `B = -rsphere / sqrt(n*(n+1)) * divspec` and `C = +rsphere / sqrt(n*(n+1)) * vrtspec` (this is `onedtotwod_vrtdiv.f` with `scale = 0.5`), and then synthesised with the vector transform; the `n = 0` coefficient has no velocity contribution.
    ///
    /// Fortran: onedtotwod_vrtdiv.f / onedtotwod_vrtdiv followed by vhsec.f / vhsec (or vhses, vhsgc, vhsgs).
    #[pyo3(signature = (vrtspec, divspec, south_to_north=false))]
    fn getuv<'py>(
        &self,
        py: Python<'py>,
        vrtspec: &Bound<'py, PyAny>,
        divspec: &Bound<'py, PyAny>,
        south_to_north: bool,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let v32 = vrtspec.extract::<PyReadonlyArray2<'py, Complex32>>().ok();
        let d32 = divspec.extract::<PyReadonlyArray2<'py, Complex32>>().ok();
        let v64 = if v32.is_none() {
            vrtspec.extract::<PyReadonlyArray2<'py, Complex64>>().ok()
        } else {
            None
        };
        let d64 = if d32.is_none() {
            divspec.extract::<PyReadonlyArray2<'py, Complex64>>().ok()
        } else {
            None
        };

        if let (Some(v), Some(d)) = (&v32, &d32) {
            let (nt, _nmdim, ntrunc) =
                validate_spec2_shape(v.shape(), self.engine.nlat - 1, "getuv")?;
            if v.shape() != d.shape() {
                return Err(PyValueError::new_err(
                    "vrtspec and divspec must have the same shape",
                ));
            }
            let (mut ov, mut od) = (Vec::new(), Vec::new());
            let vv = view2_c32(v, &mut ov);
            let dv = view2_c32(d, &mut od);
            let mut out_u = vec![0.0_f32; nt * self.engine.nlat * self.engine.nlon];
            let mut out_v = vec![0.0_f32; nt * self.engine.nlat * self.engine.nlon];
            maybe_detach(
                py,
                nt * self.engine.nlat * self.engine.nlon >= engine::PAR_WORK_THRESHOLD,
                || {
                    self.engine.getuv(
                        Some(&vv),
                        Some(&dv),
                        nt,
                        ntrunc,
                        &mut out_u,
                        &mut out_v,
                        south_to_north,
                        false,
                    )
                },
            );
            return Ok((
                to_py_grid3(py, nt, self.engine.nlat, self.engine.nlon, out_u),
                to_py_grid3(py, nt, self.engine.nlat, self.engine.nlon, out_v),
            ));
        }

        if let (Some(v), Some(d)) = (&v64, &d64) {
            let (nt, _nmdim, ntrunc) =
                validate_spec2_shape(v.shape(), self.engine.nlat - 1, "getuv")?;
            if v.shape() != d.shape() {
                return Err(PyValueError::new_err(
                    "vrtspec and divspec must have the same shape",
                ));
            }
            let (mut ov, mut od) = (Vec::new(), Vec::new());
            let vv = view2_c64(v, &mut ov);
            let dv = view2_c64(d, &mut od);
            let mut out_u = vec![0.0_f64; nt * self.engine.nlat * self.engine.nlon];
            let mut out_v = vec![0.0_f64; nt * self.engine.nlat * self.engine.nlon];
            maybe_detach(
                py,
                nt * self.engine.nlat * self.engine.nlon >= engine::PAR_WORK_THRESHOLD,
                || {
                    self.engine.getuv(
                        Some(&vv),
                        Some(&dv),
                        nt,
                        ntrunc,
                        &mut out_u,
                        &mut out_v,
                        south_to_north,
                        false,
                    )
                },
            );
            return Ok((
                to_py_grid3(py, nt, self.engine.nlat, self.engine.nlon, out_u),
                to_py_grid3(py, nt, self.engine.nlat, self.engine.nlon, out_v),
            ));
        }

        Err(PyTypeError::new_err(
            "getuv expects matching 2D (nt, nmdim) complex64 or complex128 arrays",
        ))
    }

    /// Streamfunction and velocity potential of a wind field given on the grid.
    ///
    /// Parameters: `ugrid`, `vgrid` (float32 or float64 arrays, both `(nt, nlat, nlon)`); `ntrunc` (int or None, default `nlat - 1`); `south_to_north` (bool, default False).
    ///
    /// Returns `(psi, chi)`, two real arrays of shape `(nt, nlat, nlon)`: `psi = invlap(vrt)` and `chi = invlap(div)`, where `invlap` multiplies each coefficient by `-rsphere**2 / (n*(n+1))` and zeroes `n = 0` (so the global mean of `psi` and `chi` is zero). Mixed float32/float64 inputs are promoted to float64.
    ///
    /// Raises as `getvrtdivspec`.
    ///
    /// No direct Fortran counterpart; composed of the vorticity/divergence analysis (`vhaec` family with `twodtooned_vrtdiv.f`), `invlap.f` / `invlap`, and the scalar synthesis (`shsec` family).
    #[pyo3(signature = (ugrid, vgrid, ntrunc=None, south_to_north=false))]
    fn getpsichi<'py>(
        &self,
        py: Python<'py>,
        ugrid: &Bound<'py, PyAny>,
        vgrid: &Bound<'py, PyAny>,
        ntrunc: Option<isize>,
        south_to_north: bool,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let ntrunc = resolve_ntrunc(ntrunc, self.engine.nlat - 1)?;
        let u32 = ugrid.extract::<PyReadonlyArray3<'py, f32>>().ok();
        let v32 = vgrid.extract::<PyReadonlyArray3<'py, f32>>().ok();
        let u64 = if u32.is_none() {
            ugrid.extract::<PyReadonlyArray3<'py, f64>>().ok()
        } else {
            None
        };
        let v64 = if v32.is_none() {
            vgrid.extract::<PyReadonlyArray3<'py, f64>>().ok()
        } else {
            None
        };
        // Same dtype for u and v runs natively; mixed dtypes are handled by the two promoting arms below.
        match (u64, u32, v64, v32) {
            (Some(u), None, Some(v), None) => {
                self.run_getpsichi(py, u.as_array(), v.as_array(), ntrunc, south_to_north)
            }
            (None, Some(u), None, Some(v)) => {
                self.run_getpsichi(py, u.as_array(), v.as_array(), ntrunc, south_to_north)
            }
            // u is float64, v is float32: v is promoted to float64.
            (Some(u), None, None, Some(v)) => {
                let vo = v.as_array().mapv(|x| x as f64);
                self.run_getpsichi(py, u.as_array(), vo.view(), ntrunc, south_to_north)
            }
            // u is float32, v is float64: u is promoted to float64.
            (None, Some(u), Some(v), None) => {
                let uo = u.as_array().mapv(|x| x as f64);
                self.run_getpsichi(py, uo.view(), v.as_array(), ntrunc, south_to_north)
            }
            _ => Err(PyTypeError::new_err(
                "getpsichi expects 3D (nt, nlat, nlon) float32 or float64 arrays",
            )),
        }
    }

    /// Gradient of scalar fields given by spectral coefficients.
    ///
    /// Parameters: `dataspec` (complex64 or complex128 array `(nt, nmdim)`); `south_to_north` (bool, default False).
    ///
    /// Returns `(u, v)`, two real arrays of shape `(nt, nlat, nlon)`: the zonal and meridional components of the gradient of the field on a sphere of radius `rsphere`.
    ///
    /// Raises `ValueError` for a non-2-D array, non-triangular `nmdim` or `ntrunc > nlat - 1`; `TypeError` for other dtypes.
    ///
    /// Algorithm: the gradient is synthesised as the irrotational vector field whose divergence spectrum is `-n*(n+1)/rsphere**2 * phi`, so the divergence-type coefficients are `B = +sqrt(n*(n+1))/rsphere * phi` and the vorticity-type coefficients are `C = 0`; this uses the same vector synthesis as `getuv` (internal `is_grad` flag).
    ///
    /// Fortran: gradient obtained in SPHEREPACK with `vhsec`/`vhses`/`vhsgc`/`vhsgs` on `(B, C)` built as in `onedtotwod_vrtdiv.f` (no dedicated Fortran routine in the sources read here).
    #[pyo3(signature = (dataspec, south_to_north=false))]
    fn getgrad<'py>(
        &self,
        py: Python<'py>,
        dataspec: &Bound<'py, PyAny>,
        south_to_north: bool,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        if let Ok(s32) = dataspec.extract::<PyReadonlyArray2<'py, Complex32>>() {
            let (nt, _nmdim, ntrunc) =
                validate_spec2_shape(s32.shape(), self.engine.nlat - 1, "getgrad")?;
            let mut own = Vec::new();
            let vc = view2_c32(&s32, &mut own);
            let mut out_u = vec![0.0_f32; nt * self.engine.nlat * self.engine.nlon];
            let mut out_v = vec![0.0_f32; nt * self.engine.nlat * self.engine.nlon];
            maybe_detach(
                py,
                nt * self.engine.nlat * self.engine.nlon >= engine::PAR_WORK_THRESHOLD,
                || {
                    self.engine.getuv(
                        None,
                        Some(&vc),
                        nt,
                        ntrunc,
                        &mut out_u,
                        &mut out_v,
                        south_to_north,
                        true,
                    )
                },
            );
            return Ok((
                to_py_grid3(py, nt, self.engine.nlat, self.engine.nlon, out_u),
                to_py_grid3(py, nt, self.engine.nlat, self.engine.nlon, out_v),
            ));
        }
        if let Ok(s64) = dataspec.extract::<PyReadonlyArray2<'py, Complex64>>() {
            let (nt, _nmdim, ntrunc) =
                validate_spec2_shape(s64.shape(), self.engine.nlat - 1, "getgrad")?;
            let mut own = Vec::new();
            let vc = view2_c64(&s64, &mut own);
            let mut out_u = vec![0.0_f64; nt * self.engine.nlat * self.engine.nlon];
            let mut out_v = vec![0.0_f64; nt * self.engine.nlat * self.engine.nlon];
            maybe_detach(
                py,
                nt * self.engine.nlat * self.engine.nlon >= engine::PAR_WORK_THRESHOLD,
                || {
                    self.engine.getuv(
                        None,
                        Some(&vc),
                        nt,
                        ntrunc,
                        &mut out_u,
                        &mut out_v,
                        south_to_north,
                        true,
                    )
                },
            );
            return Ok((
                to_py_grid3(py, nt, self.engine.nlat, self.engine.nlon, out_u),
                to_py_grid3(py, nt, self.engine.nlat, self.engine.nlon, out_v),
            ));
        }
        Err(PyTypeError::new_err(
            "getgrad expects a 2D (nt, nmdim) complex64 or complex128 numpy array",
        ))
    }
}

/// Spherical-harmonic transform engine for a reduced Gaussian grid (a Gaussian latitude set where row `j` has `pl[j]` longitudes).
///
/// Created with `RustReducedSpharmCore(pl, rsphere=6371200.0, legfunc="stored")`. Grid fields are `(nt, npoints)` arrays with the rows packed one after another from north to south (`npoints = sum(pl)`), spectral fields are `(nt, nmdim)` complex arrays in the triangular packing described in the module docs. Methods: `grdtospec`, `spectogrd`, `spectogrd_pair`, `getvrtdivspec`, `getvrtspec`, `getdivspec`, `getuv`, `getgrad`, `getgrad_pair`; read-only attributes `nlat`, `npoints`, `rsphere`, `legfunc`.
///
/// Fortran: `reduced_gaussian_scalar.f90` (`reduced_gaussian_grdtospec`, `reduced_gaussian_spectogrd`, `..._spectogrd_pair`, `..._getgrad`, `..._getgrad_pair`, `..._getvrtdivspec`, `..._getvrtspec`, `..._getdivspec`) through the C bindings in `reduced_bindings.f90`. Fortran header: "Direct scalar transforms for packed reduced Gaussian grids. These helpers use the same public spectral coefficient convention as the existing Spharmt Gaussian backend, but replace the rectangular longitude FFT with row-wise Fourier sums over each reduced latitude circle."
///
/// Difference from Fortran: the Legendre basis and its derivative are built lazily per dtype instead of being passed in as `basis`/`dbasis` arrays, weights and `sin_theta` are not arguments, errors are exceptions instead of `ierror`, and the per-row longitude transforms use cached real FFT plans (`fft.rs`).
#[pyclass(frozen, module = "skybornspharm._spharm_rs")]
pub struct RustReducedSpharmCore {
    engine: ReducedEngine,
}

// Internal helpers (not exposed to Python): typed workers and dtype dispatch.
impl RustReducedSpharmCore {
    /// Typed worker for the reduced-grid `grdtospec`: validate the `(nt, npoints)` shape, allocate the `(nt, nmdim)` output and run the engine analysis, releasing the GIL for large problems; engine errors become `ValueError`.
    ///
    /// Fortran: `reduced_gaussian_grdtospec` in reduced_gaussian_scalar.f90.
    fn run_grdtospec<'py, T: TypedReducedBasis + numpy::Element>(
        &self,
        py: Python<'py>,
        grid: PyReadonlyArray2<'py, T>,
        ntrunc: usize,
        south_to_north: bool,
    ) -> PyResult<Py<PyAny>>
    where
        Complex<T>: numpy::Element,
    {
        let view = grid.as_array();
        let nt = validate_reduced_grid2(view.shape(), self.engine.npoints, "grdtospec")?;
        let nmdim = nmdim_from_ntrunc(ntrunc);
        let mut out = vec![T::complex_zero(); nt * nmdim];
        maybe_detach(
            py,
            nt * self.engine.npoints >= engine::PAR_WORK_THRESHOLD,
            || {
                self.engine
                    .grdtospec(&view, &mut out, ntrunc, south_to_north)
            },
        )
        .map_err(to_py_err)?;
        Ok(to_py_spec2(py, nt, nmdim, out))
    }

    /// Typed worker for the reduced-grid `getvrtdivspec`, `getvrtspec` and `getdivspec`: validate that `u` and `v` are `(nt, npoints)` and equal in shape, then compute vorticity and/or divergence spectra according to `mode`; engine errors become `ValueError`.
    ///
    /// Fortran: `reduced_gaussian_getvrtdivspec`, `reduced_gaussian_getvrtspec`, `reduced_gaussian_getdivspec` in reduced_gaussian_scalar.f90.
    fn run_vec_analysis<'py, T: TypedReducedBasis + numpy::Element>(
        &self,
        py: Python<'py>,
        u: ArrayView2<'_, T>,
        v: ArrayView2<'_, T>,
        ntrunc: usize,
        south_to_north: bool,
        mode: VectorMode,
        op: &str,
    ) -> PyResult<Py<PyAny>>
    where
        Complex<T>: numpy::Element,
    {
        let nt = validate_reduced_grid2(u.shape(), self.engine.npoints, op)?;
        if u.shape() != v.shape() {
            return Err(PyValueError::new_err(format!(
                "ugrid shape {:?} must match vgrid shape {:?}",
                u.shape(),
                v.shape()
            )));
        }
        let nmdim = nmdim_from_ntrunc(ntrunc);
        let detach = nt * self.engine.npoints >= engine::PAR_WORK_THRESHOLD;
        match mode {
            VectorMode::VrtDiv => {
                let mut vrt = vec![T::complex_zero(); nt * nmdim];
                let mut div = vec![T::complex_zero(); nt * nmdim];
                maybe_detach(py, detach, || {
                    self.engine.getvrtdivspec(
                        &u,
                        &v,
                        Some(&mut vrt),
                        Some(&mut div),
                        ntrunc,
                        south_to_north,
                        VectorMode::VrtDiv,
                    )
                })
                .map_err(to_py_err)?;
                Ok((to_py_spec2(py, nt, nmdim, vrt), to_py_spec2(py, nt, nmdim, div))
                    .into_pyobject(py)?
                    .into_any()
                    .unbind())
            }
            VectorMode::VrtOnly => {
                let mut vrt = vec![T::complex_zero(); nt * nmdim];
                maybe_detach(py, detach, || {
                    self.engine.getvrtdivspec(
                        &u,
                        &v,
                        Some(&mut vrt),
                        None,
                        ntrunc,
                        south_to_north,
                        VectorMode::VrtOnly,
                    )
                })
                .map_err(to_py_err)?;
                Ok(to_py_spec2(py, nt, nmdim, vrt))
            }
            VectorMode::DivOnly => {
                let mut div = vec![T::complex_zero(); nt * nmdim];
                maybe_detach(py, detach, || {
                    self.engine.getvrtdivspec(
                        &u,
                        &v,
                        None,
                        Some(&mut div),
                        ntrunc,
                        south_to_north,
                        VectorMode::DivOnly,
                    )
                })
                .map_err(to_py_err)?;
                Ok(to_py_spec2(py, nt, nmdim, div))
            }
        }
    }

    /// Dtype dispatcher for the reduced-grid vector analysis methods: resolve `ntrunc` (`None` means `nlat - 1`), extract `ugrid` and `vgrid` as float32 or float64 2-D arrays and call `run_vec_analysis`. Mixed dtypes promote the float32 array to float64; unsupported input raises `TypeError` using `op_name`.
    ///
    /// No direct Fortran counterpart.
    fn dispatch_vec_analysis<'py>(
        &self,
        py: Python<'py>,
        ugrid: &Bound<'py, PyAny>,
        vgrid: &Bound<'py, PyAny>,
        ntrunc: Option<isize>,
        south_to_north: bool,
        mode: VectorMode,
        op_name: &str,
    ) -> PyResult<Py<PyAny>> {
        let ntrunc = resolve_ntrunc(ntrunc, self.engine.nlat - 1)?;
        let u32 = ugrid.extract::<PyReadonlyArray2<'py, f32>>().ok();
        let v32 = vgrid.extract::<PyReadonlyArray2<'py, f32>>().ok();
        let u64 = if u32.is_none() {
            ugrid.extract::<PyReadonlyArray2<'py, f64>>().ok()
        } else {
            None
        };
        let v64 = if v32.is_none() {
            vgrid.extract::<PyReadonlyArray2<'py, f64>>().ok()
        } else {
            None
        };
        // Same dtype for u and v runs natively; mixed dtypes are handled by the two promoting arms below.
        match (u64, u32, v64, v32) {
            (Some(u), None, Some(v), None) => self.run_vec_analysis(
                py,
                u.as_array(),
                v.as_array(),
                ntrunc,
                south_to_north,
                mode,
                op_name,
            ),
            (None, Some(u), None, Some(v)) => self.run_vec_analysis(
                py,
                u.as_array(),
                v.as_array(),
                ntrunc,
                south_to_north,
                mode,
                op_name,
            ),
            // u is float64, v is float32: v is promoted to float64.
            (Some(u), None, None, Some(v)) => {
                let vo = v.as_array().mapv(|x| x as f64);
                self.run_vec_analysis(
                    py,
                    u.as_array(),
                    vo.view(),
                    ntrunc,
                    south_to_north,
                    mode,
                    op_name,
                )
            }
            // u is float32, v is float64: u is promoted to float64.
            (None, Some(u), Some(v), None) => {
                let uo = u.as_array().mapv(|x| x as f64);
                self.run_vec_analysis(
                    py,
                    uo.view(),
                    v.as_array(),
                    ntrunc,
                    south_to_north,
                    mode,
                    op_name,
                )
            }
            _ => Err(PyTypeError::new_err(format!(
                "{op_name} expects 2D (nt, npoints) float32 or float64 arrays"
            ))),
        }
    }
}

#[pymethods]
// Python-visible methods.
impl RustReducedSpharmCore {
    /// Create a reduced Gaussian grid engine.
    ///
    /// Parameters: `pl` (sequence or numpy int32/int64 array of at least 3 positive integers) number of longitudes on each latitude row, ordered north to south, every entry `>= 4`; `rsphere` (float, > 0, default 6371200.0) sphere radius in metres; `legfunc` (`"stored"` or `"computed"`).
    ///
    /// The number of latitudes is `nlat = len(pl)` and the number of grid points is `npoints = sum(pl)`.
    ///
    /// Raises `ValueError` if any row has fewer than 4 points, `pl` has fewer than 3 rows, `rsphere` is invalid or `legfunc` is not recognised.
    ///
    /// Fortran: the `pl` argument of the `reduced_gaussian_*` routines (`ngptot = sum(pl)`). As for the full grid, `legfunc` is validated and stored but does not select a different code path here.
    #[new]
    #[pyo3(signature = (pl, rsphere=6371200.0, legfunc="stored"))]
    fn new(pl: &Bound<'_, PyAny>, rsphere: f64, legfunc: &str) -> PyResult<Self> {
        let pl_vec: Vec<usize> = if let Ok(arr) = pl.extract::<PyReadonlyArray1<'_, i32>>() {
            let v = arr.as_array();
            if v.iter().any(|&x| x < 4) {
                return Err(PyValueError::new_err(
                    "each reduced Gaussian row must contain >= 4 points",
                ));
            }
            v.iter().map(|&x| x as usize).collect()
        } else if let Ok(arr) = pl.extract::<PyReadonlyArray1<'_, i64>>() {
            let v = arr.as_array();
            if v.iter().any(|&x| x < 4) {
                return Err(PyValueError::new_err(
                    "each reduced Gaussian row must contain >= 4 points",
                ));
            }
            v.iter().map(|&x| x as usize).collect()
        } else {
            let v: Vec<isize> = pl.extract()?;
            if v.iter().any(|&x| x < 4) {
                return Err(PyValueError::new_err(
                    "each reduced Gaussian row must contain >= 4 points",
                ));
            }
            v.into_iter().map(|x| x as usize).collect()
        };
        let engine = ReducedEngine::new(pl_vec, rsphere, legfunc.to_string()).map_err(to_py_err)?;
        Ok(Self { engine })
    }

    /// Number of latitude rows, `len(pl)` (int). The maximum spectral truncation is `nlat - 1`.
    #[getter]
    fn nlat(&self) -> usize {
        self.engine.nlat
    }

    /// Total number of grid points, `sum(pl)` (int); the second dimension of grid arrays.
    #[getter]
    fn npoints(&self) -> usize {
        self.engine.npoints
    }

    /// Sphere radius used by the engine (float).
    #[getter]
    fn rsphere(&self) -> f64 {
        self.engine.rsphere
    }

    /// Legendre function mode string given at construction (`"stored"` or `"computed"`).
    #[getter]
    fn legfunc(&self) -> &str {
        &self.engine.legfunc
    }

    /// Spectral analysis of scalar fields on the reduced grid.
    ///
    /// Parameters: `datagrid` (float32 or float64 array of shape `(nt, npoints)`), rows packed north to south unless `south_to_north=True`; `ntrunc` (int or None, default `nlat - 1`), `0 <= ntrunc <= nlat - 1`; `south_to_north` (bool, default False).
    ///
    /// Returns a complex array `(nt, nmdim)` in the packed triangular order (complex64 for float32 input, complex128 for float64).
    ///
    /// Raises `ValueError` for a wrong grid shape or invalid `ntrunc`, `TypeError` if `datagrid` is not a 2-D float32/float64 array.
    ///
    /// Fortran: reduced_gaussian_scalar.f90 / reduced_gaussian_grdtospec (Gaussian quadrature weights and row-wise Fourier sums over each reduced latitude circle).
    #[pyo3(signature = (datagrid, ntrunc=None, south_to_north=false))]
    fn grdtospec<'py>(
        &self,
        py: Python<'py>,
        datagrid: &Bound<'py, PyAny>,
        ntrunc: Option<isize>,
        south_to_north: bool,
    ) -> PyResult<Py<PyAny>> {
        let ntrunc = resolve_ntrunc(ntrunc, self.engine.nlat - 1)?;
        if let Ok(a32) = datagrid.extract::<PyReadonlyArray2<'py, f32>>() {
            return self.run_grdtospec(py, a32, ntrunc, south_to_north);
        }
        if let Ok(a64) = datagrid.extract::<PyReadonlyArray2<'py, f64>>() {
            return self.run_grdtospec(py, a64, ntrunc, south_to_north);
        }
        Err(PyTypeError::new_err(
            "grdtospec expects a 2D (nt, npoints) float32 or float64 numpy array",
        ))
    }

    /// Spectral synthesis of scalar fields onto the reduced grid.
    ///
    /// Parameters: `dataspec` (complex64 or complex128 array `(nt, nmdim)`, `ntrunc` inferred and `<= nlat - 1`); `south_to_north` (bool, default False).
    ///
    /// Returns a real array `(nt, npoints)` (float32 for complex64 input, float64 for complex128).
    ///
    /// Raises `ValueError` for a non-2-D array, non-triangular `nmdim` or `ntrunc > nlat - 1`; `TypeError` for other dtypes.
    ///
    /// Fortran: reduced_gaussian_scalar.f90 / reduced_gaussian_spectogrd
    #[pyo3(signature = (dataspec, south_to_north=false))]
    fn spectogrd<'py>(
        &self,
        py: Python<'py>,
        dataspec: &Bound<'py, PyAny>,
        south_to_north: bool,
    ) -> PyResult<Py<PyAny>> {
        if let Ok(s32) = dataspec.extract::<PyReadonlyArray2<'py, Complex32>>() {
            let (nt, _nmdim, ntrunc) =
                validate_spec2_shape(s32.shape(), self.engine.nlat - 1, "spectogrd")?;
            let mut own = Vec::new();
            let vc = view2_c32(&s32, &mut own);
            let mut out = vec![0.0_f32; nt * self.engine.npoints];
            maybe_detach(
                py,
                nt * self.engine.npoints >= engine::PAR_WORK_THRESHOLD,
                || self.engine.spectogrd(&vc, &mut out, ntrunc, south_to_north),
            )
            .map_err(to_py_err)?;
            return Ok(to_py_grid2(py, nt, self.engine.npoints, out));
        }
        if let Ok(s64) = dataspec.extract::<PyReadonlyArray2<'py, Complex64>>() {
            let (nt, _nmdim, ntrunc) =
                validate_spec2_shape(s64.shape(), self.engine.nlat - 1, "spectogrd")?;
            let mut own = Vec::new();
            let vc = view2_c64(&s64, &mut own);
            let mut out = vec![0.0_f64; nt * self.engine.npoints];
            maybe_detach(
                py,
                nt * self.engine.npoints >= engine::PAR_WORK_THRESHOLD,
                || self.engine.spectogrd(&vc, &mut out, ntrunc, south_to_north),
            )
            .map_err(to_py_err)?;
            return Ok(to_py_grid2(py, nt, self.engine.npoints, out));
        }
        Err(PyTypeError::new_err(
            "spectogrd expects a 2D (nt, nmdim) complex64 or complex128 numpy array",
        ))
    }

    /// Synthesise two spectral fields at once onto the reduced grid (shares the Legendre sums and FFT work between the pair).
    ///
    /// Parameters: `spec_a`, `spec_b` (complex arrays of identical shape `(nt, nmdim)` and the same dtype); `south_to_north` (bool, default False).
    ///
    /// Returns `(grid_a, grid_b)`, two real arrays of shape `(nt, npoints)`.
    ///
    /// Raises `ValueError` if the shapes differ or the truncation is invalid; `TypeError` if both are not complex64 (or both complex128) 2-D arrays.
    ///
    /// Fortran: reduced_gaussian_scalar.f90 / reduced_gaussian_spectogrd_pair
    #[pyo3(signature = (spec_a, spec_b, south_to_north=false))]
    fn spectogrd_pair<'py>(
        &self,
        py: Python<'py>,
        spec_a: &Bound<'py, PyAny>,
        spec_b: &Bound<'py, PyAny>,
        south_to_north: bool,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        if let (Ok(a32), Ok(b32)) = (
            spec_a.extract::<PyReadonlyArray2<'py, Complex32>>(),
            spec_b.extract::<PyReadonlyArray2<'py, Complex32>>(),
        ) {
            let (nt, _nmdim, ntrunc) =
                validate_spec2_shape(a32.shape(), self.engine.nlat - 1, "spectogrd_pair")?;
            if a32.shape() != b32.shape() {
                return Err(PyValueError::new_err(
                    "spec_a and spec_b must have the same shape",
                ));
            }
            let (mut oa, mut ob) = (Vec::new(), Vec::new());
            let va = view2_c32(&a32, &mut oa);
            let vb = view2_c32(&b32, &mut ob);
            let mut out_a = vec![0.0_f32; nt * self.engine.npoints];
            let mut out_b = vec![0.0_f32; nt * self.engine.npoints];
            maybe_detach(
                py,
                nt * self.engine.npoints >= engine::PAR_WORK_THRESHOLD,
                || {
                    self.engine.spectogrd_pair(
                        &va,
                        &vb,
                        &mut out_a,
                        &mut out_b,
                        ntrunc,
                        south_to_north,
                    )
                },
            )
            .map_err(to_py_err)?;
            return Ok((
                to_py_grid2(py, nt, self.engine.npoints, out_a),
                to_py_grid2(py, nt, self.engine.npoints, out_b),
            ));
        }
        if let (Ok(a64), Ok(b64)) = (
            spec_a.extract::<PyReadonlyArray2<'py, Complex64>>(),
            spec_b.extract::<PyReadonlyArray2<'py, Complex64>>(),
        ) {
            let (nt, _nmdim, ntrunc) =
                validate_spec2_shape(a64.shape(), self.engine.nlat - 1, "spectogrd_pair")?;
            if a64.shape() != b64.shape() {
                return Err(PyValueError::new_err(
                    "spec_a and spec_b must have the same shape",
                ));
            }
            let (mut oa, mut ob) = (Vec::new(), Vec::new());
            let va = view2_c64(&a64, &mut oa);
            let vb = view2_c64(&b64, &mut ob);
            let mut out_a = vec![0.0_f64; nt * self.engine.npoints];
            let mut out_b = vec![0.0_f64; nt * self.engine.npoints];
            maybe_detach(
                py,
                nt * self.engine.npoints >= engine::PAR_WORK_THRESHOLD,
                || {
                    self.engine.spectogrd_pair(
                        &va,
                        &vb,
                        &mut out_a,
                        &mut out_b,
                        ntrunc,
                        south_to_north,
                    )
                },
            )
            .map_err(to_py_err)?;
            return Ok((
                to_py_grid2(py, nt, self.engine.npoints, out_a),
                to_py_grid2(py, nt, self.engine.npoints, out_b),
            ));
        }
        Err(PyTypeError::new_err(
            "spectogrd_pair expects matching 2D (nt, nmdim) complex64 or complex128 arrays",
        ))
    }

    /// Spectral vorticity and divergence of a wind field on the reduced grid.
    ///
    /// Parameters: `ugrid`, `vgrid` (float32 or float64 arrays, both `(nt, npoints)`); `ntrunc` (int or None, default `nlat - 1`); `south_to_north` (bool, default False).
    ///
    /// Returns `(vrtspec, divspec)`, complex arrays of shape `(nt, nmdim)`, scaled by `sqrt(n*(n+1))/rsphere` as for the full grid (vorticity with `+`, divergence with `-` sign convention of `twodtooned_vrtdiv.f`). Mixed float32/float64 inputs are promoted to float64.
    ///
    /// Raises `ValueError` for shape mismatch or invalid `ntrunc`, `TypeError` for a non-float or non-2-D input.
    ///
    /// Fortran: reduced_gaussian_scalar.f90 / reduced_gaussian_getvrtdivspec (the full-grid equivalent chain is `vhaec` plus `twodtooned_vrtdiv.f`).
    #[pyo3(signature = (ugrid, vgrid, ntrunc=None, south_to_north=false))]
    fn getvrtdivspec<'py>(
        &self,
        py: Python<'py>,
        ugrid: &Bound<'py, PyAny>,
        vgrid: &Bound<'py, PyAny>,
        ntrunc: Option<isize>,
        south_to_north: bool,
    ) -> PyResult<Py<PyAny>> {
        self.dispatch_vec_analysis(
            py,
            ugrid,
            vgrid,
            ntrunc,
            south_to_north,
            VectorMode::VrtDiv,
            "getvrtdivspec",
        )
    }

    /// Spectral vorticity only on the reduced grid. Same arguments, shapes, dtype rules and errors as the reduced-grid `getvrtdivspec`; returns a single complex array `(nt, nmdim)`.
    ///
    /// Fortran: reduced_gaussian_scalar.f90 / reduced_gaussian_getvrtspec
    #[pyo3(signature = (ugrid, vgrid, ntrunc=None, south_to_north=false))]
    fn getvrtspec<'py>(
        &self,
        py: Python<'py>,
        ugrid: &Bound<'py, PyAny>,
        vgrid: &Bound<'py, PyAny>,
        ntrunc: Option<isize>,
        south_to_north: bool,
    ) -> PyResult<Py<PyAny>> {
        self.dispatch_vec_analysis(
            py,
            ugrid,
            vgrid,
            ntrunc,
            south_to_north,
            VectorMode::VrtOnly,
            "getvrtspec",
        )
    }

    /// Spectral divergence only on the reduced grid. Same arguments, shapes, dtype rules and errors as the reduced-grid `getvrtdivspec`; returns a single complex array `(nt, nmdim)`.
    ///
    /// Fortran: reduced_gaussian_scalar.f90 / reduced_gaussian_getdivspec
    #[pyo3(signature = (ugrid, vgrid, ntrunc=None, south_to_north=false))]
    fn getdivspec<'py>(
        &self,
        py: Python<'py>,
        ugrid: &Bound<'py, PyAny>,
        vgrid: &Bound<'py, PyAny>,
        ntrunc: Option<isize>,
        south_to_north: bool,
    ) -> PyResult<Py<PyAny>> {
        self.dispatch_vec_analysis(
            py,
            ugrid,
            vgrid,
            ntrunc,
            south_to_north,
            VectorMode::DivOnly,
            "getdivspec",
        )
    }

    /// Wind components on the reduced grid from spectral vorticity and divergence.
    ///
    /// Parameters: `vrtspec`, `divspec` (complex arrays of identical shape `(nt, nmdim)`, same dtype); `south_to_north` (bool, default False).
    ///
    /// Returns `(u, v)`, real arrays of shape `(nt, npoints)`.
    ///
    /// Raises `ValueError` if the shapes differ, `nmdim` is not triangular or `ntrunc > nlat - 1`; `TypeError` for other dtypes.
    ///
    /// Algorithm (from the engine): `psispec = invlap(vrtspec)` and `chispec = invlap(divspec)` (factor `-rsphere**2 / (n*(n+1))`, zero for `n = 0`), then the gradients of `chi` and `psi` are synthesised and combined as `u = u_chi - u_psi`, `v = v_chi + v_psi`.
    ///
    /// Fortran: no dedicated reduced-grid routine; the full-grid equivalent is `onedtotwod_vrtdiv.f` plus `vhsec`.
    #[pyo3(signature = (vrtspec, divspec, south_to_north=false))]
    fn getuv<'py>(
        &self,
        py: Python<'py>,
        vrtspec: &Bound<'py, PyAny>,
        divspec: &Bound<'py, PyAny>,
        south_to_north: bool,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        if let (Ok(v32), Ok(d32)) = (
            vrtspec.extract::<PyReadonlyArray2<'py, Complex32>>(),
            divspec.extract::<PyReadonlyArray2<'py, Complex32>>(),
        ) {
            let (nt, _nmdim, ntrunc) =
                validate_spec2_shape(v32.shape(), self.engine.nlat - 1, "getuv")?;
            if v32.shape() != d32.shape() {
                return Err(PyValueError::new_err(
                    "vrtspec and divspec must have the same shape",
                ));
            }
            let (mut ov, mut od) = (Vec::new(), Vec::new());
            let vv = view2_c32(&v32, &mut ov);
            let dv = view2_c32(&d32, &mut od);
            let mut out_u = vec![0.0_f32; nt * self.engine.npoints];
            let mut out_v = vec![0.0_f32; nt * self.engine.npoints];
            maybe_detach(
                py,
                nt * self.engine.npoints >= engine::PAR_WORK_THRESHOLD,
                || {
                    self.engine
                        .getuv(&vv, &dv, &mut out_u, &mut out_v, ntrunc, south_to_north)
                },
            )
            .map_err(to_py_err)?;
            return Ok((
                to_py_grid2(py, nt, self.engine.npoints, out_u),
                to_py_grid2(py, nt, self.engine.npoints, out_v),
            ));
        }
        if let (Ok(v64), Ok(d64)) = (
            vrtspec.extract::<PyReadonlyArray2<'py, Complex64>>(),
            divspec.extract::<PyReadonlyArray2<'py, Complex64>>(),
        ) {
            let (nt, _nmdim, ntrunc) =
                validate_spec2_shape(v64.shape(), self.engine.nlat - 1, "getuv")?;
            if v64.shape() != d64.shape() {
                return Err(PyValueError::new_err(
                    "vrtspec and divspec must have the same shape",
                ));
            }
            let (mut ov, mut od) = (Vec::new(), Vec::new());
            let vv = view2_c64(&v64, &mut ov);
            let dv = view2_c64(&d64, &mut od);
            let mut out_u = vec![0.0_f64; nt * self.engine.npoints];
            let mut out_v = vec![0.0_f64; nt * self.engine.npoints];
            maybe_detach(
                py,
                nt * self.engine.npoints >= engine::PAR_WORK_THRESHOLD,
                || {
                    self.engine
                        .getuv(&vv, &dv, &mut out_u, &mut out_v, ntrunc, south_to_north)
                },
            )
            .map_err(to_py_err)?;
            return Ok((
                to_py_grid2(py, nt, self.engine.npoints, out_u),
                to_py_grid2(py, nt, self.engine.npoints, out_v),
            ));
        }
        Err(PyTypeError::new_err(
            "getuv expects matching 2D (nt, nmdim) complex64 or complex128 arrays",
        ))
    }

    /// Gradient of scalar fields given by spectral coefficients, on the reduced grid.
    ///
    /// Parameters: `dataspec` (complex64 or complex128 array `(nt, nmdim)`); `south_to_north` (bool, default False).
    ///
    /// Returns `(u, v)`, real arrays of shape `(nt, npoints)` with the zonal and meridional gradient components on a sphere of radius `rsphere`.
    ///
    /// Raises `ValueError` for a non-2-D array, non-triangular `nmdim` or `ntrunc > nlat - 1`; `TypeError` for other dtypes.
    ///
    /// Fortran: reduced_gaussian_scalar.f90 / reduced_gaussian_getgrad
    #[pyo3(signature = (dataspec, south_to_north=false))]
    fn getgrad<'py>(
        &self,
        py: Python<'py>,
        dataspec: &Bound<'py, PyAny>,
        south_to_north: bool,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        if let Ok(s32) = dataspec.extract::<PyReadonlyArray2<'py, Complex32>>() {
            let (nt, _nmdim, ntrunc) =
                validate_spec2_shape(s32.shape(), self.engine.nlat - 1, "getgrad")?;
            let mut own = Vec::new();
            let vc = view2_c32(&s32, &mut own);
            let mut out_u = vec![0.0_f32; nt * self.engine.npoints];
            let mut out_v = vec![0.0_f32; nt * self.engine.npoints];
            maybe_detach(
                py,
                nt * self.engine.npoints >= engine::PAR_WORK_THRESHOLD,
                || {
                    self.engine
                        .getgrad(&vc, &mut out_u, &mut out_v, ntrunc, south_to_north)
                },
            )
            .map_err(to_py_err)?;
            return Ok((
                to_py_grid2(py, nt, self.engine.npoints, out_u),
                to_py_grid2(py, nt, self.engine.npoints, out_v),
            ));
        }
        if let Ok(s64) = dataspec.extract::<PyReadonlyArray2<'py, Complex64>>() {
            let (nt, _nmdim, ntrunc) =
                validate_spec2_shape(s64.shape(), self.engine.nlat - 1, "getgrad")?;
            let mut own = Vec::new();
            let vc = view2_c64(&s64, &mut own);
            let mut out_u = vec![0.0_f64; nt * self.engine.npoints];
            let mut out_v = vec![0.0_f64; nt * self.engine.npoints];
            maybe_detach(
                py,
                nt * self.engine.npoints >= engine::PAR_WORK_THRESHOLD,
                || {
                    self.engine
                        .getgrad(&vc, &mut out_u, &mut out_v, ntrunc, south_to_north)
                },
            )
            .map_err(to_py_err)?;
            return Ok((
                to_py_grid2(py, nt, self.engine.npoints, out_u),
                to_py_grid2(py, nt, self.engine.npoints, out_v),
            ));
        }
        Err(PyTypeError::new_err(
            "getgrad expects a 2D (nt, nmdim) complex64 or complex128 numpy array",
        ))
    }

    /// Gradients of two spectral fields at once on the reduced grid.
    ///
    /// Parameters: `spec_a`, `spec_b` (complex arrays of identical shape `(nt, nmdim)`, same dtype); `south_to_north` (bool, default False).
    ///
    /// Returns `(a_u, a_v, b_u, b_v)`, four real arrays of shape `(nt, npoints)`.
    ///
    /// Raises `ValueError` if the shapes differ or the truncation is invalid; `TypeError` if the inputs are not matching 2-D complex64 or complex128 arrays.
    ///
    /// Fortran: reduced_gaussian_scalar.f90 / reduced_gaussian_getgrad_pair
    #[pyo3(signature = (spec_a, spec_b, south_to_north=false))]
    fn getgrad_pair<'py>(
        &self,
        py: Python<'py>,
        spec_a: &Bound<'py, PyAny>,
        spec_b: &Bound<'py, PyAny>,
        south_to_north: bool,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
        if let (Ok(a32), Ok(b32)) = (
            spec_a.extract::<PyReadonlyArray2<'py, Complex32>>(),
            spec_b.extract::<PyReadonlyArray2<'py, Complex32>>(),
        ) {
            let (nt, _nmdim, ntrunc) =
                validate_spec2_shape(a32.shape(), self.engine.nlat - 1, "getgrad_pair")?;
            if a32.shape() != b32.shape() {
                return Err(PyValueError::new_err(
                    "spec_a and spec_b must have the same shape",
                ));
            }
            let (mut oa, mut ob) = (Vec::new(), Vec::new());
            let va = view2_c32(&a32, &mut oa);
            let vb = view2_c32(&b32, &mut ob);
            let mut au = vec![0.0_f32; nt * self.engine.npoints];
            let mut av = vec![0.0_f32; nt * self.engine.npoints];
            let mut bu = vec![0.0_f32; nt * self.engine.npoints];
            let mut bv = vec![0.0_f32; nt * self.engine.npoints];
            maybe_detach(
                py,
                nt * self.engine.npoints >= engine::PAR_WORK_THRESHOLD,
                || {
                    self.engine.getgrad_pair(
                        &va,
                        &vb,
                        &mut au,
                        &mut av,
                        &mut bu,
                        &mut bv,
                        ntrunc,
                        south_to_north,
                    )
                },
            )
            .map_err(to_py_err)?;
            return Ok((
                to_py_grid2(py, nt, self.engine.npoints, au),
                to_py_grid2(py, nt, self.engine.npoints, av),
                to_py_grid2(py, nt, self.engine.npoints, bu),
                to_py_grid2(py, nt, self.engine.npoints, bv),
            ));
        }
        if let (Ok(a64), Ok(b64)) = (
            spec_a.extract::<PyReadonlyArray2<'py, Complex64>>(),
            spec_b.extract::<PyReadonlyArray2<'py, Complex64>>(),
        ) {
            let (nt, _nmdim, ntrunc) =
                validate_spec2_shape(a64.shape(), self.engine.nlat - 1, "getgrad_pair")?;
            if a64.shape() != b64.shape() {
                return Err(PyValueError::new_err(
                    "spec_a and spec_b must have the same shape",
                ));
            }
            let (mut oa, mut ob) = (Vec::new(), Vec::new());
            let va = view2_c64(&a64, &mut oa);
            let vb = view2_c64(&b64, &mut ob);
            let mut au = vec![0.0_f64; nt * self.engine.npoints];
            let mut av = vec![0.0_f64; nt * self.engine.npoints];
            let mut bu = vec![0.0_f64; nt * self.engine.npoints];
            let mut bv = vec![0.0_f64; nt * self.engine.npoints];
            maybe_detach(
                py,
                nt * self.engine.npoints >= engine::PAR_WORK_THRESHOLD,
                || {
                    self.engine.getgrad_pair(
                        &va,
                        &vb,
                        &mut au,
                        &mut av,
                        &mut bu,
                        &mut bv,
                        ntrunc,
                        south_to_north,
                    )
                },
            )
            .map_err(to_py_err)?;
            return Ok((
                to_py_grid2(py, nt, self.engine.npoints, au),
                to_py_grid2(py, nt, self.engine.npoints, av),
                to_py_grid2(py, nt, self.engine.npoints, bu),
                to_py_grid2(py, nt, self.engine.npoints, bv),
            ));
        }
        Err(PyTypeError::new_err(
            "getgrad_pair expects matching 2D (nt, nmdim) complex64 or complex128 arrays",
        ))
    }
}

/// Python module initialiser: registers the standalone functions and the two engine classes.
///
/// No direct Fortran counterpart (the Fortran build exposes the routines through f2py, see `_spherepack.pyf`).
#[pymodule]
fn _spharm_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gaussian_lats_wts, m)?)?;
    m.add_function(wrap_pyfunction!(getlegfunc, m)?)?;
    m.add_function(wrap_pyfunction!(ihgeod, m)?)?;
    m.add_function(wrap_pyfunction!(specintrp, m)?)?;
    m.add_function(wrap_pyfunction!(lap, m)?)?;
    m.add_function(wrap_pyfunction!(invlap, m)?)?;
    m.add_function(wrap_pyfunction!(multsmoothfact, m)?)?;
    m.add_class::<RustSpharmCore>()?;
    m.add_class::<RustReducedSpharmCore>()?;
    Ok(())
}
