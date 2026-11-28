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

fn slice_c32<'a>(arr: &'a [Complex32]) -> &'a [Complex<f32>] {
    unsafe { std::slice::from_raw_parts(arr.as_ptr() as *const Complex<f32>, arr.len()) }
}

fn slice_c32_mut<'a>(arr: &'a mut [Complex32]) -> &'a mut [Complex<f32>] {
    unsafe { std::slice::from_raw_parts_mut(arr.as_mut_ptr() as *mut Complex<f32>, arr.len()) }
}

fn slice_c64<'a>(arr: &'a [Complex64]) -> &'a [Complex<f64>] {
    unsafe { std::slice::from_raw_parts(arr.as_ptr() as *const Complex<f64>, arr.len()) }
}

fn slice_c64_mut<'a>(arr: &'a mut [Complex64]) -> &'a mut [Complex<f64>] {
    unsafe { std::slice::from_raw_parts_mut(arr.as_mut_ptr() as *mut Complex<f64>, arr.len()) }
}

fn to_py_err(msg: String) -> PyErr {
    PyValueError::new_err(msg)
}

#[inline]
fn validate_rsphere(rsphere: f64) -> PyResult<f64> {
    if !(rsphere > 0.0) || !rsphere.is_finite() {
        return Err(PyValueError::new_err(format!(
            "rsphere must be positive and finite, got {rsphere}"
        )));
    }
    Ok(rsphere)
}

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

fn validate_grid3_shape(shape: &[usize], nlat: usize, nlon: usize, op: &str) -> PyResult<usize> {
    if shape.len() != 3 || shape[1] != nlat || shape[2] != nlon {
        return Err(PyValueError::new_err(format!(
            "{op} expects grid shape (nt, {nlat}, {nlon}), got {:?}",
            shape
        )));
    }
    Ok(shape[0])
}

fn validate_reduced_grid2(shape: &[usize], npoints: usize, op: &str) -> PyResult<usize> {
    if shape.len() != 2 || shape[1] != npoints {
        return Err(PyValueError::new_err(format!(
            "{op} expects reduced grid shape (nt, {npoints}), got {:?}",
            shape
        )));
    }
    Ok(shape[0])
}

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

    if len_s != len_l {
        return Err(PyValueError::new_err(format!(
            "dataspec length ({len_s}) must match legfuncs length ({len_l})"
        )));
    }
    let ntrunc = ntrunc_from_nmdim(len_s).map_err(to_py_err)?;
    let detach = len_s >= 65_536;

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

fn run_lap_like<'py>(
    py: Python<'py>,
    dataspec: &Bound<'py, PyAny>,
    rsphere: f64,
    inverse: bool,
    op_name: &str,
) -> PyResult<Py<PyAny>> {
    let rsphere = validate_rsphere(rsphere)?;
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

#[pyfunction]
fn lap<'py>(
    py: Python<'py>,
    dataspec: &Bound<'py, PyAny>,
    rsphere: f64,
) -> PyResult<Py<PyAny>> {
    run_lap_like(py, dataspec, rsphere, false, "lap")
}

#[pyfunction]
fn invlap<'py>(
    py: Python<'py>,
    dataspec: &Bound<'py, PyAny>,
    rsphere: f64,
) -> PyResult<Py<PyAny>> {
    run_lap_like(py, dataspec, rsphere, true, "invlap")
}

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

#[pyclass(frozen, module = "skybornspharm._spharm_rs")]
pub struct RustSpharmCore {
    engine: FullGridEngine,
}

impl RustSpharmCore {
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
                let per_field_bytes = 4 * nmdim * std::mem::size_of::<Complex<T>>();
                let tile_max = compute_tile_size(per_field_bytes, nt);
                let mut vrt = vec![T::complex_zero(); tile_max * nmdim];
                let mut div = vec![T::complex_zero(); tile_max * nmdim];
                let mut psispec = vec![T::complex_zero(); tile_max * nmdim];
                let mut chispec = vec![T::complex_zero(); tile_max * nmdim];

                for start in (0..nt).step_by(tile_max) {
                    let t = (nt - start).min(tile_max);
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
impl RustSpharmCore {
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

    #[getter]
    fn nlon(&self) -> usize {
        self.engine.nlon
    }

    #[getter]
    fn nlat(&self) -> usize {
        self.engine.nlat
    }

    #[getter]
    fn rsphere(&self) -> f64 {
        self.engine.rsphere
    }

    #[getter]
    fn gridtype(&self) -> &'static str {
        match self.engine.gridtype {
            GridType::Regular => "regular",
            GridType::Gaussian => "gaussian",
        }
    }

    #[getter]
    fn legfunc(&self) -> &str {
        &self.engine.legfunc
    }

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
        match (u64, u32, v64, v32) {
            (Some(u), None, Some(v), None) => {
                self.run_getpsichi(py, u.as_array(), v.as_array(), ntrunc, south_to_north)
            }
            (None, Some(u), None, Some(v)) => {
                self.run_getpsichi(py, u.as_array(), v.as_array(), ntrunc, south_to_north)
            }
            (Some(u), None, None, Some(v)) => {
                let vo = v.as_array().mapv(|x| x as f64);
                self.run_getpsichi(py, u.as_array(), vo.view(), ntrunc, south_to_north)
            }
            (None, Some(u), Some(v), None) => {
                let uo = u.as_array().mapv(|x| x as f64);
                self.run_getpsichi(py, uo.view(), v.as_array(), ntrunc, south_to_north)
            }
            _ => Err(PyTypeError::new_err(
                "getpsichi expects 3D (nt, nlat, nlon) float32 or float64 arrays",
            )),
        }
    }

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

#[pyclass(frozen, module = "skybornspharm._spharm_rs")]
pub struct RustReducedSpharmCore {
    engine: ReducedEngine,
}

impl RustReducedSpharmCore {
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
impl RustReducedSpharmCore {
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

    #[getter]
    fn nlat(&self) -> usize {
        self.engine.nlat
    }

    #[getter]
    fn npoints(&self) -> usize {
        self.engine.npoints
    }

    #[getter]
    fn rsphere(&self) -> f64 {
        self.engine.rsphere
    }

    #[getter]
    fn legfunc(&self) -> &str {
        &self.engine.legfunc
    }

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
