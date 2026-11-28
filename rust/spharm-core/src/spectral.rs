//! Triangular spectral-coefficient layout and diagonal spectral operators (Laplacian, inverse Laplacian, degree-dependent smoothing, point interpolation).
//!
//! Spectral packing (triangular truncation T = `ntrunc`): the complex coefficients of one field are stored in a 1-D array of `nmdim = (T + 1) * (T + 2) / 2` entries, ordered with the zonal wavenumber `m = 0..=T` as the outer index and the degree `n = m..=T` as the inner index. With 0-based `m` and `n` the coefficient of `(m, n)` sits at `m * (2 * T - m + 3) / 2 + (n - m)`. The Fortran code (`lap.f`, `invlap.f`, `multsmoothfact.f`, `onedtotwod.f`, ...) uses 1-based `m`, `n` and computes the same position as `nm = nmstrt + n - m + 1`, `nmstrt = nmstrt + ntrunc - m + 2`. The truncation is recovered from the size with `ntrunc = -1.5 + 0.5 * sqrt(9 - 8 * (1 - nmdim))`.
//!
//! Batch layout: `nt` fields are stored C-order as `(nt, nmdim)`, which is the same memory layout as the Fortran array `dataspec(nmdim, nt)`.
//!
//! Difference from Fortran: 0-based indexing (so Fortran `n * (n - 1)` becomes `n * (n + 1)`), generic `f32`/`f64` coefficients, factors computed in `f64`, and rayon parallelism over fields for large batches.
//!
//! Derived from NCAR SPHEREPACK 3.2 (J. Adams and P. Swarztrauber), see LICENSE.spherepack.
use num_complex::Complex;
use rayon::prelude::*;

use crate::precision::ShtFloat;

/// Number of packed spectral coefficients for triangular truncation `ntrunc`: `nmdim = (ntrunc + 1) * (ntrunc + 2) / 2`.
///
/// Fortran: `twodtooned.f` / `twodtooned` declares `dataspec((ntrunc+1)*(ntrunc+2)/2, nt)`.
#[inline(always)]
pub fn nmdim_from_ntrunc(ntrunc: usize) -> usize {
    (ntrunc + 1) * (ntrunc + 2) / 2
}

/// Start index (0-based) of the zonal wavenumber block `m` inside the packed spectral array: `m * (2 * ntrunc - m + 3) / 2`.
///
/// The coefficient for `(m, n)` is at `spectral_offset(m, ntrunc) + (n - m)`, for `0 <= m <= n <= ntrunc`. Fortran (1-based): `nm = nmstrt + n - m + 1` with `nmstrt` advanced by `ntrunc - m + 2` after each `m`, in `onedtotwod.f`, `lap.f`, etc.
#[inline(always)]
pub fn spectral_offset(m: usize, ntrunc: usize) -> usize {
    m * (2 * ntrunc - m + 3) / 2
}

/// Invert `nmdim_from_ntrunc`: recover the triangular truncation `ntrunc` from the packed spectral length `nmdim`.
///
/// Returns `Err` (surfaced as `ValueError` in Python) if `nmdim == 0` or if `nmdim` is not of the form `(T + 1) * (T + 2) / 2`.
///
/// Fortran: `ntrunc = -1.5 + 0.5*sqrt(9.-8.*(1.-float(nmdim)))` at the top of `lap.f`, `invlap.f`, `multsmoothfact.f`, `onedtotwod.f`. Difference from Fortran: the real-valued root is rounded and then verified exactly, so malformed sizes are rejected instead of being silently truncated.
pub fn ntrunc_from_nmdim(nmdim: usize) -> Result<usize, String> {
    if nmdim == 0 {
        return Err("spectral dimension nmdim must be >= 1".to_string());
    }
    let disc = 9.0 + 8.0 * (nmdim as f64 - 1.0);
    let ntrunc_f = (-1.5 + 0.5 * disc.sqrt()).round();
    if ntrunc_f < 0.0 {
        return Err(format!("invalid triangular spectral dimension {nmdim}"));
    }
    let ntrunc = ntrunc_f as usize;
    if nmdim_from_ntrunc(ntrunc) != nmdim {
        return Err(format!(
            "invalid triangular spectral dimension {nmdim}: not of the form (T+1)*(T+2)/2"
        ));
    }
    Ok(ntrunc)
}

/// Fill `deg[n]` (0-based degree `n = 0..=ntrunc`) with the Laplacian eigenvalue `-n * (n + 1) / rsphere^2`.
///
/// Fortran: `lap.f` / `lap`, `-(float(n)*float(n-1)/rsphere**2)` with 1-based `n` (so `n1 * (n1 - 1)` equals `n * (n + 1)` for 0-based `n = n1 - 1`).
fn fill_lap_degree_factors<T: ShtFloat>(ntrunc: usize, rsphere: f64, deg: &mut [T]) {
    let inv_r2 = 1.0 / (rsphere * rsphere);
    for (n, slot) in deg.iter_mut().enumerate().take(ntrunc + 1) {
        let nf = n as f64;
        *slot = T::from_real(-nf * (nf + 1.0) * inv_r2);
    }
}

/// Fill `deg[n]` (0-based degree `n = 0..=ntrunc`) with the inverse-Laplacian factor `-rsphere^2 / (n * (n + 1))`, and `deg[0] = 0`.
///
/// The `n = 0` entry is set to zero because the Laplacian has no inverse on the global mean (constant) mode. Fortran: `invlap.f` / `invlap`, `-(rsphere**2/(float(n)*float(n-1)))`, loop starting at `n1 = 2` for `m = 1`, and `dataspec_ilap(1,i) = (0.,0.)`.
fn fill_invlap_degree_factors<T: ShtFloat>(ntrunc: usize, rsphere: f64, deg: &mut [T]) {
    let r2 = rsphere * rsphere;
    if !deg.is_empty() {
        deg[0] = T::ZERO;
    }
    for n in 1..=ntrunc {
        let nf = n as f64;
        deg[n] = T::from_real(-r2 / (nf * (nf + 1.0)));
    }
}

/// Multiply one packed spectral field `src` by a per-degree real factor `deg[n]` and write the result to `dst`.
///
/// Both slices have `nmdim` entries in the triangular order `m` outer, `n = m..=ntrunc` inner; entry `(m, n)` is scaled by `deg[n]`. Real and imaginary parts are scaled independently. Used when `nt == 1` to avoid building the expanded `nmdim` factor table.
///
/// Fortran: the `do m / do n=m,ntrunc+1 / nm = nmstrt + n - m + 1` double loop shared by `lap.f`, `invlap.f` and `multsmoothfact.f`.
fn apply_degree_factors_1d<T: ShtFloat>(
    src: &[Complex<T>],
    dst: &mut [Complex<T>],
    ntrunc: usize,
    deg: &[T],
) {
    let mut idx = 0usize;
    for m in 0..=ntrunc {
        for &f in &deg[m..=ntrunc] {
            let v = src[idx];
            dst[idx] = Complex::new(v.re * f, v.im * f);
            idx += 1;
        }
    }
}

/// Expand per-degree factors `deg[0..=ntrunc]` to one factor per packed coefficient (length `nmdim`): block `m` holds `deg[m..=ntrunc]`.
///
/// No direct Fortran counterpart (Fortran recomputes the factor inside the `(m, n)` loop).
fn expand_degree_factors<T: ShtFloat>(ntrunc: usize, deg: &[T]) -> Vec<T> {
    let nmdim = nmdim_from_ntrunc(ntrunc);
    let mut factors = Vec::with_capacity(nmdim);
    for m in 0..=ntrunc {
        factors.extend_from_slice(&deg[m..=ntrunc]);
    }
    factors
}

/// Build the packed table of Laplacian factors `-n * (n + 1) / rsphere^2`, one per `(m, n)` coefficient (length `nmdim`).
///
/// Fortran: `lap.f` / `lap` (factor part). No direct Fortran counterpart for the explicit table.
pub fn build_lap_factors<T: ShtFloat>(ntrunc: usize, rsphere: f64) -> Vec<T> {
    let mut deg = vec![T::ZERO; ntrunc + 1];
    fill_lap_degree_factors(ntrunc, rsphere, &mut deg);
    expand_degree_factors(ntrunc, &deg)
}

/// Build the packed table of inverse-Laplacian factors `-rsphere^2 / (n * (n + 1))`, with `0` for `n = 0`, one per `(m, n)` coefficient (length `nmdim`).
///
/// Fortran: `invlap.f` / `invlap` (factor part). No direct Fortran counterpart for the explicit table.
pub fn build_invlap_factors<T: ShtFloat>(ntrunc: usize, rsphere: f64) -> Vec<T> {
    let mut deg = vec![T::ZERO; ntrunc + 1];
    fill_invlap_degree_factors(ntrunc, rsphere, &mut deg);
    expand_degree_factors(ntrunc, &deg)
}

/// Apply the spherical Laplacian in spectral space to `nt` packed fields: `dst = -n * (n + 1) / rsphere^2 * src` for every `(m, n)`.
///
/// Parameters: `src`, `dst` are `nt * nmdim` complex values in C-order `(nt, nmdim)` (same memory layout as Fortran `dataspec(nmdim, nt)`); `nmdim` is the packed length for truncation `ntrunc`; `rsphere` is the sphere radius (same units as in the factor, any positive length). Returns nothing; `dst` is fully overwritten. `nt == 0` is a no-op.
///
/// Fortran: `lap.f` / `lap`. Difference from Fortran: degrees are 0-based; the factor is computed once per degree (stack buffer for `ntrunc < 256`, heap otherwise) rather than per coefficient; a single field (`nt == 1`) scales directly from the per-degree table, larger batches expand a packed factor table and run `apply_factors_batch`.
pub fn apply_lap_batch<T: ShtFloat>(
    src: &[Complex<T>],
    dst: &mut [Complex<T>],
    nt: usize,
    nmdim: usize,
    ntrunc: usize,
    rsphere: f64,
) {
    debug_assert_eq!(src.len(), nt * nmdim);
    debug_assert_eq!(dst.len(), nt * nmdim);
    if nt == 0 {
        return;
    }
    // Per-degree factors live on the stack when ntrunc < 256, otherwise on the heap.
    let mut stack_deg = [T::ZERO; 256];
    let mut heap_deg;
    let deg: &mut [T] = if ntrunc < 256 {
        &mut stack_deg[..ntrunc + 1]
    } else {
        heap_deg = vec![T::ZERO; ntrunc + 1];
        &mut heap_deg
    };
    // Fortran lap.f: -(float(n)*float(n-1)/rsphere**2) with 1-based n.
    fill_lap_degree_factors(ntrunc, rsphere, deg);
    // Single field: scale directly from the per-degree table, no expanded nmdim factor table needed.
    if nt == 1 {
        apply_degree_factors_1d(src, dst, ntrunc, deg);
        return;
    }
    let factors = expand_degree_factors(ntrunc, deg);
    apply_factors_batch(src, dst, nt, nmdim, &factors);
}

/// Apply the inverse spherical Laplacian in spectral space to `nt` packed fields: `dst = -rsphere^2 / (n * (n + 1)) * src` for `n >= 1`, and `dst = 0` for `n = 0` (the `(0, 0)` global-mean coefficient).
///
/// Parameters and layout as in `apply_lap_batch`. This is how streamfunction and velocity potential are obtained from vorticity and divergence (`psi = invlap(vrt)`, `chi = invlap(div)`).
///
/// Fortran: `invlap.f` / `invlap`. Difference from Fortran: Fortran skips `n = 1` (0-based `n = 0`) for `m = 1` (0-based `m = 0`) and zeroes `dataspec_ilap(1, i)` explicitly; here `deg[0] = 0` multiplies every `n = 0` coefficient, which is only the `(0, 0)` entry because `n >= m`.
pub fn apply_invlap_batch<T: ShtFloat>(
    src: &[Complex<T>],
    dst: &mut [Complex<T>],
    nt: usize,
    nmdim: usize,
    ntrunc: usize,
    rsphere: f64,
) {
    debug_assert_eq!(src.len(), nt * nmdim);
    debug_assert_eq!(dst.len(), nt * nmdim);
    if nt == 0 {
        return;
    }
    // Per-degree factors live on the stack when ntrunc < 256, otherwise on the heap.
    let mut stack_deg = [T::ZERO; 256];
    let mut heap_deg;
    let deg: &mut [T] = if ntrunc < 256 {
        &mut stack_deg[..ntrunc + 1]
    } else {
        heap_deg = vec![T::ZERO; ntrunc + 1];
        &mut heap_deg
    };
    // Fortran invlap.f: -(rsphere**2/(float(n)*float(n-1))) for n >= 2 (1-based); the (0,0) coefficient is set to zero.
    fill_invlap_degree_factors(ntrunc, rsphere, deg);
    // Single field: scale directly from the per-degree table, no expanded nmdim factor table needed.
    if nt == 1 {
        apply_degree_factors_1d(src, dst, ntrunc, deg);
        return;
    }
    let factors = expand_degree_factors(ntrunc, deg);
    apply_factors_batch(src, dst, nt, nmdim, &factors);
}

/// Multiply `nt` packed spectral fields by a degree-dependent smoothing factor: `dst(m, n) = src(m, n) * smooth[n]`.
///
/// Parameters: `src`, `dst` as in `apply_lap_batch`; `smooth` is a 1-D array indexed by 0-based total wavenumber `n` (at least `ntrunc + 1` entries are read; the Python wrapper rejects shorter arrays), of precision `S` independent of the coefficient precision `T`; each factor is converted through `f64` to `T`.
///
/// Fortran: `multsmoothfact.f` / `multsmoothfact`, `dataspec_smooth(nm,i) = dataspec(nm,i)*smooth(n)` with `smooth(nlat)` and 1-based `n`. Difference from Fortran: `smooth[n]` is 0-based and `nlat` is not an argument.
pub fn apply_multsmoothfact_batch<T: ShtFloat, S: ShtFloat>(
    src: &[Complex<T>],
    smooth: &[S],
    dst: &mut [Complex<T>],
    nt: usize,
    nmdim: usize,
    ntrunc: usize,
) {
    debug_assert_eq!(src.len(), nt * nmdim);
    debug_assert_eq!(dst.len(), nt * nmdim);
    if nt == 0 {
        return;
    }
    // Per-degree factors live on the stack when ntrunc < 256, otherwise on the heap.
    let mut stack_deg = [T::ZERO; 256];
    let mut heap_deg;
    let deg: &mut [T] = if ntrunc < 256 {
        &mut stack_deg[..ntrunc + 1]
    } else {
        heap_deg = vec![T::ZERO; ntrunc + 1];
        &mut heap_deg
    };
    // Fortran multsmoothfact.f: dataspec_smooth(nm,i) = dataspec(nm,i)*smooth(n).
    for n in 0..=ntrunc {
        deg[n] = T::from_real(smooth[n].to_f64());
    }
    // Single field: scale directly from the per-degree table, no expanded nmdim factor table needed.
    if nt == 1 {
        apply_degree_factors_1d(src, dst, ntrunc, deg);
        return;
    }
    let factors = expand_degree_factors(ntrunc, deg);
    apply_factors_batch(src, dst, nt, nmdim, &factors);
}

/// Multiply each of the `nt` rows of `src` element-wise by the packed factor table `factors` (length `nmdim`) and write to `dst`.
///
/// Runs serially for small problems (`nt <= 1` or `nt * nmdim < 1_048_576`) and in parallel over fields with rayon otherwise.
///
/// No direct Fortran counterpart (Fortran loops serially over `i = 1..nt`).
fn apply_factors_batch<T: ShtFloat>(
    src: &[Complex<T>],
    dst: &mut [Complex<T>],
    nt: usize,
    nmdim: usize,
    factors: &[T],
) {
    debug_assert_eq!(src.len(), nt * nmdim);
    debug_assert_eq!(dst.len(), nt * nmdim);
    debug_assert_eq!(factors.len(), nmdim);

    if nt <= 1 || nt * nmdim < 1_048_576 {
        for (in_row, out_row) in src.chunks_exact(nmdim).zip(dst.chunks_exact_mut(nmdim)) {
            for idx in 0..nmdim {
                let f = factors[idx];
                let v = in_row[idx];
                out_row[idx] = Complex::new(v.re * f, v.im * f);
            }
        }
    } else {
        dst.par_chunks_exact_mut(nmdim)
            .zip(src.par_chunks_exact(nmdim))
            .for_each(|(out_row, in_row)| {
                for idx in 0..nmdim {
                    let f = factors[idx];
                    let v = in_row[idx];
                    out_row[idx] = Complex::new(v.re * f, v.im * f);
                }
            });
    }
}

/// Evaluate a spectrally truncated function at one point from its packed coefficients.
///
/// Parameters: `rlon_rad` is the longitude of the point in radians; `dataspec` are the `nmdim` packed complex coefficients (precision `T`); `legfuncs` are the associated Legendre functions for the point's latitude in the same packed order (as returned by `getlegfunc`, precision `P`); `ntrunc` is the triangular truncation. Returns the interpolated value (always accumulated in `f64`).
///
/// Formula: with `S_m = sum_{n=m..ntrunc} dataspec(m, n) * legfuncs(m, n)`, the value is `Re(S_0) + sum_{m=1..ntrunc} 2 * (Re(S_m) * cos(m * lon) - Im(S_m) * sin(m * lon))`. Only the real part of the `m = 0` sum is used.
///
/// Fortran: `specintrp.f` / `specintrp`. Original comment: "given spectral coeffs of a function (datnm) truncated at T(ntrunc), longitude of pt in radians (rlon), and associated legendre polynomials for that pt (pnm), return interpolated function at that pt (ob)."
///
/// Difference from Fortran: no `scrm(ntrunc+1)` scratch array (the per-`m` sums are accumulated on the fly); `cos(m * lon)` and `sin(m * lon)` are advanced by the angle-addition recurrence from `cos(lon)`, `sin(lon)` and recomputed exactly every 64 values of `m`.
pub fn eval_specintrp<T: ShtFloat, P: ShtFloat>(
    rlon_rad: f64,
    dataspec: &[Complex<T>],
    legfuncs: &[P],
    ntrunc: usize,
) -> f64 {
    let mut offset = 0usize;
    // m = 0 block: only the real part contributes (Fortran: ob = scrm(1)).
    let mut sum_m0 = 0.0_f64;
    for _n in 0..=ntrunc {
        sum_m0 += dataspec[offset].re.to_f64() * legfuncs[offset].to_f64();
        offset += 1;
    }

    // ob accumulates the m = 0 term plus 2*Re(S_m * exp(i*m*lon)) for m >= 1 (Fortran: 2.0*real(scrm(m))*cos((m-1)*rlon) - 2.0*aimag(scrm(m))*sin((m-1)*rlon), 1-based m).
    let mut ob = sum_m0;
    if ntrunc == 0 {
        return ob;
    }
    let (s1, c1) = rlon_rad.sin_cos();
    let mut cos_m = c1;
    let mut sin_m = s1;
    for m in 1..=ntrunc {
        let mut sum_re = 0.0_f64;
        let mut sum_im = 0.0_f64;
        for _n in m..=ntrunc {
            let p = legfuncs[offset].to_f64();
            let c = dataspec[offset];
            sum_re += c.re.to_f64() * p;
            sum_im += c.im.to_f64() * p;
            offset += 1;
        }
        ob += 2.0 * (sum_re * cos_m - sum_im * sin_m);
        if m < ntrunc {
            // Every 64 steps recompute sin/cos of (m+1)*lon directly to bound the drift of the recurrence; otherwise use the angle-addition formulas.
            if (m & 63) == 0 {
                let (s, c) = (((m + 1) as f64) * rlon_rad).sin_cos();
                sin_m = s;
                cos_m = c;
            } else {
                let next_cos = cos_m * c1 - sin_m * s1;
                let next_sin = sin_m * c1 + cos_m * s1;
                cos_m = next_cos;
                sin_m = next_sin;
            }
        }
    }
    ob
}
