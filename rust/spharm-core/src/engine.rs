//! Full-grid spherical harmonic engine: scalar and vector analysis/synthesis on regular
//! (equally spaced in colatitude, poles included) and Gaussian latitude-longitude grids.
//!
//! Derived from NCAR SPHEREPACK 3.2 (J. Adams and P. Swarztrauber), see LICENSE.spherepack.
//!
//! The public entry points of [`FullGridEngine`] correspond to the following SPHEREPACK
//! routines (all for the full sphere without equatorial symmetry, i.e. `isym = 0` / `ityp = 0`):
//!
//! - `grdtospec`: scalar analysis, Fortran `shaec`/`shaes` (regular grid) and
//!   `shagc`/`shags` (Gaussian grid).
//! - `spectogrd`: scalar synthesis, Fortran `shsec`/`shses` (regular) and `shsgc`/`shsgs` (Gaussian).
//! - `getvrtdivspec`: vector analysis, Fortran `vhaec`/`vhaes` (regular) and `vhagc`/`vhags`
//!   (Gaussian), followed by the conversion of `br, bi, cr, ci` to vorticity and divergence
//!   spectra performed by `twodtooned_vrtdiv.f`.
//! - `getuv`: vector synthesis, Fortran `vhsec`/`vhses` (regular) and `vhsgc`/`vhsgs` (Gaussian),
//!   preceded by the conversion of vorticity and divergence spectra to `br, bi, cr, ci`
//!   performed by `onedtotwod_vrtdiv.f`.
//!
//! Common algorithm (all four routines):
//!
//! - The grid is split about the equator into a symmetric part `ge(i,j) = g(i,j) + g(nlat+1-i,j)`
//!   and an antisymmetric part `go(i,j) = g(i,j) - g(nlat+1-i,j)` (analysis), or recombined as
//!   `g(i,j) = ge + go` and `g(nlat+1-i,j) = ge - go` (synthesis). Associated Legendre functions
//!   of degree `n` and order `m` are symmetric about the equator when `n - m` is even and
//!   antisymmetric when `n - m` is odd, so each `m` couples `ge` only to `n - m` even harmonics
//!   and `go` only to `n - m` odd harmonics (the `zfin(2, ...)`/`zfin(1, ...)` and
//!   `alin(2, ...)`/`alin(1, ...)` calls of the Fortran sources).
//! - A real FFT along longitude produces the Fourier coefficients for `m = 0, ..., mmax-1`, where
//!   `mmax = min(nlat, nlon/2 + 1)` for scalars and `mmax = min(nlat, (nlon+1)/2)` for vectors.
//! - A Legendre transform per zonal wavenumber `m` (a matrix product here) maps between
//!   Fourier coefficients on the half sphere and the `n = m, ..., ntrunc` spectral coefficients.
//!
//! Spectral layout: complex coefficients for one field are stored in a packed triangular array of
//! length `nmdim = (ntrunc+1)*(ntrunc+2)/2`, ordered by `m` first and then by `n = m, ..., ntrunc`,
//! with index `spectral_offset(m, ntrunc) + (n - m)`. The scalar coefficient equals
//! `0.5 * (a(m+1,n+1) + i*b(m+1,n+1))` (`scale = 0.5` in `twodtooned.f`).
//!
//! Difference from Fortran: grids are C-ordered `(nt, nlat, nlon)` with a batch dimension `nt`
//! instead of `g(idg, jdg, nt)`; all indexing is 0-based; routines are generic over `f32`/`f64`;
//! `realfft` plans (see `FftCache`) replace `hrfftf`/`hrfftb`; `faer` matrix products replace the
//! explicit `do` loops over `i` and `np1`; Legendre tables come from `spharm_math` instead of the
//! `wshaec`-style workspace arrays (`zfinit`, `zvinit`, `alin`, ...), so there are no `wsave`,
//! `work`, `lwork` or `ierror` arguments; `rayon` parallelism is used across fields and
//! zonal wavenumbers.
use std::sync::OnceLock;

use faer::linalg::matmul::matmul;
use faer::mat::{MatMut, MatRef};
use faer::{Accum, Par};
use num_complex::Complex;
use numpy::ndarray;
use rayon::prelude::*;

use spharm_math::{GridType, ParityHarmonicTables, ScalarTables, SpharmFloat, VectorTables};

use crate::fft::FftCache;
use crate::precision::ShtFloat;
use crate::spectral::spectral_offset;

/// Upper bound on the number of fields (`nt` slices) processed together in one tile.
const MAX_TILE_SIZE: usize = 32;
/// Scratch-memory budget in bytes (64 MiB) used by `compute_tile_size` to cap the tile size.
const MAX_SCRATCH_BYTES: usize = 64 * 1024 * 1024;
/// Work-size threshold (in array elements) above which tile loops run in parallel with `rayon`.
///
/// No direct Fortran counterpart (SPHEREPACK is serial).
pub const PAR_WORK_THRESHOLD: usize = 32_768;

/// Floating-point types (`f32`, `f64`) for which the `faer` matrix-product helpers are provided.
///
/// No direct Fortran counterpart. Difference from Fortran: the `do` loop nests of the form
/// `a(mp1,np1,k) = a(mp1,np1,k) + zb(i,np1,i3)*ge(i,j,k)` (analysis, `shaec1`) and
/// `ge(i,j,k) = ge(i,j,k) + a(mp1,np1,k)*pb(i,np1,i3)` (synthesis, `shsec1`) become single
/// matrix products over the Legendre table of each `m`.
pub trait FaerGemmFloat: ShtFloat + SpharmFloat {
    /// Row-major general matrix product `dst = (Replace) or (dst +=) lhs * rhs` with unit scale.
    ///
    /// - `dst`: `dst_nrows x dst_ncols`, row-major (leading elements of the slice are used).
    /// - `lhs`: `lhs_nrows x lhs_ncols`, row-major if `lhs_row_major` else column-major.
    /// - `rhs`: `rhs_nrows x rhs_ncols`, row-major.
    /// - `accum`: `Accum::Replace` overwrites `dst`, `Accum::Add` accumulates into it.
    ///
    /// If `dst` is empty or the inner dimension is zero, nothing is multiplied and `dst` is
    /// zero-filled only for `Accum::Replace`.
    fn faer_gemm(
        dst: &mut [Self],
        dst_nrows: usize,
        dst_ncols: usize,
        accum: Accum,
        lhs: &[Self],
        lhs_nrows: usize,
        lhs_ncols: usize,
        lhs_row_major: bool,
        rhs: &[Self],
        rhs_nrows: usize,
        rhs_ncols: usize,
    );

    /// Matrix product that skips the first `j0` columns of `p_blk` and the first `j0` rows of `rhs`.
    ///
    /// `dst (nrows x ncols) = (Replace) or (+=) p_blk[:, j0..stride] * rhs[j0..stride, :]`, with
    /// `p_blk` row-major `nrows x stride` and `rhs` row-major `stride x ncols`. With `j0 == 0` this is
    /// the plain `faer_gemm`; with `j0 >= stride` the product is empty and `dst` is zero-filled only
    /// for `Accum::Replace`. Not used by the routines in this file.
    fn faer_gemm_subcols(
        dst: &mut [Self],
        nrows: usize,
        ncols: usize,
        accum: Accum,
        p_blk: &[Self],
        stride: usize,
        rhs: &[Self],
        j0: usize,
    );
}

// Implements `FaerGemmFloat` for a concrete float type using `faer::linalg::matmul::matmul` (sequential).
macro_rules! impl_faer_gemm {
    ($ty:ty) => {
        impl FaerGemmFloat for $ty {
            #[inline]
            fn faer_gemm(
                dst: &mut [Self],
                dst_nrows: usize,
                dst_ncols: usize,
                accum: Accum,
                lhs: &[Self],
                lhs_nrows: usize,
                lhs_ncols: usize,
                lhs_row_major: bool,
                rhs: &[Self],
                rhs_nrows: usize,
                rhs_ncols: usize,
            ) {
                if dst_nrows == 0 || dst_ncols == 0 || lhs_ncols == 0 {
                    if matches!(accum, Accum::Replace) {
                        dst[..dst_nrows * dst_ncols].fill(Self::ZERO);
                    }
                    return;
                }
                let dst_mat = MatMut::from_row_major_slice_mut(
                    &mut dst[..dst_nrows * dst_ncols],
                    dst_nrows,
                    dst_ncols,
                );
                let lhs_mat = if lhs_row_major {
                    MatRef::from_row_major_slice(&lhs[..lhs_nrows * lhs_ncols], lhs_nrows, lhs_ncols)
                } else {
                    MatRef::from_column_major_slice(
                        &lhs[..lhs_nrows * lhs_ncols],
                        lhs_nrows,
                        lhs_ncols,
                    )
                };
                let rhs_mat =
                    MatRef::from_row_major_slice(&rhs[..rhs_nrows * rhs_ncols], rhs_nrows, rhs_ncols);
                matmul(dst_mat, accum, lhs_mat, rhs_mat, 1.0 as $ty, Par::Seq);
            }

            #[inline]
            fn faer_gemm_subcols(
                dst: &mut [Self],
                nrows: usize,
                ncols: usize,
                accum: Accum,
                p_blk: &[Self],
                stride: usize,
                rhs: &[Self],
                j0: usize,
            ) {
                if j0 == 0 {
                    Self::faer_gemm(
                        dst,
                        nrows,
                        ncols,
                        accum,
                        p_blk,
                        nrows,
                        stride,
                        true,
                        rhs,
                        stride,
                        ncols,
                    );
                    return;
                }
                let active = stride.saturating_sub(j0);
                if active == 0 {
                    if matches!(accum, Accum::Replace) {
                        dst[..nrows * ncols].fill(Self::ZERO);
                    }
                    return;
                }
                let p_mat = MatRef::from_row_major_slice(p_blk, nrows, stride);
                let p_sub = p_mat.subcols(j0, active);
                let r_mat = MatRef::from_row_major_slice(rhs, stride, ncols);
                let r_sub = r_mat.subrows(j0, active);
                let d_mat =
                    MatMut::from_row_major_slice_mut(&mut dst[..nrows * ncols], nrows, ncols);
                matmul(d_mat, accum, p_sub, r_sub, 1.0 as $ty, Par::Seq);
            }
        }
    };
}

// Instantiate the GEMM helpers for single and double precision (SPHEREPACK single/double variants).
impl_faer_gemm!(f32);
impl_faer_gemm!(f64);

/// Number of fields to process per tile so that scratch memory stays within `MAX_SCRATCH_BYTES`.
///
/// - `per_field_bytes`: estimated scratch bytes needed per field.
/// - `nt`: total number of fields (the third index `k = 1, ..., nt` of the Fortran arrays).
///
/// Returns `0` if `nt == 0`, otherwise `min(nt, clamp(MAX_SCRATCH_BYTES / per_field_bytes, 1, 32))`.
///
/// No direct Fortran counterpart. Difference from Fortran: `shaec` and friends hold all `nt`
/// fields in `work` (sized by `lwork`), whereas here the batch is tiled to bound memory.
#[inline]
pub fn compute_tile_size(per_field_bytes: usize, nt: usize) -> usize {
    if nt == 0 {
        return 0;
    }
    let cap = (MAX_SCRATCH_BYTES / per_field_bytes.max(1)).clamp(1, MAX_TILE_SIZE);
    cap.min(nt)
}

/// Which vector spectra `getvrtdivspec` and `getuv` compute or use.
///
/// Corresponds to the `ityp` option of `vhaec`/`vhsec` for the full sphere without equatorial
/// symmetry: `ityp = 0` (`VrtDiv`, both `b` and `c` coefficients), `ityp = 1` (`DivOnly`, the
/// `cr, ci` coefficients are zero/skipped) and `ityp = 2` (`VrtOnly`, the `br, bi` coefficients
/// are zero/skipped). The symmetric options `ityp = 3, ..., 8` are not implemented here.
///
/// Fortran: vhaec.f / vhaec (parameter `ityp`), vhsec.f / vhsec (parameter `ityp`)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorMode {
    /// Both divergence (`br, bi`) and vorticity (`cr, ci`) coefficients (`ityp = 0`).
    VrtDiv,
    /// Vorticity only (`cr, ci`); the divergence (`br, bi`) part is skipped (`ityp = 2` style).
    VrtOnly,
    /// Divergence only (`br, bi`); the vorticity (`cr, ci`) part is skipped (`ityp = 1` style).
    DivOnly,
}

/// Spherical harmonic transform engine for one full lat-lon grid geometry.
///
/// Holds the grid geometry (`nlon`, `nlat`, `gridtype`), the sphere radius used by the vector
/// routines, and lazily built per-precision Legendre tables and FFT plans, which play the role of
/// the initialized workspace arrays `wshaec`, `wshsec`, `wvhaec`, `wvhsec`, ... of SPHEREPACK
/// (created by `shaeci`, `shseci`, `vhaeci`, `vhseci` and their Gaussian counterparts).
///
/// Difference from Fortran: tables are created on first use instead of by explicit `*i`
/// initialization routines, and are cached for both `f32` and `f64`.
pub struct FullGridEngine {
    /// Number of distinct longitude points (Fortran `nlon`), at least 4.
    pub nlon: usize,
    /// Number of grid points in colatitude on the full sphere (Fortran `nlat`), at least 3.
    pub nlat: usize,
    /// Sphere radius used to scale vorticity/divergence spectra (`rsphere` in `twodtooned_vrtdiv.f`).
    pub rsphere: f64,
    /// Latitude grid: regular (equally spaced, `shaec`/`shaes` family) or Gaussian (`shagc`/`shags` family).
    pub gridtype: GridType,
    /// Legendre function strategy requested by the caller, `stored` (as `shaes`) or `computed` (as `shaec`); validated in `new` only.
    pub legfunc: String,
    /// Lazily built single-precision parity-split Legendre tables.
    tables_f32: OnceLock<ParityHarmonicTables<f32>>,
    /// Lazily built double-precision parity-split Legendre tables.
    tables_f64: OnceLock<ParityHarmonicTables<f64>>,
    /// Real FFT plan cache for single precision (replaces `hrffti` workspace).
    fft_f32: FftCache<f32>,
    /// Real FFT plan cache for double precision (replaces `hrffti` workspace).
    fft_f64: FftCache<f64>,
}

/// Transform methods of the engine: scalar analysis/synthesis and vector analysis/synthesis.
///
/// See the module documentation for the shared algorithm and the correspondence with the
/// SPHEREPACK routines.
impl FullGridEngine {
    /// Create an engine for an `nlat x nlon` grid after validating the geometry.
    ///
    /// - `nlon`: number of longitudes, must be at least 4 (`ierror = 2` in `shaec`).
    /// - `nlat`: number of colatitudes including the poles (regular) or Gaussian points, must be at
    ///   least 3 (`ierror = 1` in `shaec`).
    /// - `rsphere`: sphere radius, must be positive and finite.
    /// - `gridtype`: `GridType::Regular` or `GridType::Gaussian`.
    /// - `legfunc`: `"stored"` or `"computed"`; mirrors the `shaes` (stored) versus `shaec`
    ///   (computed) choice but is only validated and recorded in this file.
    ///
    /// Returns an error string instead of the Fortran `ierror` codes when a check fails.
    ///
    /// Fortran: shaec.f / shaec, shaeci (`ierror` 1 and 2 checks on `nlat` and `nlon`)
    ///
    /// Difference from Fortran: `vhaec` accepts `nlon >= 1` (axisymmetric case `nlon = 1`), whereas
    /// this engine requires `nlon >= 4` for all transforms; no workspace is allocated here.
    pub fn new(
        nlon: usize,
        nlat: usize,
        rsphere: f64,
        gridtype: GridType,
        legfunc: String,
    ) -> Result<Self, String> {
        if nlon < 4 {
            return Err(format!("nlon must be >= 4, got {nlon}"));
        }
        if nlat < 3 {
            return Err(format!("nlat must be >= 3, got {nlat}"));
        }
        if !(rsphere > 0.0) || !rsphere.is_finite() {
            return Err(format!("rsphere must be positive, got {rsphere}"));
        }
        if legfunc != "stored" && legfunc != "computed" {
            return Err(format!(
                "legfunc must be 'stored' or 'computed', got '{legfunc}'"
            ));
        }
        Ok(Self {
            nlon,
            nlat,
            rsphere,
            gridtype,
            legfunc,
            tables_f32: OnceLock::new(),
            tables_f64: OnceLock::new(),
            fft_f32: FftCache::new(),
            fft_f64: FftCache::new(),
        })
    }

    /// Typed access to the lazily built parity-split Legendre tables (scalar and vector) for `T`.
    ///
    /// No direct Fortran counterpart; stands in for the workspace arrays filled by `shaeci`/`zfinit`.
    #[inline]
    fn tables<T: TypedTables>(&self) -> &ParityHarmonicTables<T> {
        T::get_tables(self)
    }

    /// Typed access to the real-FFT plan cache for `T`.
    ///
    /// No direct Fortran counterpart; stands in for `hrffti`/`whrfft` workspace initialization.
    #[inline]
    fn fft<T: TypedTables>(&self) -> &FftCache<T> {
        T::get_fft(self)
    }

    /// Scalar spherical harmonic analysis of `nt` grid fields into triangular complex spectra.
    ///
    /// For each field `g(theta_i, phi_j)` the output holds the complex coefficient
    /// `spec(m,n) = 0.5 * (a(m+1,n+1) + i*b(m+1,n+1))`, where `a`, `b` are the SPHEREPACK
    /// coefficients of `shaec` for `isym = 0`. With `phi_j = j*2*pi/nlon`, `theta_i` the
    /// colatitude of row `i` (uniform `pi*(i-1)/(nlat-1)` for regular grids, Gaussian points for
    /// Gaussian grids) and `pbar` the normalized associated Legendre functions, the Fortran
    /// definitions are
    ///
    /// ```text
    /// pbar(m,n,theta) = sqrt((2n+1) (n-m)! / (2 (n+m)!)) sin(theta)^m / (2^n n!)
    ///                   times the (n+m)th derivative of (x^2-1)^n with respect to x = cos(theta)
    /// c(m,i) = 2/nlon * sum over j of g(i,j) cos(m phi_j)
    /// s(m,i) = 2/nlon * sum over j of g(i,j) sin(m phi_j)
    /// a(m+1,n+1) = sum over i of c(m+1,i) zbar(m,n,theta_i)
    /// b(m+1,n+1) = sum over i of s(m+1,i) zbar(m,n,theta_i)
    /// mmax = min(nlat, nlon/2 + 1),  m = 0, ..., mmax-1,  n = m, ..., nlat-1
    /// ```
    ///
    /// where `zbar` is the quadrature function built from `pbar` (for the regular grid as given in the
    /// `shaec` header; for the Gaussian grid the Gaussian quadrature weights from `gaqd` enter
    /// instead, see `shagc`/`shags`). The truncation `ntrunc` limits the output to
    /// `n <= ntrunc` and `m <= ntrunc`.
    ///
    /// # Arguments
    ///
    /// - `grid`: input array of shape `(nt, nlat, nlon)`, C-order; `nt` is the number of analyses
    ///   (`k = 1, ..., nt` in Fortran).
    /// - `out_spec`: output slice of length at least `nt * nmdim`, with
    ///   `nmdim = (ntrunc+1)*(ntrunc+2)/2`; field `k` occupies `out_spec[k*nmdim..(k+1)*nmdim]` in
    ///   the packed `(m, n)` layout (`spectral_offset(m, ntrunc) + (n - m)`).
    /// - `ntrunc`: triangular truncation wavenumber.
    /// - `south_to_north`: `true` if grid row 0 is the southernmost latitude, `false` if row 0 is
    ///   the northernmost (SPHEREPACK order, `theta(1)` nearest the north pole).
    ///
    /// # Edge cases
    ///
    /// Returns immediately when `nt == 0`. `out_spec` is zero-filled first, so coefficients with
    /// `m >= mmax` or beyond the supported degree stay zero (Fortran zeroes `a(mp1,np1,k)` and
    /// `b(mp1,np1,k)` before accumulating). There are no `ierror` codes: the geometry is validated
    /// in `new`, and array shapes are not checked here (a too-short `out_spec` panics).
    ///
    /// Fortran: shaec.f / shaec (also shaes.f / shaes; for Gaussian grids shagc.f / shagc and
    /// shags.f / shags), with `twodtooned.f` for the packing into complex spectra
    ///
    /// Difference from Fortran: only the full-sphere case `isym = 0` is implemented; output is the
    /// packed complex spectrum scaled by 0.5 instead of separate `a`, `b` arrays of shape
    /// `(mdab, ndab, nt)`; the real FFT (`realfft`) replaces `hrfftf`; the per-`m` sums over `i` are
    /// matrix products (`faer`); the batch is tiled and processed in parallel (`rayon`).
    pub fn grdtospec<T: TypedTables>(
        &self,
        grid: &ndarray::ArrayView3<'_, T>,
        out_spec: &mut [Complex<T>],
        ntrunc: usize,
        south_to_north: bool,
    ) {
        // nt is the number of analyses (third index k = 1,...,nt of the Fortran arrays g, a, b).
        let nt = grid.shape()[0];
        let nmdim = crate::spectral::nmdim_from_ntrunc(ntrunc);
        if nt == 0 {
            return;
        }
        // Fortran zeroes a(mp1,np1,k) and b(mp1,np1,k) for mp1 = 1,...,mmax, np1 = mp1,...,nlat before accumulating.
        out_spec[..nt * nmdim].fill(T::complex_zero());

        let tables = self.tables::<T>();
        let sc: &ScalarTables<T> = tables.ensure_scalar();
        let imid = tables.imid;
        let imm1 = tables.imm1;
        // mmax = min0(nlat, nlon/2+1) (shaec1), here also limited to ntrunc+1 zonal wavenumbers.
        let mmax = tables.mmax_scalar.min(ntrunc + 1);
        let nlat = self.nlat;
        let nlon = self.nlon;
        let nfreq = nlon / 2 + 1;
        // modl = mod(nlat,2): for odd nlat the equator is grid row imid = (nlat+1)/2 and has no mirror row; imm1 = imid-1 (imm1 = imid for even nlat).
        let is_odd_nlat = nlat % 2 == 1;
        // For even nlon the m = nlon/2 Fourier term is the Nyquist term; Fortran halves it: ge(i,nlon,k) = .5*ge(i,nlon,k) (loop 36), and then uses only the cosine coefficient a (mdo = mmax-1 rule).
        let has_nyquist = nlon % 2 == 0 && (nlon / 2) < mmax;
        // Fortran scales the folded rows by tsn = 2/nlon (fsn = 4/nlon for hemispheric input); using 1/nlon yields the 0.5 * (a + i b) normalization of twodtooned.f.
        let inv_nlon = T::from_real(1.0 / (nlon as f64));

        let per_field_bytes =
            (nlat * nlon + 2 * mmax * imid * 2 + 2 * nlat * nfreq) * T::BYTES;
        let max_tile = compute_tile_size(per_field_bytes, nt);

        let fft_pair = self.fft::<T>().get_or_plan(nlon);
        let r2c = &fft_pair.r2c;
        let scratch_len = r2c.get_scratch_len();

        let contiguous = grid.as_slice();

        // Allocate reusable tile buffers once
        // Fourier buffers: for each m in 0..mmax, Fe_m has shape (imid, 2*max_tile), Fo_m has (imm1, 2*max_tile)
        let mut f_even = vec![T::ZERO; mmax * imid * 2 * max_tile];
        let mut f_odd = vec![T::ZERO; mmax * imm1 * 2 * max_tile];
        // Per-m spectral output buffer: C_m(max_rows, 2*max_tile)
        let max_rows_even = sc.p_even.rows_for_m(0, ntrunc).max(1);
        let mut spec_buf_m = vec![T::ZERO; mmax * max_rows_even * 2 * max_tile];

        // Pre-FFT folded rows: (tile * (imid + imm1)) rows of length nlon
        // Actually we can do fold + r2c in parallel over (k, i) and scatter into (k, i, m) then pack into f_even/f_odd,
        // OR store complex spectra directly per (k, i) in `spec_even_ki` and `spec_odd_ki`!
        let mut fft_even_ki = vec![T::complex_zero(); max_tile * imid * mmax];
        let mut fft_odd_ki = vec![T::complex_zero(); max_tile * imm1 * mmax];

        let mut start = 0usize;
        while start < nt {
            let tile = (nt - start).min(max_tile);
            let cols = 2 * tile;
            let par_fields = tile >= 2 && tile * nlat * nlon >= PAR_WORK_THRESHOLD;
            let par_m = mmax * imid * cols >= PAR_WORK_THRESHOLD;

            // Step 1: Fold N/S + r2c along lon
            let do_fold_r2c = |(tk, (fe_k, fo_k)): (usize, (&mut [Complex<T>], &mut [Complex<T>]))| {
                let k = start + tk;
                let mut real_e = vec![T::ZERO; nlon];
                let mut real_o = vec![T::ZERO; nlon];
                let mut cpx_buf = vec![T::complex_zero(); nfreq];
                let mut scratch = vec![T::complex_zero(); scratch_len];

                // Fortran (shaec1, isym = 0), for i = 1,...,imm1 with nlp1 = nlat+1:
                //   ge(i,j,k) = tsn*(g(i,j,k)+g(nlp1-i,j,k)), the part symmetric about the equator
                //   go(i,j,k) = tsn*(g(i,j,k)-g(nlp1-i,j,k)), the part antisymmetric about the equator
                // Followed by hrfftf along longitude (here a real-to-complex FFT). Row r_n is the northern row i and r_s its mirror.
                for i in 0..imm1 {
                    let r_n = if south_to_north { nlat - 1 - i } else { i };
                    let r_s = if south_to_north { i } else { nlat - 1 - i };
                    if let Some(sl) = contiguous {
                        let base_n = (k * nlat + r_n) * nlon;
                        let base_s = (k * nlat + r_s) * nlon;
                        let row_n = &sl[base_n..base_n + nlon];
                        let row_s = &sl[base_s..base_s + nlon];
                        for j in 0..nlon {
                            let vn = row_n[j];
                            let vs = row_s[j];
                            real_e[j] = (vn + vs) * inv_nlon;
                            real_o[j] = (vn - vs) * inv_nlon;
                        }
                    } else {
                        for j in 0..nlon {
                            let vn = grid[[k, r_n, j]];
                            let vs = grid[[k, r_s, j]];
                            real_e[j] = (vn + vs) * inv_nlon;
                            real_o[j] = (vn - vs) * inv_nlon;
                        }
                    }
                    r2c.process_with_scratch(&mut real_e, &mut cpx_buf, &mut scratch)
                        .expect("realfft r2c failed");
                    // Nyquist (m = nlon/2, even nlon) handling: halve the coefficient and drop its imaginary part (Fortran loop 36).
                    if has_nyquist {
                        cpx_buf[nlon / 2] =
                            Complex::new(cpx_buf[nlon / 2].re * T::HALF, T::ZERO);
                    }
                    fe_k[i * mmax..(i + 1) * mmax].copy_from_slice(&cpx_buf[..mmax]);

                    r2c.process_with_scratch(&mut real_o, &mut cpx_buf, &mut scratch)
                        .expect("realfft r2c failed");
                    if has_nyquist {
                        cpx_buf[nlon / 2] =
                            Complex::new(cpx_buf[nlon / 2].re * T::HALF, T::ZERO);
                    }
                    fo_k[i * mmax..(i + 1) * mmax].copy_from_slice(&cpx_buf[..mmax]);
                }

                // Odd nlat: equator row has no mirror. Fortran: ge(imid,j,k) = tsn*g(imid,j,k) (loop 25); it enters only the symmetric part.
                if is_odd_nlat {
                    let i = imid - 1;
                    let r_eq = imid - 1;
                    if let Some(sl) = contiguous {
                        let base_eq = (k * nlat + r_eq) * nlon;
                        let row_eq = &sl[base_eq..base_eq + nlon];
                        for j in 0..nlon {
                            real_e[j] = row_eq[j] * inv_nlon;
                        }
                    } else {
                        for j in 0..nlon {
                            real_e[j] = grid[[k, r_eq, j]] * inv_nlon;
                        }
                    }
                    r2c.process_with_scratch(&mut real_e, &mut cpx_buf, &mut scratch)
                        .expect("realfft r2c failed");
                    if has_nyquist {
                        cpx_buf[nlon / 2] =
                            Complex::new(cpx_buf[nlon / 2].re * T::HALF, T::ZERO);
                    }
                    fe_k[i * mmax..(i + 1) * mmax].copy_from_slice(&cpx_buf[..mmax]);
                }
            };
            if par_fields {
                fft_even_ki[..tile * imid * mmax]
                    .par_chunks_mut(imid * mmax)
                    .zip(fft_odd_ki[..tile * imm1 * mmax].par_chunks_mut(imm1 * mmax))
                    .enumerate()
                    .for_each(do_fold_r2c);
            } else {
                fft_even_ki[..tile * imid * mmax]
                    .chunks_mut(imid * mmax)
                    .zip(fft_odd_ki[..tile * imm1 * mmax].chunks_mut(imm1 * mmax))
                    .enumerate()
                    .for_each(do_fold_r2c);
            }

            // Step 2: Pack (k, i, m) -> f_even[m, i, 2*k..2*k+2] & f_odd[m, i, 2*k..2*k+2]
            let fe_m_stride = imid * cols;
            let fo_m_stride = imm1 * cols;
            let do_pack_m = |(m, (fe_m, fo_m)): (usize, (&mut [T], &mut [T]))| {
                for i in 0..imid {
                    let dst_row = &mut fe_m[i * cols..(i + 1) * cols];
                    for tk in 0..tile {
                        let c = fft_even_ki[(tk * imid + i) * mmax + m];
                        dst_row[2 * tk] = c.re;
                        dst_row[2 * tk + 1] = c.im;
                    }
                }
                for i in 0..imm1 {
                    let dst_row = &mut fo_m[i * cols..(i + 1) * cols];
                    for tk in 0..tile {
                        let c = fft_odd_ki[(tk * imm1 + i) * mmax + m];
                        dst_row[2 * tk] = c.re;
                        dst_row[2 * tk + 1] = c.im;
                    }
                }
            };
            if par_m {
                f_even[..mmax * fe_m_stride]
                    .par_chunks_mut(fe_m_stride)
                    .zip(f_odd[..mmax * fo_m_stride].par_chunks_mut(fo_m_stride))
                    .enumerate()
                    .for_each(do_pack_m);
            } else {
                f_even[..mmax * fe_m_stride]
                    .chunks_mut(fe_m_stride)
                    .zip(f_odd[..mmax * fo_m_stride].chunks_mut(fo_m_stride))
                    .enumerate()
                    .for_each(do_pack_m);
            }

            // Fortran (shaec1): symmetric part ge couples to n - m even harmonics, a(mp1,np1) += zb(i,np1,i3)*ge(i,2*mp1-2) and
            // b(mp1,np1) += zb(i,np1,i3)*ge(i,2*mp1-1) for np1 = mp1, mp1+2, ..., ndo (zfin(2,...)).
            // Here (cos, sin) Fourier parts are the real and imaginary parts of the FFT bin, and the sum over i is a matrix product
            // of the even Z table with f_even for each m.
            // Step 3: Legendre GEMM per m -> store into out_spec
            let spec_m_stride = max_rows_even * cols;
            let do_gemm_even = |(m, spec_m): (usize, &mut [T])| {
                let ne = sc.z_even.rows_for_m(m, ntrunc);
                if ne > 0 && imid > 0 {
                    let z_blk = sc.z_even.block_for_m(m, ntrunc);
                    let fe_m = &f_even[m * fe_m_stride..(m + 1) * fe_m_stride];
                    T::faer_gemm(
                        spec_m,
                        ne,
                        cols,
                        Accum::Replace,
                        z_blk,
                        ne,
                        imid,
                        true,
                        fe_m,
                        imid,
                        cols,
                    );
                }
            };
            if par_m {
                spec_buf_m[..mmax * spec_m_stride]
                    .par_chunks_mut(spec_m_stride)
                    .enumerate()
                    .for_each(do_gemm_even);
            } else {
                spec_buf_m[..mmax * spec_m_stride]
                    .chunks_mut(spec_m_stride)
                    .enumerate()
                    .for_each(do_gemm_even);
            }

            // Copy the n - m even rows (n = m + 2r, packed index offset(m) + 2r) into the complex output.
            // Copy even rows into out_spec
            for m in 0..mmax {
                let ne = sc.z_even.rows_for_m(m, ntrunc);
                if ne > 0 {
                    let m_base = spectral_offset(m, ntrunc);
                    let spec_m = &spec_buf_m[m * spec_m_stride..];
                    for r in 0..ne {
                        let nm = m_base + 2 * r;
                        let src_row = &spec_m[r * cols..(r + 1) * cols];
                        for tk in 0..tile {
                            out_spec[(start + tk) * nmdim + nm] =
                                Complex::new(src_row[2 * tk], src_row[2 * tk + 1]);
                        }
                    }
                }
            }

            // Fortran: antisymmetric part go couples to n - m odd harmonics (zfin(1,...), np1 = mp1+1, mp1+3, ..., ndo), summing over i = 1,...,imm1.
            // Odd parity GEMM
            let do_gemm_odd = |(m, spec_m): (usize, &mut [T])| {
                let no = sc.z_odd.rows_for_m(m, ntrunc);
                if no > 0 && imm1 > 0 {
                    let z_blk = sc.z_odd.block_for_m(m, ntrunc);
                    let fo_m = &f_odd[m * fo_m_stride..(m + 1) * fo_m_stride];
                    T::faer_gemm(
                        spec_m,
                        no,
                        cols,
                        Accum::Replace,
                        z_blk,
                        no,
                        imm1,
                        true,
                        fo_m,
                        imm1,
                        cols,
                    );
                }
            };
            if par_m {
                spec_buf_m[..mmax * spec_m_stride]
                    .par_chunks_mut(spec_m_stride)
                    .enumerate()
                    .for_each(do_gemm_odd);
            } else {
                spec_buf_m[..mmax * spec_m_stride]
                    .chunks_mut(spec_m_stride)
                    .enumerate()
                    .for_each(do_gemm_odd);
            }

            // Copy the n - m odd rows (n = m + 1 + 2r, packed index offset(m) + 1 + 2r) into the complex output.
            for m in 0..mmax {
                let no = sc.z_odd.rows_for_m(m, ntrunc);
                if no > 0 {
                    let m_base = spectral_offset(m, ntrunc);
                    let spec_m = &spec_buf_m[m * spec_m_stride..];
                    for r in 0..no {
                        let nm = m_base + 1 + 2 * r;
                        let src_row = &spec_m[r * cols..(r + 1) * cols];
                        for tk in 0..tile {
                            out_spec[(start + tk) * nmdim + nm] =
                                Complex::new(src_row[2 * tk], src_row[2 * tk + 1]);
                        }
                    }
                }
            }

            start += tile;
        }
    }

    /// Scalar spherical harmonic synthesis of `nt` triangular complex spectra onto the grid.
    ///
    /// Inverse of `grdtospec`: for each field it evaluates, with `pbar` the normalized associated
    /// Legendre functions and `mmax = min(nlat, nlon/2 + 1)`, the SPHEREPACK synthesis (`shsec`
    /// header, `isym = 0`)
    ///
    /// ```text
    /// g(i,j) = sum over n = 0..nlat-1 of  .5 * pbar(0,n,theta(i)) * a(1,n+1)
    ///        + sum over m = 1..mmax-1 of sum over n = m..nlat-1 of
    ///            pbar(m,n,theta(i)) * ( a(m+1,n+1) cos(m phi(j)) - b(m+1,n+1) sin(m phi(j)) )
    /// ```
    ///
    /// with `a(m+1,n+1) + i*b(m+1,n+1) = 2 * spec(m,n)` (the 0.5 scale of `twodtooned.f`). The sum is
    /// restricted to `n <= ntrunc`.
    ///
    /// # Arguments
    ///
    /// - `spec`: complex spectra, shape `(nt, nmdim)` with `nmdim = (ntrunc+1)*(ntrunc+2)/2`, in the
    ///   packed `(m, n)` layout; `nt = spec.shape()[0]`.
    /// - `ntrunc`: triangular truncation wavenumber.
    /// - `out_grid`: output slice of length at least `nt * nlat * nlon`, C-order `(nt, nlat, nlon)`.
    /// - `south_to_north`: `true` if grid row 0 is the southernmost latitude, `false` if row 0 is the
    ///   northernmost (SPHEREPACK order).
    ///
    /// # Edge cases
    ///
    /// Returns immediately for `nt == 0`. For even `nlon`, the `m = nlon/2` (Nyquist) term uses only
    /// the cosine part (imaginary part dropped, real part doubled), as in `shsec1` where the
    /// `mdo = mmax-1` rule keeps only `a(mmax,np1,k)` and `ge(i,nlon,k)` is multiplied by 2. For
    /// `m = 0` the imaginary part is ignored. There are no `ierror` codes; `out_grid` is fully
    /// overwritten.
    ///
    /// Fortran: shsec.f / shsec (also shses.f / shses; Gaussian grids shsgc.f / shsgc and
    /// shsgs.f / shsgs), with `onedtotwod.f` for the unpacking of the complex spectra
    ///
    /// Difference from Fortran: only `isym = 0` is implemented; input is the packed complex spectrum
    /// instead of `a`, `b` arrays; the inverse real FFT (`realfft` c2r) replaces `hrfftb`; per-`m`
    /// Legendre sums are matrix products (`faer`); batch tiling and `rayon` parallelism are used.
    pub fn spectogrd<T: TypedTables>(
        &self,
        spec: &ndarray::ArrayView2<'_, Complex<T>>,
        ntrunc: usize,
        out_grid: &mut [T],
        south_to_north: bool,
    ) {
        let nt = spec.shape()[0];
        if nt == 0 {
            return;
        }
        let tables = self.tables::<T>();
        let sc: &ScalarTables<T> = tables.ensure_scalar();
        let imid = tables.imid;
        let imm1 = tables.imm1;
        // mmax = min0(nlat, nlon/2+1) (shsec1), here also limited to ntrunc+1 zonal wavenumbers.
        let mmax = tables.mmax_scalar.min(ntrunc + 1);
        let nlat = self.nlat;
        let nlon = self.nlon;
        let nfreq = nlon / 2 + 1;
        // Odd nlat: the equator row (imid) is not mirrored; modl = mod(nlat,2), imm1 = imid-1 for odd nlat else imid (shsec1).
        let is_odd_nlat = nlat % 2 == 1;
        // Even nlon: the Nyquist term m = nlon/2 exists when it lies below mmax (mdo = mmax-1 rule of shsec1).
        let has_nyquist = nlon % 2 == 0 && (nlon / 2) < mmax;

        let per_field_bytes =
            (nlat * nlon + 2 * mmax * imid * 2 + 2 * nlat * nfreq) * T::BYTES;
        let max_tile = compute_tile_size(per_field_bytes, nt);

        let fft_pair = self.fft::<T>().get_or_plan(nlon);
        let c2r = &fft_pair.c2r;
        let scratch_len = c2r.get_scratch_len();

        let mut f_even = vec![T::ZERO; mmax * imid * 2 * max_tile];
        let mut f_odd = vec![T::ZERO; mmax * imm1 * 2 * max_tile];
        let max_rows = sc.p_even.rows_for_m(0, ntrunc).max(1);
        let mut spec_buf_m = vec![T::ZERO; mmax * max_rows * 2 * max_tile];

        let mut start = 0usize;
        while start < nt {
            let tile = (nt - start).min(max_tile);
            let cols = 2 * tile;
            let fe_m_stride = imid * cols;
            let fo_m_stride = imm1 * cols;
            let spec_m_stride = max_rows * cols;
            let par_fields = tile >= 2 && tile * nlat * nlon >= PAR_WORK_THRESHOLD;
            let par_m = mmax * imid * cols >= PAR_WORK_THRESHOLD;

            // Fortran (shsec1): ge(i,2*mp1-2,k) += a(mp1,np1,k)*pb(i,np1,i3) and ge(i,2*mp1-1,k) += b(mp1,np1,k)*pb(i,np1,i3)
            // for np1 = mp1, mp1+2, ..., ndo (n - m even, alin(2,...)). The sum over n is the matrix product P_even^T * spec_m.
            // Even parity: pack spec_m then GEMM P_even^T * spec_m -> f_even
            let do_even_m = |(m, (fe_m, spec_m)): (usize, (&mut [T], &mut [T]))| {
                let ne = sc.p_even.rows_for_m(m, ntrunc);
                if ne == 0 || imid == 0 {
                    fe_m.fill(T::ZERO);
                    return;
                }
                let m_base = spectral_offset(m, ntrunc);
                // Nyquist term (even nlon, m = nlon/2): shsec1 doubles ge(i,nlon,k) before hrfftb, because the inverse real FFT does not
                // double this bin as it does for 0 < m < nlon/2. Only the cosine (real) part is used; the imaginary part is set to zero below.
                let nyq_scale = if has_nyquist && m == nlon / 2 {
                    T::TWO
                } else {
                    T::ONE
                };
                for r in 0..ne {
                    let nm = m_base + 2 * r;
                    let dst_row = &mut spec_m[r * cols..(r + 1) * cols];
                    for tk in 0..tile {
                        let c = spec[[start + tk, nm]];
                        dst_row[2 * tk] = c.re * nyq_scale;
                        // For m = 0 (and the Nyquist term) the sine part does not exist, so the imaginary part is dropped.
                        dst_row[2 * tk + 1] = if m == 0 || (has_nyquist && m == nlon / 2) {
                            T::ZERO
                        } else {
                            c.im
                        };
                    }
                }
                let p_blk = sc.p_even.block_for_m(m, ntrunc);
                T::faer_gemm(
                    fe_m,
                    imid,
                    cols,
                    Accum::Replace,
                    p_blk,
                    imid,
                    ne,
                    false,
                    spec_m,
                    ne,
                    cols,
                );
            };
            if par_m {
                f_even[..mmax * fe_m_stride]
                    .par_chunks_mut(fe_m_stride)
                    .zip(spec_buf_m[..mmax * spec_m_stride].par_chunks_mut(spec_m_stride))
                    .enumerate()
                    .for_each(do_even_m);
            } else {
                f_even[..mmax * fe_m_stride]
                    .chunks_mut(fe_m_stride)
                    .zip(spec_buf_m[..mmax * spec_m_stride].chunks_mut(spec_m_stride))
                    .enumerate()
                    .for_each(do_even_m);
            }

            // Fortran (shsec1): go(i,2*mp1-2,k) += a(mp1,np1,k)*pb(i,np1,i3) and go(i,2*mp1-1,k) += b(mp1,np1,k)*pb(i,np1,i3)
            // for np1 = mp1+1, mp1+3, ..., ndo (n - m odd, alin(1,...)), over i = 1,...,imm1.
            // Odd parity: pack spec_m then GEMM P_odd^T * spec_m -> f_odd
            let do_odd_m = |(m, (fo_m, spec_m)): (usize, (&mut [T], &mut [T]))| {
                let no = sc.p_odd.rows_for_m(m, ntrunc);
                if no == 0 || imm1 == 0 {
                    fo_m.fill(T::ZERO);
                    return;
                }
                let m_base = spectral_offset(m, ntrunc);
                let nyq_scale = if has_nyquist && m == nlon / 2 {
                    T::TWO
                } else {
                    T::ONE
                };
                for r in 0..no {
                    let nm = m_base + 1 + 2 * r;
                    let dst_row = &mut spec_m[r * cols..(r + 1) * cols];
                    for tk in 0..tile {
                        let c = spec[[start + tk, nm]];
                        dst_row[2 * tk] = c.re * nyq_scale;
                        dst_row[2 * tk + 1] = if m == 0 || (has_nyquist && m == nlon / 2) {
                            T::ZERO
                        } else {
                            c.im
                        };
                    }
                }
                let p_blk = sc.p_odd.block_for_m(m, ntrunc);
                T::faer_gemm(
                    fo_m,
                    imm1,
                    cols,
                    Accum::Replace,
                    p_blk,
                    imm1,
                    no,
                    false,
                    spec_m,
                    no,
                    cols,
                );
            };
            if par_m {
                f_odd[..mmax * fo_m_stride]
                    .par_chunks_mut(fo_m_stride)
                    .zip(spec_buf_m[..mmax * spec_m_stride].par_chunks_mut(spec_m_stride))
                    .enumerate()
                    .for_each(do_odd_m);
            } else {
                f_odd[..mmax * fo_m_stride]
                    .chunks_mut(fo_m_stride)
                    .zip(spec_buf_m[..mmax * spec_m_stride].chunks_mut(spec_m_stride))
                    .enumerate()
                    .for_each(do_odd_m);
            }

            // Fortran (shsec1, isym = 0): after hrfftb along longitude,
            //   g(i,j,k)       = .5*(ge(i,j,k)+go(i,j,k))
            //   g(nlp1-i,j,k)  = .5*(ge(i,j,k)-go(i,j,k))
            // for i = 1,...,imm1. The factor .5 is absorbed here by the 0.5 scale of the complex spectrum.
            // Step 3: c2r + unfold across fields in tile
            let out_tile_slice = &mut out_grid[start * nlat * nlon..(start + tile) * nlat * nlon];
            let do_c2r_unfold = |(tk, field_out): (usize, &mut [T])| {
                let mut cpx_e = vec![T::complex_zero(); nfreq];
                let mut cpx_o = vec![T::complex_zero(); nfreq];
                let mut real_e = vec![T::ZERO; nlon];
                let mut real_o = vec![T::ZERO; nlon];
                let mut scratch = vec![T::complex_zero(); scratch_len];

                for i in 0..imm1 {
                    cpx_e.fill(T::complex_zero());
                    cpx_o.fill(T::complex_zero());
                    for m in 0..mmax {
                        let fe_idx = m * fe_m_stride + i * cols + 2 * tk;
                        let fo_idx = m * fo_m_stride + i * cols + 2 * tk;
                        cpx_e[m] = Complex::new(f_even[fe_idx], f_even[fe_idx + 1]);
                        cpx_o[m] = Complex::new(f_odd[fo_idx], f_odd[fo_idx + 1]);
                    }
                    // Real-FFT consistency: the m = 0 bin (and the Nyquist bin for even nlon) must be real for the inverse transform.
                    cpx_e[0].im = T::ZERO;
                    cpx_o[0].im = T::ZERO;
                    if nlon % 2 == 0 {
                        cpx_e[nlon / 2].im = T::ZERO;
                        cpx_o[nlon / 2].im = T::ZERO;
                    }
                    c2r.process_with_scratch(&mut cpx_e, &mut real_e, &mut scratch)
                        .expect("realfft c2r failed");
                    c2r.process_with_scratch(&mut cpx_o, &mut real_o, &mut scratch)
                        .expect("realfft c2r failed");

                    let r_n = if south_to_north { nlat - 1 - i } else { i };
                    let r_s = if south_to_north { i } else { nlat - 1 - i };
                    let base_n = r_n * nlon;
                    let base_s = r_s * nlon;
                    for j in 0..nlon {
                        let ge_v = real_e[j];
                        let go_v = real_o[j];
                        field_out[base_n + j] = ge_v + go_v;
                        field_out[base_s + j] = ge_v - go_v;
                    }
                }

                // Odd nlat: the equator row is purely symmetric, g(imid,j,k) = ge(imid,j,k) (no go contribution).
                if is_odd_nlat {
                    let i = imid - 1;
                    cpx_e.fill(T::complex_zero());
                    for m in 0..mmax {
                        let fe_idx = m * fe_m_stride + i * cols + 2 * tk;
                        cpx_e[m] = Complex::new(f_even[fe_idx], f_even[fe_idx + 1]);
                    }
                    cpx_e[0].im = T::ZERO;
                    if nlon % 2 == 0 {
                        cpx_e[nlon / 2].im = T::ZERO;
                    }
                    c2r.process_with_scratch(&mut cpx_e, &mut real_e, &mut scratch)
                        .expect("realfft c2r failed");
                    let base_eq = i * nlon;
                    field_out[base_eq..base_eq + nlon].copy_from_slice(&real_e);
                }
            };
            if par_fields {
                out_tile_slice
                    .par_chunks_mut(nlat * nlon)
                    .enumerate()
                    .for_each(do_c2r_unfold);
            } else {
                out_tile_slice
                    .chunks_mut(nlat * nlon)
                    .enumerate()
                    .for_each(do_c2r_unfold);
            }

            start += tile;
        }
    }

    /// Vector spherical harmonic analysis of `(u, v)` into vorticity and/or divergence spectra.
    ///
    /// For `nt` vector fields given by the eastward component `u` and northward component `v`, it
    /// computes the complex spectra of relative vorticity and divergence (the spharm
    /// `getvrtdivspec`). Internally the SPHEREPACK vector field `(v_sp, w)` is used, where `v_sp` is
    /// the colatitudinal component (positive southward, measured from the north pole) and `w` the
    /// east longitudinal component, so `v_sp = -v` and `w = u`. The Fortran analysis `vhaec`
    /// produces `br, bi, cr, ci` such that, with `b(m,n) = br(m+1,n+1) + i*bi(m+1,n+1)` and
    /// `c(m,n) = cr(m+1,n+1) + i*ci(m+1,n+1)`,
    ///
    /// ```text
    /// h(i,j) = (v_sp, w) = sum over n = 1..nlat-1 of Re[ .5*(b(0,n) bbar(0,n,theta_i) + c(0,n) cbar(0,n,theta_i)) ]
    ///        + sum over m = 1..mmax-1, n = m..nlat-1 of Re[ (b(m,n) bbar(m,n,theta_i) + c(m,n) cbar(m,n,theta_i)) exp(i m phi_j) ]
    /// bbar = (vbar, i*wbar),  cbar = (i*wbar, -vbar)
    /// vbar(m,n,theta) = d pbar(m,n,theta)/d theta / sqrt(n(n+1))
    /// wbar(m,n,theta) = m/sin(theta) * pbar(m,n,theta) / sqrt(n(n+1))
    /// mmax = min(nlat, (nlon+1)/2)  (SPHEREPACK vector wavenumber limit)
    /// ```
    ///
    /// and `twodtooned_vrtdiv.f` converts them to the output spectra (Fortran 1-based `n`, here
    /// 0-based degree `n`, factors `sqrt(n(n+1))`):
    ///
    /// ```text
    /// divspec(m,n) = -(sqrt(n(n+1)) / rsphere) * 0.5 * (br + i*bi)
    /// vrtspec(m,n) =  (sqrt(n(n+1)) / rsphere) * 0.5 * (cr + i*ci)
    /// ```
    ///
    /// # Arguments
    ///
    /// - `ugrid`, `vgrid`: eastward and northward components, shape `(nt, nlat, nlon)`, C-order.
    /// - `out_vrt`: optional output of length at least `nt * nmdim` for the vorticity spectra
    ///   (`nmdim = (ntrunc+1)*(ntrunc+2)/2`, packed `(m, n)` layout); used by `VrtDiv` and `VrtOnly`.
    /// - `out_div`: optional output for the divergence spectra, same layout; used by `VrtDiv` and
    ///   `DivOnly`.
    /// - `ntrunc`: triangular truncation wavenumber.
    /// - `south_to_north`: `true` if grid row 0 is the southernmost latitude, `false` if row 0 is the
    ///   northernmost.
    /// - `mode`: `VectorMode`, equivalent to `ityp = 0` (`VrtDiv`), `1` (`DivOnly`), `2` (`VrtOnly`).
    ///
    /// # Edge cases
    ///
    /// Returns immediately for `nt == 0`. Provided outputs are zero-filled first. Coefficients for
    /// `n = 0` and for `n = nlat-1` with even `m` are always zero (in Fortran,
    /// `br(m+1,nlat), bi, cr, ci` are assumed zero for even `m`, the `ndo1`/`ndo2` rule). Zonal
    /// wavenumbers `m >= mmax` stay zero. No `ierror` codes (Fortran `ierror` 1 to 10).
    ///
    /// Fortran: vhaec.f / vhaec (also vhaes.f / vhaes; Gaussian grids vhagc.f / vhagc and
    /// vhags.f / vhags), with `twodtooned_vrtdiv.f` / `twodtooned_vrtdiv`
    ///
    /// Difference from Fortran: only the full-sphere case (`ityp = 0, 1, 2`) is implemented; `br, bi,
    /// cr, ci` arrays of shape `(mdab, ndab, nt)` are replaced by packed complex vorticity/divergence
    /// spectra (with the 0.5 scale and `rsphere` factor applied directly); input components use the
    /// geographic sign convention (`v_sp = -v`); real FFTs, `faer` matrix products, batch tiling and
    /// `rayon` parallelism replace `hrfftf` and the explicit loops.
    pub fn getvrtdivspec<T: TypedTables>(
        &self,
        ugrid: &ndarray::ArrayView3<'_, T>,
        vgrid: &ndarray::ArrayView3<'_, T>,
        mut out_vrt: Option<&mut [Complex<T>]>,
        mut out_div: Option<&mut [Complex<T>]>,
        ntrunc: usize,
        south_to_north: bool,
        mode: VectorMode,
    ) {
        let nt = ugrid.shape()[0];
        let nmdim = crate::spectral::nmdim_from_ntrunc(ntrunc);
        if nt == 0 {
            return;
        }
        // Zero the requested outputs; spectral entries not reached below (n = 0, n = nlat-1 with m even, m >= mmax) stay zero.
        if let Some(ref mut v) = out_vrt {
            v[..nt * nmdim].fill(T::complex_zero());
        }
        if let Some(ref mut d) = out_div {
            d[..nt * nmdim].fill(T::complex_zero());
        }

        let tables = self.tables::<T>();
        let vt: &VectorTables<T> = tables.ensure_vector();
        let imid = tables.imid;
        let imm1 = tables.imm1;
        // mmax = min0(nlat, (nlon+1)/2) for vectors (vhaec1), here also limited to ntrunc+1.
        let mmax = tables.mmax_vector.min(ntrunc + 1);
        let nlat = self.nlat;
        let nlon = self.nlon;
        let nfreq = nlon / 2 + 1;
        let is_odd_nlat = nlat % 2 == 1;
        // The 0.5 scale of twodtooned_vrtdiv.f (scale = 0.5) is absorbed by folding with 1/nlon instead of the Fortran tsn = 2/nlon.
        let inv_nlon = T::from_real(1.0 / (nlon as f64));
        let inv_r = 1.0 / self.rsphere;

        // Precompute sqrt(n*(n+1))/rsphere scaling factors
        // Conversion factor sqrt(n*(n+1))/rsphere of twodtooned_vrtdiv.f (written there with the 1-based degree as sqrt(n*(n-1))).
        let mut n_factors = vec![T::ZERO; ntrunc + 1];
        for n in 1..=ntrunc {
            let nf = n as f64;
            n_factors[n] = T::from_real((nf * (nf + 1.0)).sqrt() * inv_r);
        }

        let per_field_bytes =
            (4 * nlat * nlon + 8 * mmax * imid * 2 + 4 * nlat * nfreq) * T::BYTES;
        let max_tile = compute_tile_size(per_field_bytes, nt);

        let fft_pair = self.fft::<T>().get_or_plan(nlon);
        let r2c = &fft_pair.r2c;
        let scratch_len = r2c.get_scratch_len();

        let u_contig = ugrid.as_slice();
        let v_contig = vgrid.as_slice();

        // Store FFT outputs for ve, vo, we, wo in (tk, i, m)
        let mut ve_ki = vec![T::complex_zero(); max_tile * imid * mmax];
        let mut vo_ki = vec![T::complex_zero(); max_tile * imm1 * mmax];
        let mut we_ki = vec![T::complex_zero(); max_tile * imid * mmax];
        let mut wo_ki = vec![T::complex_zero(); max_tile * imm1 * mmax];

        // Number of GEMM RHS columns per block:
        // VrtDiv -> 4*tile (B_re, B_im, C_re, C_im)
        // DivOnly -> 2*tile (B_re, B_im)
        // VrtOnly -> 2*tile (C_re, C_im)
        let col_mult = match mode {
            VectorMode::VrtDiv => 4,
            VectorMode::DivOnly | VectorMode::VrtOnly => 2,
        };
        let mut rhs_zv_even = vec![T::ZERO; mmax * imm1 * col_mult * max_tile];
        let mut rhs_zw_even = vec![T::ZERO; mmax * imid * col_mult * max_tile];
        let mut rhs_zv_odd = vec![T::ZERO; mmax * imid * col_mult * max_tile];
        let mut rhs_zw_odd = vec![T::ZERO; mmax * imm1 * col_mult * max_tile];

        let max_rows = vt.v_even.rows_for_m(0, ntrunc).max(vt.v_odd.rows_for_m(0, ntrunc)).max(1);
        let mut spec_buf_m = vec![T::ZERO; mmax * max_rows * col_mult * max_tile];

        let mut start = 0usize;
        while start < nt {
            let tile = (nt - start).min(max_tile);
            let cols = col_mult * tile;
            let par_fields = tile >= 2 && tile * nlat * nlon >= PAR_WORK_THRESHOLD;
            let par_m = mmax * imid * cols >= PAR_WORK_THRESHOLD;

            // Fortran (vhaec1, ityp = 0), for i = 1,...,imm1 with tsn = 2/nlon, nlp1 = nlat+1:
            //   ve(i,j,k) = tsn*(v(i,j,k)+v(nlp1-i,j,k)),  vo(i,j,k) = tsn*(v(i,j,k)-v(nlp1-i,j,k))
            //   we(i,j,k) = tsn*(w(i,j,k)+w(nlp1-i,j,k)),  wo(i,j,k) = tsn*(w(i,j,k)-w(nlp1-i,j,k))
            // followed by hrfftf of ve and we (here all four fields are transformed). v is the colatitudinal component (-northward) and w the east component.
            // Step 1: fold (v = -v_geo, w = +u_geo) and run r2c
            let do_vec_fold_r2c = |(tk, (((ve_k, vo_k), we_k), wo_k)): (
                usize,
                (((&mut [Complex<T>], &mut [Complex<T>]), &mut [Complex<T>]), &mut [Complex<T>]),
            )| {
                let k = start + tk;
                let mut r_ve = vec![T::ZERO; nlon];
                let mut r_vo = vec![T::ZERO; nlon];
                let mut r_we = vec![T::ZERO; nlon];
                let mut r_wo = vec![T::ZERO; nlon];
                let mut cpx_buf = vec![T::complex_zero(); nfreq];
                let mut scratch = vec![T::complex_zero(); scratch_len];

                for i in 0..imm1 {
                    let r_n = if south_to_north { nlat - 1 - i } else { i };
                    let r_s = if south_to_north { i } else { nlat - 1 - i };
                    if let (Some(usl), Some(vsl)) = (u_contig, v_contig) {
                        let bu_n = (k * nlat + r_n) * nlon;
                        let bu_s = (k * nlat + r_s) * nlon;
                        let u_n = &usl[bu_n..bu_n + nlon];
                        let u_s = &usl[bu_s..bu_s + nlon];
                        let v_n = &vsl[bu_n..bu_n + nlon];
                        let v_s = &vsl[bu_s..bu_s + nlon];
                        for j in 0..nlon {
                            let vn = -v_n[j];
                            let vs = -v_s[j];
                            let wn = u_n[j];
                            let ws = u_s[j];
                            r_ve[j] = (vn + vs) * inv_nlon;
                            r_vo[j] = (vn - vs) * inv_nlon;
                            r_we[j] = (wn + ws) * inv_nlon;
                            r_wo[j] = (wn - ws) * inv_nlon;
                        }
                    } else {
                        for j in 0..nlon {
                            let vn = -vgrid[[k, r_n, j]];
                            let vs = -vgrid[[k, r_s, j]];
                            let wn = ugrid[[k, r_n, j]];
                            let ws = ugrid[[k, r_s, j]];
                            r_ve[j] = (vn + vs) * inv_nlon;
                            r_vo[j] = (vn - vs) * inv_nlon;
                            r_we[j] = (wn + ws) * inv_nlon;
                            r_wo[j] = (wn - ws) * inv_nlon;
                        }
                    }
                    r2c.process_with_scratch(&mut r_ve, &mut cpx_buf, &mut scratch)
                        .expect("r2c ve failed");
                    ve_k[i * mmax..(i + 1) * mmax].copy_from_slice(&cpx_buf[..mmax]);

                    r2c.process_with_scratch(&mut r_vo, &mut cpx_buf, &mut scratch)
                        .expect("r2c vo failed");
                    vo_k[i * mmax..(i + 1) * mmax].copy_from_slice(&cpx_buf[..mmax]);

                    r2c.process_with_scratch(&mut r_we, &mut cpx_buf, &mut scratch)
                        .expect("r2c we failed");
                    we_k[i * mmax..(i + 1) * mmax].copy_from_slice(&cpx_buf[..mmax]);

                    r2c.process_with_scratch(&mut r_wo, &mut cpx_buf, &mut scratch)
                        .expect("r2c wo failed");
                    wo_k[i * mmax..(i + 1) * mmax].copy_from_slice(&cpx_buf[..mmax]);
                }

                // Odd nlat: equator row, Fortran ve(imid,j,k) = tsn*v(imid,j,k) and we(imid,j,k) = tsn*w(imid,j,k) (loop 6); no antisymmetric part.
                if is_odd_nlat {
                    let i = imid - 1;
                    if let (Some(usl), Some(vsl)) = (u_contig, v_contig) {
                        let b = (k * nlat + i) * nlon;
                        let u_eq = &usl[b..b + nlon];
                        let v_eq = &vsl[b..b + nlon];
                        for j in 0..nlon {
                            r_ve[j] = -v_eq[j] * inv_nlon;
                            r_we[j] = u_eq[j] * inv_nlon;
                        }
                    } else {
                        for j in 0..nlon {
                            r_ve[j] = -vgrid[[k, i, j]] * inv_nlon;
                            r_we[j] = ugrid[[k, i, j]] * inv_nlon;
                        }
                    }
                    r2c.process_with_scratch(&mut r_ve, &mut cpx_buf, &mut scratch)
                        .expect("r2c ve eq failed");
                    ve_k[i * mmax..(i + 1) * mmax].copy_from_slice(&cpx_buf[..mmax]);

                    r2c.process_with_scratch(&mut r_we, &mut cpx_buf, &mut scratch)
                        .expect("r2c we eq failed");
                    we_k[i * mmax..(i + 1) * mmax].copy_from_slice(&cpx_buf[..mmax]);
                }
            };
            if par_fields {
                ve_ki[..tile * imid * mmax]
                    .par_chunks_mut(imid * mmax)
                    .zip(vo_ki[..tile * imm1 * mmax].par_chunks_mut(imm1 * mmax))
                    .zip(we_ki[..tile * imid * mmax].par_chunks_mut(imid * mmax))
                    .zip(wo_ki[..tile * imm1 * mmax].par_chunks_mut(imm1 * mmax))
                    .enumerate()
                    .for_each(do_vec_fold_r2c);
            } else {
                ve_ki[..tile * imid * mmax]
                    .chunks_mut(imid * mmax)
                    .zip(vo_ki[..tile * imm1 * mmax].chunks_mut(imm1 * mmax))
                    .zip(we_ki[..tile * imid * mmax].chunks_mut(imid * mmax))
                    .zip(wo_ki[..tile * imm1 * mmax].chunks_mut(imm1 * mmax))
                    .enumerate()
                    .for_each(do_vec_fold_r2c);
            }

            let imm1_stride = imm1 * cols;
            let imid_stride = imid * cols;

            // Step 2 layout: for each m the right-hand side has col_mult columns per field (VrtDiv: B_re, B_im, C_re, C_im; DivOnly: B_re, B_im; VrtOnly: C_re, C_im).
            // Fortran (vhaec1, case ityp = 0), with (x.re, x.im) = (x(i,2*mp1-2), x(i,2*mp1-1)) the cos/sin Fourier parts:
            //   n - m even (loop 23, i <= imm1):  br += zv*vo(2mp1-2) + zw*we(2mp1-1);  bi += zv*vo(2mp1-1) - zw*we(2mp1-2)
            //                                      cr += -zv*wo(2mp1-2) + zw*ve(2mp1-1);  ci += -zv*wo(2mp1-1) - zw*ve(2mp1-2)
            //   n - m odd (loop 21, i <= imm1):   br += zv*ve(2mp1-2) + zw*wo(2mp1-1);  bi += zv*ve(2mp1-1) - zw*wo(2mp1-2)
            //                                      cr += -zv*we(2mp1-2) + zw*vo(2mp1-1);  ci += -zv*we(2mp1-1) - zw*vo(2mp1-2)
            // with zv, zw the vector Z tables (zvin, zwin). The packing below arranges these combinations as matrix columns.
            // Step 2: Pack RHS matrices for each m
            let do_pack_rhs_m = |(m, (((zve_m, zwe_m), zvo_m), zwo_m)): (
                usize,
                (((&mut [T], &mut [T]), &mut [T]), &mut [T]),
            )| {
                for i in 0..imm1 {
                    let r_zve = &mut zve_m[i * cols..(i + 1) * cols];
                    let r_zwo = &mut zwo_m[i * cols..(i + 1) * cols];
                    for tk in 0..tile {
                        let vo = vo_ki[(tk * imm1 + i) * mmax + m];
                        let wo = wo_ki[(tk * imm1 + i) * mmax + m];
                        match mode {
                            VectorMode::VrtDiv => {
                                r_zve[4 * tk] = vo.re;
                                r_zve[4 * tk + 1] = vo.im;
                                r_zve[4 * tk + 2] = -wo.re;
                                r_zve[4 * tk + 3] = -wo.im;
                                r_zwo[4 * tk] = wo.im;
                                r_zwo[4 * tk + 1] = -wo.re;
                                r_zwo[4 * tk + 2] = vo.im;
                                r_zwo[4 * tk + 3] = -vo.re;
                            }
                            VectorMode::DivOnly => {
                                r_zve[2 * tk] = vo.re;
                                r_zve[2 * tk + 1] = vo.im;
                                r_zwo[2 * tk] = wo.im;
                                r_zwo[2 * tk + 1] = -wo.re;
                            }
                            VectorMode::VrtOnly => {
                                r_zve[2 * tk] = -wo.re;
                                r_zve[2 * tk + 1] = -wo.im;
                                r_zwo[2 * tk] = vo.im;
                                r_zwo[2 * tk + 1] = -vo.re;
                            }
                        }
                    }
                }
                for i in 0..imid {
                    let r_zwe = &mut zwe_m[i * cols..(i + 1) * cols];
                    let r_zvo = &mut zvo_m[i * cols..(i + 1) * cols];
                    for tk in 0..tile {
                        let ve = ve_ki[(tk * imid + i) * mmax + m];
                        let we = we_ki[(tk * imid + i) * mmax + m];
                        match mode {
                            VectorMode::VrtDiv => {
                                r_zwe[4 * tk] = we.im;
                                r_zwe[4 * tk + 1] = -we.re;
                                r_zwe[4 * tk + 2] = ve.im;
                                r_zwe[4 * tk + 3] = -ve.re;
                                r_zvo[4 * tk] = ve.re;
                                r_zvo[4 * tk + 1] = ve.im;
                                r_zvo[4 * tk + 2] = -we.re;
                                r_zvo[4 * tk + 3] = -we.im;
                            }
                            VectorMode::DivOnly => {
                                r_zwe[2 * tk] = we.im;
                                r_zwe[2 * tk + 1] = -we.re;
                                r_zvo[2 * tk] = ve.re;
                                r_zvo[2 * tk + 1] = ve.im;
                            }
                            VectorMode::VrtOnly => {
                                r_zwe[2 * tk] = ve.im;
                                r_zwe[2 * tk + 1] = -ve.re;
                                r_zvo[2 * tk] = -we.re;
                                r_zvo[2 * tk + 1] = -we.im;
                            }
                        }
                    }
                }
            };
            if par_m {
                rhs_zv_even[..mmax * imm1_stride]
                    .par_chunks_mut(imm1_stride)
                    .zip(rhs_zw_even[..mmax * imid_stride].par_chunks_mut(imid_stride))
                    .zip(rhs_zv_odd[..mmax * imid_stride].par_chunks_mut(imid_stride))
                    .zip(rhs_zw_odd[..mmax * imm1_stride].par_chunks_mut(imm1_stride))
                    .enumerate()
                    .for_each(do_pack_rhs_m);
            } else {
                rhs_zv_even[..mmax * imm1_stride]
                    .chunks_mut(imm1_stride)
                    .zip(rhs_zw_even[..mmax * imid_stride].chunks_mut(imid_stride))
                    .zip(rhs_zv_odd[..mmax * imid_stride].chunks_mut(imid_stride))
                    .zip(rhs_zw_odd[..mmax * imm1_stride].chunks_mut(imm1_stride))
                    .enumerate()
                    .for_each(do_pack_rhs_m);
            }

            // Step 3a, n - m even harmonics: zv_even (rows i < imm1) acts on vo/wo, and for m > 0 zw_even (rows i < imid, including the equator for odd nlat)
            // acts on the rotated we/ve terms. For m = 0 only the zv term exists since wbar(0,n,theta) = 0 (wbar carries a factor m).
            // Step 3a: n - m EVEN GEMMs
            let spec_m_stride = max_rows * cols;
            let do_vec_even = |(m, spec_m): (usize, &mut [T])| {
                let ne = vt.zv_even.rows_for_m(m, ntrunc);
                if ne == 0 {
                    return;
                }
                let zv_blk = vt.zv_even.block_for_m(m, ntrunc);
                let zve_m = &rhs_zv_even[m * imm1_stride..(m + 1) * imm1_stride];
                T::faer_gemm(
                    spec_m,
                    ne,
                    cols,
                    Accum::Replace,
                    zv_blk,
                    ne,
                    imm1,
                    true,
                    zve_m,
                    imm1,
                    cols,
                );
                if m > 0 && imid > 0 {
                    let zw_blk = vt.zw_even.block_for_m(m, ntrunc);
                    let zwe_m = &rhs_zw_even[m * imid_stride..(m + 1) * imid_stride];
                    T::faer_gemm(
                        spec_m,
                        ne,
                        cols,
                        Accum::Add,
                        zw_blk,
                        ne,
                        imid,
                        true,
                        zwe_m,
                        imid,
                        cols,
                    );
                }
            };
            if par_m {
                spec_buf_m[..mmax * spec_m_stride]
                    .par_chunks_mut(spec_m_stride)
                    .enumerate()
                    .for_each(do_vec_even);
            } else {
                spec_buf_m[..mmax * spec_m_stride]
                    .chunks_mut(spec_m_stride)
                    .enumerate()
                    .for_each(do_vec_even);
            }

            self.scatter_vrtdiv_parity(
                &spec_buf_m,
                spec_m_stride,
                cols,
                mmax,
                ntrunc,
                nmdim,
                start,
                tile,
                true,
                vt,
                &n_factors,
                mode,
                out_vrt.as_deref_mut(),
                out_div.as_deref_mut(),
            );

            // Step 3b, n - m odd harmonics: zv_odd (rows i < imid) acts on ve/we, and for m > 0 zw_odd (rows i < imm1) on the rotated wo/vo terms (zv/zw tables of vhaec, np1 = mp2,...,ndo2).
            // Step 3b: n - m ODD GEMMs
            let do_vec_odd = |(m, spec_m): (usize, &mut [T])| {
                let no = vt.zv_odd.rows_for_m(m, ntrunc);
                if no == 0 {
                    return;
                }
                let zv_blk = vt.zv_odd.block_for_m(m, ntrunc);
                let zvo_m = &rhs_zv_odd[m * imid_stride..(m + 1) * imid_stride];
                T::faer_gemm(
                    spec_m,
                    no,
                    cols,
                    Accum::Replace,
                    zv_blk,
                    no,
                    imid,
                    true,
                    zvo_m,
                    imid,
                    cols,
                );
                if m > 0 && imm1 > 0 {
                    let zw_blk = vt.zw_odd.block_for_m(m, ntrunc);
                    let zwo_m = &rhs_zw_odd[m * imm1_stride..(m + 1) * imm1_stride];
                    T::faer_gemm(
                        spec_m,
                        no,
                        cols,
                        Accum::Add,
                        zw_blk,
                        no,
                        imm1,
                        true,
                        zwo_m,
                        imm1,
                        cols,
                    );
                }
            };
            if par_m {
                spec_buf_m[..mmax * spec_m_stride]
                    .par_chunks_mut(spec_m_stride)
                    .enumerate()
                    .for_each(do_vec_odd);
            } else {
                spec_buf_m[..mmax * spec_m_stride]
                    .chunks_mut(spec_m_stride)
                    .enumerate()
                    .for_each(do_vec_odd);
            }

            self.scatter_vrtdiv_parity(
                &spec_buf_m,
                spec_m_stride,
                cols,
                mmax,
                ntrunc,
                nmdim,
                start,
                tile,
                false,
                vt,
                &n_factors,
                mode,
                out_vrt.as_deref_mut(),
                out_div.as_deref_mut(),
            );

            start += tile;
        }
    }

    /// Scatter the per-`m` GEMM results of one parity into the packed vorticity/divergence spectra.
    ///
    /// `is_even` selects the `n - m` even (`true`) or odd (`false`) harmonics, with degree
    /// `n = m + r_offset + 2*r` for row `r` (`r_offset` 0 or 1). `spec_buf_m` holds the product
    /// for each `m` with row stride `spec_m_stride` and `cols` columns per row (4 per field for
    /// `VrtDiv`, else 2); `start` and `tile` select the fields written (`nmdim` entries per field).
    /// The results are scaled by `n_factors[n] = sqrt(n(n+1))/rsphere`, with divergence negated.
    ///
    /// Fortran: twodtooned_vrtdiv.f / twodtooned_vrtdiv, i.e.
    /// `divspec = -(sqrt(n(n-1))/rsphere)*scale*cmplx(br,bi)` and
    /// `vrtspec = (sqrt(n(n-1))/rsphere)*scale*cmplx(cr,ci)` with 1-based `n` and `scale = 0.5`
    ///
    /// Difference from Fortran: the conversion is fused with the packing of the triangular array and
    /// operates on the GEMM output directly; entries with `n == 0` or (`n == nlat-1` and even `m`)
    /// are skipped (left zero), the `ndo1`/`ndo2` rule of the Fortran vector routines.
    #[allow(clippy::too_many_arguments)]
    fn scatter_vrtdiv_parity<T: ShtFloat>(
        &self,
        spec_buf_m: &[T],
        spec_m_stride: usize,
        cols: usize,
        mmax: usize,
        ntrunc: usize,
        nmdim: usize,
        start: usize,
        tile: usize,
        is_even: bool,
        vt: &VectorTables<T>,
        n_factors: &[T],
        mode: VectorMode,
        mut out_vrt: Option<&mut [Complex<T>]>,
        mut out_div: Option<&mut [Complex<T>]>,
    ) {
        let mut r_offset = if is_even { 0 } else { 1 };
        let _ = r_offset;
        r_offset = if is_even { 0 } else { 1 };
        // Loop over zonal wavenumbers m (mp1 = m+1 in Fortran).
        for m in 0..mmax {
            let nrows = if is_even {
                vt.zv_even.rows_for_m(m, ntrunc)
            } else {
                vt.zv_odd.rows_for_m(m, ntrunc)
            };
            if nrows == 0 {
                continue;
            }
            let m_base = spectral_offset(m, ntrunc);
            let spec_m = &spec_buf_m[m * spec_m_stride..];
            for r in 0..nrows {
                let n = m + r_offset + 2 * r;
                // Skip n = 0 (no vector harmonic) and n = nlat-1 with even m: br, bi, cr, ci(m+1,nlat) are assumed zero for even m.
                if n == 0 || (n == self.nlat - 1 && m % 2 == 0) {
                    continue;
                }
                // Packed triangular index of (m, n): spectral_offset(m, ntrunc) + (n - m), the nmstrt + n - m + 1 of twodtooned_vrtdiv.f (0-based).
                let nm = m_base + (n - m);
                let nf = n_factors[n];
                let src_row = &spec_m[r * cols..(r + 1) * cols];
                match mode {
                    VectorMode::VrtDiv => {
                        if let (Some(vrt), Some(div)) = (out_vrt.as_deref_mut(), out_div.as_deref_mut()) {
                            for tk in 0..tile {
                                let out_idx = (start + tk) * nmdim + nm;
                                // divspec = -sqrt(n(n+1))/rsphere * (B_re, B_im);  vrtspec = +sqrt(n(n+1))/rsphere * (C_re, C_im)  (twodtooned_vrtdiv.f).
                                div[out_idx] = Complex::new(-nf * src_row[4 * tk], -nf * src_row[4 * tk + 1]);
                                vrt[out_idx] = Complex::new(nf * src_row[4 * tk + 2], nf * src_row[4 * tk + 3]);
                            }
                        }
                    }
                    VectorMode::DivOnly => {
                        if let Some(div) = out_div.as_deref_mut() {
                            for tk in 0..tile {
                                let out_idx = (start + tk) * nmdim + nm;
                                div[out_idx] = Complex::new(-nf * src_row[2 * tk], -nf * src_row[2 * tk + 1]);
                            }
                        }
                    }
                    VectorMode::VrtOnly => {
                        if let Some(vrt) = out_vrt.as_deref_mut() {
                            for tk in 0..tile {
                                let out_idx = (start + tk) * nmdim + nm;
                                vrt[out_idx] = Complex::new(nf * src_row[2 * tk], nf * src_row[2 * tk + 1]);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Vector spherical harmonic synthesis of vorticity/divergence spectra into `(u, v)` grids.
    ///
    /// Inverse of `getvrtdivspec`. The spectra are first converted to the `br, bi, cr, ci`
    /// coefficients (`onedtotwod_vrtdiv.f`, 1-based `n` there, `scale = 0.5`)
    ///
    /// ```text
    /// br + i*bi = -(rsphere / sqrt(n(n+1))) * divspec / 0.5
    /// cr + i*ci =  (rsphere / sqrt(n(n+1))) * vrtspec / 0.5
    /// ```
    ///
    /// (this implementation does not apply the division by `scale = 0.5` explicitly; it is implicit in
    /// the inverse real FFT, which doubles the `m > 0` terms) and then synthesized with the `vhsec`
    /// expansion, using `vbar`, `wbar` as defined in the documentation of `getvrtdivspec`:
    ///
    /// ```text
    /// v_sp(i,j) = sum over n = 1..nlat-1 of .5*br(1,n+1) vbar(0,n,theta_i)
    ///           + sum over m = 1..mmax-1, n = m..nlat-1 of
    ///               (br vbar - ci wbar) cos(m phi_j) - (bi vbar + cr wbar) sin(m phi_j)
    /// w(i,j)    = sum over n = 1..nlat-1 of -.5*cr(1,n+1) vbar(0,n,theta_i)
    ///           + sum over m = 1..mmax-1, n = m..nlat-1 of
    ///               -(cr vbar + bi wbar) cos(m phi_j) + (ci vbar - br wbar) sin(m phi_j)
    /// ```
    ///
    /// where `v_sp` is the colatitudinal (southward) and `w` the east longitudinal component. The
    /// returned components are the geographic ones, `u = w` (eastward) and `v = -v_sp` (northward).
    ///
    /// # Arguments
    ///
    /// - `vrtspec`: optional vorticity spectra, shape `(nt, nmdim)`, packed `(m, n)` layout
    ///   (`nmdim = (ntrunc+1)*(ntrunc+2)/2`); used when `!is_grad`.
    /// - `divspec`: optional divergence spectra, same shape. When `is_grad` is true it is used as the
    ///   scalar spectrum whose gradient is synthesized (see below).
    /// - `nt`: number of fields (`k = 1, ..., nt`).
    /// - `ntrunc`: triangular truncation wavenumber.
    /// - `out_u`, `out_v`: output slices of length at least `nt * nlat * nlon`, C-order
    ///   `(nt, nlat, nlon)`; eastward and northward components.
    /// - `south_to_north`: `true` if grid row 0 is the southernmost latitude, `false` if row 0 is the
    ///   northernmost.
    /// - `is_grad`: if true, only the `B` coefficients are synthesized, with factor
    ///   `+sqrt(n(n+1))/rsphere` applied to the `divspec` array instead of
    ///   `-rsphere/sqrt(n(n+1))`, and the `C` coefficients are zero; this yields the gradient of a
    ///   scalar field (an irrotational vector field). This mode has no direct Fortran counterpart.
    ///
    /// # Edge cases
    ///
    /// Returns immediately for `nt == 0`. A missing spectrum (`None`) contributes zero. Coefficients
    /// with `n = 0`, or `n = nlat-1` with even `m`, are treated as zero (`br, bi, cr, ci(m+1,nlat)` are
    /// assumed zero for even `m`). Zonal wavenumbers `m >= mmax` are not synthesized. For `m = 0`
    /// the imaginary parts of the spectra are ignored. No `ierror` codes.
    ///
    /// Fortran: vhsec.f / vhsec (also vhses.f / vhses; Gaussian grids vhsgc.f / vhsgc and
    /// vhsgs.f / vhsgs), with `onedtotwod_vrtdiv.f` / `onedtotwod_vrtdiv`
    ///
    /// Difference from Fortran: only the full-sphere case (`ityp = 0, 1, 2`) is implemented; spectra
    /// are packed complex arrays rather than `br, bi, cr, ci` of shape `(mdab, ndab, nt)`; real inverse
    /// FFTs, `faer` matrix products, batch tiling and `rayon` parallelism replace `hrfftb` and the
    /// explicit loops; output follows the geographic sign convention.
    pub fn getuv<T: TypedTables>(
        &self,
        vrtspec: Option<&ndarray::ArrayView2<'_, Complex<T>>>,
        divspec: Option<&ndarray::ArrayView2<'_, Complex<T>>>,
        nt: usize,
        ntrunc: usize,
        out_u: &mut [T],
        out_v: &mut [T],
        south_to_north: bool,
        is_grad: bool,
    ) {
        if nt == 0 {
            return;
        }
        let tables = self.tables::<T>();
        let vt: &VectorTables<T> = tables.ensure_vector();
        let imid = tables.imid;
        let imm1 = tables.imm1;
        let mmax = tables.mmax_vector.min(ntrunc + 1);
        let nlat = self.nlat;
        let nlon = self.nlon;
        let nfreq = nlon / 2 + 1;
        let is_odd_nlat = nlat % 2 == 1;

        // Precompute spectral conversion factors:
        // getuv: B = -rsphere / sqrt(n*(n+1)) * divspec, C = +rsphere / sqrt(n*(n+1)) * vrtspec
        // getgrad: divspec = -n*(n+1)/rsphere^2 * phi => B = +sqrt(n*(n+1)) / rsphere * phi, C = 0
        let mut n_factors = vec![T::ZERO; ntrunc + 1];
        for n in 1..=ntrunc {
            let nf = n as f64;
            let val = if is_grad {
                (nf * (nf + 1.0)).sqrt() / self.rsphere
            } else {
                self.rsphere / (nf * (nf + 1.0)).sqrt()
            };
            n_factors[n] = T::from_real(val);
        }

        let per_field_bytes =
            (4 * nlat * nlon + 8 * mmax * imid * 2 + 4 * nlat * nfreq) * T::BYTES;
        let max_tile = compute_tile_size(per_field_bytes, nt);

        let fft_pair = self.fft::<T>().get_or_plan(nlon);
        let c2r = &fft_pair.c2r;
        let scratch_len = c2r.get_scratch_len();

        // Vorticity coefficients are ignored for the gradient; divergence (B) coefficients are only used when a spectrum was supplied.
        let has_vrt = vrtspec.is_some() && !is_grad;
        let has_div = divspec.is_some();

        // Output of GEMM for each m:
        // ve, we: (imid, 2*tile) each; vo, wo: (imm1, 2*tile) each
        let mut ve_m_buf = vec![T::ZERO; mmax * imid * 2 * max_tile];
        let mut we_m_buf = vec![T::ZERO; mmax * imid * 2 * max_tile];
        let mut vo_m_buf = vec![T::ZERO; mmax * imm1 * 2 * max_tile];
        let mut wo_m_buf = vec![T::ZERO; mmax * imm1 * 2 * max_tile];

        let max_rows = vt.v_even.rows_for_m(0, ntrunc).max(vt.v_odd.rows_for_m(0, ntrunc)).max(1);
        let mut spec_b_buf = vec![T::ZERO; mmax * max_rows * 2 * max_tile];
        let mut spec_c_buf = vec![T::ZERO; mmax * max_rows * 2 * max_tile];
        let mut spec_rot_buf = vec![T::ZERO; mmax * max_rows * 2 * max_tile];

        let mut start = 0usize;
        while start < nt {
            let tile = (nt - start).min(max_tile);
            let cols = 2 * tile;
            let imid_stride = imid * cols;
            let imm1_stride = imm1 * cols;
            let spec_m_stride = max_rows * cols;
            let par_fields = tile >= 2 && tile * nlat * nlon >= PAR_WORK_THRESHOLD;
            let par_m = mmax * imid * cols >= PAR_WORK_THRESHOLD;

            // --- Part A: n - m EVEN (V_even -> vo, -C -> wo; W_even -> ve, we) ---
            let do_uv_even = |(m, ((((((ve_m, we_m), vo_m), wo_m), sb_m), sc_m), srot_m)): (
                usize,
                (
                    (((((&mut [T], &mut [T]), &mut [T]), &mut [T]), &mut [T]), &mut [T]),
                    &mut [T],
                ),
            )| {
                let ne = vt.v_even.rows_for_m(m, ntrunc);
                if ne == 0 {
                    ve_m.fill(T::ZERO);
                    we_m.fill(T::ZERO);
                    vo_m.fill(T::ZERO);
                    wo_m.fill(T::ZERO);
                    return;
                }
                let m_base = spectral_offset(m, ntrunc);
                for r in 0..ne {
                    let n = m + 2 * r;
                    let nm = m_base + 2 * r;
                    let nf = n_factors[n];
                    // Skip n = 0 and n = nlat-1 with even m (br, bi, cr, ci(m+1,nlat) are assumed zero for even m; ndo1/ndo2 rule of vhsec).
                    let skip = n == 0 || (n == nlat - 1 && m % 2 == 0);
                    for tk in 0..tile {
                        // B = (br, bi) from the divergence spectrum; m = 0 has no sine part so the imaginary part is zero.
                        let (bre, bim) = if skip || !has_div {
                            (T::ZERO, T::ZERO)
                        } else {
                            let d = divspec.unwrap()[[start + tk, nm]];
                            if is_grad {
                                (nf * d.re, if m == 0 { T::ZERO } else { nf * d.im })
                            } else {
                                (-nf * d.re, if m == 0 { T::ZERO } else { -nf * d.im })
                            }
                        };
                        // C = (cr, ci) from the vorticity spectrum.
                        let (cre, cim) = if skip || !has_vrt {
                            (T::ZERO, T::ZERO)
                        } else {
                            let v = vrtspec.unwrap()[[start + tk, nm]];
                            (nf * v.re, if m == 0 { T::ZERO } else { nf * v.im })
                        };
                        sb_m[r * cols + 2 * tk] = bre;
                        sb_m[r * cols + 2 * tk + 1] = bim;
                        sc_m[r * cols + 2 * tk] = -cre;
                        sc_m[r * cols + 2 * tk + 1] = -cim;
                    }
                }

                // Fortran (vhsec): the V table (vbar) maps B to the colatitudinal part of the n - m even harmonics (equatorially odd rows i < imm1),
                // vo = V_even * B and wo = -V_even * C (the sc_m buffer holds -C).
                let v_blk = vt.v_even.block_for_m(m, ntrunc);
                if imm1 > 0 {
                    if has_div {
                        T::faer_gemm(vo_m, imm1, cols, Accum::Replace, v_blk, imm1, ne, false, sb_m, ne, cols);
                    } else {
                        vo_m.fill(T::ZERO);
                    }
                    if has_vrt {
                        T::faer_gemm(wo_m, imm1, cols, Accum::Replace, v_blk, imm1, ne, false, sc_m, ne, cols);
                    } else {
                        wo_m.fill(T::ZERO);
                    }
                }

                // wbar(m,n,theta) carries a factor m, so the W-table terms only exist for m > 0. The 90-degree rotations of (re, im) below (multiplication by i) realize the
                // i*wbar components of bbar = (vbar, i*wbar) and cbar = (i*wbar, -vbar).
                if m > 0 && imid > 0 {
                    let w_blk = vt.w_even.block_for_m(m, ntrunc);
                    if has_vrt {
                        for r in 0..ne {
                            for tk in 0..tile {
                                let neg_cre = sc_m[r * cols + 2 * tk];
                                let neg_cim = sc_m[r * cols + 2 * tk + 1];
                                srot_m[r * cols + 2 * tk] = neg_cim;
                                srot_m[r * cols + 2 * tk + 1] = -neg_cre;
                            }
                        }
                        T::faer_gemm(ve_m, imid, cols, Accum::Replace, w_blk, imid, ne, false, srot_m, ne, cols);
                    } else {
                        ve_m.fill(T::ZERO);
                    }
                    if has_div {
                        for r in 0..ne {
                            for tk in 0..tile {
                                let bre = sb_m[r * cols + 2 * tk];
                                let bim = sb_m[r * cols + 2 * tk + 1];
                                srot_m[r * cols + 2 * tk] = -bim;
                                srot_m[r * cols + 2 * tk + 1] = bre;
                            }
                        }
                        T::faer_gemm(we_m, imid, cols, Accum::Replace, w_blk, imid, ne, false, srot_m, ne, cols);
                    } else {
                        we_m.fill(T::ZERO);
                    }
                } else {
                    ve_m.fill(T::ZERO);
                    we_m.fill(T::ZERO);
                }
            };
            if par_m {
                ve_m_buf[..mmax * imid_stride]
                    .par_chunks_mut(imid_stride)
                    .zip(we_m_buf[..mmax * imid_stride].par_chunks_mut(imid_stride))
                    .zip(vo_m_buf[..mmax * imm1_stride].par_chunks_mut(imm1_stride))
                    .zip(wo_m_buf[..mmax * imm1_stride].par_chunks_mut(imm1_stride))
                    .zip(spec_b_buf[..mmax * spec_m_stride].par_chunks_mut(spec_m_stride))
                    .zip(spec_c_buf[..mmax * spec_m_stride].par_chunks_mut(spec_m_stride))
                    .zip(spec_rot_buf[..mmax * spec_m_stride].par_chunks_mut(spec_m_stride))
                    .enumerate()
                    .for_each(do_uv_even);
            } else {
                ve_m_buf[..mmax * imid_stride]
                    .chunks_mut(imid_stride)
                    .zip(we_m_buf[..mmax * imid_stride].chunks_mut(imid_stride))
                    .zip(vo_m_buf[..mmax * imm1_stride].chunks_mut(imm1_stride))
                    .zip(wo_m_buf[..mmax * imm1_stride].chunks_mut(imm1_stride))
                    .zip(spec_b_buf[..mmax * spec_m_stride].chunks_mut(spec_m_stride))
                    .zip(spec_c_buf[..mmax * spec_m_stride].chunks_mut(spec_m_stride))
                    .zip(spec_rot_buf[..mmax * spec_m_stride].chunks_mut(spec_m_stride))
                    .enumerate()
                    .for_each(do_uv_even);
            }

            // --- Part B: n - m ODD (V_odd -> ve, we [Accum::Add]; W_odd -> vo, wo [Accum::Add]) ---
            let do_uv_odd = |(m, ((((((ve_m, we_m), vo_m), wo_m), sb_m), sc_m), srot_m)): (
                usize,
                (
                    (((((&mut [T], &mut [T]), &mut [T]), &mut [T]), &mut [T]), &mut [T]),
                    &mut [T],
                ),
            )| {
                let no = vt.v_odd.rows_for_m(m, ntrunc);
                if no == 0 {
                    return;
                }
                let m_base = spectral_offset(m, ntrunc);
                for r in 0..no {
                    // n = m + 1 + 2r (n - m odd): only n = nlat-1 with even m is excluded.
                    let n = m + 1 + 2 * r;
                    let nm = m_base + 1 + 2 * r;
                    let nf = n_factors[n];
                    let skip = n == nlat - 1 && m % 2 == 0;
                    for tk in 0..tile {
                        let (bre, bim) = if skip || !has_div {
                            (T::ZERO, T::ZERO)
                        } else {
                            let d = divspec.unwrap()[[start + tk, nm]];
                            if is_grad {
                                (nf * d.re, if m == 0 { T::ZERO } else { nf * d.im })
                            } else {
                                (-nf * d.re, if m == 0 { T::ZERO } else { -nf * d.im })
                            }
                        };
                        let (cre, cim) = if skip || !has_vrt {
                            (T::ZERO, T::ZERO)
                        } else {
                            let v = vrtspec.unwrap()[[start + tk, nm]];
                            (nf * v.re, if m == 0 { T::ZERO } else { nf * v.im })
                        };
                        sb_m[r * cols + 2 * tk] = bre;
                        sb_m[r * cols + 2 * tk + 1] = bim;
                        sc_m[r * cols + 2 * tk] = -cre;
                        sc_m[r * cols + 2 * tk + 1] = -cim;
                    }
                }

                // Odd parity: V_odd (rows i < imid, including the equator for odd nlat) accumulates into ve and we; W_odd (rows i < imm1) into vo and wo.
                let v_blk = vt.v_odd.block_for_m(m, ntrunc);
                if imid > 0 {
                    if has_div {
                        T::faer_gemm(ve_m, imid, cols, Accum::Add, v_blk, imid, no, false, sb_m, no, cols);
                    }
                    if has_vrt {
                        T::faer_gemm(we_m, imid, cols, Accum::Add, v_blk, imid, no, false, sc_m, no, cols);
                    }
                }

                if m > 0 && imm1 > 0 {
                    let w_blk = vt.w_odd.block_for_m(m, ntrunc);
                    if has_vrt {
                        for r in 0..no {
                            for tk in 0..tile {
                                let neg_cre = sc_m[r * cols + 2 * tk];
                                let neg_cim = sc_m[r * cols + 2 * tk + 1];
                                srot_m[r * cols + 2 * tk] = neg_cim;
                                srot_m[r * cols + 2 * tk + 1] = -neg_cre;
                            }
                        }
                        T::faer_gemm(vo_m, imm1, cols, Accum::Add, w_blk, imm1, no, false, srot_m, no, cols);
                    }
                    if has_div {
                        for r in 0..no {
                            for tk in 0..tile {
                                let bre = sb_m[r * cols + 2 * tk];
                                let bim = sb_m[r * cols + 2 * tk + 1];
                                srot_m[r * cols + 2 * tk] = -bim;
                                srot_m[r * cols + 2 * tk + 1] = bre;
                            }
                        }
                        T::faer_gemm(wo_m, imm1, cols, Accum::Add, w_blk, imm1, no, false, srot_m, no, cols);
                    }
                }
            };
            if par_m {
                ve_m_buf[..mmax * imid_stride]
                    .par_chunks_mut(imid_stride)
                    .zip(we_m_buf[..mmax * imid_stride].par_chunks_mut(imid_stride))
                    .zip(vo_m_buf[..mmax * imm1_stride].par_chunks_mut(imm1_stride))
                    .zip(wo_m_buf[..mmax * imm1_stride].par_chunks_mut(imm1_stride))
                    .zip(spec_b_buf[..mmax * spec_m_stride].par_chunks_mut(spec_m_stride))
                    .zip(spec_c_buf[..mmax * spec_m_stride].par_chunks_mut(spec_m_stride))
                    .zip(spec_rot_buf[..mmax * spec_m_stride].par_chunks_mut(spec_m_stride))
                    .enumerate()
                    .for_each(do_uv_odd);
            } else {
                ve_m_buf[..mmax * imid_stride]
                    .chunks_mut(imid_stride)
                    .zip(we_m_buf[..mmax * imid_stride].chunks_mut(imid_stride))
                    .zip(vo_m_buf[..mmax * imm1_stride].chunks_mut(imm1_stride))
                    .zip(wo_m_buf[..mmax * imm1_stride].chunks_mut(imm1_stride))
                    .zip(spec_b_buf[..mmax * spec_m_stride].chunks_mut(spec_m_stride))
                    .zip(spec_c_buf[..mmax * spec_m_stride].chunks_mut(spec_m_stride))
                    .zip(spec_rot_buf[..mmax * spec_m_stride].chunks_mut(spec_m_stride))
                    .enumerate()
                    .for_each(do_uv_odd);
            }

            // Step 3: inverse real FFT and unfold. Fortran recombines the symmetric/antisymmetric parts as ge + go (northern row) and ge - go (southern row).
            // Geographic components: u = w (east) and v = -v_sp, since SPHEREPACK's colatitudinal component points south.
            // Step 3: c2r + unfold (u_geo = w, v_geo = -v_theta)
            let u_tile = &mut out_u[start * nlat * nlon..(start + tile) * nlat * nlon];
            let v_tile = &mut out_v[start * nlat * nlon..(start + tile) * nlat * nlon];
            let do_uv_c2r = |(tk, (u_out, v_out)): (usize, (&mut [T], &mut [T]))| {
                let mut cpx_ve = vec![T::complex_zero(); nfreq];
                let mut cpx_vo = vec![T::complex_zero(); nfreq];
                let mut cpx_we = vec![T::complex_zero(); nfreq];
                let mut cpx_wo = vec![T::complex_zero(); nfreq];
                let mut r_ve = vec![T::ZERO; nlon];
                let mut r_vo = vec![T::ZERO; nlon];
                let mut r_we = vec![T::ZERO; nlon];
                let mut r_wo = vec![T::ZERO; nlon];
                let mut scratch = vec![T::complex_zero(); scratch_len];

                for i in 0..imm1 {
                    cpx_ve.fill(T::complex_zero());
                    cpx_vo.fill(T::complex_zero());
                    cpx_we.fill(T::complex_zero());
                    cpx_wo.fill(T::complex_zero());
                    for m in 0..mmax {
                        let i_imid = m * imid_stride + i * cols + 2 * tk;
                        let i_imm1 = m * imm1_stride + i * cols + 2 * tk;
                        cpx_ve[m] = Complex::new(ve_m_buf[i_imid], ve_m_buf[i_imid + 1]);
                        cpx_we[m] = Complex::new(we_m_buf[i_imid], we_m_buf[i_imid + 1]);
                        cpx_vo[m] = Complex::new(vo_m_buf[i_imm1], vo_m_buf[i_imm1 + 1]);
                        cpx_wo[m] = Complex::new(wo_m_buf[i_imm1], wo_m_buf[i_imm1 + 1]);
                    }
                    cpx_ve[0].im = T::ZERO;
                    cpx_vo[0].im = T::ZERO;
                    cpx_we[0].im = T::ZERO;
                    cpx_wo[0].im = T::ZERO;
                    if nlon % 2 == 0 {
                        cpx_ve[nlon / 2].im = T::ZERO;
                        cpx_vo[nlon / 2].im = T::ZERO;
                        cpx_we[nlon / 2].im = T::ZERO;
                        cpx_wo[nlon / 2].im = T::ZERO;
                    }
                    c2r.process_with_scratch(&mut cpx_ve, &mut r_ve, &mut scratch)
                        .expect("c2r ve failed");
                    c2r.process_with_scratch(&mut cpx_vo, &mut r_vo, &mut scratch)
                        .expect("c2r vo failed");
                    c2r.process_with_scratch(&mut cpx_we, &mut r_we, &mut scratch)
                        .expect("c2r we failed");
                    c2r.process_with_scratch(&mut cpx_wo, &mut r_wo, &mut scratch)
                        .expect("c2r wo failed");

                    let r_n = if south_to_north { nlat - 1 - i } else { i };
                    let r_s = if south_to_north { i } else { nlat - 1 - i };
                    let bn = r_n * nlon;
                    let bs = r_s * nlon;
                    for j in 0..nlon {
                        let ve_v = r_ve[j];
                        let vo_v = r_vo[j];
                        let we_v = r_we[j];
                        let wo_v = r_wo[j];
                        u_out[bn + j] = we_v + wo_v;
                        u_out[bs + j] = we_v - wo_v;
                        v_out[bn + j] = -(ve_v + vo_v);
                        v_out[bs + j] = -(ve_v - vo_v);
                    }
                }

                if is_odd_nlat {
                    let i = imid - 1;
                    cpx_ve.fill(T::complex_zero());
                    cpx_we.fill(T::complex_zero());
                    for m in 0..mmax {
                        let i_imid = m * imid_stride + i * cols + 2 * tk;
                        cpx_ve[m] = Complex::new(ve_m_buf[i_imid], ve_m_buf[i_imid + 1]);
                        cpx_we[m] = Complex::new(we_m_buf[i_imid], we_m_buf[i_imid + 1]);
                    }
                    cpx_ve[0].im = T::ZERO;
                    cpx_we[0].im = T::ZERO;
                    if nlon % 2 == 0 {
                        cpx_ve[nlon / 2].im = T::ZERO;
                        cpx_we[nlon / 2].im = T::ZERO;
                    }
                    c2r.process_with_scratch(&mut cpx_ve, &mut r_ve, &mut scratch)
                        .expect("c2r ve eq failed");
                    c2r.process_with_scratch(&mut cpx_we, &mut r_we, &mut scratch)
                        .expect("c2r we eq failed");
                    let beq = i * nlon;
                    for j in 0..nlon {
                        u_out[beq + j] = r_we[j];
                        v_out[beq + j] = -r_ve[j];
                    }
                }
            };
            if par_fields {
                u_tile
                    .par_chunks_mut(nlat * nlon)
                    .zip(v_tile.par_chunks_mut(nlat * nlon))
                    .enumerate()
                    .for_each(do_uv_c2r);
            } else {
                u_tile
                    .chunks_mut(nlat * nlon)
                    .zip(v_tile.chunks_mut(nlat * nlon))
                    .enumerate()
                    .for_each(do_uv_c2r);
            }

            start += tile;
        }
    }
}

/// Selects the precision-specific tables and FFT cache of a `FullGridEngine` for a float type.
///
/// No direct Fortran counterpart; in SPHEREPACK the single and double precision variants are
/// separate compiled routines with their own workspace arrays.
pub trait TypedTables: FaerGemmFloat {
    /// Return the lazily initialized parity-split Legendre tables for this precision.
    fn get_tables(engine: &FullGridEngine) -> &ParityHarmonicTables<Self>;
    /// Return the real-FFT plan cache for this precision.
    fn get_fft(engine: &FullGridEngine) -> &FftCache<Self>;
}

/// Single-precision tables (built on first use from `nlat`, `nlon` and `gridtype`) and FFT cache.
impl TypedTables for f32 {
    #[inline]
    fn get_tables(engine: &FullGridEngine) -> &ParityHarmonicTables<Self> {
        engine.tables_f32.get_or_init(|| {
            ParityHarmonicTables::<f32>::new(engine.nlat, engine.nlon, engine.gridtype)
                .expect("validated geometry")
        })
    }

    #[inline]
    fn get_fft(engine: &FullGridEngine) -> &FftCache<Self> {
        &engine.fft_f32
    }
}

/// Double-precision tables (built on first use from `nlat`, `nlon` and `gridtype`) and FFT cache.
impl TypedTables for f64 {
    #[inline]
    fn get_tables(engine: &FullGridEngine) -> &ParityHarmonicTables<Self> {
        engine.tables_f64.get_or_init(|| {
            ParityHarmonicTables::<f64>::new(engine.nlat, engine.nlon, engine.gridtype)
                .expect("validated geometry")
        })
    }

    #[inline]
    fn get_fft(engine: &FullGridEngine) -> &FftCache<Self> {
        &engine.fft_f64
    }
}
