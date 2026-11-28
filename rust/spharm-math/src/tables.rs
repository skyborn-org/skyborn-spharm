//! Equatorial-parity-split scalar and vector harmonic tables for regular and Gaussian grids.
//!
//! Derived from NCAR SPHEREPACK 3.2 (J. Adams and P. Swarztrauber), see LICENSE.spherepack.
//!
//! # Background (SPHEREPACK `shaes`/`shses`/`vhaes`/`vhses`/`shags`/`vhags`, `sphcom.f`)
//!
//! SPHEREPACK precomputes, in its initialisation routines, the colatitude-dependent
//! functions used by spherical harmonic analysis and synthesis: the normalised associated
//! Legendre functions $\bar P_n^m(\theta_i)$ (scalar synthesis), the vector functions
//! $\bar V_n^m$ and $\bar W_n^m$ (vector synthesis), and the analysis kernels $Z_n^m$,
//! $Z_{v,n}^m$, $Z_{w,n}^m$. Because the grid is symmetric about the equator, each function is
//! either even or odd about the equator, so only the northern hemisphere (`imid = (nlat+1)/2`
//! colatitudes) is stored. Even functions need `imid` values, odd functions need
//! `imm1 = nlat/2` values (the equator row is dropped when `nlat` is odd, since odd functions vanish there).
//!
//! Scalar functions $\bar P_n^m$ and $Z_n^m$ are even about the equator when `n - m` is even and odd
//! when `n - m` is odd. For the vector functions, $\bar V_n^m$ is odd when `n - m` is even (even when
//! `n - m` is odd), while $\bar W_n^m$ is even when `n - m` is even, as encoded in [`VectorTables`].
//!
//! Analysis kernels: on a Gaussian grid $Z = w_i \bar P$ (the Gaussian weights `wts(i)` of `gaqd`
//! multiply `pbar` in the Fortran analysis sums of `shags`); on a regular grid they
//! are the Clenshaw-Curtis quadrature functions of `regular_weights.rs`.
//!
//! Regular-grid `ndo` rules (`shaes.f`, `shses.f`, `vhaes.f`, `vhses.f`): the Fortran loops over `np1` stop at
//! `ndo` (scalar) or `ndo1`/`ndo2` (vector), which drops the degree `n = nlat - 1` for the parity
//! combinations in which its function would vanish on every grid node. The Rust tables
//! store zero rows for these entries (see [`ParityTable`] pre-zeroing).
//!
//! # Differences from Fortran
//!
//! - Tables are stored per `m` as parity-split row blocks (`ParityTable`), not as the Fortran
//!   `pmn(imid, ...)`/`zin(imid, nlat, 3)` work arrays packed in `wshaes`, `wvhaes`, ...
//! - No workspace/`wsave` arrays and no `lsave`/`lwork` sizes or `ierror` returns: construction
//!   returns `Result` and tables are initialised lazily and once (`OnceLock`).
//! - `f64` is used internally; stored values are converted to the table scalar type `T`.

use std::f64::consts::PI;
use std::sync::OnceLock;

use crate::gauss::gaussian_colats_wts;
use crate::regular_weights::{compute_regular_scalar_z, compute_regular_vector_zw};
use crate::SpharmFloat;

/// Grid type supported by `ParityHarmonicTables`.
///
/// Fortran: the `gridtype` choice made by the user between the `*es`/`*ec` (equally spaced,
/// poles included) and `*gs`/`*gc` (Gaussian) SPHEREPACK families.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridType {
    /// Equally spaced colatitudes including both poles, $\theta_i = (i-1)\pi/(nlat-1)$ (SPHEREPACK `shaes`, `shses`, `vhaes`, `vhses`).
    Regular,
    /// Gaussian colatitudes and weights from `gaqd` (SPHEREPACK `shags`, `shsgs`, `vhags`, `vhsgs`).
    Gaussian,
}

impl GridType {
    /// Parse `"regular"` or `"gaussian"` (case-insensitive).
    ///
    /// Errors: any other string yields `Err` with the message
    /// `gridtype must be 'regular' or 'gaussian', got '...'`.
    ///
    /// No direct Fortran counterpart (the Fortran API selects the grid by choosing the routine).
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "regular" => Ok(Self::Regular),
            "gaussian" => Ok(Self::Gaussian),
            _ => Err(format!("gridtype must be 'regular' or 'gaussian', got '{s}'")),
        }
    }
}

/// Parity-split basis table stored in `m`-major, row-major `(n_deg, stride)` blocks.
///
/// For each zonal wavenumber `m in 0..mmax`, the rows `row_offsets[m]..row_offsets[m + 1]`
/// contain the degrees of one parity (`n - m` even or `n - m` odd) in ascending `n` order,
/// with contiguous northern-hemisphere latitude index `i in 0..stride` (`stride = imid` for
/// equatorially even functions and `stride = imm1` for equatorially odd functions).
///
/// Row `r = row_offsets[m] + (n - m) / 2` holds the degree `n` of the given parity, i.e. `n = m`,
/// `m + 2`, ... for `is_even_nm` and `n = m + 1`, `m + 3`, ... otherwise, up to `n = nlat - 1`.
/// Entries not written by the builders (for example the SPHEREPACK `ndo` exclusions) stay zero.
///
/// Difference from Fortran: SPHEREPACK stores a full `(imid, nlat, 3)` slab per `m` (and a packed
/// triangular index `mn = m*(nlat-1) - m*(m-1)/2 + np1`); here each `m` is a dense,
/// parity-split, C-order block with 0-based `n`.
#[derive(Debug, Clone)]
pub struct ParityTable<T> {
    /// Flat storage of all blocks, `row_offsets[mmax] * stride` values.
    pub data: Vec<T>,
    /// `row_offsets[m]` is the first row of wavenumber `m`; length `mmax + 1` (last entry = total rows).
    pub row_offsets: Vec<usize>,
    /// Number of latitude values per row (`imid` for equatorially even rows, `imm1` for odd rows).
    pub stride: usize,
    /// Number of zonal wavenumbers `m = 0..mmax` stored.
    pub mmax: usize,
    /// Number of colatitudes of the full grid.
    pub nlat: usize,
    /// `true` if the rows hold `n - m` even, `false` for `n - m` odd.
    pub is_even_nm: bool,
}

impl<T> ParityTable<T> {
    /// Allocate a zero-filled table, counting rows per `m` (degrees `n = m..nlat-1` of one parity).
    ///
    /// For `is_even_nm` the count is `(n_max - m)/2 + 1` and otherwise `(n_max - m - 1)/2 + 1`
    /// (or 0 when `n_max <= m`), with `n_max = nlat - 1`.
    ///
    /// No direct Fortran counterpart (replaces the packed SPHEREPACK work-array layout).
    pub(crate) fn allocate_zero(
        nlat: usize,
        mmax: usize,
        stride: usize,
        is_even_nm: bool,
    ) -> Self
    where
        T: SpharmFloat,
    {
        let mut row_offsets = Vec::with_capacity(mmax + 1);
        let mut acc = 0usize;
        for m in 0..mmax {
            row_offsets.push(acc);
            let n_max = nlat - 1;
            let count = if is_even_nm {
                if n_max >= m {
                    (n_max - m) / 2 + 1
                } else {
                    0
                }
            } else if n_max > m {
                (n_max - m - 1) / 2 + 1
            } else {
                0
            };
            acc += count;
        }
        row_offsets.push(acc);
        Self {
            data: vec![T::ZERO; acc * stride],
            row_offsets,
            stride,
            mmax,
            nlat,
            is_even_nm,
        }
    }

    /// Number of degree rows for wavenumber `m` up to truncation `ntrunc`.
    ///
    /// Counts only degrees `n <= min(ntrunc, nlat - 1)` of the table's parity; returns 0 when
    /// `m >= mmax` or `ntrunc < m`. No direct Fortran counterpart (Fortran loops `np1 = mp1, ndo, 2`
    /// directly with `ndo` as the truncation).
    #[inline]
    pub fn rows_for_m(&self, m: usize, ntrunc: usize) -> usize {
        if m >= self.mmax || ntrunc < m {
            return 0;
        }
        let n_hi = ntrunc.min(self.nlat - 1);
        let full = self.row_offsets[m + 1] - self.row_offsets[m];
        let count = if self.is_even_nm {
            (n_hi - m) / 2 + 1
        } else if n_hi > m {
            (n_hi - m - 1) / 2 + 1
        } else {
            0
        };
        count.min(full)
    }

    /// Row-major `(rows_for_m(m, ntrunc), stride)` slice for wavenumber `m` truncated at `ntrunc`.
    ///
    /// Returns an empty slice if there are no rows or `stride == 0`. No direct Fortran counterpart.
    #[inline]
    pub fn block_for_m(&self, m: usize, ntrunc: usize) -> &[T] {
        let nrows = self.rows_for_m(m, ntrunc);
        if nrows == 0 || self.stride == 0 {
            return &[];
        }
        let start = self.row_offsets[m] * self.stride;
        &self.data[start..start + nrows * self.stride]
    }

    /// Store `values[..stride]` as the row for `(m, n)`, converting from `f64` to `T`.
    ///
    /// Requires `n - m` to match the table parity (debug-asserted). Local row is `(n - m)/2`.
    /// No direct Fortran counterpart.
    #[inline]
    pub(crate) fn set_row_from_f64(&mut self, m: usize, n: usize, values: &[f64])
    where
        T: SpharmFloat,
    {
        debug_assert!(m < self.mmax && n >= m && n < self.nlat);
        debug_assert_eq!((n - m) % 2 == 0, self.is_even_nm);
        if self.stride == 0 {
            return;
        }
        let local_row = (n - m) / 2;
        let row_idx = self.row_offsets[m] + local_row;
        let dst = &mut self.data[row_idx * self.stride..(row_idx + 1) * self.stride];
        for (d, &v) in dst.iter_mut().zip(values.iter().take(self.stride)) {
            *d = T::from_f64(v);
        }
    }
}

/// Parity-split scalar synthesis ($\bar{P}_n^m$) and analysis ($Z_n^m$) tables.
///
/// Analysis tables $Z_n^m$ come from Gaussian weights times $\bar P_n^m$ on Gaussian grids and
/// from the Clenshaw-Curtis functions on regular grids.
/// Fortran: shaes.f / shses.f / shags.f (scalar tables and `wshaes`, `wshses`, `wshags` packing).
#[derive(Debug, Clone)]
pub struct ScalarTables<T> {
    /// Synthesis $\bar{P}_n^m(\theta_i)$ for $n - m$ even (`stride = imid`).
    pub p_even: ParityTable<T>,
    /// Synthesis $\bar{P}_n^m(\theta_i)$ for $n - m$ odd (`stride = imm1`).
    pub p_odd: ParityTable<T>,
    /// Analysis quadrature kernel $Z_n^m(\theta_i)$ for $n - m$ even (`stride = imid`).
    pub z_even: ParityTable<T>,
    /// Analysis quadrature kernel $Z_n^m(\theta_i)$ for $n - m$ odd (`stride = imm1`).
    pub z_odd: ParityTable<T>,
}

/// Parity-split vector synthesis ($V_n^m, W_n^m$) and analysis ($Z_{v,n}^m, Z_{w,n}^m$) tables.
///
/// Tables are categorized by `(n - m)` parity:
/// - `v_even`, `zv_even` (`n - m` even): equatorially ODD (`stride = imm1`).
/// - `v_odd`, `zv_odd` (`n - m` odd): equatorially EVEN (`stride = imid`).
/// - `w_even`, `zw_even` (`n - m` even): equatorially EVEN (`stride = imid`).
/// - `w_odd`, `zw_odd` (`n - m` odd): equatorially ODD (`stride = imm1`).
///
/// SPHEREPACK `ndo1`/`ndo2` rule (`n = nlat - 1` with `m` even) and `n = 0` are pre-zeroed.
///
/// Fortran: the `ndo1`/`ndo2` limits in vhaes.f / vhses.f (also vhags.f): with
/// `mlat = nlat mod 2`, `ndo1 = nlat - 1` if `mlat != 0` (else `nlat`) and `ndo2 = nlat - 1` if
/// `mlat == 0` (else `nlat`). The `zv_*`/`zw_*` tables are the vector analysis kernels
/// (Clenshaw-Curtis on regular grids, Gaussian weights times $V$, $W$ on Gaussian grids).
#[derive(Debug, Clone)]
pub struct VectorTables<T> {
    /// $\bar V_n^m$ for $n - m$ even: equatorially odd (`stride = imm1`).
    pub v_even: ParityTable<T>,
    /// $\bar V_n^m$ for $n - m$ odd: equatorially even (`stride = imid`).
    pub v_odd: ParityTable<T>,
    /// $\bar W_n^m$ for $n - m$ even: equatorially even (`stride = imid`).
    pub w_even: ParityTable<T>,
    /// $\bar W_n^m$ for $n - m$ odd: equatorially odd (`stride = imm1`).
    pub w_odd: ParityTable<T>,
    /// Analysis kernel $Z_{v,n}^m$ for $n - m$ even (`stride = imm1`).
    pub zv_even: ParityTable<T>,
    /// Analysis kernel $Z_{v,n}^m$ for $n - m$ odd (`stride = imid`).
    pub zv_odd: ParityTable<T>,
    /// Analysis kernel $Z_{w,n}^m$ for $n - m$ even (`stride = imid`).
    pub zw_even: ParityTable<T>,
    /// Analysis kernel $Z_{w,n}^m$ for $n - m$ odd (`stride = imm1`).
    pub zw_odd: ParityTable<T>,
}

/// Compute northern-hemisphere $\bar{P}_n^m(\theta_i)$ in `f64` via `dnlfk`/`dnlft` ($m=0,1$)
/// and Schwarztrauber's recurrence ($m \ge 2$), calling `emit(n, m, &pbar_nh)` in ascending `n` order.
/// Keeps only 3 `n`-levels of size `(max_m + 1) * imid` in temporary memory.
///
/// Parameters:
/// - `nlat`: number of colatitudes (not needed by the recurrence, kept for symmetry with callers).
/// - `max_n`: largest degree streamed (`n = 0..=max_n`).
/// - `max_m_plus_1`: number of orders requested (`m = 0..min(max_m_plus_1, max_n + 1)`).
/// - `colats_nh`: northern-hemisphere colatitudes in radians (`imid` values).
/// - `emit(n, m_lim, cur, prev)`: `cur` holds $\bar P_n^m$ for `m = 0..=m_lim`, row-major
///   `(m, i)` with `i < imid`, `m_lim = min(n, m_cap - 1)`; `prev` holds $\bar P_{n-1}^m$ for
///   `m = 0..=min(n-1, m_cap-1)` (empty for `n = 0`). Slices are only valid during the call.
///
/// Recurrences (SPHEREPACK normalisation, $\bar P_0^0 = \sqrt{1/2}$): seeds at `n = 0, 1`,
/// then for `n >= 2` the three-term Bonnet recurrence in `n` for `m = 0, 1` and the
/// Schwarztrauber recurrence $\bar P_n^m = a\,\bar P_{n-2}^{m-2} + b\,\bar P_{n-2}^m - c\,\bar P_n^{m-2}$
/// for `m >= 2` (same coefficient structure as Fortran `abel`, `bbel`, `cbel`).
///
/// Fortran: sphcom.f / legin1 (Gaussian grids) and alin1 / rabcp1 (regular grids)
///
/// Difference from Fortran: Fortran advances in `m` for fixed `n` columns (`pmn(np1,i,km0..km2)`,
/// permuting `km0, km1, km2`); here the loop is over `n` with a 3-level ring over `n`, `n-1`,
/// `n-2`, and `m = 0, 1` also use recurrences in `n` instead of Fortran's separate `p0n`, `p1n` tables.
/// All arithmetic is `f64`.
pub(crate) fn stream_pbar_nh<F>(
    nlat: usize,
    max_n: usize,
    max_m_plus_1: usize,
    colats_nh: &[f64],
    mut emit: F,
) where
    F: FnMut(usize, usize, &[f64], &[f64]),
{
    // imid = number of northern-hemisphere colatitudes, from `0` (north pole) to the equator.
    let imid = colats_nh.len();
    let m_cap = max_m_plus_1.min(max_n + 1);
    let stride = m_cap * imid;
    // Ring buffer of three n-levels (n, n-1, n-2), each holding m_cap rows of imid values.
    let mut ring = vec![0.0_f64; 3 * stride];
    let mut sin_th = Vec::with_capacity(imid);
    let mut cos_th = Vec::with_capacity(imid);
    for &th in colats_nh {
        let (s, c) = th.sin_cos();
        sin_th.push(s);
        cos_th.push(c);
    }

    // n = 0
    // n = 0: Pbar_0^0 = sqrt(1/2) on every node (SPHEREPACK normalisation).
    let ssqr2 = 0.5_f64.sqrt();
    if m_cap >= 1 {
        ring[..imid].fill(ssqr2);
        emit(0, 0, &ring[..imid], &[]);
    }
    if max_n == 0 {
        return;
    }

    // n = 1
    {
        // n = 1: Pbar_1^0 = sqrt(3/2) cos(th) and Pbar_1^1 = sqrt(3/4) sin(th).
        let cur_off = stride;
        let m_lim = 1usize.min(m_cap - 1);
        let s32 = 1.5_f64.sqrt();
        let dst0 = &mut ring[cur_off..cur_off + imid];
        for i in 0..imid {
            dst0[i] = s32 * cos_th[i];
        }
        if m_lim >= 1 {
            let s34 = 0.75_f64.sqrt();
            let dst1 = &mut ring[cur_off + imid..cur_off + 2 * imid];
            for i in 0..imid {
                dst1[i] = s34 * sin_th[i];
            }
        }
        let cur_slice = &ring[cur_off..cur_off + (m_lim + 1) * imid];
        emit(1, m_lim, cur_slice, &ring[..imid]);
    }

    // n >= 2: level n is built from levels n-1 (nm1_off) and n-2 (nm2_off); ring offsets rotate modulo 3.
    for n in 2..=max_n {
        let nm2_off = ((n + 1) % 3) * stride;
        let nm1_off = ((n + 2) % 3) * stride;
        let cur_off = (n % 3) * stride;
        let m_lim = n.min(m_cap - 1);

        let nf = n as f64;
        let c_num = 2.0 * nf + 1.0;

        // m = 0 via 3-term Bonnet recurrence
        {
            // Pbar_n^0 = a0 cos(th) Pbar_{n-1}^0 - b0 Pbar_{n-2}^0, a0 = sqrt((2n+1)(2n-1))/n, b0 = ((n-1)/n) sqrt((2n+1)/(2n-3)).
            let a0 = (c_num * (2.0 * nf - 1.0)).sqrt() / nf;
            let b0 = ((nf - 1.0) / nf) * (c_num / (2.0 * nf - 3.0)).sqrt();
            for i in 0..imid {
                ring[cur_off + i] =
                    a0 * cos_th[i] * ring[nm1_off + i] - b0 * ring[nm2_off + i];
            }
        }

        // m = 1 via 3-term Bonnet recurrence
        if m_lim >= 1 {
            // Pbar_n^1 = a1 cos(th) Pbar_{n-1}^1 - b1 Pbar_{n-2}^1, a1 = sqrt((2n+1)(2n-1)/((n-1)(n+1))), b1 = sqrt((2n+1) n (n-2)/((2n-3)(n-1)(n+1))) (0 for n < 3).
            let denom1 = (nf - 1.0) * (nf + 1.0);
            let a1 = (c_num * (2.0 * nf - 1.0) / denom1).sqrt();
            let b1 = if n >= 3 {
                (c_num * nf * (nf - 2.0) / ((2.0 * nf - 3.0) * denom1)).sqrt()
            } else {
                0.0
            };
            for i in 0..imid {
                ring[cur_off + imid + i] = a1 * cos_th[i] * ring[nm1_off + imid + i]
                    - b1 * ring[nm2_off + imid + i];
            }
        }

        // m >= 2 via Schwarztrauber recurrence
        if m_lim >= 2 {
            // Pbar_n^m = abel Pbar_{n-2}^{m-2} + bbel Pbar_{n-2}^m - cbel Pbar_n^{m-2}, with c_num = 2n+1, c_den = 2n-3 and d1 = (m+n-1)(m+n).
            let c_den = 2.0 * nf - 3.0;
            for m in 2..=m_lim {
                let mf = m as f64;
                let d1 = (mf + nf - 1.0) * (mf + nf);
                let abel = (c_num * (mf + nf - 2.0) * (mf + nf - 3.0) / (c_den * d1)).sqrt();
                let cbel = (((nf - mf + 1.0) * (nf - mf + 2.0)) / d1).sqrt();

                // The bbel term needs Pbar_{n-2}^m, which exists only for m <= n-2 (m < n-1).
                if m < n - 1 {
                    let bbel = (c_num * (nf - mf - 1.0) * (nf - mf) / (c_den * d1)).sqrt();
                    for i in 0..imid {
                        ring[cur_off + m * imid + i] = abel * ring[nm2_off + (m - 2) * imid + i]
                            + bbel * ring[nm2_off + m * imid + i]
                            - cbel * ring[cur_off + (m - 2) * imid + i];
                    }
                } else {
                    for i in 0..imid {
                        ring[cur_off + m * imid + i] = abel * ring[nm2_off + (m - 2) * imid + i]
                            - cbel * ring[cur_off + (m - 2) * imid + i];
                    }
                }
            }
        }

        // `nlat` is not needed by the recurrence; this binding only silences the unused-parameter warning.
        let _ = nlat;
        let cur_slice = &ring[cur_off..cur_off + (m_lim + 1) * imid];
        let prev_m = (n - 1).min(m_cap - 1);
        let prev_slice = &ring[nm1_off..nm1_off + (prev_m + 1) * imid];
        emit(n, m_lim, cur_slice, prev_slice);
    }
}

/// Equatorial-parity-split spherical harmonic tables with lazy scalar/vector initialization.
///
/// Fortran: the combined initialisation of SPHEREPACK `shaesi`/`shsesi` (scalar), `vhaesi`/`vhsesi`
/// (vector) and `shagsi`/`vhagsi` (Gaussian), minus the FFT work arrays. Scalar and vector tables
/// are built independently on first use.
#[derive(Debug)]
pub struct ParityHarmonicTables<T: SpharmFloat> {
    /// Number of colatitudes (`nlat >= 3`).
    pub nlat: usize,
    /// Number of longitudes (`nlon >= 1`).
    pub nlon: usize,
    /// `(nlat + 1)/2`: number of northern-hemisphere colatitudes including the equator for odd `nlat`.
    pub imid: usize,
    /// `nlat / 2`: number of latitudes strictly north of the equator (rows of equatorially odd functions).
    pub imm1: usize,
    /// Scalar zonal wavenumber count, `min(nlat, nlon/2 + 1)` (Fortran `mmax` in shaes/shses).
    pub mmax_scalar: usize,
    /// Vector zonal wavenumber count, `min(nlat, (nlon + 1)/2)` (Fortran `mmax` in vhaes/vhses).
    pub mmax_vector: usize,
    /// Regular (equally spaced, poles included) or Gaussian grid.
    pub gridtype: GridType,
    /// Northern-hemisphere colatitudes in radians (`len = imid`).
    pub colats_nh: Vec<f64>,
    /// Northern-hemisphere Gaussian weights (`len = imid`) for Gaussian grids, empty for Regular.
    pub weights_nh: Vec<f64>,
    scalar: OnceLock<ScalarTables<T>>,
    vector: OnceLock<VectorTables<T>>,
}

impl<T: SpharmFloat> ParityHarmonicTables<T> {
    /// Construct harmonic table metadata and grid geometry (`gaqd` for Gaussian, uniform colatitudes for Regular).
    ///
    /// Errors: `nlat < 3` and `nlon < 1` return `Err` (the Fortran drivers report an invalid
    /// `nlat` as `ierror = 1` and an invalid `nlon` as `ierror = 2`).
    ///
    /// Regular grid: `colats_nh[i] = i * pi/(nlat-1)` for `i = 0..imid` (Fortran
    /// `dt = pi/(nlat-1)`, `th = (i-1)*dt`). Gaussian grid: the first `imid` colatitudes and weights
    /// of the `gaqd` nodes. `imid = (nlat+1)/2`, `imm1 = nlat/2`, `mmax_scalar = min(nlat, nlon/2+1)`,
    /// `mmax_vector = min(nlat, (nlon+1)/2)`.
    ///
    /// Fortran: shaes.f (`shaesi`), vhaes.f (`vhaesi`), gaqd.f (Gaussian nodes and weights)
    pub fn new(nlat: usize, nlon: usize, gridtype: GridType) -> Result<Self, String> {
        if nlat < 3 {
            return Err(format!("nlat must be >= 3, got {nlat}"));
        }
        if nlon < 1 {
            return Err(format!("nlon must be >= 1, got {nlon}"));
        }

        let imid = (nlat + 1) / 2;
        let imm1 = nlat / 2;
        let mmax_scalar = nlat.min(nlon / 2 + 1);
        let mmax_vector = nlat.min((nlon + 1) / 2);

        let (colats_nh, weights_nh) = match gridtype {
            GridType::Regular => {
                let dt = PI / ((nlat - 1) as f64);
                let colats = (0..imid).map(|i| (i as f64) * dt).collect();
                (colats, Vec::new())
            }
            GridType::Gaussian => {
                let (mut colats, mut wts) = gaussian_colats_wts(nlat)?;
                colats.truncate(imid);
                wts.truncate(imid);
                (colats, wts)
            }
        };

        Ok(Self {
            nlat,
            nlon,
            imid,
            imm1,
            mmax_scalar,
            mmax_vector,
            gridtype,
            colats_nh,
            weights_nh,
            scalar: OnceLock::new(),
            vector: OnceLock::new(),
        })
    }

    /// Return (lazily initializing if needed) the parity-split scalar tables.
    ///
    /// Difference from Fortran: tables are built once on first use (`OnceLock`), not eagerly in
    /// `shaesi`/`shsesi`/`shagsi`.
    pub fn ensure_scalar(&self) -> &ScalarTables<T> {
        self.scalar.get_or_init(|| self.build_scalar())
    }

    /// Return (lazily initializing if needed) the parity-split vector tables.
    ///
    /// Difference from Fortran: built once on first use (`OnceLock`), not eagerly in
    /// `vhaesi`/`vhsesi`/`vhagsi`.
    pub fn ensure_vector(&self) -> &VectorTables<T> {
        self.vector.get_or_init(|| self.build_vector())
    }

    /// Return already-initialized scalar tables if present.
    ///
    /// No direct Fortran counterpart (non-initialising accessor).
    pub fn scalar_if_init(&self) -> Option<&ScalarTables<T>> {
        self.scalar.get()
    }

    /// Return already-initialized vector tables if present.
    ///
    /// No direct Fortran counterpart (non-initialising accessor).
    pub fn vector_if_init(&self) -> Option<&VectorTables<T>> {
        self.vector.get()
    }

    /// Build all scalar tables $\bar P_n^m$ and $Z_n^m$ for `m < mmax_scalar`.
    ///
    /// $\bar P_n^m$ is streamed with `stream_pbar_nh` and split by `n - m` parity. Gaussian grids
    /// set $Z = w_i \bar P_n^m$ with the northern-hemisphere Gaussian weights; regular grids fill
    /// $Z$ from `compute_regular_scalar_z`. Rows excluded by the regular-grid `ndo` rule stay zero.
    ///
    /// Fortran: shaes.f / shaesi (via `sea1`, `ses1` in sphcom.f), shags.f / shagsi
    fn build_scalar(&self) -> ScalarTables<T> {
        let mut p_even = ParityTable::allocate_zero(self.nlat, self.mmax_scalar, self.imid, true);
        let mut p_odd = ParityTable::allocate_zero(self.nlat, self.mmax_scalar, self.imm1, false);
        let mut z_even = ParityTable::allocate_zero(self.nlat, self.mmax_scalar, self.imid, true);
        let mut z_odd = ParityTable::allocate_zero(self.nlat, self.mmax_scalar, self.imm1, false);

        let imid = self.imid;
        let is_gaussian = self.gridtype == GridType::Gaussian;
        let mut weighted_buf = vec![0.0_f64; imid];

        stream_pbar_nh(
            self.nlat,
            self.nlat - 1,
            self.mmax_scalar,
            &self.colats_nh,
            |n, m_lim, cur_p, _prev_p| {
                for m in 0..=m_lim {
                    // Parity split: n - m even rows go to the even tables (stride imid); n - m odd rows go to
                    // the odd tables (stride imm1, the equator row is dropped).
                    // SPHEREPACK regular-grid ndo rule (shaes/shses):
                    // at n == nlat - 1 with m odd, sin((nlat-1)*theta) vanishes on all nodes.
                    // Fortran shaes.f / shses.f: ndo = nlat for the first loop when nlat is odd (nlat - 1 when
                    // even), and the opposite for the second loop. Both loops skip np1 = nlat (n = nlat - 1) for odd m.
                    if !is_gaussian && n == self.nlat - 1 && m % 2 == 1 {
                        continue;
                    }
                    let row = &cur_p[m * imid..(m + 1) * imid];
                    if (n - m) % 2 == 0 {
                        p_even.set_row_from_f64(m, n, row);
                        // Gaussian analysis: Z = wts(i) * Pbar (weights multiply pbar in the Fortran analysis sums).
                        if is_gaussian {
                            for i in 0..imid {
                                weighted_buf[i] = row[i] * self.weights_nh[i];
                            }
                            z_even.set_row_from_f64(m, n, &weighted_buf);
                        }
                    } else {
                        p_odd.set_row_from_f64(m, n, row);
                        if is_gaussian {
                            for i in 0..self.imm1 {
                                weighted_buf[i] = row[i] * self.weights_nh[i];
                            }
                            z_odd.set_row_from_f64(m, n, &weighted_buf[..self.imm1]);
                        }
                    }
                }
            },
        );

        // Regular grid: Z comes from the Clenshaw-Curtis functions (Fortran zfin1/zfini1);
        // the same ndo rule (n = nlat - 1 with odd m) is skipped.
        if !is_gaussian {
            compute_regular_scalar_z(self.nlat, self.mmax_scalar, &self.colats_nh, |m, n, row| {
                if n == self.nlat - 1 && m % 2 == 1 {
                    return;
                }
                if (n - m) % 2 == 0 {
                    z_even.set_row_from_f64(m, n, row);
                } else {
                    z_odd.set_row_from_f64(m, n, row);
                }
            });
        }

        ScalarTables {
            p_even,
            p_odd,
            z_even,
            z_odd,
        }
    }

    /// Build all vector tables $\bar V_n^m$, $\bar W_n^m$, $Z_{v,n}^m$, $Z_{w,n}^m$ for `m < mmax_vector`.
    ///
    /// From the streamed $\bar P_n^m$ (in `stream_pbar_nh`, requested up to `mmax + 1` orders):
    /// - $\bar V_n^0 = -\bar P_n^1$;
    /// - $\bar V_n^m = [\sqrt{(n+m)(n-m+1)}\,\bar P_n^{m-1} - \sqrt{(n-m)(n+m+1)}\,\bar P_n^{m+1}] / \sqrt{4n(n+1)}$ for `1 <= m < n`;
    /// - $\bar V_n^n = \bar P_n^{n-1} / \sqrt{2(n+1)}$;
    /// - $\bar W_n^m = d\,[\sqrt{(n+m)(n+m-1)}\,\bar P_{n-1}^{m-1} + \sqrt{(n-m)(n-m-1)}\,\bar P_{n-1}^{m+1}]$,
    ///   $d = \sqrt{(2n+1)/(4n(n+1)(2n-1))}$, the second term absent for `m >= n - 1`.
    ///
    /// Rows `n = 0` and, on regular grids, `n = nlat - 1` with even `m` (the `ndo1`/`ndo2` rule) stay zero.
    /// Gaussian grids weight each row by the Gaussian weights to get $Z_v$, $Z_w$; regular grids use
    /// `compute_regular_vector_zw`.
    ///
    /// Fortran: sphcom.f / vbinit, wbinit, vbin, wbin (V, W), zvinit, zwinit (Z_v, Z_w); vhaes.f, vhses.f, vhags.f
    ///
    /// Difference from Fortran: V and W are formed directly from $\bar P$ here; the Fortran init
    /// routines build them from their own trigonometric-series/recurrence tables.
    fn build_vector(&self) -> VectorTables<T> {
        let mmax = self.mmax_vector;
        let mut v_even = ParityTable::allocate_zero(self.nlat, mmax, self.imm1, true);
        let mut v_odd = ParityTable::allocate_zero(self.nlat, mmax, self.imid, false);
        let mut w_even = ParityTable::allocate_zero(self.nlat, mmax, self.imid, true);
        let mut w_odd = ParityTable::allocate_zero(self.nlat, mmax, self.imm1, false);

        let mut zv_even = ParityTable::allocate_zero(self.nlat, mmax, self.imm1, true);
        let mut zv_odd = ParityTable::allocate_zero(self.nlat, mmax, self.imid, false);
        let mut zw_even = ParityTable::allocate_zero(self.nlat, mmax, self.imid, true);
        let mut zw_odd = ParityTable::allocate_zero(self.nlat, mmax, self.imm1, false);

        let imid = self.imid;
        let imm1 = self.imm1;
        let nlat = self.nlat;
        let is_gaussian = self.gridtype == GridType::Gaussian;

        // Gaussian grids: Z_v = wts * V and Z_w = wts * W, row by row; regular grids use the
        // Clenshaw-Curtis functions computed in compute_regular_vector_zw below.
        // We need P_{n}^{m+1} for V_n^m (m < n) and P_{n-1}^{m+1} for W_n^m (m < n-1),
        // so request up to mmax + 1 (clamped to nlat).
        let p_m_cap = (mmax + 1).min(nlat);
        let mut v_buf = vec![0.0_f64; imid];
        let mut w_buf = vec![0.0_f64; imid];
        let mut zv_buf = vec![0.0_f64; imid];
        let mut zw_buf = vec![0.0_f64; imid];

        stream_pbar_nh(
            nlat,
            nlat - 1,
            p_m_cap,
            &self.colats_nh,
            |n, _m_lim, cur_p, prev_p| {
                if n == 0 {
                    return;
                }
                let nf = n as f64;
                let max_m_vec = n.min(mmax - 1);

                // V_n^0 = -P_n^1 (always exists for n >= 1 since p_m_cap >= 2)
                {
                    let p1 = &cur_p[imid..2 * imid];
                    for i in 0..imid {
                        v_buf[i] = -p1[i];
                    }
                    // Fortran vhaes.f / vhses.f / vhags.f: with mlat = nlat mod 2, ndo1 = nlat - 1 if mlat != 0 and
                    // ndo2 = nlat - 1 if mlat == 0 (else nlat); the np1 loops stop there, so degree n = nlat - 1 is skipped for even m.
                    // SPHEREPACK ndo1/ndo2 rule: n == nlat - 1 with m even is zeroed
                    if !(n == nlat - 1) {
                        if n % 2 == 0 {
                            v_even.set_row_from_f64(0, n, &v_buf[..imm1]);
                            if is_gaussian {
                                for i in 0..imm1 {
                                    zv_buf[i] = v_buf[i] * self.weights_nh[i];
                                }
                                zv_even.set_row_from_f64(0, n, &zv_buf[..imm1]);
                            }
                        } else {
                            v_odd.set_row_from_f64(0, n, &v_buf);
                            if is_gaussian {
                                for i in 0..imid {
                                    zv_buf[i] = v_buf[i] * self.weights_nh[i];
                                }
                                zv_odd.set_row_from_f64(0, n, &zv_buf);
                            }
                        }
                    }
                }

                // 1 <= m <= min(n - 1, mmax - 1) for V_n^m
                // V_n^m = (sqrt((n+m)(n-m+1)) P_n^{m-1} - sqrt((n-m)(n+m+1)) P_n^{m+1}) / sqrt(4 n (n+1)).
                let dcf_v = (4.0 * nf * (nf + 1.0)).sqrt();
                let m_hi_v = (n - 1).min(mmax - 1);
                for m in 1..=m_hi_v {
                    if n == nlat - 1 && m % 2 == 0 {
                        continue;
                    }
                    let mf = m as f64;
                    let abel = ((nf + mf) * (nf - mf + 1.0)).sqrt() / dcf_v;
                    let bbel = ((nf - mf) * (nf + mf + 1.0)).sqrt() / dcf_v;
                    let p_mm1 = &cur_p[(m - 1) * imid..m * imid];
                    let p_mp1 = &cur_p[(m + 1) * imid..(m + 2) * imid];
                    for i in 0..imid {
                        v_buf[i] = abel * p_mm1[i] - bbel * p_mp1[i];
                    }
                    if (n - m) % 2 == 0 {
                        v_even.set_row_from_f64(m, n, &v_buf[..imm1]);
                        if is_gaussian {
                            for i in 0..imm1 {
                                zv_buf[i] = v_buf[i] * self.weights_nh[i];
                            }
                            zv_even.set_row_from_f64(m, n, &zv_buf[..imm1]);
                        }
                    } else {
                        v_odd.set_row_from_f64(m, n, &v_buf);
                        if is_gaussian {
                            for i in 0..imid {
                                zv_buf[i] = v_buf[i] * self.weights_nh[i];
                            }
                            zv_odd.set_row_from_f64(m, n, &zv_buf);
                        }
                    }
                }

                // m = n for V_n^n (n - m == 0, even): V_n^n = P_n^{n-1} / sqrt(2*(n+1))
                if n <= mmax - 1 && !(n == nlat - 1 && n % 2 == 0) {
                    let denom = (2.0 * (nf + 1.0)).sqrt();
                    let p_nm1 = &cur_p[(n - 1) * imid..n * imid];
                    for i in 0..imm1 {
                        v_buf[i] = p_nm1[i] / denom;
                    }
                    v_even.set_row_from_f64(n, n, &v_buf[..imm1]);
                    if is_gaussian {
                        for i in 0..imm1 {
                            zv_buf[i] = v_buf[i] * self.weights_nh[i];
                        }
                        zv_even.set_row_from_f64(n, n, &zv_buf[..imm1]);
                    }
                }

                // 1 <= m <= min(n, mmax - 1) for W_n^m
                // W_n^m = dcf_w (sqrt((n+m)(n+m-1)) P_{n-1}^{m-1} + sqrt((n-m)(n-m-1)) P_{n-1}^{m+1}); the second term is absent for m >= n-1.
                let dcf_w = ((2.0 * nf + 1.0) / (4.0 * nf * (nf + 1.0) * (2.0 * nf - 1.0))).sqrt();
                for m in 1..=max_m_vec {
                    if n == nlat - 1 && m % 2 == 0 {
                        continue;
                    }
                    let mf = m as f64;
                    let abel = dcf_w * ((nf + mf) * (nf + mf - 1.0)).sqrt();
                    let prev_mm1 = &prev_p[(m - 1) * imid..m * imid];
                    if m < n - 1 {
                        let bbel = dcf_w * ((nf - mf) * (nf - mf - 1.0)).sqrt();
                        let prev_mp1 = &prev_p[(m + 1) * imid..(m + 2) * imid];
                        for i in 0..imid {
                            w_buf[i] = abel * prev_mm1[i] + bbel * prev_mp1[i];
                        }
                    } else {
                        for i in 0..imid {
                            w_buf[i] = abel * prev_mm1[i];
                        }
                    }

                    if (n - m) % 2 == 0 {
                        w_even.set_row_from_f64(m, n, &w_buf);
                        if is_gaussian {
                            for i in 0..imid {
                                zw_buf[i] = w_buf[i] * self.weights_nh[i];
                            }
                            zw_even.set_row_from_f64(m, n, &zw_buf);
                        }
                    } else {
                        w_odd.set_row_from_f64(m, n, &w_buf[..imm1]);
                        if is_gaussian {
                            for i in 0..imm1 {
                                zw_buf[i] = w_buf[i] * self.weights_nh[i];
                            }
                            zw_odd.set_row_from_f64(m, n, &zw_buf[..imm1]);
                        }
                    }
                }
            },
        );

        // Regular grid: Z_v, Z_w from the Clenshaw-Curtis functions.
        if !is_gaussian {
            compute_regular_vector_zw(
                nlat,
                mmax,
                &self.colats_nh,
                |m, n, row| {
                    // Zero rows: n = 0, and n = nlat - 1 with even m (ndo1/ndo2 rule).
                    if n == 0 || (n == nlat - 1 && m % 2 == 0) {
                        return;
                    }
                    if (n - m) % 2 == 0 {
                        zv_even.set_row_from_f64(m, n, &row[..imm1]);
                    } else {
                        zv_odd.set_row_from_f64(m, n, row);
                    }
                },
                |m, n, row| {
                    // Zero rows: n = 0, m = 0 (W vanishes), and n = nlat - 1 with even m (ndo1/ndo2 rule).
                    if n == 0 || m == 0 || (n == nlat - 1 && m % 2 == 0) {
                        return;
                    }
                    if (n - m) % 2 == 0 {
                        zw_even.set_row_from_f64(m, n, row);
                    } else {
                        zw_odd.set_row_from_f64(m, n, &row[..imm1]);
                    }
                },
            );
        }

        VectorTables {
            v_even,
            v_odd,
            w_even,
            w_odd,
            zv_even,
            zv_odd,
            zw_even,
            zw_odd,
        }
    }
}
