//! Reduced Gaussian grid basis tables parameterized by latitude row lengths `pl`.
//! Derived from NCAR SPHEREPACK 3.2 (J. Adams and P. Swarztrauber), see LICENSE.spherepack.
//!
//! Fortran reference: `reduced_gaussian_scalar.f90` (Fortran 90 file for packed reduced Gaussian grids):
//! `reduced_gaussian_legendre_basis`, `reduced_gaussian_legendre_derivative_from_basis` and
//! `reduced_gaussian_legendre_derivative_basis`. In the Fortran code a reduced Gaussian grid is a packed
//! array with `pl(j)` longitudes on latitude circle `j` (north to south), `offsets(j+1) = offsets(j) +
//! pl(j)` and `ngptot = sum(pl)`; the transforms replace the rectangular longitude FFT by one FFT per row
//! and need the Legendre basis `basis(nlat, nmdim)` with `nmdim = (ntrunc+1)(ntrunc+2)/2`.
//!
//! Spectral packing: `nm = m * (2*ntrunc - m + 3) / 2 + n - m + 1` in Fortran (1-based), `m`-major,
//! `n = m..ntrunc` inside.
//!
//! Difference from Fortran:
//! - The Fortran basis holds all `nlat` rows. The southern rows are copies of the northern rows with
//!   the parity sign `(-1)^(n-m)` (`basis(js, nm) = parity * value`, `js = nlat - i + 1`). The Rust
//!   tables store only the northern rows (`imid = (nlat + 1) / 2`), split by the parity of `n - m` into
//!   an equatorially even table and an equatorially odd table (the latter needs only `nlat / 2`
//!   rows, since an odd function vanishes at the equator).
//! - The Fortran basis comes from `legin` using the workspace of `shagsp`; here values are streamed in
//!   `f64` by `tables::stream_pbar_nh` and narrowed to `T` when stored.
//! - The derivative basis is computed in `f64` from the unrounded `f64` values; Fortran computes it from
//!   the already rounded single precision basis.
//! - Fortran rebuilds the basis (and the FFT `wsave` tables, deliberately without any process-wide cache)
//!   on every call; here a basis is cached per `ntrunc` behind a lock. No FFT tables or workspace
//!   arrays exist in this module.

use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};

use crate::gauss::gaussian_colats_wts;
use crate::tables::{stream_pbar_nh, ParityTable};
use crate::SpharmFloat;

/// Parity-split associated Legendre basis and $\theta$-derivative basis for a reduced Gaussian grid
/// up to spectral truncation `ntrunc`.
///
/// Both `p_*` and `dp_*` are categorized by `(n - m)` degree parity:
/// - `p_even`: $\bar{P}_n^m$ for `n - m` even (`stride = imid`, equatorially even).
/// - `p_odd`: $\bar{P}_n^m$ for `n - m` odd (`stride = imm1`, equatorially odd).
/// - `dp_even`: $d\bar{P}_n^m/d\theta$ for `n - m` even (`stride = imm1`, equatorially odd).
/// - `dp_odd`: $d\bar{P}_n^m/d\theta$ for `n - m` odd (`stride = imid`, equatorially even).
///
/// Fortran: reduced_gaussian_scalar.f90 / reduced_gaussian_legendre_basis and
/// reduced_gaussian_legendre_derivative_basis (full-sphere `basis` and `dbasis` arrays there).
///
/// Difference from Fortran: only northern-hemisphere rows are stored, split by `n - m` parity (see the
/// module documentation); the southern rows follow from the symmetry with sign `(-1)^(n-m)` for `P` and
/// the opposite sign for the derivative.
#[derive(Debug, Clone)]
pub struct ReducedBasisData<T> {
    /// Spectral truncation (`ntrunc` in Fortran) for which the tables were built.
    pub ntrunc: usize,
    /// Table of $\bar{P}_n^m$ for even `n - m`.
    pub p_even: ParityTable<T>,
    /// Table of $\bar{P}_n^m$ for odd `n - m`.
    pub p_odd: ParityTable<T>,
    /// Table of $d\bar{P}_n^m/d\theta$ for even `n - m`; `None` unless the derivative basis was requested (`ensure_dbasis`).
    pub dp_even: Option<ParityTable<T>>,
    /// Table of $d\bar{P}_n^m/d\theta$ for odd `n - m`; `None` unless the derivative basis was requested (`ensure_dbasis`).
    pub dp_odd: Option<ParityTable<T>>,
}

/// Reduced Gaussian grid metadata, Gaussian geometry, and lazy `ntrunc`-keyed parity basis cache.
///
/// Fortran: reduced_gaussian_scalar.f90, the bookkeeping at the start of `reduced_gaussian_grdtospec` and
/// `reduced_gaussian_spectogrd` (`offsets`, `total`, `max_nlon`, `pair_count`, `half_nlat`,
/// `symmetric_pl`) together with `gaqd` for the Gaussian geometry.
///
/// Difference from Fortran: this struct is built once and reused, and the basis cache lives inside it.
#[derive(Debug)]
pub struct ReducedRowBasis<T: SpharmFloat> {
    /// Number of longitudes on each latitude circle, north to south (Fortran `pl(nlat)`).
    pub pl: Vec<usize>,
    /// Start offset of each row in the packed grid, length `nlat + 1`, `offsets[0] = 0` and `offsets[nlat] = npoints` (Fortran `offsets(nlat+1)`, with `offsets(1) = 0`).
    pub offsets: Vec<usize>,
    /// Number of latitude circles (Fortran `nlat`).
    pub nlat: usize,
    /// Total number of grid points, `sum(pl)` (Fortran `total` / `ngptot`).
    pub npoints: usize,
    /// Number of northern rows including the equator for odd `nlat`, `(nlat + 1) / 2` (Fortran `late` / `half_nlat`).
    pub imid: usize,
    /// Number of rows strictly north of the equator, `nlat / 2` (equal to `pair_count`).
    pub imm1: usize,
    /// Number of north-south row pairs, `nlat / 2` (Fortran `pair_count`).
    pub pair_count: usize,
    /// `true` if `pl[j] == pl[nlat - 1 - j]` for all pairs (Fortran `symmetric_pl`); this enables the pairwise north-south treatment in the transforms.
    pub symmetric_pl: bool,
    /// Largest entry of `pl` (Fortran `max_nlon`).
    pub max_nlon: usize,
    /// Full-sphere Gaussian colatitudes in radians (`0..pi`, length `nlat`).
    pub colats_rad: Vec<f64>,
    /// Full-sphere Gaussian quadrature weights in `f64` (sum = `2.0`, length `nlat`).
    pub weights_f64: Vec<f64>,
    /// Full-sphere Gaussian quadrature weights cast to `T` (length `nlat`).
    pub weights: Vec<T>,
    /// Full-sphere $\sin\theta_j$ in `f64` (length `nlat`).
    pub sin_theta_f64: Vec<f64>,
    /// Full-sphere $\sin\theta_j$ cast to `T` (length `nlat`).
    pub sin_theta: Vec<T>,
    /// Lazily built bases keyed by `ntrunc`. No Fortran counterpart (Fortran recomputes the basis on each call).
    cache: RwLock<BTreeMap<usize, Arc<ReducedBasisData<T>>>>,
}

/// Construction, validation and lazy basis computation for a reduced Gaussian grid.
impl<T: SpharmFloat> ReducedRowBasis<T> {
    /// Construct a reduced Gaussian row basis for latitude circle counts `pl` (north-to-south).
    ///
    /// Parameters: `pl`, the number of longitudes on each latitude circle, north to south.
    ///
    /// Returns the row bookkeeping, the Gaussian colatitudes and weights (via `gaqd`, i.e.
    /// [`gaussian_colats_wts`]) and $\sin\theta$ for each row, with an empty basis cache.
    ///
    /// Errors: `Err` if `pl` has fewer than 3 rows (Fortran `ierror = 1` for `nlat < 3`) or if any
    /// `pl[j] < 1` (Fortran `reduced_gaussian_build_fft_wsave`: `ierror = 1` for `pl(j) < 1`), and any
    /// error of [`gaussian_colats_wts`].
    ///
    /// Fortran: reduced_gaussian_scalar.f90 / bookkeeping at the top of `reduced_gaussian_grdtospec`
    pub fn new(pl: &[usize]) -> Result<Self, String> {
        let nlat = pl.len();
        if nlat < 3 {
            return Err(format!("pl must contain at least 3 rows, got {nlat}"));
        }
        let mut offsets = Vec::with_capacity(nlat + 1);
        offsets.push(0);
        let mut total = 0usize;
        let mut max_nlon = 0usize;
        for (idx, &row_nlon) in pl.iter().enumerate() {
            if row_nlon < 1 {
                return Err(format!("pl[{idx}] must be >= 1, got {row_nlon}"));
            }
            total += row_nlon;
            if row_nlon > max_nlon {
                max_nlon = row_nlon;
            }
            offsets.push(total);
        }

        // imid = half_nlat, imm1 = pair_count = nlat/2; symmetric_pl compares each northern row with its mirror row (Fortran loop `pl(j) /= pl(nlat - j + 1)`).
        let imid = (nlat + 1) / 2;
        let imm1 = nlat / 2;
        let pair_count = nlat / 2;
        let symmetric_pl = (0..pair_count).all(|j| pl[j] == pl[nlat - 1 - j]);

        // Gaussian geometry: the Fortran derivative routine obtains it by `call gaqd(nlat, theta, weights, ...)`. Weights and sin(theta) are kept in f64 and also narrowed to T.
        let (colats_rad, weights_f64) = gaussian_colats_wts(nlat)?;
        let sin_theta_f64: Vec<f64> = colats_rad.iter().map(|&th| th.sin()).collect();
        let weights: Vec<T> = weights_f64.iter().map(|&w| T::from_f64(w)).collect();
        let sin_theta: Vec<T> = sin_theta_f64.iter().map(|&s| T::from_f64(s)).collect();

        Ok(Self {
            pl: pl.to_vec(),
            offsets,
            nlat,
            npoints: total,
            imid,
            imm1,
            pair_count,
            symmetric_pl,
            max_nlon,
            colats_rad,
            weights_f64,
            weights,
            sin_theta_f64,
            sin_theta,
            cache: RwLock::new(BTreeMap::new()),
        })
    }

    /// For a symmetric `pl`, return the first northern-hemisphere row index `j in 0..=pair_count`
    /// where `2 * m <= pl[j]` for each `m in 0..=ntrunc` (`pair_count` means no non-equatorial pairs).
    ///
    /// Returns, for each order `m = 0..=ntrunc`, the first northern row that carries order `m`.
    ///
    /// A row with `row_nlon = pl[j]` longitudes only represents orders with `2 * m <= row_nlon`; this is the
    /// Fortran `mlimit = min(ntrunc, row_nlon / 2)` (and the test `2*m < row_nlon` before using the
    /// sine/imaginary slot `2m+1`). The scan stops at the first row satisfying the condition, so the result
    /// is only meaningful when `pl` does not decrease from the pole toward the equator (typical for reduced
    /// Gaussian grids).
    ///
    /// Errors: see [`ReducedRowBasis::validate_ntrunc`].
    ///
    /// No direct Fortran counterpart (the Fortran loops test the condition per row).
    pub fn first_pair_for_m(&self, ntrunc: usize) -> Result<Vec<usize>, String> {
        self.validate_ntrunc(ntrunc)?;
        let mut first = Vec::with_capacity(ntrunc + 1);
        for m in 0..=ntrunc {
            let twom = 2 * m;
            let mut j = 0usize;
            while j < self.pair_count && twom > self.pl[j] {
                j += 1;
            }
            first.push(j);
        }
        Ok(first)
    }

    /// For each `m in 0..=ntrunc`, return whether the equatorial row (`nlat` odd) has `2 * m <= pl[imid - 1]`.
    ///
    /// Returns, for each `m = 0..=ntrunc`, whether the equatorial row participates (`2 * m <= pl[imid - 1]`).
    /// The equatorial row exists only for odd `nlat` (`imid > pair_count`, Fortran `half_nlat > pair_count`);
    /// for even `nlat` all entries are `false`. In Fortran the equatorial row is handled separately after the
    /// north-south pair loop, with `mlimit = min(ntrunc, row_nlon / 2)`.
    ///
    /// Errors: see [`ReducedRowBasis::validate_ntrunc`].
    ///
    /// No direct Fortran counterpart as a function.
    pub fn center_active(&self, ntrunc: usize) -> Result<Vec<bool>, String> {
        self.validate_ntrunc(ntrunc)?;
        let has_equator = self.imid > self.pair_count;
        let eq_nlon = if has_equator { self.pl[self.imid - 1] } else { 0 };
        Ok((0..=ntrunc)
            .map(|m| has_equator && 2 * m <= eq_nlon)
            .collect())
    }

    /// Checks `ntrunc <= nlat - 1`. Fortran: `ierror = 2` if `ntrunc < 0 .or. ntrunc > nlat - 1`; the
    /// negative case cannot occur for `usize`.
    #[inline]
    pub fn validate_ntrunc(&self, ntrunc: usize) -> Result<(), String> {
        if ntrunc >= self.nlat {
            return Err(format!(
                "ntrunc must be <= nlat - 1 ({}), got {ntrunc}",
                self.nlat - 1
            ));
        }
        Ok(())
    }

    /// Lazily compute/fetch parity-split $\bar{P}_n^m$ basis up to `ntrunc`.
    ///
    /// Returns the cached parity-split basis for `ntrunc`, computing it on first use (no derivative tables).
    /// Errors: see [`ReducedRowBasis::validate_ntrunc`].
    ///
    /// Fortran: reduced_gaussian_scalar.f90 / reduced_gaussian_legendre_basis
    ///
    /// Difference from Fortran: cached per `ntrunc` and stored only for the northern rows (see the module
    /// documentation).
    pub fn ensure_basis(&self, ntrunc: usize) -> Result<Arc<ReducedBasisData<T>>, String> {
        self.validate_ntrunc(ntrunc)?;
        if let Some(existing) = self.cache.read().unwrap().get(&ntrunc) {
            return Ok(Arc::clone(existing));
        }

        let built = Arc::new(self.build_basis(ntrunc, false));
        let mut guard = self.cache.write().unwrap();
        let entry = guard.entry(ntrunc).or_insert_with(|| Arc::clone(&built));
        Ok(Arc::clone(entry))
    }

    /// Lazily compute/fetch parity-split $\bar{P}_n^m$ AND $d\bar{P}_n^m/d\theta$ basis up to `ntrunc`.
    ///
    /// Returns the cached basis for `ntrunc` including the derivative tables `dp_even` and `dp_odd`,
    /// rebuilding it if a cached entry has no derivative tables. Errors: see
    /// [`ReducedRowBasis::validate_ntrunc`].
    ///
    /// Fortran: reduced_gaussian_scalar.f90 / reduced_gaussian_legendre_derivative_basis (builds the basis,
    /// then calls `reduced_gaussian_legendre_derivative_from_basis`).
    pub fn ensure_dbasis(&self, ntrunc: usize) -> Result<Arc<ReducedBasisData<T>>, String> {
        self.validate_ntrunc(ntrunc)?;
        if let Some(existing) = self.cache.read().unwrap().get(&ntrunc) {
            if existing.dp_even.is_some() && existing.dp_odd.is_some() {
                return Ok(Arc::clone(existing));
            }
        }

        let built = Arc::new(self.build_basis(ntrunc, true));
        let mut guard = self.cache.write().unwrap();
        if let Some(existing) = guard.get(&ntrunc) {
            if existing.dp_even.is_some() && existing.dp_odd.is_some() {
                return Ok(Arc::clone(existing));
            }
        }
        guard.insert(ntrunc, Arc::clone(&built));
        Ok(built)
    }

    /// Builds the parity-split tables for `ntrunc`, with the derivative tables when `with_deriv` is set.
    ///
    /// Derivative relation (Fortran `reduced_gaussian_legendre_derivative_from_basis`, loop over
    /// `n = max(1, m) .. ntrunc`):
    /// $$\frac{d\bar{P}_n^m}{d\theta} = \frac{n\cos\theta\,\bar{P}_n^m
    /// - c_{nm}\,\bar{P}_{n-1}^m}{\sin\theta}, \quad
    /// c_{nm} = \sqrt{\frac{(2n+1)(n^2-m^2)}{2n-1}},$$
    /// where the second term is absent for `n == m`.
    ///
    /// Fortran: reduced_gaussian_scalar.f90 / reduced_gaussian_legendre_basis and
    /// reduced_gaussian_legendre_derivative_from_basis
    fn build_basis(&self, ntrunc: usize, with_deriv: bool) -> ReducedBasisData<T> {
        let mmax = ntrunc + 1;
        let max_n_dim = ntrunc + 1;
        // Allocate the parity tables. Row counts: P with even n-m is symmetric about the equator and needs imid rows; P with odd n-m is antisymmetric (zero on the equator) and needs imm1 rows. For the derivative the roles swap: dP/dtheta with even n-m is odd about the equator (imm1 rows), with odd n-m even (imid rows).
        let mut p_even = ParityTable::allocate_zero(max_n_dim, mmax, self.imid, true);
        let mut p_odd = ParityTable::allocate_zero(max_n_dim, mmax, self.imm1, false);

        let mut dp_even = if with_deriv {
            Some(ParityTable::allocate_zero(max_n_dim, mmax, self.imm1, true))
        } else {
            None
        };
        let mut dp_odd = if with_deriv {
            Some(ParityTable::allocate_zero(max_n_dim, mmax, self.imid, false))
        } else {
            None
        };

        let imid = self.imid;
        let imm1 = self.imm1;
        let colats_nh = &self.colats_rad[..imid];
        let mut dp_buf = if with_deriv { vec![0.0_f64; imid] } else { Vec::new() };
        let cos_nh: Vec<f64> = if with_deriv {
            colats_nh.iter().map(|&th| th.cos()).collect()
        } else {
            Vec::new()
        };
        let sin_nh = &self.sin_theta_f64[..imid];

        // Stream the northern-hemisphere P_n^m rows degree by degree: for each n the callback receives the orders 0..=m_lim in `cur_p` (m-major blocks of imid rows) and the degree n-1 rows in `prev_p`, which the derivative recurrence needs.
        stream_pbar_nh(self.nlat, ntrunc, mmax, colats_nh, |n, m_lim, cur_p, prev_p| {
            for m in 0..=m_lim {
                let row = &cur_p[m * imid..(m + 1) * imid];
                // Southern rows follow from basis(js, nm) = (-1)^(n-m) * basis(i, nm) in Fortran, so only the northern rows are stored, in the table selected by the parity of n - m.
                if (n - m) % 2 == 0 {
                    p_even.set_row_from_f64(m, n, row);
                } else {
                    p_odd.set_row_from_f64(m, n, row);
                }

                // dP/dtheta; the n = 0 function is constant, so its derivative is zero (Fortran starts at n = max(1, m)).
                if with_deriv && n >= 1 {
                    let nf = n as f64;
                    if n > m {
                        let mf = m as f64;
                        // coeff = sqrt((2n+1)(n^2 - m^2)/(2n-1)), Fortran `coeff`.
                        let coeff = (((2.0 * nf + 1.0) * (nf * nf - mf * mf))
                            / (2.0 * nf - 1.0))
                            .sqrt();
                        let prev_row = &prev_p[m * imid..(m + 1) * imid];
                        for i in 0..imid {
                            // Fortran: deriv = n*cos(theta)*basis(j,nm) - coeff*basis(j,prev_nm); dbasis(j,nm) = deriv/sin(theta).
                            dp_buf[i] = (nf * cos_nh[i] * row[i] - coeff * prev_row[i]) / sin_nh[i];
                        }
                    } else {
                        for i in 0..imid {
                            // n == m: Fortran `prev_nm = 0`, so only the first term remains.
                            dp_buf[i] = (nf * cos_nh[i] * row[i]) / sin_nh[i];
                        }
                    }

                    // Even n - m: dP/dtheta is odd about the equator, keep the imm1 rows north of it. Odd n - m: keep all imid rows.
                    if (n - m) % 2 == 0 {
                        if let Some(ref mut dpe) = dp_even {
                            dpe.set_row_from_f64(m, n, &dp_buf[..imm1]);
                        }
                    } else if let Some(ref mut dpo) = dp_odd {
                        dpo.set_row_from_f64(m, n, &dp_buf);
                    }
                }
            }
        });

        ReducedBasisData {
            ntrunc,
            p_even,
            p_odd,
            dp_even,
            dp_odd,
        }
    }
}
