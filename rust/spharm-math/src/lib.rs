//! Pure-Rust SPHEREPACK-compatible mathematical kernels.
//!
//! All Gaussian quadrature roots/weights, Clenshaw-Curtis quadrature kernels,
//! and associated Legendre recurrences are evaluated in `f64` using SPHEREPACK
//! normalization:
//! $$\int_0^\pi (\bar{P}_n^m(\theta))^2 \sin\theta\, d\theta = 1, \quad \bar{P}_0^0 = \frac{\sqrt{2}}{2}.$$
//! Output coefficient and basis tables support `T = f32` and `T = f64` via [`SpharmFloat`].
//!
//! Derived from NCAR SPHEREPACK 3.2 (J. Adams and P. Swarztrauber), see LICENSE.spherepack.
//!
//! Correspondence of the modules with the Fortran reference sources (`fortran_ref/src/fortran`
//! for the fixed-form SPHEREPACK files, `fortran_ref/src` for the Fortran 90 files):
//!
//! | Rust module | Fortran source |
//! |---|---|
//! | `gauss` | `gaqd.f` (`gaqd`, `cpdp`, `tpdp`) |
//! | `geodesic` | `ihgeod.f` (`ihgeod`, `ctos`, `stoc`) |
//! | `legendre` | `alf.f` (`alfk`, `lfim`, `lfin`, `lfpt`), `sphcom.f` (`dnlfk`, `dnlft`), `getlegfunc.f` |
//! | `reduced` | `reduced_gaussian_scalar.f90` (Legendre basis and derivative basis parts) |
//! | `regular_weights`, `tables` | see the documentation of these modules |
//!
//! Differences that apply to the whole crate: 0-based indexing, `f64` internal precision (the
//! Fortran routines mostly use single precision `real`), no workspace arrays (`wsave`, `lsave`,
//! `work`, `wlfim`), `Result<_, String>` instead of the Fortran `ierror` status code, and no
//! `hrfft` (the longitude FFT is not part of this crate).

pub mod gauss;
pub mod geodesic;
pub mod legendre;
pub mod reduced;
pub mod regular_weights;
pub mod tables;

use std::fmt::Debug;

pub use gauss::{gaussian_colats_wts, gaussian_lats_wts};
pub use geodesic::icosahedral_geodesic;
pub use legendre::{get_legendre_pbar, legendre_pbar};
pub use reduced::{ReducedBasisData, ReducedRowBasis};
pub use tables::{GridType, ParityHarmonicTables, ParityTable, ScalarTables, VectorTables};

/// Sealed floating-point storage trait implemented for `f32` and `f64`.
///
/// No direct Fortran counterpart. Describes the storage precision of the output tables: the kernels
/// compute in `f64` and results are narrowed with [`SpharmFloat::from_f64`] only when stored.
///
/// Difference from Fortran: the Fortran tables are typically single precision `real` (kernels such as
/// `gaqd` and `dnlfk` are double precision internally); here the choice between `f32` and `f64` storage
/// is a type parameter.
pub trait SpharmFloat: Copy + Send + Sync + Debug + 'static {
    /// The additive identity of the storage type, used to zero-initialize tables.
    const ZERO: Self;
    /// Converts an `f64` result into the storage type (a rounding cast for `f32`).
    fn from_f64(v: f64) -> Self;
    /// Widens a stored value back to `f64` (exact for both implementations).
    fn to_f64(self) -> f64;
}

/// Single precision storage, the precision of the Fortran `real` tables; narrowing uses `as f32`.
impl SpharmFloat for f32 {
    const ZERO: Self = 0.0;
    #[inline]
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
}

/// Double precision storage, lossless.
impl SpharmFloat for f64 {
    const ZERO: Self = 0.0;
    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, PI};

    /// For several `nlat`, checks the lengths, that the weights sum to 2 and are positive, that the
    /// latitudes are antisymmetric about the equator and strictly decreasing, that the weights are
    /// symmetric, that odd `nlat` has an equatorial latitude of 0, and that `nlat = 0` is an error
    /// (Fortran `gaqd`: `ierror = 1`).
    #[test]
    fn test_gaussian_lats_wts_properties() {
        for &nlat in &[1usize, 2, 3, 4, 16, 37, 64, 128] {
            let (lats, wts) = gaussian_lats_wts(nlat).unwrap();
            assert_eq!(lats.len(), nlat);
            assert_eq!(wts.len(), nlat);

            let wsum: f64 = wts.iter().sum();
            assert!((wsum - 2.0).abs() < 1e-13, "nlat={nlat} sum={wsum}");

            for i in 0..nlat {
                assert!(wts[i] > 0.0);
                assert!((lats[i] + lats[nlat - 1 - i]).abs() < 1e-13);
                assert!((wts[i] - wts[nlat - 1 - i]).abs() < 1e-14);
                if i + 1 < nlat {
                    assert!(lats[i] > lats[i + 1]);
                }
            }
            if nlat % 2 == 1 {
                assert!(lats[nlat / 2].abs() < 1e-13);
            }
        }
        assert!(gaussian_lats_wts(0).is_err());
    }

    /// Checks closed forms of $\bar{P}_0^0$, $\bar{P}_1^0$ and $\bar{P}_1^1$ at one colatitude, the
    /// discrete orthonormality of $\bar{P}_n^m$ over Gaussian quadrature (`nlat = 32`, `ntrunc = 15`),
    /// finite values at the pole, and that `ntrunc >= nlat` is rejected.
    #[test]
    fn test_legendre_pbar_normalization_and_exact_values() {
        let theta = 0.7_f64;
        let p = get_legendre_pbar(10, 4, theta).unwrap();
        assert_eq!(p.len(), 5 * 6 / 2);

        // P_0^0 = sqrt(2)/2
        assert!((p[0] - 0.5_f64.sqrt()).abs() < 1e-15);
        // P_1^0 = sqrt(3/2)*cos(theta)
        assert!((p[1] - (1.5_f64.sqrt() * theta.cos())).abs() < 1e-15);
        // P_1^1 = sqrt(3/4)*sin(theta) at nm = 5
        assert!((p[5] - (0.75_f64.sqrt() * theta.sin())).abs() < 1e-15);

        // Orthogonality via Gauss quadrature on nlat = 32 up to ntrunc = 15
        let nlat = 32;
        let ntrunc = 15;
        let (colats, wts) = gaussian_colats_wts(nlat).unwrap();
        let all_p: Vec<Vec<f64>> = colats
            .iter()
            .map(|&th| get_legendre_pbar(nlat, ntrunc, th).unwrap())
            .collect();
        for m in 0..=5 {
            let base = m * (2 * ntrunc - m + 3) / 2;
            for n1 in m..=10 {
                for n2 in m..=10 {
                    let dot: f64 = (0..nlat)
                        .map(|j| wts[j] * all_p[j][base + n1 - m] * all_p[j][base + n2 - m])
                        .sum();
                    let expected = if n1 == n2 { 1.0 } else { 0.0 };
                    assert!(
                        (dot - expected).abs() < 1e-13,
                        "m={m} n1={n1} n2={n2} dot={dot}"
                    );
                }
            }
        }

        // Poles must evaluate cleanly without NaN
        let p_pole = get_legendre_pbar(20, 19, 0.0).unwrap();
        assert!(p_pole.iter().all(|v| v.is_finite()));
        assert!(get_legendre_pbar(5, 5, FRAC_PI_2).is_err());
    }

    /// Checks that `m < 2` is rejected, that the point count is `10 * (m - 1)^2 + 2` (the `ihgeod` header), that the first two points are the poles, and that all values are finite and in range.
    #[test]
    fn test_icosahedral_geodesic_counts_and_poles() {
        assert!(icosahedral_geodesic(1).is_err());
        for m in [2usize, 3, 5, 10] {
            let (lats, lons) = icosahedral_geodesic(m).unwrap();
            let expected = 10 * (m - 1) * (m - 1) + 2;
            assert_eq!(lats.len(), expected);
            assert_eq!(lons.len(), expected);
            assert!((lats[0] - 90.0).abs() < 1e-6);
            assert!((lats[1] + 90.0).abs() < 1e-6);
            assert!(lats.iter().all(|v| v.is_finite() && *v >= -90.01 && *v <= 90.01));
            assert!(lons.iter().all(|v| v.is_finite() && *v >= -0.01 && *v <= 360.01));
        }
    }

    /// Checks that the scalar analysis tables are orthonormal against the synthesis tables on Regular and Gaussian grids, for even and odd `nlat`. On Regular grids `n = nlat - 1` is excluded for odd `m` (the `ndo` rule of the SPHEREPACK Fortran drivers).
    #[test]
    fn test_regular_and_gaussian_scalar_orthogonality_and_poles() {
        for &(nlat, nlon, grid) in &[
            (16usize, 32usize, GridType::Gaussian),
            (17usize, 36usize, GridType::Gaussian),
            (16usize, 32usize, GridType::Regular),
            (17usize, 36usize, GridType::Regular),
        ] {
            let tables = ParityHarmonicTables::<f64>::new(nlat, nlon, grid).unwrap();
            let sc = tables.ensure_scalar();

            // Check exact Clenshaw-Curtis / Gaussian orthogonality: sum_{i} Z_n^m * P_{n'}^m
            // Note: on full sphere, sum_full = 2 * sum_{i < imm1} + (equator if odd and n-m even)
            // On Regular grids, SPHEREPACK excludes n == nlat - 1 when m is odd (ndo rule).
            let is_odd_nlat = nlat % 2 == 1;
            for m in 0..tables.mmax_scalar.min(8) {
                let n_max = if grid == GridType::Regular && m % 2 == 1 {
                    nlat - 2
                } else {
                    nlat - 1
                };
                let ne = sc.p_even.rows_for_m(m, n_max);
                let pe = sc.p_even.block_for_m(m, n_max);
                let ze = sc.z_even.block_for_m(m, n_max);
                for r1 in 0..ne {
                    for r2 in 0..ne {
                        let mut dot = 0.0_f64;
                        for i in 0..tables.imm1 {
                            dot += 2.0 * ze[r1 * tables.imid + i] * pe[r2 * tables.imid + i];
                        }
                        if is_odd_nlat {
                            let i = tables.imid - 1;
                            dot += ze[r1 * tables.imid + i] * pe[r2 * tables.imid + i];
                        }
                        let expected = if r1 == r2 { 1.0 } else { 0.0 };
                        assert!(
                            (dot - expected).abs() < 1e-12,
                            "grid={grid:?} nlat={nlat} m={m} even r1={r1} r2={r2} dot={dot}"
                        );
                    }
                }

                let no = sc.p_odd.rows_for_m(m, n_max);
                let po = sc.p_odd.block_for_m(m, n_max);
                let zo = sc.z_odd.block_for_m(m, n_max);
                for r1 in 0..no {
                    for r2 in 0..no {
                        let mut dot = 0.0_f64;
                        for i in 0..tables.imm1 {
                            dot += 2.0 * zo[r1 * tables.imm1 + i] * po[r2 * tables.imm1 + i];
                        }
                        let expected = if r1 == r2 { 1.0 } else { 0.0 };
                        assert!(
                            (dot - expected).abs() < 1e-12,
                            "grid={grid:?} nlat={nlat} m={m} odd r1={r1} r2={r2} dot={dot}"
                        );
                    }
                }
            }
        }
    }

    /// Checks that the vector tables are finite everywhere (no polar division by zero on Regular grids) and that the vector spherical harmonics are orthonormal, including the `ndo` limit on the maximum degree for even and odd `m`.
    #[test]
    fn test_regular_and_gaussian_vector_tables_orthogonality_and_ndo_rule() {
        for &(nlat, nlon, grid) in &[
            (16usize, 32usize, GridType::Gaussian),
            (17usize, 36usize, GridType::Gaussian),
            (16usize, 32usize, GridType::Regular),
            (17usize, 36usize, GridType::Regular),
        ] {
            let tables = ParityHarmonicTables::<f64>::new(nlat, nlon, grid).unwrap();
            let vec_t = tables.ensure_vector();

            // Verify finite values everywhere (no polar division by zero on Regular grid)
            assert!(vec_t.v_even.data.iter().all(|v| v.is_finite()));
            assert!(vec_t.v_odd.data.iter().all(|v| v.is_finite()));
            assert!(vec_t.w_even.data.iter().all(|v| v.is_finite()));
            assert!(vec_t.w_odd.data.iter().all(|v| v.is_finite()));
            assert!(vec_t.zv_even.data.iter().all(|v| v.is_finite()));
            assert!(vec_t.zv_odd.data.iter().all(|v| v.is_finite()));
            assert!(vec_t.zw_even.data.iter().all(|v| v.is_finite()));
            assert!(vec_t.zw_odd.data.iter().all(|v| v.is_finite()));

            // Verify vector spherical harmonic biorthogonality:
            // integral (V_n^m * V_{n'}^m + W_n^m * W_{n'}^m) sin(theta) dtheta = delta_{n, n'}
            // for valid n in 1..nlat-1 (and n == nlat-1 when m is odd).
            let is_odd_nlat = nlat % 2 == 1;
            for m in 0..tables.mmax_vector.min(8) {
                let n_max = if m % 2 == 0 { nlat - 2 } else { nlat - 1 };

                // Even (n - m): V is odd across equator (imm1), W is even across equator (imid)
                let ne = vec_t.v_even.rows_for_m(m, n_max);
                let ve = vec_t.v_even.block_for_m(m, n_max);
                let zve = vec_t.zv_even.block_for_m(m, n_max);
                let we = vec_t.w_even.block_for_m(m, n_max);
                let zwe = vec_t.zw_even.block_for_m(m, n_max);
                for r1 in 0..ne {
                    let n1 = m + 2 * r1;
                    if n1 == 0 {
                        continue;
                    }
                    for r2 in 0..ne {
                        let n2 = m + 2 * r2;
                        if n2 == 0 {
                            continue;
                        }
                        let mut dot = 0.0_f64;
                        for i in 0..tables.imm1 {
                            dot += 2.0
                                * (zve[r1 * tables.imm1 + i] * ve[r2 * tables.imm1 + i]
                                    + zwe[r1 * tables.imid + i] * we[r2 * tables.imid + i]);
                        }
                        if is_odd_nlat {
                            let i = tables.imid - 1;
                            dot += zwe[r1 * tables.imid + i] * we[r2 * tables.imid + i];
                        }
                        let expected = if r1 == r2 { 1.0 } else { 0.0 };
                        assert!(
                            (dot - expected).abs() < 1e-11,
                            "grid={grid:?} nlat={nlat} m={m} even n1={n1} n2={n2} dot={dot}"
                        );
                    }
                }

                // Odd (n - m): V is even across equator (imid), W is odd across equator (imm1)
                let no = vec_t.v_odd.rows_for_m(m, n_max);
                let vo = vec_t.v_odd.block_for_m(m, n_max);
                let zvo = vec_t.zv_odd.block_for_m(m, n_max);
                let wo = vec_t.w_odd.block_for_m(m, n_max);
                let zwo = vec_t.zw_odd.block_for_m(m, n_max);
                for r1 in 0..no {
                    for r2 in 0..no {
                        let mut dot = 0.0_f64;
                        for i in 0..tables.imm1 {
                            dot += 2.0
                                * (zvo[r1 * tables.imid + i] * vo[r2 * tables.imid + i]
                                    + zwo[r1 * tables.imm1 + i] * wo[r2 * tables.imm1 + i]);
                        }
                        if is_odd_nlat {
                            let i = tables.imid - 1;
                            dot += zvo[r1 * tables.imid + i] * vo[r2 * tables.imid + i];
                        }
                        let expected = if r1 == r2 { 1.0 } else { 0.0 };
                        assert!(
                            (dot - expected).abs() < 1e-11,
                            "grid={grid:?} nlat={nlat} m={m} odd r1={r1} r2={r2} dot={dot}"
                        );
                    }
                }
            }
        }
    }

    /// Checks the reduced Gaussian row bookkeeping (`npoints`, `symmetric_pl`, `max_nlon`), that the `f32` basis and derivative basis are finite, and that the stored $d\bar{P}_n^m/d\theta$ equals $\sqrt{n(n+1)}$ times the vector basis $V_n^m$ of the full Gaussian tables.
    #[test]
    fn test_reduced_row_basis_and_f32_tables() {
        let pl = [16usize, 20, 24, 28, 32, 28, 24, 20, 16];
        let rb = ReducedRowBasis::<f32>::new(&pl).unwrap();
        assert_eq!(rb.nlat, 9);
        assert_eq!(rb.npoints, 208);
        assert!(rb.symmetric_pl);
        assert_eq!(rb.max_nlon, 32);

        let b5 = rb.ensure_dbasis(5).unwrap();
        let dpe = b5.dp_even.as_ref().unwrap();
        let dpo = b5.dp_odd.as_ref().unwrap();
        assert!(b5.p_even.data.iter().all(|v| v.is_finite()));
        assert!(dpe.data.iter().all(|v| v.is_finite()));
        assert!(dpo.data.iter().all(|v| v.is_finite()));

        // Verify dP/dtheta matches vector basis relation sqrt(n*(n+1))*V_n^m
        let full_tables = ParityHarmonicTables::<f64>::new(9, 32, GridType::Gaussian).unwrap();
        let vec_t = full_tables.ensure_vector();
        let rb64 = ReducedRowBasis::<f64>::new(&pl).unwrap();
        let b8 = rb64.ensure_dbasis(7).unwrap();
        let dpe64 = b8.dp_even.as_ref().unwrap();
        let dpo64 = b8.dp_odd.as_ref().unwrap();
        for m in 0..=5 {
            let ne = dpe64.rows_for_m(m, 7);
            let dp_e = dpe64.block_for_m(m, 7);
            let v_e = vec_t.v_even.block_for_m(m, 7);
            for r in 0..ne {
                let n = m + 2 * r;
                let scale = ((n * (n + 1)) as f64).sqrt();
                for i in 0..rb64.imm1 {
                    let diff = (dp_e[r * rb64.imm1 + i] - scale * v_e[r * rb64.imm1 + i]).abs();
                    assert!(diff < 1e-12, "m={m} n={n} i={i} diff={diff}");
                }
            }

            let no = dpo64.rows_for_m(m, 7);
            let dp_o = dpo64.block_for_m(m, 7);
            let v_o = vec_t.v_odd.block_for_m(m, 7);
            for r in 0..no {
                let n = m + 1 + 2 * r;
                let scale = ((n * (n + 1)) as f64).sqrt();
                for i in 0..rb64.imid {
                    let diff = (dp_o[r * rb64.imid + i] - scale * v_o[r * rb64.imid + i]).abs();
                    assert!(diff < 1e-12, "m={m} n={n} i={i} diff={diff}");
                }
            }
        }
    }

    /// Checks that `legendre_pbar` is finite at the south pole $\theta = \pi$ and that $\bar{P}_0^0 = \sqrt{2}/2$, $\bar{P}_1^0 = -\sqrt{3/2}$ there.
    #[test]
    fn test_theta_pi_colatitude_bounds() {
        let p = legendre_pbar(4, PI).unwrap();
        assert!(p.iter().all(|v| v.is_finite()));
        assert!((p[0] - 0.5_f64.sqrt()).abs() < 1e-15);
        assert!((p[1] + 1.5_f64.sqrt()).abs() < 1e-15);
    }
}
