//! Reduced Gaussian grid basis tables parameterized by latitude row lengths `pl`.

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
#[derive(Debug, Clone)]
pub struct ReducedBasisData<T> {
    pub ntrunc: usize,
    pub p_even: ParityTable<T>,
    pub p_odd: ParityTable<T>,
    pub dp_even: Option<ParityTable<T>>,
    pub dp_odd: Option<ParityTable<T>>,
}

/// Reduced Gaussian grid metadata, Gaussian geometry, and lazy `ntrunc`-keyed parity basis cache.
#[derive(Debug)]
pub struct ReducedRowBasis<T: SpharmFloat> {
    pub pl: Vec<usize>,
    pub offsets: Vec<usize>,
    pub nlat: usize,
    pub npoints: usize,
    pub imid: usize,
    pub imm1: usize,
    pub pair_count: usize,
    pub symmetric_pl: bool,
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
    cache: RwLock<BTreeMap<usize, Arc<ReducedBasisData<T>>>>,
}

impl<T: SpharmFloat> ReducedRowBasis<T> {
    /// Construct a reduced Gaussian row basis for latitude circle counts `pl` (north-to-south).
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

        let imid = (nlat + 1) / 2;
        let imm1 = nlat / 2;
        let pair_count = nlat / 2;
        let symmetric_pl = (0..pair_count).all(|j| pl[j] == pl[nlat - 1 - j]);

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
    pub fn center_active(&self, ntrunc: usize) -> Result<Vec<bool>, String> {
        self.validate_ntrunc(ntrunc)?;
        let has_equator = self.imid > self.pair_count;
        let eq_nlon = if has_equator { self.pl[self.imid - 1] } else { 0 };
        Ok((0..=ntrunc)
            .map(|m| has_equator && 2 * m <= eq_nlon)
            .collect())
    }

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

    fn build_basis(&self, ntrunc: usize, with_deriv: bool) -> ReducedBasisData<T> {
        let mmax = ntrunc + 1;
        let max_n_dim = ntrunc + 1;
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

        stream_pbar_nh(self.nlat, ntrunc, mmax, colats_nh, |n, m_lim, cur_p, prev_p| {
            for m in 0..=m_lim {
                let row = &cur_p[m * imid..(m + 1) * imid];
                if (n - m) % 2 == 0 {
                    p_even.set_row_from_f64(m, n, row);
                } else {
                    p_odd.set_row_from_f64(m, n, row);
                }

                if with_deriv && n >= 1 {
                    let nf = n as f64;
                    if n > m {
                        let mf = m as f64;
                        let coeff = (((2.0 * nf + 1.0) * (nf * nf - mf * mf))
                            / (2.0 * nf - 1.0))
                            .sqrt();
                        let prev_row = &prev_p[m * imid..(m + 1) * imid];
                        for i in 0..imid {
                            dp_buf[i] = (nf * cos_nh[i] * row[i] - coeff * prev_row[i]) / sin_nh[i];
                        }
                    } else {
                        for i in 0..imid {
                            dp_buf[i] = (nf * cos_nh[i] * row[i]) / sin_nh[i];
                        }
                    }

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
