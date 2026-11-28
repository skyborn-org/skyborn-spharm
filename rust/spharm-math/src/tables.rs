//! Equatorial-parity-split scalar and vector harmonic tables for regular and Gaussian grids.

use std::f64::consts::PI;
use std::sync::OnceLock;

use crate::gauss::gaussian_colats_wts;
use crate::regular_weights::{compute_regular_scalar_z, compute_regular_vector_zw};
use crate::SpharmFloat;

/// Grid type supported by `ParityHarmonicTables`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridType {
    Regular,
    Gaussian,
}

impl GridType {
    /// Parse `"regular"` or `"gaussian"` (case-insensitive).
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
#[derive(Debug, Clone)]
pub struct ParityTable<T> {
    pub data: Vec<T>,
    pub row_offsets: Vec<usize>,
    pub stride: usize,
    pub mmax: usize,
    pub nlat: usize,
    pub is_even_nm: bool,
}

impl<T> ParityTable<T> {
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
    #[inline]
    pub fn block_for_m(&self, m: usize, ntrunc: usize) -> &[T] {
        let nrows = self.rows_for_m(m, ntrunc);
        if nrows == 0 || self.stride == 0 {
            return &[];
        }
        let start = self.row_offsets[m] * self.stride;
        &self.data[start..start + nrows * self.stride]
    }

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
#[derive(Debug, Clone)]
pub struct VectorTables<T> {
    pub v_even: ParityTable<T>,
    pub v_odd: ParityTable<T>,
    pub w_even: ParityTable<T>,
    pub w_odd: ParityTable<T>,
    pub zv_even: ParityTable<T>,
    pub zv_odd: ParityTable<T>,
    pub zw_even: ParityTable<T>,
    pub zw_odd: ParityTable<T>,
}

/// Compute northern-hemisphere $\bar{P}_n^m(\theta_i)$ in `f64` via `dnlfk`/`dnlft` ($m=0,1$)
/// and Schwarztrauber's recurrence ($m \ge 2$), calling `emit(n, m, &pbar_nh)` in ascending `n` order.
/// Keeps only 3 `n`-levels of size `(max_m + 1) * imid` in temporary memory.
pub(crate) fn stream_pbar_nh<F>(
    nlat: usize,
    max_n: usize,
    max_m_plus_1: usize,
    colats_nh: &[f64],
    mut emit: F,
) where
    F: FnMut(usize, usize, &[f64], &[f64]),
{
    let imid = colats_nh.len();
    let m_cap = max_m_plus_1.min(max_n + 1);
    let stride = m_cap * imid;
    let mut ring = vec![0.0_f64; 3 * stride];
    let mut sin_th = Vec::with_capacity(imid);
    let mut cos_th = Vec::with_capacity(imid);
    for &th in colats_nh {
        let (s, c) = th.sin_cos();
        sin_th.push(s);
        cos_th.push(c);
    }

    // n = 0
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

    for n in 2..=max_n {
        let nm2_off = ((n + 1) % 3) * stride;
        let nm1_off = ((n + 2) % 3) * stride;
        let cur_off = (n % 3) * stride;
        let m_lim = n.min(m_cap - 1);

        let nf = n as f64;
        let c_num = 2.0 * nf + 1.0;

        // m = 0 via 3-term Bonnet recurrence
        {
            let a0 = (c_num * (2.0 * nf - 1.0)).sqrt() / nf;
            let b0 = ((nf - 1.0) / nf) * (c_num / (2.0 * nf - 3.0)).sqrt();
            for i in 0..imid {
                ring[cur_off + i] =
                    a0 * cos_th[i] * ring[nm1_off + i] - b0 * ring[nm2_off + i];
            }
        }

        // m = 1 via 3-term Bonnet recurrence
        if m_lim >= 1 {
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
            let c_den = 2.0 * nf - 3.0;
            for m in 2..=m_lim {
                let mf = m as f64;
                let d1 = (mf + nf - 1.0) * (mf + nf);
                let abel = (c_num * (mf + nf - 2.0) * (mf + nf - 3.0) / (c_den * d1)).sqrt();
                let cbel = (((nf - mf + 1.0) * (nf - mf + 2.0)) / d1).sqrt();

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

        let _ = nlat;
        let cur_slice = &ring[cur_off..cur_off + (m_lim + 1) * imid];
        let prev_m = (n - 1).min(m_cap - 1);
        let prev_slice = &ring[nm1_off..nm1_off + (prev_m + 1) * imid];
        emit(n, m_lim, cur_slice, prev_slice);
    }
}

/// Equatorial-parity-split spherical harmonic tables with lazy scalar/vector initialization.
#[derive(Debug)]
pub struct ParityHarmonicTables<T: SpharmFloat> {
    pub nlat: usize,
    pub nlon: usize,
    pub imid: usize,
    pub imm1: usize,
    pub mmax_scalar: usize,
    pub mmax_vector: usize,
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
    pub fn ensure_scalar(&self) -> &ScalarTables<T> {
        self.scalar.get_or_init(|| self.build_scalar())
    }

    /// Return (lazily initializing if needed) the parity-split vector tables.
    pub fn ensure_vector(&self) -> &VectorTables<T> {
        self.vector.get_or_init(|| self.build_vector())
    }

    /// Return already-initialized scalar tables if present.
    pub fn scalar_if_init(&self) -> Option<&ScalarTables<T>> {
        self.scalar.get()
    }

    /// Return already-initialized vector tables if present.
    pub fn vector_if_init(&self) -> Option<&VectorTables<T>> {
        self.vector.get()
    }

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
                    // SPHEREPACK regular-grid ndo rule (shaes/shses):
                    // at n == nlat - 1 with m odd, sin((nlat-1)*theta) vanishes on all nodes.
                    if !is_gaussian && n == self.nlat - 1 && m % 2 == 1 {
                        continue;
                    }
                    let row = &cur_p[m * imid..(m + 1) * imid];
                    if (n - m) % 2 == 0 {
                        p_even.set_row_from_f64(m, n, row);
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

        if !is_gaussian {
            compute_regular_vector_zw(
                nlat,
                mmax,
                &self.colats_nh,
                |m, n, row| {
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
