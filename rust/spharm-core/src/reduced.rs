use std::sync::OnceLock;

use faer::Accum;
use num_complex::Complex;
use numpy::ndarray;
use rayon::prelude::*;
use spharm_math::ReducedRowBasis;

use crate::engine::{compute_tile_size, FaerGemmFloat, VectorMode, PAR_WORK_THRESHOLD};
use crate::fft::FftCache;
use crate::spectral::{apply_invlap_batch, nmdim_from_ntrunc, spectral_offset};

pub struct ReducedEngine {
    pub pl: Vec<usize>,
    pub offsets: Vec<usize>,
    pub s2n_offsets: Vec<usize>,
    pub nlat: usize,
    pub npoints: usize,
    pub rsphere: f64,
    pub legfunc: String,
    pub symmetric_pl: bool,
    pub is_nh_monotonic: bool,
    basis_f32: OnceLock<ReducedRowBasis<f32>>,
    basis_f64: OnceLock<ReducedRowBasis<f64>>,
    fft_f32: FftCache<f32>,
    fft_f64: FftCache<f64>,
}

impl ReducedEngine {
    pub fn new(pl: Vec<usize>, rsphere: f64, legfunc: String) -> Result<Self, String> {
        if pl.len() < 3 {
            return Err(format!("pl must contain at least 3 rows, got {}", pl.len()));
        }
        if pl.iter().any(|&p| p < 4) {
            return Err("each reduced Gaussian row must contain >= 4 points".to_string());
        }
        if !(rsphere > 0.0) || !rsphere.is_finite() {
            return Err(format!("rsphere must be positive, got {rsphere}"));
        }
        if legfunc != "stored" && legfunc != "computed" {
            return Err(format!(
                "legfunc must be 'stored' or 'computed', got '{legfunc}'"
            ));
        }

        let nlat = pl.len();
        let pair_count = nlat / 2;
        let symmetric_pl = (0..pair_count).all(|j| pl[j] == pl[nlat - 1 - j]);
        let is_nh_monotonic = (1..pair_count).all(|j| pl[j] >= pl[j - 1]);

        let mut offsets = Vec::with_capacity(nlat + 1);
        offsets.push(0);
        let mut total = 0usize;
        for &p in &pl {
            total += p;
            offsets.push(total);
        }

        let mut s2n_offsets = Vec::with_capacity(nlat + 1);
        s2n_offsets.push(0);
        let mut s2n_total = 0usize;
        for &p in pl.iter().rev() {
            s2n_total += p;
            s2n_offsets.push(s2n_total);
        }

        Ok(Self {
            pl,
            offsets,
            s2n_offsets,
            nlat,
            npoints: total,
            rsphere,
            legfunc,
            symmetric_pl,
            is_nh_monotonic,
            basis_f32: OnceLock::new(),
            basis_f64: OnceLock::new(),
            fft_f32: FftCache::new(),
            fft_f64: FftCache::new(),
        })
    }

    #[inline]
    fn basis<T: TypedReducedBasis>(&self) -> &ReducedRowBasis<T> {
        T::get_basis(self)
    }

    #[inline]
    fn fft<T: TypedReducedBasis>(&self) -> &FftCache<T> {
        T::get_fft(self)
    }

    #[inline]
    fn row_range(&self, j_n2s: usize, south_to_north: bool) -> (usize, usize) {
        if !south_to_north {
            (self.offsets[j_n2s], self.pl[j_n2s])
        } else {
            let s_idx = self.nlat - 1 - j_n2s;
            (self.s2n_offsets[s_idx], self.pl[j_n2s])
        }
    }

    pub fn grdtospec<T: TypedReducedBasis>(
        &self,
        grid: &ndarray::ArrayView2<'_, T>,
        out_spec: &mut [Complex<T>],
        ntrunc: usize,
        south_to_north: bool,
    ) -> Result<(), String> {
        let nt = grid.shape()[0];
        let nmdim = nmdim_from_ntrunc(ntrunc);
        if nt == 0 {
            return Ok(());
        }
        out_spec[..nt * nmdim].fill(T::complex_zero());

        let rb = self.basis::<T>();
        let basis_data = rb.ensure_basis(ntrunc)?;
        let imid = rb.imid;
        let imm1 = rb.imm1;
        let pair_count = rb.pair_count;
        let is_odd_nlat = self.nlat % 2 == 1;
        let mmax = ntrunc + 1;
        let first_pair = if self.symmetric_pl && self.is_nh_monotonic {
            rb.first_pair_for_m(ntrunc)?
        } else {
            vec![0; mmax]
        };

        let fft_cache = self.fft::<T>();
        let plans: Vec<_> = (0..imid)
            .map(|j| fft_cache.get_or_plan(self.pl[j]))
            .collect();
        let full_plans: Vec<_> = if self.symmetric_pl {
            Vec::new()
        } else {
            (0..self.nlat)
                .map(|j| fft_cache.get_or_plan(self.pl[j]))
                .collect()
        };

        let per_field_bytes = (self.npoints + 2 * mmax * imid * 2) * T::BYTES;
        let max_tile = compute_tile_size(per_field_bytes, nt);
        let contiguous = grid.as_slice();

        if self.symmetric_pl {
            let mut f_even = vec![T::ZERO; mmax * imid * 2 * max_tile];
            let mut f_odd = vec![T::ZERO; mmax * imm1 * 2 * max_tile];
            let max_rows = basis_data.p_even.rows_for_m(0, ntrunc).max(1);
            let mut spec_buf_m = vec![T::ZERO; mmax * max_rows * 2 * max_tile];
            let mut fft_e_ki = vec![T::complex_zero(); max_tile * imid * mmax];
            let mut fft_o_ki = vec![T::complex_zero(); max_tile * imm1 * mmax];

            let mut start = 0usize;
            while start < nt {
                let tile = (nt - start).min(max_tile);
                let cols = 2 * tile;
                let par_fields = tile >= 2 && tile * self.npoints >= PAR_WORK_THRESHOLD;
                let par_m = mmax * imid * cols >= PAR_WORK_THRESHOLD;

                fft_e_ki[..tile * imid * mmax].fill(T::complex_zero());
                fft_o_ki[..tile * imm1 * mmax].fill(T::complex_zero());

                let do_fold = |(tk, (fe_k, fo_k)): (usize, (&mut [Complex<T>], &mut [Complex<T>]))| {
                    let k = start + tk;
                    let mut real_e = vec![T::ZERO; rb.max_nlon];
                    let mut real_o = vec![T::ZERO; rb.max_nlon];
                    let mut cpx_buf = vec![T::complex_zero(); rb.max_nlon / 2 + 1];
                    let mut scratch = Vec::new();

                    for j in 0..pair_count {
                        let row_nlon = self.pl[j];
                        let (off_n, _) = self.row_range(j, south_to_north);
                        let (off_s, _) = self.row_range(self.nlat - 1 - j, south_to_north);
                        let scale = T::from_real(rb.weights_f64[j] / (row_nlon as f64));

                        if let Some(sl) = contiguous {
                            let row_n = &sl[k * self.npoints + off_n..k * self.npoints + off_n + row_nlon];
                            let row_s = &sl[k * self.npoints + off_s..k * self.npoints + off_s + row_nlon];
                            for i in 0..row_nlon {
                                let vn = row_n[i];
                                let vs = row_s[i];
                                real_e[i] = (vn + vs) * scale;
                                real_o[i] = (vn - vs) * scale;
                            }
                        } else {
                            for i in 0..row_nlon {
                                let vn = grid[[k, off_n + i]];
                                let vs = grid[[k, off_s + i]];
                                real_e[i] = (vn + vs) * scale;
                                real_o[i] = (vn - vs) * scale;
                            }
                        }

                        let nfreq = row_nlon / 2 + 1;
                        let mlimit = ntrunc.min(row_nlon / 2);
                        let r2c = &plans[j].r2c;
                        let slen = r2c.get_scratch_len();
                        if scratch.len() < slen {
                            scratch.resize(slen, T::complex_zero());
                        }
                        r2c.process_with_scratch(
                            &mut real_e[..row_nlon],
                            &mut cpx_buf[..nfreq],
                            &mut scratch[..slen],
                        )
                        .expect("reduced r2c even failed");
                        fe_k[j * mmax..j * mmax + mlimit + 1]
                            .copy_from_slice(&cpx_buf[..mlimit + 1]);

                        r2c.process_with_scratch(
                            &mut real_o[..row_nlon],
                            &mut cpx_buf[..nfreq],
                            &mut scratch[..slen],
                        )
                        .expect("reduced r2c odd failed");
                        fo_k[j * mmax..j * mmax + mlimit + 1]
                            .copy_from_slice(&cpx_buf[..mlimit + 1]);
                    }

                    if is_odd_nlat {
                        let j = imid - 1;
                        let row_nlon = self.pl[j];
                        let (off_eq, _) = self.row_range(j, south_to_north);
                        let scale = T::from_real(rb.weights_f64[j] / (row_nlon as f64));
                        if let Some(sl) = contiguous {
                            let row_eq = &sl[k * self.npoints + off_eq..k * self.npoints + off_eq + row_nlon];
                            for i in 0..row_nlon {
                                real_e[i] = row_eq[i] * scale;
                            }
                        } else {
                            for i in 0..row_nlon {
                                real_e[i] = grid[[k, off_eq + i]] * scale;
                            }
                        }
                        let nfreq = row_nlon / 2 + 1;
                        let mlimit = ntrunc.min(row_nlon / 2);
                        let r2c = &plans[j].r2c;
                        let slen = r2c.get_scratch_len();
                        if scratch.len() < slen {
                            scratch.resize(slen, T::complex_zero());
                        }
                        r2c.process_with_scratch(
                            &mut real_e[..row_nlon],
                            &mut cpx_buf[..nfreq],
                            &mut scratch[..slen],
                        )
                        .expect("reduced r2c eq failed");
                        fe_k[j * mmax..j * mmax + mlimit + 1]
                            .copy_from_slice(&cpx_buf[..mlimit + 1]);
                    }
                };
                if par_fields {
                    fft_e_ki[..tile * imid * mmax]
                        .par_chunks_mut(imid * mmax)
                        .zip(fft_o_ki[..tile * imm1 * mmax].par_chunks_mut(imm1 * mmax))
                        .enumerate()
                        .for_each(do_fold);
                } else {
                    fft_e_ki[..tile * imid * mmax]
                        .chunks_mut(imid * mmax)
                        .zip(fft_o_ki[..tile * imm1 * mmax].chunks_mut(imm1 * mmax))
                        .enumerate()
                        .for_each(do_fold);
                }

                let fe_m_stride = imid * cols;
                let fo_m_stride = imm1 * cols;
                let do_pack = |(m, (fe_m, fo_m)): (usize, (&mut [T], &mut [T]))| {
                    for i in 0..imid {
                        let dst = &mut fe_m[i * cols..(i + 1) * cols];
                        for tk in 0..tile {
                            let c = fft_e_ki[(tk * imid + i) * mmax + m];
                            dst[2 * tk] = c.re;
                            dst[2 * tk + 1] = c.im;
                        }
                    }
                    for i in 0..imm1 {
                        let dst = &mut fo_m[i * cols..(i + 1) * cols];
                        for tk in 0..tile {
                            let c = fft_o_ki[(tk * imm1 + i) * mmax + m];
                            dst[2 * tk] = c.re;
                            dst[2 * tk + 1] = c.im;
                        }
                    }
                };
                if par_m {
                    f_even[..mmax * fe_m_stride]
                        .par_chunks_mut(fe_m_stride)
                        .zip(f_odd[..mmax * fo_m_stride].par_chunks_mut(fo_m_stride))
                        .enumerate()
                        .for_each(do_pack);
                } else {
                    f_even[..mmax * fe_m_stride]
                        .chunks_mut(fe_m_stride)
                        .zip(f_odd[..mmax * fo_m_stride].chunks_mut(fo_m_stride))
                        .enumerate()
                        .for_each(do_pack);
                }

                let spec_m_stride = max_rows * cols;
                let do_gemm_e = |(m, spec_m): (usize, &mut [T])| {
                    let ne = basis_data.p_even.rows_for_m(m, ntrunc);
                    if ne == 0 {
                        return;
                    }
                    let j0 = first_pair[m].min(imid);
                    let active_rows = imid - j0;
                    if active_rows == 0 {
                        spec_m[..ne * cols].fill(T::ZERO);
                        return;
                    }
                    let p_blk = basis_data.p_even.block_for_m(m, ntrunc);
                    let fe_m = &f_even[m * fe_m_stride..(m + 1) * fe_m_stride];
                    T::faer_gemm_subcols(spec_m, ne, cols, Accum::Replace, p_blk, imid, fe_m, j0);
                };
                if par_m {
                    spec_buf_m[..mmax * spec_m_stride]
                        .par_chunks_mut(spec_m_stride)
                        .enumerate()
                        .for_each(do_gemm_e);
                } else {
                    spec_buf_m[..mmax * spec_m_stride]
                        .chunks_mut(spec_m_stride)
                        .enumerate()
                        .for_each(do_gemm_e);
                }

                for m in 0..mmax {
                    let ne = basis_data.p_even.rows_for_m(m, ntrunc);
                    if ne > 0 {
                        let m_base = spectral_offset(m, ntrunc);
                        let spec_m = &spec_buf_m[m * spec_m_stride..];
                        for r in 0..ne {
                            let nm = m_base + 2 * r;
                            let row = &spec_m[r * cols..(r + 1) * cols];
                            for tk in 0..tile {
                                out_spec[(start + tk) * nmdim + nm] =
                                    Complex::new(row[2 * tk], row[2 * tk + 1]);
                            }
                        }
                    }
                }

                let do_gemm_o = |(m, spec_m): (usize, &mut [T])| {
                    let no = basis_data.p_odd.rows_for_m(m, ntrunc);
                    if no == 0 {
                        return;
                    }
                    let j0 = first_pair[m].min(imm1);
                    let active_rows = imm1 - j0;
                    if active_rows == 0 {
                        spec_m[..no * cols].fill(T::ZERO);
                        return;
                    }
                    let p_blk = basis_data.p_odd.block_for_m(m, ntrunc);
                    let fo_m = &f_odd[m * fo_m_stride..(m + 1) * fo_m_stride];
                    T::faer_gemm_subcols(spec_m, no, cols, Accum::Replace, p_blk, imm1, fo_m, j0);
                };
                if par_m {
                    spec_buf_m[..mmax * spec_m_stride]
                        .par_chunks_mut(spec_m_stride)
                        .enumerate()
                        .for_each(do_gemm_o);
                } else {
                    spec_buf_m[..mmax * spec_m_stride]
                        .chunks_mut(spec_m_stride)
                        .enumerate()
                        .for_each(do_gemm_o);
                }

                for m in 0..mmax {
                    let no = basis_data.p_odd.rows_for_m(m, ntrunc);
                    if no > 0 {
                        let m_base = spectral_offset(m, ntrunc);
                        let spec_m = &spec_buf_m[m * spec_m_stride..];
                        for r in 0..no {
                            let nm = m_base + 1 + 2 * r;
                            let row = &spec_m[r * cols..(r + 1) * cols];
                            for tk in 0..tile {
                                out_spec[(start + tk) * nmdim + nm] =
                                    Complex::new(row[2 * tk], row[2 * tk + 1]);
                            }
                        }
                    }
                }

                start += tile;
            }
        } else {
            // Non-symmetric pl fallback
            let nlat = self.nlat;
            out_spec[..nt * nmdim]
                .par_chunks_mut(nmdim)
                .enumerate()
                .for_each(|(k, out_row)| {
                    let mut real_row = vec![T::ZERO; rb.max_nlon];
                    let mut cpx_buf = vec![T::complex_zero(); rb.max_nlon / 2 + 1];
                    let mut scratch = Vec::new();
                    let mut fft_rows = vec![T::complex_zero(); nlat * mmax];

                    for j in 0..nlat {
                        let row_nlon = self.pl[j];
                        let (off, _) = self.row_range(j, south_to_north);
                        let inv_nlon = T::from_real(1.0 / (row_nlon as f64));
                        if let Some(sl) = contiguous {
                            let row = &sl[k * self.npoints + off..k * self.npoints + off + row_nlon];
                            for i in 0..row_nlon {
                                real_row[i] = row[i] * inv_nlon;
                            }
                        } else {
                            for i in 0..row_nlon {
                                real_row[i] = grid[[k, off + i]] * inv_nlon;
                            }
                        }
                        let nfreq = row_nlon / 2 + 1;
                        let mlimit = ntrunc.min(row_nlon / 2);
                        let r2c = &full_plans[j].r2c;
                        let slen = r2c.get_scratch_len();
                        if scratch.len() < slen {
                            scratch.resize(slen, T::complex_zero());
                        }
                        r2c.process_with_scratch(
                            &mut real_row[..row_nlon],
                            &mut cpx_buf[..nfreq],
                            &mut scratch[..slen],
                        )
                        .expect("reduced r2c asym failed");
                        fft_rows[j * mmax..j * mmax + mlimit + 1]
                            .copy_from_slice(&cpx_buf[..mlimit + 1]);
                    }

                    for m in 0..mmax {
                        let m_base = spectral_offset(m, ntrunc);
                        let ne = basis_data.p_even.rows_for_m(m, ntrunc);
                        let no = basis_data.p_odd.rows_for_m(m, ntrunc);
                        let pe = basis_data.p_even.block_for_m(m, ntrunc);
                        let po = basis_data.p_odd.block_for_m(m, ntrunc);
                        for j in 0..nlat {
                            if 2 * m > self.pl[j] {
                                continue;
                            }
                            let c = fft_rows[j * mmax + m];
                            let w = rb.weights_f64[j];
                            let is_south = j >= imid;
                            let j_nh = if is_south { nlat - 1 - j } else { j };
                            let w_re = w * c.re.to_f64();
                            let w_im = w * c.im.to_f64();
                            for r in 0..ne {
                                let p = pe[r * imid + j_nh].to_f64();
                                let nm = m_base + 2 * r;
                                out_row[nm] = out_row[nm] + Complex::new(T::from_real(w_re * p), T::from_real(w_im * p));
                            }
                            if j_nh < imm1 {
                                for r in 0..no {
                                    let p = po[r * imm1 + j_nh].to_f64();
                                    let s = if is_south { -p } else { p };
                                    let nm = m_base + 1 + 2 * r;
                                    out_row[nm] = out_row[nm] + Complex::new(T::from_real(w_re * s), T::from_real(w_im * s));
                                }
                            }
                        }
                    }
                });
        }

        Ok(())
    }

    pub fn spectogrd<T: TypedReducedBasis>(
        &self,
        spec: &ndarray::ArrayView2<'_, Complex<T>>,
        out_grid: &mut [T],
        ntrunc: usize,
        south_to_north: bool,
    ) -> Result<(), String> {
        self.spectogrd_batch(spec, None, out_grid, None, ntrunc, south_to_north)
    }

    pub fn spectogrd_pair<T: TypedReducedBasis>(
        &self,
        spec_a: &ndarray::ArrayView2<'_, Complex<T>>,
        spec_b: &ndarray::ArrayView2<'_, Complex<T>>,
        out_a: &mut [T],
        out_b: &mut [T],
        ntrunc: usize,
        south_to_north: bool,
    ) -> Result<(), String> {
        self.spectogrd_batch(
            spec_a,
            Some(spec_b),
            out_a,
            Some(out_b),
            ntrunc,
            south_to_north,
        )
    }

    fn spectogrd_batch<T: TypedReducedBasis>(
        &self,
        spec_a: &ndarray::ArrayView2<'_, Complex<T>>,
        spec_b: Option<&ndarray::ArrayView2<'_, Complex<T>>>,
        out_a: &mut [T],
        mut out_b: Option<&mut [T]>,
        ntrunc: usize,
        south_to_north: bool,
    ) -> Result<(), String> {
        let nt = spec_a.shape()[0];
        if nt == 0 {
            return Ok(());
        }
        let rb = self.basis::<T>();
        let basis_data = rb.ensure_basis(ntrunc)?;
        let imid = rb.imid;
        let imm1 = rb.imm1;
        let pair_count = rb.pair_count;
        let is_odd_nlat = self.nlat % 2 == 1;
        let mmax = ntrunc + 1;
        let has_pair = spec_b.is_some();
        let nfields_per_k = if has_pair { 2 } else { 1 };

        let fft_cache = self.fft::<T>();
        let plans: Vec<_> = (0..imid)
            .map(|j| fft_cache.get_or_plan(self.pl[j]))
            .collect();
        let full_plans: Vec<_> = if self.symmetric_pl {
            Vec::new()
        } else {
            (0..self.nlat)
                .map(|j| fft_cache.get_or_plan(self.pl[j]))
                .collect()
        };

        if self.symmetric_pl {
            let per_field_bytes = (self.npoints + 2 * mmax * imid * 2) * T::BYTES * nfields_per_k;
            let max_tile = compute_tile_size(per_field_bytes, nt);

            let mut f_even = vec![T::ZERO; mmax * imid * 2 * max_tile * nfields_per_k];
            let mut f_odd = vec![T::ZERO; mmax * imm1 * 2 * max_tile * nfields_per_k];
            let max_rows = basis_data.p_even.rows_for_m(0, ntrunc).max(1);
            let mut spec_buf_m = vec![T::ZERO; mmax * max_rows * 2 * max_tile * nfields_per_k];

            let mut start = 0usize;
            while start < nt {
                let tile = (nt - start).min(max_tile);
                let eff_t = tile * nfields_per_k;
                let cols = 2 * eff_t;
                let fe_m_stride = imid * cols;
                let fo_m_stride = imm1 * cols;
                let spec_m_stride = max_rows * cols;
                let par_fields = eff_t >= 2 && eff_t * self.npoints >= PAR_WORK_THRESHOLD;
                let par_m = mmax * imid * cols >= PAR_WORK_THRESHOLD;

                let do_syn_even = |(m, (fe_m, spec_m)): (usize, (&mut [T], &mut [T]))| {
                    let ne = basis_data.p_even.rows_for_m(m, ntrunc);
                    if ne == 0 || imid == 0 {
                        fe_m.fill(T::ZERO);
                        return;
                    }
                    let m_base = spectral_offset(m, ntrunc);
                    for r in 0..ne {
                        let nm = m_base + 2 * r;
                        let dst = &mut spec_m[r * cols..(r + 1) * cols];
                        for tk in 0..tile {
                            let ca = spec_a[[start + tk, nm]];
                            dst[2 * tk] = ca.re;
                            dst[2 * tk + 1] = if m == 0 { T::ZERO } else { ca.im };
                            if let Some(sb) = spec_b {
                                let cb = sb[[start + tk, nm]];
                                let idx = tile + tk;
                                dst[2 * idx] = cb.re;
                                dst[2 * idx + 1] = if m == 0 { T::ZERO } else { cb.im };
                            }
                        }
                    }
                    let p_blk = basis_data.p_even.block_for_m(m, ntrunc);
                    T::faer_gemm(fe_m, imid, cols, Accum::Replace, p_blk, imid, ne, false, spec_m, ne, cols);
                };
                if par_m {
                    f_even[..mmax * fe_m_stride]
                        .par_chunks_mut(fe_m_stride)
                        .zip(spec_buf_m[..mmax * spec_m_stride].par_chunks_mut(spec_m_stride))
                        .enumerate()
                        .for_each(do_syn_even);
                } else {
                    f_even[..mmax * fe_m_stride]
                        .chunks_mut(fe_m_stride)
                        .zip(spec_buf_m[..mmax * spec_m_stride].chunks_mut(spec_m_stride))
                        .enumerate()
                        .for_each(do_syn_even);
                }

                let do_syn_odd = |(m, (fo_m, spec_m)): (usize, (&mut [T], &mut [T]))| {
                    let no = basis_data.p_odd.rows_for_m(m, ntrunc);
                    if no == 0 || imm1 == 0 {
                        fo_m.fill(T::ZERO);
                        return;
                    }
                    let m_base = spectral_offset(m, ntrunc);
                    for r in 0..no {
                        let nm = m_base + 1 + 2 * r;
                        let dst = &mut spec_m[r * cols..(r + 1) * cols];
                        for tk in 0..tile {
                            let ca = spec_a[[start + tk, nm]];
                            dst[2 * tk] = ca.re;
                            dst[2 * tk + 1] = if m == 0 { T::ZERO } else { ca.im };
                            if let Some(sb) = spec_b {
                                let cb = sb[[start + tk, nm]];
                                let idx = tile + tk;
                                dst[2 * idx] = cb.re;
                                dst[2 * idx + 1] = if m == 0 { T::ZERO } else { cb.im };
                            }
                        }
                    }
                    let p_blk = basis_data.p_odd.block_for_m(m, ntrunc);
                    T::faer_gemm(fo_m, imm1, cols, Accum::Replace, p_blk, imm1, no, false, spec_m, no, cols);
                };
                if par_m {
                    f_odd[..mmax * fo_m_stride]
                        .par_chunks_mut(fo_m_stride)
                        .zip(spec_buf_m[..mmax * spec_m_stride].par_chunks_mut(spec_m_stride))
                        .enumerate()
                        .for_each(do_syn_odd);
                } else {
                    f_odd[..mmax * fo_m_stride]
                        .chunks_mut(fo_m_stride)
                        .zip(spec_buf_m[..mmax * spec_m_stride].chunks_mut(spec_m_stride))
                        .enumerate()
                        .for_each(do_syn_odd);
                }

                let a_ptr = out_a[start * self.npoints..(start + tile) * self.npoints].as_mut_ptr() as usize;
                let b_ptr = out_b
                    .as_deref_mut()
                    .map(|sl| sl[start * self.npoints..(start + tile) * self.npoints].as_mut_ptr() as usize);

                let do_syn_c2r = |ek: usize| {
                    let is_b = ek >= tile;
                    let tk = if is_b { ek - tile } else { ek };
                    let field_ptr = if is_b {
                        (b_ptr.unwrap() as *mut T).wrapping_add(tk * self.npoints)
                    } else {
                        (a_ptr as *mut T).wrapping_add(tk * self.npoints)
                    };

                    let mut cpx_n = vec![T::complex_zero(); rb.max_nlon / 2 + 1];
                    let mut cpx_s = vec![T::complex_zero(); rb.max_nlon / 2 + 1];
                    let mut scratch = Vec::new();

                    for j in 0..pair_count {
                        let row_nlon = self.pl[j];
                        let nfreq = row_nlon / 2 + 1;
                        let mlimit = ntrunc.min(row_nlon / 2);
                        cpx_n[..nfreq].fill(T::complex_zero());
                        cpx_s[..nfreq].fill(T::complex_zero());
                        for m in 0..=mlimit {
                            let fe_idx = m * fe_m_stride + j * cols + 2 * ek;
                            let fo_idx = m * fo_m_stride + j * cols + 2 * ek;
                            let ge_re = f_even[fe_idx];
                            let ge_im = f_even[fe_idx + 1];
                            let go_re = f_odd[fo_idx];
                            let go_im = f_odd[fo_idx + 1];
                            cpx_n[m] = Complex::new(ge_re + go_re, ge_im + go_im);
                            cpx_s[m] = Complex::new(ge_re - go_re, ge_im - go_im);
                        }
                        cpx_n[0].im = T::ZERO;
                        cpx_s[0].im = T::ZERO;
                        if row_nlon % 2 == 0 {
                            cpx_n[row_nlon / 2].im = T::ZERO;
                            cpx_s[row_nlon / 2].im = T::ZERO;
                        }
                        let c2r = &plans[j].c2r;
                        let slen = c2r.get_scratch_len();
                        if scratch.len() < slen {
                            scratch.resize(slen, T::complex_zero());
                        }
                        let (off_n, _) = self.row_range(j, south_to_north);
                        let (off_s, _) = self.row_range(self.nlat - 1 - j, south_to_north);
                        unsafe {
                            let dst_n = std::slice::from_raw_parts_mut(field_ptr.add(off_n), row_nlon);
                            let dst_s = std::slice::from_raw_parts_mut(field_ptr.add(off_s), row_nlon);
                            c2r.process_with_scratch(&mut cpx_n[..nfreq], dst_n, &mut scratch[..slen])
                                .expect("reduced c2r north failed");
                            c2r.process_with_scratch(&mut cpx_s[..nfreq], dst_s, &mut scratch[..slen])
                                .expect("reduced c2r south failed");
                        }
                    }

                    if is_odd_nlat {
                        let j = imid - 1;
                        let row_nlon = self.pl[j];
                        let nfreq = row_nlon / 2 + 1;
                        let mlimit = ntrunc.min(row_nlon / 2);
                        cpx_n[..nfreq].fill(T::complex_zero());
                        for m in 0..=mlimit {
                            let fe_idx = m * fe_m_stride + j * cols + 2 * ek;
                            cpx_n[m] = Complex::new(f_even[fe_idx], f_even[fe_idx + 1]);
                        }
                        cpx_n[0].im = T::ZERO;
                        if row_nlon % 2 == 0 {
                            cpx_n[row_nlon / 2].im = T::ZERO;
                        }
                        let c2r = &plans[j].c2r;
                        let slen = c2r.get_scratch_len();
                        if scratch.len() < slen {
                            scratch.resize(slen, T::complex_zero());
                        }
                        let (off_eq, _) = self.row_range(j, south_to_north);
                        unsafe {
                            let dst_eq = std::slice::from_raw_parts_mut(field_ptr.add(off_eq), row_nlon);
                            c2r.process_with_scratch(&mut cpx_n[..nfreq], dst_eq, &mut scratch[..slen])
                                .expect("reduced c2r eq failed");
                        }
                    }
                };
                if par_fields {
                    (0..eff_t).into_par_iter().for_each(do_syn_c2r);
                } else {
                    (0..eff_t).for_each(do_syn_c2r);
                }

                start += tile;
            }
        } else {
            // Non-symmetric pl fallback
            let nlat = self.nlat;
            let synth_one = |spec: &ndarray::ArrayView2<'_, Complex<T>>, out: &mut [T]| {
                out[..nt * self.npoints]
                    .par_chunks_mut(self.npoints)
                    .enumerate()
                    .for_each(|(k, field_out)| {
                        let mut cpx_buf = vec![T::complex_zero(); rb.max_nlon / 2 + 1];
                        let mut scratch = Vec::new();
                        for j in 0..nlat {
                            let row_nlon = self.pl[j];
                            let nfreq = row_nlon / 2 + 1;
                            let mlimit = ntrunc.min(row_nlon / 2);
                            cpx_buf[..nfreq].fill(T::complex_zero());
                            let is_south = j >= imid;
                            let j_nh = if is_south { nlat - 1 - j } else { j };
                            for m in 0..=mlimit {
                                let m_base = spectral_offset(m, ntrunc);
                                let ne = basis_data.p_even.rows_for_m(m, ntrunc);
                                let no = basis_data.p_odd.rows_for_m(m, ntrunc);
                                let pe = basis_data.p_even.block_for_m(m, ntrunc);
                                let po = basis_data.p_odd.block_for_m(m, ntrunc);
                                let mut sre = 0.0_f64;
                                let mut sim = 0.0_f64;
                                for r in 0..ne {
                                    let p = pe[r * imid + j_nh].to_f64();
                                    let c = spec[[k, m_base + 2 * r]];
                                    sre += c.re.to_f64() * p;
                                    sim += c.im.to_f64() * p;
                                }
                                if j_nh < imm1 {
                                    for r in 0..no {
                                        let p = po[r * imm1 + j_nh].to_f64();
                                        let s = if is_south { -p } else { p };
                                        let c = spec[[k, m_base + 1 + 2 * r]];
                                        sre += c.re.to_f64() * s;
                                        sim += c.im.to_f64() * s;
                                    }
                                }
                                cpx_buf[m] = Complex::new(T::from_real(sre), T::from_real(sim));
                            }
                            cpx_buf[0].im = T::ZERO;
                            if row_nlon % 2 == 0 {
                                cpx_buf[row_nlon / 2].im = T::ZERO;
                            }
                            let c2r = &full_plans[j].c2r;
                            let slen = c2r.get_scratch_len();
                            if scratch.len() < slen {
                                scratch.resize(slen, T::complex_zero());
                            }
                            let (off, _) = self.row_range(j, south_to_north);
                            c2r.process_with_scratch(
                                &mut cpx_buf[..nfreq],
                                &mut field_out[off..off + row_nlon],
                                &mut scratch[..slen],
                            )
                            .expect("reduced c2r asym failed");
                        }
                    });
            };
            synth_one(spec_a, out_a);
            if let (Some(sb), Some(ob)) = (spec_b, out_b.as_deref_mut()) {
                synth_one(sb, ob);
            }
        }

        Ok(())
    }

    pub fn getvrtdivspec<T: TypedReducedBasis>(
        &self,
        ugrid: &ndarray::ArrayView2<'_, T>,
        vgrid: &ndarray::ArrayView2<'_, T>,
        mut out_vrt: Option<&mut [Complex<T>]>,
        mut out_div: Option<&mut [Complex<T>]>,
        ntrunc: usize,
        south_to_north: bool,
        mode: VectorMode,
    ) -> Result<(), String> {
        let nt = ugrid.shape()[0];
        let nmdim = nmdim_from_ntrunc(ntrunc);
        if nt == 0 {
            return Ok(());
        }
        if let Some(ref mut v) = out_vrt {
            v[..nt * nmdim].fill(T::complex_zero());
        }
        if let Some(ref mut d) = out_div {
            d[..nt * nmdim].fill(T::complex_zero());
        }

        let rb = self.basis::<T>();
        let basis_data = rb.ensure_dbasis(ntrunc)?;
        let dp_even = basis_data.dp_even.as_ref().expect("dp_even initialized");
        let dp_odd = basis_data.dp_odd.as_ref().expect("dp_odd initialized");
        let imid = rb.imid;
        let imm1 = rb.imm1;
        let pair_count = rb.pair_count;
        let is_odd_nlat = self.nlat % 2 == 1;
        let mmax = ntrunc + 1;
        let first_pair = if self.symmetric_pl && self.is_nh_monotonic {
            rb.first_pair_for_m(ntrunc)?
        } else {
            vec![0; mmax]
        };

        let inv_r = 1.0 / self.rsphere;
        let weight_inv_r: Vec<f64> = rb.weights_f64.iter().map(|&w| w * inv_r).collect();
        let weight_inv_r_sin: Vec<f64> = weight_inv_r
            .iter()
            .zip(rb.sin_theta_f64.iter())
            .map(|(&wr, &s)| wr / s)
            .collect();

        let fft_cache = self.fft::<T>();
        let plans: Vec<_> = (0..imid)
            .map(|j| fft_cache.get_or_plan(self.pl[j]))
            .collect();
        let full_plans: Vec<_> = if self.symmetric_pl {
            Vec::new()
        } else {
            (0..self.nlat)
                .map(|j| fft_cache.get_or_plan(self.pl[j]))
                .collect()
        };

        let u_contig = ugrid.as_slice();
        let v_contig = vgrid.as_slice();
        let want_vrt = matches!(mode, VectorMode::VrtDiv | VectorMode::VrtOnly);
        let want_div = matches!(mode, VectorMode::VrtDiv | VectorMode::DivOnly);

        if self.symmetric_pl {
            let per_field_bytes = (2 * self.npoints + 8 * mmax * imid * 2) * T::BYTES;
            let max_tile = compute_tile_size(per_field_bytes, nt);

            let mut u_sum_ki = vec![T::complex_zero(); max_tile * imid * mmax];
            let mut u_diff_ki = vec![T::complex_zero(); max_tile * imm1 * mmax];
            let mut v_sum_ki = vec![T::complex_zero(); max_tile * imid * mmax];
            let mut v_diff_ki = vec![T::complex_zero(); max_tile * imm1 * mmax];

            // Note: we scale u/v Fourier coeffs directly by weight_inv_r[_sin] before packing into GEMM RHS!
            let col_mult = match mode {
                VectorMode::VrtDiv => 4,
                VectorMode::DivOnly | VectorMode::VrtOnly => 2,
            };
            let mut rhs_pe = vec![T::ZERO; mmax * imid * col_mult * max_tile];
            let mut rhs_dpe = vec![T::ZERO; mmax * imm1 * col_mult * max_tile];
            let mut rhs_po = vec![T::ZERO; mmax * imm1 * col_mult * max_tile];
            let mut rhs_dpo = vec![T::ZERO; mmax * imid * col_mult * max_tile];

            let max_rows = basis_data.p_even.rows_for_m(0, ntrunc).max(1);
            let mut spec_buf_m = vec![T::ZERO; mmax * max_rows * col_mult * max_tile];

            let mut start = 0usize;
            while start < nt {
                let tile = (nt - start).min(max_tile);
                let cols = col_mult * tile;
                let par_fields = tile >= 2 && tile * self.npoints >= PAR_WORK_THRESHOLD;
                let par_m = mmax * imid * cols >= PAR_WORK_THRESHOLD;

                u_sum_ki[..tile * imid * mmax].fill(T::complex_zero());
                u_diff_ki[..tile * imm1 * mmax].fill(T::complex_zero());
                v_sum_ki[..tile * imid * mmax].fill(T::complex_zero());
                v_diff_ki[..tile * imm1 * mmax].fill(T::complex_zero());

                // Step 1: fold N/S u,v and run r2c
                let do_vec_fold = |(tk, (((us_k, ud_k), vs_k), vd_k)): (
                    usize,
                    (((&mut [Complex<T>], &mut [Complex<T>]), &mut [Complex<T>]), &mut [Complex<T>]),
                )| {
                    let k = start + tk;
                    let mut r_us = vec![T::ZERO; rb.max_nlon];
                    let mut r_ud = vec![T::ZERO; rb.max_nlon];
                    let mut r_vs = vec![T::ZERO; rb.max_nlon];
                    let mut r_vd = vec![T::ZERO; rb.max_nlon];
                    let mut cpx_buf = vec![T::complex_zero(); rb.max_nlon / 2 + 1];
                    let mut scratch = Vec::new();

                    for j in 0..pair_count {
                        let row_nlon = self.pl[j];
                        let (off_n, _) = self.row_range(j, south_to_north);
                        let (off_s, _) = self.row_range(self.nlat - 1 - j, south_to_north);
                        let inv_nlon = T::from_real(1.0 / (row_nlon as f64));
                        if let (Some(usl), Some(vsl)) = (u_contig, v_contig) {
                            let un = &usl[k * self.npoints + off_n..k * self.npoints + off_n + row_nlon];
                            let us = &usl[k * self.npoints + off_s..k * self.npoints + off_s + row_nlon];
                            let vn = &vsl[k * self.npoints + off_n..k * self.npoints + off_n + row_nlon];
                            let vs = &vsl[k * self.npoints + off_s..k * self.npoints + off_s + row_nlon];
                            for i in 0..row_nlon {
                                r_us[i] = (un[i] + us[i]) * inv_nlon;
                                r_ud[i] = (un[i] - us[i]) * inv_nlon;
                                r_vs[i] = (vn[i] + vs[i]) * inv_nlon;
                                r_vd[i] = (vn[i] - vs[i]) * inv_nlon;
                            }
                        } else {
                            for i in 0..row_nlon {
                                let un = ugrid[[k, off_n + i]];
                                let us = ugrid[[k, off_s + i]];
                                let vn = vgrid[[k, off_n + i]];
                                let vs = vgrid[[k, off_s + i]];
                                r_us[i] = (un + us) * inv_nlon;
                                r_ud[i] = (un - us) * inv_nlon;
                                r_vs[i] = (vn + vs) * inv_nlon;
                                r_vd[i] = (vn - vs) * inv_nlon;
                            }
                        }
                        let nfreq = row_nlon / 2 + 1;
                        let mlimit = ntrunc.min(row_nlon / 2);
                        let r2c = &plans[j].r2c;
                        let slen = r2c.get_scratch_len();
                        if scratch.len() < slen {
                            scratch.resize(slen, T::complex_zero());
                        }
                        r2c.process_with_scratch(&mut r_us[..row_nlon], &mut cpx_buf[..nfreq], &mut scratch[..slen])
                            .expect("r2c us");
                        us_k[j * mmax..j * mmax + mlimit + 1].copy_from_slice(&cpx_buf[..mlimit + 1]);
                        r2c.process_with_scratch(&mut r_ud[..row_nlon], &mut cpx_buf[..nfreq], &mut scratch[..slen])
                            .expect("r2c ud");
                        ud_k[j * mmax..j * mmax + mlimit + 1].copy_from_slice(&cpx_buf[..mlimit + 1]);
                        r2c.process_with_scratch(&mut r_vs[..row_nlon], &mut cpx_buf[..nfreq], &mut scratch[..slen])
                            .expect("r2c vs");
                        vs_k[j * mmax..j * mmax + mlimit + 1].copy_from_slice(&cpx_buf[..mlimit + 1]);
                        r2c.process_with_scratch(&mut r_vd[..row_nlon], &mut cpx_buf[..nfreq], &mut scratch[..slen])
                            .expect("r2c vd");
                        vd_k[j * mmax..j * mmax + mlimit + 1].copy_from_slice(&cpx_buf[..mlimit + 1]);
                    }

                    if is_odd_nlat {
                        let j = imid - 1;
                        let row_nlon = self.pl[j];
                        let (off_eq, _) = self.row_range(j, south_to_north);
                        let inv_nlon = T::from_real(1.0 / (row_nlon as f64));
                        if let (Some(usl), Some(vsl)) = (u_contig, v_contig) {
                            let ueq = &usl[k * self.npoints + off_eq..k * self.npoints + off_eq + row_nlon];
                            let veq = &vsl[k * self.npoints + off_eq..k * self.npoints + off_eq + row_nlon];
                            for i in 0..row_nlon {
                                r_us[i] = ueq[i] * inv_nlon;
                                r_vs[i] = veq[i] * inv_nlon;
                            }
                        } else {
                            for i in 0..row_nlon {
                                r_us[i] = ugrid[[k, off_eq + i]] * inv_nlon;
                                r_vs[i] = vgrid[[k, off_eq + i]] * inv_nlon;
                            }
                        }
                        let nfreq = row_nlon / 2 + 1;
                        let mlimit = ntrunc.min(row_nlon / 2);
                        let r2c = &plans[j].r2c;
                        let slen = r2c.get_scratch_len();
                        if scratch.len() < slen {
                            scratch.resize(slen, T::complex_zero());
                        }
                        r2c.process_with_scratch(&mut r_us[..row_nlon], &mut cpx_buf[..nfreq], &mut scratch[..slen])
                            .expect("r2c ueq");
                        us_k[j * mmax..j * mmax + mlimit + 1].copy_from_slice(&cpx_buf[..mlimit + 1]);
                        r2c.process_with_scratch(&mut r_vs[..row_nlon], &mut cpx_buf[..nfreq], &mut scratch[..slen])
                            .expect("r2c veq");
                        vs_k[j * mmax..j * mmax + mlimit + 1].copy_from_slice(&cpx_buf[..mlimit + 1]);
                    }
                };
                if par_fields {
                    u_sum_ki[..tile * imid * mmax]
                        .par_chunks_mut(imid * mmax)
                        .zip(u_diff_ki[..tile * imm1 * mmax].par_chunks_mut(imm1 * mmax))
                        .zip(v_sum_ki[..tile * imid * mmax].par_chunks_mut(imid * mmax))
                        .zip(v_diff_ki[..tile * imm1 * mmax].par_chunks_mut(imm1 * mmax))
                        .enumerate()
                        .for_each(do_vec_fold);
                } else {
                    u_sum_ki[..tile * imid * mmax]
                        .chunks_mut(imid * mmax)
                        .zip(u_diff_ki[..tile * imm1 * mmax].chunks_mut(imm1 * mmax))
                        .zip(v_sum_ki[..tile * imid * mmax].chunks_mut(imid * mmax))
                        .zip(v_diff_ki[..tile * imm1 * mmax].chunks_mut(imm1 * mmax))
                        .enumerate()
                        .for_each(do_vec_fold);
                }

                // Step 2: pack RHS matrices
                let imid_stride = imid * cols;
                let imm1_stride = imm1 * cols;
                let do_vec_pack = |(m, (((pe_m, dpe_m), po_m), dpo_m)): (
                    usize,
                    (((&mut [T], &mut [T]), &mut [T]), &mut [T]),
                )| {
                    let mf = m as f64;
                    for j in 0..imid {
                        let wp = T::from_real(mf * weight_inv_r_sin[j]);
                        let wdp = T::from_real(weight_inv_r[j]);
                        let r_pe = &mut pe_m[j * cols..(j + 1) * cols];
                        let r_dpo = &mut dpo_m[j * cols..(j + 1) * cols];
                        for tk in 0..tile {
                            let us = u_sum_ki[(tk * imid + j) * mmax + m];
                            let vs = v_sum_ki[(tk * imid + j) * mmax + m];
                            match mode {
                                VectorMode::VrtDiv => {
                                    r_pe[4 * tk] = -wp * us.im;
                                    r_pe[4 * tk + 1] = wp * us.re;
                                    r_pe[4 * tk + 2] = -wp * vs.im;
                                    r_pe[4 * tk + 3] = wp * vs.re;
                                    r_dpo[4 * tk] = wdp * vs.re;
                                    r_dpo[4 * tk + 1] = wdp * vs.im;
                                    r_dpo[4 * tk + 2] = -wdp * us.re;
                                    r_dpo[4 * tk + 3] = -wdp * us.im;
                                }
                                VectorMode::DivOnly => {
                                    r_pe[2 * tk] = -wp * us.im;
                                    r_pe[2 * tk + 1] = wp * us.re;
                                    r_dpo[2 * tk] = wdp * vs.re;
                                    r_dpo[2 * tk + 1] = wdp * vs.im;
                                }
                                VectorMode::VrtOnly => {
                                    r_pe[2 * tk] = -wp * vs.im;
                                    r_pe[2 * tk + 1] = wp * vs.re;
                                    r_dpo[2 * tk] = -wdp * us.re;
                                    r_dpo[2 * tk + 1] = -wdp * us.im;
                                }
                            }
                        }
                    }
                    for j in 0..imm1 {
                        let wp = T::from_real(mf * weight_inv_r_sin[j]);
                        let wdp = T::from_real(weight_inv_r[j]);
                        let r_dpe = &mut dpe_m[j * cols..(j + 1) * cols];
                        let r_po = &mut po_m[j * cols..(j + 1) * cols];
                        for tk in 0..tile {
                            let ud = u_diff_ki[(tk * imm1 + j) * mmax + m];
                            let vd = v_diff_ki[(tk * imm1 + j) * mmax + m];
                            match mode {
                                VectorMode::VrtDiv => {
                                    r_dpe[4 * tk] = wdp * vd.re;
                                    r_dpe[4 * tk + 1] = wdp * vd.im;
                                    r_dpe[4 * tk + 2] = -wdp * ud.re;
                                    r_dpe[4 * tk + 3] = -wdp * ud.im;
                                    r_po[4 * tk] = -wp * ud.im;
                                    r_po[4 * tk + 1] = wp * ud.re;
                                    r_po[4 * tk + 2] = -wp * vd.im;
                                    r_po[4 * tk + 3] = wp * vd.re;
                                }
                                VectorMode::DivOnly => {
                                    r_dpe[2 * tk] = wdp * vd.re;
                                    r_dpe[2 * tk + 1] = wdp * vd.im;
                                    r_po[2 * tk] = -wp * ud.im;
                                    r_po[2 * tk + 1] = wp * ud.re;
                                }
                                VectorMode::VrtOnly => {
                                    r_dpe[2 * tk] = -wdp * ud.re;
                                    r_dpe[2 * tk + 1] = -wdp * ud.im;
                                    r_po[2 * tk] = -wp * vd.im;
                                    r_po[2 * tk + 1] = wp * vd.re;
                                }
                            }
                        }
                    }
                };
                if par_m {
                    rhs_pe[..mmax * imid_stride]
                        .par_chunks_mut(imid_stride)
                        .zip(rhs_dpe[..mmax * imm1_stride].par_chunks_mut(imm1_stride))
                        .zip(rhs_po[..mmax * imm1_stride].par_chunks_mut(imm1_stride))
                        .zip(rhs_dpo[..mmax * imid_stride].par_chunks_mut(imid_stride))
                        .enumerate()
                        .for_each(do_vec_pack);
                } else {
                    rhs_pe[..mmax * imid_stride]
                        .chunks_mut(imid_stride)
                        .zip(rhs_dpe[..mmax * imm1_stride].chunks_mut(imm1_stride))
                        .zip(rhs_po[..mmax * imm1_stride].chunks_mut(imm1_stride))
                        .zip(rhs_dpo[..mmax * imid_stride].chunks_mut(imid_stride))
                        .enumerate()
                        .for_each(do_vec_pack);
                }

                // Step 3a: n - m EVEN GEMMs
                let spec_m_stride = max_rows * cols;
                let do_vec_even = |(m, spec_m): (usize, &mut [T])| {
                    let ne = basis_data.p_even.rows_for_m(m, ntrunc);
                    if ne == 0 {
                        return;
                    }
                    let j0 = first_pair[m].min(imid);
                    if m > 0 {
                        let p_blk = basis_data.p_even.block_for_m(m, ntrunc);
                        let pe_m = &rhs_pe[m * imid_stride..(m + 1) * imid_stride];
                        gemm_subcols(spec_m, ne, cols, Accum::Replace, p_blk, imid, pe_m, j0);
                    } else {
                        spec_m[..ne * cols].fill(T::ZERO);
                    }
                    if imm1 > 0 {
                        let j0_o = first_pair[m].min(imm1);
                        let dp_blk = dp_even.block_for_m(m, ntrunc);
                        let dpe_m = &rhs_dpe[m * imm1_stride..(m + 1) * imm1_stride];
                        gemm_subcols(spec_m, ne, cols, Accum::Add, dp_blk, imm1, dpe_m, j0_o);
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

                scatter_reduced_vec(
                    &spec_buf_m,
                    spec_m_stride,
                    cols,
                    mmax,
                    ntrunc,
                    nmdim,
                    start,
                    tile,
                    true,
                    &basis_data.p_even,
                    mode,
                    want_div,
                    want_vrt,
                    out_div.as_deref_mut(),
                    out_vrt.as_deref_mut(),
                );

                // Step 3b: n - m ODD GEMMs
                let do_vec_odd = |(m, spec_m): (usize, &mut [T])| {
                    let no = basis_data.p_odd.rows_for_m(m, ntrunc);
                    if no == 0 {
                        return;
                    }
                    let j0 = first_pair[m].min(imid);
                    let dp_blk = dp_odd.block_for_m(m, ntrunc);
                    let dpo_m = &rhs_dpo[m * imid_stride..(m + 1) * imid_stride];
                    gemm_subcols(spec_m, no, cols, Accum::Replace, dp_blk, imid, dpo_m, j0);
                    if m > 0 && imm1 > 0 {
                        let j0_o = first_pair[m].min(imm1);
                        let p_blk = basis_data.p_odd.block_for_m(m, ntrunc);
                        let po_m = &rhs_po[m * imm1_stride..(m + 1) * imm1_stride];
                        gemm_subcols(spec_m, no, cols, Accum::Add, p_blk, imm1, po_m, j0_o);
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

                scatter_reduced_vec(
                    &spec_buf_m,
                    spec_m_stride,
                    cols,
                    mmax,
                    ntrunc,
                    nmdim,
                    start,
                    tile,
                    false,
                    &basis_data.p_odd,
                    mode,
                    want_div,
                    want_vrt,
                    out_div.as_deref_mut(),
                    out_vrt.as_deref_mut(),
                );

                start += tile;
            }
        } else {
            // Non-symmetric fallback
            let nlat = self.nlat;
            for k in 0..nt {
                let mut r_u = vec![T::ZERO; rb.max_nlon];
                let mut r_v = vec![T::ZERO; rb.max_nlon];
                let mut cpx_buf = vec![T::complex_zero(); rb.max_nlon / 2 + 1];
                let mut scratch = Vec::new();
                let mut u_fft = vec![T::complex_zero(); nlat * mmax];
                let mut v_fft = vec![T::complex_zero(); nlat * mmax];

                for j in 0..nlat {
                    let row_nlon = self.pl[j];
                    let (off, _) = self.row_range(j, south_to_north);
                    let inv_nlon = T::from_real(1.0 / (row_nlon as f64));
                    for i in 0..row_nlon {
                        r_u[i] = ugrid[[k, off + i]] * inv_nlon;
                        r_v[i] = vgrid[[k, off + i]] * inv_nlon;
                    }
                    let nfreq = row_nlon / 2 + 1;
                    let mlimit = ntrunc.min(row_nlon / 2);
                    let r2c = &full_plans[j].r2c;
                    let slen = r2c.get_scratch_len();
                    if scratch.len() < slen {
                        scratch.resize(slen, T::complex_zero());
                    }
                    r2c.process_with_scratch(&mut r_u[..row_nlon], &mut cpx_buf[..nfreq], &mut scratch[..slen])
                        .expect("r2c u asym");
                    u_fft[j * mmax..j * mmax + mlimit + 1].copy_from_slice(&cpx_buf[..mlimit + 1]);
                    r2c.process_with_scratch(&mut r_v[..row_nlon], &mut cpx_buf[..nfreq], &mut scratch[..slen])
                        .expect("r2c v asym");
                    v_fft[j * mmax..j * mmax + mlimit + 1].copy_from_slice(&cpx_buf[..mlimit + 1]);
                }

                for m in 0..mmax {
                    let m_base = spectral_offset(m, ntrunc);
                    let ne = basis_data.p_even.rows_for_m(m, ntrunc);
                    let no = basis_data.p_odd.rows_for_m(m, ntrunc);
                    let pe = basis_data.p_even.block_for_m(m, ntrunc);
                    let po = basis_data.p_odd.block_for_m(m, ntrunc);
                    let dpe = dp_even.block_for_m(m, ntrunc);
                    let dpo = dp_odd.block_for_m(m, ntrunc);
                    for j in 0..nlat {
                        if 2 * m > self.pl[j] {
                            continue;
                        }
                        let uc = u_fft[j * mmax + m];
                        let vc = v_fft[j * mmax + m];
                        let is_south = j >= imid;
                        let j_nh = if is_south { nlat - 1 - j } else { j };
                        let p_sc = (m as f64) * weight_inv_r_sin[j];
                        let dp_sc = weight_inv_r[j];
                        let ure = uc.re.to_f64();
                        let uim = uc.im.to_f64();
                        let vre = vc.re.to_f64();
                        let vim = vc.im.to_f64();

                        for r in 0..ne {
                            let p = pe[r * imid + j_nh].to_f64();
                            let dp = if j_nh < imm1 {
                                let v = dpe[r * imm1 + j_nh].to_f64();
                                if is_south { -v } else { v }
                            } else {
                                0.0
                            };
                            let nm = m_base + 2 * r;
                            if let Some(ref mut div) = out_div {
                                let dre = -p_sc * p * uim + dp_sc * dp * vre;
                                let dim = p_sc * p * ure + dp_sc * dp * vim;
                                let idx = k * nmdim + nm;
                                div[idx] = div[idx] + Complex::new(T::from_real(dre), T::from_real(dim));
                            }
                            if let Some(ref mut vrt) = out_vrt {
                                let vre_out = -dp_sc * dp * ure - p_sc * p * vim;
                                let vim_out = -dp_sc * dp * uim + p_sc * p * vre;
                                let idx = k * nmdim + nm;
                                vrt[idx] = vrt[idx] + Complex::new(T::from_real(vre_out), T::from_real(vim_out));
                            }
                        }
                        for r in 0..no {
                            let p = if j_nh < imm1 {
                                let v = po[r * imm1 + j_nh].to_f64();
                                if is_south { -v } else { v }
                            } else {
                                0.0
                            };
                            let dp = dpo[r * imid + j_nh].to_f64();
                            let nm = m_base + 1 + 2 * r;
                            if let Some(ref mut div) = out_div {
                                let dre = -p_sc * p * uim + dp_sc * dp * vre;
                                let dim = p_sc * p * ure + dp_sc * dp * vim;
                                let idx = k * nmdim + nm;
                                div[idx] = div[idx] + Complex::new(T::from_real(dre), T::from_real(dim));
                            }
                            if let Some(ref mut vrt) = out_vrt {
                                let vre_out = -dp_sc * dp * ure - p_sc * p * vim;
                                let vim_out = -dp_sc * dp * uim + p_sc * p * vre;
                                let idx = k * nmdim + nm;
                                vrt[idx] = vrt[idx] + Complex::new(T::from_real(vre_out), T::from_real(vim_out));
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    pub fn getgrad<T: TypedReducedBasis>(
        &self,
        spec: &ndarray::ArrayView2<'_, Complex<T>>,
        out_u: &mut [T],
        out_v: &mut [T],
        ntrunc: usize,
        south_to_north: bool,
    ) -> Result<(), String> {
        self.getgrad_batch(spec, None, out_u, out_v, None, None, ntrunc, south_to_north)
    }

    pub fn getgrad_pair<T: TypedReducedBasis>(
        &self,
        spec_a: &ndarray::ArrayView2<'_, Complex<T>>,
        spec_b: &ndarray::ArrayView2<'_, Complex<T>>,
        out_au: &mut [T],
        out_av: &mut [T],
        out_bu: &mut [T],
        out_bv: &mut [T],
        ntrunc: usize,
        south_to_north: bool,
    ) -> Result<(), String> {
        self.getgrad_batch(
            spec_a,
            Some(spec_b),
            out_au,
            out_av,
            Some(out_bu),
            Some(out_bv),
            ntrunc,
            south_to_north,
        )
    }

    pub fn getuv<T: TypedReducedBasis>(
        &self,
        vrtspec: &ndarray::ArrayView2<'_, Complex<T>>,
        divspec: &ndarray::ArrayView2<'_, Complex<T>>,
        out_u: &mut [T],
        out_v: &mut [T],
        ntrunc: usize,
        south_to_north: bool,
    ) -> Result<(), String> {
        let nt = vrtspec.shape()[0];
        let nmdim = nmdim_from_ntrunc(ntrunc);
        if nt == 0 {
            return Ok(());
        }
        // Compute psispec = invlap(vrtspec), chispec = invlap(divspec) into compact buffers,
        // then run getgrad_pair(chispec, psispec) and combine u = u_chi - u_psi, v = v_chi + v_psi.
        let mut vrt_c = Vec::with_capacity(nt * nmdim);
        let mut div_c = Vec::with_capacity(nt * nmdim);
        for k in 0..nt {
            for nm in 0..nmdim {
                vrt_c.push(vrtspec[[k, nm]]);
                div_c.push(divspec[[k, nm]]);
            }
        }
        let mut psispec = vec![T::complex_zero(); nt * nmdim];
        let mut chispec = vec![T::complex_zero(); nt * nmdim];
        apply_invlap_batch(&vrt_c, &mut psispec, nt, nmdim, ntrunc, self.rsphere);
        apply_invlap_batch(&div_c, &mut chispec, nt, nmdim, ntrunc, self.rsphere);
        drop(vrt_c);
        drop(div_c);

        let psi_view = ndarray::ArrayView2::from_shape((nt, nmdim), &psispec).unwrap();
        let chi_view = ndarray::ArrayView2::from_shape((nt, nmdim), &chispec).unwrap();

        let mut u_psi = vec![T::ZERO; nt * self.npoints];
        let mut v_psi = vec![T::ZERO; nt * self.npoints];
        self.getgrad_batch(
            &chi_view,
            Some(&psi_view),
            out_u,
            out_v,
            Some(&mut u_psi),
            Some(&mut v_psi),
            ntrunc,
            south_to_north,
        )?;

        // getgrad_pair(chi, psi) returns (u_chi, v_chi, v_psi_into_bu, u_psi_into_bv) -> wait:
        // reduced_gaussian.py:827-833: u_chi, v_chi, v_psi, u_psi = _synthesize_gradient_pair(chispec, psispec)
        // u = u_chi - u_psi, v = v_chi + v_psi where u_psi = vgrad(psi), v_psi = ugrad(psi)!
        // Wait: in getgrad(psi), (ugrad(psi), vgrad(psi)) = ( (1/R sin) dpsi/dphi, -(1/R) dpsi/dtheta ) = (-v_psi_rot, u_psi_rot)!
        // Specifically: u_rot = -1/R dpsi/dlat = +1/R dpsi/dtheta = -vgrad(psi),
        //               v_rot = +1/(R cos lat) dpsi/dphi = ugrad(psi)!
        // So u_total = u_chi - vgrad(psi), v_total = v_chi + ugrad(psi)!
        // Check: out_u currently holds u_chi, out_v holds v_chi, u_psi holds ugrad(psi), v_psi holds vgrad(psi):
        // Thus u = out_u - v_psi, v = out_v + u_psi!
        if out_u.len() >= 131_072 {
            out_u
                .par_iter_mut()
                .zip(v_psi.par_iter())
                .for_each(|(u, &vp)| *u = *u - vp);
            out_v
                .par_iter_mut()
                .zip(u_psi.par_iter())
                .for_each(|(v, &up)| *v = *v + up);
        } else {
            for (u, &vp) in out_u.iter_mut().zip(v_psi.iter()) {
                *u = *u - vp;
            }
            for (v, &up) in out_v.iter_mut().zip(u_psi.iter()) {
                *v = *v + up;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn getgrad_batch<T: TypedReducedBasis>(
        &self,
        spec_a: &ndarray::ArrayView2<'_, Complex<T>>,
        spec_b: Option<&ndarray::ArrayView2<'_, Complex<T>>>,
        out_au: &mut [T],
        out_av: &mut [T],
        mut out_bu: Option<&mut [T]>,
        mut out_bv: Option<&mut [T]>,
        ntrunc: usize,
        south_to_north: bool,
    ) -> Result<(), String> {
        let nt = spec_a.shape()[0];
        if nt == 0 {
            return Ok(());
        }
        let rb = self.basis::<T>();
        let basis_data = rb.ensure_dbasis(ntrunc)?;
        let dp_even = basis_data.dp_even.as_ref().expect("dp_even initialized");
        let dp_odd = basis_data.dp_odd.as_ref().expect("dp_odd initialized");
        let imid = rb.imid;
        let imm1 = rb.imm1;
        let pair_count = rb.pair_count;
        let is_odd_nlat = self.nlat % 2 == 1;
        let mmax = ntrunc + 1;
        let has_pair = spec_b.is_some();
        let nfields_per_k = if has_pair { 2 } else { 1 };
        let inv_r = 1.0 / self.rsphere;

        let fft_cache = self.fft::<T>();
        let plans: Vec<_> = (0..imid)
            .map(|j| fft_cache.get_or_plan(self.pl[j]))
            .collect();
        let full_plans: Vec<_> = if self.symmetric_pl {
            Vec::new()
        } else {
            (0..self.nlat)
                .map(|j| fft_cache.get_or_plan(self.pl[j]))
                .collect()
        };

        if self.symmetric_pl {
            let per_field_bytes = (2 * self.npoints + 4 * mmax * imid * 2) * T::BYTES * nfields_per_k;
            let max_tile = compute_tile_size(per_field_bytes, nt);

            let mut sp_e = vec![T::ZERO; mmax * imid * 2 * max_tile * nfields_per_k];
            let mut sp_o = vec![T::ZERO; mmax * imm1 * 2 * max_tile * nfields_per_k];
            let mut sdp_e = vec![T::ZERO; mmax * imm1 * 2 * max_tile * nfields_per_k];
            let mut sdp_o = vec![T::ZERO; mmax * imid * 2 * max_tile * nfields_per_k];
            let max_rows = basis_data.p_even.rows_for_m(0, ntrunc).max(1);
            let mut spec_buf_m = vec![T::ZERO; mmax * max_rows * 2 * max_tile * nfields_per_k];

            let mut start = 0usize;
            while start < nt {
                let tile = (nt - start).min(max_tile);
                let eff_t = tile * nfields_per_k;
                let cols = 2 * eff_t;
                let imid_stride = imid * cols;
                let imm1_stride = imm1 * cols;
                let spec_m_stride = max_rows * cols;
                let par_fields = eff_t >= 2 && eff_t * self.npoints >= PAR_WORK_THRESHOLD;
                let par_m = mmax * imid * cols >= PAR_WORK_THRESHOLD;

                // n - m EVEN: sp_e = p_even^T * spec_even, sdp_e = dp_even^T * spec_even
                let do_grad_even = |(m, ((spe_m, sdpe_m), spec_m)): (
                    usize,
                    ((&mut [T], &mut [T]), &mut [T]),
                )| {
                    let ne = basis_data.p_even.rows_for_m(m, ntrunc);
                    if ne == 0 {
                        spe_m.fill(T::ZERO);
                        sdpe_m.fill(T::ZERO);
                        return;
                    }
                    let m_base = spectral_offset(m, ntrunc);
                    for r in 0..ne {
                        let nm = m_base + 2 * r;
                        let dst = &mut spec_m[r * cols..(r + 1) * cols];
                        for tk in 0..tile {
                            let ca = spec_a[[start + tk, nm]];
                            dst[2 * tk] = ca.re;
                            dst[2 * tk + 1] = if m == 0 { T::ZERO } else { ca.im };
                            if let Some(sb) = spec_b {
                                let cb = sb[[start + tk, nm]];
                                let idx = tile + tk;
                                dst[2 * idx] = cb.re;
                                dst[2 * idx + 1] = if m == 0 { T::ZERO } else { cb.im };
                            }
                        }
                    }
                    let pe_blk = basis_data.p_even.block_for_m(m, ntrunc);
                    T::faer_gemm(spe_m, imid, cols, Accum::Replace, pe_blk, imid, ne, false, spec_m, ne, cols);
                    if imm1 > 0 {
                        let dpe_blk = dp_even.block_for_m(m, ntrunc);
                        T::faer_gemm(sdpe_m, imm1, cols, Accum::Replace, dpe_blk, imm1, ne, false, spec_m, ne, cols);
                    }
                };
                if par_m {
                    sp_e[..mmax * imid_stride]
                        .par_chunks_mut(imid_stride)
                        .zip(sdp_e[..mmax * imm1_stride].par_chunks_mut(imm1_stride))
                        .zip(spec_buf_m[..mmax * spec_m_stride].par_chunks_mut(spec_m_stride))
                        .enumerate()
                        .for_each(do_grad_even);
                } else {
                    sp_e[..mmax * imid_stride]
                        .chunks_mut(imid_stride)
                        .zip(sdp_e[..mmax * imm1_stride].chunks_mut(imm1_stride))
                        .zip(spec_buf_m[..mmax * spec_m_stride].chunks_mut(spec_m_stride))
                        .enumerate()
                        .for_each(do_grad_even);
                }

                // n - m ODD: sp_o = p_odd^T * spec_odd, sdp_o = dp_odd^T * spec_odd
                let do_grad_odd = |(m, ((spo_m, sdpo_m), spec_m)): (
                    usize,
                    ((&mut [T], &mut [T]), &mut [T]),
                )| {
                    let no = basis_data.p_odd.rows_for_m(m, ntrunc);
                    if no == 0 {
                        spo_m.fill(T::ZERO);
                        sdpo_m.fill(T::ZERO);
                        return;
                    }
                    let m_base = spectral_offset(m, ntrunc);
                    for r in 0..no {
                        let nm = m_base + 1 + 2 * r;
                        let dst = &mut spec_m[r * cols..(r + 1) * cols];
                        for tk in 0..tile {
                            let ca = spec_a[[start + tk, nm]];
                            dst[2 * tk] = ca.re;
                            dst[2 * tk + 1] = if m == 0 { T::ZERO } else { ca.im };
                            if let Some(sb) = spec_b {
                                let cb = sb[[start + tk, nm]];
                                let idx = tile + tk;
                                dst[2 * idx] = cb.re;
                                dst[2 * idx + 1] = if m == 0 { T::ZERO } else { cb.im };
                            }
                        }
                    }
                    if imm1 > 0 {
                        let po_blk = basis_data.p_odd.block_for_m(m, ntrunc);
                        T::faer_gemm(spo_m, imm1, cols, Accum::Replace, po_blk, imm1, no, false, spec_m, no, cols);
                    }
                    let dpo_blk = dp_odd.block_for_m(m, ntrunc);
                    T::faer_gemm(sdpo_m, imid, cols, Accum::Replace, dpo_blk, imid, no, false, spec_m, no, cols);
                };
                if par_m {
                    sp_o[..mmax * imm1_stride]
                        .par_chunks_mut(imm1_stride)
                        .zip(sdp_o[..mmax * imid_stride].par_chunks_mut(imid_stride))
                        .zip(spec_buf_m[..mmax * spec_m_stride].par_chunks_mut(spec_m_stride))
                        .enumerate()
                        .for_each(do_grad_odd);
                } else {
                    sp_o[..mmax * imm1_stride]
                        .chunks_mut(imm1_stride)
                        .zip(sdp_o[..mmax * imid_stride].chunks_mut(imid_stride))
                        .zip(spec_buf_m[..mmax * spec_m_stride].chunks_mut(spec_m_stride))
                        .enumerate()
                        .for_each(do_grad_odd);
                }

                let au_ptr = out_au[start * self.npoints..(start + tile) * self.npoints].as_mut_ptr() as usize;
                let av_ptr = out_av[start * self.npoints..(start + tile) * self.npoints].as_mut_ptr() as usize;
                let bu_ptr = out_bu
                    .as_deref_mut()
                    .map(|sl| sl[start * self.npoints..(start + tile) * self.npoints].as_mut_ptr() as usize);
                let bv_ptr = out_bv
                    .as_deref_mut()
                    .map(|sl| sl[start * self.npoints..(start + tile) * self.npoints].as_mut_ptr() as usize);

                let do_grad_c2r = |ek: usize| {
                    let is_b = ek >= tile;
                    let tk = if is_b { ek - tile } else { ek };
                    let u_ptr = if is_b {
                        (bu_ptr.unwrap() as *mut T).wrapping_add(tk * self.npoints)
                    } else {
                        (au_ptr as *mut T).wrapping_add(tk * self.npoints)
                    };
                    let v_ptr = if is_b {
                        (bv_ptr.unwrap() as *mut T).wrapping_add(tk * self.npoints)
                    } else {
                        (av_ptr as *mut T).wrapping_add(tk * self.npoints)
                    };

                    let mut u_n = vec![T::complex_zero(); rb.max_nlon / 2 + 1];
                    let mut v_n = vec![T::complex_zero(); rb.max_nlon / 2 + 1];
                    let mut u_s = vec![T::complex_zero(); rb.max_nlon / 2 + 1];
                    let mut v_s = vec![T::complex_zero(); rb.max_nlon / 2 + 1];
                    let mut scratch = Vec::new();

                    for j in 0..pair_count {
                        let row_nlon = self.pl[j];
                        let nfreq = row_nlon / 2 + 1;
                        let mlimit = ntrunc.min(row_nlon / 2);
                        let inv_r_sin = inv_r / rb.sin_theta_f64[j];
                        let t_inv_r = T::from_real(inv_r);

                        u_n[..nfreq].fill(T::complex_zero());
                        v_n[..nfreq].fill(T::complex_zero());
                        u_s[..nfreq].fill(T::complex_zero());
                        v_s[..nfreq].fill(T::complex_zero());

                        {
                            let dpe_0 = sdp_e[j * cols + 2 * ek];
                            let dpo_0 = sdp_o[j * cols + 2 * ek];
                            v_n[0] = Complex::new(-t_inv_r * (dpe_0 + dpo_0), T::ZERO);
                            v_s[0] = Complex::new(-t_inv_r * (dpo_0 - dpe_0), T::ZERO);
                        }

                        for m in 1..=mlimit {
                            let pe_idx = m * imid_stride + j * cols + 2 * ek;
                            let po_idx = m * imm1_stride + j * cols + 2 * ek;
                            let spe_re = sp_e[pe_idx];
                            let spe_im = sp_e[pe_idx + 1];
                            let spo_re = sp_o[po_idx];
                            let spo_im = sp_o[po_idx + 1];
                            let dpe_re = sdp_e[po_idx];
                            let dpe_im = sdp_e[po_idx + 1];
                            let dpo_re = sdp_o[pe_idx];
                            let dpo_im = sdp_o[pe_idx + 1];

                            let m_sc = T::from_real((m as f64) * inv_r_sin);
                            let p_n_re = spe_re + spo_re;
                            let p_n_im = spe_im + spo_im;
                            let p_s_re = spe_re - spo_re;
                            let p_s_im = spe_im - spo_im;

                            let dp_n_re = dpe_re + dpo_re;
                            let dp_n_im = dpe_im + dpo_im;
                            let dp_s_re = dpo_re - dpe_re;
                            let dp_s_im = dpo_im - dpe_im;

                            u_n[m] = Complex::new(-m_sc * p_n_im, m_sc * p_n_re);
                            v_n[m] = Complex::new(-t_inv_r * dp_n_re, -t_inv_r * dp_n_im);
                            u_s[m] = Complex::new(-m_sc * p_s_im, m_sc * p_s_re);
                            v_s[m] = Complex::new(-t_inv_r * dp_s_re, -t_inv_r * dp_s_im);
                        }

                        if row_nlon % 2 == 0 {
                            let nyq = row_nlon / 2;
                            u_n[nyq].im = T::ZERO;
                            v_n[nyq].im = T::ZERO;
                            u_s[nyq].im = T::ZERO;
                            v_s[nyq].im = T::ZERO;
                        }

                        let c2r = &plans[j].c2r;
                        let slen = c2r.get_scratch_len();
                        if scratch.len() < slen {
                            scratch.resize(slen, T::complex_zero());
                        }
                        let (off_n, _) = self.row_range(j, south_to_north);
                        let (off_s, _) = self.row_range(self.nlat - 1 - j, south_to_north);
                        unsafe {
                            let dun = std::slice::from_raw_parts_mut(u_ptr.add(off_n), row_nlon);
                            let dvn = std::slice::from_raw_parts_mut(v_ptr.add(off_n), row_nlon);
                            let dus = std::slice::from_raw_parts_mut(u_ptr.add(off_s), row_nlon);
                            let dvs = std::slice::from_raw_parts_mut(v_ptr.add(off_s), row_nlon);
                            c2r.process_with_scratch(&mut u_n[..nfreq], dun, &mut scratch[..slen])
                                .expect("c2r un");
                            c2r.process_with_scratch(&mut v_n[..nfreq], dvn, &mut scratch[..slen])
                                .expect("c2r vn");
                            c2r.process_with_scratch(&mut u_s[..nfreq], dus, &mut scratch[..slen])
                                .expect("c2r us");
                            c2r.process_with_scratch(&mut v_s[..nfreq], dvs, &mut scratch[..slen])
                                .expect("c2r vs");
                        }
                    }

                    if is_odd_nlat {
                        let j = imid - 1;
                        let row_nlon = self.pl[j];
                        let nfreq = row_nlon / 2 + 1;
                        let mlimit = ntrunc.min(row_nlon / 2);
                        let inv_r_sin = inv_r / rb.sin_theta_f64[j];
                        let t_inv_r = T::from_real(inv_r);

                        u_n[..nfreq].fill(T::complex_zero());
                        v_n[..nfreq].fill(T::complex_zero());
                        v_n[0] = Complex::new(-t_inv_r * sdp_o[j * cols + 2 * ek], T::ZERO);

                        for m in 1..=mlimit {
                            let pe_idx = m * imid_stride + j * cols + 2 * ek;
                            let spe_re = sp_e[pe_idx];
                            let spe_im = sp_e[pe_idx + 1];
                            let dpo_re = sdp_o[pe_idx];
                            let dpo_im = sdp_o[pe_idx + 1];
                            let m_sc = T::from_real((m as f64) * inv_r_sin);
                            u_n[m] = Complex::new(-m_sc * spe_im, m_sc * spe_re);
                            v_n[m] = Complex::new(-t_inv_r * dpo_re, -t_inv_r * dpo_im);
                        }
                        if row_nlon % 2 == 0 {
                            let nyq = row_nlon / 2;
                            u_n[nyq].im = T::ZERO;
                            v_n[nyq].im = T::ZERO;
                        }
                        let c2r = &plans[j].c2r;
                        let slen = c2r.get_scratch_len();
                        if scratch.len() < slen {
                            scratch.resize(slen, T::complex_zero());
                        }
                        let (off_eq, _) = self.row_range(j, south_to_north);
                        unsafe {
                            let dueq = std::slice::from_raw_parts_mut(u_ptr.add(off_eq), row_nlon);
                            let dveq = std::slice::from_raw_parts_mut(v_ptr.add(off_eq), row_nlon);
                            c2r.process_with_scratch(&mut u_n[..nfreq], dueq, &mut scratch[..slen])
                                .expect("c2r ueq");
                            c2r.process_with_scratch(&mut v_n[..nfreq], dveq, &mut scratch[..slen])
                                .expect("c2r veq");
                        }
                    }
                };
                if par_fields {
                    (0..eff_t).into_par_iter().for_each(do_grad_c2r);
                } else {
                    (0..eff_t).for_each(do_grad_c2r);
                }

                start += tile;
            }
        } else {
            // Non-symmetric fallback
            let nlat = self.nlat;
            let run_one = |spec: &ndarray::ArrayView2<'_, Complex<T>>, u_out: &mut [T], v_out: &mut [T]| {
                u_out[..nt * self.npoints]
                    .par_chunks_mut(self.npoints)
                    .zip(v_out[..nt * self.npoints].par_chunks_mut(self.npoints))
                    .enumerate()
                    .for_each(|(k, (uo, vo))| {
                        let mut u_c = vec![T::complex_zero(); rb.max_nlon / 2 + 1];
                        let mut v_c = vec![T::complex_zero(); rb.max_nlon / 2 + 1];
                        let mut scratch = Vec::new();
                        for j in 0..nlat {
                            let row_nlon = self.pl[j];
                            let nfreq = row_nlon / 2 + 1;
                            let mlimit = ntrunc.min(row_nlon / 2);
                            let inv_r_sin = inv_r / rb.sin_theta_f64[j];
                            let is_south = j >= imid;
                            let j_nh = if is_south { nlat - 1 - j } else { j };
                            u_c[..nfreq].fill(T::complex_zero());
                            v_c[..nfreq].fill(T::complex_zero());

                            for m in 0..=mlimit {
                                let m_base = spectral_offset(m, ntrunc);
                                let ne = basis_data.p_even.rows_for_m(m, ntrunc);
                                let no = basis_data.p_odd.rows_for_m(m, ntrunc);
                                let pe = basis_data.p_even.block_for_m(m, ntrunc);
                                let po = basis_data.p_odd.block_for_m(m, ntrunc);
                                let dpe = dp_even.block_for_m(m, ntrunc);
                                let dpo = dp_odd.block_for_m(m, ntrunc);
                                let mut p_re = 0.0_f64;
                                let mut p_im = 0.0_f64;
                                let mut dp_re = 0.0_f64;
                                let mut dp_im = 0.0_f64;
                                for r in 0..ne {
                                    let p = pe[r * imid + j_nh].to_f64();
                                    let dp = if j_nh < imm1 {
                                        let v = dpe[r * imm1 + j_nh].to_f64();
                                        if is_south { -v } else { v }
                                    } else {
                                        0.0
                                    };
                                    let c = spec[[k, m_base + 2 * r]];
                                    p_re += c.re.to_f64() * p;
                                    p_im += c.im.to_f64() * p;
                                    dp_re += c.re.to_f64() * dp;
                                    dp_im += c.im.to_f64() * dp;
                                }
                                for r in 0..no {
                                    let p = if j_nh < imm1 {
                                        let v = po[r * imm1 + j_nh].to_f64();
                                        if is_south { -v } else { v }
                                    } else {
                                        0.0
                                    };
                                    let dp = dpo[r * imid + j_nh].to_f64();
                                    let c = spec[[k, m_base + 1 + 2 * r]];
                                    p_re += c.re.to_f64() * p;
                                    p_im += c.im.to_f64() * p;
                                    dp_re += c.re.to_f64() * dp;
                                    dp_im += c.im.to_f64() * dp;
                                }
                                if m == 0 {
                                    v_c[0] = Complex::new(T::from_real(-inv_r * dp_re), T::ZERO);
                                } else {
                                    let m_sc = (m as f64) * inv_r_sin;
                                    u_c[m] = Complex::new(T::from_real(-m_sc * p_im), T::from_real(m_sc * p_re));
                                    v_c[m] = Complex::new(T::from_real(-inv_r * dp_re), T::from_real(-inv_r * dp_im));
                                }
                            }
                            if row_nlon % 2 == 0 {
                                let nyq = row_nlon / 2;
                                u_c[nyq].im = T::ZERO;
                                v_c[nyq].im = T::ZERO;
                            }
                            let c2r = &full_plans[j].c2r;
                            let slen = c2r.get_scratch_len();
                            if scratch.len() < slen {
                                scratch.resize(slen, T::complex_zero());
                            }
                            let (off, _) = self.row_range(j, south_to_north);
                            c2r.process_with_scratch(&mut u_c[..nfreq], &mut uo[off..off + row_nlon], &mut scratch[..slen])
                                .expect("c2r u asym");
                            c2r.process_with_scratch(&mut v_c[..nfreq], &mut vo[off..off + row_nlon], &mut scratch[..slen])
                                .expect("c2r v asym");
                        }
                    });
            };
            run_one(spec_a, out_au, out_av);
            if let (Some(sb), Some(bu), Some(bv)) = (spec_b, out_bu.as_deref_mut(), out_bv.as_deref_mut()) {
                run_one(sb, bu, bv);
            }
        }

        Ok(())
    }
}

fn gemm_subcols<T: FaerGemmFloat>(
    dst: &mut [T],
    nrows: usize,
    ncols: usize,
    accum: Accum,
    p_blk: &[T],
    stride: usize,
    rhs: &[T],
    j0: usize,
) {
    T::faer_gemm_subcols(dst, nrows, ncols, accum, p_blk, stride, rhs, j0);
}

#[allow(clippy::too_many_arguments)]
fn scatter_reduced_vec<T: crate::precision::ShtFloat>(
    spec_buf_m: &[T],
    spec_m_stride: usize,
    cols: usize,
    mmax: usize,
    ntrunc: usize,
    nmdim: usize,
    start: usize,
    tile: usize,
    is_even: bool,
    ptable: &spharm_math::ParityTable<T>,
    mode: VectorMode,
    want_div: bool,
    want_vrt: bool,
    mut out_div: Option<&mut [Complex<T>]>,
    mut out_vrt: Option<&mut [Complex<T>]>,
) {
    let r_off = if is_even { 0 } else { 1 };
    for m in 0..mmax {
        let nrows = ptable.rows_for_m(m, ntrunc);
        if nrows == 0 {
            continue;
        }
        let m_base = spectral_offset(m, ntrunc);
        let spec_m = &spec_buf_m[m * spec_m_stride..];
        for r in 0..nrows {
            let nm = m_base + r_off + 2 * r;
            let row = &spec_m[r * cols..(r + 1) * cols];
            match mode {
                VectorMode::VrtDiv => {
                    if let (Some(div), Some(vrt)) = (out_div.as_deref_mut(), out_vrt.as_deref_mut()) {
                        for tk in 0..tile {
                            let idx = (start + tk) * nmdim + nm;
                            div[idx] = Complex::new(row[4 * tk], if m == 0 { T::ZERO } else { row[4 * tk + 1] });
                            vrt[idx] = Complex::new(row[4 * tk + 2], if m == 0 { T::ZERO } else { row[4 * tk + 3] });
                        }
                    }
                }
                VectorMode::DivOnly => {
                    if want_div {
                        if let Some(div) = out_div.as_deref_mut() {
                            for tk in 0..tile {
                                let idx = (start + tk) * nmdim + nm;
                                div[idx] = Complex::new(row[2 * tk], if m == 0 { T::ZERO } else { row[2 * tk + 1] });
                            }
                        }
                    }
                }
                VectorMode::VrtOnly => {
                    if want_vrt {
                        if let Some(vrt) = out_vrt.as_deref_mut() {
                            for tk in 0..tile {
                                let idx = (start + tk) * nmdim + nm;
                                vrt[idx] = Complex::new(row[2 * tk], if m == 0 { T::ZERO } else { row[2 * tk + 1] });
                            }
                        }
                    }
                }
            }
        }
    }
}

pub trait TypedReducedBasis: FaerGemmFloat {
    fn get_basis(engine: &ReducedEngine) -> &ReducedRowBasis<Self>;
    fn get_fft(engine: &ReducedEngine) -> &FftCache<Self>;
}

impl TypedReducedBasis for f32 {
    #[inline]
    fn get_basis(engine: &ReducedEngine) -> &ReducedRowBasis<Self> {
        engine
            .basis_f32
            .get_or_init(|| ReducedRowBasis::<f32>::new(&engine.pl).expect("validated pl"))
    }

    #[inline]
    fn get_fft(engine: &ReducedEngine) -> &FftCache<Self> {
        &engine.fft_f32
    }
}

impl TypedReducedBasis for f64 {
    #[inline]
    fn get_basis(engine: &ReducedEngine) -> &ReducedRowBasis<Self> {
        engine
            .basis_f64
            .get_or_init(|| ReducedRowBasis::<f64>::new(&engine.pl).expect("validated pl"))
    }

    #[inline]
    fn get_fft(engine: &ReducedEngine) -> &FftCache<Self> {
        &engine.fft_f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_asymmetric_odd_nlat_scalar_and_vector_roundtrip() {
        let pl = vec![16, 20, 24, 22, 18];
        let engine = ReducedEngine::new(pl, 6.3712e6, "stored".to_string()).unwrap();
        let ntrunc = 3;
        let nmdim = nmdim_from_ntrunc(ntrunc);

        let mut spec = Array2::from_elem((1, nmdim), Complex::new(0.0_f64, 0.0));
        spec[[0, 0]] = Complex::new(1.5, 0.0);
        spec[[0, 1]] = Complex::new(-0.7, 0.0);
        spec[[0, 2]] = Complex::new(0.4, 0.0);
        spec[[0, 5]] = Complex::new(0.3, -0.2);

        let mut grid = vec![0.0_f64; engine.npoints];
        engine
            .spectogrd(&spec.view(), &mut grid, ntrunc, false)
            .unwrap();

        let grid_v = Array2::from_shape_vec((1, engine.npoints), grid.clone()).unwrap();
        let mut spec_back = vec![Complex::new(0.0_f64, 0.0); nmdim];
        engine
            .grdtospec(&grid_v.view(), &mut spec_back, ntrunc, false)
            .unwrap();
        for i in 0..nmdim {
            assert!((spec[[0, i]] - spec_back[i]).norm() < 1e-12);
        }

        let mut vrt = spec.clone();
        let mut div = spec.clone();
        vrt[[0, 0]] = Complex::new(0.0, 0.0);
        div[[0, 0]] = Complex::new(0.0, 0.0);
        let mut u = vec![0.0_f64; engine.npoints];
        let mut v = vec![0.0_f64; engine.npoints];
        engine
            .getuv(&vrt.view(), &div.view(), &mut u, &mut v, ntrunc, false)
            .unwrap();
        let uv = Array2::from_shape_vec((1, engine.npoints), u).unwrap();
        let vv = Array2::from_shape_vec((1, engine.npoints), v).unwrap();
        let mut vrt_b = vec![Complex::new(0.0_f64, 0.0); nmdim];
        let mut div_b = vec![Complex::new(0.0_f64, 0.0); nmdim];
        engine
            .getvrtdivspec(
                &uv.view(),
                &vv.view(),
                Some(&mut vrt_b),
                Some(&mut div_b),
                ntrunc,
                false,
                VectorMode::VrtDiv,
            )
            .unwrap();
        for i in 0..nmdim {
            assert!((vrt[[0, i]] - vrt_b[i]).norm() < 1e-12);
            assert!((div[[0, i]] - div_b[i]).norm() < 1e-12);
        }
    }
}
