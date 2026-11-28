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

const MAX_TILE_SIZE: usize = 32;
const MAX_SCRATCH_BYTES: usize = 64 * 1024 * 1024;
pub const PAR_WORK_THRESHOLD: usize = 32_768;

pub trait FaerGemmFloat: ShtFloat + SpharmFloat {
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

impl_faer_gemm!(f32);
impl_faer_gemm!(f64);

#[inline]
pub fn compute_tile_size(per_field_bytes: usize, nt: usize) -> usize {
    if nt == 0 {
        return 0;
    }
    let cap = (MAX_SCRATCH_BYTES / per_field_bytes.max(1)).clamp(1, MAX_TILE_SIZE);
    cap.min(nt)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorMode {
    VrtDiv,
    VrtOnly,
    DivOnly,
}

pub struct FullGridEngine {
    pub nlon: usize,
    pub nlat: usize,
    pub rsphere: f64,
    pub gridtype: GridType,
    pub legfunc: String,
    tables_f32: OnceLock<ParityHarmonicTables<f32>>,
    tables_f64: OnceLock<ParityHarmonicTables<f64>>,
    fft_f32: FftCache<f32>,
    fft_f64: FftCache<f64>,
}

impl FullGridEngine {
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

    #[inline]
    fn tables<T: TypedTables>(&self) -> &ParityHarmonicTables<T> {
        T::get_tables(self)
    }

    #[inline]
    fn fft<T: TypedTables>(&self) -> &FftCache<T> {
        T::get_fft(self)
    }

    pub fn grdtospec<T: TypedTables>(
        &self,
        grid: &ndarray::ArrayView3<'_, T>,
        out_spec: &mut [Complex<T>],
        ntrunc: usize,
        south_to_north: bool,
    ) {
        let nt = grid.shape()[0];
        let nmdim = crate::spectral::nmdim_from_ntrunc(ntrunc);
        if nt == 0 {
            return;
        }
        out_spec[..nt * nmdim].fill(T::complex_zero());

        let tables = self.tables::<T>();
        let sc: &ScalarTables<T> = tables.ensure_scalar();
        let imid = tables.imid;
        let imm1 = tables.imm1;
        let mmax = tables.mmax_scalar.min(ntrunc + 1);
        let nlat = self.nlat;
        let nlon = self.nlon;
        let nfreq = nlon / 2 + 1;
        let is_odd_nlat = nlat % 2 == 1;
        let has_nyquist = nlon % 2 == 0 && (nlon / 2) < mmax;
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
        let mmax = tables.mmax_scalar.min(ntrunc + 1);
        let nlat = self.nlat;
        let nlon = self.nlon;
        let nfreq = nlon / 2 + 1;
        let is_odd_nlat = nlat % 2 == 1;
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

            // Even parity: pack spec_m then GEMM P_even^T * spec_m -> f_even
            let do_even_m = |(m, (fe_m, spec_m)): (usize, (&mut [T], &mut [T]))| {
                let ne = sc.p_even.rows_for_m(m, ntrunc);
                if ne == 0 || imid == 0 {
                    fe_m.fill(T::ZERO);
                    return;
                }
                let m_base = spectral_offset(m, ntrunc);
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
        let mmax = tables.mmax_vector.min(ntrunc + 1);
        let nlat = self.nlat;
        let nlon = self.nlon;
        let nfreq = nlon / 2 + 1;
        let is_odd_nlat = nlat % 2 == 1;
        let inv_nlon = T::from_real(1.0 / (nlon as f64));
        let inv_r = 1.0 / self.rsphere;

        // Precompute sqrt(n*(n+1))/rsphere scaling factors
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
                if n == 0 || (n == self.nlat - 1 && m % 2 == 0) {
                    continue;
                }
                let nm = m_base + (n - m);
                let nf = n_factors[n];
                let src_row = &spec_m[r * cols..(r + 1) * cols];
                match mode {
                    VectorMode::VrtDiv => {
                        if let (Some(vrt), Some(div)) = (out_vrt.as_deref_mut(), out_div.as_deref_mut()) {
                            for tk in 0..tile {
                                let out_idx = (start + tk) * nmdim + nm;
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
                    let skip = n == 0 || (n == nlat - 1 && m % 2 == 0);
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

pub trait TypedTables: FaerGemmFloat {
    fn get_tables(engine: &FullGridEngine) -> &ParityHarmonicTables<Self>;
    fn get_fft(engine: &FullGridEngine) -> &FftCache<Self>;
}

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
