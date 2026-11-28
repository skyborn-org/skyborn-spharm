use num_complex::Complex;
use rayon::prelude::*;

use crate::precision::ShtFloat;

#[inline(always)]
pub fn nmdim_from_ntrunc(ntrunc: usize) -> usize {
    (ntrunc + 1) * (ntrunc + 2) / 2
}

#[inline(always)]
pub fn spectral_offset(m: usize, ntrunc: usize) -> usize {
    m * (2 * ntrunc - m + 3) / 2
}

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

fn fill_lap_degree_factors<T: ShtFloat>(ntrunc: usize, rsphere: f64, deg: &mut [T]) {
    let inv_r2 = 1.0 / (rsphere * rsphere);
    for (n, slot) in deg.iter_mut().enumerate().take(ntrunc + 1) {
        let nf = n as f64;
        *slot = T::from_real(-nf * (nf + 1.0) * inv_r2);
    }
}

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

fn expand_degree_factors<T: ShtFloat>(ntrunc: usize, deg: &[T]) -> Vec<T> {
    let nmdim = nmdim_from_ntrunc(ntrunc);
    let mut factors = Vec::with_capacity(nmdim);
    for m in 0..=ntrunc {
        factors.extend_from_slice(&deg[m..=ntrunc]);
    }
    factors
}

pub fn build_lap_factors<T: ShtFloat>(ntrunc: usize, rsphere: f64) -> Vec<T> {
    let mut deg = vec![T::ZERO; ntrunc + 1];
    fill_lap_degree_factors(ntrunc, rsphere, &mut deg);
    expand_degree_factors(ntrunc, &deg)
}

pub fn build_invlap_factors<T: ShtFloat>(ntrunc: usize, rsphere: f64) -> Vec<T> {
    let mut deg = vec![T::ZERO; ntrunc + 1];
    fill_invlap_degree_factors(ntrunc, rsphere, &mut deg);
    expand_degree_factors(ntrunc, &deg)
}

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
    let mut stack_deg = [T::ZERO; 256];
    let mut heap_deg;
    let deg: &mut [T] = if ntrunc < 256 {
        &mut stack_deg[..ntrunc + 1]
    } else {
        heap_deg = vec![T::ZERO; ntrunc + 1];
        &mut heap_deg
    };
    fill_lap_degree_factors(ntrunc, rsphere, deg);
    if nt == 1 {
        apply_degree_factors_1d(src, dst, ntrunc, deg);
        return;
    }
    let factors = expand_degree_factors(ntrunc, deg);
    apply_factors_batch(src, dst, nt, nmdim, &factors);
}

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
    let mut stack_deg = [T::ZERO; 256];
    let mut heap_deg;
    let deg: &mut [T] = if ntrunc < 256 {
        &mut stack_deg[..ntrunc + 1]
    } else {
        heap_deg = vec![T::ZERO; ntrunc + 1];
        &mut heap_deg
    };
    fill_invlap_degree_factors(ntrunc, rsphere, deg);
    if nt == 1 {
        apply_degree_factors_1d(src, dst, ntrunc, deg);
        return;
    }
    let factors = expand_degree_factors(ntrunc, deg);
    apply_factors_batch(src, dst, nt, nmdim, &factors);
}

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
    let mut stack_deg = [T::ZERO; 256];
    let mut heap_deg;
    let deg: &mut [T] = if ntrunc < 256 {
        &mut stack_deg[..ntrunc + 1]
    } else {
        heap_deg = vec![T::ZERO; ntrunc + 1];
        &mut heap_deg
    };
    for n in 0..=ntrunc {
        deg[n] = T::from_real(smooth[n].to_f64());
    }
    if nt == 1 {
        apply_degree_factors_1d(src, dst, ntrunc, deg);
        return;
    }
    let factors = expand_degree_factors(ntrunc, deg);
    apply_factors_batch(src, dst, nt, nmdim, &factors);
}

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

pub fn eval_specintrp<T: ShtFloat, P: ShtFloat>(
    rlon_rad: f64,
    dataspec: &[Complex<T>],
    legfuncs: &[P],
    ntrunc: usize,
) -> f64 {
    let mut offset = 0usize;
    let mut sum_m0 = 0.0_f64;
    for _n in 0..=ntrunc {
        sum_m0 += dataspec[offset].re.to_f64() * legfuncs[offset].to_f64();
        offset += 1;
    }

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
