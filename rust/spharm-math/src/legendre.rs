//! Associated Legendre function computation in `f64` with SPHEREPACK normalization:
//! $\int_0^\pi (\bar{P}_n^m(\theta))^2 \sin\theta\, d\theta = 1$, $\bar{P}_0^0 = \sqrt{2}/2$.

use std::f64::consts::PI;

const SC10: f64 = 1024.0;
const SC20: f64 = SC10 * SC10;
const SC40: f64 = SC20 * SC20;

/// Scale `val` by $2^{\text{exp}}$ without overflowing an intermediate `2.0_f64.powi(-exp)`.
#[inline]
fn scale_pow2(mut val: f64, mut exp: i32) -> f64 {
    while exp > 512 {
        val *= 2.0_f64.powi(512);
        exp -= 512;
    }
    while exp < -512 {
        val *= 2.0_f64.powi(-512);
        exp += 512;
    }
    val * 2.0_f64.powi(exp)
}

/// Compute Fourier coefficients `cp[0..n/2+1]` for $\bar{P}_n^m(\cos\theta)$,
/// matching SPHEREPACK `dnlfk`.
pub(crate) fn dnlfk(m: isize, n: usize, cp: &mut [f64]) {
    let ma = m.unsigned_abs();
    cp[0] = 0.0;
    if ma > n {
        return;
    }

    match n {
        0 => {
            cp[0] = 2.0_f64.sqrt();
            return;
        }
        1 => {
            if ma == 0 {
                cp[0] = 1.5_f64.sqrt();
            } else {
                cp[0] = 0.75_f64.sqrt();
                if m == -1 {
                    cp[0] = -cp[0];
                }
            }
            return;
        }
        _ => {}
    }

    let (nmms2, mut fnum, mut fnmh, pm1) = if (n + ma) % 2 == 0 {
        ((n - ma) / 2, (n + ma + 1) as f64, (n - ma + 1) as f64, 1.0_f64)
    } else {
        (
            (n - ma - 1) / 2,
            (n + ma + 2) as f64,
            (n - ma + 2) as f64,
            -1.0_f64,
        )
    };

    let mut t1 = 1.0_f64 / SC20;
    let mut nex: i32 = 20;
    let mut fden = 2.0_f64;

    for _ in 0..nmms2 {
        t1 = fnum * t1 / fden;
        if t1 > SC20 {
            t1 /= SC40;
            nex += 40;
        }
        fnum += 2.0;
        fden += 2.0;
    }

    t1 = scale_pow2(t1, nex + 1 - (n as i32));
    if (ma / 2) % 2 != 0 {
        t1 = -t1;
    }

    let mut t2 = 1.0_f64;
    for _ in 0..ma {
        t2 = fnmh * t2 / (fnmh + pm1);
        fnmh += 2.0;
    }

    let cp2 = t1 * (((n as f64) + 0.5) * t2).sqrt();
    let fnnp1 = (n * (n + 1)) as f64;
    let fnmsq = fnnp1 - 2.0 * ((ma * ma) as f64);

    let mut l = (n + 1) / 2;
    if n % 2 == 0 && ma % 2 == 0 {
        l += 1;
    }
    cp[l - 1] = cp2;
    if m < 0 && ma % 2 != 0 {
        cp[l - 1] = -cp[l - 1];
    }
    if l <= 1 {
        return;
    }

    let mut fk = n as f64;
    let a1 = (fk - 2.0) * (fk - 1.0) - fnnp1;
    let b1 = 2.0 * (fk * fk - fnmsq);
    cp[l - 2] = b1 * cp[l - 1] / a1;

    while l > 2 {
        l -= 1;
        fk -= 2.0;
        let a1 = (fk - 2.0) * (fk - 1.0) - fnnp1;
        let b1 = -2.0 * (fk * fk - fnmsq);
        let c1 = (fk + 1.0) * (fk + 2.0) - fnnp1;
        cp[l - 1 - 1] = -(b1 * cp[l - 1] + c1 * cp[l]) / a1;
    }
}

/// Evaluate $\bar{P}_n^m(\cos\theta)$ from `dnlfk` coefficients, matching SPHEREPACK `dnlft`.
#[inline]
#[allow(dead_code)]
pub(crate) fn dnlft(m: usize, n: usize, theta: f64, cp: &[f64]) -> f64 {
    if m > n {
        return 0.0;
    }
    if n == 0 {
        return 0.5 * cp[0];
    }

    let cdt = (theta + theta).cos();
    let sdt = (theta + theta).sin();

    if n % 2 == 0 {
        let kdo = n / 2;
        if m % 2 == 0 {
            let mut pb = 0.5 * cp[0];
            let mut cth = cdt;
            let mut sth = sdt;
            for k in 1..=kdo {
                pb += cp[k] * cth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
            pb
        } else {
            let mut pb = 0.0_f64;
            let mut cth = cdt;
            let mut sth = sdt;
            for k in 0..kdo {
                pb += cp[k] * sth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
            pb
        }
    } else {
        let kdo = (n + 1) / 2;
        let mut cth = theta.cos();
        let mut sth = theta.sin();
        let mut pb = 0.0_f64;
        if m % 2 == 0 {
            for k in 0..kdo {
                pb += cp[k] * cth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        } else {
            for k in 0..kdo {
                pb += cp[k] * sth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        }
        pb
    }
}

/// Compute all $\bar{P}_n^m(\theta)$ for $0 \le m \le n_{\text{trunc}}$ and $m \le n \le n_{\text{trunc}}$
/// at a single colatitude `colat_rad` in `f64`, using the unconditionally stable `f64` 3-term Bonnet
/// recurrences for $m=0,1$ and Schwarztrauber's 4-point recurrence for $m \ge 2$ in $O(n_{\text{trunc}}^2)$ time.
///
/// Output ordering matches SPHEREPACK `getlegfunc` / `shaes` triangular packing:
/// `m = 0..=ntrunc` outer, `n = m..=ntrunc` inner, index `m * (2 * ntrunc - m + 3) / 2 + (n - m)`.
pub fn legendre_pbar(ntrunc: usize, colat_rad: f64) -> Result<Vec<f64>, String> {
    if !colat_rad.is_finite() || colat_rad < -1e-12 || colat_rad > PI + 1e-12 {
        return Err(format!(
            "colat_rad must be in [0, pi], got {colat_rad}"
        ));
    }
    let theta = colat_rad.clamp(0.0, PI);
    let nmdim = (ntrunc + 1) * (ntrunc + 2) / 2;
    let mut out = vec![0.0_f64; nmdim];

    let (sin_th, cos_th) = theta.sin_cos();
    // 3 rolling n-levels: nm2 (n-2), nm1 (n-1), curr (n), each of length ntrunc + 1.
    let stride = ntrunc + 1;
    let mut ring = vec![0.0_f64; 3 * stride];

    // n = 0
    let ssqr2 = 0.5_f64.sqrt();
    ring[0] = ssqr2;
    out[0] = ssqr2;
    if ntrunc == 0 {
        return Ok(out);
    }

    // n = 1
    {
        let p0 = 1.5_f64.sqrt() * cos_th;
        let p1 = 0.75_f64.sqrt() * sin_th;
        ring[stride] = p0;
        ring[stride + 1] = p1;
        out[1] = p0;
        out[ntrunc + 1] = p1;
    }

    for n in 2..=ntrunc {
        let nm2_idx = ((n + 1) % 3) * stride;
        let nm1_idx = ((n + 2) % 3) * stride;
        let cur_idx = (n % 3) * stride;

        let nf = n as f64;
        let c_num = 2.0 * nf + 1.0;

        // m = 0 via 3-term Bonnet recurrence
        let a0 = (c_num * (2.0 * nf - 1.0)).sqrt() / nf;
        let b0 = ((nf - 1.0) / nf) * (c_num / (2.0 * nf - 3.0)).sqrt();
        let p0 = a0 * cos_th * ring[nm1_idx] - b0 * ring[nm2_idx];
        ring[cur_idx] = p0;
        out[n] = p0;

        // m = 1 via 3-term Bonnet recurrence
        let denom1 = (nf - 1.0) * (nf + 1.0);
        let a1 = (c_num * (2.0 * nf - 1.0) / denom1).sqrt();
        let b1 = if n >= 3 {
            (c_num * nf * (nf - 2.0) / ((2.0 * nf - 3.0) * denom1)).sqrt()
        } else {
            0.0
        };
        let p1 = a1 * cos_th * ring[nm1_idx + 1] - b1 * ring[nm2_idx + 1];
        ring[cur_idx + 1] = p1;
        out[(2 * ntrunc + 2) / 2 + (n - 1)] = p1;

        // m = 2..=n via Schwarztrauber recurrence
        let c_den = 2.0 * nf - 3.0;
        for m in 2..=n {
            let mf = m as f64;
            let d1 = (mf + nf - 1.0) * (mf + nf);
            let abel = (c_num * (mf + nf - 2.0) * (mf + nf - 3.0) / (c_den * d1)).sqrt();
            let cbel = (((nf - mf + 1.0) * (nf - mf + 2.0)) / d1).sqrt();

            let val = if m < n - 1 {
                let bbel = (c_num * (nf - mf - 1.0) * (nf - mf) / (c_den * d1)).sqrt();
                abel * ring[nm2_idx + m - 2] + bbel * ring[nm2_idx + m]
                    - cbel * ring[cur_idx + m - 2]
            } else {
                abel * ring[nm2_idx + m - 2] - cbel * ring[cur_idx + m - 2]
            };

            ring[cur_idx + m] = val;
            let nm = m * (2 * ntrunc - m + 3) / 2 + (n - m);
            out[nm] = val;
        }
    }

    Ok(out)
}

/// Compute associated Legendre functions $\bar{P}_n^m(\theta)$ in SPHEREPACK `getlegfunc` / `shaes`
/// triangular packing order for `ntrunc < nlat` at colatitude `colat_rad`.
pub fn get_legendre_pbar(
    nlat: usize,
    ntrunc: usize,
    colat_rad: f64,
) -> Result<Vec<f64>, String> {
    if nlat == 0 {
        return Err("nlat must be >= 1".to_string());
    }
    if ntrunc >= nlat {
        return Err(format!(
            "ntrunc must be <= nlat - 1 ({}), got {ntrunc}",
            nlat - 1
        ));
    }
    legendre_pbar(ntrunc, colat_rad)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bonnet_matches_dnlft_and_addition_theorem_high_degree() {
        let theta = (89.5_f64).to_radians();
        let ntrunc = 319usize;
        let pbar = legendre_pbar(ntrunc, theta).expect("legendre_pbar");

        // Compare m=0,1 at n=150, 319 against reference dnlfk/dnlft
        let mut cp = vec![0.0_f64; ntrunc / 2 + 2];
        for n in [2usize, 10, 150, 319] {
            for m in [0usize, 1] {
                dnlfk(m as isize, n, &mut cp);
                let ref_val = dnlft(m, n, theta, &cp);
                let idx = m * (2 * ntrunc - m + 3) / 2 + (n - m);
                assert!(
                    (pbar[idx] - ref_val).abs() < 1e-13,
                    "n={n} m={m}: got {} expected {ref_val}",
                    pbar[idx]
                );
            }
        }

        // Verify Addition Theorem: sum_{m=0}^n pbar(n, m)^2 * (1 if m==0 else 2) == (2n+1)/2
        let pbar_f480 = legendre_pbar(959, (89.5_f64).to_radians()).expect("f480");
        for &(deg, nt_cap, arr) in &[(319usize, 319usize, &pbar), (959usize, 959usize, &pbar_f480)] {
            let mut sum = 0.0_f64;
            for m in 0..=deg {
                let idx = m * (2 * nt_cap - m + 3) / 2 + (deg - m);
                let w = if m == 0 { 1.0 } else { 2.0 };
                sum += w * arr[idx] * arr[idx];
            }
            let exact = 0.5 * ((2 * deg + 1) as f64);
            assert!(
                (sum - exact).abs() < 1e-10,
                "Addition theorem failed at n={deg}: got {sum}, expected {exact}"
            );
        }
    }
}
