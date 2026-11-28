//! Associated Legendre function computation in `f64` with SPHEREPACK normalization:
//! $\int_0^\pi (\bar{P}_n^m(\theta))^2 \sin\theta\, d\theta = 1$, $\bar{P}_0^0 = \sqrt{2}/2$.
//! Derived from NCAR SPHEREPACK 3.2 (J. Adams and P. Swarztrauber), see LICENSE.spherepack.
//!
//! Fortran references:
//! - `alf.f`: `alfk` (Fourier coefficients of $\bar{P}_n^m$), `lfpt` (evaluation from those
//!   coefficients), `lfim`/`lfim1` and `lfin`/`lfin1` (four-term recurrence, unpublished notes by P. N.
//!   Swarztrauber).
//! - `sphcom.f`: `dnlfk` and `dnlft`, the double precision versions of `alfk` and `lfpt` (`dnlftd`, the
//!   derivative evaluator, has no Rust counterpart here).
//! - `getlegfunc.f`: `getlegfunc`, which fills the triangular array of all $\bar{P}_n^m$ at one latitude.
//!
//! The normalized associated Legendre functions are (Fortran header of `alfk`, `theta` is colatitude)
//! $$\bar{P}_n^m(\theta) = \sqrt{\frac{(2n+1)(n-m)!}{2(n+m)!}}\,
//! \frac{\sin^m\theta}{2^n n!} \frac{d^{n+m}}{dx^{n+m}} (x^2-1)^n, \quad x = \cos\theta,$$
//! with $\bar{P}_n^m = 0$ for $|m| > n$ and $\bar{P}_n^{-m} = (-1)^m \bar{P}_n^m$.
//!
//! Fortran trigonometric representation used by `dnlfk` / `dnlft` (`cp(k)` are the stored
//! coefficients, 1-based `k`, `th` is colatitude):
//! 1. `n` even, `m` even: $0.5\,cp(1) + \sum_{k=1}^{n/2} cp(k+1)\cos(2k\,th)$
//! 2. `n` even, `m` odd: $\sum_{k=1}^{n/2} cp(k)\sin(2k\,th)$
//! 3. `n` odd, `m` even: $\sum_{k=1}^{(n+1)/2} cp(k)\cos((2k-1)\,th)$
//! 4. `n` odd, `m` odd: $\sum_{k=1}^{(n+1)/2} cp(k)\sin((2k-1)\,th)$
//!
//! Difference from Fortran:
//! - All arithmetic is `f64`. The Fortran `getlegfunc` calls the single precision `alfk`; its source
//!   notes a known historical bug: in IEEE single precision `2.**(n-1-nex)` overflows to infinity for
//!   `n >= 149`, which turns `t1 / Inf` into 0 and silently zeroes high-degree functions (for example
//!   `m = n = 319`). The `f64` code here (with `scale_pow2`) and the Fortran 90 versions avoid this.
//! - `legendre_pbar` replaces the Fortran O(N^2) calls of `alfk` + `lfpt` (an O(N^3) total) by
//!   recurrences that cost O(N^2) in total.
//! - 0-based indexing, no workspace arrays (`wlfim`, `wlfin`), and `Result` instead of status codes.

use std::f64::consts::PI;

/// Scaling constant $2^{10}$ (Fortran `sc10`), the building block of `SC20` and `SC40`.
const SC10: f64 = 1024.0;
/// Scaling constant $2^{20}$ (Fortran `sc20`): the start value of the running product in `dnlfk` is `1/SC20`, and the product is rescaled when it exceeds `SC20`.
const SC20: f64 = SC10 * SC10;
/// Scaling constant $2^{40}$ (Fortran `sc40`): the rescaling factor applied in `dnlfk` when the running product exceeds `SC20` (the exponent counter `nex` grows by 40).
const SC40: f64 = SC20 * SC20;

/// Scale `val` by $2^{\text{exp}}$ without overflowing an intermediate `2.0_f64.powi(-exp)`.
///
/// No direct Fortran counterpart. `dnlfk` ends with `t1 = t1 / 2.d0**(n-1-nex)`; this helper computes the
/// equivalent `t1 * 2^(nex + 1 - n)` in steps of at most $2^{\pm 512}$, so that no intermediate power of
/// two overflows or underflows in `f64`.
///
/// Difference from Fortran: the exponent is applied in chunks (and as a multiplication) instead of one
/// division by a possibly huge power of two.
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
///
/// Parameters:
/// - `m`: order of $\bar{P}_n^m$; any integer, with $\bar{P}_n^{-m} = (-1)^m \bar{P}_n^m$ (negative `m` is
///   handled through `ma = |m|`).
/// - `n`: degree, nonnegative.
/// - `cp`: output slice; Fortran: "cp requires n/2+1 double precision locations", so its length must
///   be at least `n / 2 + 1` (at least 1). Only the first `l` entries are meaningful, where
///   `l = (n + 1) / 2`, plus one when `n` and `m` are both even. Its meaning depends on the parity of
///   `n` and `m` as listed in the module documentation (to be used by [`dnlft`]).
///
/// Edge cases: `|m| > n` gives `cp[0] = 0` (so $\bar{P}_n^m = 0$) and returns. `n = 0` gives
/// `cp[0] = sqrt(2)`. `n = 1` gives `sqrt(3/2)` for `m = 0` and `sqrt(3/4)` for `|m| = 1`, negated for
/// `m = -1`.
///
/// Algorithm (Fortran `alfk` header): "the highest order coefficient is determined in closed form and
/// the remaining coefficients are determined as the solution of a backward recurrence relation." The
/// closed form is accumulated as a product of ratios with the running scale `t1` kept at or below
/// $2^{20}$ by rescaling (counter `nex`).
///
/// Fortran: sphcom.f / dnlfk (single precision twin: alf.f / alfk)
///
/// Difference from Fortran: `usize` degree and 0-based `cp`, `f64` throughout, and the final power of two
/// is applied by `scale_pow2`.
pub(crate) fn dnlfk(m: isize, n: usize, cp: &mut [f64]) {
    let ma = m.unsigned_abs();
    // Fortran `cp(1) = 0.` then ma = |m|. If |m| > n then P_n^m = 0 and cp(1) = 0 is the answer.
    cp[0] = 0.0;
    if ma > n {
        return;
    }

    // Fortran arithmetic IF `if(n-1) 2,3,5`: n = 0 (label 2), n = 1 (label 3), n >= 2 (label 5, general case).
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

    // n + |m| even: nmms2 = (n-|m|)/2, fnum = n+|m|+1, fnmh = n-|m|+1, pm1 = +1.
    // n + |m| odd:  nmms2 = (n-|m|-1)/2, fnum = n+|m|+2, fnmh = n-|m|+2, pm1 = -1.
    // nmms2 is the number of factors in the closed-form product; fnmh and pm1 drive the |m|-fold product t2.
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

    // Closed form for the highest coefficient. The Fortran source keeps (commented out) the unscaled start
    // t1 = 1/2**(n-1). Here the scaled start is 1/sc20 = 2**-20 with exponent counter nex = 20, so the true
    // value is t1 * 2**(nex - (n-1)).
    let mut t1 = 1.0_f64 / SC20;
    let mut nex: i32 = 20;
    let mut fden = 2.0_f64;

    for _ in 0..nmms2 {
        t1 = fnum * t1 / fden;
        // Rescale to avoid overflow: divide by sc40 = 2**40 and account for it by nex += 40.
        if t1 > SC20 {
            t1 /= SC40;
            nex += 40;
        }
        fnum += 2.0;
        fden += 2.0;
    }

    // Fortran: t1 = t1/2.d0**(n-1-nex), applied here as t1 * 2**(nex + 1 - n).
    t1 = scale_pow2(t1, nex + 1 - (n as i32));
    // Sign (-1)**(|m|/2) of the closed form (Fortran `if(mod(ma/2,2) .ne. 0) t1 = -t1`).
    if (ma / 2) % 2 != 0 {
        t1 = -t1;
    }

    // t2: product over |m| factors fnmh/(fnmh + pm1), fnmh advancing by 2 (Fortran loop 25).
    let mut t2 = 1.0_f64;
    for _ in 0..ma {
        t2 = fnmh * t2 / (fnmh + pm1);
        fnmh += 2.0;
    }

    // cp2: the highest-order coefficient, t1 * sqrt((n + 1/2) * t2).
    let cp2 = t1 * (((n as f64) + 0.5) * t2).sqrt();
    // fnnp1 = n(n+1); fnmsq = n(n+1) - 2*m**2. Both enter the coefficients a1, b1, c1 of the backward recurrence.
    let fnnp1 = (n * (n + 1)) as f64;
    let fnmsq = fnnp1 - 2.0 * ((ma * ma) as f64);

    // l: 1-based index of the highest coefficient. For n and |m| both even the series has a constant term, so one extra coefficient is stored.
    let mut l = (n + 1) / 2;
    if n % 2 == 0 && ma % 2 == 0 {
        l += 1;
    }
    cp[l - 1] = cp2;
    // P_n^{-m} = (-1)**m P_n^m: for negative odd m the coefficients change sign.
    if m < 0 && ma % 2 != 0 {
        cp[l - 1] = -cp[l - 1];
    }
    if l <= 1 {
        return;
    }

    // Backward recurrence for the lower coefficients: first step cp(l-1) = b1*cp(l)/a1 with a1 = (fk-2)(fk-1) - n(n+1) and b1 = 2(fk**2 - fnmsq).
    let mut fk = n as f64;
    let a1 = (fk - 2.0) * (fk - 1.0) - fnnp1;
    let b1 = 2.0 * (fk * fk - fnmsq);
    cp[l - 2] = b1 * cp[l - 1] / a1;

    // General step (Fortran label 30): with c1 = (fk+1)(fk+2) - n(n+1) and b1 = -2(fk**2 - fnmsq), cp(l-1) = -(b1*cp(l) + c1*cp(l+1))/a1; fk decreases by 2 each time.
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
///
/// Parameters:
/// - `m`: order, nonnegative.
/// - `n`: degree.
/// - `theta`: colatitude in radians.
/// - `cp`: coefficients computed by [`dnlfk`] for the same `(m, n)`.
///
/// Returns $\bar{P}_n^m(\cos\theta)$, evaluated from the trigonometric series for the four cases of
/// the module documentation. Returns 0 if `m > n`.
///
/// Algorithm: the sines and cosines of the successive multiples of the angle are produced with the
/// angle-addition recurrence (`cth`, `sth` hold the cosine and sine of the current argument, `cdt`, `sdt`
/// those of `2 * theta`), so only a constant number of trigonometric evaluations is needed per call. Fortran `lfpt`
/// header: "calls to routine lfpt must be preceded by an appropriate call to routine alfk".
///
/// Fortran: sphcom.f / dnlft (single precision twin: alf.f / lfpt)
///
/// Difference from Fortran: the guards for `m > n` and `n == 0` are added here; the Fortran `dnlft`
/// has none and relies on the caller. Only nonnegative `m` is accepted.
#[inline]
#[allow(dead_code)]
pub(crate) fn dnlft(m: usize, n: usize, theta: f64, cp: &[f64]) -> f64 {
    // P_n^m = 0 for m > n (the Fortran lfpt header: pbar(n,m,theta) = 0 if abs(m) is greater than n).
    if m > n {
        return 0.0;
    }
    if n == 0 {
        return 0.5 * cp[0];
    }

    // cdt, sdt = cos(2*theta), sin(2*theta): the rotation applied to (cth, sth) at each term.
    let cdt = (theta + theta).cos();
    let sdt = (theta + theta).sin();

    if n % 2 == 0 {
        // n even: kdo = n/2 terms.
        let kdo = n / 2;
        if m % 2 == 0 {
            // n even, m even: pb = 0.5*cp(1) + sum_{k=1..n/2} cp(k+1)*cos(2*k*theta). Fortran comment: pb = pb+cp(k+1)*dcos(2*k*theta).
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
            // n even, m odd: pb = sum_{k=1..n/2} cp(k)*sin(2*k*theta). Fortran comment: pb = pb+cp(k)*dsin(2*k*theta).
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
        // n odd: kdo = (n+1)/2 terms, the recurrence starts at theta (odd multiples (2k-1)*theta).
        let kdo = (n + 1) / 2;
        let mut cth = theta.cos();
        let mut sth = theta.sin();
        let mut pb = 0.0_f64;
        // n odd, m even: pb = sum_{k=1..kdo} cp(k)*cos((2*k-1)*theta) (Fortran comment: dcos((2*k-1)*theta)). n odd, m odd: sin((2*k-1)*theta) instead.
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
///
/// Parameters:
/// - `ntrunc`: spectral truncation; functions with `0 <= m <= n <= ntrunc` are produced.
/// - `colat_rad`: colatitude in radians, in $[0, \pi]$ (values within $10^{-12}$ outside are clamped).
///
/// Returns a vector of length `(ntrunc + 1) * (ntrunc + 2) / 2` holding $\bar{P}_n^m(\theta)$ in
/// triangular packing, index `m * (2 * ntrunc - m + 3) / 2 + (n - m)`. In Fortran terms (1-based
/// `m, n` loops from `getlegfunc`) this is `nm = nmstrt + n - m + 1` with
/// `nmstrt += ntrunc - m + 2` after each `m`.
///
/// Errors: `Err` if `colat_rad` is not finite or lies outside $[0, \pi]$ by more than $10^{-12}$.
///
/// Algorithm:
/// - $m = 0$ and $m = 1$: three-term (Bonnet-type) recurrences in `n`.
/// - $m \ge 2$: Swarztrauber's recurrence from `lfim1` in `alf.f` ("four term recurrence relation,
///   unpublished notes by Paul N. Swarztrauber"), which couples levels `n` and `n - 2`. Only three
///   consecutive `n` levels are kept (rolling buffer), so memory is $O(n_{\mathrm{trunc}})$ besides the
///   output.
///
/// Fortran: getlegfunc.f / getlegfunc (values), alf.f / lfim, lfim1 (recurrence coefficients)
///
/// Difference from Fortran:
/// - `getlegfunc(legfunc, lat, ntrunc)` takes the latitude in degrees (`theta = pi/2 - (pi/180)*lat`);
///   here the colatitude is passed directly, in radians.
/// - The Fortran version evaluates every function independently with `alfk` + `lfpt`
///   (O(N^3) overall, single precision); here one recurrence sweep is used.
/// - For $m = 0, 1$ the Fortran `lfim1` initialization uses `alfk` + `lfpt`; here Bonnet recurrences
///   replace them.
/// - `f64` precision and `Result` instead of status codes; no workspace arrays.
pub fn legendre_pbar(ntrunc: usize, colat_rad: f64) -> Result<Vec<f64>, String> {
    if !colat_rad.is_finite() || colat_rad < -1e-12 || colat_rad > PI + 1e-12 {
        return Err(format!(
            "colat_rad must be in [0, pi], got {colat_rad}"
        ));
    }
    let theta = colat_rad.clamp(0.0, PI);
    // nmdim: number of (n, m) pairs with 0 <= m <= n <= ntrunc.
    let nmdim = (ntrunc + 1) * (ntrunc + 2) / 2;
    let mut out = vec![0.0_f64; nmdim];

    let (sin_th, cos_th) = theta.sin_cos();
    // Fortran `lfim1` keeps the same three levels as p1 (n-2), p2 (n-1) and p3 (n), copying p2 -> p1 and p3 -> p2 after each n; here the three slots rotate by index instead.
    // 3 rolling n-levels: nm2 (n-2), nm1 (n-1), curr (n), each of length ntrunc + 1.
    let stride = ntrunc + 1;
    let mut ring = vec![0.0_f64; 3 * stride];

    // P_0^0 = 1/sqrt(2) (Fortran lfim1: ssqrt2 = 1./sqrt(2.)).
    // n = 0
    let ssqr2 = 0.5_f64.sqrt();
    ring[0] = ssqr2;
    out[0] = ssqr2;
    if ntrunc == 0 {
        return Ok(out);
    }

    // P_1^0 = sqrt(3/2) cos(theta) and P_1^1 = sqrt(3/4) sin(theta). These agree with the n = 1 branch of dnlfk (cp = sqrt(1.5), sqrt(.75)).
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
        // Slot indices of the levels n - 2, n - 1 and n inside the rolling buffer.
        let nm2_idx = ((n + 1) % 3) * stride;
        let nm1_idx = ((n + 2) % 3) * stride;
        let cur_idx = (n % 3) * stride;

        let nf = n as f64;
        let c_num = 2.0 * nf + 1.0;

        // P_n^0 = a0 cos(theta) P_{n-1}^0 - b0 P_{n-2}^0 with a0 = sqrt((2n+1)(2n-1))/n and b0 = ((n-1)/n) sqrt((2n+1)/(2n-3)).
        // m = 0 via 3-term Bonnet recurrence
        let a0 = (c_num * (2.0 * nf - 1.0)).sqrt() / nf;
        let b0 = ((nf - 1.0) / nf) * (c_num / (2.0 * nf - 3.0)).sqrt();
        let p0 = a0 * cos_th * ring[nm1_idx] - b0 * ring[nm2_idx];
        ring[cur_idx] = p0;
        out[n] = p0;

        // Same structure for m = 1. For n = 2 the coefficient b1 is set to 0 because P_0^1 = 0.
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

        // Fortran lfim1 (loop 70): p3(m) = cc*p1(m-2) + dd*p1(m) - ee*p3(m-2), with p1 = level n-2, p3 = level n,
        // cn = (2n+1)/(2n-3), temp = (n+m)(n+m-1), and
        //   cc = sqrt(cn (n+m-3)(n+m-2) / temp)   -> abel,
        //   dd = sqrt(cn (n-m)(n-m-1) / temp)     -> bbel,
        //   ee = sqrt((n-m+1)(n-m+2) / temp)      -> cbel.
        // For m = n-1 and m = n Fortran uses special lines (71 and 75-80) without the dd term, which is
        // exactly the `else` branch below.
        // m = 2..=n via Schwarztrauber recurrence
        let c_den = 2.0 * nf - 3.0;
        for m in 2..=n {
            let mf = m as f64;
            let d1 = (mf + nf - 1.0) * (mf + nf);
            let abel = (c_num * (mf + nf - 2.0) * (mf + nf - 3.0) / (c_den * d1)).sqrt();
            let cbel = (((nf - mf + 1.0) * (nf - mf + 2.0)) / d1).sqrt();

            // m <= n-2: all three terms. m = n-1, n: P_{n-2}^m does not exist (bbel would be 0), so only abel and cbel remain.
            let val = if m < n - 1 {
                let bbel = (c_num * (nf - mf - 1.0) * (nf - mf) / (c_den * d1)).sqrt();
                abel * ring[nm2_idx + m - 2] + bbel * ring[nm2_idx + m]
                    - cbel * ring[cur_idx + m - 2]
            } else {
                abel * ring[nm2_idx + m - 2] - cbel * ring[cur_idx + m - 2]
            };

            ring[cur_idx + m] = val;
            // Triangular packing: m-major blocks of length ntrunc - m + 1, the same offset formula as `spectral_offset(m, ntrunc)` in the Fortran reduced Gaussian code (there 1-based, plus 1).
            let nm = m * (2 * ntrunc - m + 3) / 2 + (n - m);
            out[nm] = val;
        }
    }

    Ok(out)
}

/// Compute associated Legendre functions $\bar{P}_n^m(\theta)$ in SPHEREPACK `getlegfunc` / `shaes`
/// triangular packing order for `ntrunc < nlat` at colatitude `colat_rad`.
///
/// Parameters:
/// - `nlat`: number of Gaussian latitudes of the grid the truncation belongs to (at least 1).
/// - `ntrunc`: spectral truncation, must satisfy `ntrunc <= nlat - 1`.
/// - `colat_rad`: colatitude in radians.
///
/// Returns the same array as [`legendre_pbar`] (triangular packing, index
/// `m * (2 * ntrunc - m + 3) / 2 + (n - m)`). Errors: `nlat == 0`, `ntrunc >= nlat`, or any error of
/// [`legendre_pbar`]. The `ntrunc <= nlat - 1` bound mirrors the check `ntrunc > nlat - 1` (`ierror = 2`) in
/// the Fortran reduced Gaussian routines.
///
/// Fortran: getlegfunc.f / getlegfunc (no `nlat` argument there)
///
/// Difference from Fortran: colatitude in radians instead of latitude in degrees, `f64`, and `Result`
/// instead of a status code.
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

    /// Checks `legendre_pbar` against the reference `dnlfk` + `dnlft` for `m = 0, 1` and several degrees, and
    /// against the addition theorem $\sum_{m=0}^{n} w_m \bar{P}_n^m(\theta)^2 = (2n+1)/2$ (with
    /// $w_0 = 1$ and $w_m = 2$ otherwise) at `n = 319` and `n = 959`, the degrees where the single precision
    /// Fortran `alfk` underflows.
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
