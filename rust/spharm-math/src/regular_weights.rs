//! Exact SPHEREPACK Clenshaw-Curtis quadrature basis functions for regular lat-lon grids
//! (`dnzfk`/`dnzft`, `dzvk`/`dzvt`, `dzwk`/`dzwt`, `dvbk`, `dwbk`, and `rabcp`/`rabcv`/`rabcw`)
//! computed entirely in `f64` without polar division by zero.
//!
//! Derived from NCAR SPHEREPACK 3.2 (J. Adams and P. Swarztrauber), see LICENSE.spherepack.
//!
//! # Background (SPHEREPACK `sphcom.f`)
//!
//! On a regular grid the `nlat` colatitudes include both poles and are equally spaced,
//! $\theta_i = (i-1)\pi/(nlat-1)$ for `i = 1..nlat` (Fortran `zfini1`: `dt = pi/(nlat-1)`,
//! `th = (i-1)*dt`). The analysis ("Z") functions $Z_n^m$ (scalar), $Z_{v,n}^m$ and
//! $Z_{w,n}^m$ (vector) are the functions used in spherical harmonic analysis on such a grid.
//! Each is a finite trigonometric series in $\theta$: `dnzfk`, `dzvk` and `dzwk` compute the
//! series coefficients, and `dnzft`, `dzvt` and `dzwt` tabulate the series at one colatitude.
//! The coefficients are built from the trigonometric (Fourier) coefficients of
//! $\bar P_n^m$ (`dnlfk`), $\bar V_n^m$ (`dvbk`) and $\bar W_n^m$ (`dwbk`).
//!
//! The recurring factors `t1 = 1 - a^2`, `t2 = 1 - b^2` in the coefficient loops are the
//! Clenshaw-Curtis terms: for an even integer `a`, $\int_0^\pi \cos(a\theta)\sin\theta\,d\theta
//! = 2/(1-a^2)$, and `(t1 + t2)/(t1 t2) = 1/t1 + 1/t2`. The overall factor
//! `sc1 = 2/(nlat-1)` is applied to every coefficient.
//!
//! Only $m = 0, 1$ (scalar and V) and $m = 1, 2$ (W) are evaluated directly from the
//! trigonometric series (Fortran `zfini1`, `zvini1`, `zwini1`). Higher orders follow from a
//! three-term recurrence in `m` (Fortran `zfin1`, `zvin1`, `zwin1`) whose coefficients
//! `a`, `b`, `c` are produced by `rabcp`, `rabcv` and `rabcw`.
//!
//! # Differences from Fortran
//!
//! - `m`, `n` are the usual 0-based order and degree (Fortran loops use `mp1 = m + 1`,
//!   `np1 = n + 1`); array indices are 0-based, so e.g. Fortran `i = idx+idx-2` for
//!   1-based `idx` becomes `i = 2*idx` here.
//! - Everything is `f64` (Fortran stores the recurrence coefficients and the `z` tables in
//!   single precision `real`; only the coefficient routines such as `dnzfk`, `dnzft`, `dzvk` use `double precision`).
//! - There are no workspace arrays (`wzfin`, `wzvin`, `wzwin`, `abc`, `dwork`) and no `ityp`/`isym`
//!   flags: the recurrence coefficients are evaluated on the fly and rows are streamed through
//!   an `emit` callback. A 3-slice ring buffer replaces the saved `i1`, `i2`, `i3` rotation of
//!   `zfin1`/`zvin1`/`zwin1`.
//! - Rows are stored `[n * imid + i]` (C order) rather than Fortran `z(imid, nlat, 3)`.

use crate::legendre::dnlfk;

/// Fourier coefficients `cv[0..ncv]` of $\bar V_n^m(\theta)$ from those of $\bar P_n^m$.
///
/// Parameters:
/// - `m`: order (Fortran `m`).
/// - `n`: degree (Fortran `n`); for `n == 0` only `cv[0] = 0` is set.
/// - `cv`: output coefficients, `cv[0] = 0` always. Number of meaningful entries:
///   `ncv = n/2` for even `n`, `ncv = (n+1)/2` for odd `n`.
/// - `work`: scratch holding the `dnlfk` coefficients of $\bar P_n^m$ (needs `n/2 + 2` entries).
///
/// Algorithm (four parity cases, `fk` steps by 2 from 0 for even `n` and from 1 for odd `n`):
/// `cv[l] = -+ fk * (coefficient of P) / sqrt(n (n+1))`, with the sign `-` for even `m` and `+`
/// for odd `m`; for even `n` and even `m` the shifted coefficient `work[l + 1]` is used,
/// otherwise `work[l]`.
///
/// Fortran: sphcom.f / dvbk
///
/// Difference from Fortran: `cv` and `work` are separate buffers (Fortran `dzvk` calls
/// `dvbk(m,n,work,czv)`, reusing `czv` as the `dnlfk` scratch), and the unused `cf` is dropped.
#[inline]
fn dvbk(m: usize, n: usize, cv: &mut [f64], work: &mut [f64]) {
    cv[0] = 0.0;
    if n == 0 {
        return;
    }
    let fn_ = n as f64;
    let srnp1 = (fn_ * (fn_ + 1.0)).sqrt();
    dnlfk(m as isize, n, work);

    if n % 2 == 0 {
        let ncv = n / 2;
        if ncv == 0 {
            return;
        }
        let mut fk = 0.0_f64;
        if m % 2 == 0 {
            for l in 0..ncv {
                fk += 2.0;
                cv[l] = -fk * work[l + 1] / srnp1;
            }
        } else {
            for l in 0..ncv {
                fk += 2.0;
                cv[l] = fk * work[l] / srnp1;
            }
        }
    } else {
        let ncv = (n + 1) / 2;
        let mut fk = -1.0_f64;
        if m % 2 == 0 {
            for l in 0..ncv {
                fk += 2.0;
                cv[l] = -fk * work[l] / srnp1;
            }
        } else {
            for l in 0..ncv {
                fk += 2.0;
                cv[l] = fk * work[l] / srnp1;
            }
        }
    }
}

/// Fourier coefficients `cw[0..]` of $\bar W_n^m(\theta)$ from those of $\bar P_n^m$.
///
/// Parameters:
/// - `m`: order; `cw[0] = 0` and an immediate return occur for `m == 0` or `n == 0`.
/// - `n`: degree.
/// - `cw`: output coefficients (length `l_max`, with `l_max = n/2` for even `n`,
///   `(n-1)/2` for odd `n` and even `m`, `(n+1)/2` for odd `n` and odd `m`).
/// - `work`: scratch holding the `dnlfk` coefficients of $\bar P_n^m$.
///
/// Algorithm: with `cf = 2 m / sqrt(n (n+1))` the coefficients are a running (reverse)
/// cumulative sum, starting at the highest index and moving down to 1:
/// `cw[l-1] = cw[l] -+ cf * work[...]` (minus for even `m`, plus for odd `m`).
///
/// Fortran: sphcom.f / dwbk
///
/// Difference from Fortran: the Fortran `go to` ladder (`l = l-1`, test `l .le. 0`) is written
/// as `for l in (1..l_max).rev()`; indices are shifted to 0-based.
#[inline]
fn dwbk(m: usize, n: usize, cw: &mut [f64], work: &mut [f64]) {
    cw[0] = 0.0;
    if n == 0 || m == 0 {
        return;
    }
    let fn_ = n as f64;
    let srnp1 = (fn_ * (fn_ + 1.0)).sqrt();
    let cf = 2.0 * (m as f64) / srnp1;
    dnlfk(m as isize, n, work);

    if n % 2 == 0 {
        let l_max = n / 2;
        if l_max == 0 {
            return;
        }
        if m % 2 == 0 {
            cw[l_max - 1] = -cf * work[l_max];
            for l in (1..l_max).rev() {
                cw[l - 1] = cw[l] - cf * work[l];
            }
        } else {
            cw[l_max - 1] = cf * work[l_max - 1];
            for l in (1..l_max).rev() {
                cw[l - 1] = cw[l] + cf * work[l - 1];
            }
        }
    } else if m % 2 == 0 {
        let l_max = (n - 1) / 2;
        if l_max == 0 {
            return;
        }
        cw[l_max - 1] = -cf * work[l_max];
        for l in (1..l_max).rev() {
            cw[l - 1] = cw[l] - cf * work[l];
        }
    } else {
        let l_max = (n + 1) / 2;
        cw[l_max - 1] = cf * work[l_max - 1];
        for l in (1..l_max).rev() {
            cw[l - 1] = cw[l] + cf * work[l - 1];
        }
    }
}

/// Compute Clenshaw-Curtis scalar analysis coefficients `cz[0..lc]` matching SPHEREPACK `dnzfk`.
///
/// Parameters:
/// - `nlat`: number of colatitudes including the poles.
/// - `m`: order, `n`: degree of $Z_n^m$.
/// - `cz`: output, `lc = (nlat+1)/2` coefficients (Fortran requires `nlat/2 + 1` locations).
/// - `work`: scratch for the `dnlfk` coefficients of $\bar P_n^m$ (Fortran: `nlat/2 + 1`).
///
/// Algorithm: with `sc1 = 2/(nlat-1)` and `t1 = 1 - (2k + i)^2`, `t2 = 1 - (2k - i)^2`,
/// the coefficient for frequency index `i` is `sc1 * sum_k work[.] * (t1 + t2)/(t1 t2)` for even `m`
/// (cosine series) and `sc1 * sum_k work[.] * (t1 - t2)/(t1 t2)` for odd `m` (sine series).
/// The odd-`n` branches use `2k - 1` in place of `2k`.
/// The four `(n, m)` parity cases are commented inline.
///
/// Fortran: sphcom.f / dnzfk
fn dnzfk(nlat: usize, m: usize, n: usize, cz: &mut [f64], work: &mut [f64]) {
    let lc = (nlat + 1) / 2;
    let sc1 = 2.0 / ((nlat - 1) as f64);
    dnlfk(m as isize, n, work);

    if n % 2 == 0 {
        if m % 2 == 0 {
            // n even, m even: kdo = n/2 + 1, i = 2*idx (even), first term work[0]/(1 - i^2)
            // (the k = 0 term), then k = 1..n/2 with (t1 + t2)/(t1 t2).
            let kdo = n / 2 + 1;
            for idx in 0..lc {
                let i = (2 * idx) as isize;
                let mut sum = work[0] / (1.0 - (i * i) as f64);
                for kp1 in 2..=kdo {
                    let k = (kp1 - 1) as isize;
                    let a = (2 * k + i) as f64;
                    let b = (2 * k - i) as f64;
                    let t1 = 1.0 - a * a;
                    let t2 = 1.0 - b * b;
                    sum += work[kp1 - 1] * (t1 + t2) / (t1 * t2);
                }
                cz[idx] = sc1 * sum;
            }
        } else {
            // n even, m odd: kdo = n/2, i = 2*idx (even), terms k = 1..n/2 with (t1 - t2)/(t1 t2).
            let kdo = n / 2;
            for idx in 0..lc {
                let i = (2 * idx) as isize;
                let mut sum = 0.0_f64;
                for k in 1..=kdo {
                    let ki = k as isize;
                    let a = (2 * ki + i) as f64;
                    let b = (2 * ki - i) as f64;
                    let t1 = 1.0 - a * a;
                    let t2 = 1.0 - b * b;
                    sum += work[k - 1] * (t1 - t2) / (t1 * t2);
                }
                cz[idx] = sc1 * sum;
            }
        }
    } else if m % 2 == 0 {
        // n odd, m even: kdo = (n+1)/2, i = 2*(idx+1) - 1 (odd; Fortran i = idx+idx-1),
        // a = 2k-1+i, b = 2k-1-i, terms with (t1 + t2)/(t1 t2).
        let kdo = (n + 1) / 2;
        for idx in 0..lc {
            let i = (2 * (idx + 1) - 1) as isize;
            let mut sum = 0.0_f64;
            for k in 1..=kdo {
                let ki = k as isize;
                let a = (2 * ki - 1 + i) as f64;
                let b = (2 * ki - 1 - i) as f64;
                let t1 = 1.0 - a * a;
                let t2 = 1.0 - b * b;
                sum += work[k - 1] * (t1 + t2) / (t1 * t2);
            }
            cz[idx] = sc1 * sum;
        }
    } else {
        // n odd, m odd: kdo = (n+1)/2, i = 2*(idx+1) - 3 (Fortran i = idx+idx-3),
        // a = 2k-1+i, b = 2k-1-i, terms with (t1 - t2)/(t1 t2).
        let kdo = (n + 1) / 2;
        for idx in 0..lc {
            let i = 2 * (idx as isize + 1) - 3;
            let mut sum = 0.0_f64;
            for k in 1..=kdo {
                let ki = k as isize;
                let a = (2 * ki - 1 + i) as f64;
                let b = (2 * ki - 1 - i) as f64;
                let t1 = 1.0 - a * a;
                let t2 = 1.0 - b * b;
                sum += work[k - 1] * (t1 - t2) / (t1 * t2);
            }
            cz[idx] = sc1 * sum;
        }
    }
}

/// Evaluate scalar Z-function at colatitude `th` matching SPHEREPACK `dnzft`.
///
/// Parameters:
/// - `nlat`, `m`, `n`: as in `dnzfk`; `th`: colatitude in radians.
/// - `cz`: coefficients from `dnzfk` (`lc = (nlat+1)/2` entries).
///
/// Returns `zh`, the trigonometric series of $Z_n^m$ at `th`. Depending on the parity of
/// `nlat`, `n` and `m` the series is a cosine or sine sum in multiples of `th` or `2 th`; the
/// end terms (`cz[0]` and `cz[lc-1]`) carry weight 1/2 where the Fortran source does so.
///
/// Fortran: sphcom.f / dnzft
#[inline]
fn dnzft(nlat: usize, m: usize, n: usize, th: f64, cz: &[f64]) -> f64 {
    let mut zh = 0.0_f64;
    // Instead of evaluating cos(2(k-1) th) directly (the commented-out Fortran line
    // `zh = zh + cz(k)*dcos(2*(k-1)*th)`), (cth, sth) holds cos/sin of the running multiple
    // and is advanced by the angle-addition formulas with increment (cdt, sdt) = cos/sin(2 th).
    let cdt = (th + th).cos();
    let sdt = (th + th).sin();

    // nlat odd (Fortran lmod != 0): lc = (nlat+1)/2, lq = lc-1, ls = lc-2.
    if nlat % 2 != 0 {
        let lc = (nlat + 1) / 2;
        let lq = lc - 1;
        let ls = lc.saturating_sub(2);
        if n % 2 == 0 {
            if m % 2 == 0 {
                // nlat odd, n even, m even: zh = (cz[0] + cz[lc-1] cos(2 lq th))/2 + sum_{k=2}^{lq} cz[k-1] cos(2(k-1) th).
                zh = 0.5 * (cz[0] + cz[lc - 1] * (2.0 * (lq as f64) * th).cos());
                let mut cth = cdt;
                let mut sth = sdt;
                for k in 2..=lq {
                    zh += cz[k - 1] * cth;
                    let chh = cdt * cth - sdt * sth;
                    sth = sdt * cth + cdt * sth;
                    cth = chh;
                }
            } else {
                // nlat odd, n even, m odd: zh = sum_{k=1}^{ls} cz[k] sin(2k th).
                let mut cth = cdt;
                let mut sth = sdt;
                for k in 1..=ls {
                    zh += cz[k] * sth;
                    let chh = cdt * cth - sdt * sth;
                    sth = sdt * cth + cdt * sth;
                    cth = chh;
                }
            }
        } else {
            // nlat odd, n odd: multiples are odd, (cth, sth) start at cos/sin(th). m even: sum cz[k-1] cos((2k-1) th); m odd: sum cz[k] sin((2k-1) th), k = 1..lq.
            let mut cth = th.cos();
            let mut sth = th.sin();
            if m % 2 == 0 {
                for k in 1..=lq {
                    zh += cz[k - 1] * cth;
                    let chh = cdt * cth - sdt * sth;
                    sth = sdt * cth + cdt * sth;
                    cth = chh;
                }
            } else {
                for k in 1..=lq {
                    zh += cz[k] * sth;
                    let chh = cdt * cth - sdt * sth;
                    sth = sdt * cth + cdt * sth;
                    cth = chh;
                }
            }
        }
    } else {
        // nlat even: lc = nlat/2, lq = lc-1.
        let lc = nlat / 2;
        let lq = lc - 1;
        if n % 2 == 0 {
            // nlat even, n even: (cth, sth) start at cos/sin(2 th). m even: zh = cz[0]/2 + sum_{k=2}^{lc} cz[k-1] cos(2(k-1) th); m odd: sum_{k=1}^{lq} cz[k] sin(2k th).
            let mut cth = cdt;
            let mut sth = sdt;
            if m % 2 == 0 {
                zh = 0.5 * cz[0];
                for k in 2..=lc {
                    zh += cz[k - 1] * cth;
                    let chh = cdt * cth - sdt * sth;
                    sth = sdt * cth + cdt * sth;
                    cth = chh;
                }
            } else {
                for k in 1..=lq {
                    zh += cz[k] * sth;
                    let chh = cdt * cth - sdt * sth;
                    sth = sdt * cth + cdt * sth;
                    cth = chh;
                }
            }
        } else {
            // nlat even, n odd: (cth, sth) start at cos/sin(th). m even: zh = cz[lc-1] cos((nlat-1) th)/2 + sum_{k=1}^{lq} cz[k-1] cos((2k-1) th); m odd: sum_{k=1}^{lq} cz[k] sin((2k-1) th).
            let mut cth = th.cos();
            let mut sth = th.sin();
            if m % 2 == 0 {
                zh = 0.5 * cz[lc - 1] * (((nlat - 1) as f64) * th).cos();
                for k in 1..=lq {
                    zh += cz[k - 1] * cth;
                    let chh = cdt * cth - sdt * sth;
                    sth = sdt * cth + cdt * sth;
                    cth = chh;
                }
            } else {
                for k in 1..=lq {
                    zh += cz[k] * sth;
                    let chh = cdt * cth - sdt * sth;
                    sth = sdt * cth + cdt * sth;
                    cth = chh;
                }
            }
        }
    }
    zh
}

///
/// Parameters:
/// - `nlat`: number of colatitudes including the poles; `m`, `n`: order and degree of $Z_{v,n}^m$.
/// - `czv`: output, `lc = (nlat+1)/2` coefficients.
/// - `cv`: scratch for the `dvbk` coefficients of $\bar V_n^m$.
/// - `work`: scratch for the `dnlfk` coefficients (Fortran requires `nlat/2 + 1` locations).
///
/// Algorithm: same Clenshaw-Curtis sum as `dnzfk` with `sc1 = 2/(nlat-1)`,
/// `t1 = 1 - a^2`, `t2 = 1 - b^2`, applied to the coefficients `cv` of $\bar V_n^m$.
/// Each term is `(t1 - t2)/(t1 t2)` or `(t1 + t2)/(t1 t2)` depending on the `(n, m)` parity case;
/// see the inline comments.
///
/// Fortran: sphcom.f / dzvk
///
/// Difference from Fortran: for `n == 0` Fortran returns without touching `czv`; here `czv[..lc]`
/// is explicitly zeroed.
/// Compute Clenshaw-Curtis vector V analysis coefficients `czv[0..lc]` matching SPHEREPACK `dzvk`.
fn dzvk(nlat: usize, m: usize, n: usize, czv: &mut [f64], cv: &mut [f64], work: &mut [f64]) {
    let lc = (nlat + 1) / 2;
    if n == 0 {
        czv[..lc].fill(0.0);
        return;
    }

    let sc1 = 2.0 / ((nlat - 1) as f64);
    dvbk(m, n, cv, work);

    if n % 2 == 0 {
        let kdo = n / 2;
        if m % 2 == 0 {
            // n even, m even: kdo = n/2, i = 2*id, a = 2k+i, b = 2k-i, term cv[k-1] (t1 - t2)/(t1 t2).
            for id in 0..lc {
                let i = (2 * id) as isize;
                let mut sum = 0.0_f64;
                for k in 1..=kdo {
                    let ki = k as isize;
                    let a = (2 * ki + i) as f64;
                    let b = (2 * ki - i) as f64;
                    let t1 = 1.0 - a * a;
                    let t2 = 1.0 - b * b;
                    sum += cv[k - 1] * (t1 - t2) / (t1 * t2);
                }
                czv[id] = sc1 * sum;
            }
        } else {
            // n even, m odd: kdo = n/2, i = 2*id, a = 2k+i, b = 2k-i, term cv[k-1] (t1 + t2)/(t1 t2).
            for id in 0..lc {
                let i = (2 * id) as isize;
                let mut sum = 0.0_f64;
                for k in 1..=kdo {
                    let ki = k as isize;
                    let a = (2 * ki + i) as f64;
                    let b = (2 * ki - i) as f64;
                    let t1 = 1.0 - a * a;
                    let t2 = 1.0 - b * b;
                    sum += cv[k - 1] * (t1 + t2) / (t1 * t2);
                }
                czv[id] = sc1 * sum;
            }
        }
    } else {
        let kdo = (n + 1) / 2;
        if m % 2 == 0 {
            // n odd, m even: kdo = (n+1)/2, i = 2*(id+1) - 3, a = 2k-1+i, b = 2k-1-i, term cv[k-1] (t1 - t2)/(t1 t2).
            for id in 0..lc {
                let i = 2 * (id as isize + 1) - 3;
                let mut sum = 0.0_f64;
                for k in 1..=kdo {
                    let ki = k as isize;
                    let a = (2 * ki - 1 + i) as f64;
                    let b = (2 * ki - 1 - i) as f64;
                    let t1 = 1.0 - a * a;
                    let t2 = 1.0 - b * b;
                    sum += cv[k - 1] * (t1 - t2) / (t1 * t2);
                }
                czv[id] = sc1 * sum;
            }
        } else {
            // n odd, m odd: kdo = (n+1)/2, i = 2*(id+1) - 1, a = 2k-1+i, b = 2k-1-i, term cv[k-1] (t1 + t2)/(t1 t2).
            for id in 0..lc {
                let i = (2 * (id + 1) - 1) as isize;
                let mut sum = 0.0_f64;
                for k in 1..=kdo {
                    let ki = k as isize;
                    let a = (2 * ki - 1 + i) as f64;
                    let b = (2 * ki - 1 - i) as f64;
                    let t1 = 1.0 - a * a;
                    let t2 = 1.0 - b * b;
                    sum += cv[k - 1] * (t1 + t2) / (t1 * t2);
                }
                czv[id] = sc1 * sum;
            }
        }
    }
}

/// Evaluate vector V quadrature function at colatitude `th` matching SPHEREPACK `dzvt`.
///
/// Parameters:
/// - `nlat`: number of colatitudes including the poles; `m`, `n`: order and degree.
/// - `th`: colatitude in radians; `czv`: coefficients from `dzvk`.
///
/// Returns `zvh = zvbar(n, m, th)`, or 0 for `n == 0`. Cosine/sine sums in multiples of `th`
/// or `2 th` according to the parities of `nlat`, `n`, `m` (inline comments). Note that the
/// Fortran header of `dzvt` says the coefficients come from "zwk" (a typo for `dzvk`).
///
/// Fortran: sphcom.f / dzvt
#[inline]
fn dzvt(nlat: usize, m: usize, n: usize, th: f64, czv: &[f64]) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let mut zvh = 0.0_f64;
    let lc = (nlat + 1) / 2;
    let lq = lc - 1;
    let ls = lc.saturating_sub(2);
    // (cth, sth) = cos/sin(th), (cdt, sdt) = cos/sin(2 th) by the double-angle formulas; the
    // series terms are generated by repeated angle addition instead of calling cos/sin per term.
    let mut cth = th.cos();
    let mut sth = th.sin();
    let cdt = cth * cth - sth * sth;
    let sdt = 2.0 * sth * cth;

    // nlat odd: lc = (nlat+1)/2, lq = lc-1, ls = lc-2 (see the branches below).
    if nlat % 2 != 0 {
        if n % 2 == 0 {
            // nlat odd, n even: start at 2 th.
            cth = cdt;
            sth = sdt;
            if m % 2 == 0 {
                // m even: sum_{k=1}^{ls} czv[k] sin(2k th).
                for k in 1..=ls {
                    zvh += czv[k] * sth;
                    let chh = cdt * cth - sdt * sth;
                    sth = sdt * cth + cdt * sth;
                    cth = chh;
                }
            } else {
                // m odd: czv[0]/2 + sum_{k=2}^{lq} czv[k-1] cos(2(k-1) th) + czv[lc-1] cos((nlat-1) th)/2.
                zvh = 0.5 * czv[0];
                for k in 2..=lq {
                    zvh += czv[k - 1] * cth;
                    let chh = cdt * cth - sdt * sth;
                    sth = sdt * cth + cdt * sth;
                    cth = chh;
                }
                zvh += 0.5 * czv[lc - 1] * (((nlat - 1) as f64) * th).cos();
            }
        } else if m % 2 == 0 {
            // nlat odd, n odd, m even: sum_{k=1}^{lq} czv[k] sin((2k-1) th).
            for k in 1..=lq {
                zvh += czv[k] * sth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        } else {
            // nlat odd, n odd, m odd: sum_{k=1}^{lq} czv[k-1] cos((2k-1) th).
            for k in 1..=lq {
                zvh += czv[k - 1] * cth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        }
    } else if n % 2 == 0 {
        // nlat even, n even: start at 2 th.
        cth = cdt;
        sth = sdt;
        if m % 2 == 0 {
            // m even: sum_{k=1}^{lq} czv[k] sin(2k th).
            for k in 1..=lq {
                zvh += czv[k] * sth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        } else {
            // m odd: czv[0]/2 + sum_{k=2}^{lc} czv[k-1] cos(2(k-1) th).
            zvh = 0.5 * czv[0];
            for k in 2..=lc {
                zvh += czv[k - 1] * cth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        }
    } else if m % 2 == 0 {
        // nlat even, n odd, m even: sum_{k=1}^{lq} czv[k] sin((2k-1) th).
        for k in 1..=lq {
            zvh += czv[k] * sth;
            let chh = cdt * cth - sdt * sth;
            sth = sdt * cth + cdt * sth;
            cth = chh;
        }
    } else {
        // nlat even, n odd, m odd: czv[lc-1] cos((nlat-1) th)/2 + sum_{k=1}^{lq} czv[k-1] cos((2k-1) th).
        zvh = 0.5 * czv[lc - 1] * (((nlat - 1) as f64) * th).cos();
        for k in 1..=lq {
            zvh += czv[k - 1] * cth;
            let chh = cdt * cth - sdt * sth;
            sth = sdt * cth + cdt * sth;
            cth = chh;
        }
    }
    zvh
}

///
/// Parameters:
/// - `nlat`: number of colatitudes including the poles; `m`, `n`: order and degree of $Z_{w,n}^m$.
/// - `czw`: output, `lc = (nlat+1)/2` coefficients.
/// - `cw`: scratch for the `dwbk` coefficients of $\bar W_n^m$.
/// - `work`: scratch for the `dnlfk` coefficients (Fortran requires `nlat/2 + 1` locations).
///
/// Algorithm: same Clenshaw-Curtis sum as `dnzfk` / `dzvk` with `sc1 = 2/(nlat-1)`, applied
/// to `cw`. $\bar W_n^m$ vanishes for `m = 0`, hence the early return.
///
/// Fortran: sphcom.f / dzwk
///
/// Difference from Fortran: Fortran returns early only for `n <= 0`; here `czw[..lc]` is zeroed
/// and returned for both `n == 0` and `m == 0`.
/// Compute Clenshaw-Curtis vector W analysis coefficients `czw[0..lc]` matching SPHEREPACK `dzwk`.
fn dzwk(nlat: usize, m: usize, n: usize, czw: &mut [f64], cw: &mut [f64], work: &mut [f64]) {
    let lc = (nlat + 1) / 2;
    if n == 0 || m == 0 {
        czw[..lc].fill(0.0);
        return;
    }

    let sc1 = 2.0 / ((nlat - 1) as f64);
    dwbk(m, n, cw, work);

    if n % 2 == 0 {
        let kdo = n / 2;
        if m % 2 == 0 {
            // n even, m even: kdo = n/2, i = 2*(id+1) - 3, a = 2k-1+i, b = 2k-1-i, term cw[k-1] (t1 - t2)/(t1 t2).
            for id in 0..lc {
                let i = 2 * (id as isize + 1) - 3;
                let mut sum = 0.0_f64;
                for k in 1..=kdo {
                    let ki = k as isize;
                    let a = (2 * ki - 1 + i) as f64;
                    let b = (2 * ki - 1 - i) as f64;
                    let t1 = 1.0 - a * a;
                    let t2 = 1.0 - b * b;
                    sum += cw[k - 1] * (t1 - t2) / (t1 * t2);
                }
                czw[id] = sc1 * sum;
            }
        } else {
            // n even, m odd: kdo = n/2, i = 2*(id+1) - 1, a = 2k-1+i, b = 2k-1-i, term cw[k-1] (t1 + t2)/(t1 t2).
            for id in 0..lc {
                let i = (2 * (id + 1) - 1) as isize;
                let mut sum = 0.0_f64;
                for k in 1..=kdo {
                    let ki = k as isize;
                    let a = (2 * ki - 1 + i) as f64;
                    let b = (2 * ki - 1 - i) as f64;
                    let t1 = 1.0 - a * a;
                    let t2 = 1.0 - b * b;
                    sum += cw[k - 1] * (t1 + t2) / (t1 * t2);
                }
                czw[id] = sc1 * sum;
            }
        }
    } else if m % 2 == 0 {
        // n odd, m even: kdo = (n-1)/2, i = 2*id, a = 2k+i, b = 2k-i, term cw[k-1] (t1 - t2)/(t1 t2).
        let kdo = (n - 1) / 2;
        for id in 0..lc {
            let i = (2 * id) as isize;
            let mut sum = 0.0_f64;
            for k in 1..=kdo {
                let ki = k as isize;
                let a = (2 * ki + i) as f64;
                let b = (2 * ki - i) as f64;
                let t1 = 1.0 - a * a;
                let t2 = 1.0 - b * b;
                sum += cw[k - 1] * (t1 - t2) / (t1 * t2);
            }
            czw[id] = sc1 * sum;
        }
    } else {
        // n odd, m odd: kdo = (n+1)/2, i = 2*id; first term cw[0]/(1 - i^2) (k = 0), then k = 1..kdo-1 with (t1 + t2)/(t1 t2).
        let kdo = (n + 1) / 2;
        for id in 0..lc {
            let i = (2 * id) as isize;
            let mut sum = cw[0] / (1.0 - (i * i) as f64);
            for kp1 in 2..=kdo {
                let k = (kp1 - 1) as isize;
                let a = (2 * k + i) as f64;
                let b = (2 * k - i) as f64;
                let t1 = 1.0 - a * a;
                let t2 = 1.0 - b * b;
                sum += cw[kp1 - 1] * (t1 + t2) / (t1 * t2);
            }
            czw[id] = sc1 * sum;
        }
    }
}

/// Evaluate vector W quadrature function at colatitude `th` matching SPHEREPACK `dzwt`.
///
/// Parameters:
/// - `nlat`: number of colatitudes including the poles; `m`, `n`: order and degree.
/// - `th`: colatitude in radians; `czw`: coefficients from `dzwk`.
///
/// Returns `zwh = zwbar(n, m, th)`, or 0 for `n == 0` or `m == 0`. The Fortran header of `dzwt`
/// states that `nlat` must be odd, but the source (and this port) implements both parities.
///
/// Fortran: sphcom.f / dzwt
#[inline]
fn dzwt(nlat: usize, m: usize, n: usize, th: f64, czw: &[f64]) -> f64 {
    if n == 0 || m == 0 {
        return 0.0;
    }
    let mut zwh = 0.0_f64;
    let lc = (nlat + 1) / 2;
    let lq = lc - 1;
    let ls = lc.saturating_sub(2);
    // (cth, sth) = cos/sin(th), (cdt, sdt) = cos/sin(2 th) (double-angle); terms advance by
    // angle addition.
    let mut cth = th.cos();
    let mut sth = th.sin();
    let cdt = cth * cth - sth * sth;
    let sdt = 2.0 * sth * cth;

    // nlat odd: lc = (nlat+1)/2, lq = lc-1, ls = lc-2.
    if nlat % 2 != 0 {
        if n % 2 == 0 {
            if m % 2 == 0 {
                // nlat odd, n even, m even: sum_{k=1}^{lq} czw[k] sin((2k-1) th).
                for k in 1..=lq {
                    zwh += czw[k] * sth;
                    let chh = cdt * cth - sdt * sth;
                    sth = sdt * cth + cdt * sth;
                    cth = chh;
                }
            } else {
                // nlat odd, n even, m odd: sum_{k=1}^{lq} czw[k-1] cos((2k-1) th).
                for k in 1..=lq {
                    zwh += czw[k - 1] * cth;
                    let chh = cdt * cth - sdt * sth;
                    sth = sdt * cth + cdt * sth;
                    cth = chh;
                }
            }
        } else {
            // nlat odd, n odd: start at 2 th.
            cth = cdt;
            sth = sdt;
            if m % 2 == 0 {
                // m even: sum_{k=1}^{ls} czw[k] sin(2k th).
                for k in 1..=ls {
                    zwh += czw[k] * sth;
                    let chh = cdt * cth - sdt * sth;
                    sth = sdt * cth + cdt * sth;
                    cth = chh;
                }
            } else {
                // m odd: czw[0]/2 + sum_{k=2}^{lq} czw[k-1] cos(2(k-1) th) + czw[lc-1] cos((nlat-1) th)/2.
                zwh = 0.5 * czw[0];
                for k in 2..=lq {
                    zwh += czw[k - 1] * cth;
                    let chh = cdt * cth - sdt * sth;
                    sth = sdt * cth + cdt * sth;
                    cth = chh;
                }
                zwh += 0.5 * czw[lc - 1] * (((nlat - 1) as f64) * th).cos();
            }
        }
    } else if n % 2 == 0 {
        if m % 2 == 0 {
            // nlat even, n even, m even: sum_{k=1}^{lq} czw[k] sin((2k-1) th).
            for k in 1..=lq {
                zwh += czw[k] * sth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        } else {
            // nlat even, n even, m odd: czw[lc-1] cos((nlat-1) th)/2 + sum_{k=1}^{lq} czw[k-1] cos((2k-1) th).
            zwh = 0.5 * czw[lc - 1] * (((nlat - 1) as f64) * th).cos();
            for k in 1..=lq {
                zwh += czw[k - 1] * cth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        }
    } else {
        // nlat even, n odd: start at 2 th.
        cth = cdt;
        sth = sdt;
        if m % 2 == 0 {
            // m even: sum_{k=1}^{lq} czw[k] sin(2k th).
            for k in 1..=lq {
                zwh += czw[k] * sth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        } else {
            // m odd: czw[0]/2 + sum_{k=2}^{lc} czw[k-1] cos(2(k-1) th).
            zwh = 0.5 * czw[0];
            for k in 2..=lc {
                zwh += czw[k - 1] * cth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        }
    }
    zwh
}

/// Compute scalar Clenshaw-Curtis quadrature basis $Z_n^m(\theta_i)$ in `f64` for
/// `m = 0..mmax`, `n = m..nlat`, `i = 0..imid` (stored row-major `(n_idx, i)` per `m` via callback).
///
/// Parameters:
/// - `nlat`: number of colatitudes; `mmax`: number of zonal wavenumbers (Fortran
///   `mmax = min(nlat, nlon/2 + 1)`), orders `m = 0..mmax`.
/// - `colats_nh`: northern-hemisphere colatitudes (`imid = (nlat+1)/2` values, 0 to the equator).
/// - `emit(m, n, row)`: called once per `(m, n)` with `row.len() == imid`, in ascending `m`
///   and, within each `m`, ascending `n`. The row is only valid during the call.
///
/// Orders `m = 0, 1` are evaluated with `dnzfk`/`dnzft` and the first (pole) entry is halved
/// (Fortran `zfini1`: `z(1,np1,mp1) = .5*z(1,np1,mp1)`). Orders `m >= 2` follow from the
/// recurrence with `a`, `b`, `c` of `rabcp1` (Fortran `zfin1` with `isym = 0`):
/// `Z(m, n) = a Z(m-2, n-2) + b Z(m, n-2) - c Z(m-2, n)`,
/// with the `b` term absent for `n = m` and `n = m + 1`. The halved pole weight is inherited.
///
/// Fortran: sphcom.f / zfini1, zfin1, rabcp, rabcp1
///
/// Difference from Fortran: the `a`, `b`, `c` coefficients are computed in the loop (Fortran
/// stores them in `abc` at index `((m-2)*(nlat+nlat-m-1))/2 + n + 1`); a ring of three `m`-slices
/// replaces `i1`, `i2`, `i3`; the `Z` rows are streamed instead of stored in `wzfin`.
pub(crate) fn compute_regular_scalar_z<F>(
    nlat: usize,
    mmax: usize,
    colats_nh: &[f64],
    mut emit: F,
) where
    F: FnMut(usize, usize, &[f64]),
{
    let imid = colats_nh.len();
    let mut cz = vec![0.0_f64; imid + 1];
    let mut work = vec![0.0_f64; nlat / 2 + 2];

    // 3 rolling m-slices: m-2, m-1, m, each indexed [n * imid + i]
    let slice_len = nlat * imid;
    let mut ring = vec![0.0_f64; 3 * slice_len];

    // m = 0 and m = 1 via dnzfk/dnzft (with pole half-weight at i=0)
    for m in 0..mmax.min(2) {
        let cur = m * slice_len;
        for n in m..nlat {
            dnzfk(nlat, m, n, &mut cz, &mut work);
            let dst = &mut ring[cur + n * imid..cur + (n + 1) * imid];
            for (i, &th) in colats_nh.iter().enumerate() {
                dst[i] = dnzft(nlat, m, n, th, &cz);
            }
            // Half weight at the pole (i = 0), as in Fortran zfini1.
            dst[0] *= 0.5;
            emit(m, n, dst);
        }
    }

    // m >= 2 via rabcp / zfin1 in f64
    // Recurrence for m >= 2 (Fortran zfin1 with isym = 0; coefficients as in rabcp1 for
    // mp1 = 3..mmax). m2 points at the slice for order m-2, cur at the slice for order m.
    for m in 2..mmax {
        let m2 = ((m - 2) % 3) * slice_len;
        let cur = (m % 3) * slice_len;
        let fm = m as f64;
        let tm = fm + fm;

        // n = m
        {
            // n = m (Fortran np1 = m+1): z(i,m+1,i3) = a(ns)*z(i,m-1,i1) - c(ns)*z(i,m+1,i1),
            // a = sqrt((2m+1)(2m-2)/(2m(2m-1))), c = sqrt(2/(2m(2m-1))).
            let temp = tm * (tm - 1.0);
            let a = ((tm + 1.0) * (tm - 2.0) / temp).sqrt();
            let c = (2.0 / temp).sqrt();
            for i in 0..imid {
                ring[cur + m * imid + i] =
                    a * ring[m2 + (m - 2) * imid + i] - c * ring[m2 + m * imid + i];
            }
            emit(m, m, &ring[cur + m * imid..cur + (m + 1) * imid]);
        }

        // n = m + 1
        if m + 1 < nlat {
            // n = m + 1 (Fortran np1 = m+2; skipped when m == nlat-1):
            // a = sqrt((2m+3)(2m-2)/(2m(2m+1))), c = sqrt(6/(2m(2m+1))).
            let temp = tm * (tm + 1.0);
            let a = ((tm + 3.0) * (tm - 2.0) / temp).sqrt();
            let c = (6.0 / temp).sqrt();
            for i in 0..imid {
                ring[cur + (m + 1) * imid + i] =
                    a * ring[m2 + (m - 1) * imid + i] - c * ring[m2 + (m + 1) * imid + i];
            }
            emit(m, m + 1, &ring[cur + (m + 1) * imid..cur + (m + 2) * imid]);
        }

        // n = m + 2 .. nlat
        // n >= m + 2 (Fortran np1 = m+3..nlat): full three-term recurrence,
        // a = sqrt(cn (n+m-3)(n+m-2)/temp), b = sqrt(cn (n-m)(n-m-1)/temp),
        // c = sqrt((n-m+1)(n-m+2)/temp), temp = (n+m)(n+m-1), cn = (2n+1)/(2n-3).
        for n in (m + 2)..nlat {
            let fn_ = n as f64;
            let tn = fn_ + fn_;
            let cn = (tn + 1.0) / (tn - 3.0);
            let fnpm = fn_ + fm;
            let fnmm = fn_ - fm;
            let temp = fnpm * (fnpm - 1.0);
            let a = (cn * (fnpm - 3.0) * (fnpm - 2.0) / temp).sqrt();
            let b = (cn * fnmm * (fnmm - 1.0) / temp).sqrt();
            let c = ((fnmm + 1.0) * (fnmm + 2.0) / temp).sqrt();

            for i in 0..imid {
                ring[cur + n * imid + i] = a * ring[m2 + (n - 2) * imid + i]
                    + b * ring[cur + (n - 2) * imid + i]
                    - c * ring[m2 + n * imid + i];
            }
            emit(m, n, &ring[cur + n * imid..cur + (n + 1) * imid]);
        }
    }
}

/// Compute vector Clenshaw-Curtis quadrature bases $Z_{v,n}^m(\theta_i)$ and $Z_{w,n}^m(\theta_i)$
/// in `f64` for `m = 0..mmax`, `n = m..nlat`, `i = 0..imid`.
///
/// Parameters:
/// - `nlat`: number of colatitudes; `mmax`: number of zonal wavenumbers (Fortran vector
///   `mmax = min(nlat, (nlon+1)/2)`), orders `m = 0..mmax`.
/// - `colats_nh`: northern-hemisphere colatitudes (`imid = (nlat+1)/2` values).
/// - `emit_v(m, n, row)` and `emit_w(m, n, row)`: callbacks receiving the $Z_v$ and $Z_w$ rows
///   (`row.len() == imid`). `emit_v` is called for all `m` first, then `emit_w`.
///
/// $Z_v$: orders `m = 0, 1` from `dzvk`/`dzvt` (Fortran `zvini1`, `mdo = min(2, nlat, (nlon+1)/2)`),
/// pole entry halved, then `m >= 2` by the recurrence with `rabcv1` coefficients (Fortran `zvin1`
/// with `ityp = 0`). $Z_w$: $m = 0$ is identically zero, orders `m = 1, 2` from
/// `dzwk`/`dzwt` (Fortran `zwini1`, `mdo = min(3, ...)`, requires at least 2), then `m >= 3`
/// by the recurrence with `rabcw1` coefficients (Fortran `zwin1`).
///
/// Fortran: sphcom.f / zvini1, zvin1, rabcv, rabcv1 (Z_v); zwini1, zwin1, rabcw, rabcw1 (Z_w)
///
/// Difference from Fortran: as for `compute_regular_scalar_z`; additionally the W rows for
/// `m = 0` are emitted as explicit zeros (Fortran `zwin1` never stores them).
pub(crate) fn compute_regular_vector_zw<FV, FW>(
    nlat: usize,
    mmax: usize,
    colats_nh: &[f64],
    mut emit_v: FV,
    mut emit_w: FW,
) where
    FV: FnMut(usize, usize, &[f64]),
    FW: FnMut(usize, usize, &[f64]),
{
    let imid = colats_nh.len();
    let mut cz = vec![0.0_f64; imid + 1];
    let mut cvw = vec![0.0_f64; nlat / 2 + 2];
    let mut work = vec![0.0_f64; nlat / 2 + 2];

    let slice_len = nlat * imid;
    let mut ring = vec![0.0_f64; 3 * slice_len];

    // --- ZV: m = 0, 1 via dzvk/dzvt; m >= 2 via rabcv / zvin1 ---
    // Orders m = 0, 1 (mmax.min(2)), Fortran zvini1: dzvk/dzvt, then halve the pole entry.
    for m in 0..mmax.min(2) {
        let cur = m * slice_len;
        for n in m..nlat {
            dzvk(nlat, m, n, &mut cz, &mut cvw, &mut work);
            let dst = &mut ring[cur + n * imid..cur + (n + 1) * imid];
            for (i, &th) in colats_nh.iter().enumerate() {
                dst[i] = dzvt(nlat, m, n, th, &cz);
            }
            // Half weight at the pole (i = 0), as in Fortran zvini1.
            dst[0] *= 0.5;
            emit_v(m, n, dst);
        }
    }

    // Recurrence for m >= 2 (Fortran zvin1, coefficients as in rabcv1 for mp1 = 3..mmax):
    // same form as the scalar case with the extra factor tpn inside the square roots.
    for m in 2..mmax {
        let m2 = ((m - 2) % 3) * slice_len;
        let cur = (m % 3) * slice_len;
        let fm = m as f64;
        let tm = fm + fm;

        // n = m
        {
            // n = m: a = sqrt(tpn (2m+1)(2m-2)/temp), tpn = (m-2)(m-1)/(m(m+1)), c = sqrt(2/temp), temp = 2m(2m-1).
            let temp = tm * (tm - 1.0);
            let tpn = (fm - 2.0) * (fm - 1.0) / (fm * (fm + 1.0));
            let a = (tpn * (tm + 1.0) * (tm - 2.0) / temp).sqrt();
            let c = (2.0 / temp).sqrt();
            for i in 0..imid {
                ring[cur + m * imid + i] =
                    a * ring[m2 + (m - 2) * imid + i] - c * ring[m2 + m * imid + i];
            }
            emit_v(m, m, &ring[cur + m * imid..cur + (m + 1) * imid]);
        }

        // n = m + 1
        if m + 1 < nlat {
            // n = m + 1: tpn = (m-1)m/((m+1)(m+2)), a = sqrt(tpn (2m+3)(2m-2)/temp), c = sqrt(6/temp), temp = 2m(2m+1).
            let temp = tm * (tm + 1.0);
            let tpn = (fm - 1.0) * fm / ((fm + 1.0) * (fm + 2.0));
            let a = (tpn * (tm + 3.0) * (tm - 2.0) / temp).sqrt();
            let c = (6.0 / temp).sqrt();
            for i in 0..imid {
                ring[cur + (m + 1) * imid + i] =
                    a * ring[m2 + (m - 1) * imid + i] - c * ring[m2 + (m + 1) * imid + i];
            }
            emit_v(m, m + 1, &ring[cur + (m + 1) * imid..cur + (m + 2) * imid]);
        }

        // n = m + 2 .. nlat
        // n >= m + 2: tpn = (n-2)(n-1)/(n(n+1)); a, b carry tpn*cn inside the square root, c does not.
        for n in (m + 2)..nlat {
            let fn_ = n as f64;
            let tn = fn_ + fn_;
            let cn = (tn + 1.0) / (tn - 3.0);
            let tpn = (fn_ - 2.0) * (fn_ - 1.0) / (fn_ * (fn_ + 1.0));
            let fnpm = fn_ + fm;
            let fnmm = fn_ - fm;
            let temp = fnpm * (fnpm - 1.0);
            let a = (tpn * cn * (fnpm - 3.0) * (fnpm - 2.0) / temp).sqrt();
            let b = (tpn * cn * fnmm * (fnmm - 1.0) / temp).sqrt();
            let c = ((fnmm + 1.0) * (fnmm + 2.0) / temp).sqrt();

            for i in 0..imid {
                ring[cur + n * imid + i] = a * ring[m2 + (n - 2) * imid + i]
                    + b * ring[cur + (n - 2) * imid + i]
                    - c * ring[m2 + n * imid + i];
            }
            emit_v(m, n, &ring[cur + n * imid..cur + (n + 1) * imid]);
        }
    }

    // --- ZW: m = 0 is zero; m = 1, 2 via dzwk/dzwt; m >= 3 via rabcw / zwin1 ---
    ring.fill(0.0);
    // W_n^0 vanishes identically (dwbk returns zeros for m = 0), so emit explicit zero rows.
    if mmax >= 1 {
        let zero_row = vec![0.0_f64; imid];
        for n in 0..nlat {
            emit_w(0, n, &zero_row);
        }
    }

    // Orders m = 1, 2 (mmax.min(3)), Fortran zwini1: dzwk/dzwt, then halve the pole entry.
    for m in 1..mmax.min(3) {
        let cur = (m % 3) * slice_len;
        for n in m..nlat {
            dzwk(nlat, m, n, &mut cz, &mut cvw, &mut work);
            let dst = &mut ring[cur + n * imid..cur + (n + 1) * imid];
            for (i, &th) in colats_nh.iter().enumerate() {
                dst[i] = dzwt(nlat, m, n, th, &cz);
            }
            // Half weight at the pole (i = 0), as in Fortran zwini1.
            dst[0] *= 0.5;
            emit_w(m, n, dst);
        }
    }

    // Recurrence for m >= 3 (Fortran zwin1, coefficients as in rabcw1 for mp1 = 4..mmax):
    // as for Z_v but with the extra factor tph = m/(m-2) on a and c.
    for m in 3..mmax {
        let m2 = ((m - 2) % 3) * slice_len;
        let cur = (m % 3) * slice_len;
        let fm = m as f64;
        let tm = fm + fm;
        let tph = fm / (fm - 2.0);

        // n = m
        {
            // n = m: a = tph sqrt(tpn (2m+1)(2m-2)/temp), c = tph sqrt(2/temp).
            let temp = tm * (tm - 1.0);
            let tpn = (fm - 2.0) * (fm - 1.0) / (fm * (fm + 1.0));
            let a = tph * (tpn * (tm + 1.0) * (tm - 2.0) / temp).sqrt();
            let c = tph * (2.0 / temp).sqrt();
            for i in 0..imid {
                ring[cur + m * imid + i] =
                    a * ring[m2 + (m - 2) * imid + i] - c * ring[m2 + m * imid + i];
            }
            emit_w(m, m, &ring[cur + m * imid..cur + (m + 1) * imid]);
        }

        // n = m + 1
        if m + 1 < nlat {
            // n = m + 1: a = tph sqrt(tpn (2m+3)(2m-2)/temp), c = tph sqrt(6/temp).
            let temp = tm * (tm + 1.0);
            let tpn = (fm - 1.0) * fm / ((fm + 1.0) * (fm + 2.0));
            let a = tph * (tpn * (tm + 3.0) * (tm - 2.0) / temp).sqrt();
            let c = tph * (6.0 / temp).sqrt();
            for i in 0..imid {
                ring[cur + (m + 1) * imid + i] =
                    a * ring[m2 + (m - 1) * imid + i] - c * ring[m2 + (m + 1) * imid + i];
            }
            emit_w(m, m + 1, &ring[cur + (m + 1) * imid..cur + (m + 2) * imid]);
        }

        // n = m + 2 .. nlat
        // n >= m + 2 (note: in rabcw1 the b coefficient has no tph factor, unlike a and c).
        for n in (m + 2)..nlat {
            let fn_ = n as f64;
            let tn = fn_ + fn_;
            let cn = (tn + 1.0) / (tn - 3.0);
            let fnpm = fn_ + fm;
            let fnmm = fn_ - fm;
            let temp = fnpm * (fnpm - 1.0);
            let tpn = (fn_ - 2.0) * (fn_ - 1.0) / (fn_ * (fn_ + 1.0));
            let a = tph * (tpn * cn * (fnpm - 3.0) * (fnpm - 2.0) / temp).sqrt();
            let b = (tpn * cn * fnmm * (fnmm - 1.0) / temp).sqrt();
            let c = tph * ((fnmm + 1.0) * (fnmm + 2.0) / temp).sqrt();

            for i in 0..imid {
                ring[cur + n * imid + i] = a * ring[m2 + (n - 2) * imid + i]
                    + b * ring[cur + (n - 2) * imid + i]
                    - c * ring[m2 + n * imid + i];
            }
            emit_w(m, n, &ring[cur + n * imid..cur + (n + 1) * imid]);
        }
    }
}
