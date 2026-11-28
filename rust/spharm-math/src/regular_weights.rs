//! Exact SPHEREPACK Clenshaw-Curtis quadrature basis functions for regular lat-lon grids
//! (`dnzfk`/`dnzft`, `dzvk`/`dzvt`, `dzwk`/`dzwt`, `dvbk`, `dwbk`, and `rabcp`/`rabcv`/`rabcw`)
//! computed entirely in `f64` without polar division by zero.

use crate::legendre::dnlfk;

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
fn dnzfk(nlat: usize, m: usize, n: usize, cz: &mut [f64], work: &mut [f64]) {
    let lc = (nlat + 1) / 2;
    let sc1 = 2.0 / ((nlat - 1) as f64);
    dnlfk(m as isize, n, work);

    if n % 2 == 0 {
        if m % 2 == 0 {
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
#[inline]
fn dnzft(nlat: usize, m: usize, n: usize, th: f64, cz: &[f64]) -> f64 {
    let mut zh = 0.0_f64;
    let cdt = (th + th).cos();
    let sdt = (th + th).sin();

    if nlat % 2 != 0 {
        let lc = (nlat + 1) / 2;
        let lq = lc - 1;
        let ls = lc.saturating_sub(2);
        if n % 2 == 0 {
            if m % 2 == 0 {
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
        let lc = nlat / 2;
        let lq = lc - 1;
        if n % 2 == 0 {
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
#[inline]
fn dzvt(nlat: usize, m: usize, n: usize, th: f64, czv: &[f64]) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let mut zvh = 0.0_f64;
    let lc = (nlat + 1) / 2;
    let lq = lc - 1;
    let ls = lc.saturating_sub(2);
    let mut cth = th.cos();
    let mut sth = th.sin();
    let cdt = cth * cth - sth * sth;
    let sdt = 2.0 * sth * cth;

    if nlat % 2 != 0 {
        if n % 2 == 0 {
            cth = cdt;
            sth = sdt;
            if m % 2 == 0 {
                for k in 1..=ls {
                    zvh += czv[k] * sth;
                    let chh = cdt * cth - sdt * sth;
                    sth = sdt * cth + cdt * sth;
                    cth = chh;
                }
            } else {
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
            for k in 1..=lq {
                zvh += czv[k] * sth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        } else {
            for k in 1..=lq {
                zvh += czv[k - 1] * cth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        }
    } else if n % 2 == 0 {
        cth = cdt;
        sth = sdt;
        if m % 2 == 0 {
            for k in 1..=lq {
                zvh += czv[k] * sth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        } else {
            zvh = 0.5 * czv[0];
            for k in 2..=lc {
                zvh += czv[k - 1] * cth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        }
    } else if m % 2 == 0 {
        for k in 1..=lq {
            zvh += czv[k] * sth;
            let chh = cdt * cth - sdt * sth;
            sth = sdt * cth + cdt * sth;
            cth = chh;
        }
    } else {
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
#[inline]
fn dzwt(nlat: usize, m: usize, n: usize, th: f64, czw: &[f64]) -> f64 {
    if n == 0 || m == 0 {
        return 0.0;
    }
    let mut zwh = 0.0_f64;
    let lc = (nlat + 1) / 2;
    let lq = lc - 1;
    let ls = lc.saturating_sub(2);
    let mut cth = th.cos();
    let mut sth = th.sin();
    let cdt = cth * cth - sth * sth;
    let sdt = 2.0 * sth * cth;

    if nlat % 2 != 0 {
        if n % 2 == 0 {
            if m % 2 == 0 {
                for k in 1..=lq {
                    zwh += czw[k] * sth;
                    let chh = cdt * cth - sdt * sth;
                    sth = sdt * cth + cdt * sth;
                    cth = chh;
                }
            } else {
                for k in 1..=lq {
                    zwh += czw[k - 1] * cth;
                    let chh = cdt * cth - sdt * sth;
                    sth = sdt * cth + cdt * sth;
                    cth = chh;
                }
            }
        } else {
            cth = cdt;
            sth = sdt;
            if m % 2 == 0 {
                for k in 1..=ls {
                    zwh += czw[k] * sth;
                    let chh = cdt * cth - sdt * sth;
                    sth = sdt * cth + cdt * sth;
                    cth = chh;
                }
            } else {
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
            for k in 1..=lq {
                zwh += czw[k] * sth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        } else {
            zwh = 0.5 * czw[lc - 1] * (((nlat - 1) as f64) * th).cos();
            for k in 1..=lq {
                zwh += czw[k - 1] * cth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        }
    } else {
        cth = cdt;
        sth = sdt;
        if m % 2 == 0 {
            for k in 1..=lq {
                zwh += czw[k] * sth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        } else {
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
            dst[0] *= 0.5;
            emit(m, n, dst);
        }
    }

    // m >= 2 via rabcp / zfin1 in f64
    for m in 2..mmax {
        let m2 = ((m - 2) % 3) * slice_len;
        let cur = (m % 3) * slice_len;
        let fm = m as f64;
        let tm = fm + fm;

        // n = m
        {
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
    for m in 0..mmax.min(2) {
        let cur = m * slice_len;
        for n in m..nlat {
            dzvk(nlat, m, n, &mut cz, &mut cvw, &mut work);
            let dst = &mut ring[cur + n * imid..cur + (n + 1) * imid];
            for (i, &th) in colats_nh.iter().enumerate() {
                dst[i] = dzvt(nlat, m, n, th, &cz);
            }
            dst[0] *= 0.5;
            emit_v(m, n, dst);
        }
    }

    for m in 2..mmax {
        let m2 = ((m - 2) % 3) * slice_len;
        let cur = (m % 3) * slice_len;
        let fm = m as f64;
        let tm = fm + fm;

        // n = m
        {
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
    if mmax >= 1 {
        let zero_row = vec![0.0_f64; imid];
        for n in 0..nlat {
            emit_w(0, n, &zero_row);
        }
    }

    for m in 1..mmax.min(3) {
        let cur = (m % 3) * slice_len;
        for n in m..nlat {
            dzwk(nlat, m, n, &mut cz, &mut cvw, &mut work);
            let dst = &mut ring[cur + n * imid..cur + (n + 1) * imid];
            for (i, &th) in colats_nh.iter().enumerate() {
                dst[i] = dzwt(nlat, m, n, th, &cz);
            }
            dst[0] *= 0.5;
            emit_w(m, n, dst);
        }
    }

    for m in 3..mmax {
        let m2 = ((m - 2) % 3) * slice_len;
        let cur = (m % 3) * slice_len;
        let fm = m as f64;
        let tm = fm + fm;
        let tph = fm / (fm - 2.0);

        // n = m
        {
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
