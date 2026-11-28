//! Gaussian quadrature colatitudes, latitudes, and weights matching SPHEREPACK `gaqd`.

use std::f64::consts::PI;

/// Kahan compensated summation for improved accuracy on large arrays.
#[inline]
fn kahan_sum(values: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    let mut c = 0.0_f64;
    for &v in values {
        let y = v - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

/// Compute Fourier-cosine series coefficients `cz, cp, dcp` for $P_n(\cos\theta)$
/// matching SPHEREPACK `cpdp`.
fn cpdp(n: usize) -> (f64, Vec<f64>, Vec<f64>) {
    let ncp = (n + 1) / 2;
    let mut cp = vec![0.0_f64; ncp];
    let mut dcp = vec![0.0_f64; ncp];

    let mut t1 = -1.0_f64;
    let mut t2 = (n + 1) as f64;
    let mut t3 = 0.0_f64;
    let mut t4 = (2 * n + 1) as f64;

    let cz;
    if n % 2 == 0 {
        cp[ncp - 1] = 1.0;
        for j in (2..=ncp).rev() {
            t1 += 2.0;
            t2 -= 1.0;
            t3 += 1.0;
            t4 -= 2.0;
            cp[j - 2] = (t1 * t2) / (t3 * t4) * cp[j - 1];
        }
        t1 += 2.0;
        t2 -= 1.0;
        t3 += 1.0;
        t4 -= 2.0;
        cz = (t1 * t2) / (t3 * t4) * cp[0];
        for j in 0..ncp {
            dcp[j] = (2 * (j + 1)) as f64 * cp[j];
        }
    } else {
        cp[ncp - 1] = 1.0;
        for j in (1..ncp).rev() {
            t1 += 2.0;
            t2 -= 1.0;
            t3 += 1.0;
            t4 -= 2.0;
            cp[j - 1] = (t1 * t2) / (t3 * t4) * cp[j];
        }
        cz = 0.0;
        for j in 0..ncp {
            dcp[j] = (2 * j + 1) as f64 * cp[j];
        }
    }

    (cz, cp, dcp)
}

/// Evaluate $P_n(\cos\theta)$ and $dP_n(\cos\theta)/d\theta$ matching SPHEREPACK `tpdp`.
#[inline]
fn tpdp(n: usize, theta: f64, cz: f64, cp: &[f64], dcp: &[f64]) -> (f64, f64) {
    let cdt = (theta + theta).cos();
    let sdt = (theta + theta).sin();

    if n % 2 == 0 {
        let kdo = n / 2;
        let mut pb = 0.5 * cz;
        let mut dpb = 0.0_f64;
        if n > 0 {
            let mut cth = cdt;
            let mut sth = sdt;
            for k in 0..kdo {
                pb += cp[k] * cth;
                dpb -= dcp[k] * sth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        }
        (pb, dpb)
    } else {
        let kdo = (n + 1) / 2;
        let mut pb = 0.0_f64;
        let mut dpb = 0.0_f64;
        let mut cth = theta.cos();
        let mut sth = theta.sin();
        for k in 0..kdo {
            pb += cp[k] * cth;
            dpb -= dcp[k] * sth;
            let chh = cdt * cth - sdt * sth;
            sth = sdt * cth + cdt * sth;
            cth = chh;
        }
        (pb, dpb)
    }
}

/// Compute Gaussian colatitudes (in radians, `0..pi`) and quadrature weights (summing to `2.0`),
/// matching SPHEREPACK `gaqd` in double precision.
pub fn gaussian_colats_wts(nlat: usize) -> Result<(Vec<f64>, Vec<f64>), String> {
    if nlat == 0 {
        return Err("nlat must be >= 1".to_string());
    }

    if nlat == 1 {
        return Ok((vec![0.0_f64.acos()], vec![2.0]));
    }
    if nlat == 2 {
        let x = (1.0_f64 / 3.0_f64).sqrt();
        return Ok((vec![x.acos(), (-x).acos()], vec![1.0, 1.0]));
    }

    let mut theta = vec![0.0_f64; nlat];
    let mut wts = vec![0.0_f64; nlat];

    let eps_sqrt = f64::EPSILON.sqrt();
    let eps = eps_sqrt * eps_sqrt.sqrt();
    let pis2 = PI * 0.5;
    let mnlat = nlat % 2;
    let ns2 = nlat / 2;
    let nhalf = (nlat + 1) / 2;

    let (cz, cp, dcp) = cpdp(nlat);

    let dtheta = pis2 / (nhalf as f64);
    let dthalf = 0.5 * dtheta;
    let cmax = 0.2 * dtheta;

    let mut zero;
    let mut zprev = pis2;
    let mut nix;
    if mnlat != 0 {
        zero = pis2 - dtheta;
        nix = nhalf - 1;
    } else {
        zero = pis2 - dthalf;
        nix = nhalf;
    }

    while nix > 0 {
        let mut it = 0usize;
        let mut zlast;
        let mut pb;
        let mut dpb;
        loop {
            it += 1;
            zlast = zero;
            (pb, dpb) = tpdp(nlat, zero, cz, &cp, &dcp);

            let mut dcor = if dpb.abs() < f64::EPSILON * 1000.0 {
                eps.copysign(pb)
            } else {
                pb / dpb
            };

            let sgnd = if dcor != 0.0 { dcor / dcor.abs() } else { 1.0 };
            dcor = sgnd * dcor.abs().min(cmax);
            zero -= dcor;

            if (zero - zlast).abs() <= eps * zero.abs().max(1.0) || it > 50 {
                break;
            }
        }

        let idx = nix - 1;
        theta[idx] = zero;
        let zhold = zero;

        let sin_z = zlast.sin();
        if sin_z.abs() > f64::EPSILON * 1000.0 {
            let denom = dpb + pb * zlast.cos() / sin_z;
            wts[idx] = ((2 * nlat + 1) as f64) / (denom * denom);
        } else {
            wts[idx] = ((2 * nlat + 1) as f64) / (dpb * dpb);
        }

        nix -= 1;
        if nix == 0 {
            break;
        }
        if nix == nhalf - 1 {
            zero = 3.0 * zero - PI;
        } else if nix < nhalf - 1 {
            zero = zero + zero - zprev;
        }
        zprev = zhold;
    }

    if mnlat != 0 {
        theta[nhalf - 1] = pis2;
        let (_pb, dpb) = tpdp(nlat, pis2, cz, &cp, &dcp);
        wts[nhalf - 1] = ((2 * nlat + 1) as f64) / (dpb * dpb);
    }

    for i in 0..ns2 {
        wts[nlat - 1 - i] = wts[i];
        theta[nlat - 1 - i] = PI - theta[i];
    }

    let mut sum = kahan_sum(&wts);
    if sum.abs() < f64::EPSILON * (nlat as f64) {
        sum = 2.0;
    }
    let scale = 2.0 / sum;
    for w in &mut wts {
        *w *= scale;
    }

    Ok((theta, wts))
}

/// Compute Gaussian latitudes (degrees north, descending north-to-south) and quadrature weights,
/// matching `skybornspharm.gaussian_lats_wts`.
pub fn gaussian_lats_wts(nlat: usize) -> Result<(Vec<f64>, Vec<f64>), String> {
    let (colats, wts) = gaussian_colats_wts(nlat)?;
    let rad_to_deg = 180.0 / PI;
    let lats = colats.into_iter().map(|th| 90.0 - th * rad_to_deg).collect();
    Ok((lats, wts))
}
