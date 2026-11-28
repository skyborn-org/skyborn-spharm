//! Gaussian quadrature colatitudes, latitudes, and weights matching SPHEREPACK `gaqd`.
//! Derived from NCAR SPHEREPACK 3.2 (J. Adams and P. Swarztrauber), see LICENSE.spherepack.
//!
//! Fortran reference: `gaqd.f` (`gaqd`, `cpdp`, `tpdp`, `dzeps`). The Fortran header states that the
//! Gauss points and weights are computed with the Fourier-Newton method described in P. N.
//! Swarztrauber, "Computing the points and weights for Gauss-Legendre quadrature", SIAM J. Sci.
//! Comput., 24 (2002), pp. 945-954. This is the version introduced in SPHEREPACK 3.1; the Fortran
//! header describes it as faster and more accurate than the older routine of the same name in
//! SPHEREPACK 3.0. The `w` and `lwork` arguments of the Fortran `gaqd` are dummies kept only so the
//! routine can be exchanged with the old `gaqd`.
//!
//! Algorithm outline (from the Fortran source):
//! 1. $P_n(\cos\theta)$ is written as a finite cosine series in $\theta$ whose coefficients come
//!    from a backward recurrence (`cpdp`). The series and its $\theta$-derivative are summed with an
//!    angle-addition recurrence (`tpdp`).
//! 2. The roots in $(0, \pi/2]$ are found one at a time by Newton iteration, starting next to
//!    $\theta = \pi/2$. The step is limited to `0.2 * dtheta` (`cmax`). Later starting guesses are
//!    extrapolated from the previous roots.
//! 3. Weights use Yakimiw's formula, which permits reusing the last `pb` and `dpb` of the Newton
//!    iteration. The remaining points and weights follow from the north-south symmetry, and the
//!    weights are finally rescaled so that they sum to 2.
//!
//! Difference from Fortran: everything is computed in `f64`, there are no `w`/`lwork` dummy arguments,
//! `ierror` is replaced by `Result`, and the coefficient arrays of `cpdp` are separate `Vec`s instead of
//! being stored temporarily in the upper half of `theta` and `wts` (Fortran passes `theta(ns2+1)` and
//! `wts(ns2+1)` as scratch).

use std::f64::consts::PI;

/// Kahan compensated summation for improved accuracy on large arrays.
///
/// No direct Fortran counterpart. The Fortran `gaqd` accumulates the weight sum with a plain loop
/// (`sum = sum + wts(i)`); this helper uses compensated (Kahan) summation for the same purpose, the
/// final normalization of the weights.
///
/// Difference from Fortran: compensated instead of plain summation.
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
///
/// Computes the Fourier coefficients of the Legendre polynomial $P_n^0$ and of its derivative.
///
/// Fortran header: "computes the fourier coefficients of the legendre polynomial p_n^0 and its
/// derivative. n is the degree and n/2 or (n+1)/2 coefficients are returned in cp depending on whether
/// n is even or odd. The same number of coefficients are returned in dcp. For n even the constant
/// coefficient is returned in cz."
///
/// Parameters:
/// - `n`: degree of the polynomial (the Gaussian point count `nlat` in the caller).
///
/// Returns `(cz, cp, dcp)`:
/// - `cz`: constant coefficient (`0.0` for odd `n`).
/// - `cp`: `(n + 1) / 2` coefficients. Series, with 1-based `k` as in Fortran:
///   - `n` even: $P_n(\cos\theta) = cz/2 + \sum_{k=1}^{n/2} cp(k) \cos(2k\theta)$
///   - `n` odd: $P_n(\cos\theta) = \sum_{k=1}^{(n+1)/2} cp(k) \cos((2k-1)\theta)$
/// - `dcp`: derivative coefficients, `dcp(k) = 2k * cp(k)` for `n` even and
///   `dcp(k) = (2k-1) * cp(k)` for `n` odd.
///
/// The highest coefficient is set to 1 and the others follow from a backward recurrence in the ratios
/// `t1*t2/(t3*t4)`. The coefficients are therefore not normalized. This is harmless here because the
/// Newton step `pb / dpb` and the final weight normalization are both insensitive to a common scale.
///
/// Fortran: gaqd.f / cpdp
///
/// Difference from Fortran: 0-based indexing (`cp[j - 1]` is Fortran `cp(j)`), and the vectors have
/// exactly `(n + 1) / 2` entries instead of the `n/2 + 1` dimension declared in Fortran.
fn cpdp(n: usize) -> (f64, Vec<f64>, Vec<f64>) {
    let ncp = (n + 1) / 2;
    let mut cp = vec![0.0_f64; ncp];
    let mut dcp = vec![0.0_f64; ncp];

    // Recurrence state: the four running terms t1..t4 form the ratio t1*t2/(t3*t4) used for each
    // backward step (Fortran `t1 = -1`, `t2 = n + 1`, `t3 = 0`, `t4 = 2n + 1`).
    let mut t1 = -1.0_f64;
    let mut t2 = (n + 1) as f64;
    let mut t3 = 0.0_f64;
    let mut t4 = (2 * n + 1) as f64;

    let cz;
    if n % 2 == 0 {
        // n even: the highest coefficient (index ncp, Fortran cp(ncp)) is set to 1, then the lower ones follow from the backward recurrence down to cp(1); cz comes from one more step.
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
        // One more recurrence step gives the constant coefficient cz (Fortran `cz = (t1*t2)/(t3*t4)*cp(1)`).
        cz = (t1 * t2) / (t3 * t4) * cp[0];
        for j in 0..ncp {
            // Derivative coefficient: Fortran `dcp(j) = (j+j)*cp(j)` with 1-based j, d/dtheta of cos(2*j*theta) gives the factor 2*j.
            dcp[j] = (2 * (j + 1)) as f64 * cp[j];
        }
    } else {
        // n odd: same start (cp(ncp) = 1) but the loop stops at j = 1 and there is no constant coefficient (cz = 0).
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
            // Derivative coefficient: Fortran `dcp(j) = (j+j-1)*cp(j)` with 1-based j, the factor is (2*j - 1) for cos((2*j-1)*theta).
            dcp[j] = (2 * j + 1) as f64 * cp[j];
        }
    }

    (cz, cp, dcp)
}

/// Evaluate $P_n(\cos\theta)$ and $dP_n(\cos\theta)/d\theta$ matching SPHEREPACK `tpdp`.
///
/// Computes $P_n(\cos\theta)$ and its derivative with respect to $\theta$.
///
/// Fortran header: "computes pn(theta) and its derivative dpb(theta) with respect to theta".
///
/// Parameters:
/// - `n`: degree of the polynomial.
/// - `theta`: colatitude in radians.
/// - `cz`, `cp`, `dcp`: coefficients from [`cpdp`] for the same `n`.
///
/// Returns `(pb, dpb)`, the polynomial value and its $\theta$-derivative.
///
/// Algorithm: the cosine series of `cpdp` is summed term by term. Instead of calling `cos`/`sin` for
/// every term, the angle-addition recurrence
/// `cos((k+1)a) = cos(a) cos(ka) - sin(a) sin(ka)` and
/// `sin((k+1)a) = sin(a) cos(ka) + cos(a) sin(ka)` with `a = 2 * theta` is used.
///
/// Fortran: gaqd.f / tpdp
#[inline]
fn tpdp(n: usize, theta: f64, cz: f64, cp: &[f64], dcp: &[f64]) -> (f64, f64) {
    // cdt, sdt: cos and sin of 2*theta, the fixed step of the angle-addition recurrence (cth, sth hold cos and sin of the current multiple).
    let cdt = (theta + theta).cos();
    let sdt = (theta + theta).sin();

    if n % 2 == 0 {
        // n even: kdo = n/2 terms. pb = cz/2 + sum_{k=1..kdo} cp(k) cos(2*k*theta). For n = 0 only the constant term remains.
        let kdo = n / 2;
        let mut pb = 0.5 * cz;
        let mut dpb = 0.0_f64;
        if n > 0 {
            let mut cth = cdt;
            let mut sth = sdt;
            for k in 0..kdo {
                // Fortran comment: pb = pb + cp(k)*cos(2*k*theta), evaluated through the recurrence value cth.
                pb += cp[k] * cth;
                // Fortran comment: dpb = dpb - (k+k)*cp(k)*sin(2*k*theta), with dcp(k) = (k+k)*cp(k).
                dpb -= dcp[k] * sth;
                // Advance (cth, sth) from angle j*a to (j+1)*a; chh keeps the old sth from being overwritten too early.
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        }
        (pb, dpb)
    } else {
        // n odd: kdo = (n+1)/2 terms. pb = sum_{k=1..kdo} cp(k) cos((2*k-1)*theta); the recurrence starts at theta itself.
        let kdo = (n + 1) / 2;
        let mut pb = 0.0_f64;
        let mut dpb = 0.0_f64;
        let mut cth = theta.cos();
        let mut sth = theta.sin();
        for k in 0..kdo {
            // Fortran comment: pb = pb + cp(k)*cos((2*k-1)*theta).
            pb += cp[k] * cth;
            // Fortran comment: dpb = dpb - (k+k-1)*cp(k)*sin((2*k-1)*theta).
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
///
/// Parameters:
/// - `nlat`: number of Gaussian colatitudes in $(0, \pi)$ (between the two poles); must be at least 1.
///
/// Returns `(theta, wts)`, each of length `nlat`: `theta` holds the colatitudes in radians in
/// increasing order (index 0 is the point nearest the north pole), `wts` the Gaussian weights, which are
/// rescaled to sum to 2.
///
/// Errors: Fortran `ierror = 1` if `nlat <= 0`; here `Err("nlat must be >= 1")` for `nlat == 0`
/// (`nlat` is unsigned). Fortran `ierror = 0` corresponds to `Ok`.
///
/// Algorithm (Fortran `gaqd`): `nlat = 1` and `nlat = 2` are closed forms ($\theta = \pi/2$, weight 2;
/// and $\theta = \arccos(\pm 1/\sqrt{3})$, weights 1). Otherwise Newton iteration is run for the
/// `nhalf = (nlat + 1) / 2` roots in the northern half, the first one starting next to
/// $\theta = \pi/2$, and the other points and weights are obtained from the symmetries
/// `theta(nlat - i + 1) = pi - theta(i)` and `wts(nlat - i + 1) = wts(i)`. For odd `nlat` the
/// equatorial point is $\pi/2$.
///
/// Fortran: gaqd.f / gaqd
///
/// Difference from Fortran:
/// - The Fortran routine takes the dummy arguments `w` and `lwork` and reports status in `ierror`;
///   here the result is a `Result` and no work arrays exist.
/// - Machine epsilon: Fortran computes `eps = sqrt(dzeps(1.0d0))` and then `eps = eps*sqrt(eps)`, where
///   `dzeps` estimates the unit roundoff (and is declared `real`, so it returns a default `REAL`);
///   here `f64::EPSILON` is used in the same formula, giving `eps = EPSILON^(3/4)`.
/// - Newton safeguards added in Rust: if `|dpb| < 1000 * EPSILON` the correction is replaced by
///   `eps` carrying the sign of `pb` (avoiding a division by a vanishing derivative); the stopping test is
///   `|zero - zlast| <= eps * max(|zero|, 1)` instead of `eps * |zero|`; and at most 50 iterations
///   are performed per root. The Fortran loop has none of these.
/// - The weight sum uses compensated summation, and if the sum is smaller than `EPSILON * nlat` it is
///   replaced by 2 before the normalization (a guard absent in Fortran).
pub fn gaussian_colats_wts(nlat: usize) -> Result<(Vec<f64>, Vec<f64>), String> {
    if nlat == 0 {
        return Err("nlat must be >= 1".to_string());
    }

    // Compute weights and points analytically when nlat = 1, 2 (Fortran `gaqd` does the same).
    if nlat == 1 {
        return Ok((vec![0.0_f64.acos()], vec![2.0]));
    }
    if nlat == 2 {
        let x = (1.0_f64 / 3.0_f64).sqrt();
        return Ok((vec![x.acos(), (-x).acos()], vec![1.0, 1.0]));
    }

    let mut theta = vec![0.0_f64; nlat];
    let mut wts = vec![0.0_f64; nlat];

    // Newton tolerance eps = (machine eps)^(3/4), Fortran: `eps = sqrt(dzeps(1.0d0)); eps = eps*sqrt(eps)`.
    let eps_sqrt = f64::EPSILON.sqrt();
    let eps = eps_sqrt * eps_sqrt.sqrt();
    let pis2 = PI * 0.5;
    // mnlat: parity of nlat (1 = odd, an equatorial point exists). ns2: points per hemisphere excluding the equator. nhalf: points in the northern half including the equator.
    let mnlat = nlat % 2;
    let ns2 = nlat / 2;
    let nhalf = (nlat + 1) / 2;

    // Fourier coefficients of P_nlat and its derivative (Fortran stores them in theta(ns2+1) and wts(ns2+1) as scratch).
    let (cz, cp, dcp) = cpdp(nlat);

    // dtheta: nominal point spacing near the equator. dthalf: half of it. cmax: largest Newton correction allowed per step (0.2 * dtheta).
    let dtheta = pis2 / (nhalf as f64);
    let dthalf = 0.5 * dtheta;
    let cmax = 0.2 * dtheta;

    // Estimate the first point next to theta = pi/2: one spacing away for odd nlat (the equator itself is a root), half a spacing away for even nlat.
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

    // Roots are found from the equator toward the north pole; nix is the 1-based position of the root being computed (Fortran label 9).
    while nix > 0 {
        let mut it = 0usize;
        let mut zlast;
        let mut pb;
        let mut dpb;
        loop {
            it += 1;
            zlast = zero;
            // Newton iterations (Fortran label 10): evaluate P_nlat and its derivative at the current estimate.
            (pb, dpb) = tpdp(nlat, zero, cz, &cp, &dcp);

            let mut dcor = if dpb.abs() < f64::EPSILON * 1000.0 {
                eps.copysign(pb)
            } else {
                pb / dpb
            };

            // Limit the correction to cmax in magnitude while keeping its sign (Fortran: `dcor = sgnd*min(dabs(dcor),cmax)`).
            let sgnd = if dcor != 0.0 { dcor / dcor.abs() } else { 1.0 };
            dcor = sgnd * dcor.abs().min(cmax);
            zero -= dcor;

            if (zero - zlast).abs() <= eps * zero.abs().max(1.0) || it > 50 {
                break;
            }
        }

        let idx = nix - 1;
        // Converged root. Fortran comment: `wts(nix) = (nlat+nlat+1)/(dpb*dpb)` is the textbook weight; Yakimiw's formula below permits using the old pb and dpb.
        theta[idx] = zero;
        let zhold = zero;

        // Yakimiw's formula: wts = (2*nlat + 1) / (dpb + pb*cos(zlast)/sin(zlast))^2. The guard on sin(zlast) is a Rust addition that falls back to the plain formula near the pole.
        let sin_z = zlast.sin();
        if sin_z.abs() > f64::EPSILON * 1000.0 {
            let denom = dpb + pb * zlast.cos() / sin_z;
            wts[idx] = ((2 * nlat + 1) as f64) / (denom * denom);
        } else {
            wts[idx] = ((2 * nlat + 1) as f64) / (dpb * dpb);
        }

        // Starting guess for the next root: after the first root use 3*zero - pi, afterwards linear extrapolation zero + zero - zprev (Fortran lines `if(nix.eq.nhalf-1) zero = 3.0*zero-pi` and `if(nix.lt.nhalf-1) zero = zero+zero-zprev`).
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

    // Extend points and weights via symmetries (Fortran label 30). For odd nlat the equatorial point is exactly pi/2 and its weight uses the plain formula (2*nlat + 1) / dpb^2.
    if mnlat != 0 {
        theta[nhalf - 1] = pis2;
        let (_pb, dpb) = tpdp(nlat, pis2, cz, &cp, &dcp);
        wts[nhalf - 1] = ((2 * nlat + 1) as f64) / (dpb * dpb);
    }

    // Mirror the northern half: wts(nlat - i + 1) = wts(i) and theta(nlat - i + 1) = pi - theta(i) (1-based Fortran indices).
    for i in 0..ns2 {
        wts[nlat - 1 - i] = wts[i];
        theta[nlat - 1 - i] = PI - theta[i];
    }

    // Normalise so that the weights sum to 2 (Fortran: `wts(i) = 2.0d0*wts(i)/sum`).
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
///
/// No direct Fortran counterpart: SPHEREPACK `gaqd` returns colatitudes only. This wrapper converts the
/// `gaqd` colatitudes to latitudes, `lat = 90 - colat_deg`, so index 0 is the northernmost latitude.
///
/// Parameters: `nlat`, the number of Gaussian latitudes (at least 1). Returns `(lats, wts)`, each of
/// length `nlat`; the weights are those of [`gaussian_colats_wts`] and sum to 2. Errors are propagated
/// from [`gaussian_colats_wts`].
pub fn gaussian_lats_wts(nlat: usize) -> Result<(Vec<f64>, Vec<f64>), String> {
    let (colats, wts) = gaussian_colats_wts(nlat)?;
    let rad_to_deg = 180.0 / PI;
    let lats = colats.into_iter().map(|th| 90.0 - th * rad_to_deg).collect();
    Ok((lats, wts))
}
