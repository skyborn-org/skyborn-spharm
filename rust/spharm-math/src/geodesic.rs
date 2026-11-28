//! Icosahedral geodesic grid generator matching SPHEREPACK `ihgeod` and `getgeodesicpts`.

use std::f64::consts::PI;

#[inline]
fn stoc(r: f64, theta: f64, phi: f64) -> (f64, f64, f64) {
    let (st, ct) = theta.sin_cos();
    let (sp, cp) = phi.sin_cos();
    (r * st * cp, r * st * sp, r * ct)
}

#[inline]
fn ctos(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let r1 = x * x + y * y;
    if r1 < 1.0e-14 {
        let theta = if z < 0.0 { PI } else { 0.0 };
        return (z.abs(), theta, 0.0);
    }
    let r1_sqrt = r1.sqrt();
    let r = (r1 + z * z).sqrt();
    let phi = y.atan2(x);
    let theta = r1_sqrt.atan2(z);
    (r, theta, phi)
}

/// Compute `(latitudes_deg, longitudes_deg)` of the SPHEREPACK icosahedral geodesic mesh
/// of resolution `m >= 2` (total points = `10 * (m - 1)^2 + 2`).
///
/// Output ordering and conventions match `skybornspharm.getgeodesicpts(m)` / `ihgeod`.
pub fn icosahedral_geodesic(m: usize) -> Result<(Vec<f32>, Vec<f32>), String> {
    if m < 2 {
        return Err(format!("m must be >= 2, got {m}"));
    }

    let idp = 2 * m - 1;
    let jdp = m;
    let len = idp * jdp * 5;
    let mut x = vec![0.0_f64; len];
    let mut y = vec![0.0_f64; len];
    let mut z = vec![0.0_f64; len];

    // 0-based indexer corresponding to Fortran (j, i, k) in (idp, jdp, 5)
    let idx = |j: usize, i: usize, k: usize| -> usize { (k * jdp + i) * idp + j };

    let dphi = 0.4 * PI;
    let beta = dphi.cos();
    let theta1 = (beta / (1.0 - beta)).acos();
    let theta2 = PI - theta1;
    let hdphi = 0.5 * dphi;
    let tdphi = 3.0 * hdphi;
    let mf = (m - 1) as f64;

    for k in 0..5 {
        let phi = (k as f64) * dphi;
        let (x1, y1, z1) = stoc(1.0, theta2, phi);
        let (x2, y2, z2) = stoc(1.0, PI, phi + hdphi);
        let (x3, y3, z3) = stoc(1.0, theta2, phi + dphi);

        // First triangle
        let dxi = (x2 - x1) / mf;
        let dyi = (y2 - y1) / mf;
        let dzi = (z2 - z1) / mf;
        let dxj = (x3 - x2) / mf;
        let dyj = (y3 - y2) / mf;
        let dzj = (z3 - z2) / mf;
        for i in 0..m {
            let fi = i as f64;
            let xs = x1 + fi * dxi;
            let ys = y1 + fi * dyi;
            let zs = z1 + fi * dzi;
            for j in 0..=i {
                let fj = j as f64;
                let p = idx(j, i, k);
                x[p] = xs + fj * dxj;
                y[p] = ys + fj * dyj;
                z[p] = zs + fj * dzj;
            }
        }

        // Second triangle
        let (x4, y4, z4) = stoc(1.0, theta1, phi + hdphi);
        let dxi = (x3 - x4) / mf;
        let dyi = (y3 - y4) / mf;
        let dzi = (z3 - z4) / mf;
        let dxj = (x4 - x1) / mf;
        let dyj = (y4 - y1) / mf;
        let dzj = (z4 - z1) / mf;
        for j in 0..m {
            let fj = j as f64;
            let xs = x1 + fj * dxj;
            let ys = y1 + fj * dyj;
            let zs = z1 + fj * dzj;
            for i in 0..=j {
                let fi = i as f64;
                let p = idx(j, i, k);
                x[p] = xs + fi * dxi;
                y[p] = ys + fi * dyi;
                z[p] = zs + fi * dzi;
            }
        }

        // Third triangle
        let (x5, y5, z5) = stoc(1.0, theta1, phi + tdphi);
        let dxj = (x5 - x3) / mf;
        let dyj = (y5 - y3) / mf;
        let dzj = (z5 - z3) / mf;
        for i in 0..m {
            let fi = i as f64;
            let xs = x4 + fi * dxi;
            let ys = y4 + fi * dyi;
            let zs = z4 + fi * dzi;
            for j in 0..=i {
                let fj = j as f64;
                let p = idx(j + m - 1, i, k);
                x[p] = xs + fj * dxj;
                y[p] = ys + fj * dyj;
                z[p] = zs + fj * dzj;
            }
        }

        // Fourth triangle
        let (x6, y6, z6) = stoc(1.0, 0.0, phi + dphi);
        let dxi = (x5 - x6) / mf;
        let dyi = (y5 - y6) / mf;
        let dzi = (z5 - z6) / mf;
        let dxj = (x6 - x4) / mf;
        let dyj = (y6 - y4) / mf;
        let dzj = (z6 - z4) / mf;
        for j in 0..m {
            let fj = j as f64;
            let xs = x4 + fj * dxj;
            let ys = y4 + fj * dyj;
            let zs = z4 + fj * dzj;
            for i in 0..=j {
                let fi = i as f64;
                let p = idx(j + m - 1, i, k);
                x[p] = xs + fi * dxi;
                y[p] = ys + fi * dyi;
                z[p] = zs + fi * dzi;
            }
        }
    }

    // Reproject to unit sphere just like ihgeod lines 148-156
    for k in 0..5 {
        for j in 0..idp {
            for i in 0..m {
                let p = idx(j, i, k);
                let (_rad, theta, phi) = ctos(x[p], y[p], z[p]);
                (x[p], y[p], z[p]) = stoc(1.0, theta, phi);
            }
        }
    }

    let total_points = 10 * (m - 1) * (m - 1) + 2;
    let mut lats = Vec::with_capacity(total_points);
    let mut lons = Vec::with_capacity(total_points);
    lats.push(90.0_f32);
    lats.push(-90.0_f32);
    lons.push(0.0_f32);
    lons.push(0.0_f32);

    let rad_to_deg = 180.0 / PI;
    // Match NumPy `lats[0 : 2*(m-1), 0 : m-1, :].flatten()` (C-order: j outer, i middle, k inner)
    for j in 0..(2 * (m - 1)) {
        for i in 0..(m - 1) {
            for k in 0..5 {
                let p = idx(j, i, k);
                let xv = x[p] as f32;
                let yv = y[p] as f32;
                let zv = z[p] as f32;
                let r1 = (xv * xv + yv * yv).sqrt();
                let xtmp = if xv != 0.0 || yv != 0.0 { xv } else { 1.0 };
                let ztmp = if r1 != 0.0 || zv != 0.0 { zv } else { 1.0 };
                let lon = (rad_to_deg as f32) * yv.atan2(xtmp) + 180.0_f32;
                let lat = (rad_to_deg as f32) * r1.atan2(ztmp) - 90.0_f32;
                lats.push(lat);
                lons.push(lon);
            }
        }
    }

    Ok((lats, lons))
}
