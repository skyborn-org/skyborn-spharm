//! Icosahedral geodesic grid generator matching SPHEREPACK `ihgeod` and `getgeodesicpts`.
//! Derived from NCAR SPHEREPACK 3.2 (J. Adams and P. Swarztrauber), see LICENSE.spherepack.
//!
//! Fortran reference: `ihgeod.f` (`ihgeod`, `ctos`, `stoc`). The mesh is built on an unfolded
//! icosahedron: for each of 5 longitudinal sectors, four flat triangles are subdivided into `m - 1`
//! segments per edge, and every point is then projected radially onto the unit sphere.
//!
//! Difference from Fortran: `ihgeod` writes Cartesian `x, y, z` arrays of shape `(idp, jdp, 5)` in
//! single precision; here the same arrays are built in `f64` (stored column-major, first index fastest,
//! to mirror the Fortran layout), and only the latitude and longitude of the distinct points are
//! returned (see [`icosahedral_geodesic`]).

use std::f64::consts::PI;

/// Spherical to Cartesian conversion.
///
/// Parameters: `r` radius, `theta` colatitude in radians, `phi` longitude in radians. Returns
/// `(x, y, z) = (r sin(theta) cos(phi), r sin(theta) sin(phi), r cos(theta))`.
///
/// Fortran: ihgeod.f / stoc
///
/// Difference from Fortran: returns a tuple instead of writing output arguments; `f64` instead of
/// single precision.
#[inline]
fn stoc(r: f64, theta: f64, phi: f64) -> (f64, f64, f64) {
    let (st, ct) = theta.sin_cos();
    let (sp, cp) = phi.sin_cos();
    (r * st * cp, r * st * sp, r * ct)
}

/// Cartesian to spherical conversion.
///
/// Parameters: Cartesian coordinates `x, y, z`. Returns `(r, theta, phi)` with `theta` the colatitude
/// and `phi = atan2(y, x)` the longitude, both in radians. On the polar axis (`x` and `y` both
/// vanishing) `phi = 0` and `theta` is `0`, or `pi` when `z < 0`.
///
/// Fortran: ihgeod.f / ctos
///
/// Difference from Fortran: the Fortran test for the polar axis is exact (`r1 .ne. 0.`) and leaves `r`
/// unset on that branch; here the test is `x*x + y*y < 1e-14` and `r = |z|` is returned.
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
///
/// Parameters: `m`, the number of points on the edge of a single geodesic triangle (Fortran header).
/// Must be at least 2; Fortran divides by `m - 1`.
///
/// Returns `(lats, lons)` in degrees as `f32`, each of length `10 * (m - 1)^2 + 2`. The first two
/// entries are the poles, `(90, 0)` and `(-90, 0)`, followed by the remaining points flattened in the
/// order j (outer), i (middle), k (inner) over `j in 0..2*(m-1)`, `i in 0..m-1`, `k in 0..5`. The
/// longitude is computed as `atan2(y, x)` in degrees plus 180, and the latitude as
/// `atan2(sqrt(x^2 + y^2), z)` in degrees minus 90, in `f32` arithmetic. This final conversion and
/// flattening has no counterpart in `ihgeod.f`.
///
/// Errors: `Err` if `m < 2`.
///
/// Fortran header of `ihgeod(m, idp, jdp, x, y, z)`: the points are `x(i,j,k), y(i,j,k), z(i,j,k)`
/// with `i = 1..m+m-1`, `j = 1..m`, `k = 1..5` (the loops in the code name the first index `j` and the
/// second `i`). The indices are defined on the unfolded icosahedron as follows for `m = 3`
/// (theta1 and theta2 are the colatitudes of the two rings of five vertices, `pi` is the south pole,
/// `0` the north pole; the pattern repeats for k = 2,3,4,5 in the longitudinal direction):
///
/// ```text
///                 north pole
///
///                  (5,1)          0      l
///         i     (4,1) (5,2)              a    (repeated for
///            (3,1) (4,2) (5,3)  theta1   t    k=2,3,4,5 in
///         (2,1) (3,2) (4,3)              i        -->
///      (1,1) (2,2) (3,3)        theta2   t    the longitudinal
///         (1,2) (2,3)                    u    direction)
///            (1,3)                pi     d
///       j                                e
///          south pole
/// ```
///
/// Total number of points is `10*(m-1)**2 + 2`, total number of triangles `20*(m-1)**2`, total number of
/// edges `30*(m-1)**2`.
///
/// Fortran: ihgeod.f / ihgeod
///
/// Difference from Fortran:
/// - `idp = 2m - 1` and `jdp = m` are derived internally instead of being passed in.
/// - 0-based indices; `f64` internal precision (Fortran: single precision).
/// - The internal `x, y, z` arrays are zero-initialized, so entries that are never written (outside
///   the four triangles) hold zeros instead of undefined values; they are not part of the output.
pub fn icosahedral_geodesic(m: usize) -> Result<(Vec<f32>, Vec<f32>), String> {
    if m < 2 {
        return Err(format!("m must be >= 2, got {m}"));
    }

    // idp = m + m - 1 and jdp = m are the leading dimensions of the Fortran arrays x, y, z(idp, jdp, 5).
    let idp = 2 * m - 1;
    let jdp = m;
    let len = idp * jdp * 5;
    let mut x = vec![0.0_f64; len];
    let mut y = vec![0.0_f64; len];
    let mut z = vec![0.0_f64; len];

    // 0-based indexer corresponding to Fortran (j, i, k) in (idp, jdp, 5)
    let idx = |j: usize, i: usize, k: usize| -> usize { (k * jdp + i) * idp + j };

    // dphi = 0.4*pi (72 degrees) is the longitudinal width of one sector. beta = cos(dphi);
    // theta1 = acos(beta/(1 - beta)) is the colatitude of the upper ring of five vertices (acos(1/sqrt(5)),
    // about 63.43 degrees), theta2 = pi - theta1 the lower ring. hdphi is half a sector, tdphi = 3*hdphi.
    let dphi = 0.4 * PI;
    let beta = dphi.cos();
    let theta1 = (beta / (1.0 - beta)).acos();
    let theta2 = PI - theta1;
    let hdphi = 0.5 * dphi;
    let tdphi = 3.0 * hdphi;
    let mf = (m - 1) as f64;

    for k in 0..5 {
        let phi = (k as f64) * dphi;
        // Vertices of the first triangle: 1 = lower ring at phi, 2 = south pole (placed at phi + hdphi), 3 = lower ring at phi + dphi.
        let (x1, y1, z1) = stoc(1.0, theta2, phi);
        let (x2, y2, z2) = stoc(1.0, PI, phi + hdphi);
        let (x3, y3, z3) = stoc(1.0, theta2, phi + dphi);

        // Edge 1->2 is stepped by dxi along i, edge 2->3 by dxj along j; points (j <= i) fill the lower triangle of the (j, i) block.
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

        // Vertices 1, 4 and 3, with vertex 4 on the upper ring at phi + hdphi. Here the roles swap: j runs along 1->4 and i along 4->3, filling points with i <= j.
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

        // Vertices 4, 3 and 5 (5 = upper ring at phi + 3*hdphi). Stored with the first index shifted by m - 1; dxi, dyi, dzi are reused from the second triangle, as in the Fortran code.
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

        // Vertices 4, 6 and 5, where 6 is the north pole (theta = 0). Also stored with the first index shifted by m - 1.
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

    // The triangle points are linear interpolations on flat faces and lie inside the unit sphere; converting to spherical coordinates (ctos) and back with r = 1 (stoc) projects them radially onto the sphere. All idp * m entries per sector are visited, as in Fortran.
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
