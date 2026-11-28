# Legacy Fortran / SPHEREPACK Archive (`fortran_ref/`)

This directory archives the original F77/F90 SPHEREPACK sources, C extension glue, and Meson/setuptools build scripts from `skyborn-spharm` `0.1.x`. It is excluded from Pure-Rust wheel builds via `[tool.maturin] exclude` in `pyproject.toml`.

## Known Historical Bugs Fixed & Superseded by Rust (`_spharm_rs`)

1. **`F480 (960 x 1920)` 32-bit Integer Overflow -> Global `NaN` (`vhags` / `shags`)**
   - Original F77/F90 SPHEREPACK computed `(2*n+1)*(m+n-2)*(m+n-3)` using default 32-bit signed integers before converting to floating point.
   - At `F480` (`nlat = 960`, `n = m = 959`), `(2*n-3)*(m+n-1)*(m+n)` exceeds $7.04 \times 10^9 > 2^{31}-1$, wrapping negative so `sqrt(negative)` produces `NaN`.
   - Fixed in `Skyborn` commits `6d30e63` / `b94bafd` and structurally eliminated in Rust (`usize` indices + `f64` recurrence arithmetic).

2. **`getlegfunc` / `alfk` Single-Precision Underflow to Zero for $n \ge 149$ (`alf.f`, `getlegfunc.f`)**
   - In `fortran_ref/src/fortran/alf.f` line 142 (`20 t1 = t1/2.**(n-1-nex)`), when `n - 1 - nex >= 128` (starting at $n \ge 149$), single-precision `2.0**(n-1-nex)` overflows IEEE-754 `real32` (`> 2^128`) to `+Infinity`, causing `t1 / Inf = 0.0` and silently zeroing high-degree sectoral modes (Addition Theorem sum at `lat = 0.5, n = 319` drops from `319.5` to `150.337`).
   - Additionally, `getlegfunc` repeated `alfk` + `lfpt` in $O(N^3)$ time (`~40.5 ms` at `ntrunc=319`, `~131 ms` at `ntrunc=479`).
   - **Resolution**: Marked in untouched F77 archival files (`fortran_ref/src/fortran/{alf.f,getlegfunc.f}`), fixed in `fortran_ref/src/{alf.f90,getlegfunc.f90}` using IEEE `real64` + $O(N^2)$ Bonnet + Schwarztrauber recurrence, and implemented natively in `rust/spharm-math/src/legendre.rs` (`~1.1 ms` pre-opt / sub-ms post-opt, Addition Theorem error `~1e-12`).

3. **Forced `float32` Precision Ceiling**
   - Legacy F2PY / C wrappers coerced `float64` user arrays down to `float32`, capping spectral accuracy at `~1e-7`. `_spharm_rs` preserves native `float64` / `complex128` end-to-end (`~1e-15`–`1e-14` roundtrip error).

4. **F-order & North-to-South Mandatory Layout Copies**
   - Legacy SPHEREPACK required Fortran-contiguous `(nlat, nlon, nt)` north-to-south arrays, forcing callers with C-order `(time, level, lat, lon)` south-to-north data to reverse, transpose, and copy before every call. `_spharm_rs` natively indexes `(nt, nlat, nlon)` C-order and folds `south_to_north` rows in-place.
