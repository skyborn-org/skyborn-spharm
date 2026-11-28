<p align="center">
  <img src="assets/skyborn_spharm_logo.svg" alt="skyborn-spharm" width="220">
</p>

<h1 align="center">skyborn-spharm (<code>skybornspharm</code>)</h1>

<p align="center">
  <strong>Pure-Rust spherical harmonic transforms for atmospheric &amp; oceanic sciences in Python</strong>
</p>

<p align="center">
  Part of the <a href="https://github.com/skyborn-org"><strong>Skyborn</strong> (<code>skyborn-org</code>)</a> ecosystem.
</p>

---

## Why `skyborn-spharm`?

- **Pure-Rust `_spharm_rs` Core (v0.2.0)**: Backed by `faer` SIMD matrix kernels, `realfft` / `rustfft`, and adaptive `rayon` multithreading with zero `gfortran` / `libgfortran` / OpenBLAS runtime dependencies.
- **Zero-Copy C-Order & South-to-North (`S2N`) Support**: Accepts modern `(..., nlat, nlon)` C-contiguous arrays and ERA5/CMIP6 `S2N` (`-90° -> +90°`) grids directly without Python flips or Fortran transpositions, while automatically recognizing legacy `(nlat, nlon, ...)` calls.
- **Native `float32` & `float64` Precision**: Full `f64` internal Legendre quadrature and recurrence tables supporting both fast single-precision (`~1e-6` round-trip) and machine-precision double (`~1e-14` round-trip).
- **High-Degree & Ultra-High-Resolution Stability**: Eliminates legacy `SPHEREPACK 3.2` `int32` workspace overflow (`F480` `960×1920`) and `real32` Legendre $2^{n-1-\text{nex}}$ underflow ($n \ge 149$).
- **Packed Reduced Gaussian Grids**: Native `ReducedGaussianTransform` / `ReducedGaussianSpharmt` for ECMWF IFS / ERA5 reduced Gaussian grids (`O80`–`O1280`, `N320`, etc.).
- **Drop-In `pyspharm` Compatibility**: Full Python 3.9–3.15 and NumPy 1.24–2.x support with `SpharmTransform` (C-order default) and `Spharmt` (`auto` layout).

---

## Spherical Harmonics Visual Primer: Analysis, Spectral Calculus & Synthesis

### Waves on a Sphere: Orthogonal Spherical Harmonic Basis

<p align="center"><img src="assets/skyborn_spharm_modes_primer.svg" alt="Spherical Harmonic Modes Primer" width="920"/></p>

Just as a 1D Fourier series decomposes a signal around a circle into pure sine and cosine waves, **spherical harmonics** $Y_n^m(\theta,\lambda) = \bar{P}_n^m(\sin\theta)e^{im\lambda}$ decompose global geophysical fields (such as wind, vorticity, geopotential, and temperature) into smooth, orthogonal 2D wave patterns over the entire sphere:

- **Zonal modes ($m = 0$)**: Symmetric latitude belts with no variation in longitude, capturing the pole-to-equator mean state and zonal jet bands.
- **Tesseral modes ($0 < m < n$)**: Checkerboard wave cells across midlatitudes, capturing planetary Rossby waves and storm tracks.
- **Sectorial modes ($m = n$)**: Orange-slice wave sectors peaking along the equator, capturing tropical wave trains and Walker-scale oscillations.

Because $Y_n^m(\theta,\lambda)$ are smooth global eigenfunctions of the spherical Laplacian, spectral transforms completely eliminate the latitude-longitude **"pole problem"** caused by converging meridians and polar finite-difference singularities (Swarztrauber 1979, 1993, 1996; Adams & Swarztrauber 1999; Swarztrauber & Spotz 2000).

### From Grid to Spectrum and Back: Analysis, Spectral Calculus & Synthesis

<p align="center"><img src="assets/skyborn_spharm_analysis_synthesis.svg" alt="Spherical Harmonic Analysis and Synthesis" width="920"/></p>

<table>
  <tr>
    <td width="50%" align="center" valign="top">
      <img src="assets/skyborn_analysis_logo.svg" alt="Spherical Harmonic Analysis (Grid to Spectrum)" width="190"/><br/>
      <strong>Analysis (Forward Transform, Grid &rarr; Spectrum)</strong><br/>
      <code>Zonal Real FFT</code> along longitude (&lambda; &rarr; <i>m</i>) followed by <code>Gauss-Legendre Quadrature</code> along latitude (&theta; &rarr; <i>n</i>) via <code>grdtospec</code> and <code>getvrtdivspec</code>, projecting physical grids into triangularly truncated spectral coefficients <i>c<sub>n</sub><sup>m</sup></i> (0 &le; <i>m</i> &le; <i>n</i> &le; <i>N</i>).
    </td>
    <td width="50%" align="center" valign="top">
      <img src="assets/skyborn_synthesis_logo.svg" alt="Spherical Harmonic Synthesis (Spectrum to Grid)" width="190"/><br/>
      <strong>Synthesis (Inverse Transform, Spectrum &rarr; Grid)</strong><br/>
      <code>Legendre Summation</code> across degree <i>n</i> followed by <code>Inverse Zonal FFT</code> (<i>m</i> &rarr; &lambda;) via <code>spectogrd</code>, <code>getuv</code>, and <code>getpsichi</code>, reconstructing smooth scalar fields or rotational and divergent wind components cleanly.
    </td>
  </tr>
</table>

- **Algebraic Spectral Calculus (`invlap` / `lap`)**: In spectral space, horizontal differentiation and Poisson inversion reduce to exact diagonal arithmetic: $\nabla^2 \leftrightarrow -n(n+1)/a^2$ and $\nabla^{-2} \leftrightarrow -\frac{a^2}{n(n+1)}$ (`lap` / `invlap`). Streamfunction $\psi$ and velocity potential $\chi$ are inverted directly from vorticity $\zeta$ and divergence $\delta$ with zero polar coordinate singularities.

---

## Pure-Rust (`_spharm_rs`) vs. Legacy Fortran (`_spherepack`) Comparison

All benchmarks below were measured after warmup (median wall time and physical peak `ru_maxrss` delta) on Apple Silicon (`arm64`, Python 3.14, `--release` build) using realistic `S2N` C-order inputs.

### 1. Architecture, Numerical Precision & Packaging

| Dimension | Pure-Rust `_spharm_rs` (v0.2.0) | Legacy Fortran `_spherepack` (SPHEREPACK 3.2) |
| :--- | :--- | :--- |
| Compute Precision | Native `float32` & `float64` (`f64` internal tables) | Hardcoded `float32` (`real32`) transforms |
| Array Memory Layout | Zero-copy C-order `(..., lat, lon)` + auto `legacy` | Fortran-order `(lat, lon, ...)` only (requires copy) |
| Latitude Ordering | Zero-copy `N2S` & `S2N` (`lat=` / `lat_order="S2N"`) | `N2S` only (requires `::-1` copy on `S2N` data) |
| High-Res `F480` (`960×1920`) | Supported (`50.22 ms` `grdtospec` for 8 fields) | Fails (`int32` workspace index overflow, `ierror=4`) |
| High-Degree `legendre` ($n \ge 149$) | Exact `f64` Bonnet + Belousov (`1e-13` accuracy) | `real32` $2^{n-1-\text{nex}}$ underflows to `0.0` at $n \ge 149$ |
| Runtime Dependencies | `numpy` only (self-contained `.so` / `.pyd`) | Requires `libgfortran` / `libquadmath` toolchain |
| Compiled Extension Size | `2.9 MB` (`f32` + `f64` + `faer` + FFT + Rayon) | `1.3 MB` (+ external `libgfortran` shared library) |

### 2. End-to-End Speed & Spectral Round-Trip Error Across Grids

*(Warmed median runtime on `S2N` C-order `float32` batches; Fortran timings include the required Python `S2N -> N2S` flip and `(lat, lon, batch)` Fortran transpose/copy.)*

| Grid / Operator | Batch | Rust `_spharm_rs` | Fortran E2E | Speedup | Round-Trip Max Error (`R32` / `F32` / `R64`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `2.5°` Regular (`73×144`, `T72`) `grdtospec` / `getvrtdivspec` | `nt=64` | `2.05 / 1.83 ms` | `3.73 / 7.33 ms` | **1.82x / 4.00x** | `5.36e-7` / `9.56e-7` / **`2.40e-14`** |
| `1.0°` Regular (`181×360`, `T180`) `grdtospec` / `getvrtdivspec` | `nt=64` | `5.15 / 13.64 ms` | `49.59 / 104.69 ms` | **9.63x / 7.68x** | `1.45e-6` / `2.78e-6` / **`1.77e-13`** |
| `0.5°` Regular (`361×720`, `T360`) `grdtospec` / `getvrtdivspec` | `nt=32` | `20.71 / 32.69 ms` | `189.88 / 314.95 ms` | **9.17x / 9.64x** | `1.97e-6` / `8.11e-6` / **`2.05e-13`** |
| `F160` Gaussian (`160×320`, `T159`) `grdtospec` / `spectogrd` | `nt=96` | `5.49 / 5.50 ms` | `21.55 / 25.24 ms` | **3.92x / 4.59x** | – |
| `F480` Gaussian (`960×1920`, `T959`) `grdtospec` / `getvrtdivspec` | `nt=8` | `50.22 / 227.03 ms` | Overflow (`ierror=4`) | **Rust only** | – |
| `O80` Reduced Gaussian (`27,200` pts) `grdtospec` / `spectogrd` | `nt=32` | `2.30 / 1.79 ms` | `14.47 / 4.67 ms` | **6.31x / 2.61x** | – |
| `regrid` (`73×144` → `160×320`) | `nt=96` | `4.53 ms` | `31.95 ms` | **7.06x** | – |
| `legendre(35.5°, T319)` / `specintrp(T319)` | `nt=1` | `0.123 / 0.033 ms` | `35.173 / 0.081 ms` | **287.2x / 2.44x** | – |
| `lap` / `invlap` (`T63` / `T319`) | `nt=1` | `0.0005 / 0.0071 ms` | `0.0006 / 0.0119 ms` | **1.15x–1.68x** | `0.00e+0` (`f32` exact) |

### 3. Peak Working Memory Overhead (`ru_maxrss` Delta Above Input Arrays)

Because the Rust engine streams large batches through bounded tiles (`MAX_TILE_SIZE = 32`, `MAX_SCRATCH_BYTES = 64 MB`) and reads `S2N` C-order inputs zero-copy, its peak working memory savings scale rapidly with both **spatial resolution** and **batch count**:

| Scaling Case | Operator | Input Volume | Rust Peak Overhead | Fortran E2E Peak Overhead | RAM Saved |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Grid Scaling**: `2.5°` (`73×144`, `nt=64`) | `grdtospec` / `getvrtdivspec` | `2.6 / 5.1 MB` | `+6.5 / +14.8 MB` | `+11.8 / +23.2 MB` | **5.3 / 8.4 MB** (`1.82x / 1.57x`) |
| **Grid Scaling**: `1.0°` (`181×360`, `nt=64`) | `grdtospec` / `getvrtdivspec` | `15.9 / 31.8 MB` | `+33.7 / +77.3 MB` | `+80.9 / +144.3 MB` | **47.2 / 67.0 MB** (`2.40x / 1.87x`) |
| **Grid Scaling**: `0.5°` (`361×720`, `nt=64`) | `grdtospec` / `getvrtdivspec` | `63.5 / 126.9 MB` | `+148.6 / +258.0 MB` | `+457.4 / +709.6 MB` | **308.9 / 451.6 MB** (`3.08x / 2.75x`) |
| **Grid Scaling**: `0.25°` (`721×1440`, `nt=32`) | `grdtospec` / `getvrtdivspec` | `126.7 / 253.5 MB` | `+779.5 / +1535.1 MB` | `+2504.0 / +3003.3 MB` | **1,724.4 / 1,468.2 MB** (`3.21x / 1.96x`) |
| **Batch Scaling**: `1.0°` (`181×360`, `nt=128`) | `grdtospec` / `getvrtdivspec` | `31.8 / 63.6 MB` | `+33.9 / +85.0 MB` | `+64.2 / +207.8 MB` | **30.3 / 122.8 MB** (`1.89x / 2.44x`) |
| **Batch Scaling**: `1.0°` (`181×360`, `nt=512`) | `grdtospec` / `getvrtdivspec` / `getpsichi` | `127.3 / 254.5 / 254.5 MB` | `+33.7 / +134.1 / +388.7 MB` | `+255.6 / +829.8 / +1340.4 MB` | **221.9 / 695.7 / 951.7 MB** (`7.58x / 6.19x / 3.45x`) |

---

## Installation

```bash
pip install skyborn-spharm
```

## Quick Start

### Modern C-order & South-to-North (`S2N`) Zero-Copy API

```python
import numpy as np
from skybornspharm import SpharmTransform

# Typical ERA5 / CMIP6 C-order batch: (time, level, lat, lon), south-to-north
nlon, nlat = 360, 181
lats_s2n = np.linspace(-90.0, 90.0, nlat)
data_s2n = np.random.randn(12, 8, nlat, nlon).astype(np.float32)

sht = SpharmTransform(
    nlon=nlon,
    nlat=nlat,
    gridtype="regular",
    lat=lats_s2n,          # auto-detects S2N with zero Python flipping
    precision="auto",      # float32 -> single, float64 -> double
)

spec = sht.grdtospec(data_s2n)       # shape: (12, 8, nmdim)
inv_spec = sht.invlap(spec)          # spectral inverse Laplacian (-a^2 / [n(n+1)])
reconstructed = sht.spectogrd(spec)  # shape: (12, 8, 181, 360), S2N preserved
```

Standalone spectral utilities such as `skybornspharm.invlap(spec, rsphere=...)` also support 1D, C-order `(..., nmdim)`, and legacy `(nmdim, ...)` spectral arrays directly.

### Legacy `pyspharm`-Compatible & Reduced Gaussian Usage

```python
import numpy as np
from skybornspharm import Spharmt, ReducedGaussianTransform

# Drop-in Spharmt (auto-detects both (lat, lon, ...) legacy and (..., lat, lon) C-order)
sht = Spharmt(nlon=144, nlat=73, gridtype="gaussian")
field = np.random.randn(73, 144)
spec = sht.grdtospec(field)
reconstructed = sht.spectogrd(spec)

# Packed ECMWF reduced Gaussian grid (C-order leading batch axes)
pl = np.array([20, 28, 36, 40, 40, 36, 28, 20], dtype=np.int32)
rg = ReducedGaussianTransform(pl, precision="double")
packed_spec = rg.grdtospec(np.random.randn(4, int(pl.sum())))
```

Note: `skyborn.spharm` and `skyborn.windspharm` in the `skyborn` package seamlessly wrap `skybornspharm` (with automatic C-order detection enabled by default).

## License

- Python wrappers, Pure-Rust engine, and modern extensions: BSD 3-Clause License ([LICENSE](LICENSE))
- SPHEREPACK mathematical reference attribution: SPHEREPACK License ([LICENSE.spherepack](LICENSE.spherepack))
