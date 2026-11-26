# skyborn-spharm (`skybornspharm`)

Modern, prebuilt-wheel spherical harmonic transforms in Python backed by NCAR SPHEREPACK Fortran/OpenMP kernels and native **reduced Gaussian grid** support (ECMWF IFS / ERA5 style).

Part of the [**Skyborn** (`skyborn-org`)](https://github.com/skyborn-org) ecosystem.

## Why `skyborn-spharm`?

- **Drop-in modern replacement for `pyspharm`**: Full Python 3.9–3.14 and NumPy 1.24–2.x support with prebuilt wheels for Linux (`manylinux2014` & `manylinux_2_28`), macOS (`arm64` & `x86_64`), and Windows (`win_amd64`).
- **Zero extra runtime dependencies**: Depends only on `numpy`.
- **Thread-safe & OpenMP accelerated**: Modernized Fortran 90 kernels with Meson build system.
- **Reduced Gaussian Grid Support**: Direct transforms on packed reduced Gaussian grids (`ReducedGaussianSpharmt`).

## Installation

```bash
pip install skyborn-spharm
```

## Quick Start

```python
import numpy as np
from skybornspharm import Spharmt, gaussian_lats_wts

nlon, nlat = 144, 73
sht = Spharmt(nlon=nlon, nlat=nlat, gridtype="gaussian")
field = np.random.randn(nlat, nlon)

spec = sht.grdtospec(field)
reconstructed = sht.spectogrd(spec)
```

Note: `skyborn.spharm` in the `skyborn` package seamlessly re-exports `skybornspharm`.

## License

- Python wrappers and modern extensions: BSD 3-Clause License ([LICENSE](LICENSE))
- Underlying SPHEREPACK routines: SPHEREPACK License ([LICENSE.spherepack](LICENSE.spherepack))
