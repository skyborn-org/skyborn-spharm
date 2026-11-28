"""Live differential parity, C-order layout, and S2N zero-copy tests for Rust backend."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Optional, Tuple

sys.dont_write_bytecode = True

import numpy as np
import pytest

from skybornspharm import (
    ReducedGaussianSpharmt,
    Spharmt,
    SpharmTransform,
    ValidationError,
    _spharm_rs,
    gaussian_lats_wts,
    getgeodesicpts,
    legendre,
    specintrp,
)

ORACLE_PATH = (
    Path(__file__).resolve().parent / "_fortran_ref" / "_spherepack.cpython-314-darwin.so"
)


def _load_fortran_oracle():
    if not ORACLE_PATH.exists():
        pytest.skip(f"Fortran oracle not found at {ORACLE_PATH}")
    spec = importlib.util.spec_from_file_location("_spherepack", ORACLE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FortranSpharmOracle:
    """Minimal pure-oracle wrapper around legacy _spherepack for differential tests."""

    def __init__(
        self,
        oracle_mod,
        nlon: int,
        nlat: int,
        rsphere: float = 6.3712e6,
        gridtype: str = "regular",
        legfunc: str = "stored",
    ) -> None:
        self.mod = oracle_mod
        self.nlon = nlon
        self.nlat = nlat
        self.rsphere = rsphere
        self.gridtype = gridtype
        self.legfunc = legfunc
        self._n1 = min(nlat, (nlon + 1) // 2 if nlon % 2 else (nlon + 2) // 2)
        self._n2 = (nlat + 1) // 2 if nlat % 2 else nlat // 2

        n1, n2 = self._n1, self._n2
        if (gridtype, legfunc) == ("regular", "stored"):
            lshaes = (n1 * n2 * (nlat + nlat - n1 + 1)) // 2 + nlon + 15
            lwork = 5 * nlat * n2 + 3 * ((n1 - 2) * (nlat + nlat - n1 - 1)) // 2
            self.wshaes, e1 = oracle_mod.shaesi(nlat, nlon, lshaes, lwork, nlat + 1)
            self.wshses, e2 = oracle_mod.shsesi(nlat, nlon, lshaes, lwork, nlat + 1)
            lvhaes = n1 * n2 * (nlat + nlat - n1 + 1) + nlon + 15
            lwork_v = 3 * (max(n1 - 2, 0) * (nlat + nlat - n1 - 1)) // 2 + 5 * n2 * nlat
            self.wvhaes, e3 = oracle_mod.vhaesi(nlat, nlon, lvhaes, lwork_v, 2 * (nlat + 1))
            self.wvhses, e4 = oracle_mod.vhsesi(nlat, nlon, lvhaes, lwork_v, 2 * (nlat + 1))
            assert (e1, e2, e3, e4) == (0, 0, 0, 0)
        elif (gridtype, legfunc) == ("regular", "computed"):
            lshaec = (
                2 * nlat * n2
                + 3 * ((n1 - 2) * (nlat + nlat - n1 - 1)) // 2
                + nlon
                + 15
            )
            self.wshaec, e1 = oracle_mod.shaeci(nlat, nlon, lshaec, 2 * (nlat + 1))
            self.wshsec, e2 = oracle_mod.shseci(nlat, nlon, lshaec, 2 * (nlat + 1))
            lvhaec = 4 * nlat * n2 + 3 * max(n1 - 2, 0) * (2 * nlat - n1 - 1) + nlon + 15
            self.wvhaec, e3 = oracle_mod.vhaeci(nlat, nlon, lvhaec, 2 * (nlat + 1))
            self.wvhsec, e4 = oracle_mod.vhseci(nlat, nlon, lvhaec, 2 * (nlat + 1))
            assert (e1, e2, e3, e4) == (0, 0, 0, 0)
        elif (gridtype, legfunc) == ("gaussian", "stored"):
            lshags = (
                nlat * (3 * (n1 + n2) - 2)
                + (n1 - 1) * (n2 * (2 * nlat - n1) - 3 * n1) // 2
                + nlon
                + 15
            )
            self.wshags, e1 = oracle_mod.shagsi(
                nlat, nlon, lshags, 4 * nlat * (nlat + 2) + 2, nlat * (nlat + 4)
            )
            self.wshsgs = self.wshags
            lvhags = (nlat + 1) * (nlat + 1) * nlat // 2 + nlon + 15
            ldwork_v = (3 * nlat * (nlat + 3) + 2) // 2
            self.wvhags, e3 = oracle_mod.vhagsi(nlat, nlon, lvhags, ldwork_v)
            lvhsgs = n1 * n2 * (nlat + nlat - n1 + 1) + nlon + 15 + 2 * nlat
            self.wvhsgs, e4 = oracle_mod.vhsgsi(nlat, nlon, lvhsgs, ldwork_v)
            assert (e1, e3, e4) == (0, 0, 0)
        elif (gridtype, legfunc) == ("gaussian", "computed"):
            lshagc = (
                nlat * (2 * n2 + 3 * n1 - 2) + 3 * n1 * (1 - n1) // 2 + nlon + 15
            )
            ldwork = nlat * (nlat + 4)
            self.wshagc, e1 = oracle_mod.shagci(nlat, nlon, lshagc, ldwork)
            self.wshsgc = self.wshagc
            lvhagc = (
                4 * nlat * n2
                + 3 * max(n1 - 2, 0) * (2 * nlat - n1 - 1)
                + nlon
                + n2
                + 15
            )
            ldwork_v = 2 * nlat * (nlat + 1) + 1
            self.wvhagc, e3 = oracle_mod.vhagci(nlat, nlon, lvhagc, ldwork_v)
            lvhsgc = (
                4 * nlat * n2 + 3 * max(n1 - 2, 0) * (2 * nlat - n1 - 1) + nlon + 15
            )
            self.wvhsgc, e4 = oracle_mod.vhsgci(nlat, nlon, lvhsgc, ldwork_v)
            assert (e1, e3, e4) == (0, 0, 0)
        else:
            raise ValueError((gridtype, legfunc))

    def _to_fgrid(self, arr: np.ndarray) -> Tuple[np.ndarray, int, Tuple[int, ...]]:
        arr = np.asarray(arr)
        if arr.ndim == 2:
            return np.asfortranarray(arr[:, :, np.newaxis], dtype=np.float32), 1, ()
        extra = tuple(arr.shape[2:])
        nt = int(np.prod(extra))
        return np.asfortranarray(arr.reshape(self.nlat, self.nlon, nt), dtype=np.float32), nt, extra

    def _to_fspec(self, spec: np.ndarray) -> Tuple[np.ndarray, int, Tuple[int, ...]]:
        spec = np.asarray(spec)
        if spec.ndim == 1:
            return np.asfortranarray(spec[:, np.newaxis], dtype=np.complex64), 1, ()
        extra = tuple(spec.shape[1:])
        nt = int(np.prod(extra))
        return np.asfortranarray(spec.reshape(spec.shape[0], nt), dtype=np.complex64), nt, extra

    def grdtospec(self, data: np.ndarray, ntrunc: Optional[int] = None) -> np.ndarray:
        grid, nt, extra = self._to_fgrid(data)
        ntrunc = self.nlat - 1 if ntrunc is None else ntrunc
        if (self.gridtype, self.legfunc) == ("regular", "stored"):
            a, b, ierr = self.mod.shaes(grid, self.wshaes, (nt + 1) * self.nlat * self.nlon)
        elif (self.gridtype, self.legfunc) == ("regular", "computed"):
            lw = self.nlat * (nt * self.nlon + max(3 * self._n2, self.nlon))
            a, b, ierr = self.mod.shaec(grid, self.wshaec, lw)
        elif (self.gridtype, self.legfunc) == ("gaussian", "stored"):
            a, b, ierr = self.mod.shags(grid, self.wshags, self.nlat * self.nlon * (nt + 1))
        else:
            lw = self.nlat * (self.nlon * nt + max(3 * self._n2, self.nlon))
            a, b, ierr = self.mod.shagc(grid, self.wshagc, lw)
        assert ierr == 0
        spec = self.mod.twodtooned(a, b, ntrunc)
        return spec[:, 0] if not extra else spec.reshape((spec.shape[0],) + extra)

    def spectogrd(self, spec: np.ndarray) -> np.ndarray:
        nspec, nt, extra = self._to_fspec(spec)
        a, b = self.mod.onedtotwod(nspec, self.nlat)
        if (self.gridtype, self.legfunc) == ("regular", "stored"):
            out, ierr = self.mod.shses(self.nlon, a, b, self.wshses, (nt + 1) * self.nlat * self.nlon)
        elif (self.gridtype, self.legfunc) == ("regular", "computed"):
            lw = self.nlat * (nt * self.nlon + max(3 * self._n2, self.nlon))
            out, ierr = self.mod.shsec(self.nlon, a, b, self.wshsec, lw)
        elif (self.gridtype, self.legfunc) == ("gaussian", "stored"):
            out, ierr = self.mod.shsgs(self.nlon, a, b, self.wshsgs, self.nlat * self.nlon * (nt + 1))
        else:
            lw = self.nlat * (self.nlon * nt + max(3 * self._n2, self.nlon))
            out, ierr = self.mod.shsgc(self.nlon, a, b, self.wshsgc, lw)
        assert ierr == 0
        return out[:, :, 0] if not extra else out.reshape((self.nlat, self.nlon) + extra)

    def _vha(self, u: np.ndarray, v: np.ndarray, ityp: int):
        w, nt, extra = self._to_fgrid(u)
        vf, _, _ = self._to_fgrid(-np.asarray(v))
        if (self.gridtype, self.legfunc) == ("regular", "stored"):
            br, bi, cr, ci, ierr = self.mod.vhaes(
                vf, w, self.wvhaes, (2 * nt + 1) * self.nlat * self.nlon, ityp
            )
        elif (self.gridtype, self.legfunc) == ("regular", "computed"):
            lw = self.nlat * (2 * nt * self.nlon + max(6 * self._n2, self.nlon))
            br, bi, cr, ci, ierr = self.mod.vhaec(vf, w, self.wvhaec, lw, ityp)
        elif (self.gridtype, self.legfunc) == ("gaussian", "stored"):
            br, bi, cr, ci, ierr = self.mod.vhags(
                vf, w, self.wvhags, (2 * nt + 1) * self.nlat * self.nlon, ityp
            )
        else:
            lw = 2 * self.nlat * (2 * self.nlon * nt + 3 * self._n2)
            br, bi, cr, ci, ierr = self.mod.vhagc(vf, w, self.wvhagc, lw, ityp)
        assert ierr == 0
        return br, bi, cr, ci, extra

    def getvrtdivspec(
        self, u: np.ndarray, v: np.ndarray, ntrunc: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        ntrunc = self.nlat - 1 if ntrunc is None else ntrunc
        br, bi, cr, ci, extra = self._vha(u, v, 0)
        vrt, div = self.mod.twodtooned_vrtdiv(br, bi, cr, ci, ntrunc, self.rsphere)
        if not extra:
            return vrt[:, 0], div[:, 0]
        return vrt.reshape((vrt.shape[0],) + extra), div.reshape((div.shape[0],) + extra)

    def getvrtspec(
        self, u: np.ndarray, v: np.ndarray, ntrunc: Optional[int] = None
    ) -> np.ndarray:
        ntrunc = self.nlat - 1 if ntrunc is None else ntrunc
        _, _, cr, ci, extra = self._vha(u, v, 2)
        vrt = self.mod.twodtooned_vrt(cr, ci, ntrunc, self.rsphere)
        return vrt[:, 0] if not extra else vrt.reshape((vrt.shape[0],) + extra)

    def getdivspec(
        self, u: np.ndarray, v: np.ndarray, ntrunc: Optional[int] = None
    ) -> np.ndarray:
        ntrunc = self.nlat - 1 if ntrunc is None else ntrunc
        br, bi, _, _, extra = self._vha(u, v, 1)
        div = self.mod.twodtooned_div(br, bi, ntrunc, self.rsphere)
        return div[:, 0] if not extra else div.reshape((div.shape[0],) + extra)

    def _vhs(self, br, bi, cr, ci, nt: int, extra: Tuple[int, ...], ityp: int):
        lw_reg = (2 * nt + 1) * self.nlat * self.nlon
        lw_cmp = self.nlat * (2 * nt * self.nlon + max(6 * self._n2, self.nlon))
        if (self.gridtype, self.legfunc) == ("regular", "stored"):
            v, w, ierr = self.mod.vhses(self.nlon, br, bi, cr, ci, self.wvhses, lw_reg, ityp)
        elif (self.gridtype, self.legfunc) == ("regular", "computed"):
            v, w, ierr = self.mod.vhsec(self.nlon, br, bi, cr, ci, self.wvhsec, lw_cmp, ityp)
        elif (self.gridtype, self.legfunc) == ("gaussian", "stored"):
            v, w, ierr = self.mod.vhsgs(self.nlon, br, bi, cr, ci, self.wvhsgs, lw_reg, ityp)
        else:
            v, w, ierr = self.mod.vhsgc(self.nlon, br, bi, cr, ci, self.wvhsgc, lw_cmp, ityp)
        assert ierr == 0
        u_out = w[:, :, 0] if not extra else w.reshape((self.nlat, self.nlon) + extra)
        v_out = -v[:, :, 0] if not extra else -v.reshape((self.nlat, self.nlon) + extra)
        return u_out, v_out

    def getuv(self, vrtspec: np.ndarray, divspec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        vrt_f, nt, extra = self._to_fspec(vrtspec)
        div_f, _, _ = self._to_fspec(divspec)
        br, bi, cr, ci = self.mod.onedtotwod_vrtdiv(vrt_f, div_f, self.nlat, self.rsphere)
        return self._vhs(br, bi, cr, ci, nt, extra, 0)

    def getgrad(self, spec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        spec_f, nt, extra = self._to_fspec(spec)
        div_f = self.mod.lap(spec_f, self.rsphere)
        br, bi = self.mod.onedtotwod_div(div_f, self.nlat, self.rsphere)
        lw_reg = (2 * nt + 1) * self.nlat * self.nlon
        lw_cmp = self.nlat * (2 * nt * self.nlon + max(6 * self._n2, self.nlon))
        if (self.gridtype, self.legfunc) == ("regular", "stored"):
            v, w, ierr = self.mod.vhsesdiv(self.nlon, br, bi, self.wvhses, lw_reg)
        elif (self.gridtype, self.legfunc) == ("regular", "computed"):
            v, w, ierr = self.mod.vhsecdiv(self.nlon, br, bi, self.wvhsec, lw_cmp)
        elif (self.gridtype, self.legfunc) == ("gaussian", "stored"):
            v, w, ierr = self.mod.vhsgsdiv(self.nlon, br, bi, self.wvhsgs, lw_reg)
        else:
            v, w, ierr = self.mod.vhsgcdiv(self.nlon, br, bi, self.wvhsgc, lw_cmp)
        assert ierr == 0
        u_out = w[:, :, 0] if not extra else w.reshape((self.nlat, self.nlon) + extra)
        v_out = -v[:, :, 0] if not extra else -v.reshape((self.nlat, self.nlon) + extra)
        return u_out, v_out

    def getpsichi(
        self, u: np.ndarray, v: np.ndarray, ntrunc: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        vrt, div = self.getvrtdivspec(u, v, ntrunc=ntrunc)
        vrt_f, _, extra = self._to_fspec(vrt)
        div_f, _, _ = self._to_fspec(div)
        psi_f = self.mod.invlap(vrt_f, self.rsphere)
        chi_f = self.mod.invlap(div_f, self.rsphere)
        psi_in = psi_f[:, 0] if not extra else psi_f.reshape((psi_f.shape[0],) + extra)
        chi_in = chi_f[:, 0] if not extra else chi_f.reshape((chi_f.shape[0],) + extra)
        return self.spectogrd(psi_in), self.spectogrd(chi_in)


class _FortranReducedOracle:
    """Minimal oracle wrapper around _spherepack reduced_gaussian_* functions."""

    def __init__(self, oracle_mod, pl: np.ndarray, rsphere: float = 6.3712e6) -> None:
        self.mod = oracle_mod
        self.pl = np.asfortranarray(pl, dtype=np.int32)
        self.nlat = int(self.pl.size)
        self.npoints = int(self.pl.sum())
        self.rsphere = float(rsphere)
        colats, wts, ierr = self.mod.gaqd(self.nlat)
        assert ierr == 0
        self.weights = np.asfortranarray(wts, dtype=np.float32)
        self.sin_theta = np.asfortranarray(np.sin(colats), dtype=np.float32)

    def _basis(self, ntrunc: int):
        b, ierr = self.mod.reduced_gaussian_legendre_basis(self.nlat, ntrunc)
        assert ierr == 0
        db, ierr2 = self.mod.reduced_gaussian_legendre_derivative_from_basis(b, ntrunc)
        assert ierr2 == 0
        return (
            np.asfortranarray(b, dtype=np.float32),
            np.asfortranarray(np.asarray(b, dtype=np.float32).T),
            np.asfortranarray(np.asarray(db, dtype=np.float32).T),
        )

    def grdtospec(self, grid: np.ndarray, ntrunc: int) -> np.ndarray:
        b, _, _ = self._basis(ntrunc)
        g2d = np.asfortranarray(grid[:, np.newaxis] if grid.ndim == 1 else grid, dtype=np.float32)
        spec, ierr = self.mod.reduced_gaussian_grdtospec(
            g2d, self.pl, self.weights, b, ntrunc
        )
        assert ierr == 0
        return spec[:, 0] if grid.ndim == 1 else spec

    def spectogrd(self, spec: np.ndarray, ntrunc: int) -> np.ndarray:
        _, b_t, _ = self._basis(ntrunc)
        s2d = np.asfortranarray(spec[:, np.newaxis] if spec.ndim == 1 else spec, dtype=np.complex64)
        grid, ierr = self.mod.reduced_gaussian_spectogrd(
            s2d, self.pl, b_t, ntrunc, self.npoints
        )
        assert ierr == 0
        return grid[:, 0] if spec.ndim == 1 else grid

    def getvrtdivspec(self, u: np.ndarray, v: np.ndarray, ntrunc: int):
        _, b_t, db_t = self._basis(ntrunc)
        u2d = np.asfortranarray(u[:, np.newaxis] if u.ndim == 1 else u, dtype=np.float32)
        v2d = np.asfortranarray(v[:, np.newaxis] if v.ndim == 1 else v, dtype=np.float32)
        vrt, div, ierr = self.mod.reduced_gaussian_getvrtdivspec(
            u2d, v2d, self.pl, self.weights, b_t, db_t, self.sin_theta, ntrunc, self.rsphere
        )
        assert ierr == 0
        if u.ndim == 1:
            return vrt[:, 0], div[:, 0]
        return vrt, div

    def getuv(self, vrt: np.ndarray, div: np.ndarray, ntrunc: int):
        _, b_t, db_t = self._basis(ntrunc)
        vrt2d = np.asfortranarray(vrt[:, np.newaxis] if vrt.ndim == 1 else vrt, dtype=np.complex64)
        div2d = np.asfortranarray(div[:, np.newaxis] if div.ndim == 1 else div, dtype=np.complex64)
        psi_s = self.mod.invlap(vrt2d, self.rsphere)
        chi_s = self.mod.invlap(div2d, self.rsphere)
        pu, pv, cu, cv, ierr = self.mod.reduced_gaussian_getgrad_pair(
            psi_s, chi_s, self.pl, b_t, db_t, self.sin_theta, ntrunc, self.npoints, self.rsphere
        )
        assert ierr == 0
        u_out = cu - pv
        v_out = cv + pu
        if vrt.ndim == 1:
            return u_out[:, 0], v_out[:, 0]
        return u_out, v_out

    def getgrad(self, spec: np.ndarray, ntrunc: int):
        _, b_t, db_t = self._basis(ntrunc)
        s2d = np.asfortranarray(spec[:, np.newaxis] if spec.ndim == 1 else spec, dtype=np.complex64)
        ug, vg, ierr = self.mod.reduced_gaussian_getgrad(
            s2d, self.pl, b_t, db_t, self.sin_theta, ntrunc, self.npoints, self.rsphere
        )
        assert ierr == 0
        if spec.ndim == 1:
            return ug[:, 0], vg[:, 0]
        return ug, vg


@pytest.fixture(scope="module")
def oracle_mod():
    return _load_fortran_oracle()


def test_utilities_and_spectral_operators_differential_parity(oracle_mod):
    """Compare standalone utilities and spectral operators against Fortran oracle."""
    for nlat in (8, 17, 32, 64, 181):
        colats, f_wts, ierr = oracle_mod.gaqd(nlat)
        assert ierr == 0
        f_lats = 90.0 - np.rad2deg(colats)
        r_lats, r_wts = gaussian_lats_wts(nlat)
        assert r_lats.dtype == np.float64 and r_wts.dtype == np.float64
        np.testing.assert_allclose(r_lats, f_lats, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(r_wts, f_wts, rtol=1e-12, atol=1e-12)

    for lat in (-87.5, -30.0, 0.0, 45.0, 89.5):
        f_leg = oracle_mod.getlegfunc(lat, 18)
        r_leg = legendre(lat, 18)
        assert r_leg.dtype == np.float64
        np.testing.assert_allclose(r_leg, f_leg, rtol=1e-5, atol=1e-6)

    for m in (2, 3, 5):
        x, y, z = oracle_mod.ihgeod(m)
        r2d = x * x + y * y
        r1 = np.sqrt(r2d)
        xtmp = np.where(np.logical_or(x, y), x, np.ones(x.shape, np.float32))
        ztmp = np.where(np.logical_or(r1, z), z, np.ones(z.shape, np.float32))
        lons = (180.0 / math.pi) * np.arctan2(y, xtmp) + 180.0
        lats = (180.0 / math.pi) * np.arctan2(r1, ztmp) - 90.0
        npts = 10 * (m - 1) ** 2 + 2
        f_lat = np.zeros(npts, np.float32)
        f_lon = np.zeros(npts, np.float32)
        f_lat[0], f_lat[1] = 90.0, -90.0
        f_lat[2:] = lats[0 : 2 * (m - 1), 0 : m - 1, :].flatten()
        f_lon[2:] = lons[0 : 2 * (m - 1), 0 : m - 1, :].flatten()
        r_lat, r_lon = getgeodesicpts(m)
        np.testing.assert_allclose(r_lat, f_lat, rtol=1e-5, atol=1e-5)
        lon_diff = ((r_lon - f_lon + 180.0) % 360.0) - 180.0
        np.testing.assert_allclose(lon_diff, 0.0, atol=1e-4)

    rng = np.random.default_rng(20261128)
    ntrunc = 12
    nmdim = (ntrunc + 1) * (ntrunc + 2) // 2
    spec_1d = (rng.standard_normal(nmdim) + 1j * rng.standard_normal(nmdim)).astype(
        np.complex64
    )
    spec_1d[: ntrunc + 1] = spec_1d[: ntrunc + 1].real.astype(np.complex64)

    f_leg = oracle_mod.getlegfunc(37.5, ntrunc)
    r_leg = legendre(37.5, ntrunc)
    f_val = oracle_mod.specintrp(math.radians(123.4), ntrunc, spec_1d, f_leg)
    r_val = specintrp(123.4, spec_1d, r_leg)
    np.testing.assert_allclose(r_val, f_val, rtol=1e-5, atol=1e-5)

    # Batch complex64 and complex128 lap / invlap / multsmoothfact
    spec_batch_c64 = (
        rng.standard_normal((5, nmdim)) + 1j * rng.standard_normal((5, nmdim))
    ).astype(np.complex64)
    smooth_f32 = np.linspace(1.0, 0.2, ntrunc + 1, dtype=np.float32)
    rsphere = 6.3712e6
    f_spec_for = np.asfortranarray(spec_batch_c64.T)

    np.testing.assert_allclose(
        _spharm_rs.lap(spec_batch_c64, rsphere),
        oracle_mod.lap(f_spec_for, rsphere).T,
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _spharm_rs.invlap(spec_batch_c64, rsphere),
        oracle_mod.invlap(f_spec_for, rsphere).T,
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _spharm_rs.multsmoothfact(spec_batch_c64, smooth_f32),
        oracle_mod.multsmoothfact(f_spec_for, smooth_f32).T,
        rtol=1e-6,
        atol=1e-12,
    )

    spec_batch_c128 = spec_batch_c64.astype(np.complex128)
    r_lap_64 = _spharm_rs.lap(spec_batch_c128, rsphere)
    r_inv_64 = _spharm_rs.invlap(r_lap_64, rsphere)
    assert r_lap_64.dtype == np.complex128
    assert r_inv_64.dtype == np.complex128
    # n=0 mode is zeroed by invlap
    np.testing.assert_allclose(r_inv_64[:, 1:], spec_batch_c128[:, 1:], rtol=1e-12, atol=1e-12)
    assert np.all(r_inv_64[:, 0] == 0.0)


@pytest.mark.parametrize("gridtype", ["regular", "gaussian"])
@pytest.mark.parametrize("legfunc", ["stored", "computed"])
@pytest.mark.parametrize(("nlat", "nlon"), [(19, 36), (32, 64)])
def test_spharmt_differential_parity_float32_and_float64(
    oracle_mod, gridtype, legfunc, nlat, nlon
):
    """Verify full transform parity against Fortran oracle in float32 and float64."""
    rng = np.random.default_rng(100 + nlat + (1 if gridtype == "gaussian" else 0))
    oracle = _FortranSpharmOracle(
        oracle_mod, nlon=nlon, nlat=nlat, gridtype=gridtype, legfunc=legfunc
    )
    rust_legacy = Spharmt(
        nlon=nlon, nlat=nlat, gridtype=gridtype, legfunc=legfunc, layout="legacy"
    )
    rust_c = SpharmTransform(nlon=nlon, nlat=nlat, gridtype=gridtype, legfunc=legfunc)

    # Create smooth bandlimited scalar and vector fields so even float32 roundtrip is crisp
    ntrunc = nlat - 1
    nspec = (ntrunc + 1) * (ntrunc + 2) // 2
    s_raw = (rng.standard_normal((nlat, nlon, 3)) * 5.0).astype(np.float32)
    u_raw = (rng.standard_normal((nlat, nlon, 3)) * 10.0).astype(np.float32)
    v_raw = (rng.standard_normal((nlat, nlon, 3)) * 10.0).astype(np.float32)

    # 1. Float32 differential parity (legacy shape)
    f_spec = oracle.grdtospec(s_raw)
    r_spec = rust_legacy.grdtospec(s_raw)
    assert r_spec.dtype == np.complex64
    np.testing.assert_allclose(r_spec, f_spec, rtol=1e-4, atol=1e-4)

    f_grid = oracle.spectogrd(f_spec)
    r_grid = rust_legacy.spectogrd(r_spec)
    assert r_grid.dtype == np.float32
    np.testing.assert_allclose(r_grid, f_grid, rtol=1e-4, atol=1e-4)

    f_vrt, f_div = oracle.getvrtdivspec(u_raw, v_raw)
    r_vrt, r_div = rust_legacy.getvrtdivspec(u_raw, v_raw)
    r_vrt_only = rust_legacy.getvrtspec(u_raw, v_raw)
    r_div_only = rust_legacy.getdivspec(u_raw, v_raw)
    assert r_vrt.dtype == np.complex64
    np.testing.assert_allclose(r_vrt, f_vrt, rtol=1e-4, atol=1e-10)
    np.testing.assert_allclose(r_div, f_div, rtol=1e-4, atol=1e-10)
    np.testing.assert_allclose(r_vrt_only, r_vrt, rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(r_div_only, r_div, rtol=1e-5, atol=1e-12)

    f_u, f_v = oracle.getuv(f_vrt, f_div)
    r_u, r_v = rust_legacy.getuv(r_vrt, r_div)
    assert r_u.dtype == np.float32 and r_v.dtype == np.float32
    np.testing.assert_allclose(r_u, f_u, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(r_v, f_v, rtol=1e-4, atol=1e-4)

    f_psi, f_chi = oracle.getpsichi(u_raw, v_raw)
    r_psi, r_chi = rust_legacy.getpsichi(u_raw, v_raw)
    r_psi_only = rust_legacy.getpsi(u_raw, v_raw)
    r_chi_only = rust_legacy.getchi(u_raw, v_raw)
    psi_atol = max(1.0, 1e-5 * float(np.max(np.abs(f_psi))))
    chi_atol = max(1.0, 1e-5 * float(np.max(np.abs(f_chi))))
    np.testing.assert_allclose(r_psi, f_psi, rtol=1e-4, atol=psi_atol)
    np.testing.assert_allclose(r_chi, f_chi, rtol=1e-4, atol=chi_atol)
    np.testing.assert_allclose(r_psi_only, r_psi, rtol=1e-5, atol=psi_atol)
    np.testing.assert_allclose(r_chi_only, r_chi, rtol=1e-5, atol=chi_atol)

    f_gu, f_gv = oracle.getgrad(f_spec)
    r_gu, r_gv = rust_legacy.getgrad(r_spec)
    np.testing.assert_allclose(r_gu, f_gu, rtol=1e-4, atol=1e-9)
    np.testing.assert_allclose(r_gv, f_gv, rtol=1e-4, atol=1e-9)

    # 2. Float64 true double precision parity + high-precision roundtrip
    s_c64 = np.moveaxis(s_raw, -1, 0).astype(np.float64)
    u_c64 = np.moveaxis(u_raw, -1, 0).astype(np.float64)
    v_c64 = np.moveaxis(v_raw, -1, 0).astype(np.float64)

    r_spec64 = rust_c.grdtospec(s_c64)
    assert r_spec64.dtype == np.complex128
    np.testing.assert_allclose(np.moveaxis(r_spec64, 0, -1), f_spec, rtol=1e-4, atol=1e-4)
    r_grid64 = rust_c.spectogrd(r_spec64)
    assert r_grid64.dtype == np.float64
    r_spec64_rt = rust_c.grdtospec(r_grid64)
    # Truncated roundtrip in true f64 must achieve ~1e-12 accuracy (Fortran f32 cannot)
    np.testing.assert_allclose(r_spec64_rt, r_spec64, rtol=1e-11, atol=1e-11)

    r_vrt64, r_div64 = rust_c.getvrtdivspec(u_c64, v_c64)
    assert r_vrt64.dtype == np.complex128
    np.testing.assert_allclose(np.moveaxis(r_vrt64, 0, -1), f_vrt, rtol=1e-4, atol=1e-10)
    np.testing.assert_allclose(np.moveaxis(r_div64, 0, -1), f_div, rtol=1e-4, atol=1e-10)
    r_u64, r_v64 = rust_c.getuv(r_vrt64, r_div64)
    assert r_u64.dtype == np.float64
    r_vrt64_rt, r_div64_rt = rust_c.getvrtdivspec(r_u64, r_v64)
    np.testing.assert_allclose(r_vrt64_rt, r_vrt64, rtol=1e-10, atol=1e-18)
    np.testing.assert_allclose(r_div64_rt, r_div64, rtol=1e-10, atol=1e-18)
    assert nspec == r_spec64.shape[-1]


def test_reduced_gaussian_differential_parity_float32_and_float64(oracle_mod):
    """Verify ReducedGaussianSpharmt parity against Fortran oracle in f32 and f64."""
    pl = np.array([16, 20, 24, 28, 32, 28, 24, 20, 16], dtype=np.int32)
    ntrunc = 5
    rng = np.random.default_rng(20260601)
    npts = int(pl.sum())

    oracle = _FortranReducedOracle(oracle_mod, pl)
    rust_leg = ReducedGaussianSpharmt(pl, layout="legacy")
    rust_c = ReducedGaussianSpharmt(pl, layout="c")

    s_f = rng.standard_normal((npts, 4)).astype(np.float32)
    u_f = rng.standard_normal((npts, 4)).astype(np.float32)
    v_f = rng.standard_normal((npts, 4)).astype(np.float32)

    f_spec = oracle.grdtospec(s_f, ntrunc)
    r_spec = rust_leg.grdtospec(s_f, ntrunc=ntrunc)
    np.testing.assert_allclose(r_spec, f_spec, rtol=2e-5, atol=2e-5)

    f_grd = oracle.spectogrd(f_spec, ntrunc)
    r_grd = rust_leg.spectogrd(r_spec)
    np.testing.assert_allclose(r_grd, f_grd, rtol=2e-5, atol=2e-5)

    f_vrt, f_div = oracle.getvrtdivspec(u_f, v_f, ntrunc)
    r_vrt, r_div = rust_leg.getvrtdivspec(u_f, v_f, ntrunc=ntrunc)
    np.testing.assert_allclose(r_vrt, f_vrt, rtol=2e-5, atol=1e-11)
    np.testing.assert_allclose(r_div, f_div, rtol=2e-5, atol=1e-11)
    np.testing.assert_allclose(
        rust_leg.getvrtspec(u_f, v_f, ntrunc=ntrunc), r_vrt, rtol=1e-6, atol=1e-12
    )
    np.testing.assert_allclose(
        rust_leg.getdivspec(u_f, v_f, ntrunc=ntrunc), r_div, rtol=1e-6, atol=1e-12
    )

    f_u, f_v = oracle.getuv(f_vrt, f_div, ntrunc)
    r_u, r_v = rust_leg.getuv(r_vrt, r_div)
    np.testing.assert_allclose(r_u, f_u, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(r_v, f_v, rtol=2e-5, atol=2e-5)

    f_gu, f_gv = oracle.getgrad(f_spec, ntrunc)
    r_gu, r_gv = rust_leg.getgrad(r_spec)
    np.testing.assert_allclose(r_gu, f_gu, rtol=2e-5, atol=1e-10)
    np.testing.assert_allclose(r_gv, f_gv, rtol=2e-5, atol=1e-10)

    # C-order (2, 2, npoints) float64 parity + high-precision roundtrip
    s_c64 = np.ascontiguousarray(s_f.T.reshape(2, 2, npts), dtype=np.float64)
    u_c64 = np.ascontiguousarray(u_f.T.reshape(2, 2, npts), dtype=np.float64)
    v_c64 = np.ascontiguousarray(v_f.T.reshape(2, 2, npts), dtype=np.float64)
    r_spec_c = rust_c.grdtospec(s_c64, ntrunc=ntrunc)
    assert r_spec_c.shape == (2, 2, (ntrunc + 1) * (ntrunc + 2) // 2)
    assert r_spec_c.dtype == np.complex128
    np.testing.assert_allclose(
        r_spec_c.reshape(4, -1).T, f_spec, rtol=2e-5, atol=2e-5
    )
    r_grd_c = rust_c.spectogrd(r_spec_c)
    assert r_grd_c.shape == (2, 2, npts) and r_grd_c.dtype == np.float64
    np.testing.assert_allclose(
        rust_c.grdtospec(r_grd_c, ntrunc=ntrunc), r_spec_c, rtol=1e-11, atol=1e-11
    )
    r_vrt_c, r_div_c = rust_c.getvrtdivspec(u_c64, v_c64, ntrunc=ntrunc)
    r_u_c, r_v_c = rust_c.getuv(r_vrt_c, r_div_c)
    assert r_u_c.shape == (2, 2, npts) and r_v_c.shape == (2, 2, npts)
    vrt_rt, div_rt = rust_c.getvrtdivspec(r_u_c, r_v_c, ntrunc=ntrunc)
    np.testing.assert_allclose(vrt_rt, r_vrt_c, rtol=1e-10, atol=1e-17)
    np.testing.assert_allclose(div_rt, r_div_c, rtol=1e-10, atol=1e-17)


@pytest.mark.parametrize("gridtype", ["regular", "gaussian"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_corder_4d_3d_2d_vs_legacy_parity(gridtype, dtype):
    """C-order (time, level, lat, lon)/(time, lat, lon)/(nt, lat, lon) matches legacy."""
    nlat, nlon = 19, 36
    rng = np.random.default_rng(777)
    s_4d = np.ascontiguousarray(rng.standard_normal((3, 4, nlat, nlon)), dtype=dtype)
    u_4d = np.ascontiguousarray(rng.standard_normal((3, 4, nlat, nlon)), dtype=dtype)
    v_4d = np.ascontiguousarray(rng.standard_normal((3, 4, nlat, nlon)), dtype=dtype)

    s_3d = s_4d.reshape(12, nlat, nlon)
    s_leg = np.asfortranarray(np.moveaxis(s_4d, (2, 3), (0, 1)))
    u_leg = np.asfortranarray(np.moveaxis(u_4d, (2, 3), (0, 1)))
    v_leg = np.asfortranarray(np.moveaxis(v_4d, (2, 3), (0, 1)))

    sht_c = SpharmTransform(nlon=nlon, nlat=nlat, gridtype=gridtype)
    sht_leg = Spharmt(nlon=nlon, nlat=nlat, gridtype=gridtype, layout="legacy")

    spec_4d = sht_c.grdtospec(s_4d)
    spec_3d = sht_c.grdtospec(s_3d)
    spec_2d_slice = sht_c.grdtospec(s_4d[1, 2])
    spec_leg = sht_leg.grdtospec(s_leg)

    assert spec_4d.shape == (3, 4, nlat * (nlat + 1) // 2)
    assert spec_3d.shape == (12, nlat * (nlat + 1) // 2)
    assert spec_2d_slice.shape == (nlat * (nlat + 1) // 2,)
    np.testing.assert_allclose(spec_4d.reshape(12, -1), spec_3d, rtol=0, atol=0)
    np.testing.assert_allclose(spec_4d[1, 2], spec_2d_slice, rtol=0, atol=0)
    np.testing.assert_allclose(np.moveaxis(spec_4d, -1, 0), spec_leg, rtol=1e-12, atol=1e-12)

    rec_4d = sht_c.spectogrd(spec_4d)
    rec_leg = sht_leg.spectogrd(spec_leg)
    assert rec_4d.shape == (3, 4, nlat, nlon)
    np.testing.assert_allclose(np.moveaxis(rec_4d, (2, 3), (0, 1)), rec_leg, rtol=1e-12, atol=1e-12)

    vrt_4d, div_4d = sht_c.getvrtdivspec(u_4d, v_4d)
    vrt_leg, div_leg = sht_leg.getvrtdivspec(u_leg, v_leg)
    np.testing.assert_allclose(np.moveaxis(vrt_4d, -1, 0), vrt_leg, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.moveaxis(div_4d, -1, 0), div_leg, rtol=1e-12, atol=1e-12)

    psi_4d, chi_4d = sht_c.getpsichi(u_4d, v_4d)
    psi_leg, chi_leg = sht_leg.getpsichi(u_leg, v_leg)
    np.testing.assert_allclose(np.moveaxis(psi_4d, (2, 3), (0, 1)), psi_leg, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.moveaxis(chi_4d, (2, 3), (0, 1)), chi_leg, rtol=1e-12, atol=1e-12)

    gu_4d, gv_4d = sht_c.getgrad(spec_4d)
    gu_leg, gv_leg = sht_leg.getgrad(spec_leg)
    np.testing.assert_allclose(np.moveaxis(gu_4d, (2, 3), (0, 1)), gu_leg, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.moveaxis(gv_4d, (2, 3), (0, 1)), gv_leg, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("gridtype", ["regular", "gaussian"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_south_to_north_autodetect_and_zero_copy(gridtype, dtype):
    """Verify S2N vs N2S parity, lat auto-detection, validation, and zero copy."""
    nlat, nlon = 19, 36
    if gridtype == "regular":
        lat_n2s = np.linspace(90.0, -90.0, nlat)
    else:
        lat_n2s, _ = gaussian_lats_wts(nlat)
    lat_s2n = lat_n2s[::-1].copy()

    rng = np.random.default_rng(20260909)
    s_n2s = np.ascontiguousarray(rng.standard_normal((2, 3, nlat, nlon)), dtype=dtype)
    u_n2s = np.ascontiguousarray(rng.standard_normal((2, 3, nlat, nlon)), dtype=dtype)
    v_n2s = np.ascontiguousarray(rng.standard_normal((2, 3, nlat, nlon)), dtype=dtype)

    s_s2n = np.ascontiguousarray(s_n2s[:, :, ::-1, :])
    u_s2n = np.ascontiguousarray(u_n2s[:, :, ::-1, :])
    v_s2n = np.ascontiguousarray(v_n2s[:, :, ::-1, :])
    s_s2n_snapshot = s_s2n.copy()

    sht_n2s = SpharmTransform(nlon=nlon, nlat=nlat, gridtype=gridtype, lat=lat_n2s)
    sht_s2n = SpharmTransform(nlon=nlon, nlat=nlat, gridtype=gridtype, lat=lat_s2n)
    assert sht_n2s.south_to_north is False
    assert sht_s2n.south_to_north is True

    # Zero-copy assertion: _prepare_grid_input must alias the caller's C-contiguous S2N buffer
    _, nt, c_3d, batch_shape = sht_s2n._prepare_grid_input(s_s2n, "zero_copy_check")
    assert nt == 6 and batch_shape == (2, 3)
    assert np.shares_memory(s_s2n, c_3d)

    # Scalar analysis & synthesis parity
    spec_n2s = sht_n2s.grdtospec(s_n2s)
    spec_s2n = sht_s2n.grdtospec(s_s2n)
    np.testing.assert_allclose(spec_s2n, spec_n2s, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(s_s2n, s_s2n_snapshot)

    grd_n2s = sht_n2s.spectogrd(spec_n2s)
    grd_s2n = sht_s2n.spectogrd(spec_s2n)
    np.testing.assert_allclose(grd_s2n, grd_n2s[:, :, ::-1, :], rtol=1e-12, atol=1e-12)

    # Vector analysis & synthesis parity
    vrt_n2s, div_n2s = sht_n2s.getvrtdivspec(u_n2s, v_n2s)
    vrt_s2n, div_s2n = sht_s2n.getvrtdivspec(u_s2n, v_s2n)
    np.testing.assert_allclose(vrt_s2n, vrt_n2s, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(div_s2n, div_n2s, rtol=1e-12, atol=1e-12)

    u_back_n, v_back_n = sht_n2s.getuv(vrt_n2s, div_n2s)
    u_back_s, v_back_s = sht_s2n.getuv(vrt_s2n, div_s2n)
    np.testing.assert_allclose(u_back_s, u_back_n[:, :, ::-1, :], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(v_back_s, v_back_n[:, :, ::-1, :], rtol=1e-12, atol=1e-12)

    psi_n, chi_n = sht_n2s.getpsichi(u_n2s, v_n2s)
    psi_s, chi_s = sht_s2n.getpsichi(u_s2n, v_s2n)
    np.testing.assert_allclose(psi_s, psi_n[:, :, ::-1, :], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(chi_s, chi_n[:, :, ::-1, :], rtol=1e-12, atol=1e-12)

    gu_n, gv_n = sht_n2s.getgrad(spec_n2s)
    gu_s, gv_s = sht_s2n.getgrad(spec_s2n)
    np.testing.assert_allclose(gu_s, gu_n[:, :, ::-1, :], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(gv_s, gv_n[:, :, ::-1, :], rtol=1e-12, atol=1e-12)

    # Validation errors on contradictory lat / lat_order or non-monotonic lat
    with pytest.raises(ValidationError):
        SpharmTransform(
            nlon=nlon, nlat=nlat, gridtype=gridtype, lat=lat_s2n, lat_order="N2S"
        )
    with pytest.raises(ValidationError):
        SpharmTransform(
            nlon=nlon, nlat=nlat, gridtype=gridtype, lat=lat_n2s, lat_order="S2N"
        )
    bad_lat = lat_n2s.copy()
    bad_lat[3] = bad_lat[4]
    with pytest.raises(ValidationError):
        SpharmTransform(nlon=nlon, nlat=nlat, gridtype=gridtype, lat=bad_lat)


def test_reduced_gaussian_s2n_parity_and_zero_copy():
    """Verify ReducedGaussianSpharmt S2N lat auto-detection and zero copy."""
    pl = np.array([16, 20, 24, 28, 32, 28, 24, 20, 16], dtype=np.int32)
    lat_n2s, _ = gaussian_lats_wts(len(pl))
    lat_s2n = lat_n2s[::-1].copy()

    rng = np.random.default_rng(20260910)
    npts = int(pl.sum())
    s_n2s = rng.standard_normal((3, npts)).astype(np.float64)
    u_n2s = rng.standard_normal((3, npts)).astype(np.float64)
    v_n2s = rng.standard_normal((3, npts)).astype(np.float64)

    offsets = np.concatenate(([0], np.cumsum(pl)))
    s_s2n = np.empty_like(s_n2s)
    u_s2n = np.empty_like(u_n2s)
    v_s2n = np.empty_like(v_n2s)
    out_off = 0
    for j in range(len(pl) - 1, -1, -1):
        a, b = offsets[j], offsets[j + 1]
        w = b - a
        s_s2n[:, out_off : out_off + w] = s_n2s[:, a:b]
        u_s2n[:, out_off : out_off + w] = u_n2s[:, a:b]
        v_s2n[:, out_off : out_off + w] = v_n2s[:, a:b]
        out_off += w

    rg_n2s = ReducedGaussianSpharmt(pl, layout="c", lat=lat_n2s)
    rg_s2n = ReducedGaussianSpharmt(pl, layout="c", lat=lat_s2n)
    assert rg_n2s.south_to_north is False
    assert rg_s2n.south_to_north is True

    _, nt, c_2d, _ = rg_s2n._prepare_grid_input(s_s2n, "rg_zero_copy")
    assert nt == 3 and np.shares_memory(s_s2n, c_2d)

    np.testing.assert_allclose(
        rg_s2n.grdtospec(s_s2n, ntrunc=5),
        rg_n2s.grdtospec(s_n2s, ntrunc=5),
        rtol=1e-12,
        atol=1e-12,
    )
    vrt_n, div_n = rg_n2s.getvrtdivspec(u_n2s, v_n2s, ntrunc=5)
    vrt_s, div_s = rg_s2n.getvrtdivspec(u_s2n, v_s2n, ntrunc=5)
    np.testing.assert_allclose(vrt_s, vrt_n, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(div_s, div_n, rtol=1e-12, atol=1e-12)


def test_canonical_aliases_and_reduced_psi_chi_parity():
    """Verify SpherepackError, ReducedGaussianTransform, and ReducedGaussianSpharmt psi/chi."""
    from skybornspharm import (
        ReducedGaussianTransform,
        SpherepackError,
        SpheremackError,
        getspecindx,
    )

    assert SpherepackError is SpheremackError
    assert hasattr(SpharmTransform(16, 8), "invlap")
    assert hasattr(SpharmTransform(16, 8), "invlapspec")
    assert not hasattr(SpharmTransform(16, 8), "lapspec")

    indxm0, indxn0 = getspecindx(0)
    assert indxm0.shape == (1,) and indxn0.shape == (1,)
    assert int(indxm0[0]) == 0 and int(indxn0[0]) == 0

    pl = np.array([16, 20, 24, 28, 32, 28, 24, 20, 16], dtype=np.int32)
    npts = int(pl.sum())
    rng = np.random.default_rng(20261128)
    u_c = rng.standard_normal((2, npts)).astype(np.float64)
    v_c = rng.standard_normal((2, npts)).astype(np.float64)
    s_c = rng.standard_normal((2, npts)).astype(np.float64)

    rgt = ReducedGaussianTransform(pl)
    assert rgt.layout == "c"
    assert hasattr(rgt, "invlap")
    assert hasattr(rgt, "invlapspec")
    assert not hasattr(rgt, "lapspec")

    rgs = ReducedGaussianSpharmt(pl, layout="c")
    psi, chi = rgt.getpsichi(u_c, v_c, ntrunc=5)
    np.testing.assert_allclose(psi, rgt.getpsi(u_c, v_c, ntrunc=5), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(chi, rgt.getchi(u_c, v_c, ntrunc=5), rtol=1e-12, atol=1e-12)
    psispec, chispec = rgt.getpsichispec(u_c, v_c, ntrunc=5)
    np.testing.assert_allclose(psispec, rgt.getpsispec(u_c, v_c, ntrunc=5), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(chispec, rgt.getchispec(u_c, v_c, ntrunc=5), rtol=1e-12, atol=1e-12)

    spec = rgs.grdtospec(s_c, ntrunc=5)
    gx_pos, gy_pos = rgs.getgrad(spec)
    gx_kw, gy_kw = rgs.getgrad(chispec=spec)
    np.testing.assert_allclose(gx_pos, gx_kw, rtol=0, atol=0)
    np.testing.assert_allclose(gy_pos, gy_kw, rtol=0, atol=0)


def test_4d_and_5d_zero_copy_reshape_and_exact_parity():
    """Verify 4D/5D grids flatten to zero-copy 3D views internally and restore exact shapes."""
    from skybornspharm import ReducedGaussianTransform

    rng = np.random.default_rng(20261130)
    nlat, nlon = 19, 36
    sp_auto = Spharmt(nlon, nlat, gridtype="regular", layout="auto")
    sp_c = SpharmTransform(nlon, nlat, gridtype="regular")

    for dtype, batch_shape in [(np.float32, (4, 3)), (np.float64, (2, 2, 3))]:
        nt = int(np.prod(batch_shape))
        u_nd = rng.standard_normal(batch_shape + (nlat, nlon)).astype(dtype)
        v_nd = rng.standard_normal(batch_shape + (nlat, nlon)).astype(dtype)

        # C-contiguous N-D inputs share memory with the flattened (nt, nlat, nlon) 3D view
        resolved_layout, got_nt, c_3d, got_batch = sp_auto._prepare_grid_input(u_nd, "test")
        assert resolved_layout == "c"
        assert got_nt == nt
        assert got_batch == batch_shape
        assert c_3d.shape == (nt, nlat, nlon)
        assert np.shares_memory(c_3d, u_nd)

        spec_nd = sp_c.grdtospec(u_nd, ntrunc=11)
        spec_3d = sp_c.grdtospec(u_nd.reshape(nt, nlat, nlon), ntrunc=11)
        assert spec_nd.shape == batch_shape + (spec_3d.shape[-1],)
        np.testing.assert_allclose(
            spec_nd, spec_3d.reshape(batch_shape + (spec_3d.shape[-1],)), rtol=0, atol=0
        )

        rec_nd = sp_c.spectogrd(spec_nd)
        rec_3d = sp_c.spectogrd(spec_3d)
        assert rec_nd.shape == batch_shape + (nlat, nlon)
        assert rec_nd.dtype == dtype
        np.testing.assert_allclose(
            rec_nd, rec_3d.reshape(batch_shape + (nlat, nlon)), rtol=0, atol=0
        )

        psi_nd, chi_nd = sp_c.getpsichi(u_nd, v_nd, ntrunc=11)
        psi_3d, chi_3d = sp_c.getpsichi(
            u_nd.reshape(nt, nlat, nlon), v_nd.reshape(nt, nlat, nlon), ntrunc=11
        )
        assert psi_nd.shape == batch_shape + (nlat, nlon)
        assert chi_nd.shape == batch_shape + (nlat, nlon)
        np.testing.assert_allclose(
            psi_nd, psi_3d.reshape(batch_shape + (nlat, nlon)), rtol=0, atol=0
        )
        np.testing.assert_allclose(
            chi_nd, chi_3d.reshape(batch_shape + (nlat, nlon)), rtol=0, atol=0
        )

    # Legacy 4D shape (nlat, nlon, 4, 3) also roundtrips identically to flattened 3D
    u_4d = rng.standard_normal((4, 3, nlat, nlon)).astype(np.float32)
    u_4d_leg = np.moveaxis(u_4d, (2, 3), (0, 1)).copy()
    sp_leg = Spharmt(nlon, nlat, gridtype="regular", layout="legacy")
    spec_leg = sp_leg.grdtospec(u_4d_leg, ntrunc=11)
    assert spec_leg.shape == (spec_leg.shape[0], 4, 3)
    rec_leg = sp_leg.spectogrd(spec_leg)
    assert rec_leg.shape == (nlat, nlon, 4, 3)
    np.testing.assert_allclose(
        np.moveaxis(rec_leg, (0, 1), (2, 3)),
        sp_c.spectogrd(sp_c.grdtospec(u_4d, ntrunc=11)),
        rtol=0,
        atol=0,
    )

    # Reduced Gaussian 3D (2, 3, sum(pl)) -> flattened (6, sum(pl)) -> restored (2, 3, sum(pl))
    pl = np.array([16, 20, 24, 28, 32, 28, 24, 20, 16], dtype=np.int32)
    npts = int(pl.sum())
    rgt = ReducedGaussianTransform(pl)
    u_rg = rng.standard_normal((2, 3, npts)).astype(np.float32)
    v_rg = rng.standard_normal((2, 3, npts)).astype(np.float32)
    _, got_rg_nt, rg_2d, rg_batch = rgt._prepare_grid_input(u_rg, "test_rg")
    assert got_rg_nt == 6 and rg_batch == (2, 3)
    assert np.shares_memory(rg_2d, u_rg)
    psi_rg, chi_rg = rgt.getpsichi(u_rg, v_rg, ntrunc=5)
    psi_rg2, chi_rg2 = rgt.getpsichi(u_rg.reshape(6, npts), v_rg.reshape(6, npts), ntrunc=5)
    assert psi_rg.shape == (2, 3, npts) and psi_rg.dtype == np.float32
    np.testing.assert_allclose(psi_rg, psi_rg2.reshape(2, 3, npts), rtol=0, atol=0)
    np.testing.assert_allclose(chi_rg, chi_rg2.reshape(2, 3, npts), rtol=0, atol=0)


def test_public_invlap_and_invlapspec_apis():
    """Verify Spharmt/SpharmTransform/ReducedGaussianTransform invlap/invlapspec and standalone invlap."""
    from skybornspharm import (
        ReducedGaussianTransform,
        ValidationError,
        getspecindx,
        invlap,
    )

    rng = np.random.default_rng(20261201)
    nlat, nlon = 19, 36
    rsphere = 6.3712e6
    ntrunc = 8
    nmdim = (ntrunc + 1) * (ntrunc + 2) // 2
    _, indxn = getspecindx(ntrunc)

    expected_scale = np.zeros(nmdim, dtype=np.float64)
    nonzero = indxn > 0
    expected_scale[nonzero] = -(rsphere**2) / (indxn[nonzero] * (indxn[nonzero] + 1))

    # 1. Exact analytical values, n=0 zeroing, and 1D / 3D / 4D C-order & legacy layouts
    spec_1d_64 = (
        rng.standard_normal(nmdim) + 1j * rng.standard_normal(nmdim)
    ).astype(np.complex128)
    spec_1d_64[0] = 5.0 + 2.0j
    inv_1d_64 = invlap(spec_1d_64, rsphere=rsphere)
    assert inv_1d_64.shape == (nmdim,)
    assert inv_1d_64.dtype == np.complex128
    assert inv_1d_64[0] == 0.0 + 0.0j
    np.testing.assert_allclose(inv_1d_64, spec_1d_64 * expected_scale, rtol=1e-14, atol=1e-14)

    spec_1d_32 = spec_1d_64.astype(np.complex64)
    inv_1d_32 = invlap(spec_1d_32, rsphere=rsphere)
    assert inv_1d_32.shape == (nmdim,)
    assert inv_1d_32.dtype == np.complex64
    assert inv_1d_32[0] == 0.0 + 0.0j
    np.testing.assert_allclose(
        inv_1d_32,
        (spec_1d_32 * expected_scale.astype(np.float32)),
        rtol=1e-6,
        atol=1e-6,
    )

    spec_4d_c = (
        rng.standard_normal((2, 3, 4, nmdim))
        + 1j * rng.standard_normal((2, 3, 4, nmdim))
    ).astype(np.complex64)
    inv_4d_c = invlap(spec_4d_c, rsphere=rsphere, layout="c")
    assert inv_4d_c.shape == (2, 3, 4, nmdim)
    assert inv_4d_c.dtype == np.complex64
    np.testing.assert_allclose(
        inv_4d_c,
        spec_4d_c * expected_scale.astype(np.float32),
        rtol=1e-6,
        atol=1e-6,
    )

    spec_3d_leg = np.moveaxis(spec_4d_c[0], -1, 0).copy()
    inv_3d_leg = invlap(spec_3d_leg, rsphere=rsphere, layout="legacy")
    assert inv_3d_leg.shape == (nmdim, 3, 4)
    assert inv_3d_leg.dtype == np.complex64
    np.testing.assert_allclose(
        np.moveaxis(inv_3d_leg, 0, -1),
        inv_4d_c[0],
        rtol=0,
        atol=0,
    )

    # Empty batch (nt == 0) preserves shape and dtype
    empty_c = np.empty((0, 3, nmdim), dtype=np.complex64)
    inv_empty = invlap(empty_c, rsphere=rsphere, layout="c")
    assert inv_empty.shape == (0, 3, nmdim)
    assert inv_empty.dtype == np.complex64

    # 2. SpharmTransform and Spharmt methods match standalone invlap and Poisson identity
    sht_c = SpharmTransform(nlon, nlat, rsphere=rsphere, gridtype="regular", precision="auto")
    sht_auto = Spharmt(nlon, nlat, rsphere=rsphere, gridtype="regular", layout="auto")
    sht_leg = Spharmt(nlon, nlat, rsphere=rsphere, gridtype="regular", layout="legacy")

    np.testing.assert_allclose(sht_c.invlap(spec_4d_c), inv_4d_c, rtol=0, atol=0)
    np.testing.assert_allclose(sht_c.invlapspec(spec_4d_c), inv_4d_c, rtol=0, atol=0)
    np.testing.assert_allclose(sht_auto.invlap(spec_1d_64), inv_1d_64, rtol=0, atol=0)
    np.testing.assert_allclose(sht_leg.invlap(spec_3d_leg), inv_3d_leg, rtol=0, atol=0)

    u_c = rng.standard_normal((2, 3, nlat, nlon)).astype(np.float64)
    v_c = rng.standard_normal((2, 3, nlat, nlon)).astype(np.float64)
    vrt_spec, div_spec = sht_c.getvrtdivspec(u_c, v_c)
    psi_from_invlap = sht_c.spectogrd(sht_c.invlap(vrt_spec))
    chi_from_invlap = sht_c.spectogrd(sht_c.invlap(div_spec))
    psi_ref, chi_ref = sht_c.getpsichi(u_c, v_c)
    np.testing.assert_allclose(psi_from_invlap, psi_ref, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(chi_from_invlap, chi_ref, rtol=1e-13, atol=1e-13)

    # 3. ReducedGaussianTransform invlap / invlapspec
    pl = np.array([16, 20, 24, 28, 32, 28, 24, 20, 16], dtype=np.int32)
    npts = int(pl.sum())
    rgt = ReducedGaussianTransform(pl, rsphere=rsphere, precision="double")
    u_rg = rng.standard_normal((2, npts)).astype(np.float64)
    v_rg = rng.standard_normal((2, npts)).astype(np.float64)
    vrt_rg, div_rg = rgt.getvrtdivspec(u_rg, v_rg, ntrunc=5)
    psi_rg, chi_rg = rgt.getpsichi(u_rg, v_rg, ntrunc=5)
    np.testing.assert_allclose(rgt.spectogrd(rgt.invlap(vrt_rg)), psi_rg, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(rgt.spectogrd(rgt.invlapspec(div_rg)), chi_rg, rtol=1e-13, atol=1e-13)

    # 4. Validation errors on invalid rsphere or spectral shape
    with pytest.raises(ValidationError, match="rsphere must be positive"):
        invlap(spec_1d_64, rsphere=0.0)
    with pytest.raises(ValidationError, match="rsphere must be positive"):
        invlap(spec_1d_64, rsphere=-10.0)
    with pytest.raises(ValidationError, match="rsphere must be positive"):
        invlap(spec_1d_64, rsphere=float("nan"))
    with pytest.raises(ValidationError, match="invalid triangular spectral"):
        invlap(np.zeros(7, dtype=np.complex128))
    with pytest.raises(ValidationError, match="invalid triangular spectral"):
        invlap(np.zeros((2, 7), dtype=np.complex128), layout="c")

