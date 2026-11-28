//! Floating-point abstraction used by the transform kernels: a single sealed trait, `ShtFloat`, implemented for `f32` and `f64`.
//!
//! Purpose: every kernel in `engine.rs`, `reduced.rs`, `spectral.rs` and `fft.rs` is generic over `T: ShtFloat`, so one code path serves both single and double precision data (numpy `float32`/`complex64` and `float64`/`complex128`).
//!
//! Difference from Fortran: SPHEREPACK is compiled with a single default `real`/`complex` kind per build (the f2py layer exposes `real` arguments, plus `double precision` workspaces in some routines such as `gaqd`). Here the precision is chosen per call from the dtype of the input array, and the trait adds the `FftNum` bound so the same `T` drives `realfft`.
//!
//! Derived from NCAR SPHEREPACK 3.2 (J. Adams and P. Swarztrauber), see LICENSE.spherepack.
use num_complex::Complex;
use realfft::FftNum;
use spharm_math::SpharmFloat;

/// Sealing module: only `f32` and `f64` can implement `ShtFloat`, so downstream crates cannot add other element types.
pub(crate) mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Real scalar type (`f32` or `f64`) over which all spherical-harmonic kernels are generic.
///
/// Requirements (supertraits): `SpharmFloat` from `spharm_math` (constants such as `ZERO`, conversions `from_f64`/`to_f64`), `Copy + Default + Send + Sync`, `realfft::FftNum` (usable as a real FFT element), `AddAssign`.
///
/// No direct Fortran counterpart (Fortran selects its real kind at compile time).
pub trait ShtFloat:
    sealed::Sealed
    + SpharmFloat
    + Copy
    + Default
    + Send
    + Sync
    + FftNum
    + std::ops::AddAssign
    + 'static
{
    /// Multiplicative identity (`1`) in the implementing precision.
    const ONE: Self;
    /// The constant `0.5`. The Fortran packing routines (`onedtotwod.f`, `twodtooned.f`) use `scale = 0.5` for the relation between the real `(a, b)` coefficient arrays and the complex packed coefficients.
    const HALF: Self;
    /// The constant `2.0`.
    const TWO: Self;
    /// Size in bytes of one element; used to size tiles so that work buffers stay within a memory budget.
    const BYTES: usize = std::mem::size_of::<Self>();

    /// Fused multiply-add `self * a + b` (single rounding) in the implementing precision.
    fn mul_add(self, a: Self, b: Self) -> Self;

    /// Convert an `f64` constant to `Self`, rounding when `Self` is `f32`. Used for factors computed in `f64` (such as `-n(n+1)/rsphere^2`) and then stored in the working precision.
    #[inline(always)]
    fn from_real(v: f64) -> Self {
        <Self as SpharmFloat>::from_f64(v)
    }

    /// Complex zero `0 + 0i` in the implementing precision.
    fn complex_zero() -> Complex<Self> {
        Complex::new(Self::ZERO, Self::ZERO)
    }
}

/// Single-precision implementation (numpy `float32` / `complex64`).
impl ShtFloat for f32 {
    const ONE: Self = 1.0;
    const HALF: Self = 0.5;
    const TWO: Self = 2.0;

    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        self.mul_add(a, b)
    }
}

/// Double-precision implementation (numpy `float64` / `complex128`).
impl ShtFloat for f64 {
    const ONE: Self = 1.0;
    const HALF: Self = 0.5;
    const TWO: Self = 2.0;

    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        self.mul_add(a, b)
    }
}
