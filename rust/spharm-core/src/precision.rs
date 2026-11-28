use num_complex::Complex;
use realfft::FftNum;
use spharm_math::SpharmFloat;

pub(crate) mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

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
    const ONE: Self;
    const HALF: Self;
    const TWO: Self;
    const BYTES: usize = std::mem::size_of::<Self>();

    fn mul_add(self, a: Self, b: Self) -> Self;

    #[inline(always)]
    fn from_real(v: f64) -> Self {
        <Self as SpharmFloat>::from_f64(v)
    }

    fn complex_zero() -> Complex<Self> {
        Complex::new(Self::ZERO, Self::ZERO)
    }
}

impl ShtFloat for f32 {
    const ONE: Self = 1.0;
    const HALF: Self = 0.5;
    const TWO: Self = 2.0;

    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        self.mul_add(a, b)
    }
}

impl ShtFloat for f64 {
    const ONE: Self = 1.0;
    const HALF: Self = 0.5;
    const TWO: Self = 2.0;

    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        self.mul_add(a, b)
    }
}
