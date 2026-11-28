//! Real FFT plan cache used for the longitudinal transforms.
//!
//! Purpose: the SPHEREPACK routines transform each latitude circle along longitude with the real periodic FFT `hrfft.f` (FFTPACK `rfftf`/`rfftb`, with explicit `wsave` trigonometric tables that the caller builds with `hrffti`). Here the transform is delegated to `rustfft`/`realfft`: `RealFftPlanner` provides a real-to-complex forward plan and a complex-to-real inverse plan for a given length, and `FftCache` memoises those plans per length so repeated calls with the same `nlon` (or the same reduced-grid row length) reuse them.
//!
//! Difference from Fortran:
//! - no `wsave` work array and no initialisation call: plans are created lazily by `FftCache::get_or_plan` and shared through `Arc`;
//! - the real FFT output is the half spectrum of `len / 2 + 1` complex values (`realfft` convention), not the FFTPACK packed `r0, r1, i1, r2, i2, ...` real array of `hrfft.f`;
//! - `realfft` transforms are unnormalised in both directions, so the `1/nlon` scaling is applied by the callers;
//! - element type is generic (`f32` or `f64`, see `precision.rs`), whereas Fortran uses the default real kind.
//!
//! Derived from NCAR SPHEREPACK 3.2 (J. Adams and P. Swarztrauber), see LICENSE.spherepack.
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

use crate::precision::ShtFloat;

/// Forward and inverse real FFT plans for one transform length.
///
/// - `r2c`: real-to-complex forward plan (analysis, grid to Fourier coefficients). Fortran: `hrfft.f` / `hrfftf` (forward real FFT).
/// - `c2r`: complex-to-real inverse plan (synthesis, Fourier coefficients to grid). Fortran: `hrfft.f` / `hrfftb` (backward real FFT).
///
/// Plans are `Arc` handles, so cloning a pair is cheap.
#[derive(Clone)]
pub struct FftPair<T: ShtFloat> {
    pub r2c: Arc<dyn RealToComplex<T>>,
    pub c2r: Arc<dyn ComplexToReal<T>>,
}

/// Thread-safe cache of `FftPair` plans keyed by transform length.
///
/// Plans are guarded by an `RwLock` so concurrent readers (for example rayon workers) share them without contention once planned. Fortran: replaces the per-call `wsave` table built by `hrffti` in `hrfft.f`.
pub struct FftCache<T: ShtFloat> {
    plans: RwLock<HashMap<usize, FftPair<T>>>,
}

// Equivalent to `FftCache::new()` (empty cache).
impl<T: ShtFloat> Default for FftCache<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: ShtFloat> FftCache<T> {
    /// Create an empty cache. No plan is built until `get_or_plan` is called. Fortran: counterpart of `hrffti` (table initialisation), performed lazily.
    pub fn new() -> Self {
        Self {
            plans: RwLock::new(HashMap::new()),
        }
    }

    /// Return the forward and inverse real FFT plans for length `len`, building them on first use.
    ///
    /// Parameters: `len` is the real transform length (number of longitudes `nlon`, or the number of points on a reduced-grid latitude row).
    ///
    /// Locking: a read lock serves cache hits; on a miss the write lock is taken and the cache is checked again before planning (double-checked pattern), and a poisoned lock is recovered with `into_inner`.
    pub fn get_or_plan(&self, len: usize) -> FftPair<T> {
        if let Ok(guard) = self.plans.read() {
            if let Some(pair) = guard.get(&len) {
                return pair.clone();
            }
        }
        // Cache miss: re-check under the write lock in case another thread inserted the plan meanwhile.
        let mut guard = self
            .plans
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(pair) = guard.get(&len) {
            return pair.clone();
        }
        let mut planner = RealFftPlanner::<T>::new();
        let pair = FftPair {
            r2c: planner.plan_fft_forward(len),
            c2r: planner.plan_fft_inverse(len),
        };
        guard.insert(len, pair.clone());
        pair
    }
}
