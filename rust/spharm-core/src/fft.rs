use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

use crate::precision::ShtFloat;

#[derive(Clone)]
pub struct FftPair<T: ShtFloat> {
    pub r2c: Arc<dyn RealToComplex<T>>,
    pub c2r: Arc<dyn ComplexToReal<T>>,
}

pub struct FftCache<T: ShtFloat> {
    plans: RwLock<HashMap<usize, FftPair<T>>>,
}

impl<T: ShtFloat> Default for FftCache<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: ShtFloat> FftCache<T> {
    pub fn new() -> Self {
        Self {
            plans: RwLock::new(HashMap::new()),
        }
    }

    pub fn get_or_plan(&self, len: usize) -> FftPair<T> {
        if let Ok(guard) = self.plans.read() {
            if let Some(pair) = guard.get(&len) {
                return pair.clone();
            }
        }
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
