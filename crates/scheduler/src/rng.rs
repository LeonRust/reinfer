//! Deterministic RNG core (plan D5).
//!
//! Every random decision in the scheduler derives from `base_seed` (CLI
//! `--seed` or `REINFER_SEED`) plus request identity and position, so that
//! replaying the same arrival sequence with the same seed is bit-identical.
//! No system RNG is ever consulted.
//!
//! `splitmix64` is a bijection on `u64` (xor-shifts and odd-constant
//! multiplications are permutations), so distinct inputs never collide.

/// Golden-ratio constant used by SplitMix64.
pub const GOLDEN_RATIO_64: u64 = 0x9E37_79B9_7F4A_7C15;

/// Mix a single `u64` (the `z` stage of SplitMix64, without the state
/// increment). Well-mixed output for any input.
#[inline]
fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// One-shot SplitMix64 mixing function: `splitmix64(x) = mix(x + GOLDEN_RATIO)`.
///
/// Bijective on `u64` (see module doc), so `x != y` implies
/// `splitmix64(x) != splitmix64(y)`.
#[inline]
pub fn splitmix64(x: u64) -> u64 {
    mix(x.wrapping_add(GOLDEN_RATIO_64))
}

/// Streaming SplitMix64 generator (deterministic; used by tests and scenario
/// builders). Standard SplitMix64 semantics: state += golden, output = mix(state).
#[derive(Debug, Clone, Copy)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// New generator with the given seed.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next well-mixed `u64`.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(GOLDEN_RATIO_64);
        mix(self.state)
    }
}

/// Raw `u64` draw: `SplitMix64(hash(seed_i, pos, vocab))` (plan D5).
#[inline]
pub fn rng_u64(seed_i: u64, pos: usize, vocab: u32) -> u64 {
    let mixed = (pos as u64).wrapping_mul(GOLDEN_RATIO_64)
        ^ (vocab as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
    splitmix64(seed_i ^ mixed)
}

/// D5 `rng(seed_i, pos, vocab) -> [0, 1)`: pure function with a single,
/// fixed definition (CPU/GPU bit-agreement convention). The mapping uses the
/// standard SplitMix64 double extraction (top 53 bits of the draw over 2^53).
#[inline]
pub fn rng(seed_i: u64, pos: usize, vocab: u32) -> f32 {
    (rng_u64(seed_i, pos, vocab) >> 11) as f32 / (1u64 << 53) as f32
}

/// Bounded draw in `[0, bound)` via the Lemire multiplication method
/// (deterministic and near-uniform without modulo bias).
#[inline]
pub fn rng_usize(seed_i: u64, pos: usize, vocab: u32, bound: usize) -> usize {
    debug_assert!(bound > 0, "rng_usize bound must be positive");
    let m = (rng_u64(seed_i, pos, vocab) as u128) * (bound as u128);
    (m >> 64) as usize
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn splitmix64_known_vector() {
        // Reference SplitMix64: seed 0, first output.
        assert_eq!(splitmix64(0), 0xE220_A839_7B1D_CDAF);
        assert_eq!(splitmix64(1), 0x910A_2DEC_8902_5CC1);
    }

    #[test]
    fn streaming_matches_oneshot() {
        // The stream and the one-shot agree exactly at the first draw
        // (both mix seed + golden); afterwards the stream state advances by
        // golden per draw (standard SplitMix64 semantics).
        let mut g = SplitMix64::new(0);
        assert_eq!(g.next_u64(), splitmix64(0));
        assert_eq!(g.next_u64(), mix(GOLDEN_RATIO_64.wrapping_mul(2)));
        assert_eq!(g.next_u64(), mix(GOLDEN_RATIO_64.wrapping_mul(3)));
    }

    #[test]
    fn splitmix64_is_injective_on_prefix() {
        // Bijection: first N outputs are pairwise distinct.
        let mut seen = std::collections::HashSet::new();
        for i in 0..1000u64 {
            assert!(seen.insert(splitmix64(i)), "collision at {i}");
        }
    }

    #[test]
    fn rng_is_in_unit_range_and_deterministic() {
        let a = rng(0xDEAD, 3, 128);
        let b = rng(0xDEAD, 3, 128);
        assert_eq!(a, b, "same inputs must give the same draw");
        assert!((0.0..1.0).contains(&a));
        let c = rng(0xDEAD, 4, 128);
        assert_ne!(a, c, "position changes the draw");
        let d = rng(0xDEAD, 3, 129);
        assert_ne!(a, d, "vocab changes the draw");
    }

    #[test]
    fn rng_usize_respects_bound_and_is_deterministic() {
        for pos in 0..32usize {
            let v = rng_usize(7, pos, 16, 16);
            assert!(v < 16, "bound violated");
        }
        assert_eq!(rng_usize(7, 3, 16, 16), rng_usize(7, 3, 16, 16));
        assert!(rng_usize(7, 0, 16, 16) != rng_usize(7, 1, 16, 16));
    }
}
