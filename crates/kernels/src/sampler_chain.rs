//! Sampler chain interface layer (006-2 T3A/T3B): the `SamplerChain` trait,
//! the 005 D5 full parameter surface (`SamplerParams`), the CPU adapter over
//! `crates/samplers` (llm-samplers 0.0.7), and the GPU>CPU fallback selector
//! with the 006-style counters.
//!
//! Determinism contract (006-2 plan D2, three tiers):
//! - temp=0: no-RNG path (software/hardware argmax; tie-break = FIRST max,
//!   pinned by 005 D5 and 012 semantics). NOTE: llm-samplers `SampleGreedy`
//!   breaks ties on the LAST max; the CPU adapter preserves the existing
//!   `crates/samplers` behavior bit-identically and records the deviation on
//!   every sample via [`TokenOut::tie_break`] (recorded difference, D2 tier 1).
//! - temp>0: only same-distribution with the CPU path is promised; the GPU
//!   implementation must advance a pure-function `(i,p,v)` index (005 D5
//!   math), not a stream-ordered draw.
//! - CPU lineage: llm-samplers 0.0.7 with rand `StdRng` = ChaCha12. This is a
//!   known state (not SplitMix64); it stays until the 005 pure-function
//!   migration unifies the lineage (D2 tier-3 note, inside this spec's promise
//!   scope).
//!
//! Fallback semantics (vLLM style): the selector orders providers by
//! [`SamplerImpl`] (GPU < CPU). A GPU `sample` returning
//! [`SampleError::NotSupported`] is re-dispatched to the CPU implementation
//! automatically and counted as `eager_fallback`. An implementation MUST NOT
//! consume `rng` or mutate its own state before returning `NotSupported`, so
//! re-dispatch is atomic.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::LaunchError;
use crate::logits::LogitsView;
use crate::sampler::SplitMix64;

/// Tie-break rule that produced the sampled token (006-2 D2 tier 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TieBreak {
    /// First maximum wins — the D2/005 D5 pin (012 `argmax_first` semantics).
    FirstMax,
    /// Last maximum wins — llm-samplers `SampleGreedy` (`max_by` + `total_cmp`).
    /// The CPU adapter preserves this legacy behavior; the deviation from the
    /// D2 pin is recorded (see module docs).
    LastMax,
}

/// One sampling outcome (the "TokenOut-ish" of 006-2 Interface Contracts).
#[derive(Debug, Clone, PartialEq)]
pub struct TokenOut {
    /// Sampled token id.
    pub token: u32,
    /// Tie-break rule that produced `token` (deviation marker for temp=0).
    pub tie_break: TieBreak,
}

/// A parameter of the 005 D5 chain not covered by an implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnsupportedParam {
    /// D5 stage 1 `logit_bias` (additive per-token bias).
    LogitBias,
    /// D5 stage 2 `frequency_penalty`.
    FrequencyPenalty,
    /// D5 stage 2 `presence_penalty`.
    PresencePenalty,
    /// D5 stage 3 `bad_words` (blocked token sequences).
    BadWords,
    /// D5 stage 5 `min_p` (log-domain threshold `max_val + ln(min_p)`).
    MinP,
    /// D5 stage 8 `gumbel` (pure-function Gumbel noise, 005 D5 domain).
    Gumbel,
}

/// Sampling error surface (fail-closed: no silent wrong sampling).
#[derive(Debug, Clone, PartialEq)]
pub enum SampleError {
    /// This implementation does not cover the parameter; the selector
    /// re-dispatches to the next provider (fallback channel).
    NotSupported(UnsupportedParam),
    /// Logits conversion/shape error (mirrors `reinfer_samplers::SamplerError`).
    Logits(String),
    /// Sampling chain error.
    Chain(String),
    /// The chain produced no token (empty/all-filtered distribution).
    NoToken,
}

impl fmt::Display for SampleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SampleError::NotSupported(p) => write!(f, "sampler: parameter not supported: {p:?}"),
            SampleError::Logits(e) => write!(f, "sampler: logits: {e}"),
            SampleError::Chain(e) => write!(f, "sampler: chain: {e}"),
            SampleError::NoToken => write!(f, "sampler: produced no token"),
        }
    }
}

impl std::error::Error for SampleError {}

/// Full sampling parameter surface, 005 D5 order: bias → penalties
/// (freq/presence/repeat) → bad words → temperature → min_p → top_k → top_p →
/// gumbel → argmax. At least one token is always kept (min_keep=1 in the
/// llm-samplers filters). Parameters an implementation cannot process are
/// reported via [`SampleError::NotSupported`] — never silently ignored.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplerParams {
    /// D5 stage 1: additive logit bias per token id (`(id, bias)`).
    pub logit_bias: Vec<(u32, f32)>,
    /// D5 stage 2: frequency penalty (Some(0.0) = off, vLLM semantics).
    pub frequency_penalty: Option<f32>,
    /// D5 stage 2: presence penalty (Some(0.0) = off).
    pub presence_penalty: Option<f32>,
    /// D5 stage 2: repetition penalty (only >1.0 enables; llama.cpp semantics:
    /// logits >0 divided, <0 multiplied).
    pub repeat_penalty: Option<f32>,
    /// Repetition penalty window (recent token history length).
    pub repeat_last_n: usize,
    /// D5 stage 3: blocked token sequences (bad words).
    pub bad_words: Vec<Vec<u32>>,
    /// D5 stage 4: temperature (≤0 → greedy/argmax, no RNG).
    pub temperature: f32,
    /// D5 stage 5: min-p threshold (None or Some(≤0.0) = off).
    pub min_p: Option<f32>,
    /// D5 stage 6: top-k (Some(0) = off; k > vocab is a no-op keep-all, same
    /// legacy llm-samplers semantics).
    pub top_k: Option<usize>,
    /// D5 stage 7: top-p (Some(0.0) or Some(1.0) = off, legacy filter
    /// `0 < p < 1`).
    pub top_p: Option<f32>,
    /// D5 stage 8: gumbel noise before argmax (pure-function domain).
    pub gumbel: bool,
    /// Legacy CPU RNG seed (D2 tier 3: StdRng = ChaCha12 lineage; not the 005
    /// SplitMix64 lineage — see [`RngState`]).
    pub seed: Option<u64>,
}

impl Default for SamplerParams {
    fn default() -> Self {
        Self {
            logit_bias: Vec::new(),
            frequency_penalty: None,
            presence_penalty: None,
            repeat_penalty: None,
            repeat_last_n: 64,
            bad_words: Vec::new(),
            temperature: 0.0,
            min_p: None,
            top_k: None,
            top_p: None,
            gumbel: false,
            seed: None,
        }
    }
}

/// Engine-owned sampling RNG state: the 005 D5 pure-function `rng(i,p,v)`
/// index base. Constructed once per generation (014 r2: seed injection point
/// = one SplitMix64 construction; temp=0 never consumes it — D2 tier 1). The
/// GPU implementation (T3D) advances this index; the CPU adapter keeps the
/// legacy internal StdRng (ChaCha12, D2 tier 3) and does not consume it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RngState {
    mix: SplitMix64,
}

impl RngState {
    /// New state seeded from `seed` (same seed → same stream).
    pub fn new(seed: u64) -> Self {
        Self { mix: SplitMix64::new(seed) }
    }

    /// The underlying SplitMix64 generator (GPU path consumption point).
    pub fn mix(&mut self) -> &mut SplitMix64 {
        &mut self.mix
    }
}

/// Alarm threshold for `padding_ratio` (006-2 Interface Contracts: >20%).
pub const PADDING_ALARM_THRESHOLD: f64 = 0.20;

/// 006-style sampler counters: `sampler_gpu` / `eager_fallback` /
/// `padding_ratio` (fallback ⊆ eager share; >20% alarm interface reserved —
/// the 5-minute windowed aggregation is wired by T6).
#[derive(Debug, Default)]
pub struct SamplerCounters {
    sampler_gpu: AtomicU64,
    eager_fallback: AtomicU64,
}

impl SamplerCounters {
    /// New, zeroed counters.
    pub fn new() -> Self {
        Self::default()
    }

    /// A sample was taken on the GPU path.
    pub fn record_gpu(&self) {
        self.sampler_gpu.fetch_add(1, Ordering::Relaxed);
    }

    /// A sample fell back to the CPU (eager) path.
    pub fn record_eager_fallback(&self) {
        self.eager_fallback.fetch_add(1, Ordering::Relaxed);
    }

    /// GPU-path sample count.
    pub fn sampler_gpu(&self) -> u64 {
        self.sampler_gpu.load(Ordering::Relaxed)
    }

    /// CPU fallback sample count.
    pub fn eager_fallback(&self) -> u64 {
        self.eager_fallback.load(Ordering::Relaxed)
    }

    /// Total samples observed.
    pub fn total(&self) -> u64 {
        self.sampler_gpu() + self.eager_fallback()
    }

    /// `padding_ratio`: fallback share of total samples (回退 ⊆ eager 比例);
    /// 0.0 when nothing observed.
    pub fn padding_ratio(&self) -> f64 {
        let total = self.total();
        if total == 0 { 0.0 } else { self.eager_fallback() as f64 / total as f64 }
    }

    /// Alarm interface (reserved): instantaneous check that `padding_ratio`
    /// exceeds [`PADDING_ALARM_THRESHOLD`]. The 006 contract ("5 minutes with
    /// the ratio above 20% → alarm") needs windowed aggregation, which lands
    /// with T6; this method is the reserved hook.
    pub fn padding_alarm(&self) -> bool {
        self.padding_ratio() > PADDING_ALARM_THRESHOLD
    }
}

/// Sampler implementation identity. The `repr(u8)` discriminant doubles as
/// the TuneDb variant number (006 plan D4: TuneDb 记录带变体号).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum SamplerImpl {
    /// GPU sampler chain (006-2 T3D: single-launch penalty+softmax+topk/argmax).
    GpuSampler = 0,
    /// CPU adapter over llm-samplers 0.0.7 (legacy bit-identical reference and
    /// fallback path).
    CpuAdapter = 1,
}

impl SamplerImpl {
    /// TuneDb variant number (006 plan D4).
    pub const fn tune_db_variant(self) -> u8 {
        self as u8
    }
}

/// A sampling chain: `sample` consumes logits and returns a token.
///
/// Contract (006-2 D2): `temp=0` must not consume `rng`; `temp>0` only
/// promises same-distribution vs the CPU path. On
/// [`SampleError::NotSupported`] the implementation MUST NOT consume `rng` or
/// mutate itself — the selector re-dispatches to the next provider atomically.
pub trait SamplerChain: Send + Sync {
    /// Sample one token from `logits` under `params`, advancing `rng`.
    fn sample(
        &mut self,
        logits: &LogitsView,
        params: &SamplerParams,
        rng: &mut RngState,
    ) -> Result<TokenOut, SampleError>;

    /// Implementation identity (TuneDb variant number, selection key).
    fn variant(&self) -> SamplerImpl;

    /// Runtime counters if this chain is instrumented (composite chains);
    /// `None` for bare implementations.
    fn counters(&self) -> Option<&SamplerCounters> {
        None
    }
}

/// CPU implementation: adapter over `crates/samplers` (llm-samplers 0.0.7).
///
/// Behavior contract: for the legacy parameter surface (temperature / top_k /
/// top_p / repeat_penalty / repeat_last_n / seed) this adapter produces
/// bit-identical output to calling `reinfer_samplers::Sampler` directly with
/// the same inputs — it constructs the same llm-samplers chain in the same
/// order (repeat → top-k → top-p → temperature → greedy|rand-distrib) and
/// self-feeds the sampled token exactly like the pipeline's post-sample
/// `feed()` call (014 T9 / bin pipeline.rs), so the penalty-history state
/// sequence is identical. Parameters are per-call: when they differ from the
/// parameters the chain was built with, the chain is rebuilt (which resets
/// RNG and penalty history — with a stable parameter set this never happens,
/// matching today's construct-once semantics).
///
/// Chain order note (recorded deviation): the legacy chain order
/// repeat → top-k → top-p → temperature is kept verbatim to preserve existing
/// behavior; the 005 D5 order (temperature → min_p → top_k → top_p) applies to
/// the GPU implementation and, after the 005 pure-function migration, to the
/// CPU path (D2 tier 3). At temp=0 the effective order coincides with D5
/// (penalties → top-k → top-p → argmax) for the covered surface.
///
/// Uncovered surface (explicit [`SampleError::NotSupported`], fail-closed):
/// logit_bias, frequency_penalty, presence_penalty, bad_words, min_p, gumbel.
/// llm-samplers 0.0.7 has samplers for the first five, but inserting them
/// would change the legacy chain order (and thus break bit-identical
/// preservation); they land on the CPU path with the 005 migration.
///
/// Tie-break (recorded deviation, D2 tier 1): llm-samplers `SampleGreedy`
/// picks the LAST maximum; the D2/005 D5 pin is the FIRST maximum. This
/// adapter preserves the existing behavior and reports `TieBreak::LastMax`
/// on every sample — callers that require the D2 pin must treat LastMax on
/// ties as a recorded difference.
#[derive(Debug)]
pub struct CpuSamplerChain {
    inner: reinfer_samplers::Sampler,
    built: SamplerParams,
}

impl CpuSamplerChain {
    /// Build the adapter. Parameters with an uncovered surface are rejected
    /// with `NotSupported` at build time (and again at sample time after a
    /// parameter change, via the lazy-rebuild path).
    pub fn new(params: &SamplerParams) -> Result<Self, SampleError> {
        let legacy = Self::map(params)?;
        let inner = reinfer_samplers::Sampler::new(&legacy)
            .map_err(|e| SampleError::Chain(e.to_string()))?;
        Ok(Self { inner, built: params.clone() })
    }

    /// Map the D5 surface onto the legacy `crates/samplers` parameter type,
    /// rejecting parameters this adapter does not cover. The mapping is a
    /// field-for-field copy — llm-samplers call semantics are untouched.
    fn map(params: &SamplerParams) -> Result<reinfer_samplers::SamplingParams, SampleError> {
        if !params.logit_bias.is_empty() {
            return Err(SampleError::NotSupported(UnsupportedParam::LogitBias));
        }
        if params.frequency_penalty.is_some_and(|p| p != 0.0) {
            return Err(SampleError::NotSupported(UnsupportedParam::FrequencyPenalty));
        }
        if params.presence_penalty.is_some_and(|p| p != 0.0) {
            return Err(SampleError::NotSupported(UnsupportedParam::PresencePenalty));
        }
        if !params.bad_words.is_empty() {
            return Err(SampleError::NotSupported(UnsupportedParam::BadWords));
        }
        if params.min_p.is_some_and(|p| p > 0.0) {
            return Err(SampleError::NotSupported(UnsupportedParam::MinP));
        }
        if params.gumbel {
            return Err(SampleError::NotSupported(UnsupportedParam::Gumbel));
        }
        Ok(reinfer_samplers::SamplingParams {
            temperature: params.temperature,
            top_k: params.top_k,
            top_p: params.top_p,
            repeat_penalty: params.repeat_penalty,
            repeat_last_n: params.repeat_last_n,
            seed: params.seed,
        })
    }
}

impl SamplerChain for CpuSamplerChain {
    fn sample(
        &mut self,
        logits: &LogitsView,
        params: &SamplerParams,
        _rng: &mut RngState,
    ) -> Result<TokenOut, SampleError> {
        // Lazy rebuild on parameter change (per-call params contract). On
        // NotSupported the adapter is left untouched (atomic fallback).
        if params != &self.built {
            *self = Self::new(params)?;
        }
        let host = logits.to_host();
        let tok = self.inner.sample(&host).map_err(|e| match e {
            reinfer_samplers::SamplerError::Logits(m) => SampleError::Logits(m),
            reinfer_samplers::SamplerError::Chain(m) => SampleError::Chain(m),
            reinfer_samplers::SamplerError::NoToken => SampleError::NoToken,
        })?;
        // Mirror the pipeline's post-sample feed() (pipeline.rs): identical
        // penalty-history state sequence as today.
        self.inner.feed(tok);
        Ok(TokenOut { token: tok, tie_break: TieBreak::LastMax })
    }

    fn variant(&self) -> SamplerImpl {
        SamplerImpl::CpuAdapter
    }
}

/// Composite chain: primary provider with automatic fallback (vLLM-style
/// fallback chain). A `NotSupported` from the primary is re-dispatched to the
/// next provider in order and counted as `eager_fallback`; GPU successes are
/// counted as `sampler_gpu`. When every provider rejects the parameter, the
/// primary's `NotSupported` is propagated.
pub struct FallbackSamplerChain {
    primary: Box<dyn SamplerChain>,
    fallbacks: Vec<Box<dyn SamplerChain>>,
    counters: Arc<SamplerCounters>,
}

impl FallbackSamplerChain {
    /// Composite with `primary` tried first, then `fallbacks` in order.
    pub fn new(primary: Box<dyn SamplerChain>, fallbacks: Vec<Box<dyn SamplerChain>>) -> Self {
        Self { primary, fallbacks, counters: Arc::new(SamplerCounters::new()) }
    }
}

impl SamplerChain for FallbackSamplerChain {
    fn sample(
        &mut self,
        logits: &LogitsView,
        params: &SamplerParams,
        rng: &mut RngState,
    ) -> Result<TokenOut, SampleError> {
        match self.primary.sample(logits, params, rng) {
            Ok(out) => {
                if self.primary.variant() == SamplerImpl::GpuSampler {
                    self.counters.record_gpu();
                }
                Ok(out)
            }
            Err(SampleError::NotSupported(param)) => {
                self.counters.record_eager_fallback();
                for fb in &mut self.fallbacks {
                    match fb.sample(logits, params, rng) {
                        Ok(out) => return Ok(out),
                        Err(SampleError::NotSupported(_)) => continue,
                        Err(e) => return Err(e),
                    }
                }
                Err(SampleError::NotSupported(param))
            }
            Err(e) => Err(e),
        }
    }

    fn variant(&self) -> SamplerImpl {
        self.primary.variant()
    }

    fn counters(&self) -> Option<&SamplerCounters> {
        Some(&self.counters)
    }
}

impl fmt::Debug for FallbackSamplerChain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let fallbacks: Vec<SamplerImpl> = self.fallbacks.iter().map(|p| p.variant()).collect();
        f.debug_struct("FallbackSamplerChain")
            .field("primary", &self.primary.variant())
            .field("fallbacks", &fallbacks)
            .field("counters", &self.counters)
            .finish()
    }
}

/// Select a sampler chain from registered providers: GPU preferred over CPU
/// (variant order, `GpuSampler < CpuAdapter`), vLLM-style fallback semantics.
/// A single provider is returned directly (no composite); multiple providers
/// yield a [`FallbackSamplerChain`] that re-dispatches `NotSupported`
/// automatically and counts `sampler_gpu` / `eager_fallback`. An empty
/// provider list is an explicit error (fail-closed, mirroring
/// `provider::select` — never a silent CPU fallback).
pub fn select_sampler(
    providers: Vec<Box<dyn SamplerChain>>,
) -> Result<Box<dyn SamplerChain>, LaunchError> {
    if providers.is_empty() {
        return Err(LaunchError::Fatal);
    }
    let mut ordered = providers;
    ordered.sort_by_key(|p| p.variant()); // stable: registration order kept within a tier
    let mut iter = ordered.into_iter();
    let primary = iter.next().ok_or(LaunchError::Fatal)?;
    let fallbacks: Vec<_> = iter.collect();
    if fallbacks.is_empty() {
        Ok(primary)
    } else {
        Ok(Box::new(FallbackSamplerChain::new(primary, fallbacks)))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)] // 测试断言崩溃即失败
    use super::*;
    use crate::logits::{DeviceBuffer, LogitsView};
    use reinfer_core::DeviceId;
    use std::sync::Arc;

    /// Mock logits view: the copy closure reads a captured host slice,
    /// simulating the backend's device→host copy (no GPU required).
    fn view(logits: &[f32]) -> LogitsView {
        let src: Arc<Vec<f32>> = Arc::new(logits.to_vec());
        let copy = src.clone();
        LogitsView::new(
            DeviceId::new(0),
            DeviceBuffer::new(0x1000, logits.len() * 4),
            logits.len(),
            Box::new(move || copy.as_ref().clone()),
        )
    }

    #[test]
    fn greedy_tie_break_is_last_max_recorded_deviation() {
        // Ties at index 1 and 2: llm-samplers SampleGreedy picks the LAST max
        // (idx 2); the D2/005 D5 pin is the FIRST max (idx 1). The adapter
        // preserves existing crates/samplers behavior and records the
        // deviation via TokenOut::tie_break (006-2 D2 tier-1 note).
        let logits = [1.0f32, 2.0, 2.0, 1.5];
        let params = SamplerParams::default(); // temp=0 → greedy
        let mut chain = CpuSamplerChain::new(&params).unwrap();
        let out = chain.sample(&view(&logits), &params, &mut RngState::new(1)).unwrap();
        assert_eq!(out.token, 2, "legacy llm-samplers tie-break = last max");
        assert_eq!(out.tie_break, TieBreak::LastMax);
        // Reference: direct crates/samplers call with the same input.
        let mut ref_s = reinfer_samplers::Sampler::new(&reinfer_samplers::SamplingParams {
            temperature: 0.0,
            ..Default::default()
        })
        .unwrap();
        assert_eq!(ref_s.sample(&logits).unwrap(), 2);
    }

    #[test]
    fn cpu_adapter_matches_direct_sampler_calls() {
        // Deterministic mock logits (no NaN; NaN is rejected by the caller,
        // not by the sampler — 014 T9).
        let logits = [
            0.1f32, -2.0, 1.5, 3.0, 0.5, -1.0, 2.0, 0.0, 1.0, 1.8, -0.5, 0.25, -3.0, 4.0, 0.75,
            -1.5,
        ];
        let cases = [
            SamplerParams { temperature: 0.0, ..Default::default() },
            SamplerParams {
                temperature: 0.8,
                top_k: Some(10),
                top_p: Some(0.9),
                repeat_penalty: Some(1.1),
                repeat_last_n: 8,
                seed: Some(42),
                ..Default::default()
            },
            SamplerParams { temperature: 1.0, seed: Some(7), ..Default::default() },
            SamplerParams { temperature: 0.5, top_k: Some(3), seed: Some(1), ..Default::default() },
            SamplerParams {
                temperature: 1.2,
                repeat_penalty: Some(1.05),
                repeat_last_n: 4,
                seed: Some(99),
                ..Default::default()
            },
            SamplerParams {
                temperature: 0.9,
                top_p: Some(0.8),
                seed: Some(3),
                ..Default::default()
            },
        ];
        for params in cases {
            let legacy = reinfer_samplers::SamplingParams {
                temperature: params.temperature,
                top_k: params.top_k,
                top_p: params.top_p,
                repeat_penalty: params.repeat_penalty,
                repeat_last_n: params.repeat_last_n,
                seed: params.seed,
            };
            let mut ref_s = reinfer_samplers::Sampler::new(&legacy).unwrap();
            let mut chain = CpuSamplerChain::new(&params).unwrap();
            let mut rng = RngState::new(params.seed.unwrap_or(0));
            for _ in 0..32 {
                // Direct crates/samplers path: sample then feed (pipeline.rs
                // two-step). Adapter path: single sample (self-feed). Outputs
                // must be identical every step.
                let t_ref = ref_s.sample(&logits).unwrap();
                ref_s.feed(t_ref);
                let t_chain = chain.sample(&view(&logits), &params, &mut rng).unwrap().token;
                assert_eq!(t_ref, t_chain, "adapter drifted from direct sampler call");
            }
        }
    }

    #[test]
    fn unsupported_params_are_explicit() {
        let unsupported = [
            (
                SamplerParams { logit_bias: vec![(0, -1.0)], ..Default::default() },
                UnsupportedParam::LogitBias,
            ),
            (
                SamplerParams { frequency_penalty: Some(0.5), ..Default::default() },
                UnsupportedParam::FrequencyPenalty,
            ),
            (
                SamplerParams { presence_penalty: Some(-0.3), ..Default::default() },
                UnsupportedParam::PresencePenalty,
            ),
            (
                SamplerParams { bad_words: vec![vec![1, 2, 3]], ..Default::default() },
                UnsupportedParam::BadWords,
            ),
            (SamplerParams { min_p: Some(0.05), ..Default::default() }, UnsupportedParam::MinP),
            (SamplerParams { gumbel: true, ..Default::default() }, UnsupportedParam::Gumbel),
        ];
        for (params, expected) in unsupported {
            // Build-time rejection.
            assert!(
                matches!(CpuSamplerChain::new(&params), Err(SampleError::NotSupported(p)) if p == expected)
            );
            // Sample-time rejection via the lazy-rebuild path (chain built
            // with default params, then sample() with the uncovered surface).
            let mut chain = CpuSamplerChain::new(&SamplerParams::default()).unwrap();
            let err = chain
                .sample(&view(&[1.0f32, 2.0, 3.0]), &params, &mut RngState::new(1))
                .unwrap_err();
            assert!(matches!(err, SampleError::NotSupported(p) if p == expected));
        }
    }

    #[test]
    fn effectively_off_params_are_accepted() {
        // All values that the legacy filters treat as "off" must be accepted
        // and behave like the default greedy chain.
        let off = SamplerParams {
            min_p: Some(0.0),
            frequency_penalty: Some(0.0),
            presence_penalty: Some(0.0),
            top_k: Some(0),
            top_p: Some(1.0),
            repeat_penalty: Some(1.0), // not >1.0 → legacy filter treats as off
            temperature: 0.0,
            ..Default::default()
        };
        let mut chain = CpuSamplerChain::new(&off).unwrap();
        let out = chain.sample(&view(&[1.0f32, 3.0, 2.0]), &off, &mut RngState::new(1)).unwrap();
        assert_eq!(out.token, 1, "greedy argmax over [1.0, 3.0, 2.0]");
    }

    #[test]
    fn rng_state_deterministic_same_seed() {
        let mut a = RngState::new(42);
        let mut b = RngState::new(42);
        for _ in 0..8 {
            assert_eq!(a.mix().next_u64(), b.mix().next_u64());
        }
        assert_ne!(RngState::new(1).mix().next_u64(), RngState::new(2).mix().next_u64());
        assert_eq!(SamplerImpl::GpuSampler.tune_db_variant(), 0);
        assert_eq!(SamplerImpl::CpuAdapter.tune_db_variant(), 1);
    }

    /// Test GPU stub: rejects an optional parameter, otherwise returns a
    /// fixed token (does not consume `rng` — NotSupported contract).
    struct StubGpu {
        reject: Option<UnsupportedParam>,
        token: u32,
    }

    impl SamplerChain for StubGpu {
        fn sample(
            &mut self,
            _logits: &LogitsView,
            _params: &SamplerParams,
            _rng: &mut RngState,
        ) -> Result<TokenOut, SampleError> {
            match self.reject {
                Some(p) => Err(SampleError::NotSupported(p)),
                None => Ok(TokenOut { token: self.token, tie_break: TieBreak::FirstMax }),
            }
        }

        fn variant(&self) -> SamplerImpl {
            SamplerImpl::GpuSampler
        }
    }

    #[test]
    fn selector_empty_is_error() {
        assert!(matches!(select_sampler(vec![]), Err(LaunchError::Fatal)));
    }

    #[test]
    fn selector_returns_cpu_when_no_gpu_provider() {
        // No GPU provider (non-NPU/CPU-only CI) → the selector must yield the
        // CPU adapter directly (T3 verification item ④).
        let cpu: Box<dyn SamplerChain> =
            Box::new(CpuSamplerChain::new(&SamplerParams::default()).unwrap());
        let mut chain = select_sampler(vec![cpu]).unwrap();
        assert_eq!(chain.variant(), SamplerImpl::CpuAdapter);
        assert!(chain.counters().is_none());
        let params = SamplerParams::default();
        let out = chain.sample(&view(&[1.0f32, 3.0, 2.0]), &params, &mut RngState::new(1)).unwrap();
        assert_eq!(out.token, 1);
    }

    #[test]
    fn selector_prefers_gpu_and_counts() {
        let cpu: Box<dyn SamplerChain> =
            Box::new(CpuSamplerChain::new(&SamplerParams::default()).unwrap());
        // Passed first on purpose: variant ordering must still pick the GPU.
        let gpu: Box<dyn SamplerChain> = Box::new(StubGpu { reject: None, token: 7 });
        let mut chain = select_sampler(vec![cpu, gpu]).unwrap();
        assert_eq!(chain.variant(), SamplerImpl::GpuSampler);
        let params = SamplerParams::default();
        let out = chain.sample(&view(&[1.0f32, 2.0, 3.0]), &params, &mut RngState::new(1)).unwrap();
        assert_eq!(out.token, 7);
        assert_eq!(out.tie_break, TieBreak::FirstMax);
        let counters = chain.counters().unwrap();
        assert_eq!(counters.sampler_gpu(), 1);
        assert_eq!(counters.eager_fallback(), 0);
        assert_eq!(counters.padding_ratio(), 0.0);
        assert!(!counters.padding_alarm());
    }

    #[test]
    fn selector_falls_back_to_cpu_on_not_supported_and_counts() {
        let gpu: Box<dyn SamplerChain> =
            Box::new(StubGpu { reject: Some(UnsupportedParam::MinP), token: 7 });
        let cpu: Box<dyn SamplerChain> =
            Box::new(CpuSamplerChain::new(&SamplerParams::default()).unwrap());
        let mut chain = select_sampler(vec![gpu, cpu]).unwrap();
        let params = SamplerParams::default(); // no uncovered surface → CPU handles it
        let mut rng = RngState::new(1);
        let out = chain.sample(&view(&[1.0f32, 3.0, 2.0]), &params, &mut rng).unwrap();
        assert_eq!(out.token, 1, "CPU fallback result");
        assert_eq!(out.tie_break, TieBreak::LastMax, "came from the CPU adapter");
        let counters = chain.counters().unwrap();
        assert_eq!(counters.sampler_gpu(), 0);
        assert_eq!(counters.eager_fallback(), 1);
        assert_eq!(counters.padding_ratio(), 1.0);
        assert!(counters.padding_alarm());
        // rng untouched by the whole flow (GPU NotSupported must not consume
        // it; the CPU path does not use RngState).
        assert_eq!(rng.mix().next_u64(), RngState::new(1).mix().next_u64());
    }

    #[test]
    fn selector_propagates_when_all_providers_unsupported() {
        let gpu: Box<dyn SamplerChain> =
            Box::new(StubGpu { reject: Some(UnsupportedParam::MinP), token: 7 });
        let cpu: Box<dyn SamplerChain> =
            Box::new(CpuSamplerChain::new(&SamplerParams::default()).unwrap());
        let mut chain = select_sampler(vec![gpu, cpu]).unwrap();
        let params = SamplerParams { min_p: Some(0.05), ..Default::default() };
        let err =
            chain.sample(&view(&[1.0f32, 3.0, 2.0]), &params, &mut RngState::new(1)).unwrap_err();
        assert_eq!(err, SampleError::NotSupported(UnsupportedParam::MinP));
        let counters = chain.counters().unwrap();
        assert_eq!(counters.eager_fallback(), 1, "fallback attempt counted");
    }

    #[test]
    fn counters_are_zero_before_any_sample() {
        let c = SamplerCounters::new();
        assert_eq!(c.sampler_gpu(), 0);
        assert_eq!(c.eager_fallback(), 0);
        assert_eq!(c.padding_ratio(), 0.0);
        assert!(!c.padding_alarm());
    }
}
