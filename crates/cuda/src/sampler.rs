//! GPU sampler chain (006-2 T3C): single-launch penalty+softmax+topk/argmax.
//!
//! One kernel launch per step on a single stream (`sampler_kernel.cu`,
//! one block of 256 threads). The kernel applies, in one pass over the
//! vocabulary: logit_bias (binary search over host-sorted pairs) →
//! frequency/presence penalties → repetition penalty (window scan) →
//! temperature (÷) → softmax (f32 max-subtracted) → min_p → top_k
//! (64-round 64-bit pair-key bisection, exact llm-samplers truncate) →
//! top_p (30-round boundary-value bisection) → gumbel-max argmax with the
//! LastMax tie-break (largest tid wins on equal scores).
//!
//! At temp=0 the kernel skips softmax/filters entirely (they are inert for
//! the greedy result — the max value always survives both truncations and
//! the boundary tie runs are cut to their largest-tid members) and runs a
//! pure hardware argmax with the LastMax rule: bit-identical with the
//! llm-samplers `SampleGreedy` semantics that the CPU adapter preserves
//! (006-2 plan D2 tier 1; deviation recording via `TokenOut::tie_break`).
//!
//! RNG contract (D2 tier 2): the host advances `RngState` exactly once per
//! step at temp>0 (the (i,p) index fold); the kernel derives per-token noise
//! as a pure function u(v) = splitmix64(base ^ splitmix64(v)) — no
//! stream-ordered RNG. temp=0 never consumes `rng` (D2 tier 1).
//!
//! NotSupported (fail-closed, atomic fallback): `bad_words` and `gumbel`
//! return [`SampleError::NotSupported`] BEFORE touching `rng` or `self` —
//! the selector (`FallbackSamplerChain`) re-dispatches to the CPU adapter
//! and counts `eager_fallback`; GPU successes are counted via the selector's
//! `sampler_gpu` counter and mirrored by [`GpuSamplerChain::launch_count`]
//! (one kernel launch per successful sample — the single-launch gate).

use std::ffi::c_void;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use reinfer_core::DeviceId;
use reinfer_jit::compile::{compile_cubin, gencode_flags};
use reinfer_jit::{JitCache, JitKey, KernelSource, probe_toolchain_for_arch};
use reinfer_kernels::{
    LaunchError, LogitsView, RngState, SampleError, SamplerChain, SamplerImpl, SamplerParams,
    TieBreak, TokenOut, UnsupportedParam,
};

use crate::buffer::{DeviceBuffer, HostBuffer, MemRef, copy};
use crate::jit::{CtxGuard, JLib, KernelFn};
use crate::stream::CudaStream;

/// Sampler kernel block size (`SAMPLER_BLOCK` in sampler_kernel.cu).
const SAMPLER_BLOCK: u32 = 256;
/// out[2] status value for a successful sample (kernel contract;
/// any other value = no finite logit).
const STATUS_OK: u32 = 0;

/// Lazily-allocated device scratch (reallocated on growth).
#[derive(Debug)]
struct Scratch {
    val: DeviceBuffer,  // [vocab] f32 penalized/temperature-scaled values
    prob: DeviceBuffer, // [vocab] f32 softmax probs (temp>0 only)
    vocab: usize,
    window: DeviceBuffer, // [cap] u32 penalty-history window
    window_cap: usize,
    bias_ids: DeviceBuffer,  // [cap] u32 logit-bias ids (ascending)
    bias_vals: DeviceBuffer, // [cap] f32 logit-bias values
    bias_cap: usize,
    out: DeviceBuffer, // [3] u32 token / tie / status
}

// SAFETY: device pointers are context-bound handles usable from any thread
// that establishes the driver context (CtxGuard per launch, jit.rs C3);
// all device access through this chain happens via `&mut self` methods
// (single logical owner), mirroring buffer.rs's ownership semantics.
// `SamplerChain` requires Send + Sync.
unsafe impl Send for Scratch {}
unsafe impl Sync for Scratch {}

impl Scratch {
    fn alloc(
        dev: DeviceId,
        vocab: usize,
        window_cap: usize,
        bias_cap: usize,
    ) -> Result<Self, LaunchError> {
        Ok(Self {
            val: DeviceBuffer::alloc(dev, vocab.max(1) * 4)?,
            prob: DeviceBuffer::alloc(dev, vocab.max(1) * 4)?,
            vocab,
            window: DeviceBuffer::alloc(dev, window_cap.max(1) * 4)?,
            window_cap,
            bias_ids: DeviceBuffer::alloc(dev, bias_cap.max(1) * 4)?,
            bias_vals: DeviceBuffer::alloc(dev, bias_cap.max(1) * 4)?,
            bias_cap,
            out: DeviceBuffer::alloc(dev, 3 * 4)?,
        })
    }
}

/// 006-2 T3C GPU sampler chain: single-launch kernel + RngState pure-function
/// index. Thread-safe (`SamplerChain: Send + Sync`): `sample` is `&mut self`
/// but the chain holds no shared mutable state across threads.
#[derive(Debug)]
pub struct GpuSamplerChain {
    dev: u32,
    stream: CudaStream,
    /// Loaded cubin (kept alive; the driver must see the code bytes until
    /// unload — JLib owns them).
    #[allow(dead_code)]
    lib: JLib,
    kernel: KernelFn,
    scratch: Option<Scratch>,
    /// Penalty history: sampled tokens fed back each step (mirror of the CPU
    /// adapter's llm-samplers `last_tokens` sequence).
    history: Vec<u32>,
    /// Successful kernel launches (single-launch gate).
    launch_count: AtomicU64,
}

impl GpuSamplerChain {
    /// Load the sampler kernel (JitCache pipeline, mirror of
    /// [`crate::engine::DenseKernels::new`]).
    pub fn new(dev: DeviceId, arch: &str, cache_dir: Option<PathBuf>) -> Result<Self, LaunchError> {
        let tc = probe_toolchain_for_arch(arch)?;
        let src = KernelSource {
            name: "gpu_sampler_kernel",
            src: include_str!("../kernels/sampler_kernel.cu"),
            headers: vec![],
            flags: gencode_flags(arch)?,
            arch: arch.to_string(),
            toolchain_ver: tc.ver_line.clone(),
        };
        let cache = JitCache::open(cache_dir)?;
        let key = JitKey::new(&src, &tc);
        let (_, cubin_path) = cache.build_once(&key, &src, || compile_cubin(&src, &tc))?;
        let bytes = std::fs::read(&cubin_path).map_err(|_| LaunchError::Fatal)?;
        let lib = JLib::from_bytes(bytes)?;
        let kernel = lib.kernel("gpu_sampler_kernel")?;
        let stream = CudaStream::new(dev)?;
        Ok(Self {
            dev: dev.index(),
            stream,
            lib,
            kernel,
            scratch: None,
            history: Vec::new(),
            launch_count: AtomicU64::new(0),
        })
    }

    /// Device index the chain runs on.
    pub fn device(&self) -> u32 {
        self.dev
    }

    /// Stream used for sampler work.
    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }

    /// Successful kernel launches (one per successful sample).
    pub fn launch_count(&self) -> u64 {
        self.launch_count.load(Ordering::Relaxed)
    }

    /// Current penalty-history length (mirror of the CPU adapter's fed tokens).
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    fn ensure_scratch(
        &mut self,
        vocab: usize,
        window_cap: usize,
        bias_cap: usize,
    ) -> Result<(), SampleError> {
        let ok = matches!(
            &self.scratch,
            Some(s) if s.vocab >= vocab && s.window_cap >= window_cap && s.bias_cap >= bias_cap
        );
        if !ok {
            let s = Scratch::alloc(DeviceId::new(self.dev), vocab, window_cap, bias_cap)
                .map_err(|e| SampleError::Chain(e.to_string()))?;
            self.scratch = Some(s);
        }
        Ok(())
    }
}

impl SamplerChain for GpuSamplerChain {
    fn sample(
        &mut self,
        logits: &LogitsView,
        params: &SamplerParams,
        rng: &mut RngState,
    ) -> Result<TokenOut, SampleError> {
        // NotSupported gate BEFORE any rng/self consumption (atomic fallback
        // contract: the selector re-dispatches without side effects).
        if !params.bad_words.is_empty() {
            return Err(SampleError::NotSupported(UnsupportedParam::BadWords));
        }
        if params.gumbel {
            return Err(SampleError::NotSupported(UnsupportedParam::Gumbel));
        }
        if logits.device().index() != self.dev {
            return Err(SampleError::Chain(format!(
                "logits on device {} but sampler chain on device {}",
                logits.device().index(),
                self.dev
            )));
        }
        let vocab = logits.vocab();
        if vocab == 0 {
            return Err(SampleError::Logits("empty logits".into()));
        }
        // temp normalization: only finite >0 samples; anything else is greedy
        // (no RNG) — mirrors the legacy "≤0 → greedy" semantics.
        let temperature = if params.temperature.is_finite() && params.temperature > 0.0 {
            params.temperature
        } else {
            0.0
        };
        // (i,p) index fold: exactly one RngState u64 per temp>0 step.
        let rng_base = if temperature > 0.0 { rng.mix().next_u64() } else { 0 };

        // Parameter mapping onto kernel off-values (kernel treats these as off).
        let freq_pen = params.frequency_penalty.unwrap_or(0.0);
        let pres_pen = params.presence_penalty.unwrap_or(0.0);
        let rep_pen = params.repeat_penalty.unwrap_or(1.0);
        let top_k: u32 = match params.top_k {
            Some(k) if k > 0 && k < vocab => k as u32,
            _ => 0, // off: None / Some(0) / keep-all (k >= vocab)
        };
        let top_p = params.top_p.unwrap_or(1.0);
        let min_p = params.min_p.unwrap_or(0.0);
        let window_len = params.repeat_last_n.min(self.history.len());

        // logit-bias pairs sorted ascending by id (kernel binary search).
        let mut pairs = params.logit_bias.clone();
        pairs.sort_by_key(|&(id, _)| id);
        let n_bias = pairs.len();
        let (mut bias_ids, mut bias_vals) = (Vec::new(), Vec::new());
        for (id, v) in &pairs {
            bias_ids.push(*id);
            bias_vals.push(*v);
        }

        self.ensure_scratch(vocab, params.repeat_last_n, n_bias)?;
        let scratch = self.scratch.as_ref().unwrap();

        // Uploads (all ordered on the sampler stream — single stream, then
        // one kernel launch, then sync; per-step ≤1 launch).
        if window_len > 0 {
            let hb =
                HostBuffer::alloc(window_len * 4).map_err(|e| SampleError::Chain(e.to_string()))?;
            // SAFETY: tail of `history` (window_len u32s) → pinned host buffer
            // of exactly window_len*4 bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.history.as_ptr().add(self.history.len() - window_len) as *const u8,
                    hb.as_ptr() as *mut u8,
                    window_len * 4,
                );
            }
            copy(
                &mut MemRef::Device(&scratch.window),
                &MemRef::Host(&hb),
                window_len * 4,
                Some(&self.stream),
            )
            .map_err(|e| SampleError::Chain(e.to_string()))?;
        }
        if n_bias > 0 {
            let hib =
                HostBuffer::alloc(n_bias * 4).map_err(|e| SampleError::Chain(e.to_string()))?;
            let hvb =
                HostBuffer::alloc(n_bias * 4).map_err(|e| SampleError::Chain(e.to_string()))?;
            // SAFETY: vectors are n_bias elements; pinned buffers are
            // n_bias*4 bytes each.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bias_ids.as_ptr() as *const u8,
                    hib.as_ptr() as *mut u8,
                    n_bias * 4,
                );
                std::ptr::copy_nonoverlapping(
                    bias_vals.as_ptr() as *const u8,
                    hvb.as_ptr() as *mut u8,
                    n_bias * 4,
                );
            }
            copy(
                &mut MemRef::Device(&scratch.bias_ids),
                &MemRef::Host(&hib),
                n_bias * 4,
                Some(&self.stream),
            )
            .map_err(|e| SampleError::Chain(e.to_string()))?;
            copy(
                &mut MemRef::Device(&scratch.bias_vals),
                &MemRef::Host(&hvb),
                n_bias * 4,
                Some(&self.stream),
            )
            .map_err(|e| SampleError::Chain(e.to_string()))?;
        }

        // Single kernel launch (C3 discipline: every arg is a local variable
        // taken by address; driver launch needs the current context guard).
        {
            let _guard =
                CtxGuard::set_current(self.dev).map_err(|e| SampleError::Chain(e.to_string()))?;
            let logits_p: *const f32 = logits.buffer().ptr() as *const f32;
            let window_p: *const u32 = scratch.window.as_ptr() as *const u32;
            let window_len_v: u32 = window_len as u32;
            let bias_ids_p: *const u32 = scratch.bias_ids.as_ptr() as *const u32;
            let bias_vals_p: *const f32 = scratch.bias_vals.as_ptr() as *const f32;
            let n_bias_v: u32 = n_bias as u32;
            let freq_v: f32 = freq_pen;
            let pres_v: f32 = pres_pen;
            let rep_v: f32 = rep_pen;
            let temp_v: f32 = temperature;
            let top_k_v: u32 = top_k;
            let top_p_v: f32 = top_p;
            let min_p_v: f32 = min_p;
            let rng_base_v: u64 = rng_base;
            let vocab_v: u32 = vocab as u32;
            let val_p: *mut f32 = scratch.val.as_ptr() as *mut f32;
            let prob_p: *mut f32 = scratch.prob.as_ptr() as *mut f32;
            let out_p: *mut u32 = scratch.out.as_ptr() as *mut u32;
            let mut args: [*mut c_void; 18] = [
                (&logits_p as *const *const f32) as *mut c_void,
                (&window_p as *const *const u32) as *mut c_void,
                (&window_len_v as *const u32) as *mut c_void,
                (&bias_ids_p as *const *const u32) as *mut c_void,
                (&bias_vals_p as *const *const f32) as *mut c_void,
                (&n_bias_v as *const u32) as *mut c_void,
                (&freq_v as *const f32) as *mut c_void,
                (&pres_v as *const f32) as *mut c_void,
                (&rep_v as *const f32) as *mut c_void,
                (&temp_v as *const f32) as *mut c_void,
                (&top_k_v as *const u32) as *mut c_void,
                (&top_p_v as *const f32) as *mut c_void,
                (&min_p_v as *const f32) as *mut c_void,
                (&rng_base_v as *const u64) as *mut c_void,
                (&vocab_v as *const u32) as *mut c_void,
                (&val_p as *const *mut f32) as *mut c_void,
                (&prob_p as *const *mut f32) as *mut c_void,
                (&out_p as *const *mut u32) as *mut c_void,
            ];
            unsafe {
                crate::jit::launch_row(
                    self.kernel,
                    &self.stream,
                    self.dev,
                    SAMPLER_BLOCK,
                    args.as_mut_ptr(),
                )
            }
            .map_err(|e| SampleError::Chain(e.to_string()))?;
        }
        self.stream.synchronize().map_err(|e| SampleError::Chain(e.to_string()))?;

        // Readback the outcome slot.
        let hb = HostBuffer::alloc(3 * 4).map_err(|e| SampleError::Chain(e.to_string()))?;
        copy(&mut MemRef::Host(&hb), &MemRef::Device(&scratch.out), 3 * 4, None)
            .map_err(|e| SampleError::Chain(e.to_string()))?;
        let status = unsafe { *(hb.as_ptr() as *const u32).add(2) };
        if status != STATUS_OK {
            return Err(SampleError::NoToken);
        }
        let token = unsafe { *(hb.as_ptr() as *const u32) };
        self.history.push(token);
        self.launch_count.fetch_add(1, Ordering::Relaxed);
        Ok(TokenOut { token, tie_break: TieBreak::LastMax })
    }

    fn variant(&self) -> SamplerImpl {
        SamplerImpl::GpuSampler
    }
}
