//! S2-D: the single-threaded scheduler event loop (plan D1) + the serve
//! wiring surface for `REINFER_SCHEDULER=on`.
//!
//! # Architecture
//!
//! ```text
//!                    ┌──────────────────────────────────────────────┐
//!   HTTP handler ──► │ SchedHandle (std mpsc command channel)       │
//!   (axum/SSE)  ◄──  │   Submit / Abort{token} / Shutdown           │
//!                    └───────────────┬──────────────────────────────┘
//!                                    ▼
//!              ┌─────────────────────────────────────────────────┐
//!              │ SchedLoop<E> (one thread, plan D1)              │
//!              │   arrive → admission (D2) → select_batch (D3/D7)│
//!              │   → prefill (chunked, page-aligned)             │
//!              │   → decode batch (req_id-sorted)                │
//!              │   → per-request sampling → frames → terminal    │
//!              └───────────────┬─────────────────────────────────┘
//!                              │ BatchExecutor (per-request KV segments)
//!              ┌───────────────▼─────────────────────────────────┐
//!              │ CudaBatchExecutor (shared KvSegmentPool + the   │
//!              │   engine; the singleton/commit/stage scheme)    │
//!              └─────────────────────────────────────────────────┘
//! ```
//!
//! ## The event loop (D1)
//!
//! One thread owns every piece of mutable state: the `Req` state machines
//! (S2-A), the `KvSegmentPool` (S2-C), the engine and the sampler chains.
//! The loop blocks on the command channel while idle, otherwise it drains
//! all queued commands (arrivals/aborts in command order — the
//! deterministic input) and runs one step: admission (D2 gates) →
//! `select_batch` (decode-first, D7 victims, chunked prefill) → dispatch.
//! Prefill chunks are staged into the engine's own pool synchronously
//! (short, ~10 ms) and then committed page-exactly into the request's
//! segment; the decode batch runs in one `batch_decode_step` call; each
//! request samples on its own CPU sampler chain (seeded from the request
//! seed, D5); frames (delta text + optional logprobs) are pushed through a
//! per-request bounded tokio channel. Terminal events (EOS / stop string /
//! max output / abort) release the segment exactly once (`take_release`).
//!
//! ## Determinism (the SchedDeterminism integration)
//!
//! Everything the loop decides is a pure function of (base seed, command
//! order): arrivals are numbered by arrival sequence, batches are req_id-
//! sorted, the pool hands out segments deterministically (S2-C contract),
//! each request's sampler chain is seeded from its own seed, and the mock
//! executor's logits are a pure function of (request id, position). Two
//! runs with the same command sequence produce bit-identical transition
//! traces (asserted in `tests::loop_replays_bit_identically`).
//!
//! ## The KV pool and the anchor window
//!
//! The executor allocates one shared device buffer of `2 × kv_pages`
//! physical pages (KvStore layout: K region `[0, kv_pages)`, V region
//! `[kv_pages, 2·kv_pages)`) and a `KvSegmentPool` over the K region. Each
//! request gets a segment of `n_layer × ceil(max_model_len/32)` pages (the
//! full window — the batch kernels use an identity page table of that
//! size). The **top window** `[kv_pages − window, kv_pages)` is anchored:
//! allocated once with `alloc_from_end` and never freed, so request
//! segments (first-fit from the front) can never reach it.
//!
//! The batch kernels derive each pool's V region as
//! `kv_base + pool_pages·page_bytes` where
//! `pool_pages = max(base_pages + n_layer·pp)` over the batch. For the V
//! slots to be stable across batches — the singleton/commit copies below
//! address them directly — `pool_pages` must be constant. The executor
//! therefore appends the **anchor as a fixed phantom request** (token 0,
//! pos 0, kv_len 1) to every B≥2 decode batch: the anchor's segment extent
//! is `(kv_pages − window) + window = kv_pages`, so `pool_pages ==
//! kv_pages` for every batch regardless of the real requests' segments,
//! and the V region is exactly the KvStore layout
//! (`kv_base + kv_pages·page_bytes`, i.e. `KvStore::v_ptr()`). The anchor's
//! logits row is discarded; its K/V slots live at the very top of both
//! regions, untouched by real requests.
//!
//! ## The singleton scheme (engine-pool staging)
//!
//! `batch_decode_step` routes B=1 to the engine's `step` (its own pool;
//! `SegRef` ignored). The loop keeps the **singleton**: the request whose
//! KV currently lives in the engine pool. A lone decoder decodes there
//! (B=1, zero copies); when a batch forms, the singleton's KV is flushed
//! into its segment (page-exact D2D copies) before the batch runs, and a
//! lone decoder whose KV is in a segment copies it into the engine pool
//! first. Prefill stages into the engine pool; a single-request world
//! adopts the staged chunk (plus the segment prefix for chunked prefill),
//! a multi-request world commits the chunk to the segment. All copies are
//! synchronous `cudaMemcpy` (D2D) under `CtxGuard` — the engine's stream
//! work is complete (logits readback) whenever the loop runs them.
//!
//! ## V1 scope and known limits
//!
//! - Prefill is serial (one chunk per step) and chunked prefill is inert
//!   under the full-window admission bound (each admitted request fits, so
//!   the batch token budget never runs dry — D7 victims never trigger);
//!   the code path is wired and tested through the mock executor.
//! - Sampling runs on per-request CPU chains (host logits); the GPU chain
//!   is a later wave. `temperature=0` (greedy) is bit-identical to the
//!   serial path.
//! - The frame channel is bounded (256); a slow client stalls the loop
//!   (blocking_send). A dropped receiver aborts the request.
//! - `stop` strings are token patterns (the serving layer's job to encode,
//!   OpenAI-style stop text); the serve layer currently passes none.

use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::sync::Arc;
use std::sync::mpsc as std_mpsc;

use reinfer_kernels::{CpuSamplerChain, LogitsView, RngState, SamplerChain, SamplerParams};
use reinfer_memory::segment::{KvPoolStats, KvSegment, KvSegmentPool};
use reinfer_scheduler::admission::{
    AdmissionConfig, AdmissionVerdict, EmaTracker, EstimateInput, RequestEstimate, check_admission,
    estimate, is_busy,
};
use reinfer_scheduler::batch::{DecodingReq, WaitingReq, select_batch};
use reinfer_scheduler::policy::SchedulePolicy;
use reinfer_scheduler::radix::{PrefixHit, TokenRadixCache};
use reinfer_scheduler::replay::{Event, TraceEntry};
use reinfer_scheduler::req::{ConfirmEvent, Req, ReqId, ReqState};
use reinfer_scheduler::rng::splitmix64;

use crate::pipeline::{GenParams, TokenOut, token_out};

/// Engine `BLOCK_LEN` (engine.rs) — the loop's page length in tokens.
pub const BLOCK_LEN: usize = 32;

// ---------------------------------------------------------------------------
// Executor abstraction (the engine stays behind this interface; S2-B+ batch
// efficiency and prefill-seg backends plug in here without touching the loop)
// ---------------------------------------------------------------------------

/// One request of a decode batch step, req_id-sorted (design report: decode
/// batch 一律按 req_id 排序).
#[derive(Debug, Clone, Copy)]
pub struct ExecReq {
    /// Request id (also the batch ordering key).
    pub id: ReqId,
    /// The token to embed and predict from.
    pub token: u32,
    /// Absolute sequence position of `token` (`cached_len - 1`).
    pub pos: usize,
    /// KV window length (`cached_len`; the attention window).
    pub kv_len: usize,
    /// The request's segment (pool physical pages).
    pub seg: KvSegment,
}

/// Executor failure (engine / pool / device copy).
#[derive(Debug)]
pub enum ExecError {
    /// Engine-level failure (launch/JIT/embedding/...).
    Engine(String),
    /// KV pool allocation failure.
    Pool(String),
    /// Device copy failure.
    Copy(String),
    /// Any other executor failure.
    Msg(String),
}

impl fmt::Display for ExecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExecError::Engine(m) | ExecError::Pool(m) | ExecError::Copy(m) | ExecError::Msg(m) => {
                f.write_str(m)
            }
        }
    }
}

impl std::error::Error for ExecError {}

/// The backend behind the loop (one instance owns the KV pool + engine).
///
/// The staging/singleton vocabulary mirrors the engine-pool scheme above;
/// the loop drives it as a pure state machine and the executor implements
/// the device reality (or a deterministic mock in tests).
pub trait BatchExecutor: Send {
    /// Allocate a request segment of `n_pages` pages.
    fn alloc_segment(&mut self, n_pages: usize) -> Result<KvSegment, ExecError>;
    /// Release a request segment (exactly once per request, mirroring the
    /// Req machine's release guard).
    fn free_segment(&mut self, seg: KvSegment);
    /// Prefill `ids` into the engine pool (flushing the current singleton
    /// first). The engine pool afterwards holds the staged chunk.
    fn prefill(&mut self, ids: &[u32]) -> Result<(), ExecError>;
    /// P3-01/016 r2 D3: prefix-cache hit prefill — a single sequential path
    /// (flush the singleton, copy the cached prefix run into the engine pool,
    /// decode-step the remaining suffix tokens one at a time; each step's
    /// attention reads `[0, pos+1]` including the copied prefix — the FMHA
    /// batch prefill is context-blind, so hits must not use it). Afterwards
    /// the engine pool holds the full prompt KV; the loop adopts the
    /// singleton with `adopt_singleton(id, seg, 0, prompt_len)` (no copy —
    /// the pool is already complete).
    fn prefill_prefix_hit(&mut self, hit: PrefixHit, ids_suffix: &[u32]) -> Result<(), ExecError>;
    /// P3-01/016 r2 D2: refill the first `prefix_pages` pages of `seg` into
    /// the cache on a **normal Done release** — per-layer `ref_` on the
    /// prefix run, then free the whole segment (the prefix pages drop 2 → 1
    /// = cache-owned, the suffix pages drop to 0 = back to the pool).
    /// Infallible on the host side. Abort/preempt release MUST keep using
    /// plain `free_segment` (their segments are unreliable — review #4).
    ///
    /// **Flush first** (review #2, the real fix): a B=1 world keeps the
    /// request's KV in the engine pool — the segment is only materialized
    /// when it is adopted/committed/flushed. If `id` is the current
    /// singleton, its KV must be copied into `seg` BEFORE the prefix run
    /// is refilled, or the cache would hold never-written pages.
    fn refill_prefix(&mut self, id: ReqId, seg: KvSegment, prefix_pages: u32);
    /// P3-01/016 D2: release one cached run's pool references (LRU eviction
    /// callback — the front-end's `Evicted` list; per-layer unref, mirror of
    /// `refill_prefix`).
    fn unref_prefix(&mut self, base_page: u32, pages: u32);
    /// Commit the staged chunk to `seg`: tokens `[start, start + len)` are
    /// copied from the engine pool into the segment at page offset
    /// `start / block_len`.
    fn commit_stage(&mut self, seg: KvSegment, start: usize, len: usize) -> Result<(), ExecError>;
    /// Adopt the staged chunk as the engine-pool singleton: `[0, start)`
    /// (earlier chunks, when the request is chunked) is copied from the
    /// segment into the engine pool, and the engine pool now owns the
    /// request's full KV `[0, start + len)`.
    fn adopt_singleton(
        &mut self,
        id: ReqId,
        seg: KvSegment,
        start: usize,
        len: usize,
    ) -> Result<(), ExecError>;
    /// Drop the singleton if it belongs to `id` (terminal / abort /
    /// preempt — its engine-pool KV becomes garbage, overwritten by the
    /// next prefill).
    fn drop_singleton(&mut self, id: ReqId);
    /// Decode the batch (req_id-sorted); returns one logits row per request
    /// in batch order. B=1 routes to the engine's single-request path.
    fn decode_batch(&mut self, reqs: &[ExecReq]) -> Result<Vec<Vec<f32>>, ExecError>;
    /// KV pool statistics (S2-C surface for serve diagnostics; the loop
    /// itself only frees segments, conservation is asserted in tests).
    #[allow(dead_code)]
    fn pool_stats(&self) -> KvPoolStats;
}

// ---------------------------------------------------------------------------
// Loop configuration and the serve-facing surface
// ---------------------------------------------------------------------------

/// Static scheduler-loop configuration (derive deterministic behavior).
#[derive(Clone)]
pub struct SchedLoopConfig {
    /// Base seed (`REINFER_SEED`, default 0) — part of the deterministic
    /// input (plan D5).
    pub base_seed: u64,
    /// Sampling vocabulary size.
    pub vocab: usize,
    /// Compute device index (logits-view tag only in this wave).
    pub dev: u32,
    /// Transformer layers (window = `n_layer × ceil(max_model_len/32)`).
    pub n_layer: usize,
    /// Token slots per KV page (must equal the engine's BLOCK_LEN).
    pub block_len: usize,
    /// Effective context window (`--max-model-len`).
    pub max_model_len: usize,
    /// Shared KV pool capacity in pages.
    pub kv_pages: usize,
    /// User cap on concurrent running requests (`--max-num-seqs`).
    pub max_num_seqs: usize,
    /// Max tokens per prefill chunk (chunk budget; V1 default:
    /// `max_model_len`, i.e. whole prompts — the chunked machinery stays
    /// wired for later waves).
    pub chunk_size: usize,
    /// Hard step cap (0 = unlimited; termination safety for tests).
    pub max_steps: usize,
    /// Detokenization (delta-text frames) — the loop is tokenizer-agnostic.
    pub detok: Arc<dyn Fn(&[u32]) -> String + Send + Sync>,
    /// P3-01/016: prefix-cache page budget (per-layer pages; 0 = cache off).
    pub prefix_cache_pages: u64,
}

impl SchedLoopConfig {
    /// Pages per request window.
    pub fn window_pages(&self) -> usize {
        self.n_layer * self.max_model_len.div_ceil(self.block_len)
    }

    /// Token capacity of the allocatable pool (the anchor window reserved).
    pub fn pool_tokens(&self) -> u64 {
        (self.kv_pages - self.window_pages()) as u64 * self.block_len as u64
    }

    /// Concurrent-request cap: full windows that fit in the allocatable
    /// pool, bounded by the user's `--max-num-seqs`.
    pub fn admit_cap(&self) -> usize {
        ((self.kv_pages - self.window_pages()) / self.window_pages()).min(self.max_num_seqs)
    }
}

/// One generated-token / terminal frame pushed to the request's channel.
#[derive(Debug, Clone)]
pub enum SchedFrame {
    /// One generated token (delta detokenized text + optional logprobs).
    Token {
        /// Detokenized delta text since the previous frame (may be empty).
        delta: String,
        /// Per-token logprobs when requested.
        out: Option<TokenOut>,
    },
    /// Generation finished (EOS / stop / max output).
    Done {
        /// Whether generation stopped on EOS (OpenAI `finish_reason`).
        stopped_by_eos: bool,
        /// Whether generation stopped on a `stop` sequence (S3-1).
        stopped_by_stop: bool,
        /// Generated token count.
        tokens: usize,
        /// Prompt token count.
        prompt_tokens: usize,
    },
    /// Generation failed; the request was aborted by the loop.
    Error { message: String },
}

/// One client request handed to the loop.
#[derive(Debug)]
pub struct SubmitRequest {
    /// Prompt token ids.
    pub ids: Vec<u32>,
    /// Sampling parameters.
    pub params: GenParams,
    /// EOS token id.
    pub eos: Option<u32>,
    /// Max generated output tokens.
    pub max_tokens: usize,
    /// Stop token patterns (matched on the token stream, D8).
    pub stop: Vec<Vec<u32>>,
    /// Top-N logprobs per token (0 = none).
    pub logprobs_top_n: usize,
    /// Opaque client handle for `SchedHandle::abort`.
    pub token: u64,
    /// Frame channel (the loop blocks while it is full; a dropped receiver
    /// aborts the request).
    pub tx: tokio::sync::mpsc::Sender<SchedFrame>,
}

enum SchedCmd {
    Submit(SubmitRequest),
    /// Explicit abort by client token. The current serve path aborts on
    /// disconnect via the failed blocking_send instead; the API stays for
    /// multi-client control (and is exercised by the abort tests).
    #[allow(dead_code)]
    Abort {
        token: u64,
    },
    Shutdown,
}

/// Cross-thread handle to a running [`SchedLoop`].
#[derive(Debug)]
pub struct SchedHandle {
    tx: std_mpsc::Sender<SchedCmd>,
    join: Option<std::thread::JoinHandle<()>>,
}

impl SchedHandle {
    /// Spawn the loop thread. `init` runs on the loop thread (CUDA context,
    /// engine load, pool build — the scheduler thread owns the device) and
    /// must return the executor plus the loop configuration. Blocks until
    /// `init` completes (engine load happens at serve startup, like the
    /// serial path).
    pub fn spawn<E, F>(init: F) -> Result<SchedHandle, String>
    where
        E: BatchExecutor + 'static,
        F: FnOnce() -> Result<(E, SchedLoopConfig), String> + Send + 'static,
    {
        let (tx, rx) = std_mpsc::channel::<SchedCmd>();
        let (done_tx, done_rx) = std_mpsc::channel::<Result<(), String>>();
        let join = std::thread::Builder::new()
            .name("reinfer-sched".into())
            .spawn(move || {
                // init 结束即握手（run 是无限事件循环——done 若等 run
                // 完成，主线程 `done_rx.recv()` 永不返回，serve 永不
                // listen；2026-09-01 验收发现并修复）。
                match init() {
                    Ok((exec, cfg)) => {
                        let _ = done_tx.send(Ok(()));
                        let mut loop_ = SchedLoop::new(exec, cfg, rx);
                        loop_.run();
                    }
                    Err(e) => {
                        let _ = done_tx.send(Err(e));
                    }
                }
            })
            .map_err(|e| format!("spawn sched thread: {e}"))?;
        match done_rx.recv() {
            Ok(Ok(())) => Ok(SchedHandle { tx, join: Some(join) }),
            Ok(Err(e)) => Err(e),
            Err(_) => Err("sched thread died during init".into()),
        }
    }

    /// Submit a request (the loop assigns the deterministic request id).
    pub fn submit(&self, req: SubmitRequest) -> Result<(), String> {
        self.tx.send(SchedCmd::Submit(req)).map_err(|e| format!("scheduler stopped: {e}"))
    }

    /// Abort a submitted request by its opaque client token (idempotent).
    /// The current serve path aborts via the failed blocking_send (client
    /// disconnect); this explicit channel stays for multi-client control.
    #[allow(dead_code)]
    pub fn abort(&self, token: u64) {
        let _ = self.tx.send(SchedCmd::Abort { token });
    }
}

impl Drop for SchedHandle {
    fn drop(&mut self) {
        let _ = self.tx.send(SchedCmd::Shutdown);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

/// `REINFER_SCHEDULER` switch: on for "1"/"on"/"true"/"yes", off otherwise
/// (default off — the serial path stays the stable default).
pub fn scheduler_env_on() -> bool {
    std::env::var("REINFER_SCHEDULER")
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "on" | "true" | "yes"))
        .unwrap_or(false)
}

/// Base seed for the scheduler loop (`REINFER_SEED`, default 0).
pub fn base_seed_env() -> u64 {
    std::env::var("REINFER_SEED").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(0)
}

/// `REINFER_PREFIX_CACHE` switch: off for "0"/"off"/"false"/"no", on
/// otherwise (default on with the scheduler — 016 r2).
pub fn prefix_cache_env_on() -> bool {
    std::env::var("REINFER_PREFIX_CACHE")
        .map(|v| !matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "off" | "false" | "no"))
        .unwrap_or(true)
}

/// Prefix-cache page budget (per-layer pages): `REINFER_PREFIX_CACHE_PAGES`
/// if set, else 10% of `kv_pages` (≥ 1); 0 when the cache is off.
/// `minimum` — usually `MIN_BLOCKS` per entry — is only validated by the
/// cache front-end (an entry that exceeds the budget is rejected at refill).
pub fn prefix_cache_pages_env(kv_pages: usize, enabled: bool) -> u64 {
    if !enabled {
        return 0;
    }
    std::env::var("REINFER_PREFIX_CACHE_PAGES")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or_else(|| ((kv_pages as u64) * 10) / 100)
        .max(1)
}

// ---------------------------------------------------------------------------
// The loop
// ---------------------------------------------------------------------------

/// Per-request state the loop owns (the Req machine + serving concerns).
struct ReqMeta {
    prompt: Vec<u32>,
    max_output: usize,
    /// Request-owned sampler chain (per-request penalty window, seeded
    /// deterministically — D5).
    chain: CpuSamplerChain,
    sampler_params: SamplerParams,
    rng: RngState,
    /// Current input token (last prompt token at PrefillDone).
    cur: u32,
    /// Confirmed generated tokens.
    generated: Vec<u32>,
    /// Detokenize delta watermark.
    last_len: usize,
    logprobs_top_n: usize,
    tx: tokio::sync::mpsc::Sender<SchedFrame>,
}

/// Outcome of a loop run (determinism/accounting diagnostics).
/// Test-only surface: the serve path runs the loop to completion inside the
/// loop thread and discards the accounting.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg(test)]
pub struct SchedOutcome {
    /// Transition trace (the SchedDeterminism record).
    pub trace: Vec<TraceEntry>,
    /// Final `(id, state, cached_len, device_len)` per request, req_id-sorted.
    pub finals: Vec<(ReqId, ReqState, usize, usize)>,
    /// Total tokens dispatched ("先分配").
    pub dispatched: u64,
    /// Total tokens returned at release ("后释放").
    pub returned: u64,
    /// Steps executed.
    pub steps: usize,
}

/// The single-threaded scheduler event loop (plan D1).
pub struct SchedLoop<E: BatchExecutor> {
    cfg: SchedLoopConfig,
    exec: E,
    cmd_rx: std_mpsc::Receiver<SchedCmd>,
    reqs: BTreeMap<ReqId, Req>,
    meta: BTreeMap<ReqId, ReqMeta>,
    /// Waiting queue (preempted requests go to the head, D7).
    waiting: VecDeque<ReqId>,
    /// Decoding requests (regenerated deterministically each step).
    decoding: Vec<ReqId>,
    /// Live request segments (pool physical pages).
    segs: BTreeMap<ReqId, KvSegment>,
    /// Client token → request id.
    token_ids: BTreeMap<u64, ReqId>,
    /// Live request estimates (D2 working set).
    estimates: BTreeMap<ReqId, RequestEstimate>,
    /// Output-length EMA (D2), updated on request completion.
    ema: EmaTracker,
    arrival_seq: u64,
    step: usize,
    step_first_chunk_sum: u64,
    trace: Vec<TraceEntry>,
    dispatched: u64,
    returned: u64,
    /// P3-01/016: prefix cache front-end (None = disabled). Owned here, used
    /// on the single thread; the executor performs the pool ops.
    cache: Option<TokenRadixCache>,
    cache_hits: usize,
    cache_refills: usize,
}

impl<E: BatchExecutor> SchedLoop<E> {
    /// New loop over the given executor and configuration.
    fn new(exec: E, cfg: SchedLoopConfig, cmd_rx: std_mpsc::Receiver<SchedCmd>) -> Self {
        let cache_pages = cfg.prefix_cache_pages;
        Self {
            cache: (cache_pages > 0).then(|| TokenRadixCache::new(cache_pages)),
            cfg,
            exec,
            cmd_rx,
            reqs: BTreeMap::new(),
            meta: BTreeMap::new(),
            waiting: VecDeque::new(),
            decoding: Vec::new(),
            segs: BTreeMap::new(),
            token_ids: BTreeMap::new(),
            estimates: BTreeMap::new(),
            ema: EmaTracker::new(),
            arrival_seq: 0,
            step: 0,
            step_first_chunk_sum: 0,
            trace: Vec::new(),
            dispatched: 0,
            returned: 0,
            cache_hits: 0,
            cache_refills: 0,
        }
    }

    /// The event loop: block for commands while idle, otherwise drain the
    /// command queue (deterministic arrival order) and run one step.
    /// Returns when the command channel closes or a Shutdown command
    /// arrives.
    pub fn run(&mut self) {
        loop {
            if self.is_idle() {
                match self.cmd_rx.recv() {
                    Ok(cmd) => {
                        if self.handle_cmd(cmd) {
                            return; // shutdown
                        }
                    }
                    Err(_) => return, // all handles dropped
                }
            } else {
                while let Ok(cmd) = self.cmd_rx.try_recv() {
                    if self.handle_cmd(cmd) {
                        return;
                    }
                }
                if self.cfg.max_steps > 0 && self.step >= self.cfg.max_steps {
                    eprintln!("reinfer: sched: step cap {} hit", self.cfg.max_steps);
                    return;
                }
                self.iterate();
            }
        }
    }

    /// Run to completion and return the outcome plus the executor (tests).
    #[cfg(test)]
    fn finish(mut self) -> (SchedOutcome, E) {
        self.run();
        let finals: Vec<(ReqId, ReqState, usize, usize)> = self
            .reqs
            .iter()
            .map(|(&id, r)| (id, r.state(), r.cached_len(), r.device_len()))
            .collect();
        let out = SchedOutcome {
            trace: self.trace,
            finals,
            dispatched: self.dispatched,
            returned: self.returned,
            steps: self.step,
        };
        (out, self.exec)
    }

    // -- command handling ------------------------------------------------

    /// Handle one command; returns true on shutdown.
    fn handle_cmd(&mut self, cmd: SchedCmd) -> bool {
        match cmd {
            SchedCmd::Submit(req) => {
                self.submit(req);
                false
            }
            SchedCmd::Abort { token } => {
                if let Some(&id) = self.token_ids.get(&token) {
                    self.request_abort(id, None);
                }
                false
            }
            SchedCmd::Shutdown => true,
        }
    }

    /// Enqueue a request: derive the deterministic id, build the per-
    /// request sampler state, join the waiting queue.
    fn submit(&mut self, req: SubmitRequest) {
        if req.ids.is_empty() {
            let _ = req.tx.blocking_send(SchedFrame::Error { message: "empty prompt".into() });
            return;
        }
        let id = ReqId::derive(self.cfg.base_seed, self.arrival_seq);
        let arrival_seq = self.arrival_seq;
        self.arrival_seq += 1;
        let r = Req::new(id, self.cfg.base_seed, arrival_seq, req.stop.clone(), req.eos);
        self.reqs.insert(id, r);
        self.trace(id, Event::Arrived);
        // Sampling seed: an explicit user seed wins; otherwise the D5
        // per-request derivation (identical to the Req machine's seed_i).
        let sampler_params = SamplerParams {
            temperature: req.params.temperature,
            top_k: req.params.top_k,
            top_p: req.params.top_p,
            repeat_penalty: req.params.repeat_penalty,
            frequency_penalty: req.params.frequency_penalty,
            presence_penalty: req.params.presence_penalty,
            repeat_last_n: 64, // legacy pipeline penalty window (unchanged)
            seed: Some(
                req.params.seed.unwrap_or_else(|| splitmix64(self.cfg.base_seed ^ id.as_u64())),
            ),
            ..SamplerParams::default()
        };
        let chain = match CpuSamplerChain::new(&sampler_params) {
            Ok(c) => c,
            Err(e) => {
                let _ = req
                    .tx
                    .blocking_send(SchedFrame::Error { message: format!("sampler init: {e}") });
                return;
            }
        };
        self.token_ids.insert(req.token, id);
        self.meta.insert(
            id,
            ReqMeta {
                prompt: req.ids.clone(),
                max_output: req.max_tokens,
                chain,
                sampler_params,
                rng: RngState::new(0), // the CPU chain self-seeds; unused
                cur: 0,
                generated: Vec::new(),
                last_len: 0,
                logprobs_top_n: req.logprobs_top_n,
                tx: req.tx,
            },
        );
        // Estimates are inserted on admission only — the D2 working set is
        // decoding + candidate, never the waiting queue (waiting requests
        // would trip the TooManyRequests gate forever).
        self.waiting.push_back(id);
    }

    // -- the step --------------------------------------------------------

    /// One scheduling step: admission → batch selection → prefill dispatch →
    /// decode batch.
    fn iterate(&mut self) {
        self.admit_schedule();
        self.decode_step();
        self.step += 1;
    }

    fn is_idle(&self) -> bool {
        self.waiting.is_empty() && self.decoding.is_empty()
    }

    /// Record a transition with the resulting cursors.
    fn trace(&mut self, req: ReqId, event: Event) {
        let r = &self.reqs[&req];
        self.trace.push(TraceEntry {
            step: self.step,
            req,
            event,
            cached_len: r.cached_len(),
            device_len: r.device_len(),
        });
    }

    fn kv_usage_tokens(&self) -> u64 {
        self.reqs.values().map(|r| r.device_len() as u64).sum()
    }

    fn is_busy(&self) -> bool {
        is_busy(self.kv_usage_tokens(), self.cfg.pool_tokens(), 0.8)
    }

    fn current_estimate(&self, id: ReqId) -> RequestEstimate {
        let r = &self.reqs[&id];
        let m = &self.meta[&id];
        let chunked_remaining = if matches!(r.state(), ReqState::Chunked) {
            r.prompt_len().saturating_sub(r.cached_len()) as u64
        } else {
            0
        };
        estimate(EstimateInput {
            input_len: m.prompt.len() as u64,
            max_new_tokens: m.max_output as u64,
            has_out_len: r.output_tokens() as u64,
            shm_kv_len: 0,
            chunked_remaining,
            chunk_size: self.cfg.chunk_size as u64,
            busy: self.is_busy(),
            ema_req_out_len: self.ema.ema().round() as u64,
            max_waiting_tokens: 16,
        })
    }

    fn admission_cfg(&self) -> AdmissionConfig {
        AdmissionConfig {
            max_total_pages: (self.cfg.kv_pages - self.cfg.window_pages()) as u64,
            page_size: self.cfg.block_len,
            running_max_req_size: self.cfg.admit_cap(),
            batch_max_tokens: self.cfg.pool_tokens(),
            chunked_budget_multiplier: 2,
            router_token_ratio: 0.8,
            max_waiting_tokens: 16,
        }
    }

    /// Admission (D2) + batch selection + prefill dispatch/confirm, in the
    /// S2-A order (mirrors the SchedDeterminism harness).
    fn admit_schedule(&mut self) {
        self.step_first_chunk_sum = 0;
        // Admission eligibility in waiting-queue order (D2 gates).
        let queue: Vec<ReqId> = self.waiting.iter().copied().collect();
        let mut eligible: Vec<WaitingReq> = Vec::new();
        for id in queue {
            let est = self.current_estimate(id);
            let mut working: Vec<(ReqId, RequestEstimate)> =
                self.estimates.iter().map(|(&k, &v)| (k, v)).collect();
            if let Some(slot) = working.iter_mut().find(|(k, _)| *k == id) {
                slot.1 = est; // refresh the stale estimate
            } else {
                working.push((id, est));
            }
            let first_chunk = if self.reqs[&id].state() == ReqState::Waiting {
                Some(self.cfg.chunk_size.min(self.meta[&id].prompt.len()) as u64)
            } else {
                None
            };
            match check_admission(
                &working,
                first_chunk,
                self.step_first_chunk_sum,
                self.is_busy(),
                &self.admission_cfg(),
            ) {
                AdmissionVerdict::Admitted => {
                    if let Some(fc) = first_chunk {
                        self.step_first_chunk_sum += fc;
                    }
                    self.estimates.insert(id, est);
                    let r = &self.reqs[&id];
                    eligible.push(WaitingReq {
                        id,
                        arrival_seq: r.arrival_seq(),
                        resume: r.resume(),
                        prefix_id: None,
                        prompt_len: r.prompt_len().max(self.meta[&id].prompt.len()),
                        cached_len: r.cached_len(),
                        max_chunk: self.cfg.chunk_size,
                    });
                    self.trace(id, Event::Admitted);
                }
                AdmissionVerdict::Denied { .. } => {
                    self.trace(id, Event::Waited);
                }
            }
        }
        // Batch selection (decode-first, D7 victims, chunked prefill). The
        // step budget is the still-available KV capacity (lightllm:
        // `max_total_token_num - total_token_size`).
        let budget = self.cfg.pool_tokens().saturating_sub(self.kv_usage_tokens()) as usize;
        let decoding_view: Vec<DecodingReq> = self
            .decoding
            .iter()
            .map(|&id| {
                let r = &self.reqs[&id];
                DecodingReq { id, arrival_seq: r.arrival_seq(), resume: r.resume() }
            })
            .collect();
        let sel = select_batch(&eligible, &decoding_view, budget, 0, SchedulePolicy::Fcfc);
        // D7 victims: preempt (cursors zeroed), release the segment, back
        // to the waiting-queue head.
        for id in &sel.preempted {
            let dev = self.reqs[id].device_len();
            self.reqs.get_mut(id).expect("victim live").preempt().expect("decoding victim");
            self.returned += dev as u64;
            self.trace(*id, Event::Preempted);
            self.remove_from_decoding(*id);
            self.waiting.push_front(*id);
            if let Some(seg) = self.segs.remove(id) {
                self.exec.free_segment(seg);
            }
            self.exec.drop_singleton(*id);
        }
        // Prefill assignments: dispatch → stage into the engine → confirm
        // synchronously (D1: single-threaded, CPU state mutated serially).
        for &(id, start, end) in &sel.prefill {
            let prompt_len = self.meta[&id].prompt.len();
            // Page-aligned chunk end (page-exact D2D copies); the final
            // chunk is never rounded. A chunk that aligns to zero tokens
            // (budget < one page) waits for a better budget.
            let end = align_chunk_end(end, prompt_len, self.cfg.block_len);
            if end <= start {
                self.trace(id, Event::Waited);
                continue;
            }
            match self.reqs[&id].state() {
                ReqState::Waiting | ReqState::Preempted => {
                    let seg = match self.exec.alloc_segment(self.cfg.window_pages()) {
                        Ok(s) => s,
                        Err(e) => {
                            eprintln!("reinfer: sched: segment alloc failed: {e}");
                            self.trace(id, Event::Waited);
                            continue;
                        }
                    };
                    self.segs.insert(id, seg);
                    self.reqs
                        .get_mut(&id)
                        .expect("live req")
                        .start_prefill(prompt_len, self.meta[&id].max_output, end - start)
                        .expect("start prefill");
                    self.trace(id, Event::PrefillStart);
                }
                ReqState::Chunked => {
                    self.reqs
                        .get_mut(&id)
                        .expect("live req")
                        .dispatch_chunk(end)
                        .expect("dispatch chunk");
                }
                s => unreachable!("prefill assignment to {s:?}"),
            }
            self.dispatched += (end - start) as u64;
            // P3-01/016 r2 D3: prefix-cache hit for the FIRST chunk of a
            // Waiting request (start == 0; chunked continuations stay on the
            // full path). The hit short-circuits the staged chunk into the
            // engine pool directly (copy the cached prefix run + decode-step
            // the suffix), then falls into the usual confirm/commit/adopt
            // flow unchanged — the engine pool afterwards holds the whole
            // chunk (adopt copies nothing, commit copies page-exact).
            // Accounting: the hit prefix tokens are NOT recomputed by this
            // request; `dispatched` counts them anyway (they were produced
            // by the earlier request that filled the cache) — the
            // dispatch/return conservation check stays balanced, and the
            // cache's own pages are covered by the pool-refcount
            // conservation (asserted separately in tests).
            let mut cache_hit = false;
            // v1: a hit only when the WHOLE prompt is a single chunk
            // (`end == prompt.len()` — final chunks are never rounded);
            // chunked continuations stay on the full path (their hit
            // coverage vs. chunk accounting is a v2 concern).
            if start == 0 && end == self.meta[&id].prompt.len() {
                if let Some(c) = &self.cache
                    && let Some(hit) = c.lookup(&self.meta[&id].prompt)
                {
                    let suffix = &self.meta[&id].prompt[hit.key_len..end];
                    if let Err(e) = self.exec.prefill_prefix_hit(hit, suffix) {
                        self.request_abort(id, Some(&format!("prefix hit: {e}")));
                        continue;
                    }
                    self.cache_hits += 1;
                    cache_hit = true;
                }
            }
            // Stage the chunk into the engine pool (short serial work),
            // then confirm. (Skipped on a hit — already staged.)
            if !cache_hit && let Err(e) = self.exec.prefill(&self.meta[&id].prompt[start..end]) {
                self.request_abort(id, Some(&format!("prefill: {e}")));
                continue;
            }
            let seg = self.segs[&id];
            match self.reqs.get_mut(&id).expect("live req").confirm() {
                ConfirmEvent::ChunkConfirmed => {
                    self.trace(id, Event::ChunkDone);
                    if let Err(e) = self.exec.commit_stage(seg, start, end - start) {
                        self.request_abort(id, Some(&format!("commit: {e}")));
                        continue;
                    }
                }
                ConfirmEvent::PrefillDone => {
                    self.trace(id, Event::PrefillDone);
                    self.add_to_decoding(id);
                    self.remove_from_waiting(id);
                    let m = self.meta.get_mut(&id).expect("live meta");
                    m.cur = *m.prompt.last().expect("non-empty prompt");
                    // Singleton adoption: a lone decoder's KV lives in the
                    // engine pool (B=1 path, zero copies); a multi-request
                    // world commits the staged chunk to the segment.
                    if self.decoding.len() == 1 {
                        if let Err(e) = self.exec.adopt_singleton(id, seg, start, end - start) {
                            self.request_abort(id, Some(&format!("adopt: {e}")));
                            continue;
                        }
                    } else if let Err(e) = self.exec.commit_stage(seg, start, end - start) {
                        self.request_abort(id, Some(&format!("commit: {e}")));
                        continue;
                    }
                }
                ev => unreachable!("{ev:?}"),
            }
        }
    }

    /// Generate and confirm one decode token per decoding request, in
    /// req_id order (design report: decode batch 按 req_id 排序).
    fn decode_step(&mut self) {
        self.decoding.sort_unstable();
        let ids: Vec<ReqId> = self.decoding.clone();
        if ids.is_empty() {
            return;
        }
        let exec_reqs: Vec<ExecReq> = ids
            .iter()
            .map(|&id| ExecReq {
                id,
                token: self.meta[&id].cur,
                pos: self.reqs[&id].cached_len() - 1,
                kv_len: self.reqs[&id].cached_len(),
                seg: self.segs[&id],
            })
            .collect();
        let logits_all = match self.exec.decode_batch(&exec_reqs) {
            Ok(l) => l,
            Err(e) => {
                // Engine-level failure: abort the whole batch determin-
                // istically (per-request error frames).
                let msg = format!("decode: {e}");
                for id in &ids {
                    if !matches!(self.reqs[id].state(), ReqState::Done | ReqState::Aborted) {
                        self.request_abort(*id, Some(&msg));
                    }
                }
                return;
            }
        };
        for (i, &id) in ids.iter().enumerate() {
            if self.reqs[&id].state() != ReqState::Decode {
                continue; // aborted mid-step
            }
            let logits = &logits_all[i];
            if logits.iter().all(|l| l.is_nan()) {
                self.request_abort(id, Some("logits contain only NaN — refuse to sample"));
                continue;
            }
            let sampled = {
                let m = self.meta.get_mut(&id).expect("live meta");
                sample_token(m, logits, self.cfg.dev, self.cfg.vocab)
            };
            let (token, tokout) = match sampled {
                Ok(v) => v,
                Err(e) => {
                    self.request_abort(id, Some(&e));
                    continue;
                }
            };
            self.dispatched += 1;
            let ev = self.reqs.get_mut(&id).expect("live req").decode_step(token);
            match ev {
                ConfirmEvent::DecodeConfirmed { token } => {
                    self.trace(id, Event::DecodeToken { token });
                    let m = self.meta.get_mut(&id).expect("live meta");
                    m.cur = token;
                    m.generated.push(token);
                    let full = (self.cfg.detok)(&m.generated);
                    let delta = if full.len() > m.last_len {
                        let d = full[m.last_len..].to_string();
                        m.last_len = full.len();
                        d
                    } else {
                        String::new()
                    };
                    if m.tx.blocking_send(SchedFrame::Token { delta, out: tokout }).is_err() {
                        // The receiver (client) is gone — abort.
                        self.request_abort(id, None);
                    }
                }
                ConfirmEvent::Stopped => {
                    self.trace(id, Event::Stopped);
                    self.terminal(id, false, true);
                }
                ConfirmEvent::Eos => {
                    self.trace(id, Event::Eos);
                    self.terminal(id, true, false);
                }
                ConfirmEvent::MaxOutput => {
                    self.trace(id, Event::MaxOutput);
                    self.terminal(id, false, false);
                }
                ev => unreachable!("{ev:?}"),
            }
        }
    }

    /// Terminal accounting: Done frame, exactly-once release guard, EMA
    /// update, working-set removal, segment release ("后释放").
    fn terminal(&mut self, id: ReqId, stopped_by_eos: bool, stopped_by_stop: bool) {
        let m = self.meta.get_mut(&id).expect("live meta");
        let frame = SchedFrame::Done {
            stopped_by_eos,
            stopped_by_stop,
            tokens: m.generated.len(),
            prompt_tokens: m.prompt.len(),
        };
        let _ = m.tx.blocking_send(frame);
        let dev = self.reqs[&id].device_len();
        self.returned += dev as u64;
        self.reqs.get_mut(&id).expect("live req").take_release();
        let out = self.reqs[&id].output_tokens() as f64;
        self.ema.update(out);
        self.estimates.remove(&id);
        // P3-01/016 r2 D2: ONE release point for the cache refill (only the
        // normal Done path; abort/preempt keep plain `free_segment` — their
        // segments are unreliable, review #4). The refill itself flushes
        // the request's singleton (a B=1 world keeps its KV in the engine
        // pool; a refilled-but-never-flushed segment would poison the next
        // cache hit — review #2). `drop_singleton` stays for the paths
        // that do not refill (cache off / cache declined).
        if let Some(seg) = self.segs.remove(&id) {
            self.refill_prefix_release(seg, id);
        } else {
            self.exec.drop_singleton(id);
        }
        self.remove_from_decoding(id);
        self.remove_from_waiting(id);
        self.token_ids.retain(|_, v| *v != id);
        self.meta.remove(&id);
    }

    /// P3-01/016 r2 D2: refill decision on a normal Done release (called
    /// only from `terminal` — the single cache refill point; abort and
    /// preempt keep plain `free_segment`). Order matters:
    ///
    /// ```text
    /// L  = floor(prompt_len / block_len)            // page-aligned blocks
    /// L < MIN_BLOCKS (or 0)         → free(seg)
    /// same aligned key present      → copy/copied…  → free(seg) + touch(key)
    ///                                  (review #3: no ref_ — the old entry
    ///                                   already owns the pool references)
    /// new key                       → insert(key, base, L):
    ///                                  Ok(evicted)  → unref evicted runs,
    ///                                  then refill_prefix(seg, L) → refs +
    ///                                  free (infallible host-side)
    ///                                  Err          → free(seg)
    /// ```
    fn refill_prefix_release(&mut self, seg: KvSegment, id: ReqId) {
        let Some(c) = self.cache.as_mut() else {
            self.exec.free_segment(seg);
            return;
        };
        let prompt = &self.meta[&id].prompt;
        let L = prompt.len() / self.cfg.block_len;
        if L == 0 || L < reinfer_scheduler::radix::MIN_BLOCKS {
            self.exec.free_segment(seg);
            return;
        }
        let aligned = L * self.cfg.block_len;
        let key = &prompt[..aligned];
        if c.lookup(key).map(|h| h.key_len == aligned).unwrap_or(false) {
            // 同键（review #3）：不 ref_ ——先看树是否已有该精确键;裸 free，
            // 旧 entry 的 recency 用 touch 刷新（新键的 insert 才 ref）。
            self.exec.free_segment(seg);
            c.touch(key);
            return;
        }
        match c.insert(key, seg.base_page as u32, L as u32) {
            Ok(evicted) => {
                for e in evicted {
                    self.exec.unref_prefix(e.base_page, e.pages);
                }
                self.exec.refill_prefix(id, seg, L as u32);
                self.cache_refills += 1;
            }
            Err(e) => {
                eprintln!("reinfer: sched: prefix-cache refill declined: {e:?}");
                self.exec.free_segment(seg);
            }
        }
    }

    /// Abort a live request (`frame` = Some → push an Error frame first;
    /// None → silent, e.g. client disconnect). Idempotent (tombstone).
    fn request_abort(&mut self, id: ReqId, frame: Option<&str>) {
        if matches!(self.reqs[&id].state(), ReqState::Done | ReqState::Aborted) {
            return;
        }
        if let Some(msg) = frame {
            if let Some(m) = self.meta.get(&id) {
                let _ = m.tx.blocking_send(SchedFrame::Error { message: msg.to_string() });
            }
        }
        let dev = self.reqs[&id].device_len();
        self.reqs.get_mut(&id).expect("live req").abort();
        self.trace(id, Event::Aborted);
        self.returned += dev as u64;
        self.estimates.remove(&id);
        if let Some(seg) = self.segs.remove(&id) {
            self.exec.free_segment(seg);
        }
        self.exec.drop_singleton(id);
        self.remove_from_decoding(id);
        self.remove_from_waiting(id);
        self.token_ids.retain(|_, v| *v != id);
        self.meta.remove(&id);
    }

    fn add_to_decoding(&mut self, id: ReqId) {
        if !self.decoding.contains(&id) {
            self.decoding.push(id);
        }
    }

    fn remove_from_decoding(&mut self, id: ReqId) {
        self.decoding.retain(|&x| x != id);
    }

    fn remove_from_waiting(&mut self, id: ReqId) {
        self.waiting.retain(|&x| x != id);
    }
}

/// Round a chunk end down to a whole-page boundary (page-exact D2D copies
/// for the commit/adopt copies); the final chunk (end == prompt_len) is
/// never rounded.
fn align_chunk_end(end: usize, prompt_len: usize, block_len: usize) -> usize {
    if end >= prompt_len { prompt_len } else { end / block_len * block_len }
}

/// Sample one token on the request's own chain (host logits — the batch
/// returns rows; each request builds a host-backed view).
fn sample_token(
    meta: &mut ReqMeta,
    logits: &[f32],
    dev: u32,
    vocab: usize,
) -> Result<(u32, Option<TokenOut>), String> {
    let t0 = std::time::Instant::now();
    let view = LogitsView::new(
        reinfer_core::DeviceId::new(dev),
        reinfer_kernels::DeviceBuffer::new(0, 0),
        vocab,
        {
            let lg = logits.to_vec();
            move || lg.clone()
        },
    );
    let out = meta
        .chain
        .sample(&view, &meta.sampler_params, &mut meta.rng)
        .map_err(|e| format!("sampler: {e}"))?;
    let tokout = if meta.logprobs_top_n > 0 {
        Some(token_out(logits, out.token, meta.logprobs_top_n))
    } else {
        None
    };
    if std::env::var("REINFER_PROF_SAMPLER").as_deref() == Ok("1") {
        eprintln!("reinfer: prof: sampler host ({vocab} vocab) took {:?}", t0.elapsed());
    }
    Ok((out.token, tokout))
}

// ---------------------------------------------------------------------------
// CUDA executor (the S2-B engine behind the BatchExecutor interface)
// ---------------------------------------------------------------------------

/// The CUDA batch executor: shared KV pool + engine + the singleton /
/// staging scheme (see the module docs).
#[cfg(feature = "cuda")]
pub struct CudaBatchExecutor {
    engine: reinfer_cuda::engine::Engine,
    dev: u32,
    /// Shared pool bookkeeping (physical pages).
    pool: KvSegmentPool,
    /// Shared pool device store (K + V regions, KvStore layout).
    store: reinfer_cuda::decode::KvStore,
    /// The top window, allocated once, never freed (see module docs).
    anchor: KvSegment,
    n_layer: usize,
    /// Pages per layer per window (`ceil(max_model_len / block_len)`).
    pp: usize,
    kv_heads: usize,
    d: usize,
    block_len: usize,
    /// The engine-pool singleton (see module docs).
    singleton: Option<Singleton>,
}

/// The request whose KV currently lives in the engine pool.
#[cfg(feature = "cuda")]
struct Singleton {
    id: ReqId,
    seg: KvSegment,
    /// Confirmed KV length in tokens (`[0, kv_len)` written in the pool).
    kv_len: usize,
}

#[cfg(feature = "cuda")]
impl CudaBatchExecutor {
    /// Load the engine and build the shared KV pool (the serve path).
    ///
    /// `kv_pages` must be at least one window (`n_layer ×
    /// ceil(max_len/32)`) — the anchor plus the allocatable pool.
    pub fn load(
        dev: u32,
        model_dir: &std::path::Path,
        max_len: usize,
        kv_pages: usize,
    ) -> Result<Self, String> {
        use reinfer_core::DeviceId;
        use reinfer_cuda::{CudaContext, CudaStream};
        let ctx = CudaContext::init(DeviceId::new(dev))
            .map_err(|e| format!("cuda init (device {dev}): {e}"))?;
        let _stream = CudaStream::new(ctx.device_id()).map_err(|e| format!("stream: {e}"))?;
        let arch = reinfer_cuda::arch::resolve_arch().map_err(|e| format!("arch: {e}"))?;
        let engine = reinfer_cuda::engine::Engine::load(
            ctx.device_id().clone(),
            &arch,
            Some(std::env::temp_dir().join("reinfer-jit-dense")),
            model_dir,
            max_len,
        )
        .map_err(|e| format!("engine load: {e}"))?;
        let cfg = engine.config();
        let block_len = BLOCK_LEN;
        let pp = max_len.div_ceil(block_len);
        let n_layer = cfg.n_layer;
        let kv_heads = cfg.kv_heads;
        let d = cfg.head_dim;
        let window = n_layer * pp;
        if kv_pages < window {
            return Err(format!(
                "KV pool {kv_pages} pages < one window ({window}) — raise --max-model-len or free memory"
            ));
        }
        let store = reinfer_cuda::decode::KvStore::alloc(
            DeviceId::new(dev),
            kv_pages,
            block_len,
            cfg.kv_heads,
            cfg.head_dim,
        )
        .map_err(|e| format!("kv pool alloc: {e}"))?;
        let mut pool = KvSegmentPool::new(reinfer_memory::pool::BlockLen::B32, kv_pages);
        // The anchor: the top window, allocated once, never freed. The
        // executor appends it to every B≥2 batch as a phantom request so
        // the kernels' pool_pages == kv_pages (fixed V region — see the
        // module docs).
        let anchor = pool.alloc_from_end(window).map_err(|e| format!("anchor alloc: {e:?}"))?;
        Ok(Self {
            engine,
            dev,
            pool,
            store,
            anchor,
            n_layer,
            pp,
            kv_heads,
            d,
            block_len,
            singleton: None,
        })
    }

    /// Bytes of one physical page (all layers, f16 K+V).
    fn page_bytes(&self) -> usize {
        self.block_len * self.kv_heads * self.d * 2
    }

    /// Flush the singleton into its segment: copy its engine-pool KV
    /// `[0, kv_len)` into the segment (page-exact, per layer).
    fn flush_singleton(&mut self) -> Result<(), ExecError> {
        if let Some(s) = self.singleton.take() {
            self.copy_engine_to_pool(s.seg, 0, s.kv_len.div_ceil(self.block_len))?;
        }
        Ok(())
    }

    /// Copy `pages` pages per layer from the engine pool into the shared
    /// pool at `seg`'s layer run offset `dst_layer0` (pages).
    fn copy_engine_to_pool(
        &self,
        seg: KvSegment,
        dst_layer0: usize,
        pages: usize,
    ) -> Result<(), ExecError> {
        use reinfer_cuda::_cudarc::runtime::sys;
        use reinfer_cuda::jit::CtxGuard;
        if pages == 0 {
            return Ok(());
        }
        let pb = self.page_bytes();
        let e = self.engine.kv_store();
        let k_src = e.k_ptr() as *const u8;
        let v_src = e.v_ptr() as *const u8;
        let k_dst = self.store.k_ptr() as *mut u8;
        let v_dst = self.store.v_ptr() as *mut u8;
        let _guard =
            CtxGuard::set_current(self.dev).map_err(|e| ExecError::Copy(format!("{e}")))?;
        let bytes = pages * pb;
        for li in 0..self.n_layer {
            let src_off = li * self.pp * pb;
            let dst_off = (seg.base_page + dst_layer0 + li * self.pp) * pb;
            unsafe {
                sys::cudaMemcpy(
                    k_dst.add(dst_off) as *mut core::ffi::c_void,
                    k_src.add(src_off) as *const core::ffi::c_void,
                    bytes,
                    sys::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                )
            }
            .result()
            .map_err(|e| ExecError::Copy(format!("K region: {e:?}")))?;
            unsafe {
                sys::cudaMemcpy(
                    v_dst.add(dst_off) as *mut core::ffi::c_void,
                    v_src.add(src_off) as *const core::ffi::c_void,
                    bytes,
                    sys::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                )
            }
            .result()
            .map_err(|e| ExecError::Copy(format!("V region: {e:?}")))?;
        }
        Ok(())
    }

    /// Copy `pages` pages per layer from the shared pool (at `seg`'s layer
    /// run offset `src_layer0`) into the engine pool.
    fn copy_pool_to_engine(
        &self,
        seg: KvSegment,
        src_layer0: usize,
        pages: usize,
    ) -> Result<(), ExecError> {
        use reinfer_cuda::_cudarc::runtime::sys;
        use reinfer_cuda::jit::CtxGuard;
        if pages == 0 {
            return Ok(());
        }
        let pb = self.page_bytes();
        let e = self.engine.kv_store();
        let k_dst = e.k_ptr() as *mut u8;
        let v_dst = e.v_ptr() as *mut u8;
        let k_src = self.store.k_ptr() as *const u8;
        let v_src = self.store.v_ptr() as *const u8;
        let _guard =
            CtxGuard::set_current(self.dev).map_err(|e| ExecError::Copy(format!("{e}")))?;
        let bytes = pages * pb;
        for li in 0..self.n_layer {
            let src_off = (seg.base_page + src_layer0 + li * self.pp) * pb;
            let dst_off = li * self.pp * pb;
            unsafe {
                sys::cudaMemcpy(
                    k_dst.add(dst_off) as *mut core::ffi::c_void,
                    k_src.add(src_off) as *const core::ffi::c_void,
                    bytes,
                    sys::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                )
            }
            .result()
            .map_err(|e| ExecError::Copy(format!("K region: {e:?}")))?;
            unsafe {
                sys::cudaMemcpy(
                    v_dst.add(dst_off) as *mut core::ffi::c_void,
                    v_src.add(src_off) as *const core::ffi::c_void,
                    bytes,
                    sys::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                )
            }
            .result()
            .map_err(|e| ExecError::Copy(format!("V region: {e:?}")))?;
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl BatchExecutor for CudaBatchExecutor {
    fn alloc_segment(&mut self, n_pages: usize) -> Result<KvSegment, ExecError> {
        self.pool.alloc(n_pages).map_err(|e| ExecError::Pool(format!("segment alloc: {e:?}")))
    }

    fn free_segment(&mut self, seg: KvSegment) {
        self.pool.free(seg);
    }

    fn prefill(&mut self, ids: &[u32]) -> Result<(), ExecError> {
        // The engine pool must not hold a stale singleton across the new
        // staged chunk.
        self.flush_singleton()?;
        self.engine.prefill_batch(ids).map_err(|e| ExecError::Engine(format!("prefill: {e}")))?;
        Ok(())
    }

    fn prefill_prefix_hit(&mut self, hit: PrefixHit, ids_suffix: &[u32]) -> Result<(), ExecError> {
        // 016 r2 D3: one sequential path — flush first (a stale singleton
        // must be committed BEFORE its pool pages are overwritten — review
        // #6), copy the cached prefix run (pool K/V, per-layer `li*pp`
        // stride, exactly like copy_engine_to_pool's inverse layout), then
        // decode-step the suffix tokens: each step writes its KV slot at
        // `prefix_tokens + i` and its attention reads the full window
        // `[0, pos+1]` (T1a tests: full-prefix copies are bit-identical to
        // all-write runs; partial copies are NOT — untested prefix slots
        // would be read as garbage by layer 0, so the copy must cover
        // `[0, prefix_tokens)` completely, which the hit's page-aligned run
        // does by construction).
        self.flush_singleton()?;
        let prefix_tokens = hit.key_len;
        let prefix_pages = hit.pages as usize;
        let run = KvSegment { base_page: hit.base_page as usize, n_pages: prefix_pages };
        self.copy_pool_to_engine(run, 0, prefix_pages)?;
        for (i, &tok) in ids_suffix.iter().enumerate() {
            let pos = prefix_tokens + i;
            self.engine
                .step(tok, pos, pos + 1)
                .map_err(|e| ExecError::Engine(format!("prefix-hit step: {e}")))?;
        }
        Ok(())
    }

    fn refill_prefix(&mut self, id: ReqId, seg: KvSegment, prefix_pages: u32) {
        // 016 r2 #2 (the real fix): a B=1 world keeps the request's KV in
        // the engine pool — the segment is only materialized on flush/
        // commit. Flush the singleton FIRST (page-exact, same direction as
        // `flush_singleton`) so the refilled prefix holds written KV.
        if let Some(s) = self.singleton.take().filter(|s| s.id == id) {
            if let Err(e) = self.copy_engine_to_pool(seg, 0, s.kv_len.div_ceil(self.block_len)) {
                // Cannot happen under normal execution (D2D under CtxGuard);
                // regardless, release the whole segment and skip the cache.
                eprintln!("reinfer: sched: refill flush failed: {e} — plain release");
                self.pool.free(seg);
                return;
            }
        }
        if prefix_pages == 0 {
            self.pool.free(seg);
            return;
        }
        // 016 r2 D2: the cache's per-layer run is `(base + li*pp, +L)` —
        // take one reference per layer BEFORE freeing the segment (the
        // prefix pages drop 2→1, the suffix pages 1→0 back to the pool).
        for li in 0..self.n_layer {
            self.pool.ref_(KvSegment {
                base_page: seg.base_page + li * self.pp,
                n_pages: prefix_pages as usize,
            });
        }
        self.pool.free(seg);
    }

    fn unref_prefix(&mut self, base_page: u32, pages: u32) {
        for li in 0..self.n_layer {
            self.pool.unref(KvSegment {
                base_page: (base_page as usize) + li * self.pp,
                n_pages: pages as usize,
            });
        }
    }

    fn commit_stage(&mut self, seg: KvSegment, start: usize, len: usize) -> Result<(), ExecError> {
        self.copy_engine_to_pool(seg, start / self.block_len, len.div_ceil(self.block_len))
    }

    fn adopt_singleton(
        &mut self,
        id: ReqId,
        seg: KvSegment,
        start: usize,
        len: usize,
    ) -> Result<(), ExecError> {
        // Chunked continuation: the earlier chunks already live in the
        // segment — bring them back into the engine pool so the singleton's
        // KV is contiguous `[0, start + len)` there.
        if start > 0 {
            self.copy_pool_to_engine(seg, 0, start.div_ceil(self.block_len))?;
        }
        self.singleton = Some(Singleton { id, seg, kv_len: start + len });
        Ok(())
    }

    fn drop_singleton(&mut self, id: ReqId) {
        if self.singleton.as_ref().map(|s| s.id) == Some(id) {
            self.singleton = None;
        }
    }

    fn decode_batch(&mut self, reqs: &[ExecReq]) -> Result<Vec<Vec<f32>>, ExecError> {
        use reinfer_cuda::engine::{BatchReq, SegRef};
        if reqs.is_empty() {
            return Err(ExecError::Msg("empty decode batch".into()));
        }
        if reqs.len() == 1 {
            let r = &reqs[0];
            if self.singleton.as_ref().map(|s| s.id) != Some(r.id) {
                // B=1 with the KV in a segment: copy it into the engine
                // pool (the engine's own pool is the singleton staging).
                self.copy_pool_to_engine(r.seg, 0, r.kv_len.div_ceil(self.block_len))?;
            }
            let lg = self
                .engine
                .batch_decode_step(&[BatchReq {
                    token: r.token,
                    pos: r.pos,
                    kv: SegRef::engine(r.kv_len),
                }])
                .map_err(|e| ExecError::Engine(format!("decode: {e}")))?;
            self.singleton = Some(Singleton { id: r.id, seg: r.seg, kv_len: r.kv_len });
            return Ok(lg);
        }
        // B>1: flush the singleton first (its KV must move into its
        // segment — the batch only touches the shared pool), then run the
        // batch on the shared pool (the A3 shared-pool shape). The anchor
        // rides along as a phantom request so the kernels' pool_pages ==
        // kv_pages for every batch (fixed V region — module docs); its
        // logits row is dropped.
        self.flush_singleton()?;
        let kv_ptr = self.store.k_ptr() as *mut u16;
        let mut b_reqs: Vec<BatchReq> = reqs
            .iter()
            .map(|r| BatchReq {
                token: r.token,
                pos: r.pos,
                kv: SegRef { kv: kv_ptr, base_pages: r.seg.base_page as u32, len: r.kv_len },
            })
            .collect();
        b_reqs.push(BatchReq {
            token: 0,
            pos: 0,
            kv: SegRef { kv: kv_ptr, base_pages: self.anchor.base_page as u32, len: 1 },
        });
        let mut logits = self
            .engine
            .batch_decode_step(&b_reqs)
            .map_err(|e| ExecError::Engine(format!("decode: {e}")))?;
        logits.pop(); // the anchor's row
        Ok(logits)
    }

    fn pool_stats(&self) -> KvPoolStats {
        self.pool.stats()
    }
}

// ---------------------------------------------------------------------------
// Tests: deterministic mock executor + the SchedDeterminism integration
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use std::collections::BTreeMap as Map;
    use std::sync::mpsc as std_mpsc;

    /// Pseudo-logit: a pure function of (request id, position, vocab
    /// entry) — deterministic token generation without a model (mirrors
    /// the SchedDeterminism harness).
    fn pseudo_logit(seed: u64, pos: usize, v: u32) -> u64 {
        let h = (pos as u64)
            .wrapping_mul(reinfer_scheduler::rng::GOLDEN_RATIO_64)
            .wrapping_add((v as u64).wrapping_mul(0x94D0_49BB_1331_11EB));
        splitmix64(seed ^ h)
    }

    /// Deterministic mock: implements the staging/singleton fiction
    /// faithfully (segment contents tracked by length; every flush/commit/
    /// adopt invariant asserted) so the loop's executor call order is
    /// validated without a GPU.
    struct MockExecutor {
        pool: KvSegmentPool,
        vocab: usize,
        /// Engine-pool fiction: the staged chunk's token ids.
        staged: Option<Vec<u32>>,
        /// Per-segment stored tokens, keyed by base page (the segment KV
        /// fiction; content length matters, values are placeholders).
        segs: Map<usize, Vec<u32>>,
        /// Singleton fiction: (id, seg, kv_len).
        singleton: Option<(ReqId, KvSegment, usize)>,
        /// P3-01/016: prefix-hit adoption marker — set by
        /// `prefill_prefix_hit` (the engine-pool fiction already holds the
        /// full prompt KV), consumed by `adopt_singleton(start=0)`.
        cache_adopt: Option<usize>,
        cache_hits: usize,
        allocs: usize,
        frees: usize,
    }

    impl MockExecutor {
        fn new(kv_pages: usize, vocab: usize) -> Self {
            Self {
                pool: KvSegmentPool::new(reinfer_memory::pool::BlockLen::B32, kv_pages),
                vocab,
                staged: None,
                segs: Map::new(),
                singleton: None,
                cache_adopt: None,
                cache_hits: 0,
                allocs: 0,
                frees: 0,
            }
        }

        fn flush(&mut self) {
            if let Some((_, seg, kv_len)) = self.singleton.take() {
                let s = self.segs.get_mut(&seg.base_page).expect("singleton segment live");
                assert!(s.len() <= kv_len, "flush cannot overshoot");
                s.extend(std::iter::repeat(0u32).take(kv_len - s.len()));
            }
        }
    }

    impl BatchExecutor for MockExecutor {
        fn alloc_segment(&mut self, n_pages: usize) -> Result<KvSegment, ExecError> {
            self.allocs += 1;
            let seg = self.pool.alloc(n_pages).map_err(|e| ExecError::Pool(format!("{e:?}")))?;
            self.segs.insert(seg.base_page, Vec::new());
            Ok(seg)
        }

        fn free_segment(&mut self, seg: KvSegment) {
            self.frees += 1;
            assert!(
                self.segs.remove(&seg.base_page).is_some(),
                "free of an unknown/unreleased segment {seg:?}"
            );
            self.pool.free(seg);
        }

        fn prefill(&mut self, ids: &[u32]) -> Result<(), ExecError> {
            self.flush();
            assert!(self.staged.is_none(), "one staged chunk at a time");
            self.staged = Some(ids.to_vec());
            Ok(())
        }

        fn prefill_prefix_hit(
            &mut self,
            hit: PrefixHit,
            ids_suffix: &[u32],
        ) -> Result<(), ExecError> {
            // Engine-pool fiction: the full prompt KV is present (the hit
            // run was "copied", the suffix "decoded") — nothing staged; the
            // loop adopts the singleton with start=0 immediately.
            self.flush();
            assert!(self.staged.is_none(), "one staged chunk at a time");
            assert!(self.cache_adopt.is_none(), "unconsumed cache adopt");
            self.cache_adopt = Some(hit.key_len + ids_suffix.len());
            self.cache_hits += 1;
            Ok(())
        }

        fn commit_stage(
            &mut self,
            seg: KvSegment,
            start: usize,
            len: usize,
        ) -> Result<(), ExecError> {
            let s = self.segs.get_mut(&seg.base_page).expect("commit into a live segment");
            assert_eq!(s.len(), start, "committed chunks are contiguous");
            let staged = self.staged.take().expect("a staged chunk to commit");
            assert_eq!(staged.len(), len, "staged length matches the chunk");
            s.extend(staged);
            Ok(())
        }

        fn adopt_singleton(
            &mut self,
            id: ReqId,
            seg: KvSegment,
            start: usize,
            len: usize,
        ) -> Result<(), ExecError> {
            // P3-01/016: a prefix hit staged nothing — the engine-pool
            // fiction already holds the full prompt KV; skip the staged
            // assert (nothing was staged) and take the singleton as-is.
            if let Some(kv_len) = self.cache_adopt.take() {
                assert_eq!(kv_len, start + len, "hit adopt length == prompt");
                assert_eq!(start, 0, "hit adopt starts at 0");
                self.singleton = Some((id, seg, kv_len));
                return Ok(());
            }
            let staged = self.staged.take().expect("a staged chunk to adopt");
            assert_eq!(staged.len(), len);
            if start > 0 {
                let s = self.segs.get(&seg.base_page).expect("chunked prefix lives in the segment");
                assert_eq!(s.len(), start, "adopt copies the [0, start) prefix");
            }
            self.singleton = Some((id, seg, start + len));
            Ok(())
        }

        fn drop_singleton(&mut self, id: ReqId) {
            if self.singleton.as_ref().map(|s| s.0) == Some(id) {
                self.singleton = None;
            }
        }

        fn decode_batch(&mut self, reqs: &[ExecReq]) -> Result<Vec<Vec<f32>>, ExecError> {
            if reqs.len() >= 2 {
                self.flush();
            }
            // Per-request logits: a pure function of (id, pos) — greedy
            // sampling then replays bit-identically.
            let mut out = Vec::with_capacity(reqs.len());
            for r in reqs {
                let seed = r.id.as_u64() ^ (r.pos as u64).wrapping_mul(0x2545_F491_4F6C_DD1D);
                let lg: Vec<f32> =
                    (0..self.vocab as u32).map(|v| pseudo_logit(seed, r.pos, v) as f32).collect();
                out.push(lg);
            }
            Ok(out)
        }

        fn refill_prefix(&mut self, id: ReqId, seg: KvSegment, prefix_pages: u32) {
            // 016 r2 #2: flush the singleton fiction first (mirrors the
            // real executor: B=1 KV lives in the engine pool). When the
            // terminal drop_singleton already ran, this is a no-op — the
            // fresh guarded release below then also holds (a flushed-then-
            // dropped singleton is invisible; segments must be refilled
            // with materialized KV only, so the loop's flush-on-refill
            // path guarantees it).
            if let Some((_sid, sseg, kv_len)) = self.singleton.take().filter(|s| s.0 == id) {
                debug_assert_eq!(sseg.base_page, seg.base_page);
                let s = self.segs.get_mut(&seg.base_page).expect("singleton segment live");
                assert!(s.len() <= kv_len, "flush cannot overshoot");
                s.extend(std::iter::repeat(0u32).take(kv_len - s.len()));
            }
            // Mock segments are laid out as one continuous run (no layer
            // interleave), so the prefix run is a single ref_ — the
            // refcount distribution (prefix 2→1, suffix→0) is identical to
            // the real per-layer run sequence in aggregate.
            self.frees += 1;
            assert!(self.segs.remove(&seg.base_page).is_some(), "refill of an unknown segment");
            self.pool.ref_(KvSegment { base_page: seg.base_page, n_pages: prefix_pages as usize });
            self.pool.free(seg);
        }

        fn unref_prefix(&mut self, base_page: u32, pages: u32) {
            self.pool.unref(KvSegment { base_page: base_page as usize, n_pages: pages as usize });
        }

        fn pool_stats(&self) -> KvPoolStats {
            self.pool.stats()
        }
    }

    fn cfg(vocab: usize, kv_pages: usize, max_num_seqs: usize) -> SchedLoopConfig {
        SchedLoopConfig {
            base_seed: 0,
            vocab,
            dev: 0,
            n_layer: 2,
            block_len: BLOCK_LEN,
            max_model_len: 128,
            kv_pages,
            max_num_seqs,
            chunk_size: 32,
            max_steps: 0,
            detok: Arc::new(|ids: &[u32]| {
                ids.iter().map(|t| format!("T{t}")).collect::<Vec<_>>().join("")
            }),
            prefix_cache_pages: 0, // off by default in tests; hit tests enable it
        }
    }

    fn submit(
        tx: &std_mpsc::Sender<SchedCmd>,
        ids: Vec<u32>,
        max_tokens: usize,
        params: GenParams,
    ) -> (u64, tokio::sync::mpsc::Receiver<SchedFrame>) {
        let (ftx, frx) = tokio::sync::mpsc::channel::<SchedFrame>(1024);
        let token = splitmix64(ids.len() as u64) ^ max_tokens as u64 ^ (params.seed.unwrap_or(7));
        tx.send(SchedCmd::Submit(SubmitRequest {
            ids,
            params,
            eos: Some(0xDEAD),
            max_tokens,
            stop: vec![],
            logprobs_top_n: 0,
            token,
            tx: ftx,
        }))
        .unwrap();
        (token, frx)
    }

    fn token_seq(trace: &[TraceEntry], req: ReqId) -> Vec<u32> {
        trace
            .iter()
            .filter(|e| e.req == req)
            .filter_map(|e| match e.event {
                Event::DecodeToken { token } => Some(token),
                _ => None,
            })
            .collect()
    }

    fn greedy() -> GenParams {
        GenParams {
            temperature: 0.0,
            top_k: None,
            top_p: None,
            repeat_penalty: None,
            frequency_penalty: None,
            presence_penalty: None,
            seed: Some(42),
        }
    }

    #[test]
    fn loop_replays_bit_identically() {
        // Two identical runs (same commands, same config) must produce
        // bit-identical transition traces — the SchedDeterminism contract
        // at the loop level (mock executor: no device). SubmitRequest is
        // not Clone, so the command stream is rebuilt twice.
        let build = |abort: bool| {
            let c = cfg(32, 200, 4);
            let (tx, rx) = std_mpsc::channel::<SchedCmd>();
            // The receivers stay alive until the run ends — a dropped
            // receiver aborts the request at its first frame (the
            // disconnect path), which would mask the terminal events.
            let mut frxs: Vec<tokio::sync::mpsc::Receiver<SchedFrame>> = Vec::new();
            let (_, f) = submit(&tx, (0..40).collect(), 12, greedy());
            frxs.push(f);
            let (_, f) = submit(&tx, (40..200).collect(), 8, greedy());
            frxs.push(f);
            let (tok, f) = submit(&tx, (200..240).collect(), 16, greedy());
            frxs.push(f);
            if abort {
                tx.send(SchedCmd::Abort { token: tok }).unwrap();
            }
            drop(tx);
            (c, rx, frxs)
        };
        let (c1, rx1, _f1) = build(true);
        let (a, _) = SchedLoop::new(MockExecutor::new(200, 32), c1, rx1).finish();
        let (c2, rx2, _f2) = build(true);
        let (b, _) = SchedLoop::new(MockExecutor::new(200, 32), c2, rx2).finish();
        assert_eq!(a, b, "same command stream must replay bit-identically");
        // The scenario must exercise the interesting transitions.
        assert!(a.trace.iter().any(|e| e.event == Event::PrefillStart), "prefill exercised");
        assert!(
            a.trace.iter().any(|e| matches!(e.event, Event::DecodeToken { .. })),
            "decode exercised"
        );
        assert!(
            a.trace.iter().any(|e| e.event == Event::Aborted),
            "the deterministic abort must fire"
        );
        assert!(
            a.trace.iter().any(|e| matches!(e.event, Event::MaxOutput | Event::Eos)),
            "a terminal event must fire"
        );
        // Release-once accounting: everything dispatched is returned.
        assert_eq!(a.dispatched, a.returned, "dispatch/return conservation (先分配、后释放)");
    }

    #[test]
    fn abort_isolates_other_requests() {
        // Removing the abort must not change the survivors' token
        // sequences (spec: 其余请求输出与无 abort 基线运行逐 token 一致).
        let build = |abort: bool| {
            let c = cfg(32, 200, 4);
            let (tx, rx) = std_mpsc::channel::<SchedCmd>();
            let (tok, _) = submit(&tx, (0..40).collect(), 8, greedy());
            let (_, _) = submit(&tx, (40..200).collect(), 8, greedy());
            if abort {
                tx.send(SchedCmd::Abort { token: tok }).unwrap();
            }
            drop(tx);
            (c, rx)
        };
        let (c1, rx1) = build(true);
        let (a, _) = SchedLoop::new(MockExecutor::new(200, 32), c1, rx1).finish();
        let aborted: Vec<ReqId> =
            a.trace.iter().filter(|e| e.event == Event::Aborted).map(|e| e.req).collect();
        assert!(!aborted.is_empty(), "the abort must fire");
        let (c2, rx2) = build(false);
        let (clean, _) = SchedLoop::new(MockExecutor::new(200, 32), c2, rx2).finish();
        let aborted_set: std::collections::HashSet<ReqId> = aborted.iter().copied().collect();
        let ids: Vec<ReqId> = (0..2u64).map(|s| ReqId::derive(0, s)).collect();
        for id in ids {
            if aborted_set.contains(&id) {
                continue;
            }
            assert_eq!(token_seq(&a.trace, id), token_seq(&clean.trace, id), "survivor {id}");
        }
    }

    #[test]
    fn greedy_single_request_completes_with_frames() {
        // One request, greedy: completes with a Done frame, all segments
        // released, the pool conserved (mock pool has no anchor).
        let c = cfg(16, 200, 2);
        let (tx, rx) = std_mpsc::channel::<SchedCmd>();
        let (_, mut frx) = submit(&tx, (3..20).collect(), 6, greedy());
        drop(tx);
        let (out, exec) = SchedLoop::new(MockExecutor::new(200, 16), c, rx).finish();
        // Done via max output (no EOS in the vocab stream).
        assert!(
            out.finals.iter().all(|(_, s, _, _)| *s == ReqState::Done),
            "single request done: {:?}",
            out.finals
        );
        assert_eq!(out.dispatched, out.returned);
        assert_eq!(exec.pool.in_use(), 0, "all pages released");
        assert_eq!(exec.allocs, exec.frees, "segment alloc/free symmetry");
        // Frames: 6 token frames (each "T<id>") + Done.
        let mut tokens = 0usize;
        let mut done = false;
        while let Ok(f) = frx.try_recv() {
            match f {
                SchedFrame::Token { delta, out } => {
                    tokens += 1;
                    assert!(delta.starts_with('T'));
                    assert!(out.is_none());
                }
                SchedFrame::Done {
                    stopped_by_eos,
                    stopped_by_stop: _,
                    tokens: n,
                    prompt_tokens,
                } => {
                    done = true;
                    assert!(!stopped_by_eos, "terminated by max output");
                    assert_eq!(n, 6);
                    assert_eq!(prompt_tokens, 17);
                }
                SchedFrame::Error { .. } => panic!("no error expected"),
            }
        }
        assert_eq!(tokens, 6, "six token frames");
        assert!(done, "Done frame delivered");
    }

    #[test]
    fn chunked_prefill_via_chunk_budget() {
        // A small chunk budget forces multi-chunk prefill (the chunked
        // lifecycle: ChunkDone × N → PrefillDone); alignment keeps the
        // chunks page-aligned except the final one.
        let c = SchedLoopConfig { max_model_len: 1024, chunk_size: 64, ..cfg(16, 200, 2) };
        let (tx, rx) = std_mpsc::channel::<SchedCmd>();
        let (_, mut frx) = submit(&tx, (0..200).collect(), 4, greedy()); // 200 > 64×3
        drop(tx);
        let (out, exec) = SchedLoop::new(MockExecutor::new(200, 16), c, rx).finish();
        let id = ReqId::derive(0, 0);
        // ChunkDone fires for every intermediate chunk; the final chunk
        // lands as PrefillDone (both carry the cumulative cached_len).
        let chunks: Vec<&TraceEntry> = out
            .trace
            .iter()
            .filter(|e| {
                e.req == id
                    && matches!(
                        e.event,
                        Event::ChunkDone | Event::PrefillStart | Event::PrefillDone
                    )
            })
            .collect();
        assert!(chunks.len() >= 2, "prompt must be chunked: {:?}", chunks);
        // Chunk boundaries are page-aligned (64 % 32 == 0, so alignment is
        // a no-op here — but the final chunk ends at the prompt length).
        let last = chunks.last().unwrap();
        assert!(matches!(last.event, Event::PrefillDone), "final chunk: {:?}", last);
        assert!(last.cached_len >= 200, "prefill completed");
        assert!(out.finals.iter().all(|(_, s, _, _)| *s == ReqState::Done));
        assert_eq!(exec.pool.in_use(), 0);
        // Frames: 4 token frames + Done.
        let mut frames = 0usize;
        let mut tokens = 0usize;
        while let Ok(f) = frx.try_recv() {
            frames += 1;
            match f {
                SchedFrame::Token { .. } => tokens += 1,
                SchedFrame::Done { .. } => {}
                SchedFrame::Error { .. } => panic!("no error expected"),
            }
        }
        assert_eq!(tokens, 4);
        assert_eq!(frames, 5);
    }

    #[test]
    fn stop_pattern_terminates_generation() {
        // D8: a stop pattern on the token stream fires Stopped (the matched
        // tokens are consumed, not emitted). Use a tiny vocab + greedy and
        // a pattern whose tokens the deterministic stream produces.
        let mut c = cfg(8, 200, 2);
        c.chunk_size = 32;
        let (tx, rx) = std_mpsc::channel::<SchedCmd>();
        let (ftx, frx) = tokio::sync::mpsc::channel::<SchedFrame>(1024);
        tx.send(SchedCmd::Submit(SubmitRequest {
            ids: (0..5).collect(),
            params: greedy(),
            eos: None,
            max_tokens: 64,
            stop: vec![vec![1, 2]],
            logprobs_top_n: 0,
            token: 9,
            tx: ftx,
        }))
        .unwrap();
        drop(tx);
        let mut frx = frx;
        let (out, _) = SchedLoop::new(MockExecutor::new(200, 8), c, rx).finish();
        let id = ReqId::derive(0, 0);
        assert!(
            out.trace.iter().any(|e| e.req == id && e.event == Event::Stopped),
            "stop pattern [1,2] must fire on the deterministic stream: {:?}",
            token_seq(&out.trace, id)
        );
        assert!(!out.finals.is_empty() && out.finals[0].1 == ReqState::Done);
        // The stop-matching tokens are consumed: the Done frame reports
        // fewer tokens than generated.
        while let Ok(f) = frx.try_recv() {
            if let SchedFrame::Done { tokens, .. } = f {
                assert!(tokens <= 64);
            }
        }
    }

    #[test]
    fn admission_caps_concurrency_deterministically() {
        // Three arrivals against a 2-slot pool: the third waits (Waited)
        // until a slot frees; the waiting order is the arrival order.
        let c = cfg(16, 200, 2);
        let (tx, rx) = std_mpsc::channel::<SchedCmd>();
        // The receivers stay alive until the run ends — a dropped receiver
        // aborts the request at its first frame (the disconnect path).
        let mut frxs: Vec<tokio::sync::mpsc::Receiver<SchedFrame>> = Vec::new();
        for i in 0..3u64 {
            let (ftx, frx) = tokio::sync::mpsc::channel::<SchedFrame>(1024);
            let ids: Vec<u32> = (0..32).map(|v| v as u32 + i as u32).collect();
            tx.send(SchedCmd::Submit(SubmitRequest {
                ids,
                params: greedy(),
                eos: None,
                max_tokens: 6,
                stop: vec![],
                logprobs_top_n: 0,
                token: i,
                tx: ftx,
            }))
            .unwrap();
            frxs.push(frx);
        }
        drop(tx);
        let (out, exec) = SchedLoop::new(MockExecutor::new(200, 16), c, rx).finish();
        // All three complete; the third starts after the first finishes.
        assert!(
            out.finals.iter().all(|(_, s, _, _)| *s == ReqState::Done),
            "all complete: {:?}",
            out.finals
        );
        let third_start = out
            .trace
            .iter()
            .find(|e| e.req == ReqId::derive(0, 2) && e.event == Event::PrefillStart)
            .map(|e| e.step)
            .unwrap();
        let first_done = out
            .trace
            .iter()
            .find(|e| e.req == ReqId::derive(0, 0) && e.event == Event::MaxOutput)
            .map(|e| e.step)
            .unwrap();
        assert!(third_start > first_done, "third request waits for a slot");
        assert!(out.trace.iter().any(|e| e.event == Event::Waited), "admission exercised");
        assert_eq!(exec.pool.in_use(), 0);
        assert_eq!(exec.allocs, exec.frees);
    }

    #[test]
    fn align_chunk_end_rounds_down_unless_final() {
        assert_eq!(align_chunk_end(37, 100, 32), 32);
        assert_eq!(align_chunk_end(64, 100, 32), 64);
        assert_eq!(align_chunk_end(100, 100, 32), 100, "final chunk untouched");
        assert_eq!(align_chunk_end(5, 5, 32), 5, "short prompt untouched");
        assert_eq!(align_chunk_end(130, 120, 32), 120, "end capped at the prompt");
    }

    // ------------------------------------------------------------------
    // P3-01/016: prefix-cache loop integration (mock executor)
    // ------------------------------------------------------------------

    /// 016 r2 D2/D3: one same-prompt request refills the cache; the next
    /// identical request takes the hit path (`prefill_prefix_hit` +
    /// adopt(start=0)) and complete conserving the pool (the cache keeps
    /// exactly `L` prefix pages, everything else returns).
    /// Drain a request's frames until the Done; returns the token count.
    /// The frame channel must stay alive on the sending side (a dropped
    /// receiver is the disconnect-abort path).
    fn drain_done(frx: &mut tokio::sync::mpsc::Receiver<SchedFrame>) -> usize {
        let mut n = 0;
        loop {
            match frx.blocking_recv() {
                Some(SchedFrame::Token { .. }) => n += 1,
                Some(SchedFrame::Done { .. }) => return n,
                Some(SchedFrame::Error { message }) => panic!("request errored: {message}"),
                None => panic!("channel closed before Done"),
            }
        }
    }

    /// Serve-style single-loop run: the loop lives on its own thread (one
    /// cache for its lifetime); the test thread submits and drains.
    fn loop_runner(
        c: SchedLoopConfig,
    ) -> (std_mpsc::Sender<SchedCmd>, std::thread::JoinHandle<(SchedOutcome, MockExecutor)>) {
        let (tx, rx) = std_mpsc::channel::<SchedCmd>();
        let h =
            std::thread::spawn(move || SchedLoop::new(MockExecutor::new(200, 16), c, rx).finish());
        (tx, h)
    }

    #[test]
    fn prefix_cache_hit_refills_and_reuses() {
        let mut c = cfg(16, 200, 2);
        c.prefix_cache_pages = 16; // budgets ≫ L pages
        c.chunk_size = 128; // single-chunk prefill (serve V1: chunk_size = max_model_len)
        let prompt = (0..100).collect::<Vec<u32>>(); // L = floor(100/32) = 3 blocks
        let (tx, h) = loop_runner(c);
        // Request 1 (cold): refills the cache — L=3 pages stay owned.
        let (_, mut frx1) = submit(&tx, prompt.clone(), 8, greedy());
        drain_done(&mut frx1);
        // Request 2 (warm): same prompt → the hit path.
        let (_, mut frx2) = submit(&tx, prompt.clone(), 8, greedy());
        drain_done(&mut frx2);
        drop(tx);
        let (out, exec) = h.join().unwrap();
        assert_eq!(exec.cache_hits, 1, "the second request hits the cache");
        assert_eq!(exec.pool.in_use(), 3, "cache owns exactly L pages, no leak");
        assert_eq!(out.dispatched, out.returned, "dispatch/return conservation");
        assert!(
            out.trace.iter().any(|e| matches!(e.event, Event::MaxOutput | Event::Eos)),
            "both requests terminate"
        );
        drop(frx1);
        drop(frx2);
    }

    /// 016 r2 review #3 regression: same-key releases (identical prompts
    /// after the first) must NOT accumulate references — an unbounded run
    /// of same-prompt requests keeps `in_use == L` forever.
    #[test]
    fn prefix_cache_same_key_does_not_leak() {
        let mut c = cfg(16, 200, 2);
        c.prefix_cache_pages = 16;
        c.chunk_size = 128; // single-chunk prefill (hits require one chunk)
        let prompt = (0..100).collect::<Vec<u32>>();
        let (tx, h) = loop_runner(c);
        let mut frxs = Vec::new();
        for _ in 0..5 {
            let (_, mut frx) = submit(&tx, prompt.clone(), 8, greedy());
            drain_done(&mut frx);
            frxs.push(frx);
        }
        drop(tx);
        let (_, exec) = h.join().unwrap();
        assert_eq!(exec.cache_hits, 4, "requests 2..5 all hit");
        assert_eq!(exec.pool.in_use(), 3, "steady state: L pages, no leak");
        drop(frxs);
    }

    /// 016 r2 D2: abort release must NOT refill (plain free, no cache
    /// growth) — an aborted request's segment is unreliable.
    #[test]
    fn prefix_cache_abort_does_not_refill() {
        let mut c = cfg(16, 200, 2);
        c.prefix_cache_pages = 16;
        let prompt = (0..100).collect::<Vec<u32>>();
        let (tx, h) = loop_runner(c);
        let (tok, frx) = submit(&tx, prompt, 3, greedy());
        let _frx = frx; // keep the receiver alive — the Abort command drives it
        tx.send(SchedCmd::Abort { token: tok }).unwrap();
        drop(tx);
        let (_, exec) = h.join().unwrap();
        assert_eq!(exec.cache_hits, 0);
        assert_eq!(exec.pool.in_use(), 0, "no cached pages after the abort release");
    }
}
