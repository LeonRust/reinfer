//! reinfer-scheduler — the pure-logic scheduling core (spec 005, S2 slice).
//!
//! This crate is the deterministic decision layer of the serving engine. It
//! holds no GPU state and drives no inference; it decides. All state lives in
//! per-request cursors, all decisions are pure functions of the inputs, so a
//! given arrival sequence + seed + generation policy replays bit-identically
//! (`replay::SchedDeterminism`).
//!
//! Modules:
//!
//! - [`req`] — the `ReqState` machine (Waiting → Prefill → Chunked → Decode →
//!   Done/Aborted/Preempted) with the dual-cursor accounting (cached_len /
//!   device_len) that is the only ledger the scheduler keeps (design report
//!   D8). `ReqId` is deterministically derived from the base seed and the
//!   arrival sequence (SplitMix64).
//! - [`admission`] — the lightllm admission estimate (design report D2):
//!   per-request (a, b) token estimates, the busy heuristic, the running
//!   EMA, the peak-footprint formula and the pages/token budget gates.
//! - [`batch`] — `select_batch`: decode-first fill, D7 victim preemption
//!   (newest arrival first, resume requests protected), prefill chunked by
//!   per-request chunk budget in policy order.
//! - [`policy`] — waiting-queue ordering (FCFS / LPM, D3) and the D7 victim
//!   order; resume (preempted) requests always outrank new arrivals.
//! - [`radix`] — the token-prefix KV cache front-end (spec 016 P3-01):
//!   page-aligned trie lookups, budget LRU eviction, pure CPU (no CUDA).
//! - [`stop`] — incremental stop-pattern matching over the generated token
//!   stream (longest-prefix partial match state, spec D8).
//! - [`rng`] — the deterministic per-request RNG (design report D5):
//!   `rng(seed_i, pos, vocab)` via SplitMix64 + Lemire bounded sampling.
//! - [`replay`] — the end-to-end step loop that ties everything together:
//!   arrivals, admission, batch selection, chunk dispatch, decode, stop/EOS/
//!   max-output, abort, preempt, and the 2× bit-identical replay contract.
//!
//! Scope: pure logic only. RadixCache prefix data is P3 (spec 005 Non-Goals);
//! the memory allocator lives in `reinfer-memory` and is consumed read-only
//! at the engine boundary, never here.

pub mod admission;
pub mod batch;
pub mod policy;
pub mod radix;
pub mod replay;
pub mod req;
pub mod rng;
pub mod stop;
