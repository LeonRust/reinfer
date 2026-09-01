//! Request state machine and deterministic accounting (plan D8).
//!
//! The dual cursors `cached_len` / `device_len` are the single source of
//! truth for per-request accounting:
//!
//! - `cached_len`: tokens confirmed resident in the KV cache (refcount held);
//! - `device_len`: tokens dispatched to the device (confirmed or in flight);
//!
//! Every derived quantity (current prefill chunk, generated output length) is
//! computed from the cursors — there is no independent chunk counter.
//!
//! # `ReqId` derivation — decision record
//!
//! Neither the 005 spec nor the plan pins a derivation formula: the spec only
//! requires "req_id ordering + explicit seed; deterministic input = arrival
//! order", and plan D5 uses `req_id` inside `seed_i = SplitMix64(base_seed ⊕
//! req_id)`. Per the S2-A directive ("no explicit definition → arrival order
//! plus seed-derived SplitMix64"), we derive:
//!
//! ```text
//! req_id = SplitMix64(base_seed ^ arrival_seq)
//! ```
//!
//! SplitMix64 is a bijection on `u64`, so distinct arrival sequences never
//! collide; the same (seed, arrival order) always yields the same ids, and a
//! different seed yields different ids (hence different D5 sampling streams).
//! Note that SplitMix64 does not preserve numeric order, so req_id order
//! differs from arrival order — that is harmless: the id order is itself
//! deterministic and is used as the decode-batch sort key (design report:
//! "decode batch 一律按 req_id 排序").

use crate::rng::splitmix64;
use crate::stop::{StopMatcher, StopOutcome};

/// Deterministic request id (see module doc for the derivation record).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ReqId(u64);

impl ReqId {
    /// Derive from (base seed, arrival order) — see the module doc record.
    pub fn derive(base_seed: u64, arrival_seq: u64) -> Self {
        ReqId(splitmix64(base_seed ^ arrival_seq))
    }

    /// Raw `u64` value (ordering key).
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl core::fmt::Display for ReqId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "req_{:016x}", self.0)
    }
}

/// Request lifecycle states (spec AC: Waiting→Prefill→(Chunked)→Decode→
/// Done/Aborted/Preempted).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReqState {
    /// Queued, never scheduled.
    Waiting,
    /// Single-chunk prefill: the whole prompt fits the assigned chunk.
    Prefill,
    /// Multi-chunk prefill in progress; between chunks the request stays in
    /// the waiting queue with its cursors intact.
    Chunked,
    /// Auto-regressive output generation (one token per scheduling step).
    Decode,
    /// Finished (EOS / stop / max output); resources released exactly once.
    Done,
    /// Aborted (tombstone); resources released exactly once.
    Aborted,
    /// D7 preemption marker: cursors zeroed (all blocks released; shared
    /// prefix blocks retained at the pool level), sits at the waiting-queue
    /// head with resume priority, and restarts prefill from scratch
    /// (cached = 0). Only a record — no independent resources.
    Preempted,
}

/// Outcome of a confirmation step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfirmEvent {
    /// A prefill chunk was confirmed; more prompt tokens remain.
    ChunkConfirmed,
    /// The whole prompt is confirmed; the request entered Decode.
    PrefillDone,
    /// A decode token was confirmed (id in the payload).
    DecodeConfirmed {
        /// The confirmed token id.
        token: u32,
    },
    /// A stop string matched; the request is Done (stop tokens consumed).
    Stopped,
    /// The EOS token was generated; the request is Done.
    Eos,
    /// `max_output` tokens were generated and the next token would exceed
    /// the cap; the request is Done (the exceeding token is consumed, not
    /// emitted — the cap-reaching token itself was already confirmed).
    MaxOutput,
}

/// State-machine operation errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReqError {
    /// The operation is not allowed from the current state.
    InvalidTransition {
        /// State the request was in.
        from: ReqState,
        /// Operation that was attempted.
        op: &'static str,
    },
    /// A prefill chunk of 0 tokens was requested.
    ZeroChunk,
    /// The chunk end is outside the uncomputed prompt range.
    ChunkOutOfRange,
}

/// A single request: state plus dual cursors (D8 single source of truth).
#[derive(Debug, Clone)]
pub struct Req {
    id: ReqId,
    state: ReqState,
    seed_i: u64,
    arrival_seq: u64,
    prompt_len: usize,
    max_output: usize,
    cached_len: usize,
    device_len: usize,
    stop: StopMatcher,
    eos_id: Option<u32>,
    /// D3: preempted earlier → resume priority while queued.
    resume: bool,
    /// Exactly-once release guard shared by finish/abort/eos/stop.
    released: bool,
}

impl Req {
    /// New request in Waiting state with zero cursors.
    pub fn new(
        id: ReqId,
        base_seed: u64,
        arrival_seq: u64,
        stop_patterns: Vec<Vec<u32>>,
        eos_id: Option<u32>,
    ) -> Self {
        let seed_i = splitmix64(base_seed ^ id.as_u64());
        Self {
            id,
            state: ReqState::Waiting,
            seed_i,
            arrival_seq,
            prompt_len: 0,
            max_output: 0,
            cached_len: 0,
            device_len: 0,
            stop: StopMatcher::new(stop_patterns),
            eos_id,
            resume: false,
            released: false,
        }
    }

    /// Request id.
    pub fn id(&self) -> ReqId {
        self.id
    }

    /// Current state.
    pub fn state(&self) -> ReqState {
        self.state
    }

    /// D5 per-request sampling seed: `SplitMix64(base_seed ⊕ req_id)`.
    pub fn seed_i(&self) -> u64 {
        self.seed_i
    }

    /// Arrival sequence (deterministic input order).
    pub fn arrival_seq(&self) -> u64 {
        self.arrival_seq
    }

    /// Prompt token count (set by `start_prefill`).
    pub fn prompt_len(&self) -> usize {
        self.prompt_len
    }

    /// Max generated output tokens (set by `start_prefill`).
    pub fn max_output(&self) -> usize {
        self.max_output
    }

    /// Confirmed tokens resident in the KV cache.
    pub fn cached_len(&self) -> usize {
        self.cached_len
    }

    /// Tokens dispatched to the device (confirmed or in flight).
    pub fn device_len(&self) -> usize {
        self.device_len
    }

    /// D3 resume flag (preempted request waiting to be re-scheduled).
    pub fn resume(&self) -> bool {
        self.resume
    }

    /// Confirmed generated output tokens (`cached_len - prompt_len`, ≥ 0).
    pub fn output_tokens(&self) -> usize {
        self.cached_len.saturating_sub(self.prompt_len)
    }

    /// Whether the request is in a prefill state (Prefill or Chunked).
    pub fn is_prefilling(&self) -> bool {
        matches!(self.state, ReqState::Prefill | ReqState::Chunked)
    }

    /// Start (or resume after preemption, D7) prefill with the assigned chunk.
    ///
    /// `chunk_tokens` is the token budget the batch selector committed to:
    /// the whole prompt fits → `Prefill`, otherwise `Chunked`. Cursors are
    /// zeroed first (resume restarts from scratch, `cached = 0`).
    pub fn start_prefill(
        &mut self,
        prompt_len: usize,
        max_output: usize,
        chunk_tokens: usize,
    ) -> Result<(), ReqError> {
        if !matches!(self.state, ReqState::Waiting | ReqState::Preempted) {
            return Err(ReqError::InvalidTransition { from: self.state, op: "start_prefill" });
        }
        if chunk_tokens == 0 {
            return Err(ReqError::ZeroChunk);
        }
        self.prompt_len = prompt_len;
        self.max_output = max_output;
        self.cached_len = 0;
        self.device_len = chunk_tokens.min(prompt_len);
        self.state =
            if self.device_len >= prompt_len { ReqState::Prefill } else { ReqState::Chunked };
        self.resume = false;
        self.assert_cursors_valid();
        Ok(())
    }

    /// Dispatch the next prefill chunk of a `Chunked` request (the batch
    /// selector decides the end from the current cursors and its budget).
    pub fn dispatch_chunk(&mut self, chunk_end: usize) -> Result<(), ReqError> {
        if self.state != ReqState::Chunked {
            return Err(ReqError::InvalidTransition { from: self.state, op: "dispatch_chunk" });
        }
        if chunk_end <= self.cached_len || chunk_end > self.prompt_len {
            return Err(ReqError::ChunkOutOfRange);
        }
        self.device_len = chunk_end;
        self.assert_cursors_valid();
        Ok(())
    }

    /// Confirm the in-flight prefill chunk (`cached = device`).
    ///
    /// When the whole prompt is confirmed the request enters `Decode`
    /// (`PrefillDone`); otherwise it stays in `Chunked` and the next chunk is
    /// dispatched by the selector in a later step.
    pub fn confirm(&mut self) -> ConfirmEvent {
        debug_assert!(
            matches!(self.state, ReqState::Prefill | ReqState::Chunked),
            "confirm() outside prefill: {:?}",
            self.state
        );
        self.cached_len = self.device_len;
        if self.cached_len >= self.prompt_len {
            self.state = ReqState::Decode;
            ConfirmEvent::PrefillDone
        } else {
            ConfirmEvent::ChunkConfirmed
        }
    }

    /// Generate-and-confirm one decode token.
    ///
    /// Dispatches one token (device = cached + 1), then checks stop (checked
    /// first, vLLM order) → EOS → max output. On a terminal outcome the
    /// request becomes `Done` and arms the exactly-once release guard. The
    /// token that matched stop/EOS is consumed, not emitted.
    ///
    /// Max-output boundary: the cap-reaching token (k = max_output) IS
    /// confirmed and emitted — exactly `max_output` tokens are generated for
    /// `max_output = n`, matching the serial pipeline and llama.cpp `-n`
    /// semantics. `MaxOutput` fires on the next call, whose token would
    /// exceed the cap (device > prompt + max_output); that token is consumed
    /// like stop/EOS (dispatched, not cached, not emitted).
    pub fn decode_step(&mut self, token: u32) -> ConfirmEvent {
        debug_assert_eq!(self.state, ReqState::Decode, "decode_step outside Decode");
        self.device_len = self.cached_len + 1;
        match self.stop.push(token) {
            StopOutcome::Match { .. } => {
                self.set_done();
                return ConfirmEvent::Stopped;
            }
            StopOutcome::Continue => {}
        }
        if self.eos_id == Some(token) {
            self.set_done();
            return ConfirmEvent::Eos;
        }
        if self.device_len > self.prompt_len + self.max_output {
            self.set_done();
            return ConfirmEvent::MaxOutput;
        }
        self.cached_len = self.device_len;
        ConfirmEvent::DecodeConfirmed { token }
    }

    /// Decode → Done. Idempotent no-op from any other state (false = nothing
    /// transitioned; the release guard is consumed via `take_release`).
    pub fn finish(&mut self) -> bool {
        if self.state != ReqState::Decode {
            return false;
        }
        self.set_done();
        true
    }

    /// Abort from any non-terminal state; idempotent no-op after Done/Aborted
    /// (spec AC: abort-after-done 幂等). Returns true iff this call created
    /// the tombstone (i.e. armed the release guard — resources are released
    /// exactly once, shared with the finish path).
    pub fn abort(&mut self) -> bool {
        match self.state {
            ReqState::Done | ReqState::Aborted => false,
            _ => {
                self.state = ReqState::Aborted;
                self.arm_release()
            }
        }
    }

    /// D7 preemption: allowed from any active state (Prefill/Chunked/Decode).
    /// Zeroes both cursors (all blocks released at the pool level; shared
    /// prefix blocks are retained there), records the Preempted marker and
    /// resume priority. The caller re-queues the request at the waiting-queue
    /// head; on rescheduling, `start_prefill` restarts from scratch
    /// (cached = 0).
    pub fn preempt(&mut self) -> Result<(), ReqError> {
        match self.state {
            ReqState::Prefill | ReqState::Chunked | ReqState::Decode => {}
            s => return Err(ReqError::InvalidTransition { from: s, op: "preempt" }),
        }
        self.state = ReqState::Preempted;
        self.cached_len = 0;
        self.device_len = 0;
        self.resume = true;
        self.stop.reset();
        Ok(())
    }

    /// Next prefill chunk of prompt tokens to compute, derived from the
    /// cursors (D8): `[cached_len, min(device_len, cached_len + budget,
    /// prompt_len))`. None when nothing is dispatchable (not prefilling, or
    /// `device_len == cached_len` between steps).
    pub fn current_chunk(&self, chunk_budget: usize) -> Option<(usize, usize)> {
        if !self.is_prefilling() {
            return None;
        }
        let start = self.cached_len;
        let end = self.device_len.min(self.prompt_len).min(start.saturating_add(chunk_budget));
        (start < end).then_some((start, end))
    }

    /// Consume the exactly-once release guard (shared by finish/abort/
    /// eos/stop): true on the first call only, false afterwards.
    pub fn take_release(&mut self) -> bool {
        let r = self.released;
        self.released = false;
        r
    }

    /// Cursor invariant: `cached ≤ device` always, and `device` never
    /// exceeds the prompt during prefill, nor prompt + max output in decode
    /// (the terminal step that would exceed the cap transitions to Done
    /// before this check is evaluated).
    pub fn assert_cursors_valid(&self) {
        debug_assert!(self.cached_len <= self.device_len, "cached > device");
        match self.state {
            ReqState::Prefill | ReqState::Chunked => {
                debug_assert!(self.device_len <= self.prompt_len, "prefill past prompt");
            }
            ReqState::Decode => {
                debug_assert!(
                    self.device_len <= self.prompt_len + self.max_output,
                    "decode past max output"
                );
            }
            _ => {}
        }
    }

    fn set_done(&mut self) {
        self.state = ReqState::Done;
        self.arm_release();
    }

    fn arm_release(&mut self) -> bool {
        let fresh = !self.released;
        self.released = true;
        fresh
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn rid(n: u64) -> ReqId {
        ReqId::derive(0, n)
    }

    fn new_req(n: u64) -> Req {
        Req::new(rid(n), 0, n, vec![], None)
    }

    #[test]
    fn req_id_derivation_is_deterministic_and_distinct() {
        // Same (seed, arrival order) → same id; different seed → different id.
        assert_eq!(ReqId::derive(42, 3), ReqId::derive(42, 3));
        assert_ne!(ReqId::derive(42, 3), ReqId::derive(42, 4));
        assert_ne!(ReqId::derive(42, 3), ReqId::derive(43, 3));
        // Bijection: no collisions over a prefix of arrival sequences.
        let mut seen = std::collections::HashSet::new();
        for seq in 0..256u64 {
            assert!(seen.insert(ReqId::derive(7, seq)));
        }
    }

    #[test]
    fn start_prefill_single_chunk_enters_prefill() {
        let mut r = new_req(0);
        assert_eq!(r.state(), ReqState::Waiting);
        r.start_prefill(10, 5, 16).unwrap();
        assert_eq!(r.state(), ReqState::Prefill);
        assert_eq!((r.cached_len(), r.device_len()), (0, 10));
        assert_eq!(r.current_chunk(16), Some((0, 10)));
    }

    #[test]
    fn start_prefill_chunked_enters_chunked_with_cursors() {
        let mut r = new_req(1);
        r.start_prefill(20, 8, 6).unwrap();
        assert_eq!(r.state(), ReqState::Chunked);
        assert_eq!((r.cached_len(), r.device_len()), (0, 6));
        assert_eq!(r.current_chunk(6), Some((0, 6)));
    }

    #[test]
    fn chunk_cycle_confirms_and_advances() {
        let mut r = new_req(1);
        r.start_prefill(10, 8, 4).unwrap();
        // chunk 1
        assert_eq!(r.confirm(), ConfirmEvent::ChunkConfirmed);
        assert_eq!((r.cached_len(), r.device_len()), (4, 4));
        assert_eq!(r.current_chunk(4), None, "nothing dispatched yet");
        // chunk 2
        r.dispatch_chunk(8).unwrap();
        assert_eq!(r.current_chunk(4), Some((4, 8)));
        assert_eq!(r.confirm(), ConfirmEvent::ChunkConfirmed);
        assert_eq!((r.cached_len(), r.device_len()), (8, 8));
        // final chunk
        r.dispatch_chunk(10).unwrap();
        assert_eq!(r.confirm(), ConfirmEvent::PrefillDone);
        assert_eq!(r.state(), ReqState::Decode);
        assert_eq!((r.cached_len(), r.device_len()), (10, 10));
        assert_eq!(r.current_chunk(4), None, "prefill finished");
        assert_eq!(r.output_tokens(), 0);
    }

    #[test]
    fn prefill_done_keeps_cursors_settled() {
        let mut r = new_req(2);
        r.start_prefill(5, 4, 5).unwrap();
        assert_eq!(r.confirm(), ConfirmEvent::PrefillDone);
        assert_eq!((r.cached_len(), r.device_len()), (5, 5), "settled between steps");
    }

    #[test]
    fn decode_token_and_max_output() {
        let mut r = new_req(3);
        r.start_prefill(3, 2, 3).unwrap();
        assert_eq!(r.confirm(), ConfirmEvent::PrefillDone);
        assert_eq!(r.decode_step(11), ConfirmEvent::DecodeConfirmed { token: 11 });
        assert_eq!((r.cached_len(), r.device_len()), (4, 4));
        assert_eq!(r.output_tokens(), 1);
        // The cap-reaching token (k = max_output) IS confirmed and emitted.
        assert_eq!(r.decode_step(12), ConfirmEvent::DecodeConfirmed { token: 12 });
        assert_eq!((r.cached_len(), r.device_len()), (5, 5));
        assert_eq!(r.output_tokens(), 2);
        assert_eq!(r.state(), ReqState::Decode);
        // The next token would exceed the cap → MaxOutput (consumed, not
        // emitted: dispatched to device, not cached).
        assert_eq!(r.decode_step(13), ConfirmEvent::MaxOutput);
        assert_eq!(r.state(), ReqState::Done);
        assert_eq!((r.cached_len(), r.device_len()), (5, 6));
        assert!(r.take_release(), "release armed on first terminal");
        assert!(!r.take_release(), "release guard consumed exactly once");
    }

    #[test]
    fn decode_eos_terminates() {
        let mut r = Req::new(rid(4), 0, 4, vec![], Some(9));
        r.start_prefill(2, 8, 2).unwrap();
        assert_eq!(r.confirm(), ConfirmEvent::PrefillDone);
        assert_eq!(r.decode_step(1), ConfirmEvent::DecodeConfirmed { token: 1 });
        assert_eq!(r.decode_step(9), ConfirmEvent::Eos);
        assert_eq!(r.state(), ReqState::Done);
    }

    #[test]
    fn decode_stop_string_terminates_first() {
        // Stop string takes priority over EOS (vLLM check order).
        let mut r = Req::new(rid(5), 0, 5, vec![vec![3, 3]], Some(9));
        r.start_prefill(2, 8, 2).unwrap();
        assert_eq!(r.confirm(), ConfirmEvent::PrefillDone);
        assert_eq!(r.decode_step(3), ConfirmEvent::DecodeConfirmed { token: 3 });
        // The token that completes the stop (3) is consumed, not emitted.
        assert_eq!(r.decode_step(3), ConfirmEvent::Stopped);
        assert_eq!(r.state(), ReqState::Done);
    }

    #[test]
    fn abort_from_every_state_and_idempotence() {
        // Waiting
        let mut r = new_req(6);
        assert!(r.abort());
        assert_eq!(r.state(), ReqState::Aborted);
        assert!(!r.abort(), "abort-after-abort is a no-op");
        // Abort from any live state arms the exactly-once release guard
        // (uniform rule: every terminal request owes exactly one release).
        assert!(r.take_release(), "abort arms the release guard");
        assert!(!r.take_release(), "guard consumed exactly once");
        // Prefill
        let mut r = new_req(7);
        r.start_prefill(4, 4, 4).unwrap();
        assert!(r.abort());
        assert_eq!(r.state(), ReqState::Aborted);
        // Chunked
        let mut r = new_req(8);
        r.start_prefill(10, 4, 4).unwrap();
        assert!(r.abort());
        // Decode
        let mut r = new_req(9);
        r.start_prefill(4, 4, 4).unwrap();
        r.confirm();
        assert!(r.abort());
        assert_eq!(r.state(), ReqState::Aborted);
        // Preempted
        let mut r = new_req(10);
        r.start_prefill(4, 4, 4).unwrap();
        r.confirm();
        r.preempt().unwrap();
        assert_eq!(r.state(), ReqState::Preempted);
        assert!(r.abort());
        assert_eq!(r.state(), ReqState::Aborted);
        // abort-after-done is idempotent and does not double-release
        let mut r = new_req(11);
        r.start_prefill(2, 1, 2).unwrap();
        r.confirm();
        // max_output = 1: the first token is confirmed, the second (which
        // would exceed the cap) fires MaxOutput.
        assert_eq!(r.decode_step(5), ConfirmEvent::DecodeConfirmed { token: 5 });
        assert_eq!(r.decode_step(6), ConfirmEvent::MaxOutput);
        assert_eq!(r.state(), ReqState::Done);
        assert!(r.take_release());
        assert!(!r.abort(), "abort-after-done no-op");
        assert!(!r.take_release(), "no second release");
    }

    #[test]
    fn preempt_zeroes_cursors_and_restarts_from_scratch() {
        let mut r = new_req(12);
        r.start_prefill(6, 4, 6).unwrap();
        r.confirm();
        assert_eq!(r.decode_step(1), ConfirmEvent::DecodeConfirmed { token: 1 });
        r.preempt().unwrap();
        assert_eq!(r.state(), ReqState::Preempted);
        assert_eq!((r.cached_len(), r.device_len()), (0, 0), "D7: all blocks released");
        assert!(r.resume(), "resume priority set");
        // Resume restarts prefill from scratch (cached = 0).
        r.start_prefill(6, 4, 6).unwrap();
        assert_eq!(r.state(), ReqState::Prefill);
        assert_eq!(r.cached_len(), 0);
        assert!(!r.resume(), "priority consumed at scheduling");
        r.confirm();
        assert_eq!(
            r.decode_step(1),
            ConfirmEvent::DecodeConfirmed { token: 1 },
            "token stream restarts deterministically at the same position"
        );
    }

    #[test]
    fn preempt_rejected_in_wrong_states() {
        let mut r = new_req(13);
        assert!(matches!(
            r.preempt(),
            Err(ReqError::InvalidTransition { from: ReqState::Waiting, .. })
        ));
        r.start_prefill(4, 4, 4).unwrap();
        r.confirm();
        assert_eq!(r.decode_step(1), ConfirmEvent::DecodeConfirmed { token: 1 });
        r.preempt().unwrap();
        assert!(matches!(
            r.preempt(),
            Err(ReqError::InvalidTransition { from: ReqState::Preempted, .. })
        ));
    }

    #[test]
    fn invalid_transitions_and_zero_chunk() {
        let mut r = new_req(14);
        assert_eq!(r.start_prefill(4, 4, 0), Err(ReqError::ZeroChunk));
        assert_eq!(r.start_prefill(4, 4, 4).err(), None);
        assert!(matches!(
            r.dispatch_chunk(4),
            Err(ReqError::InvalidTransition { from: ReqState::Prefill, .. })
        ));
        let mut r = new_req(15);
        r.start_prefill(10, 4, 4).unwrap();
        // Re-dispatching the in-flight chunk is a no-op (cursors unchanged),
        // not an error — the invalid range is (cached_len, prompt_len].
        assert_eq!(r.dispatch_chunk(4), Ok(()), "in-flight chunk is re-dispatchable");
        assert_eq!(r.dispatch_chunk(11), Err(ReqError::ChunkOutOfRange), "past prompt");
        r.confirm(); // cached = 4, still Chunked
        assert_eq!(
            r.dispatch_chunk(4),
            Err(ReqError::ChunkOutOfRange),
            "no progress: end <= cached"
        );
        assert!(!r.finish(), "finish outside Decode is a no-op");
    }

    #[test]
    fn current_chunk_derives_from_cursors_only() {
        let mut r = new_req(16);
        assert_eq!(r.current_chunk(8), None, "waiting");
        r.start_prefill(12, 4, 4).unwrap();
        assert_eq!(r.current_chunk(4), Some((0, 4)), "budget caps the chunk");
        assert_eq!(r.current_chunk(64), Some((0, 4)), "device cap dominates");
        r.confirm();
        assert_eq!(r.current_chunk(4), None);
        r.dispatch_chunk(8).unwrap();
        assert_eq!(r.current_chunk(4), Some((4, 8)));
        assert_eq!(r.current_chunk(2), Some((4, 6)), "budget smaller than dispatched");
        r.dispatch_chunk(12).unwrap();
        r.confirm();
        assert_eq!(r.state(), ReqState::Decode);
        assert_eq!(r.current_chunk(4), None, "no prefill chunk in Decode");
    }

    #[test]
    fn output_tokens_derived_from_cursor() {
        let mut r = new_req(17);
        r.start_prefill(3, 4, 3).unwrap();
        assert_eq!(r.output_tokens(), 0);
        r.confirm();
        assert_eq!(r.decode_step(1), ConfirmEvent::DecodeConfirmed { token: 1 });
        assert_eq!(r.output_tokens(), 1);
    }
}
