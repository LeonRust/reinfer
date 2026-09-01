//! Deterministic batch selection (pure logic).
//!
//! Given the waiting and decoding sets (per-request cursors and chunk
//! budgets) plus the step token budget, produce the next batch:
//!
//! - decode requests have priority (1 token each) and are all schedulable —
//!   if the budget cannot hold all of them, D7 victims are preempted
//!   (newest arrival first, resume requests last) and go back to the
//!   waiting-queue head;
//! - the remaining budget feeds prefill in policy order (D3), chunked by
//!   each request's `max_chunk` (chunk budget, D2 estimation lives in
//!   `admission`);
//! - the decode batch is emitted req_id-sorted (design report: decode batch
//!   一律按 req_id 排序), and every choice is a pure function of the inputs,
//!   so the output is bit-identical across replays.

use crate::policy::{SchedulePolicy, order_victims, order_waiting};
use crate::req::ReqId;

/// A waiting request as seen by the batch selector.
#[derive(Debug, Clone, Copy)]
pub struct WaitingReq {
    /// Request id (also the ordering key for ties).
    pub id: ReqId,
    /// Arrival sequence (FCFS order).
    pub arrival_seq: u64,
    /// Preempted earlier → resume priority (D3).
    pub resume: bool,
    /// Prompt prefix group for LPM ordering (RadixCache data is P3).
    pub prefix_id: Option<u64>,
    /// Prompt token count.
    pub prompt_len: usize,
    /// Confirmed tokens (0 for new/resumed requests — D7 zeroes on preempt).
    pub cached_len: usize,
    /// Max tokens for this step's prefill chunk (chunk budget).
    pub max_chunk: usize,
}

/// A decoding request as seen by the batch selector.
#[derive(Debug, Clone, Copy)]
pub struct DecodingReq {
    /// Request id.
    pub id: ReqId,
    /// Arrival sequence (D7 victim order).
    pub arrival_seq: u64,
    /// Preempted earlier → preempted last (D3/D7).
    pub resume: bool,
}

/// One scheduling step's output.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BatchSelection {
    /// Decode ids kept in the batch, req_id-sorted.
    pub decode: Vec<ReqId>,
    /// Prefill assignments as token slices `(id, start, end)`.
    pub prefill: Vec<(ReqId, usize, usize)>,
    /// D7 victims (newest/lowest-priority first) — preempted, back to the
    /// waiting-queue head.
    pub preempted: Vec<ReqId>,
    /// Waiting requests that could not be scheduled this step.
    pub kept_waiting: Vec<ReqId>,
}

/// Select the next batch (decode-first fill, prefill by chunk budget).
///
/// `budget` is the step token budget; `kv_floor` is the token capacity that
/// must stay free (decode safety), so the effective budget is
/// `budget - kv_floor`. Pure: no state is mutated.
pub fn select_batch(
    waiting: &[WaitingReq],
    decoding: &[DecodingReq],
    budget: usize,
    kv_floor: usize,
    policy: SchedulePolicy,
) -> BatchSelection {
    let eff = budget.saturating_sub(kv_floor);
    let mut sel = BatchSelection::default();
    // Phase 1 — decode: every decoding request needs exactly 1 token; victims
    // are preempted when the budget cannot hold all of them (D7).
    let keep = decoding.len().min(eff);
    let victims = order_victims(decoding);
    sel.preempted = victims[..decoding.len() - keep].iter().map(|&i| decoding[i].id).collect();
    let mut kept: Vec<ReqId> =
        victims[decoding.len() - keep..].iter().map(|&i| decoding[i].id).collect();
    kept.sort_unstable();
    sel.decode = kept;
    let mut remaining = eff.saturating_sub(keep);
    // Phase 2 — prefill: waiting in policy order, chunked by budget.
    for i in order_waiting(waiting, policy) {
        let w = &waiting[i];
        let need = w.prompt_len.saturating_sub(w.cached_len);
        let chunk = need.min(w.max_chunk).min(remaining);
        if chunk == 0 {
            sel.kept_waiting.push(w.id);
            continue;
        }
        sel.prefill.push((w.id, w.cached_len, w.cached_len + chunk));
        remaining -= chunk;
    }
    sel
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn rid(n: u64) -> ReqId {
        ReqId::derive(0, n)
    }

    fn w(id: u64, arrival: u64, prompt: usize, cached: usize, max_chunk: usize) -> WaitingReq {
        WaitingReq {
            id: rid(id),
            arrival_seq: arrival,
            resume: false,
            prefix_id: None,
            prompt_len: prompt,
            cached_len: cached,
            max_chunk,
        }
    }

    fn d(id: u64, arrival: u64, resume: bool) -> DecodingReq {
        DecodingReq { id: rid(id), arrival_seq: arrival, resume }
    }

    #[test]
    fn decode_first_then_prefill_by_budget() {
        let waiting = vec![w(0, 0, 40, 0, 16)];
        let decoding = vec![d(1, 1, false), d(2, 2, false), d(3, 3, false)];
        // budget 10, floor 2 → eff 8: decode 3 × 1, prefill 5
        let sel = select_batch(&waiting, &decoding, 10, 2, SchedulePolicy::Fcfc);
        // Decode batch is req_id-sorted (value order — SplitMix64 is a
        // bijection, not monotonic, so arrival order ≠ id order).
        let mut want = vec![rid(1), rid(2), rid(3)];
        want.sort_unstable();
        assert_eq!(sel.decode, want, "decode batch req_id-sorted");
        assert_eq!(sel.prefill, vec![(rid(0), 0, 5)]);
        assert!(sel.preempted.is_empty());
        assert!(sel.kept_waiting.is_empty());
    }

    #[test]
    fn kv_floor_is_reserved() {
        let waiting = vec![w(0, 0, 100, 0, 16)];
        let decoding = vec![d(1, 1, false)];
        // budget 10, floor 10 → eff 0: nothing runs
        let sel = select_batch(&waiting, &decoding, 10, 10, SchedulePolicy::Fcfc);
        assert!(sel.decode.is_empty());
        assert_eq!(sel.preempted, vec![rid(1)], "decode cannot run → preempted");
        assert!(sel.prefill.is_empty());
        assert_eq!(sel.kept_waiting, vec![rid(0)]);
    }

    #[test]
    fn preempts_newest_victims_and_keeps_resume() {
        let decoding = vec![d(10, 10, false), d(11, 11, true), d(12, 12, false)];
        // budget 2, floor 1 → eff 1: keep 1, preempt 2 (newest non-resume first)
        let sel = select_batch(&[], &decoding, 2, 1, SchedulePolicy::Fcfc);
        assert_eq!(sel.preempted, vec![rid(12), rid(10)]);
        assert_eq!(sel.decode, vec![rid(11)], "resume request kept");
    }

    #[test]
    fn prefill_chunks_by_max_chunk_and_budget() {
        // Two waiting: A prompt 40 (max_chunk 8), B prompt 10 (max_chunk 8);
        // eff 20 → A gets 8, B gets 8, remaining 4 unused
        let waiting = vec![w(0, 0, 40, 0, 8), w(1, 1, 10, 0, 8)];
        let sel = select_batch(&waiting, &[], 20, 0, SchedulePolicy::Fcfc);
        assert_eq!(sel.prefill, vec![(rid(0), 0, 8), (rid(1), 0, 8)]);
    }

    #[test]
    fn chunked_continuation_resumes_at_cached_len() {
        // Continuation: cached 12, prompt 30, max_chunk 8 → chunk [12, 20)
        let waiting = vec![w(0, 0, 30, 12, 8)];
        let sel = select_batch(&waiting, &[], 64, 0, SchedulePolicy::Fcfc);
        assert_eq!(sel.prefill, vec![(rid(0), 12, 20)]);
        // finished prompt never scheduled
        let waiting = vec![w(1, 1, 12, 12, 8)];
        let sel = select_batch(&waiting, &[], 64, 0, SchedulePolicy::Fcfc);
        assert!(sel.prefill.is_empty());
        assert_eq!(sel.kept_waiting, vec![rid(1)]);
    }

    #[test]
    fn resume_request_scheduled_before_new() {
        // B is a resume request that arrived later — must go first (D3).
        let waiting = [w(0, 0, 10, 0, 8), w(1, 5, 10, 0, 8)];
        let mut b = waiting[1];
        b.resume = true;
        let waiting = [waiting[0], b];
        let sel = select_batch(&waiting, &[], 12, 0, SchedulePolicy::Fcfc);
        assert_eq!(sel.prefill, vec![(rid(1), 0, 8), (rid(0), 0, 4)]);
    }

    #[test]
    fn lpm_orders_waiting_for_selection() {
        let mut a = w(0, 0, 10, 0, 4);
        a.prefix_id = Some(7);
        let mut b = w(1, 1, 10, 0, 4);
        b.prefix_id = Some(9);
        let mut c = w(2, 2, 10, 0, 4);
        c.prefix_id = Some(7);
        let sel = select_batch(&[a, b, c], &[], 64, 0, SchedulePolicy::Lpm);
        assert_eq!(sel.prefill, vec![(rid(0), 0, 4), (rid(2), 0, 4), (rid(1), 0, 4)]);
    }

    #[test]
    fn everything_waits_when_budget_is_zero() {
        let waiting = vec![w(0, 0, 10, 0, 8)];
        let decoding = vec![d(1, 1, false)];
        let sel = select_batch(&waiting, &decoding, 0, 0, SchedulePolicy::Fcfc);
        assert!(sel.decode.is_empty());
        assert_eq!(sel.preempted, vec![rid(1)]);
        assert!(sel.prefill.is_empty());
        assert_eq!(sel.kept_waiting, vec![rid(0)]);
    }

    #[test]
    fn empty_sets_produce_empty_batch() {
        let sel = select_batch(&[], &[], 64, 0, SchedulePolicy::Fcfc);
        assert!(sel.decode.is_empty());
        assert!(sel.prefill.is_empty());
        assert!(sel.preempted.is_empty());
        assert!(sel.kept_waiting.is_empty());
    }
}
