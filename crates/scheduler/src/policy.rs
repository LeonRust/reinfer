//! D3 queue-ordering policies.
//!
//! `SchedulePolicy` orders the waiting queue (FCFS / LPM). The resume
//! priority rule ("恢复请求>新进" — preempted requests must not be starved) is
//! baked into every ordering produced here, and into the D7 victim order
//! (preempted requests are the last to be preempted again).

use std::cmp::Reverse;
use std::collections::HashMap;

use crate::batch::{DecodingReq, WaitingReq};

/// Waiting-queue schedule policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SchedulePolicy {
    /// First-come-first-served by arrival sequence.
    #[default]
    Fcfc,
    /// Longest-prefix-match grouping: requests sharing a prefix id are
    /// scheduled together. This slice only orders — the RadixCache data
    /// structures behind real prefix matching are P3 (spec Non-Goals).
    Lpm,
}

/// Order the waiting queue for scheduling.
///
/// Resume (preempted) requests always rank before new arrivals (D3); within
/// each class the order is FCFS by arrival sequence, or prefix-grouped for
/// LPM (groups ordered by first appearance).
pub fn order_waiting(waiting: &[WaitingReq], policy: SchedulePolicy) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..waiting.len()).collect();
    match policy {
        SchedulePolicy::Fcfc => {
            idx.sort_by_key(|&i| (Reverse(waiting[i].resume), waiting[i].arrival_seq));
        }
        SchedulePolicy::Lpm => {
            // Deterministic group ranks: first-appearance order of prefix ids
            // (None is its own group). The map is only read via lookups, so
            // iteration order never affects the result.
            let mut rank: HashMap<Option<u64>, usize> = HashMap::new();
            let mut next = 0usize;
            for w in waiting {
                rank.entry(w.prefix_id).or_insert_with(|| {
                    let r = next;
                    next += 1;
                    r
                });
            }
            idx.sort_by_key(|&i| {
                (Reverse(waiting[i].resume), rank[&waiting[i].prefix_id], waiting[i].arrival_seq)
            });
        }
    }
    idx
}

/// D7 victim order: newest arrivals first; resume requests are preempted
/// last (they have already lost their blocks once).
pub fn order_victims(decoding: &[DecodingReq]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..decoding.len()).collect();
    idx.sort_by_key(|&i| (decoding[i].resume, Reverse(decoding[i].arrival_seq)));
    idx
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use crate::req::ReqId;

    fn w(id: u64, arrival: u64, resume: bool, prefix: Option<u64>) -> WaitingReq {
        WaitingReq {
            id: ReqId::derive(0, id),
            arrival_seq: arrival,
            resume,
            prefix_id: prefix,
            prompt_len: 8,
            cached_len: 0,
            max_chunk: 8,
        }
    }

    #[test]
    fn fcfs_by_arrival_with_resume_first() {
        let q = vec![w(1, 1, false, None), w(2, 2, true, None), w(3, 3, false, None)];
        let order = order_waiting(&q, SchedulePolicy::Fcfc);
        // resume request first, then arrival order
        assert_eq!(order, vec![1, 0, 2]);
    }

    #[test]
    fn lpm_groups_prefixes() {
        let q = vec![
            w(1, 1, false, Some(7)),
            w(2, 2, false, Some(9)),
            w(3, 3, false, Some(7)),
            w(4, 4, false, None),
        ];
        let order = order_waiting(&q, SchedulePolicy::Lpm);
        // group 7 (first appearance at idx 0), then group 9, then None
        assert_eq!(order, vec![0, 2, 1, 3]);
    }

    #[test]
    fn lpm_resume_still_wins_over_groups() {
        let q = vec![w(1, 1, false, Some(7)), w(2, 2, true, Some(9)), w(3, 3, false, Some(7))];
        let order = order_waiting(&q, SchedulePolicy::Lpm);
        assert_eq!(order, vec![1, 0, 2], "resume first, then prefix groups");
    }

    #[test]
    fn victims_newest_first_resume_last() {
        let d = vec![
            DecodingReq { id: ReqId::derive(0, 10), arrival_seq: 10, resume: false },
            DecodingReq { id: ReqId::derive(0, 11), arrival_seq: 11, resume: true },
            DecodingReq { id: ReqId::derive(0, 12), arrival_seq: 12, resume: false },
        ];
        let order = order_victims(&d);
        // non-resume newest first (12, 10), resume last (11)
        assert_eq!(order, vec![2, 0, 1]);
    }
}
