//! Determinism harness (S2-A): replay a synthetic arrival stream through the
//! full pure-logic scheduler pipeline — arrival → admission (D2) → batch
//! selection (decode-first, D7 preemption) → state machine (D8) → stop/EOS/
//! max-output — with a deterministic generation policy, and record a trace
//! that must be bit-identical across replays of the same (seed, arrival
//! order). This is the `SchedDeterminism` contract: same input rerun 2× →
//! identical transition sequences and token cursors.

use std::collections::{BTreeMap, HashMap, VecDeque};

use crate::admission::{
    AdmissionConfig, AdmissionVerdict, EmaTracker, EstimateInput, RequestEstimate, check_admission,
    estimate, is_busy,
};
use crate::batch::{DecodingReq, WaitingReq, select_batch};
use crate::policy::SchedulePolicy;
use crate::req::{ConfirmEvent, Req, ReqId, ReqState};
use crate::rng::{GOLDEN_RATIO_64, rng_usize, splitmix64};

/// Deterministic token generation policy (no real model in this slice).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenPolicy {
    /// argmax over seed-derived pseudo-logits (no RNG draws, like greedy).
    Greedy,
    /// D5 RNG draw: `token = rng_usize(seed_i, pos, vocab, vocab)`.
    Sample,
}

/// Deterministic scheduler configuration (the seed is part of the input).
#[derive(Debug, Clone)]
pub struct SchedConfig {
    /// Base seed (CLI `--seed` / `REINFER_SEED`).
    pub base_seed: u64,
    /// Sampling vocabulary size.
    pub vocab: u32,
    /// Optional EOS token id.
    pub eos_id: Option<u32>,
    /// Token generation policy.
    pub gen_policy: GenPolicy,
    /// Per-step token budget handed to the batch selector.
    pub step_budget: usize,
    /// Token floor that must stay free (decode safety).
    pub kv_floor: usize,
    /// Max tokens per chunked-prefill chunk.
    pub chunk_size: usize,
    /// Waiting-queue ordering policy (D3).
    pub schedule: SchedulePolicy,
    /// Hard step cap (termination guarantee for tests).
    pub max_steps: usize,
    /// KV token capacity (busy ratio denominator; D2).
    pub max_total_tokens: u64,
    /// D2 admission gates.
    pub admission: AdmissionConfig,
}

/// One synthetic request. `at_step` must be non-decreasing across the slice
/// (the arrival list order is part of the deterministic input).
#[derive(Debug, Clone)]
pub struct Arrival {
    /// Step at which the request arrives at the scheduler.
    pub at_step: usize,
    /// Prompt token count.
    pub prompt_len: usize,
    /// `max_new_tokens`.
    pub max_output: usize,
    /// Chunked prefill (multi-chunk prompt processing).
    pub chunked: bool,
    /// Stop token patterns (matched on the token stream, D8).
    pub stop_patterns: Vec<Vec<u32>>,
    /// Deterministic abort injection: step at which the request is aborted
    /// (no-op if already terminal).
    pub abort_at_step: Option<usize>,
    /// Deterministic preempt injection: step at which the request is
    /// preempted (only when it is decoding).
    pub preempt_at_step: Option<usize>,
}

/// One recorded state-machine transition with the resulting cursors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceEntry {
    /// Global step.
    pub step: usize,
    /// Request id.
    pub req: ReqId,
    /// Transition.
    pub event: Event,
    /// `cached_len` after the transition.
    pub cached_len: usize,
    /// `device_len` after the transition.
    pub device_len: usize,
}

/// Transition events recorded in the trace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    /// Request enqueued.
    Arrived,
    /// Request passed admission and is eligible this step.
    Admitted,
    /// Request denied by admission this step (stays in the waiting queue).
    Waited,
    /// Prefill started (first chunk dispatched).
    PrefillStart,
    /// A prefill chunk was confirmed.
    ChunkDone,
    /// The whole prompt was confirmed; the request entered Decode.
    PrefillDone,
    /// A decode token was generated and confirmed (payload = token id).
    DecodeToken {
        /// The generated token id.
        token: u32,
    },
    /// A stop string matched; Done.
    Stopped,
    /// EOS generated; Done.
    Eos,
    /// `max_output` reached; Done.
    MaxOutput,
    /// Aborted (tombstone).
    Aborted,
    /// Budget-driven preemption (D7 victim).
    Preempted,
    /// Explicitly injected preemption.
    PreemptInjected,
}

/// Full replay outcome; derives `PartialEq` so two replays can be compared
/// bit-identically.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayOutcome {
    /// The transition trace (events in deterministic order).
    pub trace: Vec<TraceEntry>,
    /// Final `(id, state, cached_len, device_len)` per request, req_id-sorted.
    pub finals: Vec<(ReqId, ReqState, usize, usize)>,
    /// High-water mark of the sum of `device_len` over live requests
    /// (allocated-token accounting proxy).
    pub high_water: usize,
    /// Steps executed.
    pub steps: usize,
    /// Total tokens dispatched ("先分配").
    pub total_dispatched: u64,
    /// Total tokens returned (preempt releases + terminal releases, "后释放").
    pub total_returned: u64,
}

/// SchedDeterminism: replay driver over a fixed configuration.
#[derive(Debug, Clone)]
pub struct SchedDeterminism {
    cfg: SchedConfig,
}

impl SchedDeterminism {
    /// New driver with the given configuration.
    pub fn new(cfg: SchedConfig) -> Self {
        debug_assert!(cfg.vocab > 0, "vocab must be positive");
        Self { cfg }
    }

    /// Configuration.
    pub fn config(&self) -> &SchedConfig {
        &self.cfg
    }

    /// Replay the arrival stream through the pure-logic pipeline.
    pub fn replay(&self, arrivals: &[Arrival]) -> ReplayOutcome {
        replay(&self.cfg, arrivals)
    }

    /// Replay twice and assert the outcomes are bit-identical (panics on
    /// mismatch). Returns the first outcome.
    pub fn assert_bit_identical(&self, arrivals: &[Arrival]) -> ReplayOutcome {
        let a = replay(&self.cfg, arrivals);
        let b = replay(&self.cfg, arrivals);
        assert_eq!(a, b, "same (seed, arrival order) must replay bit-identically");
        a
    }
}

/// Replay the arrival stream with the given configuration.
pub fn replay(cfg: &SchedConfig, arrivals: &[Arrival]) -> ReplayOutcome {
    let mut r = Replay::new(cfg, arrivals);
    // `is_finished()` is vacuously true before the first arrival, so keep
    // running while arrivals remain or any request is still live.
    while r.step < cfg.max_steps && (r.arrival_idx < r.arrivals.len() || !r.is_finished()) {
        r.arrive();
        r.abort_injected();
        r.preempt_injected();
        r.schedule_step();
        r.decode_step();
        r.record_high_water();
        r.step += 1;
    }
    let finals: Vec<(ReqId, ReqState, usize, usize)> = r
        .reqs
        .iter()
        .map(|(&id, req)| (id, req.state(), req.cached_len(), req.device_len()))
        .collect();
    ReplayOutcome {
        trace: r.trace,
        finals,
        high_water: r.high_water,
        steps: r.step,
        total_dispatched: r.dispatched,
        total_returned: r.returned,
    }
}

/// Per-request metadata held by the harness (harness concerns only — the Req
/// state machine itself only needs the cursors).
#[derive(Debug, Clone)]
struct Meta {
    prompt_len: usize,
    max_output: usize,
    chunked: bool,
    abort_at: Option<usize>,
    preempt_at: Option<usize>,
}

struct Replay<'a> {
    cfg: &'a SchedConfig,
    arrivals: &'a [Arrival],
    reqs: BTreeMap<ReqId, Req>,
    meta: HashMap<ReqId, Meta>,
    /// Waiting queue (preempted requests go to the head).
    waiting: VecDeque<ReqId>,
    /// Decoding requests (regenerated deterministically each step).
    decoding: Vec<ReqId>,
    /// Live request estimates (D2 working set).
    estimates: HashMap<ReqId, RequestEstimate>,
    /// Output-length EMA (D2), updated on request completion.
    ema: EmaTracker,
    arrival_idx: usize,
    arrival_seq: u64,
    step_first_chunk_sum: u64,
    step: usize,
    trace: Vec<TraceEntry>,
    dispatched: u64,
    returned: u64,
    high_water: usize,
}

impl<'a> Replay<'a> {
    fn new(cfg: &'a SchedConfig, arrivals: &'a [Arrival]) -> Self {
        debug_assert!(
            arrivals.windows(2).all(|w| w[0].at_step <= w[1].at_step),
            "arrival list must be ordered by at_step (order is part of the input)"
        );
        Self {
            cfg,
            arrivals,
            reqs: BTreeMap::new(),
            meta: HashMap::new(),
            waiting: VecDeque::new(),
            decoding: Vec::new(),
            estimates: HashMap::new(),
            ema: EmaTracker::new(),
            arrival_idx: 0,
            arrival_seq: 0,
            step_first_chunk_sum: 0,
            step: 0,
            trace: Vec::new(),
            dispatched: 0,
            returned: 0,
            high_water: 0,
        }
    }

    fn is_finished(&self) -> bool {
        self.reqs.values().all(|r| matches!(r.state(), ReqState::Done | ReqState::Aborted))
    }

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
        is_busy(
            self.kv_usage_tokens(),
            self.cfg.max_total_tokens,
            self.cfg.admission.router_token_ratio,
        )
    }

    fn current_estimate(&self, id: ReqId) -> RequestEstimate {
        let r = &self.reqs[&id];
        let m = &self.meta[&id];
        let chunked_remaining =
            if m.chunked { r.prompt_len().saturating_sub(r.cached_len()) as u64 } else { 0 };
        estimate(EstimateInput {
            input_len: m.prompt_len as u64,
            max_new_tokens: m.max_output as u64,
            has_out_len: r.output_tokens() as u64,
            shm_kv_len: 0,
            chunked_remaining,
            chunk_size: self.cfg.chunk_size as u64,
            busy: self.is_busy(),
            ema_req_out_len: self.ema.ema().round() as u64,
            max_waiting_tokens: self.cfg.admission.max_waiting_tokens,
        })
    }

    /// Enqueue the arrivals scheduled for this step.
    fn arrive(&mut self) {
        while self.arrival_idx < self.arrivals.len()
            && self.arrivals[self.arrival_idx].at_step <= self.step
        {
            let a = &self.arrivals[self.arrival_idx];
            let id = ReqId::derive(self.cfg.base_seed, self.arrival_seq);
            let req = Req::new(
                id,
                self.cfg.base_seed,
                self.arrival_seq,
                a.stop_patterns.clone(),
                self.cfg.eos_id,
            );
            self.reqs.insert(id, req);
            self.trace(id, Event::Arrived);
            self.meta.insert(
                id,
                Meta {
                    prompt_len: a.prompt_len,
                    max_output: a.max_output,
                    chunked: a.chunked,
                    abort_at: a.abort_at_step,
                    preempt_at: a.preempt_at_step,
                },
            );
            self.estimates.insert(id, self.current_estimate(id));
            self.waiting.push_back(id);
            self.arrival_idx += 1;
            self.arrival_seq += 1;
        }
    }

    /// Apply deterministic abort injections (tombstone semantics, D8).
    fn abort_injected(&mut self) {
        let ids: Vec<ReqId> = self.reqs.keys().copied().collect();
        for id in ids {
            if self.meta[&id].abort_at == Some(self.step) {
                let live = !matches!(self.reqs[&id].state(), ReqState::Done | ReqState::Aborted);
                if live {
                    let dev = self.reqs[&id].device_len();
                    self.reqs.get_mut(&id).expect("live req").abort();
                    self.returned += dev as u64;
                    self.estimates.remove(&id);
                    self.remove_from_waiting(id);
                    self.trace(id, Event::Aborted);
                }
            }
        }
    }

    /// Apply deterministic preempt injections (only while decoding).
    fn preempt_injected(&mut self) {
        let ids: Vec<ReqId> = self.decoding.clone();
        for id in ids {
            if self.meta[&id].preempt_at == Some(self.step) {
                if self.reqs[&id].state() != ReqState::Decode {
                    continue;
                }
                let dev = self.reqs[&id].device_len();
                self.reqs.get_mut(&id).expect("decode req").preempt().expect("decoding");
                self.returned += dev as u64;
                self.trace(id, Event::PreemptInjected);
                self.remove_from_decoding(id);
                self.waiting.push_front(id);
            }
        }
    }

    /// Admission + batch selection + dispatch/confirm for this step.
    fn schedule_step(&mut self) {
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
                Some(self.cfg.chunk_size.min(self.meta[&id].prompt_len) as u64)
            } else {
                None
            };
            match check_admission(
                &working,
                first_chunk,
                self.step_first_chunk_sum,
                self.is_busy(),
                &self.cfg.admission,
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
                        prompt_len: r.prompt_len().max(self.meta[&id].prompt_len),
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
        // step budget is capped by the still-available KV capacity — real
        // schedulers shrink the budget as allocated tokens grow (lightllm:
        // `max_total_token_num - total_token_size`), which is also what makes
        // budget-driven preemption (D7) reachable.
        let budget = self
            .cfg
            .step_budget
            .min(self.cfg.max_total_tokens.saturating_sub(self.kv_usage_tokens()) as usize);
        let decoding_view: Vec<DecodingReq> = self
            .decoding
            .iter()
            .map(|&id| {
                let r = &self.reqs[&id];
                DecodingReq { id, arrival_seq: r.arrival_seq(), resume: r.resume() }
            })
            .collect();
        let sel =
            select_batch(&eligible, &decoding_view, budget, self.cfg.kv_floor, self.cfg.schedule);
        // D7 victims: preempt, zero cursors, back to the waiting-queue head.
        for id in &sel.preempted {
            let dev = self.reqs[id].device_len();
            self.reqs.get_mut(id).expect("victim live").preempt().expect("decoding victim");
            self.returned += dev as u64;
            self.trace(*id, Event::Preempted);
            self.remove_from_decoding(*id);
            self.waiting.push_front(*id);
        }
        // Prefill assignments: dispatch then confirm synchronously (D1:
        // single-threaded event loop; CPU state mutated serially).
        for &(id, start, end) in &sel.prefill {
            debug_assert_eq!(
                start,
                self.reqs[&id].cached_len(),
                "selector chunk starts at cached_len"
            );
            let chunk_tokens = end - start;
            match self.reqs[&id].state() {
                ReqState::Waiting | ReqState::Preempted => {
                    let m = &self.meta[&id];
                    self.reqs
                        .get_mut(&id)
                        .expect("live req")
                        .start_prefill(m.prompt_len, m.max_output, chunk_tokens)
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
            self.dispatched += chunk_tokens as u64;
            match self.reqs.get_mut(&id).expect("live req").confirm() {
                ConfirmEvent::ChunkConfirmed => {
                    self.trace(id, Event::ChunkDone);
                }
                ConfirmEvent::PrefillDone => {
                    self.trace(id, Event::PrefillDone);
                    self.add_to_decoding(id);
                    self.remove_from_waiting(id);
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
        for id in ids {
            if self.reqs[&id].state() != ReqState::Decode {
                continue; // aborted mid-step
            }
            let seed_i = self.reqs[&id].seed_i();
            let pos = self.reqs[&id].cached_len();
            let token = gen_token(self.cfg, seed_i, pos);
            self.dispatched += 1;
            let ev = self.reqs.get_mut(&id).expect("live req").decode_step(token);
            match ev {
                ConfirmEvent::DecodeConfirmed { token } => {
                    self.trace(id, Event::DecodeToken { token });
                }
                ConfirmEvent::Stopped => {
                    self.trace(id, Event::Stopped);
                    self.terminal(id);
                }
                ConfirmEvent::Eos => {
                    self.trace(id, Event::Eos);
                    self.terminal(id);
                }
                ConfirmEvent::MaxOutput => {
                    self.trace(id, Event::MaxOutput);
                    self.terminal(id);
                }
                ev => unreachable!("{ev:?}"),
            }
        }
    }

    /// Terminal accounting: exactly-once release guard, EMA update, working
    /// set removal ("后释放").
    fn terminal(&mut self, id: ReqId) {
        let dev = self.reqs[&id].device_len();
        self.returned += dev as u64;
        self.reqs.get_mut(&id).expect("live req").take_release();
        let out = self.reqs[&id].output_tokens() as f64;
        self.ema.update(out);
        self.estimates.remove(&id);
        self.remove_from_decoding(id);
        self.remove_from_waiting(id);
    }

    fn add_to_decoding(&mut self, id: ReqId) {
        if !self.decoding.contains(&id) {
            self.decoding.push(id);
        }
    }

    fn remove_from_decoding(&mut self, id: ReqId) {
        self.decoding.retain(|&x| x != id);
    }

    /// Remove a request from the waiting queue (it left via PrefillDone →
    /// Decode, or terminal). Keeps the deque a real queue: a request is in it
    /// iff Waiting/Chunked/Preempted, so re-queueing (D7) can never duplicate.
    fn remove_from_waiting(&mut self, id: ReqId) {
        self.waiting.retain(|&x| x != id);
    }

    fn record_high_water(&mut self) {
        let s: usize = self.reqs.values().map(|r| r.device_len()).sum();
        self.high_water = self.high_water.max(s);
    }
}

/// Deterministic pseudo-logit for the greedy policy: a pure function of
/// (seed_i, position, vocab entry).
fn pseudo_logit(seed_i: u64, pos: usize, v: u32) -> u64 {
    let h = (pos as u64)
        .wrapping_mul(GOLDEN_RATIO_64)
        .wrapping_add((v as u64).wrapping_mul(0x94D0_49BB_1331_11EB));
    splitmix64(seed_i ^ h)
}

/// One deterministic token draw (position = `cached_len`, i.e. the next
/// output slot `prompt_len + output index`).
fn gen_token(cfg: &SchedConfig, seed_i: u64, pos: usize) -> u32 {
    match cfg.gen_policy {
        GenPolicy::Sample => rng_usize(seed_i, pos, cfg.vocab, cfg.vocab as usize) as u32,
        GenPolicy::Greedy => {
            let mut best: u32 = 0;
            let mut best_logit: u64 = 0;
            for v in 0..cfg.vocab {
                let l = pseudo_logit(seed_i, pos, v);
                if l >= best_logit {
                    best_logit = l;
                    best = v;
                }
            }
            best
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn base_cfg(seed: u64) -> SchedConfig {
        SchedConfig {
            base_seed: seed,
            vocab: 256,
            eos_id: Some(255),
            gen_policy: GenPolicy::Sample,
            step_budget: 64,
            kv_floor: 8,
            chunk_size: 16,
            schedule: SchedulePolicy::Fcfc,
            max_steps: 400,
            max_total_tokens: 1024,
            admission: AdmissionConfig {
                max_total_pages: 64,
                page_size: 16,
                running_max_req_size: 8,
                batch_max_tokens: 128,
                chunked_budget_multiplier: 2,
                router_token_ratio: 0.8,
                max_waiting_tokens: 16,
            },
        }
    }

    /// Deterministic pseudo-random scenario: 12 arrivals with mixed chunked
    /// prefill, stop patterns (incl. overlapping), abort and preempt
    /// injections — all derived from the seed.
    fn scenario(seed: u64) -> (SchedConfig, Vec<Arrival>) {
        let mut g = crate::rng::SplitMix64::new(seed);
        let cfg = base_cfg(seed);
        let patterns_pool: Vec<Vec<Vec<u32>>> = vec![
            vec![vec![1, 1, 1]],
            vec![vec![3, 1]],
            vec![vec![7]],
            vec![vec![5, 5], vec![5, 5, 5]],
            vec![vec![2, 4, 2, 4]],
            vec![],
        ];
        let mut arrivals = Vec::new();
        for i in 0..12u64 {
            let prompt_len = (4 + g.next_u64() % 60) as usize;
            let max_output = (2 + g.next_u64() % 14) as usize;
            let chunked = g.next_u64() % 2 == 1;
            let stop_patterns =
                patterns_pool[(g.next_u64() % patterns_pool.len() as u64) as usize].clone();
            let abort_at_step =
                if i % 3 == 1 { Some(3 + (g.next_u64() % 5) as usize) } else { None };
            let preempt_at_step =
                if i % 3 == 2 { Some(2 + (g.next_u64() % 4) as usize) } else { None };
            arrivals.push(Arrival {
                at_step: i as usize,
                prompt_len,
                max_output,
                chunked,
                stop_patterns,
                abort_at_step,
                preempt_at_step,
            });
        }
        (cfg, arrivals)
    }

    /// Token sequence of one request from the trace (in generation order).
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

    #[test]
    fn replay_is_bit_identical_across_20_seeds() {
        for seed in 0..20u64 {
            let (cfg, arrivals) = scenario(seed);
            let drv = SchedDeterminism::new(cfg.clone());
            let a = drv.assert_bit_identical(&arrivals);
            // The scenario must exercise the interesting transitions.
            assert!(
                a.trace.iter().any(|e| e.event == Event::PrefillStart),
                "seed {seed}: prefill exercised"
            );
            assert!(
                a.trace.iter().any(|e| matches!(e.event, Event::DecodeToken { .. })),
                "seed {seed}: decode exercised"
            );
            assert!(
                a.trace.iter().any(|e| {
                    matches!(
                        e.event,
                        Event::Stopped | Event::Eos | Event::MaxOutput | Event::Aborted
                    )
                }),
                "seed {seed}: terminal event exercised"
            );
            // Release-once accounting: everything dispatched is returned.
            if a.finals.iter().all(|(_, s, _, _)| matches!(s, ReqState::Done | ReqState::Aborted)) {
                assert_eq!(
                    a.total_dispatched, a.total_returned,
                    "seed {seed}: dispatch/return conservation (先分配、后释放)"
                );
            }
            // Allocated tokens stay inside the KV capacity (admission bound).
            assert!(
                a.high_water as u64 <= cfg.max_total_tokens,
                "seed {seed}: high-water <= KV cap"
            );
        }
    }

    #[test]
    fn different_seeds_produce_different_traces() {
        let (mut cfg, arrivals) = scenario(42);
        let drv_a = SchedDeterminism::new(cfg.clone());
        let a = drv_a.replay(&arrivals);
        cfg.base_seed = 43; // same arrivals, different seed
        let b = SchedDeterminism::new(cfg).replay(&arrivals);
        assert_ne!(a.trace, b.trace, "different seeds must diverge (sample policy)");
    }

    #[test]
    fn greedy_policy_has_no_rng_draws() {
        // Greedy is deterministic per (seed, pos) — still bit-identical.
        let mut cfg = base_cfg(5);
        cfg.gen_policy = GenPolicy::Greedy;
        let drv = SchedDeterminism::new(cfg);
        let arrivals = scenario(5).1;
        let a = drv.assert_bit_identical(&arrivals);
        assert!(a.trace.iter().any(|e| matches!(e.event, Event::DecodeToken { .. })));
    }

    #[test]
    fn abort_isolates_other_requests() {
        // Removing an abort must not change the survivors' token sequences
        // (spec: 其余请求输出与无 abort 基线运行逐 token 一致). Generation is a
        // pure function of (seed_i, pos), so survivors are unaffected.
        let cfg = base_cfg(7);
        let arrivals = vec![
            Arrival {
                at_step: 0,
                prompt_len: 8,
                max_output: 8,
                chunked: false,
                stop_patterns: vec![],
                abort_at_step: Some(3), // definitely live (decoding) at step 3
                preempt_at_step: None,
            },
            Arrival {
                at_step: 1,
                prompt_len: 16,
                max_output: 6,
                chunked: true,
                stop_patterns: vec![vec![3, 1]],
                abort_at_step: None,
                preempt_at_step: None,
            },
            Arrival {
                at_step: 2,
                prompt_len: 24,
                max_output: 4,
                chunked: true,
                stop_patterns: vec![vec![7]],
                abort_at_step: None,
                preempt_at_step: None,
            },
        ];
        let drv = SchedDeterminism::new(cfg.clone());
        let base = drv.replay(&arrivals);
        let aborted: Vec<ReqId> =
            base.trace.iter().filter(|e| e.event == Event::Aborted).map(|e| e.req).collect();
        assert!(!aborted.is_empty(), "the injection must fire");
        let clean: Vec<Arrival> =
            arrivals.iter().map(|a| Arrival { abort_at_step: None, ..a.clone() }).collect();
        let clean_out = drv.replay(&clean);
        // Survivors' token streams are bit-identical with the no-abort
        // baseline (generation is a pure function of (seed_i, pos); the
        // aborted request itself is expected to differ — it keeps generating
        // in the clean run). Step-level traces are NOT compared: the abort
        // frees budget and shifts the survivors' step timing.
        let aborted_set: std::collections::HashSet<ReqId> = aborted.iter().copied().collect();
        let ids: Vec<ReqId> =
            (0..arrivals.len() as u64).map(|s| ReqId::derive(cfg.base_seed, s)).collect();
        for id in ids {
            if aborted_set.contains(&id) {
                continue;
            }
            assert_eq!(
                token_seq(&base.trace, id),
                token_seq(&clean_out.trace, id),
                "survivor {id}"
            );
        }
    }

    #[test]
    fn preempted_request_resumes_from_zero_with_priority() {
        // A: arrives step 0, single-chunk prefill, decodes from step 1;
        // preempted at step 3 while decoding. B arrives step 4. Budget is
        // tight so only one request runs at a time.
        let mut cfg = base_cfg(11);
        cfg.step_budget = 16;
        cfg.kv_floor = 0;
        cfg.chunk_size = 8;
        let arrivals = vec![
            Arrival {
                at_step: 0,
                prompt_len: 8,
                max_output: 8,
                chunked: false,
                stop_patterns: vec![],
                abort_at_step: None,
                preempt_at_step: Some(3),
            },
            Arrival {
                at_step: 4,
                prompt_len: 8,
                max_output: 8,
                chunked: false,
                stop_patterns: vec![],
                abort_at_step: None,
                preempt_at_step: None,
            },
        ];
        let drv = SchedDeterminism::new(cfg.clone());
        let out = drv.replay(&arrivals);
        let aid = ReqId::derive(cfg.base_seed, 0);
        let bid = ReqId::derive(cfg.base_seed, 1);
        // Preempted with zeroed cursors (D7: all blocks released).
        let pe = out
            .trace
            .iter()
            .find(|e| e.req == aid && e.event == Event::PreemptInjected)
            .expect("injected preempt fires");
        assert_eq!((pe.cached_len, pe.device_len), (0, 0));
        // The request restarts prefill from scratch (chunk starts at 0) and
        // generates the same tokens again (deterministic pure RNG).
        let resumed: Vec<TraceEntry> = out
            .trace
            .iter()
            .filter(|e| e.req == aid && e.event == Event::PrefillStart)
            .cloned()
            .collect();
        assert!(resumed.len() >= 2, "prefill restarts after preemption");
        assert_eq!(resumed[1].cached_len, 0, "restart from cached = 0");
        let post_start = resumed[1].step;
        // Tokens generated before the preemption vs after the restart: same
        // (seed_i, pos) pairs → same tokens (positions reset with the cursors).
        let before: Vec<u32> = out
            .trace
            .iter()
            .filter(|e| e.req == aid && e.step < post_start)
            .filter_map(|e| match e.event {
                Event::DecodeToken { token } => Some(token),
                _ => None,
            })
            .collect();
        let post: Vec<u32> = out
            .trace
            .iter()
            .filter(|e| e.req == aid && e.step >= post_start)
            .filter_map(|e| match e.event {
                Event::DecodeToken { token } => Some(token),
                _ => None,
            })
            .collect();
        assert!(
            post.len() >= before.len(),
            "restart regenerates at least the pre-preemption prefix ({} vs {})",
            post.len(),
            before.len()
        );
        assert_eq!(&post[..before.len()], &before[..], "token stream restarts deterministically");
        // Resume priority (D3): A is re-scheduled before the new arrival B.
        let a_start = resumed[1].step;
        let b_start = out
            .trace
            .iter()
            .find(|e| e.req == bid && e.event == Event::PrefillStart)
            .map(|e| e.step)
            .expect("B starts prefill");
        assert!(a_start < b_start, "resume before new: {a_start} < {b_start}");
    }

    #[test]
    fn budget_victims_are_newest_and_resume_protected() {
        // Budget-driven preemption (D7) is the hard insurance for admission
        // estimate misses: here the KV capacity (256) is well below what
        // admission allows (64 pages × 16 = 1024), simulating a surprise
        // footprint growth — allocated tokens saturate the capacity, the
        // step budget shrinks to 0, and victims are preempted.
        let cfg = base_cfg(3);
        let mut cfg2 = cfg.clone();
        cfg2.max_total_tokens = 256; // KV capacity (busy ratio + budget cap)
        cfg2.step_budget = 2;
        cfg2.kv_floor = 0;
        cfg2.chunk_size = 16;
        let mk = |at: usize| Arrival {
            at_step: at,
            prompt_len: 16,
            max_output: 60,
            chunked: false,
            stop_patterns: vec![],
            abort_at_step: None,
            preempt_at_step: None,
        };
        let arrivals = vec![mk(0), mk(1), mk(2), mk(3)];
        let out = SchedDeterminism::new(cfg2).replay(&arrivals);
        let preempted: Vec<ReqId> =
            out.trace.iter().filter(|e| e.event == Event::Preempted).map(|e| e.req).collect();
        assert!(!preempted.is_empty(), "KV saturation must trigger victims");
        // First victim = newest arrival at that point (arrival_seq 3).
        assert_eq!(preempted[0], ReqId::derive(cfg.base_seed, 3));
        // Every preempted request is re-prefilled from scratch later
        // (resume works; cursors zeroed per D7).
        for v in &preempted {
            let resumed = out
                .trace
                .iter()
                .find(|e| e.req == *v && e.event == Event::PrefillStart && e.step > 3);
            assert!(resumed.is_some(), "victim {v} resumes");
            assert_eq!(resumed.unwrap().cached_len, 0, "resume restarts at cached = 0");
        }
    }

    #[test]
    fn stop_pattern_terminates_generation() {
        // A single-token stop pattern must eventually fire for *some* seed:
        // with a tiny vocab the drawn sequence has no particular structure,
        // so search seeds 0..40 for one where the pattern lands (a fixed
        // seed would be flaky — the pattern may simply never occur).
        let mut fired = false;
        for seed in 0..40u64 {
            let mut cfg = base_cfg(seed);
            cfg.vocab = 4; // tiny vocab: draws are in {0,1,2,3}
            cfg.eos_id = None;
            cfg.step_budget = 32;
            cfg.kv_floor = 0;
            let arrivals = vec![Arrival {
                at_step: 0,
                prompt_len: 2,
                max_output: 64,
                chunked: false,
                stop_patterns: vec![vec![2, 1]],
                abort_at_step: None,
                preempt_at_step: None,
            }];
            let out = SchedDeterminism::new(cfg).replay(&arrivals);
            if out.trace.iter().any(|e| e.event == Event::Stopped) {
                fired = true;
                break;
            }
        }
        assert!(fired, "stop pattern [2,1] must fire for some seed in 0..40");
    }

    #[test]
    fn eos_and_max_output_terminate() {
        let mut cfg = base_cfg(13);
        cfg.vocab = 64;
        cfg.eos_id = Some(7);
        cfg.step_budget = 64;
        cfg.kv_floor = 0;
        let arrivals = vec![
            Arrival {
                at_step: 0,
                prompt_len: 2,
                max_output: 4,
                chunked: false,
                stop_patterns: vec![],
                abort_at_step: None,
                preempt_at_step: None,
            },
            Arrival {
                at_step: 1,
                prompt_len: 2,
                max_output: 2,
                chunked: false,
                stop_patterns: vec![],
                abort_at_step: None,
                preempt_at_step: None,
            },
        ];
        let out = SchedDeterminism::new(cfg).replay(&arrivals);
        assert!(
            out.finals.iter().all(|(_, s, _, _)| matches!(s, ReqState::Done)),
            "both requests finish: {:?}",
            out.finals
        );
    }
}
