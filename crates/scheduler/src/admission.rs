//! D2 admission estimation (lightllm 口径, complete formula set).
//!
//! Per-request estimate is the `(a, b)` pair:
//!
//! - `a = max(input_len + has_out_len + 1, shm_kv_len + 1)` — peak footprint;
//! - `b`: busy → `max_new_tokens` (pessimistic), else
//!   `min(max_new_tokens, max(1.1 × has_out_len, ema_req_out_len))`;
//! - `ADDED_OUTPUT_LEN = 16` slack on `b` (lightllm adds it to every request;
//!   plan D2's text mentions it in the chunked sentence — we follow lightllm);
//! - chunked requests additionally add the lifecycle extension term
//!   `ceil(remaining_prefill / chunk_size) × (max_waiting_token + 1)`;
//! - busy judgement: KV occupancy / max_total_token_num ≥ router_token_ratio;
//! - EMA of request output length: init 2048, floor 64, adaptive α
//!   (0.9 when the actual exceeds the previous EMA, else 0.6);
//! - peak formula (lightllm): sort by `b` descending, then
//!   `need_max = max(left_out_len[k] × k + cum_run_len[k])` with 1-based `k`,
//!   `left_out_len[k] = b` and `cum_run_len[k] = Σ a[j] for j ≤ k` — i.e. the
//!   loop `cur += a; need_max = max(need_max, b × (i + 1) + cur)`;
//! - token→page conversion: ceiling, plus `page_size - 1` intra-page slack
//!   per running request (lightllm's tgt_len / page conversion).
//!
//! Admission gates (in order): request count ≤ `running_max_req_size`; the
//! lightllm totals check (`Σa + Σb` when busy, `Σa`/`Σb` separately when not);
//! peak ≤ `max_total_pages × page_size` tokens (lightllm's
//! `need_max > max_total_token_num`); and the per-step first prefill-chunk
//! budget `batch_max_tokens` (doubled in chunked mode).

use crate::req::ReqId;

/// Output slack (lightllm `ADDED_OUTPUT_LEN`).
pub const ADDED_OUTPUT_LEN: u64 = 16;

/// Initial EMA of the request output length (plan D2).
pub const EMA_INIT: f64 = 2048.0;

/// Lower bound of the EMA (plan D2: 下限 64).
pub const EMA_FLOOR: f64 = 64.0;

/// Lightllm-style per-request estimate `(a, b)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestEstimate {
    /// Peak token footprint `max(input_len + has_out_len + 1, shm_kv_len + 1)`.
    pub a: u64,
    /// Tokens per decode step (busy: pessimistic `max_new_tokens`).
    pub b: u64,
}

/// Inputs for `estimate`.
#[derive(Debug, Clone, Copy)]
pub struct EstimateInput {
    /// Prompt token count.
    pub input_len: u64,
    /// `max_new_tokens` of the request.
    pub max_new_tokens: u64,
    /// Already generated (confirmed) output tokens.
    pub has_out_len: u64,
    /// Shared (prefix-cache) KV length; 0 until P3 RadixCache lands (D9).
    pub shm_kv_len: u64,
    /// Remaining prefill tokens when the request is chunked (0 otherwise).
    pub chunked_remaining: u64,
    /// Chunk size for chunked prefill.
    pub chunk_size: u64,
    /// Busy judgement (D2: KV occupancy ratio).
    pub busy: bool,
    /// Current EMA of request output length.
    pub ema_req_out_len: u64,
    /// `max_waiting_token` (chunked lifecycle extension term).
    pub max_waiting_tokens: u64,
}

/// D2 `(a, b)` estimation formula (lightllm 口径).
pub fn estimate(input: EstimateInput) -> RequestEstimate {
    let a = (input.input_len + input.has_out_len + 1).max(input.shm_kv_len + 1);
    let b = if input.busy {
        input.max_new_tokens
    } else {
        let observed = ((1.1 * input.has_out_len as f64).ceil() as u64).max(input.ema_req_out_len);
        input.max_new_tokens.min(observed)
    };
    let mut b = b.saturating_add(ADDED_OUTPUT_LEN);
    if input.chunked_remaining > 0 && input.chunk_size > 0 {
        let chunks = input.chunked_remaining.div_ceil(input.chunk_size);
        b = b.saturating_add(chunks.saturating_mul(input.max_waiting_tokens + 1));
    }
    RequestEstimate { a, b }
}

/// EMA of the observed request output length (lightllm adaptive α, floored).
#[derive(Debug, Clone)]
pub struct EmaTracker {
    ema: f64,
    alpha: f64,
}

impl Default for EmaTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl EmaTracker {
    /// New tracker at `EMA_INIT` with α = 0.6.
    pub fn new() -> Self {
        Self { ema: EMA_INIT, alpha: 0.6 }
    }

    /// Current EMA value.
    pub fn ema(&self) -> f64 {
        self.ema
    }

    /// Current adaptive α.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Update with an observed output length and return the new EMA.
    /// α is chosen by comparing the actual against the previous EMA
    /// (0.9 when larger, else 0.6); the result is floored at `EMA_FLOOR`.
    pub fn update(&mut self, actual_out_len: f64) -> f64 {
        let alpha = if actual_out_len > self.ema { 0.9 } else { 0.6 };
        let next = (alpha * self.ema + (1.0 - alpha) * actual_out_len).max(EMA_FLOOR);
        self.alpha = alpha;
        self.ema = next;
        next
    }
}

/// D2 busy judgement: KV occupation ratio ≥ `router_token_ratio`.
pub fn is_busy(kv_used_tokens: u64, max_total_tokens: u64, ratio: f64) -> bool {
    max_total_tokens > 0 && ratio <= 1.0 && kv_used_tokens as f64 / max_total_tokens as f64 >= ratio
}

/// D2 peak formula: sort by `b` descending (stable), then
/// `need_max = max_k(b[k] × (k+1) + Σ_{j≤k} a[j])` (lightllm; plan D2's
/// 1-based `left_out_len[k] × k + cum_run_len[k]` with `cum_run_len`
/// including the current request).
pub fn peak_tokens(estimates: &[(ReqId, RequestEstimate)]) -> u64 {
    let mut sorted: Vec<&(ReqId, RequestEstimate)> = estimates.iter().collect();
    sorted.sort_by_key(|(_, e)| std::cmp::Reverse(e.b)); // b desc; stable → arrival order within ties
    let mut peak = 0u64;
    let mut cur = 0u64;
    for (i, (_, e)) in sorted.iter().enumerate() {
        cur += e.a;
        peak = peak.max(e.b.saturating_mul(i as u64 + 1).saturating_add(cur));
    }
    peak
}

/// Ceiling token→page conversion.
pub fn pages_needed(tokens: u64, page_size: usize) -> u64 {
    tokens.div_ceil(page_size as u64)
}

/// D2 conversion: ceiling plus `page_size - 1` intra-page slack per running
/// request.
pub fn token_budget_with_slack(tokens: u64, page_size: usize, running_reqs: usize) -> u64 {
    pages_needed(tokens, page_size) + running_reqs as u64 * (page_size as u64 - 1)
}

/// D2 admission gates.
#[derive(Debug, Clone, Copy)]
pub struct AdmissionConfig {
    /// KV pool capacity in pages (hard peak gate; token equivalent =
    /// `max_total_pages × page_size`).
    pub max_total_pages: u64,
    /// KV page length in tokens.
    pub page_size: usize,
    /// Max concurrent running requests (`running_max_req_size`).
    pub running_max_req_size: usize,
    /// Per-step cap on the sum of first prefill chunks.
    pub batch_max_tokens: u64,
    /// Chunked-prefill mode multiplies the batch token budget (D2: 翻倍).
    pub chunked_budget_multiplier: u64,
    /// Busy threshold (KV occupancy ratio).
    pub router_token_ratio: f64,
    /// `max_waiting_token` for the chunked lifecycle extension term.
    pub max_waiting_tokens: u64,
}

/// Result of an admission check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionVerdict {
    /// The working set fits.
    Admitted,
    /// Rejected; the request stays in the waiting queue.
    Denied {
        /// Which gate rejected the request.
        reason: DenyReason,
    },
}

/// Why a request was denied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenyReason {
    /// Peak (or busy totals) exceed the KV budget.
    PeakBudget,
    /// More requests than `running_max_req_size`.
    TooManyRequests,
    /// The per-step first-chunk sum would exceed `batch_max_tokens` (× the
    /// chunked multiplier).
    FirstChunkSum,
}

/// Check admission for the whole working set (`working` includes the request
/// under review, so chunked continuations re-check their own slot).
///
/// `first_chunk`: the request's first prefill chunk size — `Some` only when
/// the request starts prefill for the first time (the per-step chunk-sum gate
/// applies once per request); `None` skips that gate.
pub fn check_admission(
    working: &[(ReqId, RequestEstimate)],
    first_chunk: Option<u64>,
    step_first_chunk_sum: u64,
    busy: bool,
    config: &AdmissionConfig,
) -> AdmissionVerdict {
    if working.len() > config.running_max_req_size {
        return AdmissionVerdict::Denied { reason: DenyReason::TooManyRequests };
    }
    let max_total_tokens = config.max_total_pages * config.page_size as u64;
    let sum_a: u64 = working.iter().map(|(_, e)| e.a).sum();
    let sum_b: u64 = working.iter().map(|(_, e)| e.b).sum();
    // Lightllm totals check: busy → Σa + Σb must fit; idle → each alone.
    // (Note: dominated by the peak gate below — peak ≤ totals always — but
    // kept for lightllm parity.)
    let totals_ok = if busy {
        sum_a.saturating_add(sum_b) <= max_total_tokens
    } else {
        sum_a <= max_total_tokens && sum_b <= max_total_tokens
    };
    if !totals_ok {
        return AdmissionVerdict::Denied { reason: DenyReason::PeakBudget };
    }
    // Peak gate in tokens (lightllm: `need_max > max_total_token_num`).
    // `token_budget_with_slack` is the batch-side token→page conversion
    // (D2 换算), applied when budgets are expressed in pages (T4).
    if peak_tokens(working) > max_total_tokens {
        return AdmissionVerdict::Denied { reason: DenyReason::PeakBudget };
    }
    if let Some(fc) = first_chunk {
        let cap = config.batch_max_tokens.saturating_mul(config.chunked_budget_multiplier);
        if step_first_chunk_sum.saturating_add(fc) > cap {
            return AdmissionVerdict::Denied { reason: DenyReason::FirstChunkSum };
        }
    }
    AdmissionVerdict::Admitted
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn est(a: u64, b: u64) -> RequestEstimate {
        RequestEstimate { a, b }
    }

    fn input(len: u64, max_new: u64) -> EstimateInput {
        EstimateInput {
            input_len: len,
            max_new_tokens: max_new,
            has_out_len: 0,
            shm_kv_len: 0,
            chunked_remaining: 0,
            chunk_size: 0,
            busy: false,
            ema_req_out_len: EMA_INIT.round() as u64,
            max_waiting_tokens: 16,
        }
    }

    #[test]
    fn estimate_a_holds_prompt_plus_first_output() {
        let e = estimate(input(100, 64));
        assert_eq!(e.a, 101, "a = input_len + 0 outputs + 1");
        let mut i = input(100, 64);
        i.has_out_len = 7;
        assert_eq!(estimate(i).a, 108);
        i.shm_kv_len = 200;
        assert_eq!(estimate(i).a, 201, "shared prefix dominates");
    }

    #[test]
    fn estimate_b_busy_pessimistic_vs_idle() {
        // idle: b = min(max_new, max(1.1×out, ema)) + 16 (lightllm slack)
        let e = estimate(input(8, 64));
        assert_eq!(e.b, 80, "ema 2048 dominates, capped by max_new, +16");
        let mut i = input(8, 300);
        i.has_out_len = 20;
        // max(1.1×20=22, ema 2048) = 2048 → min(300, 2048) = 300 → +16
        assert_eq!(estimate(i).b, 316);
        i.ema_req_out_len = 10;
        // max(22, 10) = 22 → min(300, 22) = 22 → +16
        assert_eq!(estimate(i).b, 38, "1.1×20=22 wins over ema 10");
        // busy: b = max_new_tokens + 16 (pessimistic, D2)
        i.busy = true;
        assert_eq!(estimate(i).b, 316);
        i.busy = false;
        i.max_new_tokens = 8;
        assert_eq!(estimate(i).b, 24, "max_new caps the observed");
    }

    #[test]
    fn estimate_chunked_lifecycle_extension() {
        // 20 remaining prefill tokens / chunk 8 → 3 chunks × (16 + 1) = 51
        let mut i = input(20, 300);
        i.chunked_remaining = 20;
        i.chunk_size = 8;
        let e = estimate(i);
        assert_eq!(e.b, 300 + ADDED_OUTPUT_LEN + 51);
        assert_eq!(e.a, 21);
    }

    #[test]
    fn ema_update_floor_and_adaptive_alpha() {
        let mut ema = EmaTracker::new();
        assert_eq!(ema.ema(), EMA_INIT);
        // actual below EMA → alpha 0.6
        let v = ema.update(100.0);
        assert_eq!(v, 0.6 * EMA_INIT + 0.4 * 100.0);
        assert_eq!(ema.alpha(), 0.6);
        // actual above previous EMA → alpha 0.9
        let v2 = ema.update(5000.0);
        assert_eq!(v2, 0.9 * v + 0.1 * 5000.0);
        assert_eq!(ema.alpha(), 0.9);
        // floor: even a tiny actual keeps EMA ≥ 64
        for _ in 0..20 {
            ema.update(1.0);
        }
        assert!(ema.ema() >= EMA_FLOOR);
    }

    #[test]
    fn busy_threshold() {
        assert!(is_busy(80, 100, 0.8));
        assert!(!is_busy(79, 100, 0.8));
        assert!(!is_busy(0, 0, 0.8), "degenerate pool is not busy");
    }

    #[test]
    fn peak_formula_hand_computed() {
        // sorted by b desc: [(a10,b8), (a5,b3)]
        // i0: cur=10, 8×1+10=18; i1: cur=15, 3×2+15=21 → peak 21
        let working = vec![(ReqId::derive(0, 1), est(10, 8)), (ReqId::derive(0, 2), est(5, 3))];
        assert_eq!(peak_tokens(&working), 21);
        // order by b: single request → b + a
        let one = vec![(ReqId::derive(0, 1), est(9, 24))];
        assert_eq!(peak_tokens(&one), 33);
        // tie on b: stable arrival order decides, same peak either way
        let tie = vec![(ReqId::derive(0, 1), est(4, 5)), (ReqId::derive(0, 2), est(6, 5))];
        assert_eq!(peak_tokens(&tie), 20); // i0: 5+4=9; i1: 5×2+10=20
    }

    #[test]
    fn pages_ceil_and_slack() {
        assert_eq!(pages_needed(0, 16), 0);
        assert_eq!(pages_needed(1, 16), 1);
        assert_eq!(pages_needed(16, 16), 1);
        assert_eq!(pages_needed(17, 16), 2);
        // 40 tokens → 3 pages; plus 2 running × 15 slack → 33
        assert_eq!(token_budget_with_slack(40, 16, 2), 33);
    }

    fn cfg() -> AdmissionConfig {
        AdmissionConfig {
            max_total_pages: 64,
            page_size: 16,
            running_max_req_size: 4,
            batch_max_tokens: 100,
            chunked_budget_multiplier: 2,
            router_token_ratio: 0.8,
            max_waiting_tokens: 16,
        }
    }

    #[test]
    fn admission_boundaries() {
        let c = cfg(); // max_total_tokens = 64 pages × 16 = 1024
        // single small request admitted (idle)
        let one = vec![(ReqId::derive(0, 1), est(9, 24))];
        assert_eq!(check_admission(&one, None, 0, false, &c), AdmissionVerdict::Admitted);
        // count gate: 5 > running_max_req_size 4
        let many: Vec<_> = (0..5).map(|i| (ReqId::derive(0, i), est(9, 24))).collect();
        assert_eq!(
            check_admission(&many, None, 0, false, &c),
            AdmissionVerdict::Denied { reason: DenyReason::TooManyRequests }
        );
        // peak gate: peak = 600 + 600 = 1200 > 1024
        let big = vec![(ReqId::derive(0, 1), est(600, 600))];
        assert_eq!(
            check_admission(&big, None, 0, false, &c),
            AdmissionVerdict::Denied { reason: DenyReason::PeakBudget }
        );
        // busy totals branch (lightllm parity): Σa+Σb > max_total_tokens,
        // while the peak stays under the budget (peak ≤ totals always).
        // Custom config: 3 pages × 4 tokens = 12 token budget.
        let tiny = AdmissionConfig { max_total_pages: 3, page_size: 4, ..c };
        // (a7,b3), (a1,b2): peak = max(3+7, 2×2+8) = 12 ≤ 12; totals = 13 > 12
        let busy = vec![(ReqId::derive(0, 1), est(7, 3)), (ReqId::derive(0, 2), est(1, 2))];
        assert_eq!(
            check_admission(&busy, None, 0, true, &tiny),
            AdmissionVerdict::Denied { reason: DenyReason::PeakBudget },
            "busy totals 13 > 12"
        );
        // the same working set passes when idle (Σa, Σb, peak each ≤ 12)
        assert_eq!(check_admission(&busy, None, 0, false, &tiny), AdmissionVerdict::Admitted);
        // first-chunk gate: cap = batch_max_tokens × chunked multiplier
        // (D2: chunked prefill doubles the per-step batch budget).
        let one = vec![(ReqId::derive(0, 1), est(60, 24))];
        assert_eq!(check_admission(&one, Some(60), 0, false, &c), AdmissionVerdict::Admitted);
        assert_eq!(
            check_admission(&one, Some(60), 141, false, &c),
            AdmissionVerdict::Denied { reason: DenyReason::FirstChunkSum },
            "60 + 141 > 200 (cap = 100 × multiplier 2)"
        );
        // multiplier 1 caps at 100: 60 ≤ 100, then 60 + 50 > 100
        let mut c2 = c;
        c2.chunked_budget_multiplier = 1;
        assert_eq!(check_admission(&one, Some(60), 0, false, &c2), AdmissionVerdict::Admitted);
        assert_eq!(
            check_admission(&one, Some(60), 50, false, &c2),
            AdmissionVerdict::Denied { reason: DenyReason::FirstChunkSum },
            "60 + 50 > 100 (multiplier 1)"
        );
    }

    #[test]
    fn admission_rejects_continuations_without_gate() {
        // Continuations pass `first_chunk: None` → no chunk-sum gate.
        let c = cfg();
        let small = vec![(ReqId::derive(0, 1), est(60, 24))];
        assert_eq!(check_admission(&small, None, u64::MAX, false, &c), AdmissionVerdict::Admitted);
        // the peak gate still applies to continuations
        let big = vec![(ReqId::derive(0, 1), est(1000, 1000))];
        assert_eq!(
            check_admission(&big, None, u64::MAX, false, &c),
            AdmissionVerdict::Denied { reason: DenyReason::PeakBudget }
        );
    }
}
