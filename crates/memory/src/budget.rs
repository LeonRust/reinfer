//! KV cache budget formulas (spec 005 D2 / S2-C, vLLM semantics).
//!
//! All functions here are pure and **model-agnostic**: they consume shape
//! numbers from the model config (the `LlamaConfig` fields) and device
//! memory figures — never a model identity. The scheduler (005 T2) derives
//! the per-request token estimates (the lightllm (a, b) pair, busy
//! threshold, EMA, peak formula) and feeds the page counts from this module.
//!
//! ## KV budget formula (90% semantics — vLLM `CacheConfig` 口径)
//!
//! ```text
//! kv_capacity = util × mem_total − weights − graph_pool − misc
//! pages       = max(0, floor(kv_capacity / page_bytes))
//! page_bytes  = n_layer × block_len × kv_heads × head_dim × 2 × 2
//! ```
//!
//! `page_bytes` mirrors the engine allocation (`crates/cuda/src/decode.rs`
//! `KvStore::alloc`: `per_region = total_pages × block_len × kv_heads × d ×
//! 2` bytes, K and V regions → ×2): one physical page spans all layers, one
//! block of f16 K and V slots. `util = 0.9` is the vLLM default
//! `gpu_memory_utilization`; the floor matches vLLM's `//` division, and the
//! clamp at 0 matches `max(..., 0)`. The 003-era workspace budget
//! (B-M11: `weights + KV + 0.5×weights`) is a different, per-engine-session
//! convention — this formula is the scheduler-layer KV pool budget.
//!
//! ## max-num-seqs (D2 corner)
//!
//! [`admit_max_seqs`] gates the concurrent sequence count jointly on the
//! page budget and the worst-case per-request window: one request at a full
//! `max_seq_len` window needs `n_layer × ceil(max_seq_len / block_len)`
//! pages, so
//!
//! ```text
//! admit = min(user_max_seqs, floor(total_pages / pages_per_req_full_window))
//! ```

/// KV geometry derived from the model config — the only model-dependent
/// numbers the budget needs (no model identity, only shapes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvGeometry {
    /// Transformer layers (`LlamaConfig::n_layer`).
    pub n_layer: usize,
    /// KV heads (`LlamaConfig::kv_heads`).
    pub kv_heads: usize,
    /// Head dimension (`LlamaConfig::head_dim`).
    pub head_dim: usize,
    /// Token slots per page (engine `BLOCK_LEN` = 32; `BlockLen::B32`).
    pub block_len: usize,
}

impl KvGeometry {
    /// Bytes of one physical KV page across **all layers**, f16 storage:
    /// `n_layer × block_len × kv_heads × head_dim × 2 (f16) × 2 (K+V)`.
    /// Mirrors `KvStore::alloc` in `crates/cuda/src/decode.rs`.
    pub const fn page_bytes_f16(&self) -> u64 {
        (self.n_layer as u64)
            * (self.block_len as u64)
            * (self.kv_heads as u64)
            * (self.head_dim as u64)
            * 2
            * 2
    }
}

/// Budget inputs: device memory total and the fixed reservations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KvBudgetInput {
    /// Device memory total visible to the allocator (bytes).
    pub mem_total_bytes: u64,
    /// Device-resident model weights (f16 upload), bytes.
    pub weights_bytes: u64,
    /// CUDA graph pool reservation, bytes.
    pub graph_pool_bytes: u64,
    /// Other fixed reservations (workspace, activation scratch, misc), bytes.
    pub misc_bytes: u64,
    /// Fraction of device memory usable for KV — vLLM
    /// `gpu_memory_utilization`, default 0.9. Valid range (0, 1].
    pub utilization: f64,
}

impl Default for KvBudgetInput {
    fn default() -> Self {
        Self {
            mem_total_bytes: 0,
            weights_bytes: 0,
            graph_pool_bytes: 0,
            misc_bytes: 0,
            utilization: 0.9,
        }
    }
}

/// vLLM-style KV capacity: `util × mem_total − weights − graph − misc`,
/// clamped at 0.
///
/// # Panics
///
/// In debug builds when `utilization` is not finite or outside (0, 1].
pub fn kv_capacity_bytes(input: &KvBudgetInput) -> u64 {
    let u = input.utilization;
    debug_assert!(u.is_finite() && u > 0.0 && u <= 1.0, "invalid utilization {u}");
    let budget = (input.mem_total_bytes as f64 * u) as u64;
    budget.saturating_sub(input.weights_bytes + input.graph_pool_bytes + input.misc_bytes)
}

/// How many physical pages fit in the 90% KV budget — the pool capacity
/// (`max_kv_pages`) for [`KvSegmentPool`](crate::segment::KvSegmentPool).
///
/// ```text
/// pages = max(0, floor(kv_capacity_bytes(input) / page_bytes))
/// ```
///
/// A result of 0 means no KV space: admission is impossible until
/// reservations shrink (or the pool is not constructed).
pub fn kv_budget_pages(input: &KvBudgetInput, geom: &KvGeometry) -> u64 {
    kv_capacity_bytes(input) / geom.page_bytes_f16()
}

/// Pages one request needs for `len` tokens across all layers:
/// `n_layer × ceil(len / block_len)` — the D2 换算 (token → page, ceiling).
pub const fn req_pages_for_len(geom: &KvGeometry, len: usize) -> u64 {
    geom.n_layer as u64 * len.div_ceil(geom.block_len) as u64
}

/// Pages one request needs at a full `max_seq_len` window (the max-num-seqs
/// worst case). Degenerate `max_seq_len == 0` clamps to the page granularity
/// (≥ 1 page per layer is not meaningful either — see [`admit_max_seqs`]).
pub const fn req_pages_full_window(geom: &KvGeometry, max_seq_len: usize) -> u64 {
    req_pages_for_len(geom, max_seq_len)
}

/// max-num-seqs joint gate (D2 corner): the concurrent request cap from the
/// KV page budget and the worst-case per-request window, combined with the
/// caller's configured cap:
///
/// ```text
/// per_req = max(1, n_layer × ceil(max_seq_len / block_len))
/// by_mem  = floor(total_pages / per_req)
/// admit   = min(max_seqs_user, by_mem)
/// ```
///
/// `total_pages` is the pool capacity from [`kv_budget_pages`]; a zero
/// budget admits nothing. This is only the *capacity* corner of D2 — the
/// per-request token estimates (a/b pair, EMA, busy, peak) are 005 T2,
/// scheduler-side.
pub fn admit_max_seqs(
    total_pages: u64,
    geom: &KvGeometry,
    max_seq_len: usize,
    max_seqs_user: usize,
) -> usize {
    let per_req = req_pages_for_len(geom, max_seq_len).max(1);
    let by_mem = (total_pages / per_req) as usize;
    max_seqs_user.min(by_mem)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Qwen3-0.6B-shaped geometry (spec 013 fixture values).
    const QWEN3_06B: KvGeometry =
        KvGeometry { n_layer: 28, kv_heads: 2, head_dim: 128, block_len: 32 };

    #[test]
    fn page_bytes_matches_engine_layout() {
        // decode.rs KvStore::alloc: per_region = total_pages × block_len ×
        // kv_heads × d × 2 bytes, two regions (K, V) → ×2 in total.
        let expect = 28 * 32 * 2 * 128 * 2 * 2;
        assert_eq!(QWEN3_06B.page_bytes_f16() as usize, expect);
        assert_eq!(expect, 917_504);
    }

    #[test]
    fn default_utilization_is_90pct() {
        assert_eq!(KvBudgetInput::default().utilization, 0.9);
    }

    #[test]
    fn kv_budget_pages_vllm_90pct() {
        let input = KvBudgetInput {
            mem_total_bytes: 100_000_000,
            weights_bytes: 10_000_000,
            graph_pool_bytes: 2_000_000,
            misc_bytes: 500_000,
            utilization: 0.9,
        };
        // kv_capacity = 90M − 10M − 2M − 0.5M = 77.5M; floor(/917_504) = 84
        assert_eq!(kv_capacity_bytes(&input), 77_500_000);
        assert_eq!(kv_budget_pages(&input, &QWEN3_06B), 84);
        let page = QWEN3_06B.page_bytes_f16();
        assert!(84 * page <= 77_500_000);
    }

    #[test]
    fn kv_budget_zero_when_reservations_exceed_util_budget() {
        let geom = KvGeometry { n_layer: 2, kv_heads: 1, head_dim: 64, block_len: 16 };
        let input = KvBudgetInput {
            mem_total_bytes: 100_000,
            weights_bytes: 95_000,
            ..KvBudgetInput::default()
        };
        assert_eq!(kv_capacity_bytes(&input), 0);
        assert_eq!(kv_budget_pages(&input, &geom), 0);
    }

    #[test]
    fn kv_budget_floor_boundary() {
        let geom = KvGeometry { n_layer: 1, kv_heads: 1, head_dim: 16, block_len: 16 };
        let page = geom.page_bytes_f16(); // 1×16×1×16×2×2 = 1024
        assert_eq!(page, 1024);
        // kv_capacity == page_bytes exactly → 1 page; one byte less → 0.
        // (kv_capacity = 0.9 × 100_000 = 90_000, so weights pin the margin.)
        let full = KvBudgetInput {
            mem_total_bytes: 100_000,
            weights_bytes: 90_000 - page,
            ..KvBudgetInput::default()
        };
        assert_eq!(kv_capacity_bytes(&full), page);
        assert_eq!(kv_budget_pages(&full, &geom), 1);
        let shy = KvBudgetInput {
            mem_total_bytes: 100_000,
            weights_bytes: 90_000 - page + 1,
            ..KvBudgetInput::default()
        };
        assert_eq!(kv_capacity_bytes(&shy), page - 1);
        assert_eq!(kv_budget_pages(&shy, &geom), 0);
    }

    #[test]
    fn graph_and_misc_are_subtracted() {
        let input = KvBudgetInput {
            mem_total_bytes: 100_000,
            weights_bytes: 10_000,
            graph_pool_bytes: 5_000,
            misc_bytes: 3_000,
            utilization: 0.9,
        };
        // 90_000 − 10_000 − 5_000 − 3_000 = 72_000
        assert_eq!(kv_capacity_bytes(&input), 72_000);
    }

    #[test]
    fn req_pages_ceiling_across_layers() {
        assert_eq!(req_pages_for_len(&QWEN3_06B, 0), 0);
        assert_eq!(req_pages_for_len(&QWEN3_06B, 1), 28);
        assert_eq!(req_pages_for_len(&QWEN3_06B, 32), 28);
        assert_eq!(req_pages_for_len(&QWEN3_06B, 33), 56);
        assert_eq!(req_pages_for_len(&QWEN3_06B, 2048), 28 * 64);
    }

    #[test]
    fn admit_max_seqs_joint_gate() {
        let per_req = req_pages_full_window(&QWEN3_06B, 2048); // 1792
        // Memory admits 3 full-window requests; the user cap (64) is higher.
        assert_eq!(admit_max_seqs(per_req * 3 + 100, &QWEN3_06B, 2048, 64), 3);
        // Plenty of memory: the user cap dominates.
        assert_eq!(admit_max_seqs(per_req * 100, &QWEN3_06B, 2048, 64), 64);
        // Zero budget admits nothing.
        assert_eq!(admit_max_seqs(0, &QWEN3_06B, 2048, 64), 0);
        // Degenerate window: clamped to ≥ 1 page per request, so memory
        // alone can never undercut the user cap.
        assert_eq!(admit_max_seqs(per_req * 2, &QWEN3_06B, 0, 64), 64);
    }
}
