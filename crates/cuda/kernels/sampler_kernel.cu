// gpu_sampler_kernel: 006-2 T3C single-launch GPU sampler chain (single block).
//
// D2 三层确定性契约的 GPU 侧实现（specs/006-2-decoding-kernel-performance/plan.md D2）:
//   - temp=0: no-RNG hardware argmax; tie-break = LAST max, i.e. the largest
//     token id among equal maxima — identical to llm-samplers `SampleGreedy`
//     (`max_by` + `total_cmp` over the raw list) and to greedy over the
//     top-k/top-p-truncated sorted list (ties are ordered larger-tid-first in
//     llm-samplers' insertion sort, and `max_by` keeps the first of a tie run,
//     so both reduce to "largest tid among maxima"). Bit-identical with the
//     CPU adapter for the covered surface; see Phase-G note for the top-k/
//     top-p inertness proof at temp=0.
//   - temp>0: same-distribution only; the draw uses a pure-function (i,p,v)
//     Gumbel noise — u(v) = splitmix64(base ^ splitmix64(v)) where `base` is
//     the single 64-bit value the host advances from RngState per step
//     ((i,p) index fold). No stream-ordered RNG. Gumbel-max over the filtered
//     set samples from the renormalized softmax exactly (llm-samplers
//     recomputes the softmax over the survivors before WeightedIndex; both
//     yield the categorical over the same survivor set).
//
// Chain order (005 D5): logit_bias → frequency/presence penalties →
// repetition penalty → temperature (÷) → softmax (f32 max-subtracted) →
// min_p → top_k → top_p → gumbel-max argmax. At temp=0 the effective order
// coincides with the legacy llm-samplers chain (penalties → top-k → top-p →
// greedy) for the covered surface.
//
// Filter exactness (single launch, single block — no extra launches):
//   - top_k: the k-th largest (value, tid) pair-key is found by a 64-round
//     in-kernel bisection over the 64-bit key space
//     ((ascending-sortable value) << 32 | ~tid); survivors = key >= tau_k.
//     This reproduces llm-samplers' stable-sorted truncate(k) exactly,
//     including boundary ties (smaller tid kept first — llm-samplers
//     `sort_by` is stable, so tie runs keep their original ascending-tid
//     order).
//   - top_p: llm-samplers truncates the sorted prefix whose sequential f32
//     cumsum first reaches p. Sequential cumsum in sorted order cannot be
//     reproduced in an unsorted single launch; instead the kernel computes the
//     boundary VALUE theta* = max{theta : sum_{prob >= theta} prob >= p*Z_S}
//     (Z_S = renormalization mass of the top-k survivor set) by a 30-round
//     in-kernel bisection, and keeps all tokens with prob >= theta*. For
//     continuous logits this set equals the llm-samplers prefix; when the
//     boundary value is TIED the GPU keeps the whole tie group (over-keep).
//     Recorded deviation, D2 tier-2 same-distribution scope.
//   - min_p: keep prob >= max_prob * min_p (llm-samplers SampleMinP, min_keep
//     trivially satisfied by the max).
//
// Determinism: fixed element→thread mapping (strided, ascending id), fixed
// tree reductions (warp shuffle + shared stages) — same seed + same inputs
// give bit-identical tokens.
//
// Non-finite logits are excluded everywhere (mirror of llm-samplers
// `Logits::try_from_iter` is_finite filter): the penalized value is computed
// only for finite inputs, otherwise the sentinel -FLT_MAX is stored — it
// ranks below every real value and has softmax prob 0, so it can never be
// sampled; if no finite logit exists the kernel reports STATUS_NO_TOKEN.

#include <float.h>
#include <math.h>

#define SAMPLER_BLOCK 256

// ---- SplitMix64 mixing (mirror of crates/kernels sampler.rs) --------------
__device__ __forceinline__ unsigned long long smix_mix(unsigned long long z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
// Unit uniform in [0,1) from the 24 high bits (mirror of SplitMix64::next_f32_unit).
__device__ __forceinline__ float smix_unit(unsigned long long x) {
    return (float)(x >> 40) * (1.0f / 16777216.0f);
}
// Gumbel noise for candidate `v`: pure function of (base, v); base folds the
// (i,p) index (advanced once per step by the host). u==0 → -ln(-ln(0)) = -inf
// (that candidate never wins); u<1 by construction, no NaN.
__device__ __forceinline__ float gumbel_noise(unsigned long long base, unsigned v) {
    float u = smix_unit(smix_mix(base ^ smix_mix((unsigned long long)v)));
    return -logf(-logf(u));
}

// ---- 64-bit pair key: (ascending-sortable value, tid-complement) ----------
// Larger key = larger value; equal values ordered by SMALLER tid first.
// llm-samplers orders ties by stable `sort_by` (original ascending-tid order),
// so its sorted-desc list and the truncate(k) survivor set are reproduced
// exactly by "key >= tau" over this key space — boundary tie groups are cut
// to their smallest-tid members, matching the CPU chain's survivor set (the
// temp>0 categorical over the survivors must be identical to the CPU's).
__device__ __forceinline__ unsigned long long pair_key(float val, unsigned tid) {
    unsigned bits = __float_as_uint(val);
    unsigned asc = (bits >> 31) ? ~bits : (bits ^ 0x80000000u);  // ascending by value
    return ((unsigned long long)asc << 32) | (0xFFFFFFFFu - tid);
}

// ---- block reductions (deterministic tree: shuffle + shared stage) --------
// All three helpers: lanes of warp 0 finish the second stage; threads of warp
// 0 that did not read a shared slot contribute the neutral element.

__device__ __forceinline__ void block_sum_f32(
    unsigned tid, float v, float* s_red, float* out)
{
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    if ((tid & 31u) == 0) s_red[tid >> 5] = v;
    __syncthreads();
    if (tid < 32) {
        v = (tid < (SAMPLER_BLOCK >> 5)) ? s_red[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down_sync(0xffffffffu, v, off);
        }
        if (tid == 0) *out = v;
    }
}

__device__ __forceinline__ void block_count_u64(
    unsigned tid, unsigned long long c, unsigned long long* s_cnt, unsigned long long* out)
{
    for (int off = 16; off > 0; off >>= 1) {
        c += __shfl_down_sync(0xffffffffu, c, off);
    }
    if ((tid & 31u) == 0) s_cnt[tid >> 5] = c;
    __syncthreads();
    if (tid < 32) {
        c = (tid < (SAMPLER_BLOCK >> 5)) ? s_cnt[tid] : 0ull;
        for (int off = 16; off > 0; off >>= 1) {
            c += __shfl_down_sync(0xffffffffu, c, off);
        }
        if (tid == 0) *out = c;
    }
}

// Pair (value, id) max with the LastMax rule: (v,i) wins over (ov,oi) iff
// v > ov || (v == ov && i > oi).
__device__ __forceinline__ void block_pair_max(
    unsigned tid, float v, unsigned i,
    float* s_v, unsigned* s_i, float* out_v, unsigned* out_i)
{
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_down_sync(0xffffffffu, v, off);
        unsigned oi = __shfl_down_sync(0xffffffffu, i, off);
        if (ov > v || (ov == v && oi > i)) { v = ov; i = oi; }
    }
    if ((tid & 31u) == 0) { s_v[tid >> 5] = v; s_i[tid >> 5] = i; }
    __syncthreads();
    if (tid < 32) {
        if (tid >= (SAMPLER_BLOCK >> 5)) { v = -FLT_MAX; i = 0u; }
        else { v = s_v[tid]; i = s_i[tid]; }
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_down_sync(0xffffffffu, v, off);
            unsigned oi = __shfl_down_sync(0xffffffffu, i, off);
            if (ov > v || (ov == v && oi > i)) { v = ov; i = oi; }
        }
        if (tid == 0) { *out_v = v; *out_i = i; }
    }
}

// Out layout (host contract): out[0] token, out[1] tie_break (1=LastMax),
// out[2] status (0 ok, 1 no finite logit).
extern "C" __global__ void gpu_sampler_kernel(
    const float* __restrict__ logits,          // [vocab] raw device logits
    const unsigned* __restrict__ window,       // [window_len] last tokens (penalty history)
    unsigned window_len,
    const unsigned* __restrict__ bias_ids,     // [n_bias] logit-bias ids, sorted ascending
    const float* __restrict__ bias_vals,       // [n_bias] logit-bias values
    unsigned n_bias,
    float freq_pen,                            // 0.0 = off
    float pres_pen,                            // 0.0 = off
    float rep_pen,                             // <= 1.0 = off
    float temperature,                         // <= 0.0 = greedy (no RNG)
    unsigned top_k,                            // 0 = off; else min(k, vocab)
    float top_p,                               // <= 0.0 or >= 1.0 = off
    float min_p,                               // <= 0.0 = off
    unsigned long long rng_base,               // (i,p) pure-function index; unused at temp<=0
    unsigned vocab,
    float* __restrict__ val_scratch,           // [vocab] penalized/temperature-scaled values
    float* __restrict__ prob_scratch,          // [vocab] softmax probs (temp>0 only)
    unsigned* __restrict__ out)                // [3] token / tie / status
{
    __shared__ float  s_red[SAMPLER_BLOCK];
    __shared__ unsigned long long s_cnt[SAMPLER_BLOCK];
    __shared__ float  s_v[SAMPLER_BLOCK >> 5];
    __shared__ unsigned s_i[SAMPLER_BLOCK >> 5];
    __shared__ float  s_pair_v;
    __shared__ unsigned s_pair_i;
    __shared__ unsigned s_any_finite;
    __shared__ float  s_maxval, s_sume, s_maxprob, s_z_s, s_target, s_flo, s_fhi, s_fmid;
    __shared__ unsigned long long s_lo, s_hi, s_mid, s_tau_pair;

    const unsigned tid = threadIdx.x;

    // ---------- Phase A: logit_bias + penalties + temperature --------------
    bool any_finite_local = false;
    for (unsigned v = tid; v < vocab; v += SAMPLER_BLOCK) {
        float l = logits[v];
        if (!isfinite(l)) {
            val_scratch[v] = -FLT_MAX;  // excluded sentinel
            continue;
        }
        any_finite_local = true;
        if (n_bias > 0) {
            // binary search over ascending ids (host-sorted pairs, L2-resident)
            unsigned lo = 0, hi = n_bias;
            while (lo < hi) {
                unsigned mid = (lo + hi) >> 1;
                if (bias_ids[mid] < v) lo = mid + 1; else hi = mid;
            }
            if (lo < n_bias && bias_ids[lo] == v) l += bias_vals[lo];
        }
        if (window_len > 0 && (freq_pen != 0.0f || pres_pen != 0.0f || rep_pen > 1.0f)) {
            unsigned cnt = 0;
            for (unsigned w = 0; w < window_len; ++w) {
                if (window[w] == v) ++cnt;
            }
            if (cnt > 0) {
                // llm-samplers SampleFreqPresence: logit -= cnt*freq + (cnt>0)*pres
                l -= freq_pen * (float)cnt;
                l -= pres_pen;
                // llm-samplers SampleRepetition: logit<=0 -> *rep; logit>0 -> /rep
                if (rep_pen > 1.0f) l = l > 0.0f ? l / rep_pen : l * rep_pen;
            }
        }
        if (temperature > 0.0f) l /= temperature;
        val_scratch[v] = l;
    }
    // any-finite flag (count of 1s)
    {
        unsigned long long c = any_finite_local ? 1ull : 0ull;
        block_count_u64(tid, c, s_cnt, &s_tau_pair);  // scratch slot, rewritten below
        __syncthreads();
        if (tid == 0) s_any_finite = (unsigned)(s_tau_pair & 0xFFFFFFFFull);
        __syncthreads();
    }

    // ---------- Phase G: temp=0 greedy (hardware argmax, LastMax) ----------
    // NOTE (top-k/top-p inertness at temp=0): the max VALUE always survives
    // top-k and top-p (the truncated prefix starts with the max), so the
    // greedy TOKEN depends only on penalties + argmax. Tie-break: this kernel
    // pins the LastMax rule (spec 006-2 r2 — llm-samplers `max_by` semantics,
    // matching the CPU adapter on the no-filter surface, bit-identical).
    // RECORDED DEVIATION: with top-k/top-p ENABLED, llm-samplers runs its
    // stable sort and `SampleGreedy` takes the truncated list's FIRST element,
    // so the CPU chain yields the FIRST max when the global max has ties
    // (a stable-sort artifact of the legacy chain). The GPU ignores filters
    // at temp=0 and always yields the LAST max (D2 tier-1 pin, r2 revision);
    // the two differ only when the max value is tied AND a filter is enabled.
    // Hence the greedy token depends only on
    // penalties + argmax(LastMax) — bit-identical with the llm-samplers chain
    // without computing softmax or filters at all.
    if (temperature <= 0.0f) {
        float best_v = -FLT_MAX;
        unsigned best_i = 0;
        for (unsigned v = tid; v < vocab; v += SAMPLER_BLOCK) {
            float lv = val_scratch[v];
            if (lv > best_v || (lv == best_v && v > best_i)) { best_v = lv; best_i = v; }
        }
        block_pair_max(tid, best_v, best_i, s_v, s_i, &s_pair_v, &s_pair_i);
        __syncthreads();
        if (tid == 0) {
            out[0] = s_pair_i;
            out[1] = 1u;  // TieBreak::LastMax
            out[2] = s_any_finite != 0u ? 0u : 1u;  // STATUS_NO_TOKEN when no finite logit
        }
        return;
    }

    // ---------- Phase S: softmax (f32 max-subtracted) ----------------------
    {
        float loc_max = -FLT_MAX;
        for (unsigned v = tid; v < vocab; v += SAMPLER_BLOCK) {
            float lv = val_scratch[v];
            if (lv > loc_max) loc_max = lv;
        }
        block_pair_max(tid, loc_max, 0u, s_v, s_i, &s_maxval, &s_pair_i);
        __syncthreads();
        float loc_sum = 0.0f;
        for (unsigned v = tid; v < vocab; v += SAMPLER_BLOCK) {
            loc_sum += expf(val_scratch[v] - s_maxval);
        }
        block_sum_f32(tid, loc_sum, s_red, &s_sume);
        __syncthreads();
        if (tid == 0) s_maxprob = 1.0f / s_sume;
        __syncthreads();
        for (unsigned v = tid; v < vocab; v += SAMPLER_BLOCK) {
            prob_scratch[v] = expf(val_scratch[v] - s_maxval) / s_sume;
        }
        __syncthreads();
    }

    // ---------- Phase C1: top_k — 64-round bisection on the pair key -------
    // tau_pair = max{mid : #{key >= mid} >= top_k}; survivors key >= tau_pair
    // (exactly the top_k largest (value, tid) pairs, ties smallest-tid-first
    // = llm-samplers stable-sort truncate).
    if (tid == 0) {
        if (top_k > 0 && top_k < vocab) { s_lo = 0ull; s_hi = 0xFFFFFFFFFFFFFFFFull; }
        else { s_tau_pair = 0ull; }  // off -> all keys pass
    }
    __syncthreads();
    if (top_k > 0 && top_k < vocab) {
        // vocab < 2^32 guarantees max key < 2^64-1, so F(hi)=0 < top_k: the
        // invariant F(lo) >= top_k, F(hi) < top_k holds at entry.
        for (int round = 0; round < 64; ++round) {
            unsigned long long d = s_hi - s_lo;
            unsigned long long mid = s_lo + (d >> 1) + (d & 1ull);  // upper mid, no overflow
            if (tid == 0) s_mid = mid;
            __syncthreads();
            unsigned long long c = 0ull;
            for (unsigned v = tid; v < vocab; v += SAMPLER_BLOCK) {
                if (pair_key(val_scratch[v], v) >= s_mid) ++c;
            }
            block_count_u64(tid, c, s_cnt, &s_tau_pair);
            __syncthreads();
            if (tid == 0) {
                if (s_tau_pair >= (unsigned long long)top_k) s_lo = s_mid;
                else s_hi = s_mid - 1ull;
            }
            __syncthreads();
        }
        if (tid == 0) s_tau_pair = s_lo;
        __syncthreads();
    }

    // ---------- Phase C2: Z_S + top_p boundary bisection -------------------
    // Z_S = sum of probs over the top_k survivors (renormalization mass of
    // llm-samplers' post-top_k softmax recompute); target T = p * Z_S.
    {
        float loc_z = 0.0f;
        for (unsigned v = tid; v < vocab; v += SAMPLER_BLOCK) {
            if (pair_key(val_scratch[v], v) >= s_tau_pair) loc_z += prob_scratch[v];
        }
        block_sum_f32(tid, loc_z, s_red, &s_z_s);
        __syncthreads();
        if (tid == 0) s_target = s_z_s * top_p;  // top_p in (0,1) when enabled
        __syncthreads();
    }
    if (tid == 0) {
        if (top_p > 0.0f && top_p < 1.0f) { s_flo = 0.0f; s_fhi = s_maxprob; }
        else { s_flo = -FLT_MAX; }  // off -> no threshold
    }
    __syncthreads();
    if (top_p > 0.0f && top_p < 1.0f) {
        // theta* = max{theta : sum_{key>=tau, prob>=theta} prob >= T} in
        // [0, maxprob]; 30 rounds resolve the f32 boundary (F strictly
        // decreases at every distinct prob value, so the bisection lands on
        // the exact boundary value).
        for (int round = 0; round < 30; ++round) {
            float mid = (s_flo + s_fhi) * 0.5f;
            if (tid == 0) s_fmid = mid;
            __syncthreads();
            float loc = 0.0f;
            for (unsigned v = tid; v < vocab; v += SAMPLER_BLOCK) {
                if (pair_key(val_scratch[v], v) >= s_tau_pair && prob_scratch[v] >= s_fmid) {
                    loc += prob_scratch[v];
                }
            }
            block_sum_f32(tid, loc, s_red, &s_target);  // s_target now holds F(mid)
            __syncthreads();
            if (tid == 0) {
                float need = s_z_s * top_p;
                if (s_target >= need) s_flo = s_fmid;
                else s_fhi = s_fmid;
            }
            __syncthreads();
        }
    }

    // ---------- Phase D: gumbel-max over the survivor set ------------------
    {
        float thresh = s_flo;  // -FLT_MAX when top_p off; else theta*
        if (min_p > 0.0f) {
            float mt = s_maxprob * min_p;
            if (mt > thresh) thresh = mt;
        }

        float best_score = -FLT_MAX;
        unsigned best_i = 0;
        bool any_surv_local = false;
        for (unsigned v = tid; v < vocab; v += SAMPLER_BLOCK) {
            float p = prob_scratch[v];
            if (p < thresh) continue;
            if (pair_key(val_scratch[v], v) < s_tau_pair) continue;
            any_surv_local = true;
            float score = val_scratch[v] + gumbel_noise(rng_base, v);
            if (score > best_score || (score == best_score && v > best_i)) {
                best_score = score;
                best_i = v;
            }
        }
        block_pair_max(tid, best_score, best_i, s_v, s_i, &s_pair_v, &s_pair_i);
        __syncthreads();
        unsigned long long c = any_surv_local ? 1ull : 0ull;
        block_count_u64(tid, c, s_cnt, &s_tau_pair);
        __syncthreads();
        if (tid == 0) {
            out[0] = s_pair_i;
            out[1] = 1u;  // TieBreak::LastMax (deterministic larger-tid on equal scores)
            out[2] = (s_any_finite != 0u && s_tau_pair != 0ull) ? 0u : 1u;
        }
    }
}
