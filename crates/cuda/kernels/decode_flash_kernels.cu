// 006-2 T2: flash-style decode attention kernel (S1-5 suspension clause
// triggered — S1-1 profile: decode-step attn segment 14.17 ms/step, 63.4%,
// with kv bandwidth only ~37 us/step, i.e. the naive paged GQA kernel is
// ~3% bandwidth efficiency). This kernel is the JIT tier replacement.
//
// Failure mode analysis (why the old kernel is slow): decode_step_gqa
// assigns one thread per output element i and recomputes the FULL q.k_t dot
// for every token t inside that i-loop — the QK^T work is duplicated d
// times per (b, h) CTA, and the serial i-major structure leaves the CTA
// latency-bound (d x kv_len FMA chain). kv bandwidth (656 tok * 8 heads *
// 128 d * 2 B * 2 K/V ~= 2.7 MB/layer) is ~37 us at 1.4 TB/s, so the
// kernel is compute/latency-bound, not bandwidth-bound.
//
// This kernel: one CTA per (b, q_head), 512 threads, flash-style three
// phases with a single launch per layer:
//   A) QK^T: thread tid computes scores for tokens t = tid, tid+512, ...
//      (fixed stride-512 order, ascending j per dot, two-token ILP
//      interleave with a fixed 4-accumulator pattern); scores in smem.
//   B) softmax: block-wide max (fixed warp butterfly + 16-lane tree),
//      strided exp-sum (same fixed reduction), p[t] = exp(score - max)*inv
//      written back to smem.
//   C) PV: 512 threads split as (output i-pair = 2*(tid % (d/2)), token
//      chunk s = tid / (d/2)) — each thread computes two adjacent outputs
//      from one 4-byte V load (2 f16), ascending t within a chunk,
//      cross-chunk reduction in ascending chunk order via smem. Output rows
//      [b, h, :] — layout identical to decode_step_gqa.
//
// Performance notes (sm_120a, measured): the per-element f16->f32 and
// f32->f16 conversions use the hardware F2F/F2H instructions (exact IEEE
// semantics; the software bit path differed only in NaN payload quieting
// and the f32->half denormal-flush boundary, both outside the D7
// tolerance). The software bit-path conversion was the original kernel's
// dominant ALU cost (~10 instructions/element); 512 threads/CTA + 4-byte
// pair loads in phase C + hardware conversions took the kernel from
// ~138 us to ~53 us at kv_len 646 (Qwen3-0.6B shapes).
//
// Determinism (012 scope): no atomics; every reduction is a fixed tree
// (decode_dot convention: xor butterfly within warps, fixed 16-lane tree
// for the block stage); all per-thread loops use fixed stride/ascending
// orders. Accumulation is fp32 throughout (014 32F-acc judge tier; D7
// table: f16-in/f32-out rel 1e-4 + atol 1e-6 — the residual reorder noise
// vs the serial reference is ~sqrt(kv_len)*2^-24 << 1 fp16 ulp).
//
// Layout contract (identical to decode_gqa_kernels.cu):
//   K region: [total_pages][block_len][kv_heads][d] f16 (row-major)
//   V region: same size immediately following K region
//   page:     [logical_pages] u32 physical page ids (shared K/V table)
//   q:        [B, QH, d] f16 (or f32 — the parity-f32 tier, q already
//             scaled by 1/sqrt(d) upstream)
//   kv_lens:  [B] u32
//   out:      [B, QH, d] f16 (or f32)
//
// identity fast path: when `identity == 1` the page table is the S1-2
// static identity table (page[j] = base + j), so token t's K/V row is
// kv + ((page[0]*block_len + t)*kv_heads + kv_h)*d — fully contiguous
// reads, no per-token page lookups. The page parameter surface is kept
// for future dynamic page tables (`identity == 0`).
//
// Resource contract: dynamic smem = (d + max_kv) * 4 bytes <= 48 KB
// (caller guards); d <= 256 (d divides 256 for the PV split); d even for
// the 4-byte pair loads in phase C.

#include <cuda_fp16.h>
#include <math.h>

// fp16 -> fp32, hardware F2F (exact IEEE; NaN payloads quieted — no NaN
// data on the decode path). fp32 -> fp16 RNE, hardware F2H (only
// difference vs the software bit path: [2^-25, 2^-24) rounds to a
// denormal half instead of zero — outside the D7 atol 1e-6 band).
__device__ __forceinline__ float hbits_to_f32(unsigned short h) {
    return __half2float(*(const __half*)&h);
}
__device__ __forceinline__ unsigned short f32_to_hbits(float f) {
    return (unsigned short)__half_as_ushort(__float2half_rn(f));
}

template <bool F32>
__device__ __forceinline__ float qload(const void* p, int idx) {
    if (F32) {
        return ((const float*)p)[idx];
    }
    return hbits_to_f32(((const unsigned short*)p)[idx]);
}

template <bool F32>
__device__ __forceinline__ void qstore(void* p, int idx, float v) {
    if (F32) {
        ((float*)p)[idx] = v;
    } else {
        ((unsigned short*)p)[idx] = f32_to_hbits(v);
    }
}

// K row of token t (identity fast path vs paged lookup).
__device__ __forceinline__ const unsigned short* krow_ptr(
    const unsigned short* kv, const unsigned int* page, int t, int block_len,
    int kv_h, int kv_heads, int d, int identity, unsigned int base_page) {
    if (identity) {
        return kv + ((size_t)(base_page * block_len + t) * kv_heads + kv_h) * d;
    }
    int lp = t / block_len;
    int off = t % block_len;
    return kv + (((size_t)page[lp] * block_len + off) * kv_heads + kv_h) * d;
}

// V row of token t (identity fast path vs paged lookup); kv_base = V region
// base offset (total_pages * block_len * kv_heads * d).
__device__ __forceinline__ const unsigned short* vrow_ptr(
    const unsigned short* kv, size_t kv_base, const unsigned int* page, int t,
    int block_len, int kv_h, int kv_heads, int d, int identity,
    unsigned int base_page) {
    if (identity) {
        return kv + kv_base + ((size_t)(base_page * block_len + t) * kv_heads + kv_h) * d;
    }
    int lp = t / block_len;
    int off = t % block_len;
    return kv + kv_base + (((size_t)page[lp] * block_len + off) * kv_heads + kv_h) * d;
}

// Fixed 5-step warp butterfly (decode_dot convention).
template <typename C>
__device__ __forceinline__ float warp_reduce(float v, C combine) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v = combine(v, __shfl_xor_sync(0xffffffffu, v, off));
    }
    return v;
}

__device__ __forceinline__ float fmaxf_c(float a, float b) {
    return fmaxf(a, b);
}
__device__ __forceinline__ float faddf_c(float a, float b) {
    return a + b;
}

// 512-thread block reduction: per-warp butterfly, then warp 0 finishes with
// a fixed xor tree over lanes 0..15 (mask 0xffff — only mask members execute
// the shfl). Deterministic order; result broadcast via `slots[0]`.
// Threads whose data does not participate must contribute the neutral
// element (max: -inf, sum: 0.0).
__device__ __forceinline__ float block_reduce_512(
    float v, int tid, float* slots, float (*combine)(float, float)) {
    int lane = tid & 31;
    int warp = tid >> 5;
    v = warp_reduce(v, combine);
    if (lane == 0) {
        slots[warp] = v;
    }
    __syncthreads();
    if (warp == 0 && lane < 16) {
        float w = slots[lane];
#pragma unroll
        for (int off = 8; off > 0; off >>= 1) {
            w = combine(w, __shfl_xor_sync(0xffffu, w, off));
        }
        if (lane == 0) {
            slots[0] = w;
        }
    }
    __syncthreads();
    return slots[0];
}

constexpr int FLASH_TPB = 512;

template <bool F32>
__device__ void decode_flash_impl(
    const void* __restrict__ q,          // [B, QH, d] f16 or f32
    const unsigned int* __restrict__ page,     // [logical_pages]
    const unsigned short* __restrict__ kv,     // K [P][BL][KH][d] then V
    const unsigned int* __restrict__ kv_lens,  // [B]
    void* __restrict__ out,              // [B, QH, d] f16 or f32
    int B, int QH, int d, int block_len, int kv_ratio, int kv_heads,
    int max_kv, int total_pages, int identity) {
    int cta = blockIdx.x;
    if (cta >= B * QH) {
        return;
    }
    int b = cta / QH;
    int h = cta % QH;
    int kv_h = h / kv_ratio;
    int kv_len = (int)kv_lens[b];
    int tid = threadIdx.x;
    if (kv_len <= 0 || kv_len > max_kv) {
        return;  // caller guards; out stays untouched (same as decode_step_gqa)
    }
    const int per_tok_kv = kv_heads * d;

    // Dynamic smem: [0, d) q row f32; [d, d + kv_len) scores.
    extern __shared__ float sm[];
    float* sq = sm;
    float* ss = sm + d;
    // Static smem: 16 warp partial slots for block reductions (also the
    // broadcast slot) + 512 PV partials (2 per thread).
    __shared__ float s_slots[16];
    __shared__ float s_part[FLASH_TPB * 2];

    // Base physical page of this layer's contiguous KV run (identity path).
    unsigned int base_page = identity ? page[0] : 0u;

    // q row -> smem (f32), row offset (b*QH + h)*d.
    {
        const void* qrow =
            (const char*)q + (size_t)(b * QH + h) * d * (F32 ? 4u : 2u);
        for (int i = tid; i < d; i += FLASH_TPB) {
            sq[i] = qload<F32>(qrow, i);
        }
        __syncthreads();
    }

    // Phase A: QK^T — fixed stride-512 token assignment, ascending j per
    // dot; two tokens interleaved for ILP (fixed pattern, no per-j branch).
    // Each per-token dot uses 4 independent j accumulators (d % 4 == 0 by
    // contract) — the serial 128-FMA chain is latency-bound otherwise.
    float m = -1e30f;
    {
        int t = tid;
        while (t + FLASH_TPB < kv_len) {
            const unsigned short* krow = krow_ptr(
                kv, page, t, block_len, kv_h, kv_heads, d, identity, base_page);
            const unsigned short* krow2 = krow_ptr(
                kv, page, t + FLASH_TPB, block_len, kv_h, kv_heads, d, identity,
                base_page);
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
            for (int j = 0; j < d; j += 4) {
                float k0 = hbits_to_f32(krow[j]);
                float k1 = hbits_to_f32(krow[j + 1]);
                float k2 = hbits_to_f32(krow[j + 2]);
                float k3 = hbits_to_f32(krow[j + 3]);
                a0 += sq[j] * k0;
                a1 += sq[j + 1] * k1;
                a2 += sq[j + 2] * k2;
                a3 += sq[j + 3] * k3;
                float m0 = hbits_to_f32(krow2[j]);
                float m1 = hbits_to_f32(krow2[j + 1]);
                float m2 = hbits_to_f32(krow2[j + 2]);
                float m3 = hbits_to_f32(krow2[j + 3]);
                b0 += sq[j] * m0;
                b1 += sq[j + 1] * m1;
                b2 += sq[j + 2] * m2;
                b3 += sq[j + 3] * m3;
            }
            float acc = (a0 + a1) + (a2 + a3);    // fixed tree
            float acc2 = (b0 + b1) + (b2 + b3);
            ss[t] = acc;
            ss[t + FLASH_TPB] = acc2;
            m = fmaxf(m, fmaxf(acc, acc2));
            t += 2 * FLASH_TPB;
        }
        if (t < kv_len) {
            const unsigned short* krow = krow_ptr(
                kv, page, t, block_len, kv_h, kv_heads, d, identity, base_page);
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            for (int j = 0; j < d; j += 4) {
                float k0 = hbits_to_f32(krow[j]);
                float k1 = hbits_to_f32(krow[j + 1]);
                float k2 = hbits_to_f32(krow[j + 2]);
                float k3 = hbits_to_f32(krow[j + 3]);
                a0 += sq[j] * k0;
                a1 += sq[j + 1] * k1;
                a2 += sq[j + 2] * k2;
                a3 += sq[j + 3] * k3;
            }
            float acc = (a0 + a1) + (a2 + a3);
            ss[t] = acc;
            m = fmaxf(m, acc);
        }
    }
    float maxv = block_reduce_512(m, tid, s_slots, &fmaxf_c);

    // Phase B: strided exp-sum (fixed order), inv, p[] write-back.
    float sumv = 0.0f;
    for (int t = tid; t < kv_len; t += FLASH_TPB) {
        sumv += expf(ss[t] - maxv);
    }
    sumv = block_reduce_512(sumv, tid, s_slots, &faddf_c);
    float inv = sumv != 0.0f ? 1.0f / sumv : 0.0f;
    for (int t = tid; t < kv_len; t += FLASH_TPB) {
        ss[t] = expf(ss[t] - maxv) * inv;
    }
    __syncthreads();

    // Phase C: PV — (output i-pair, token chunk) split. Each thread owns
    // outputs (i0, i0+1) with i0 = 2 * (tid % ng), ng = d/2 i-groups, and
    // token chunk s = tid / ng (FLASH_TPB/ng chunks; d even by contract),
    // ascending t within a chunk, cross-chunk reduction in ascending chunk
    // order. Two adjacent f16 V elements are loaded as one 4-byte word
    // (halves the load count).
    {
        const size_t kv_base = (size_t)total_pages * block_len * per_tok_kv;
        const int ng = d / 2;            // i-pair groups per row
        const int nc = FLASH_TPB / ng;   // token chunks per output pair
        const int seg = (kv_len + nc - 1) / nc;
        const int i0 = 2 * (tid % ng);
        const int s = tid / ng;
        float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
        float c4 = 0.0f, c5 = 0.0f, c6 = 0.0f, c7 = 0.0f;
        int t = s * seg;
        int t_end = (s + 1) * seg < kv_len ? (s + 1) * seg : kv_len;
        // 4 independent t chains (fixed t += 4 order) x 2 outputs.
        for (; t + 3 < t_end; t += 4) {
            const unsigned int* v0 = (const unsigned int*)vrow_ptr(
                kv, kv_base, page, t, block_len, kv_h, kv_heads, d, identity,
                base_page);
            const unsigned int* v1 = (const unsigned int*)vrow_ptr(
                kv, kv_base, page, t + 1, block_len, kv_h, kv_heads, d,
                identity, base_page);
            const unsigned int* v2 = (const unsigned int*)vrow_ptr(
                kv, kv_base, page, t + 2, block_len, kv_h, kv_heads, d,
                identity, base_page);
            const unsigned int* v3 = (const unsigned int*)vrow_ptr(
                kv, kv_base, page, t + 3, block_len, kv_h, kv_heads, d,
                identity, base_page);
            unsigned int w0 = v0[i0 / 2];
            unsigned int w1 = v1[i0 / 2];
            unsigned int w2 = v2[i0 / 2];
            unsigned int w3 = v3[i0 / 2];
            c0 += ss[t] * hbits_to_f32((unsigned short)(w0 & 0xffffu));
            c1 += ss[t] * hbits_to_f32((unsigned short)(w0 >> 16));
            c2 += ss[t + 1] * hbits_to_f32((unsigned short)(w1 & 0xffffu));
            c3 += ss[t + 1] * hbits_to_f32((unsigned short)(w1 >> 16));
            c4 += ss[t + 2] * hbits_to_f32((unsigned short)(w2 & 0xffffu));
            c5 += ss[t + 2] * hbits_to_f32((unsigned short)(w2 >> 16));
            c6 += ss[t + 3] * hbits_to_f32((unsigned short)(w3 & 0xffffu));
            c7 += ss[t + 3] * hbits_to_f32((unsigned short)(w3 >> 16));
        }
        for (; t < t_end; ++t) {
            const unsigned int* v = (const unsigned int*)vrow_ptr(
                kv, kv_base, page, t, block_len, kv_h, kv_heads, d, identity,
                base_page);
            unsigned int w = v[i0 / 2];
            c0 += ss[t] * hbits_to_f32((unsigned short)(w & 0xffffu));
            c1 += ss[t] * hbits_to_f32((unsigned short)(w >> 16));
        }
        s_part[tid * 2] = (c0 + c2) + (c4 + c6);      // fixed tree
        s_part[tid * 2 + 1] = (c1 + c3) + (c5 + c7);
        __syncthreads();
        if (s == 0) {
            float o0 = s_part[tid * 2];
            float o1 = s_part[tid * 2 + 1];
            for (int k = 1; k < nc; ++k) {
                o0 += s_part[(tid + k * ng) * 2];
                o1 += s_part[(tid + k * ng) * 2 + 1];
            }
            qstore<F32>(out, (size_t)(b * QH + h) * d + i0, o0);
            qstore<F32>(out, (size_t)(b * QH + h) * d + i0 + 1, o1);
        }
    }
}

extern "C" __global__ void decode_step_gqa_flash(
    const void* __restrict__ q, const unsigned int* __restrict__ page,
    const unsigned short* __restrict__ kv, const unsigned int* __restrict__ kv_lens,
    void* __restrict__ out, int B, int QH, int d, int block_len, int kv_ratio,
    int kv_heads, int max_kv, int total_pages, int identity) {
    decode_flash_impl<false>(q, page, kv, kv_lens, out, B, QH, d, block_len,
                             kv_ratio, kv_heads, max_kv, total_pages, identity);
}

extern "C" __global__ void decode_step_gqa_flash_f32(
    const void* __restrict__ q, const unsigned int* __restrict__ page,
    const unsigned short* __restrict__ kv, const unsigned int* __restrict__ kv_lens,
    void* __restrict__ out, int B, int QH, int d, int block_len, int kv_ratio,
    int kv_heads, int max_kv, int total_pages, int identity) {
    decode_flash_impl<true>(q, page, kv, kv_lens, out, B, QH, d, block_len,
                            kv_ratio, kv_heads, max_kv, total_pages, identity);
}

// ---------------------------------------------------------------------------
// S2-B+: batched flash decode — the whole batch in ONE launch (grid =
// B*QH), replacing the per-request loop of B launches. Each (b, h) CTA
// resolves request b's own page row and pool base from the batch tables
// instead of receiving a per-request pointer:
//   pages:    [B][n_layer][pp] u32 identity page tables (row (b, li) at
//             pages[(b*n_layer + li)*pp ..]; page[0] = the layer's first
//             physical page — the identity fast path, as in the engine's
//             batch scratch);
//   kv:       [B] pool K-region bases — the shared-pool call has uniform
//             entries; per-request-pool calls (the S2-B A3 shape) index
//             each request's own segment;
//   kv_lens:  [B] u32 per-request attention windows.
// The per-(b, h) arithmetic (phases A/B/C, q row -> smem, out row write)
// is verbatim `decode_flash_impl<false>` — bit-identical per request to
// the single-request flash, so the batch step's attention matches the
// single path bitwise. pp is derived as max_kv / block_len (the engine
// guarantees pp * block_len == max_kv).
// ---------------------------------------------------------------------------
extern "C" __global__ void decode_step_gqa_flash_batch(
    const void* __restrict__ q,               // [B, QH, d] f16
    const unsigned int* __restrict__ pages,   // [B][n_layer][pp] identity tables
    const unsigned short* const* __restrict__ kv,      // [B] pool K-region bases
    const unsigned int* __restrict__ kv_lens, // [B]
    void* __restrict__ out,                   // [B, QH, d] f16
    int B, int QH, int d, int block_len, int kv_ratio, int kv_heads,
    int max_kv, int total_pages, int n_layer, int li, int identity) {
    int cta = blockIdx.x;
    if (cta >= B * QH) {
        return;
    }
    int b = cta / QH;
    int h = cta % QH;
    int pp = max_kv / block_len;
    const unsigned int* page = pages + ((size_t)b * n_layer + li) * pp;
    decode_flash_impl<false>(q, page, kv[b], kv_lens, out, B, QH, d, block_len,
                             kv_ratio, kv_heads, max_kv, total_pages, identity);
}

// ---------------------------------------------------------------------------
// S1-9 fused decode: decode_step_gqa_flash_fused — the flash attention
// kernel with the kv-cache write fused in (f16 tier only; grid = B*QH
// blocks of FLASH_TPB threads, same as the split flash):
//   - kv-cache write of the current step's slot (k16/v16 -> the K/V
//     regions at (phys, off)) — redundant idempotent writes by every
//     block, ordered by a block-local __syncthreads() before the
//     attention reads (byte-identical to kv_write_row's copy; replaces
//     the separate kv_write launch);
// The o-projection phase-1 is NOT folded in: it reads the whole
// attention row (all B*QH heads' outputs), which the block's own phase C
// writes only for its own head — cross-block reads would race. It stays
// a separate launch (the o plan row in the multi phase-1 kernel) after
// this kernel completes.
// The attention arithmetic (phases A/B/C) is verbatim identical to
// decode_flash_impl<false>.
// ---------------------------------------------------------------------------
extern "C" __global__ void decode_step_gqa_flash_fused(
    const void* __restrict__ q, const unsigned int* __restrict__ page,
    const unsigned short* __restrict__ kv, const unsigned int* __restrict__ kv_lens,
    void* __restrict__ out, int B, int QH, int d, int block_len, int kv_ratio,
    int kv_heads, int max_kv, int total_pages, int identity,
    const unsigned short* __restrict__ k16, const unsigned short* __restrict__ v16) {
    int cta = blockIdx.x;
    if (cta >= B * QH) {
        return;
    }
    int b = cta / QH;
    int h = cta % QH;
    int kv_h = h / kv_ratio;
    int kv_len = (int)kv_lens[b];
    int tid = threadIdx.x;
    if (kv_len <= 0 || kv_len > max_kv) {
        return;  // same guard as decode_step_gqa_flash
    }
    const int per_tok_kv = kv_heads * d;

    // 1. Current step's kv slot: k16/v16 -> kv K/V regions (identical
    //    bytes to kv_write_row; every block writes the same values —
    //    idempotent redundant writes, no cross-block ordering needed).
    {
        int lp = (kv_len - 1) / block_len;
        int off = (kv_len - 1) % block_len;
        unsigned int phys = identity ? page[0] + lp : page[lp];
        const size_t slot = ((size_t)phys * block_len + off) * per_tok_kv;
        const size_t v_base = (size_t)total_pages * block_len * per_tok_kv;
        for (int c = tid; c < per_tok_kv; c += FLASH_TPB) {
            ((unsigned short*)kv)[slot + c] = k16[c];
            ((unsigned short*)kv)[v_base + slot + c] = v16[c];
        }
        __syncthreads();
    }

    // Dynamic smem: [0, d) q row f32; [d, d + kv_len) scores.
    extern __shared__ float sm[];
    float* sq = sm;
    float* ss = sm + d;
    // Static smem: 16 warp partial slots for block reductions (also the
    // broadcast slot) + 512 PV partials (2 per thread).
    __shared__ float s_slots[16];
    __shared__ float s_part[FLASH_TPB * 2];

    // Base physical page of this layer's contiguous KV run (identity path).
    unsigned int base_page = identity ? page[0] : 0u;

    // q row -> smem (f32), row offset (b*QH + h)*d.
    {
        const void* qrow =
            (const char*)q + (size_t)(b * QH + h) * d * 2u;
        for (int i = tid; i < d; i += FLASH_TPB) {
            sq[i] = hbits_to_f32(((const unsigned short*)qrow)[i]);
        }
        __syncthreads();
    }

    // Phase A: QK^T — fixed stride-512 token assignment, ascending j per
    // dot; two tokens interleaved for ILP (fixed pattern, no per-j branch).
    // Each per-token dot uses 4 independent j accumulators (d % 4 == 0 by
    // contract) — the serial 128-FMA chain is latency-bound otherwise.
    float m = -1e30f;
    {
        int t = tid;
        while (t + FLASH_TPB < kv_len) {
            const unsigned short* krow = krow_ptr(
                kv, page, t, block_len, kv_h, kv_heads, d, identity, base_page);
            const unsigned short* krow2 = krow_ptr(
                kv, page, t + FLASH_TPB, block_len, kv_h, kv_heads, d, identity,
                base_page);
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
            for (int j = 0; j < d; j += 4) {
                float k0 = hbits_to_f32(krow[j]);
                float k1 = hbits_to_f32(krow[j + 1]);
                float k2 = hbits_to_f32(krow[j + 2]);
                float k3 = hbits_to_f32(krow[j + 3]);
                a0 += sq[j] * k0;
                a1 += sq[j + 1] * k1;
                a2 += sq[j + 2] * k2;
                a3 += sq[j + 3] * k3;
                float m0 = hbits_to_f32(krow2[j]);
                float m1 = hbits_to_f32(krow2[j + 1]);
                float m2 = hbits_to_f32(krow2[j + 2]);
                float m3 = hbits_to_f32(krow2[j + 3]);
                b0 += sq[j] * m0;
                b1 += sq[j + 1] * m1;
                b2 += sq[j + 2] * m2;
                b3 += sq[j + 3] * m3;
            }
            float acc = (a0 + a1) + (a2 + a3);    // fixed tree
            float acc2 = (b0 + b1) + (b2 + b3);
            ss[t] = acc;
            ss[t + FLASH_TPB] = acc2;
            m = fmaxf(m, fmaxf(acc, acc2));
            t += 2 * FLASH_TPB;
        }
        if (t < kv_len) {
            const unsigned short* krow = krow_ptr(
                kv, page, t, block_len, kv_h, kv_heads, d, identity, base_page);
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            for (int j = 0; j < d; j += 4) {
                float k0 = hbits_to_f32(krow[j]);
                float k1 = hbits_to_f32(krow[j + 1]);
                float k2 = hbits_to_f32(krow[j + 2]);
                float k3 = hbits_to_f32(krow[j + 3]);
                a0 += sq[j] * k0;
                a1 += sq[j + 1] * k1;
                a2 += sq[j + 2] * k2;
                a3 += sq[j + 3] * k3;
            }
            float acc = (a0 + a1) + (a2 + a3);
            ss[t] = acc;
            m = fmaxf(m, acc);
        }
    }
    float maxv = block_reduce_512(m, tid, s_slots, &fmaxf_c);

    // Phase B: strided exp-sum (fixed order), inv, p[] write-back.
    float sumv = 0.0f;
    for (int t = tid; t < kv_len; t += FLASH_TPB) {
        sumv += expf(ss[t] - maxv);
    }
    sumv = block_reduce_512(sumv, tid, s_slots, &faddf_c);
    float inv = sumv != 0.0f ? 1.0f / sumv : 0.0f;
    for (int t = tid; t < kv_len; t += FLASH_TPB) {
        ss[t] = expf(ss[t] - maxv) * inv;
    }
    __syncthreads();

    // Phase C: PV — (output i-pair, token chunk) split. Each thread owns
    // outputs (i0, i0+1) with i0 = 2 * (tid % ng), ng = d/2 i-groups, and
    // token chunk s = tid / ng (FLASH_TPB/ng chunks; d even by contract),
    // ascending t within a chunk, cross-chunk reduction in ascending chunk
    // order. Two adjacent f16 V elements are loaded as one 4-byte word
    // (halves the load count).
    {
        const size_t kv_base = (size_t)total_pages * block_len * per_tok_kv;
        const int ng = d / 2;            // i-pair groups per row
        const int nc = FLASH_TPB / ng;   // token chunks per output pair
        const int seg = (kv_len + nc - 1) / nc;
        const int i0 = 2 * (tid % ng);
        const int s = tid / ng;
        float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
        float c4 = 0.0f, c5 = 0.0f, c6 = 0.0f, c7 = 0.0f;
        int t = s * seg;
        int t_end = (s + 1) * seg < kv_len ? (s + 1) * seg : kv_len;
        // 4 independent t chains (fixed t += 4 order) x 2 outputs.
        for (; t + 3 < t_end; t += 4) {
            const unsigned int* v0 = (const unsigned int*)vrow_ptr(
                kv, kv_base, page, t, block_len, kv_h, kv_heads, d, identity,
                base_page);
            const unsigned int* v1 = (const unsigned int*)vrow_ptr(
                kv, kv_base, page, t + 1, block_len, kv_h, kv_heads, d,
                identity, base_page);
            const unsigned int* v2 = (const unsigned int*)vrow_ptr(
                kv, kv_base, page, t + 2, block_len, kv_h, kv_heads, d,
                identity, base_page);
            const unsigned int* v3 = (const unsigned int*)vrow_ptr(
                kv, kv_base, page, t + 3, block_len, kv_h, kv_heads, d,
                identity, base_page);
            unsigned int w0 = v0[i0 / 2];
            unsigned int w1 = v1[i0 / 2];
            unsigned int w2 = v2[i0 / 2];
            unsigned int w3 = v3[i0 / 2];
            c0 += ss[t] * hbits_to_f32((unsigned short)(w0 & 0xffffu));
            c1 += ss[t] * hbits_to_f32((unsigned short)(w0 >> 16));
            c2 += ss[t + 1] * hbits_to_f32((unsigned short)(w1 & 0xffffu));
            c3 += ss[t + 1] * hbits_to_f32((unsigned short)(w1 >> 16));
            c4 += ss[t + 2] * hbits_to_f32((unsigned short)(w2 & 0xffffu));
            c5 += ss[t + 2] * hbits_to_f32((unsigned short)(w2 >> 16));
            c6 += ss[t + 3] * hbits_to_f32((unsigned short)(w3 & 0xffffu));
            c7 += ss[t + 3] * hbits_to_f32((unsigned short)(w3 >> 16));
        }
        for (; t < t_end; ++t) {
            const unsigned int* v = (const unsigned int*)vrow_ptr(
                kv, kv_base, page, t, block_len, kv_h, kv_heads, d, identity,
                base_page);
            unsigned int w = v[i0 / 2];
            c0 += ss[t] * hbits_to_f32((unsigned short)(w & 0xffffu));
            c1 += ss[t] * hbits_to_f32((unsigned short)(w >> 16));
        }
        s_part[tid * 2] = (c0 + c2) + (c4 + c6);      // fixed tree
        s_part[tid * 2 + 1] = (c1 + c3) + (c5 + c7);
        __syncthreads();
        if (s == 0) {
            float o0 = s_part[tid * 2];
            float o1 = s_part[tid * 2 + 1];
            for (int k = 1; k < nc; ++k) {
                o0 += s_part[(tid + k * ng) * 2];
                o1 += s_part[(tid + k * ng) * 2 + 1];
            }
            ((unsigned short*)out)[(size_t)(b * QH + h) * d + i0] = f32_to_hbits(o0);
            ((unsigned short*)out)[(size_t)(b * QH + h) * d + i0 + 1] = f32_to_hbits(o1);
        }
    }
    __syncthreads();
}
