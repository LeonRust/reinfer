// S1-9 decode-step kernel fusion: the per-layer 27-node launch sequence is
// replaced by 8 nodes (228 total for Qwen3-0.6B, down from 760) by merging
// the two-phase gemv launches and the small dense kernels into one fused
// kernel per group. Every fused kernel is bit-identical to the split
// sequence it replaces: the per-column fixed accumulation orders (jgemm's
// ascending-slab phase-2 sums, the per-thread strided rms sums with the
// 128/256-tree, the f16 round/widen/round chains) are preserved exactly,
// and the f16 conversions use the identical software RNE bit construction
// as dense_kernels.cu / diff_kernels.cu.
//
// Fusion groups (per layer):
//   1. gemv_m1_f16f32_multi     — phase-1 of the m=1 plans in four launches
//                                 via a device PlanRow table: p1_qkv (q,k,v
//                                 at layer start), p1_o (o after the flash —
//                                 its own node: it reads the full attention
//                                 row, which the flash blocks write only per
//                                 head), p1_gu (g,u after p2_o). 496 blocks
//                                 for Qwen3-0.6B instead of 14 phase-1
//                                 launches (the down phase-1 is folded into
//                                 group 3 — block-local there).
//   2. gemv_p2_qkv_cast_hn_rope — phase-2 reductions of q/k/v + f16 casts +
//                                 q/k head-norm + RoPE in one kernel
//                                 (replaces 3 phase-2 + 3 casts + 2 head-norm
//                                 + 2 rope launches).
//   3. gemv_p2_gu_p1d_swiglu    — gate/up phase-2 reductions + the fused
//                                 cast-SiLU-GLU + the down plan's phase-1 in
//                                 one kernel (the down plan's full phase-1
//                                 tile grid): each block redundantly writes
//                                 the down activation stripe its phase-2
//                                 k-range lies in, then computes the down
//                                 tile (bx/nslabs_d, bx%nslabs_d) (valid
//                                 iff nslabs_d == 2*ncols_g && slab_k_d <=
//                                 128 — build_plans gate; block-local, no
//                                 cross-block reads).
//   4. gemv_p2_add_rms          — phase-2 reduction + residual add
//                                 (exact add_cast semantics) + RMSNorm row;
//                                 used for the o projection (residual into x,
//                                 norm -> xn, ffn_norm) and the down
//                                 projection (residual into x, norm -> xn,
//                                 next layer's attn_norm).
// kv write is fused into decode_step_gqa_flash_fused (decode_flash_kernels.cu,
// block-local idempotent writes).
//
// Bit-identity contracts (each against the split sequence):
//   - phase-1 per-(col, slab): identical arithmetic to gemv_m1_f16f32 —
//     four ILP accumulators over stride-4 k, fixed (acc0+acc1)+(acc2+acc3);
//   - phase-2 per column: ascending-slab sum (identical to
//     gemv_m1_f16f32_reduce);
//   - casts: f32_to_hbits RNE bit construction (identical to diff_kernels.cu
//     cast_f32_to_f16);
//   - head-norm: per-head squares + tree over 128 slots starting at st=64.
//     The split rms_norm_heads_f16 tree starts at st=128, but its st=128
//     step (and, for d < 128, its st=64 step) only adds +0.0 pads, and
//     a + 0.0 == a bit-wise for every finite a (squares are never -0.0), so
//     skipping the no-op steps is bit-identical;
//   - rope: identical theta/rotation code to rope_heads_f16 (same
//     f16-round/widen/scale/round chain);
//   - residual add: x[i] = f16(f32(x[i]) + f16(c)) — same rounding order as
//     add_cast_f16; the rms passes then read the widened stored f16 bits in
//     the same per-thread strided order as rms_norm_row_f16;
//   - swiglu: g = widen(f16(gate)), u = widen(f16(up)), silu = g/(1+exp(-g)),
//     out = f16(silu * u) — the fused_cast_swiglu_f16 expression.
//
// Host side: crates/cuda/src/fused.rs (loader, plan table, launches);
// engine wiring: crates/cuda/src/engine.rs (REINFER_FUSED gate, fused
// step launch + GraphStepDecl::build_fused).

#include <cuda_fp16.h>  // __half / __half2float (same as gemm_m1.cu)

// Host-mirrored plan row (40 bytes, 8-aligned) — one per gemv plan. The
// multi-kernel decodes its (col, slab) tile from the row, so every
// per-(col, slab) computation is identical to the single-plan kernel.
struct PlanRow {
    const __half* a;        // [k] f16 activation row
    const __half* b;        // [k x n] f16 row-major weight matrix
    float* partials;        // [nslabs x n] s-major slab partials
                            // (partials[slab*n + col] — one 128B line per
                            // warp per slab; layout shared with gemm_m1.cu)
    int n;                  // output columns
    int k;                  // reduction length
    int nslabs;             // k slabs (phase-1 grid split)
    int col_off;            // linearized (ncols*nslabs) block offset of this
                            // plan's segment in the multi-kernel grid
};

__device__ __forceinline__ float hbits_to_f32(unsigned short h) {
    // Identical software widening to dense_kernels.cu (same bit construction).
    unsigned int sign = (unsigned int)(h >> 15) << 31;
    unsigned int exp  = (h >> 10) & 0x1f;
    unsigned int man  = h & 0x03ff;
    if (exp == 0) {
        if (man == 0) {
            return __uint_as_float(sign);
        }
        unsigned int m = man;
        unsigned int s = 0;
        while ((m & 0x0400) == 0) {
            m <<= 1;
            s += 1;
        }
        return __uint_as_float(sign | ((113 - s) << 23) | ((m & 0x03ff) << 13));
    }
    if (exp == 0x1f) {
        return __uint_as_float(sign | 0x7f800000u | (man << 13));
    }
    return __uint_as_float(sign | ((exp + 112) << 23) | (man << 13));
}

__device__ __forceinline__ unsigned short f32_to_hbits(float f) {
    // Identical RNE bit construction to dense_kernels.cu / diff_kernels.cu
    // cast_f32_to_f16 (bit-level identity of every fused cast depends on it).
    unsigned int bits = __float_as_uint(f);
    unsigned int sign = (bits >> 16) & 0x8000u;
    int exp = (int)((bits >> 23) & 0xff);
    unsigned int man = bits & 0x7fffffu;
    if (exp == 0xff) {
        return (unsigned short)(sign | 0x7c00u | ((man >> 13) & 0x3ffu));
    }
    int half_exp = exp - 127 + 15;
    if (half_exp <= 0) {
        if (half_exp < -10) {
            return (unsigned short)sign;
        }
        unsigned int subm = (man | 0x800000u) >> (1 - half_exp + 13);
        return (unsigned short)(sign | subm);
    }
    if (half_exp >= 31) {
        return (unsigned short)(sign | 0x7c00u);
    }
    return (unsigned short)(sign | ((unsigned int)half_exp << 10) | (man >> 13));
}

// ---------------------------------------------------------------------------
// Group 1: phase-1 of the seven m=1 plans in one launch.
// grid = sum over plans of (ncols*nslabs) (672 for Qwen3-0.6B), block = 256.
// The plan row is decoded from the (bx - col_off) tile offset; the per-tile
// arithmetic is byte-for-byte the gemm_m1.cu phase-1 body, so the partials
// are bit-identical to the split phase-1 launches.
// ---------------------------------------------------------------------------
// Phase-1 accumulation for one (col, slab) tile: four ILP accumulators over
// stride-4 k, fixed (acc0+acc1)+(acc2+acc3) tree — identical to
// gemm_m1_f16f32. When the slab's k-range is a multiple of 4 (every real
// plan: slab_k divides k), the per-position guards are all no-ops and the
// unguarded unrolled loop lets ptxas software-pipeline the B loads deeper
// (more loads in flight — the decode FFN kernels are DRAM-latency-bound,
// measured: fewer registers/guards make them slower, not faster).
__device__ __forceinline__ void gemv_phase1(const PlanRow& row, int col,
                                            int ks, int ke, int slab) {
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    if ((ke - ks) % 4 == 0) {
        #pragma unroll 8
        for (int k0 = ks; k0 < ke; k0 += 4) {
            acc0 += __half2float(__ldg(&row.a[k0])) *
                    __half2float(__ldg(&row.b[(size_t)k0 * row.n + col]));
            acc1 += __half2float(__ldg(&row.a[k0 + 1])) *
                    __half2float(__ldg(&row.b[(size_t)(k0 + 1) * row.n + col]));
            acc2 += __half2float(__ldg(&row.a[k0 + 2])) *
                    __half2float(__ldg(&row.b[(size_t)(k0 + 2) * row.n + col]));
            acc3 += __half2float(__ldg(&row.a[k0 + 3])) *
                    __half2float(__ldg(&row.b[(size_t)(k0 + 3) * row.n + col]));
        }
    } else {
        // General k >= 1 contract: per-position guards keep every tail
        // exact (identical order, identical values — the fast path above
        // is bit-identical to this loop for slab lengths divisible by 4).
        for (int k0 = ks; k0 < ke; k0 += 4) {
            acc0 += __half2float(__ldg(&row.a[k0])) *
                    __half2float(__ldg(&row.b[(size_t)k0 * row.n + col]));
            if (k0 + 1 < ke) {
                acc1 += __half2float(__ldg(&row.a[k0 + 1])) *
                        __half2float(__ldg(&row.b[(size_t)(k0 + 1) * row.n + col]));
            }
            if (k0 + 2 < ke) {
                acc2 += __half2float(__ldg(&row.a[k0 + 2])) *
                        __half2float(__ldg(&row.b[(size_t)(k0 + 2) * row.n + col]));
            }
            if (k0 + 3 < ke) {
                acc3 += __half2float(__ldg(&row.a[k0 + 3])) *
                        __half2float(__ldg(&row.b[(size_t)(k0 + 3) * row.n + col]));
            }
        }
    }
    float* dst = row.partials + (size_t)slab * row.n + col;
    *dst = (acc0 + acc1) + (acc2 + acc3);
}

extern "C" __global__ void __launch_bounds__(256, 2) gemv_m1_f16f32_multi(const PlanRow* __restrict__ plans,
                                                int nplans) {
    int bx = blockIdx.x;
    // Linear scan (nplans <= 8; the decode is per-layer 7 + lm 1).
    int pl = 0;
    while (pl + 1 < nplans && bx >= plans[pl + 1].col_off) {
        ++pl;
    }
    const PlanRow row = plans[pl];
    int local = bx - row.col_off;
    int col = (local / row.nslabs) * 256 + threadIdx.x;
    if (col >= row.n) {
        return;
    }
    int slab = local % row.nslabs;
    int slab_k = (row.k + row.nslabs - 1) / row.nslabs;  // ceil; last slab shorter
    int ks = slab * slab_k;
    int ke = ks + slab_k;
    if (ke > row.k) {
        ke = row.k;
    }
    gemv_phase1(row, col, ks, ke, slab);
}

// ---------------------------------------------------------------------------
// Group 2: q/k/v phase-2 reductions + f16 casts + q/k head-norm + RoPE.
// grid = ceil(nq/256) + ceil(nk/256) + ceil(nv/256) (16 for Qwen3-0.6B),
// block = 256. Precondition (load-time gate): 256 % d == 0, d >= 32 — every
// block covers exactly 256/d complete heads, so each head's RMSNorm runs
// entirely inside one block. Per-block head regions of 128 smem slots:
// [0..d) hold the head's element squares, [d..128) are +0.0 pads, and the
// tree runs st = 64..1 — bit-identical to rms_norm_heads_f16's full 256-slot
// tree with its no-op +0.0 steps (a + 0.0 == a for all finite a).
// RoPE reads the block's cast (or head-normed) q16/k16 — a separate launch
// in the split path, so a __syncthreads() separates the write phases here.
// ---------------------------------------------------------------------------
extern "C" __global__ void gemv_p2_qkv_cast_hn_rope(
    const float* __restrict__ pq,     // q phase-2 partials [nslabs_q * nq] s-major
    const float* __restrict__ pk,     // k phase-2 partials [nslabs_k * nk] s-major
    const float* __restrict__ pv,     // v phase-2 partials [nslabs_v * nv] s-major
    unsigned short* __restrict__ q16,
    unsigned short* __restrict__ k16,
    unsigned short* __restrict__ v16,
    const unsigned short* __restrict__ wq,  // q head-norm weights [d]; null if !hn
    const unsigned short* __restrict__ wk,  // k head-norm weights [d]; null if !hn
    int nq, int nk, int nv,
    int nslabs_q, int nslabs_k, int nslabs_v,
    int d, int half, int pos,
    float eta, float scale_q, float scale_k,
    float eps, int hn) {
    int bx = blockIdx.x;
    int tid = threadIdx.x;
    // Max head regions: (256/d) heads * 128 slots <= 1024 (d >= 32 gate).
    __shared__ float s_sh[1024];
    int cq = (nq + 255) / 256;
    if (bx < cq) {
        // --- q segment: reduce -> cast -> (hn) -> rope ---------------------
        int col = bx * 256 + tid;
        if (col < nq) {
            float acc = 0.0f;
            for (int s = 0; s < nslabs_q; ++s) {
                acc += pq[(size_t)s * nq + col];
            }
            q16[col] = f32_to_hbits(acc);
        }
        if (hn) {
            int head = tid / d;
            int e = tid % d;
            float v = (col < nq) ? hbits_to_f32(q16[col]) : 0.0f;
            s_sh[head * 128 + e] = v * v;
            __syncthreads();
            // Tree over the 128-slot region, st = 64..1 (the split's st=128
            // step and, for d < 128, its st=64 step add only +0.0 pads).
            for (int st = 64; st > 0; st >>= 1) {
                if (e < st) {
                    s_sh[head * 128 + e] += s_sh[head * 128 + e + st];
                }
                __syncthreads();
            }
            if (tid < 256 / d) {
                s_sh[tid * 128] = rsqrtf(s_sh[tid * 128] / (float)d + eps);
            }
            __syncthreads();
            if (col < nq) {
                float rstd = s_sh[(tid / d) * 128];
                q16[col] = f32_to_hbits(hbits_to_f32(q16[col]) * rstd * hbits_to_f32(wq[tid % d]));
            }
            __syncthreads();
        } else {
            __syncthreads();
        }
        // RoPE (q scale): identical rotation/rounding to rope_heads_f16.
        if (col < nq) {
            int e = tid % d;
            if (e < half) {
                int base = col - e;
                unsigned short* xr = q16 + base;
                float theta = (float)pos * powf(eta, -2.f * (float)e / (2.f * (float)half));
                float c = cosf(theta), s = sinf(theta);
                float a = hbits_to_f32(xr[e]);
                float b = hbits_to_f32(xr[e + half]);
                float v1 = a * c - b * s;
                float v2 = a * s + b * c;
                xr[e] = f32_to_hbits(hbits_to_f32(f32_to_hbits(v1)) * scale_q);
                xr[e + half] = f32_to_hbits(hbits_to_f32(f32_to_hbits(v2)) * scale_q);
            }
        }
        return;
    }
    int ck = (nk + 255) / 256;
    if (bx < cq + ck) {
        // --- k segment: same pipeline, scale_k, no rotation is skipped -----
        int lb = bx - cq;
        int col = lb * 256 + tid;
        if (col < nk) {
            float acc = 0.0f;
            for (int s = 0; s < nslabs_k; ++s) {
                acc += pk[(size_t)s * nk + col];
            }
            k16[col] = f32_to_hbits(acc);
        }
        if (hn) {
            int head = tid / d;
            int e = tid % d;
            float v = (col < nk) ? hbits_to_f32(k16[col]) : 0.0f;
            s_sh[head * 128 + e] = v * v;
            __syncthreads();
            for (int st = 64; st > 0; st >>= 1) {
                if (e < st) {
                    s_sh[head * 128 + e] += s_sh[head * 128 + e + st];
                }
                __syncthreads();
            }
            if (tid < 256 / d) {
                s_sh[tid * 128] = rsqrtf(s_sh[tid * 128] / (float)d + eps);
            }
            __syncthreads();
            if (col < nk) {
                float rstd = s_sh[(tid / d) * 128];
                k16[col] = f32_to_hbits(hbits_to_f32(k16[col]) * rstd * hbits_to_f32(wk[tid % d]));
            }
            __syncthreads();
        } else {
            __syncthreads();
        }
        if (col < nk) {
            int e = tid % d;
            if (e < half) {
                int base = col - e;
                unsigned short* xr = k16 + base;
                float theta = (float)pos * powf(eta, -2.f * (float)e / (2.f * (float)half));
                float c = cosf(theta), s = sinf(theta);
                float a = hbits_to_f32(xr[e]);
                float b = hbits_to_f32(xr[e + half]);
                float v1 = a * c - b * s;
                float v2 = a * s + b * c;
                xr[e] = f32_to_hbits(hbits_to_f32(f32_to_hbits(v1)) * scale_k);
                xr[e + half] = f32_to_hbits(hbits_to_f32(f32_to_hbits(v2)) * scale_k);
            }
        }
        return;
    }
    // --- v segment: reduce + cast only -------------------------------------
    int lb = bx - (cq + ck);
    int col = lb * 256 + tid;
    if (col < nv) {
        float acc = 0.0f;
        for (int s = 0; s < nslabs_v; ++s) {
            acc += pv[(size_t)s * nv + col];
        }
        v16[col] = f32_to_hbits(acc);
    }
}

// ---------------------------------------------------------------------------
// Group 3: gate/up phase-2 reductions + fused cast-SiLU-GLU.
// grid = ceil(ffn/256), block = 256. Each thread reduces its gate and up
// columns (identical ascending-slab orders to the split phase-2 kernels,
// interleaved per thread — each accumulator's order is unchanged), casts
// both to f16 via the identical RNE construction, and computes the
// fused_cast_swiglu_f16 expression: silu = g/(1+exp(-g)), out = f16(silu*u).
// ---------------------------------------------------------------------------
extern "C" __global__ void gemv_p2_gu_swiglu(const float* __restrict__ pg,
                                             const float* __restrict__ pu,
                                             unsigned short* __restrict__ down,
                                             int n, int nslabs) {
    int col = blockIdx.x * 256 + threadIdx.x;
    if (col >= n) {
        return;
    }
    float gacc = 0.0f, uacc = 0.0f;
    for (int s = 0; s < nslabs; ++s) {
        gacc += pg[(size_t)s * n + col];
        uacc += pu[(size_t)s * n + col];
    }
    float gv = hbits_to_f32(f32_to_hbits(gacc));
    float uv = hbits_to_f32(f32_to_hbits(uacc));
    float silu = gv / (1.f + expf(-gv));
    down[col] = f32_to_hbits(silu * uv);
}

// ---------------------------------------------------------------------------
// Group 4: phase-2 reduction + residual add + RMSNorm row.
// grid = 1, block = 256, n <= 1024. Serves the o projection (partials of
// the o plan, x += o, rms over the post-add x into xn with ffn_norm) and
// the down projection (partials of the down plan, x += down, rms into xn
// with the next layer's attn_norm / final_norm) — the identical computation
// in both slots of the layer.
// Bit-identity: the add is add_cast_f16's expression (round the addend to
// f16, add in f32, round the sum to f16 — the exact f16(c) semantics), the
// widened stored f16 sums are cached in registers, and the rms passes are
// rms_norm_row_f16's per-thread strided order with its 256-slot tree —
// the fused_add_rms_f16 shape exactly.
// ---------------------------------------------------------------------------
// S1-9b: the partials layout is s-major (partials[s*n + i], see PlanRow),
// so one warp's 32 columns at slab s sit in 128 contiguous bytes — one
// coalesced line per load instead of 32 scattered 4B sector requests at
// 192B stride. That plus the 8-deep software pipeline below keeps the
// 48-slab down sum off the critical path: the adds chain in the SAME
// ascending single-accumulator order — bit identical to the split
// gemv_m1_f16f32_reduce sum — while the window loads stay in flight.
// (c-major layout + pipeline-8 measured 20.5us vs 11.0us at nslabs 48
// vs 24; the s-major layout brings the 48-slab cost down to the 24-slab
// level — the LSU transaction count was the limiter, not the latency.)
extern "C" __global__ void gemv_p2_add_rms(const float* __restrict__ partials,
                                           unsigned short* __restrict__ x,
                                           unsigned short* __restrict__ out,
                                           const unsigned short* __restrict__ w,
                                           int n, int nslabs, float eps) {
    int tid = threadIdx.x;
    __shared__ float s_sh[256];
    float v[4];  // n <= 1024, 256 threads -> at most 4 elements per thread
    int cnt = 0;
    for (int i = tid; i < n; i += 256) {
        const float* p = partials + i;  // s-major: slab s at p[(size_t)s * n]
        float acc = 0.0f;
        int s = 0;
        // Software pipeline: p0..p7 always hold p[s..s+7] at the top of
        // the main loop — the 8 window loads stay in flight while the
        // adds chain in ascending order (bit-identical to the plain
        // single-accumulator sum). Main loop consumes full windows;
        // the tail (<= 11) is finished one element at a time.
        float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f, p4 = 0.f, p5 = 0.f, p6 = 0.f, p7 = 0.f;
        if (nslabs > 0) { p0 = p[0]; }
        if (nslabs > 1) { p1 = p[(size_t)1 * n]; }
        if (nslabs > 2) { p2 = p[(size_t)2 * n]; }
        if (nslabs > 3) { p3 = p[(size_t)3 * n]; }
        if (nslabs > 4) { p4 = p[(size_t)4 * n]; }
        if (nslabs > 5) { p5 = p[(size_t)5 * n]; }
        if (nslabs > 6) { p6 = p[(size_t)6 * n]; }
        if (nslabs > 7) { p7 = p[(size_t)7 * n]; }
        for (; s + 8 <= nslabs; s += 8) {
            acc += p0; acc += p1; acc += p2; acc += p3;
            acc += p4; acc += p5; acc += p6; acc += p7;
            if (s + 8 < nslabs) { p0 = p[(size_t)(s + 8) * n]; }
            if (s + 9 < nslabs) { p1 = p[(size_t)(s + 9) * n]; }
            if (s + 10 < nslabs) { p2 = p[(size_t)(s + 10) * n]; }
            if (s + 11 < nslabs) { p3 = p[(size_t)(s + 11) * n]; }
            if (s + 12 < nslabs) { p4 = p[(size_t)(s + 12) * n]; }
            if (s + 13 < nslabs) { p5 = p[(size_t)(s + 13) * n]; }
            if (s + 14 < nslabs) { p6 = p[(size_t)(s + 14) * n]; }
            if (s + 15 < nslabs) { p7 = p[(size_t)(s + 15) * n]; }
        }
        if (s + 4 <= nslabs) {
            acc += p0; acc += p1; acc += p2; acc += p3;
            s += 4;
            p0 = p4; p1 = p5; p2 = p6; p3 = p7;
        }
        for (; s < nslabs; ++s) {
            acc += p0;
            if (s + 1 < nslabs) { p0 = p[(size_t)(s + 1) * n]; }
        }
        float sum = hbits_to_f32(x[i]) + hbits_to_f32(f32_to_hbits(acc));
        unsigned short v16 = f32_to_hbits(sum);
        x[i] = v16;
        v[cnt++] = hbits_to_f32(v16);
    }
    float s = 0.f;
    for (int k = 0; k < cnt; ++k) {
        s += v[k] * v[k];
    }
    s_sh[tid] = s;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (tid < st) {
            s_sh[tid] += s_sh[tid + st];
        }
        __syncthreads();
    }
    if (tid == 0) {
        float mean_sq = s_sh[0] / (float)n;
        s_sh[0] = rsqrtf(mean_sq + eps);
    }
    __syncthreads();
    float rstd = s_sh[0];
    for (int i = tid, k = 0; i < n; i += 256, ++k) {
        out[i] = f32_to_hbits(v[k] * rstd * hbits_to_f32(w[i]));
    }
}

// ---------------------------------------------------------------------------
// Group 5: gate/up phase-2 + fused cast-SiLU-GLU + the down plan's phase-1
// in one kernel (S1-9 — the 8-node layer's seventh node).
// grid = ncols_d * nslabs_d (192 for Qwen3-0.6B after the S1-9b nslabs
// tuning: 4 col-groups x 48 slabs) — the down plan's full phase-1 tile
// grid, identical to the split gemv_m1_f16f32 launch. block = 256.
// plans[0..2] are the layer's gate, up and down rows (3 rows).
// Phase 1 (per column, identical to gemv_p2_gu_swiglu): ascending-slab
// reduction of the gate/up partials, cast to f16, silu = g/(1+exp(-g)),
// down[col] = f16(silu * u). Block bx redundantly writes the 256-col
// stripe c1 = (bx % nslabs_d) / (256/slab_k) — the stripe its phase-2
// k-range lies in (build_plans' gate: every slab s's k-range is inside
// [(s/per_stripe)*256, (s/per_stripe)*256 + 256), per_stripe =
// 256/slab_k — slab_k in {256, 128, 64}) — so after __syncthreads()
// every phase-2 tile reads only the columns its own block wrote. The
// nslabs_d/per_stripe writers per stripe all write identical bits
// (idempotent).
// Phase 2 (the down phase-1, per-(col, slab) tiles byte-identical to
// gemv_m1_f16f32): block bx computes tile (c2, slab) with c2 = bx /
// nslabs_d and slab = bx % nslabs_d — the split kernel's exact
// (local / nslabs, local % nslabs) tile mapping with local = bx — the
// same ceil-slab split, the same 4-ILP accumulation tree and the same
// (acc0+acc1)+(acc2+acc3) combination, so the partials are bit-identical.
// ---------------------------------------------------------------------------
extern "C" __global__ void gemv_p2_gu_p1d_swiglu(const PlanRow* __restrict__ plans,
                                                 int nplans) {
    int bx = blockIdx.x;
    int tid = threadIdx.x;
    const PlanRow rg = plans[0];
    const PlanRow ru = plans[1];
    const PlanRow rd = plans[2];

    // Phase 1: gate/up reductions + swiglu -> down[col] for the stripe
    // c1 = (bx % nslabs_d) / (256 / slab_k) — this block's phase-2
    // k-range. (down = the down row's `a` activation buffer; the write
    // is redundant across blocks and idempotent). The dynamic divisor
    // admits slab_k in {256, 128, 64} (1/2/4 slabs per 256-stripe —
    // S1-9b nslabs tuning: 12/24/48 slabs for k=3072); the build_plans
    // gate mirrors this mapping and rejects other configs (split
    // fallback).
    const int slab_k = (rd.k + rd.nslabs - 1) / rd.nslabs;
    int c1 = (bx % rd.nslabs) / (256 / slab_k);
    int col = c1 * 256 + tid;
    if (col < rg.n) {
        float gacc = 0.0f, uacc = 0.0f;
        for (int s = 0; s < rg.nslabs; ++s) {
            gacc += rg.partials[(size_t)s * rg.n + col];
            uacc += ru.partials[(size_t)s * ru.n + col];
        }
        float gv = hbits_to_f32(f32_to_hbits(gacc));
        float uv = hbits_to_f32(f32_to_hbits(uacc));
        float silu = gv / (1.f + expf(-gv));
        ((unsigned short*)rd.a)[col] = f32_to_hbits(silu * uv);
    }
    __syncthreads();

    // Phase 2: the down phase-1 tile (c2, slab) — the split kernel's
    // (local / nslabs, local % nslabs) tile. The loads use __half2float
    // on the __half pointers, LITERALLY gemv_m1_f16f32's expression —
    // hbits_to_f32 would go through an implicit __half -> float ->
    // unsigned short conversion (a value change, not a bit
    // re-interpret), corrupting the weight bits. The per-tile k-walk is
    // the shared unguarded fast path (slab_k divides k for every real
    // plan; the guarded general fallback is bit-identical).
    {
        int slab = bx % rd.nslabs;
        int ks = slab * slab_k;
        int ke = ks + slab_k;
        if (ke > rd.k) {
            ke = rd.k;
        }
        int c2 = bx / rd.nslabs;
        int ccmax = c2 * 256 + 256;
        if (ccmax > rd.n) {
            ccmax = rd.n;
        }
        for (int cc = c2 * 256 + tid; cc < ccmax; cc += 256) {
            gemv_phase1(rd, cc, ks, ke, slab);
        }
    }
}
