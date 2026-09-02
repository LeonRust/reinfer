// S1-10 decode-step deep fusion: the whole layer in ONE kernel launch.
//
// The S1-9 fused step still launches 8 kernels per layer (p1_qkv, p2_qkv,
// flash_fused, p1_o, p2_o, p1_gu, p2_gu_d, p2_down — 229 nodes/step for
// Qwen3-0.6B). Between dependent kernels the GPU pays a full drain +
// grid-ramp per boundary, and the small stages (p2_qkv: 8 blocks, p2_o:
// 1 block) leave the device mostly idle. S1-10 merges the eight kernels
// into ONE persistent kernel per layer: a fixed grid of co-resident
// blocks runs the eight stages in sequence, separated by a device-side
// grid barrier (arrive/spin on a generation counter, released by the last
// arriving block). Per step: 28 layer launches + gather + rms0 + lm pair
// = 32 nodes (down from 229).
//
// Bit-level contract: every stage preserves the exact per-column / per-tile
// arithmetic and accumulation orders of the kernels it replaces (the
// ascending-slab phase-2 sums, the 4-ILP phase-1 tree, the per-thread
// strided rms sums with the 256-slot tree, the f16 round/widen/round
// chains, the software RNE f16 conversions, the flash phases A/B/C and the
// kv-slot write). The only differences are mechanical:
//   - the block size is 512 threads (the flash kernel already uses 512;
//     the other stages' per-thread work is unchanged — a 512-col tile
//     replaces two 256-col tiles, and the per-(col, slab) computations
//     are byte-identical because each tile is still computed by exactly
//     one thread with the same k-walk);
//   - stage work tiles are grid-strided over the fixed grid (tile t goes
//     to block t % G deterministically — every tile is computed by
//     exactly one block);
//   - the add_rms stages use only threads 0..255 (the rest idle at the
//     block syncs) — the per-thread strided sums and the 256-slot tree
//     are unchanged;
//   - the head-norm guard becomes 512/d heads per block (per-head
//     arithmetic unchanged).
// Stage order is exactly the S1-9 launch order, so the inter-stage data
// flow is identical; the grid barriers order the stages device-side.
//
// Grid-size gate: the barrier spins deadlock unless every block is
// co-resident. The host computes the exact co-resident block count via
// cuOccupancyMaxActiveBlocksPerMultiprocessor and launches grid =
// min(stage tile max, occupancy * SM count); stage tiles grid-stride so
// any co-resident grid is correct. If even that cannot be satisfied the
// loader fails (Fatal) and the engine falls back to the S1-9 fused path.
//
// S1-11 (specs/017) block-width wave: the "single-block serial" stages
// (p1_qkv, p2_qkv, p1_gu) are widened by splitting their 512-col tiles
// into 256-col tiles, one PAIR of adjacent tiles per block (threads
// 0..255 on tile 2p, threads 256..511 on tile 2p+1 — both halves run the
// same k-walk concurrently). Only the (column -> block, thread)
// assignment changes; every (col, slab) value is still computed by
// exactly one thread with the identical scan order, so the outputs and
// the partials bytes are unchanged. The widened kernel is the second
// entry point `decode_step_layer_fused_bw2` (W=2, __launch_bounds__(512,
// 2) — 64 registers/thread so two blocks per SM fit); the W=1 entry is
// the S1-10 kernel verbatim (REINFER_FUSED_BW=off keeps it). The host
// (layer_fused.rs) picks the entry and the grid; the barrier participant
// sets below are recomputed from the widened tile counts automatically
// (the P formulas read the const's tile counts, which the host uploads
// per width).
//
// 017-d per-thread column width (specs/017-decode-block-width): the
// DRAM-bound phase-1 stages (p1_qkv, p1_gu, and the down phase-1 in
// stage 7) widen the b-row loads of the W=2 kernel further — each thread
// owns WC = 2 or 4 CONSECUTIVE columns and fetches them as one vector
// load (LDG.32 / LDG.64; scalar stride-2 loads would waste half of every
// 32B sector). A block-half's tile is then 256*WC columns wide and the
// stage-7 tile 512*WC wide. Per-column arithmetic is untouched (each
// column still gets the identical 4-ILP chain, and the reduction trees /
// aggregation orders never see WC) — see gemv_phase1_wc. WC=1 keeps
// this file's S1-11 code verbatim; entries `decode_step_layer_fused_
// bw2_wc2` / `..._wc4` (both __launch_bounds__(512, 2)) carry WC=2/4,
// selected by REINFER_FUSED_WC.
//
// Host side: crates/cuda/src/layer_fused.rs; engine wiring:
// crates/cuda/src/engine.rs (REINFER_LAYER_FUSED gate, S1-10 step launch
// + GraphStepDecl::build_layer_fused).

#include <cuda_fp16.h>  // __half / __half2float (same as gemm_m1.cu)

// Host-mirrored plan row (40 bytes, 8-aligned) — identical to
// decode_fused_kernels.cu (the layer kernel reads the SAME plan table
// the S1-9 fused path uploads).
struct PlanRow {
    const __half* a;        // [k] f16 activation row
    const __half* b;        // [k x n] f16 row-major weight matrix
    float* partials;        // [nslabs x n] s-major slab partials
    int n;                  // output columns
    int k;                  // reduction length
    int nslabs;             // k slabs (phase-1 grid split)
    int col_off;            // unused here (kept for layout identity)
};

// Host-mirrored static geometry (uploaded once; all pointers stable).
struct LayerFusedConst {
    const unsigned short* embed;   // gather source
    unsigned short* x;             // residual stream (mutated in place)
    unsigned short* xn;            // norm output (layer input of the next plan group)
    unsigned short* q16;           // q/k/v projection outputs (f16)
    unsigned short* k16;
    unsigned short* v16;
    unsigned short* attn;          // flash output (o-projection input)
    unsigned short* kv;            // kv cache (K region then V region)
    const unsigned int* lens;      // [B] kv lengths
    const unsigned int* pages;     // identity page table base (li*pp offset inside)
    const unsigned short* wnorm0;  // layer 0 attn_norm (the rms0 stage)
    unsigned int* stage_ts;        // per-stage clock64 slots (null when off)
    int h, nqk, kvk, d, half, ffn;
    int nslabs_q, nslabs_k, nslabs_v, nslabs_o, nslabs_g, nslabs_d;
    int tiles_qkv;                 // p1_qkv stage tiles (sum over q/k/v rows)
    int tiles_o;                   // p1_o stage tiles
    int tiles_gu;                  // p1_gu stage tiles
    int tiles_gu_d;                // p2_gu_d stage tiles (down plan tiles)
    int q_heads, kv_heads, ratio, block_len, max_kv, total_pages, pp;
    float eta, scale_q, eps;
    int hn;                        // head_norm on (q_norm/k_norm present)
};

__device__ __forceinline__ float hbits_to_f32(unsigned short h) {
    // Identical software widening to decode_fused_kernels.cu.
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
    // Identical RNE bit construction to decode_fused_kernels.cu.
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
// Grid barrier: arrive + spin on a generation counter. Only the CALLER's
// participant blocks call it (the participant prefix {bx < p}; full-grid
// barriers pass p == gridDim.x). The last arriving participant resets the
// arrival counter and bumps the generation; the others spin until they
// observe the bump. Memory ordering: each block fences before its arrival
// (its stage writes become visible) and after the spin (others' stage
// writes become visible before the following reads). The grid must be
// fully co-resident (host gate) — otherwise the spin deadlocks.
// `cnt`/`gen` slots are distinct per barrier index and self-reset, so no
// per-launch initialization is needed (the buffer is zeroed once at build;
// the first launch's barrier 0 sees cnt == 0).
__device__ __forceinline__ void grid_barrier(volatile unsigned int* cnt,
                                             volatile unsigned int* gen,
                                             int p) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();  // release: my stage writes are visible before arrival
        unsigned int arrive = atomicAdd((unsigned int*)cnt, 1);
        if ((int)arrive == p - 1) {
            *cnt = 0;  // reset for the next barrier
            __threadfence();
            atomicAdd((unsigned int*)gen, 1);  // release the others
        } else {
            unsigned int s = *gen;
            while (*gen == s) {  // spin until the last block releases
            }
            __threadfence();  // acquire: others' stage writes are visible now
        }
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------
// Phase-1 accumulation for one (col, slab) tile — LITERALLY
// decode_fused_kernels.cu's gemv_phase1 (4 ILP accumulators over stride-4
// k, fixed (acc0+acc1)+(acc2+acc3) tree, unguarded fast path when the
// slab's k-range is a multiple of 4).
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// S1-11d (specs/017-d): the phase-1 b-row walk widened — one thread
// accumulates WC CONSECUTIVE output columns with independent per-column
// 4-ILP chains. Per (col, slab) the k sequence, the (acc0+acc1)+(acc2+acc3)
// tree and the partials write are byte-identical to gemv_phase1 (every
// column is still computed by exactly one thread in the identical order —
// only the column -> thread assignment changed). The consecutive columns
// make the per-(thread, k) b loads contiguous, so WC columns per row are
// fetched as one vector LDG.32 (WC=2) / LDG.64 (WC=4) instead of WC scalar
// 2B loads (scalar per-column loads at stride-2 would fetch each 32B
// sector half-used). Load width never enters the arithmetic — the widen
// path changes only memory instruction shapes.
//
// `wc` = the number of valid columns at col0 (== WC on the vector fast
// path; a tile that straddles the plan's n edge falls back to the guarded
// per-column scalar path for its remainder).
// ---------------------------------------------------------------------------
template <int WC>
__device__ __forceinline__ void gemv_phase1_wc(const PlanRow& row, int col0,
                                               int ks, int ke, int slab,
                                               int wc) {
    float a[WC][4];
    for (int j = 0; j < WC; ++j) {
        a[j][0] = 0.0f;
        a[j][1] = 0.0f;
        a[j][2] = 0.0f;
        a[j][3] = 0.0f;
    }
    if ((ke - ks) % 4 == 0 && wc == WC) {
        // Fast path (no per-k guards — the slab k-range is a multiple of
        // 4, exactly like gemv_phase1's fast path). Vector loads require
        // 4B (WC=2) / 8B (WC=4) alignment: col0 is a multiple of WC and
        // the row stride is even (checked below); plans with an odd n
        // fall back to the scalar loop (identical arithmetic).
        if constexpr (WC == 2) {
            if ((row.n & 1) == 0) {
                #pragma unroll 8
                for (int k0 = ks; k0 < ke; k0 += 4) {
                    const unsigned int* b = (const unsigned int*)(row.b + (size_t)k0 * row.n + col0);
                    float av0 = __half2float(__ldg(&row.a[k0]));
                    float av1 = __half2float(__ldg(&row.a[k0 + 1]));
                    float av2 = __half2float(__ldg(&row.a[k0 + 2]));
                    float av3 = __half2float(__ldg(&row.a[k0 + 3]));
                    unsigned int w0 = __ldg(b);
                    unsigned int w1 = __ldg(b + row.n / 2);
                    unsigned int w2 = __ldg(b + 2 * (row.n / 2));
                    unsigned int w3 = __ldg(b + 3 * (row.n / 2));
                    a[0][0] += av0 * hbits_to_f32((unsigned short)(w0 & 0xffffu));
                    a[0][1] += av1 * hbits_to_f32((unsigned short)(w1 & 0xffffu));
                    a[0][2] += av2 * hbits_to_f32((unsigned short)(w2 & 0xffffu));
                    a[0][3] += av3 * hbits_to_f32((unsigned short)(w3 & 0xffffu));
                    a[1][0] += av0 * hbits_to_f32((unsigned short)(w0 >> 16));
                    a[1][1] += av1 * hbits_to_f32((unsigned short)(w1 >> 16));
                    a[1][2] += av2 * hbits_to_f32((unsigned short)(w2 >> 16));
                    a[1][3] += av3 * hbits_to_f32((unsigned short)(w3 >> 16));
                }
            } else {
                // Scalar fast path (n odd — vector loads misaligned).
                for (int k0 = ks; k0 < ke; k0 += 4) {
                    for (int i = 0; i < 4; ++i) {
                        float av = __half2float(__ldg(&row.a[k0 + i]));
                        a[0][i] += av * __half2float(__ldg(&row.b[(size_t)(k0 + i) * row.n + col0]));
                        a[1][i] += av * __half2float(__ldg(&row.b[(size_t)(k0 + i) * row.n + col0 + 1]));
                    }
                }
            }
        } else {  // WC == 4
            if ((row.n & 3) == 0) {
                #pragma unroll 8
                for (int k0 = ks; k0 < ke; k0 += 4) {
                    const unsigned long long* b =
                        (const unsigned long long*)(row.b + (size_t)k0 * row.n + col0);
                    float av0 = __half2float(__ldg(&row.a[k0]));
                    float av1 = __half2float(__ldg(&row.a[k0 + 1]));
                    float av2 = __half2float(__ldg(&row.a[k0 + 2]));
                    float av3 = __half2float(__ldg(&row.a[k0 + 3]));
                    unsigned long long w0 = __ldg(b);
                    unsigned long long w1 = __ldg(b + row.n / 4);
                    unsigned long long w2 = __ldg(b + 2 * (row.n / 4));
                    unsigned long long w3 = __ldg(b + 3 * (row.n / 4));
                    a[0][0] += av0 * hbits_to_f32((unsigned short)(w0 & 0xffffu));
                    a[1][0] += av0 * hbits_to_f32((unsigned short)((w0 >> 16) & 0xffffu));
                    a[2][0] += av0 * hbits_to_f32((unsigned short)((w0 >> 32) & 0xffffu));
                    a[3][0] += av0 * hbits_to_f32((unsigned short)(w0 >> 48));
                    a[0][1] += av1 * hbits_to_f32((unsigned short)(w1 & 0xffffu));
                    a[1][1] += av1 * hbits_to_f32((unsigned short)((w1 >> 16) & 0xffffu));
                    a[2][1] += av1 * hbits_to_f32((unsigned short)((w1 >> 32) & 0xffffu));
                    a[3][1] += av1 * hbits_to_f32((unsigned short)(w1 >> 48));
                    a[0][2] += av2 * hbits_to_f32((unsigned short)(w2 & 0xffffu));
                    a[1][2] += av2 * hbits_to_f32((unsigned short)((w2 >> 16) & 0xffffu));
                    a[2][2] += av2 * hbits_to_f32((unsigned short)((w2 >> 32) & 0xffffu));
                    a[3][2] += av2 * hbits_to_f32((unsigned short)(w2 >> 48));
                    a[0][3] += av3 * hbits_to_f32((unsigned short)(w3 & 0xffffu));
                    a[1][3] += av3 * hbits_to_f32((unsigned short)((w3 >> 16) & 0xffffu));
                    a[2][3] += av3 * hbits_to_f32((unsigned short)((w3 >> 32) & 0xffffu));
                    a[3][3] += av3 * hbits_to_f32((unsigned short)(w3 >> 48));
                }
            } else {
                for (int k0 = ks; k0 < ke; k0 += 4) {
                    for (int i = 0; i < 4; ++i) {
                        float av = __half2float(__ldg(&row.a[k0 + i]));
                        for (int j = 0; j < 4; ++j) {
                            a[j][i] += av *
                                __half2float(__ldg(&row.b[(size_t)(k0 + i) * row.n + col0 + j]));
                        }
                    }
                }
            }
        }
    } else {
        // Guarded path (k tail or the tile's n-edge straddle — per-column
        // guards mirror gemv_phase1's guarded loop for each column; the
        // j<wc guards are per-column compile-time unrolled so the a[j][i]
        // accumulators stay in registers).
        for (int k0 = ks; k0 < ke; k0 += 4) {
            float av0 = __half2float(__ldg(&row.a[k0]));
            #pragma unroll
            for (int j = 0; j < WC; ++j) {
                if (j < wc) {
                    a[j][0] += av0 * __half2float(__ldg(&row.b[(size_t)k0 * row.n + col0 + j]));
                }
            }
            if (k0 + 1 < ke) {
                float av1 = __half2float(__ldg(&row.a[k0 + 1]));
                #pragma unroll
                for (int j = 0; j < WC; ++j) {
                    if (j < wc) {
                        a[j][1] += av1 *
                            __half2float(__ldg(&row.b[(size_t)(k0 + 1) * row.n + col0 + j]));
                    }
                }
            }
            if (k0 + 2 < ke) {
                float av2 = __half2float(__ldg(&row.a[k0 + 2]));
                #pragma unroll
                for (int j = 0; j < WC; ++j) {
                    if (j < wc) {
                        a[j][2] += av2 *
                            __half2float(__ldg(&row.b[(size_t)(k0 + 2) * row.n + col0 + j]));
                    }
                }
            }
            if (k0 + 3 < ke) {
                float av3 = __half2float(__ldg(&row.a[k0 + 3]));
                #pragma unroll
                for (int j = 0; j < WC; ++j) {
                    if (j < wc) {
                        a[j][3] += av3 *
                            __half2float(__ldg(&row.b[(size_t)(k0 + 3) * row.n + col0 + j]));
                    }
                }
            }
        }
    }
    float* dst = row.partials + (size_t)slab * row.n + col0;
    dst[0] = (a[0][0] + a[0][1]) + (a[0][2] + a[0][3]);
    if (wc > 1) {
        dst[1] = (a[1][0] + a[1][1]) + (a[1][2] + a[1][3]);
    }
    if (wc > 2) {
        dst[2] = (a[2][0] + a[2][1]) + (a[2][2] + a[2][3]);
    }
    if (wc > 3) {
        dst[3] = (a[3][0] + a[3][1]) + (a[3][2] + a[3][3]);
    }
}

// ---------------------------------------------------------------------------
// Stage 1: phase-1 of the q/k/v plans (the layer input xn). Tiles =
// sum over plans of (ceil(n/tw) * nslabs); each tile is one tw-col
// (col-block, slab) pair, one thread per column, the identical per-tile
// arithmetic as gemv_m1_f16f32_multi. `tw` is the widened tile width
// (512 for the plain stages, 256 for the S1-11 widened stages).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void stage_p1(const LayerFusedConst* c,
                                         const PlanRow* table,
                                         int row_base, int nrows,
                                         int tile, int tw, int tl) {
    int row = row_base;
    for (int r = 0; r < nrows; ++r) {
        const PlanRow R = table[row_base + r];
        int ncols = (R.n + tw - 1) / tw;
        int nt = ncols * R.nslabs;
        if (tile < nt) {
            row = row_base + r;
            break;
        }
        tile -= nt;
    }
    const PlanRow R = table[row];
    int col = (tile / R.nslabs) * tw + tl;
    if (col >= R.n) {
        return;
    }
    int slab = tile % R.nslabs;
    int slab_k = (R.k + R.nslabs - 1) / R.nslabs;
    int ks = slab * slab_k;
    int ke = ks + slab_k;
    if (ke > R.k) {
        ke = R.k;
    }
    gemv_phase1(R, col, ks, ke, slab);
}

// 017-d: stage_p1's WC-wide variant — the block-half's tile covers
// tw = 256*WC columns and thread tl (0..255) owns WC CONSECUTIVE columns
// starting at col0 = blk*tw + tl*WC (the same columns stage_p1 would
// assign to threads tl*WC..tl*WC+WC-1 of a 512/1024-col tile, so every
// (col, slab) is still computed exactly once, in the identical order).
// The tile decode is stage_p1's with tw = 256*WC; the k-range and the
// per-column chain arithmetic come from gemv_phase1_wc<WC>.
template <int WC>
__device__ __forceinline__ void stage_p1_wc(const LayerFusedConst* c,
                                            const PlanRow* table,
                                            int row_base, int nrows,
                                            int tile, int tl) {
    const int tw = 256 * WC;
    int row = row_base;
    for (int r = 0; r < nrows; ++r) {
        const PlanRow R = table[row_base + r];
        int ncols = (R.n + tw - 1) / tw;
        int nt = ncols * R.nslabs;
        if (tile < nt) {
            row = row_base + r;
            break;
        }
        tile -= nt;
    }
    const PlanRow R = table[row];
    int col0 = (tile / R.nslabs) * tw + tl * WC;
    if (col0 >= R.n) {
        return;
    }
    int wc = R.n - col0;
    if (wc > WC) {
        wc = WC;
    }
    int slab = tile % R.nslabs;
    int slab_k = (R.k + R.nslabs - 1) / R.nslabs;
    int ks = slab * slab_k;
    int ke = ks + slab_k;
    if (ke > R.k) {
        ke = R.k;
    }
    gemv_phase1_wc<WC>(R, col0, ks, ke, slab, wc);
}

// The plain (512-col) tile loop — grid-strided, all 512 threads per tile.
// Used by the p1_o stage in both widths and by the widened stages when W=1.
__device__ __forceinline__ void p1_tiles_plain(const LayerFusedConst* c,
                                               const PlanRow* table,
                                               int row_base, int nrows,
                                               int tiles, int bx, int G,
                                               int tid) {
    for (int t = bx; t < tiles; t += G) {
        stage_p1(c, table, row_base, nrows, t, 512, tid);
    }
}

// The widened (S1-11) tile loop: tw-col tiles, one adjacent PAIR per
// block — threads 0..255 process tile 2p, threads 256..511 process tile
// 2p+1, both k-walks concurrently (per-thread arithmetic unchanged; only
// the column -> (block, thread) assignment differs). W=1 keeps the plain
// grid-stride loop verbatim.
//
// 017-d: with WC > 1 each half's tile is 256*WC columns wide and every
// thread covers WC consecutive columns via stage_p1_wc (the tw argument
// is then only the W=2/WC=1 tile width; stage_p1_wc derives its own).
template <int W, int WC>
__device__ __forceinline__ void p1_tiles(const LayerFusedConst* c,
                                         const PlanRow* table,
                                         int row_base, int nrows,
                                         int tiles, int tw, int bx, int G,
                                         int tid) {
    if constexpr (W == 1) {
        for (int t = bx; t < tiles; t += G) {
            stage_p1(c, table, row_base, nrows, t, 512, tid);
        }
    } else if constexpr (WC == 1) {
        const int half = tid >> 8;
        const int tl = tid & 255;
        for (int p = bx; 2 * p < tiles; p += G) {
            if (2 * p + half < tiles) {
                stage_p1(c, table, row_base, nrows, 2 * p + half, tw, tl);
            }
        }
    } else {
        const int half = tid >> 8;
        const int tl = tid & 255;
        for (int p = bx; 2 * p < tiles; p += G) {
            if (2 * p + half < tiles) {
                stage_p1_wc<WC>(c, table, row_base, nrows, 2 * p + half, tl);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Stage 2: q/k/v phase-2 reductions + f16 casts + q/k head-norm + RoPE —
// the 512-thread form of gemv_p2_qkv_cast_hn_rope. Tiles: q (ceil(nq/tw)),
// k, v. Per-column ascending-slab sums, per-head 128-slot tree (st =
// 64..1) and the rope pair rounds are unchanged; only the block's head
// count grows from 256/d to 512/d (the rstd broadcast guard becomes
// tid < 512/d). `tw` = 512 for W=1, 256 for the widened W=2 entry.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void stage_p2_qkv_w1(const LayerFusedConst* c,
                                                const PlanRow* table,
                                                int tile, int tid,
                                                unsigned int pos,
                                                const unsigned short* wq,
                                                const unsigned short* wk) {
    const int cq = (c->nqk + 511) / 512;
    const int ck = (c->kvk + 511) / 512;
    const int d = c->d;
    const int half = c->half;
    __shared__ float s_sh[1024];  // (512/d) heads * 128 slots, d >= 32
    if (tile < cq) {
        // --- q segment: reduce -> cast -> (hn) -> rope ---------------------
        const PlanRow R = table[0];
        int col = tile * 512 + tid;
        if (col < R.n) {
            float acc = 0.0f;
            for (int s = 0; s < R.nslabs; ++s) {
                acc += R.partials[(size_t)s * R.n + col];
            }
            c->q16[col] = f32_to_hbits(acc);
        }
        if (c->hn) {
            int head = tid / d;
            int e = tid % d;
            float v = (col < R.n) ? hbits_to_f32(c->q16[col]) : 0.0f;
            s_sh[head * 128 + e] = v * v;
            __syncthreads();
            for (int st = 64; st > 0; st >>= 1) {
                if (e < st) {
                    s_sh[head * 128 + e] += s_sh[head * 128 + e + st];
                }
                __syncthreads();
            }
            if (tid < 512 / d) {
                s_sh[tid * 128] = rsqrtf(s_sh[tid * 128] / (float)d + c->eps);
            }
            __syncthreads();
            if (col < R.n) {
                float rstd = s_sh[(tid / d) * 128];
                c->q16[col] =
                    f32_to_hbits(hbits_to_f32(c->q16[col]) * rstd * hbits_to_f32(wq[e]));
            }
            __syncthreads();
        } else {
            __syncthreads();
        }
        if (col < R.n) {
            int e = tid % d;
            if (e < half) {
                int base = col - e;
                unsigned short* xr = c->q16 + base;
                float theta = (float)pos * powf(c->eta, -2.f * (float)e / (2.f * (float)half));
                float co = cosf(theta), si = sinf(theta);
                float a = hbits_to_f32(xr[e]);
                float b = hbits_to_f32(xr[e + half]);
                float v1 = a * co - b * si;
                float v2 = a * si + b * co;
                xr[e] = f32_to_hbits(hbits_to_f32(f32_to_hbits(v1)) * c->scale_q);
                xr[e + half] = f32_to_hbits(hbits_to_f32(f32_to_hbits(v2)) * c->scale_q);
            }
        }
        return;
    }
    if (tile < cq + ck) {
        // --- k segment: same pipeline, scale 1.0 ---------------------------
        const PlanRow R = table[1];
        int lb = tile - cq;
        int col = lb * 512 + tid;
        if (col < R.n) {
            float acc = 0.0f;
            for (int s = 0; s < R.nslabs; ++s) {
                acc += R.partials[(size_t)s * R.n + col];
            }
            c->k16[col] = f32_to_hbits(acc);
        }
        if (c->hn) {
            int head = tid / d;
            int e = tid % d;
            float v = (col < R.n) ? hbits_to_f32(c->k16[col]) : 0.0f;
            s_sh[head * 128 + e] = v * v;
            __syncthreads();
            for (int st = 64; st > 0; st >>= 1) {
                if (e < st) {
                    s_sh[head * 128 + e] += s_sh[head * 128 + e + st];
                }
                __syncthreads();
            }
            if (tid < 512 / d) {
                s_sh[tid * 128] = rsqrtf(s_sh[tid * 128] / (float)d + c->eps);
            }
            __syncthreads();
            if (col < R.n) {
                float rstd = s_sh[(tid / d) * 128];
                c->k16[col] =
                    f32_to_hbits(hbits_to_f32(c->k16[col]) * rstd * hbits_to_f32(wk[e]));
            }
            __syncthreads();
        } else {
            __syncthreads();
        }
        if (col < R.n) {
            int e = tid % d;
            if (e < half) {
                int base = col - e;
                unsigned short* xr = c->k16 + base;
                float theta = (float)pos * powf(c->eta, -2.f * (float)e / (2.f * (float)half));
                float co = cosf(theta), si = sinf(theta);
                float a = hbits_to_f32(xr[e]);
                float b = hbits_to_f32(xr[e + half]);
                float v1 = a * co - b * si;
                float v2 = a * si + b * co;
                xr[e] = f32_to_hbits(hbits_to_f32(f32_to_hbits(v1)));
                xr[e + half] = f32_to_hbits(hbits_to_f32(f32_to_hbits(v2)));
            }
        }
        return;
    }
    // --- v segment: reduce + cast only -------------------------------------
    const PlanRow R = table[2];
    int lb = tile - (cq + ck);
    int col = lb * 512 + tid;
    if (col < R.n) {
        float acc = 0.0f;
        for (int s = 0; s < R.nslabs; ++s) {
            acc += R.partials[(size_t)s * R.n + col];
        }
        c->v16[col] = f32_to_hbits(acc);
    }
}

// The widened (S1-11) q/k/v phase-2: two 256-col tiles per block (threads
// 0..255 on tile 2p, 256..511 on tile 2p+1). The per-column arithmetic is
// the W=1 form's (ascending-slab sums, casts, head-norm tree, rope pair
// rounds); the head-norm tree runs on a UNIFORM skeleton so both halves
// hit the same __syncthreads() regardless of their segments (the v
// segment contributes zero slots and skips the writes — the tree
// arithmetic per head is unchanged).
__device__ __forceinline__ void stage_p2_qkv_w2(const LayerFusedConst* c,
                                                const PlanRow* table,
                                                int tile, int tl, int tid,
                                                unsigned int pos,
                                                const unsigned short* wq,
                                                const unsigned short* wk) {
    const int cq = (c->nqk + 255) / 256;
    const int ck = (c->kvk + 255) / 256;
    const int d = c->d;
    const int half = c->half;
    __shared__ float s_sh[1024];  // (512/d) heads * 128 slots, d >= 32
    int seg;
    int col;
    if (tile < cq) {
        seg = 0;
        col = tile * 256 + tl;
    } else if (tile < cq + ck) {
        seg = 1;
        col = (tile - cq) * 256 + tl;
    } else {
        seg = 2;
        col = (tile - cq - ck) * 256 + tl;
    }
    const PlanRow R = table[seg];
    unsigned short* out = seg == 0 ? c->q16 : (seg == 1 ? c->k16 : c->v16);
    if (col < R.n) {
        float acc = 0.0f;
        for (int s = 0; s < R.nslabs; ++s) {
            acc += R.partials[(size_t)s * R.n + col];
        }
        out[col] = f32_to_hbits(acc);
    }
    if (c->hn) {
        int head = tid / d;
        int e = tid % d;
        float v = 0.0f;
        if (seg != 2 && col < R.n) {
            v = hbits_to_f32(out[col]);
        }
        s_sh[head * 128 + e] = v * v;
        __syncthreads();
        for (int st = 64; st > 0; st >>= 1) {
            if (e < st) {
                s_sh[head * 128 + e] += s_sh[head * 128 + e + st];
            }
            __syncthreads();
        }
        if (tid < 512 / d) {
            s_sh[tid * 128] = rsqrtf(s_sh[tid * 128] / (float)d + c->eps);
        }
        __syncthreads();
        if (seg != 2 && col < R.n) {
            float rstd = s_sh[(tid / d) * 128];
            out[col] = f32_to_hbits(hbits_to_f32(out[col]) * rstd *
                                    hbits_to_f32(seg == 0 ? wq[e] : wk[e]));
        }
        __syncthreads();
    } else {
        __syncthreads();
    }
    if (seg != 2 && col < R.n) {
        int e = tid % d;
        if (e < half) {
            int base = col - e;
            unsigned short* xr = out + base;
            float theta = (float)pos * powf(c->eta, -2.f * (float)e / (2.f * (float)half));
            float co = cosf(theta), si = sinf(theta);
            float a = hbits_to_f32(xr[e]);
            float b = hbits_to_f32(xr[e + half]);
            float v1 = a * co - b * si;
            float v2 = a * si + b * co;
            if (seg == 0) {
                // q: scale_q on the rounded pair (the W=1 q path verbatim).
                xr[e] = f32_to_hbits(hbits_to_f32(f32_to_hbits(v1)) * c->scale_q);
                xr[e + half] = f32_to_hbits(hbits_to_f32(f32_to_hbits(v2)) * c->scale_q);
            } else {
                // k: no scale (the W=1 k path verbatim).
                xr[e] = f32_to_hbits(hbits_to_f32(f32_to_hbits(v1)));
                xr[e + half] = f32_to_hbits(hbits_to_f32(f32_to_hbits(v2)));
            }
        }
    }
}

// The stage-2 tile loop: W=1 keeps the plain grid-stride form (one 512-col
// tile per block per iteration); W=2 runs one adjacent 256-col tile pair
// per block.
template <int W>
__device__ __forceinline__ void p2qkv_tiles(const LayerFusedConst* c,
                                            const PlanRow* table,
                                            int tiles, int bx, int G,
                                            int tid, unsigned int pos,
                                            const unsigned short* wq,
                                            const unsigned short* wk) {
    if (W == 1) {
        for (int t = bx; t < tiles; t += G) {
            stage_p2_qkv_w1(c, table, t, tid, pos, wq, wk);
        }
    } else {
        const int half = tid >> 8;
        const int tl = tid & 255;
        for (int p = bx; 2 * p < tiles; p += G) {
            if (2 * p + half < tiles) {
                stage_p2_qkv_w2(c, table, 2 * p + half, tl, tid, pos, wq, wk);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Stage 3: the fused flash (kv write of the current slot + flash decode
// attention) — the decode_step_gqa_flash_fused body verbatim (FLASH_TPB =
// 512 = the kernel's block size). Grid-strided over the q-heads; the kv
// page base is derived from li * pp inside. The conversions use the
// hardware F2F / F2H instructions exactly like decode_flash_kernels.cu
// (the software bit paths differ on NaN payloads and the [2^-25, 2^-24)
// f32->f16 rounding band — the flash stage must reproduce the S1-9
// kernel's bits, so it gets its own local helpers).
// ---------------------------------------------------------------------------
#define FLASH_TPB 512

__device__ __forceinline__ float fl_hbits_to_f32(unsigned short h) {
    return __half2float(*(const __half*)&h);
}
__device__ __forceinline__ unsigned short fl_f32_to_hbits(float f) {
    return (unsigned short)__half_as_ushort(__float2half_rn(f));
}

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

__device__ __forceinline__ void stage_flash(const LayerFusedConst* c,
                                            int head, int tid, int li) {
    int b = 0;
    int h = head;
    int kv_h = h / c->ratio;
    int kv_len = (int)c->lens[b];
    if (kv_len <= 0 || kv_len > c->max_kv) {
        return;  // caller guards; out stays untouched (same as the split)
    }
    const int per_tok_kv = c->kv_heads * c->d;
    const unsigned int* page = c->pages + (size_t)li * c->pp;

    // 1. Current step's kv slot: k16/v16 -> kv K/V regions (identical bytes
    //    to kv_write_row; every block writes the same values — idempotent
    //    redundant writes, no cross-block ordering needed).
    {
        int lp = (kv_len - 1) / c->block_len;
        int off = (kv_len - 1) % c->block_len;
        unsigned int phys = page[0] + lp;
        const size_t slot = ((size_t)phys * c->block_len + off) * per_tok_kv;
        const size_t v_base = (size_t)c->total_pages * c->block_len * per_tok_kv;
        for (int cc = tid; cc < per_tok_kv; cc += FLASH_TPB) {
            ((unsigned short*)c->kv)[slot + cc] = c->k16[cc];
            ((unsigned short*)c->kv)[v_base + slot + cc] = c->v16[cc];
        }
        __syncthreads();
    }

    // Dynamic smem: [0, d) q row f32; [d, d + kv_len) scores.
    extern __shared__ float sm[];
    float* sq = sm;
    float* ss = sm + c->d;
    // Static smem: 16 warp partial slots + 512 PV partials (2 per thread).
    __shared__ float s_slots[16];
    __shared__ float s_part[FLASH_TPB * 2];

    unsigned int base_page = page[0];

    // q row -> smem (f32), row offset (b*QH + h)*d.
    {
        const unsigned short* qrow = c->q16 + (size_t)(b * c->q_heads + h) * c->d;
        for (int i = tid; i < c->d; i += FLASH_TPB) {
            sq[i] = fl_hbits_to_f32(qrow[i]);
        }
        __syncthreads();
    }

    // Phase A: QK^T — fixed stride-512 token assignment, ascending j per
    // dot; two tokens interleaved for ILP (fixed pattern, no per-j branch).
    // S1-10c: each per-token 128-f16 kv row is fetched as 16B uint4 loads
    // (d gated to {64, 128, 256} and kv rows are 16B-aligned, so the cast
    // is always legal) — the loads are independent of the FMA chains, so
    // one batched wave replaces the 32 dependent 4B loads; the dot
    // arithmetic (4 accumulator chains in ascending-j order, the
    // (a0+a1)+(a2+a3) tree, the fl_hbits_to_f32 / hbits_to_f32 widening
    // mix, and the per-token fixed assignment) is byte-identical.
    float m = -1e30f;
    {
        int t = tid;
        while (t + FLASH_TPB < kv_len) {
            const uint4* krow = (const uint4*)krow_ptr(
                c->kv, page, t, c->block_len, kv_h, c->kv_heads, c->d, 1, base_page);
            const uint4* krow2 = (const uint4*)krow_ptr(
                c->kv, page, t + FLASH_TPB, c->block_len, kv_h, c->kv_heads, c->d, 1,
                base_page);
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < c->d; j += 8) {
                uint4 w = krow[j >> 3];
                uint4 wb = krow2[j >> 3];
                a0 += sq[j] * fl_hbits_to_f32((unsigned short)(w.x & 0xffffu));
                a1 += sq[j + 1] * fl_hbits_to_f32((unsigned short)(w.x >> 16));
                a2 += sq[j + 2] * fl_hbits_to_f32((unsigned short)(w.y & 0xffffu));
                a3 += sq[j + 3] * fl_hbits_to_f32((unsigned short)(w.y >> 16));
                a0 += sq[j + 4] * fl_hbits_to_f32((unsigned short)(w.z & 0xffffu));
                a1 += sq[j + 5] * fl_hbits_to_f32((unsigned short)(w.z >> 16));
                a2 += sq[j + 6] * fl_hbits_to_f32((unsigned short)(w.w & 0xffffu));
                a3 += sq[j + 7] * fl_hbits_to_f32((unsigned short)(w.w >> 16));
                b0 += sq[j] * fl_hbits_to_f32((unsigned short)(wb.x & 0xffffu));
                b1 += sq[j + 1] * fl_hbits_to_f32((unsigned short)(wb.x >> 16));
                b2 += sq[j + 2] * fl_hbits_to_f32((unsigned short)(wb.y & 0xffffu));
                b3 += sq[j + 3] * fl_hbits_to_f32((unsigned short)(wb.y >> 16));
                b0 += sq[j + 4] * fl_hbits_to_f32((unsigned short)(wb.z & 0xffffu));
                b1 += sq[j + 5] * fl_hbits_to_f32((unsigned short)(wb.z >> 16));
                b2 += sq[j + 6] * fl_hbits_to_f32((unsigned short)(wb.w & 0xffffu));
                b3 += sq[j + 7] * fl_hbits_to_f32((unsigned short)(wb.w >> 16));
            }
            float acc = (a0 + a1) + (a2 + a3);
            float acc2 = (b0 + b1) + (b2 + b3);
            ss[t] = acc;
            ss[t + FLASH_TPB] = acc2;
            m = fmaxf(m, fmaxf(acc, acc2));
            t += 2 * FLASH_TPB;
        }
        if (t < kv_len) {
            const uint4* krow = (const uint4*)krow_ptr(
                c->kv, page, t, c->block_len, kv_h, c->kv_heads, c->d, 1, base_page);
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < c->d; j += 8) {
                uint4 w = krow[j >> 3];
                a0 += sq[j] * fl_hbits_to_f32((unsigned short)(w.x & 0xffffu));
                a1 += sq[j + 1] * fl_hbits_to_f32((unsigned short)(w.x >> 16));
                a2 += sq[j + 2] * hbits_to_f32((unsigned short)(w.y & 0xffffu));
                a3 += sq[j + 3] * hbits_to_f32((unsigned short)(w.y >> 16));
                a0 += sq[j + 4] * fl_hbits_to_f32((unsigned short)(w.z & 0xffffu));
                a1 += sq[j + 5] * fl_hbits_to_f32((unsigned short)(w.z >> 16));
                a2 += sq[j + 6] * hbits_to_f32((unsigned short)(w.w & 0xffffu));
                a3 += sq[j + 7] * hbits_to_f32((unsigned short)(w.w >> 16));
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

    // Phase C: PV — (output i-pair, token chunk) split (identical
    // assignment and reduction order to the split kernel).
    {
        const size_t kv_base = (size_t)c->total_pages * c->block_len * per_tok_kv;
        const int ng = c->d / 2;
        const int nc = FLASH_TPB / ng;
        const int seg = (kv_len + nc - 1) / nc;
        const int i0 = 2 * (tid % ng);
        const int s = tid / ng;
        float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
        float c4 = 0.0f, c5 = 0.0f, c6 = 0.0f, c7 = 0.0f;
        int t = s * seg;
        int t_end = (s + 1) * seg < kv_len ? (s + 1) * seg : kv_len;
        // S1-10c: unroll so the per-token 4B v loads (4 independent per
        // iteration) issue as one batched wave — the 8 accumulator chains
        // and the per-token (i0, i0+1) assignment are unchanged.
        #pragma unroll 4
        for (; t + 3 < t_end; t += 4) {
            const unsigned int* v0 = (const unsigned int*)vrow_ptr(
                c->kv, kv_base, page, t, c->block_len, kv_h, c->kv_heads, c->d, 1,
                base_page);
            const unsigned int* v1 = (const unsigned int*)vrow_ptr(
                c->kv, kv_base, page, t + 1, c->block_len, kv_h, c->kv_heads, c->d, 1,
                base_page);
            const unsigned int* v2 = (const unsigned int*)vrow_ptr(
                c->kv, kv_base, page, t + 2, c->block_len, kv_h, c->kv_heads, c->d, 1,
                base_page);
            const unsigned int* v3 = (const unsigned int*)vrow_ptr(
                c->kv, kv_base, page, t + 3, c->block_len, kv_h, c->kv_heads, c->d, 1,
                base_page);
            unsigned int w0 = v0[i0 / 2];
            unsigned int w1 = v1[i0 / 2];
            unsigned int w2 = v2[i0 / 2];
            unsigned int w3 = v3[i0 / 2];
            c0 += ss[t] * fl_hbits_to_f32((unsigned short)(w0 & 0xffffu));
            c1 += ss[t] * fl_hbits_to_f32((unsigned short)(w0 >> 16));
            c2 += ss[t + 1] * fl_hbits_to_f32((unsigned short)(w1 & 0xffffu));
            c3 += ss[t + 1] * fl_hbits_to_f32((unsigned short)(w1 >> 16));
            c4 += ss[t + 2] * fl_hbits_to_f32((unsigned short)(w2 & 0xffffu));
            c5 += ss[t + 2] * fl_hbits_to_f32((unsigned short)(w2 >> 16));
            c6 += ss[t + 3] * fl_hbits_to_f32((unsigned short)(w3 & 0xffffu));
            c7 += ss[t + 3] * fl_hbits_to_f32((unsigned short)(w3 >> 16));
        }
        for (; t < t_end; ++t) {
            const unsigned int* v = (const unsigned int*)vrow_ptr(
                c->kv, kv_base, page, t, c->block_len, kv_h, c->kv_heads, c->d, 1,
                base_page);
            unsigned int w = v[i0 / 2];
            c0 += ss[t] * fl_hbits_to_f32((unsigned short)(w & 0xffffu));
            c1 += ss[t] * fl_hbits_to_f32((unsigned short)(w >> 16));
        }
        s_part[tid * 2] = (c0 + c2) + (c4 + c6);
        s_part[tid * 2 + 1] = (c1 + c3) + (c5 + c7);
        __syncthreads();
        if (s == 0) {
            float o0 = s_part[tid * 2];
            float o1 = s_part[tid * 2 + 1];
            for (int k = 1; k < nc; ++k) {
                o0 += s_part[(tid + k * ng) * 2];
                o1 += s_part[(tid + k * ng) * 2 + 1];
            }
            ((unsigned short*)c->attn)[(size_t)(b * c->q_heads + h) * c->d + i0] =
                fl_f32_to_hbits(o0);
            ((unsigned short*)c->attn)[(size_t)(b * c->q_heads + h) * c->d + i0 + 1] =
                fl_f32_to_hbits(o1);
        }
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------
// Stages 5/8, S1-10c split into two passes (bit-level, no D7):
//
//   A. stage_add_columns (ALL blocks): element-wise residual add over a
//      contiguous per-block column stripe. The per-column arithmetic is
//      byte-identical to the single-block stage (same 8-ILP slab sum in
//      the same addition order, same f16 round/widen chain) — only the
//      column -> (block, thread) assignment changed, so every x[i] value
//      is bit-identical. The stripe is contiguous (each column belongs to
//      exactly one block; n/G columns per block, remainder in the tail
//      blocks).
//   B. stage_rms_out (block 0 only): RMSNorm row in the ORIGINAL order —
//      per-thread strided squared sums (i = tid; i < n; i += 256) over the
//      rounded x values, the 256-slot tree, rsqrt, then the out pass with
//      the same v[k] * rstd * w[i] f16 chain. Bit-level: pass B re-reads
//      x[i] (which pass A left as the rounded sum), so its operands are
//      exactly the register-cached values the single-block stage used.
//      The out pass stays in block 0, so rstd never leaves the block and
//      no extra broadcast barrier is needed.
// Two new full-grid barriers (slots 7/8) order A -> B, and bar3/bar6
// become full-grid (every block now consumes the p1_o / p2_gu_d
// partials). Serves the o projection (x += o, rms -> xn with the layer's
// ffn_norm) and the down projection (x += down, rms -> xn with the next
// layer's attn_norm / final_norm).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void stage_add_columns(const float* partials,
                                                  unsigned short* x,
                                                  int n, int nslabs,
                                                  int bx, int G, int tid) {
    const int cols = (n + G - 1) / G;
    const int my = bx * cols;
    const int me = my + cols < n ? my + cols : n;
    for (int i = my + tid; i < me; i += blockDim.x) {
        const float* p = partials + i;
        float acc = 0.0f;
        int s = 0;
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
        x[i] = f32_to_hbits(sum);
    }
}

__device__ __forceinline__ void stage_rms_out(unsigned short* x,
                                              unsigned short* out,
                                              const unsigned short* w,
                                              int n, float eps, int tid) {
    __shared__ float s_sh[256];
    float v[4];
    int cnt = 0;
    if (tid < 256) {
        float s = 0.f;
        for (int i = tid; i < n; i += 256, ++cnt) {
            float val = hbits_to_f32(x[i]);
            v[cnt] = val;
            s += val * val;
        }
        s_sh[tid] = s;
    }
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
    if (tid < 256) {
        for (int i = tid, k = 0; i < n; i += 256, ++k) {
            out[i] = f32_to_hbits(v[k] * rstd * hbits_to_f32(w[i]));
        }
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------
// Stage 7: gate/up phase-2 + fused cast-SiLU-GLU + the down plan's phase-1
// in one stage — gemv_p2_gu_p1d_swiglu with 512-col stripes. Tile =
// (c2, slab) of the down plan (c2 = tile / nslabs_d, slab = tile %
// nslabs_d). The block redundantly writes the 512-col stripe its phase-2
// k-range lies in (c1 = (tile % nslabs_d) / (512 / slab_k)), then computes
// its down tile block-locally (the build gate admits slab_k in
// {64, 128, 256, 512} with 512 % slab_k == 0 and the stripes covering the
// whole k-range). Per-column arithmetic identical to the S1-9 kernel.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void stage_p2_gu_d(const LayerFusedConst* c,
                                              const PlanRow* table,
                                              int tile, int tid) {
    const PlanRow rg = table[4];
    const PlanRow ru = table[5];
    const PlanRow rd = table[6];

    const int slab_k = (rd.k + rd.nslabs - 1) / rd.nslabs;
    int c1 = (tile % rd.nslabs) / (512 / slab_k);
    int col = c1 * 512 + tid;
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

    int slab = tile % rd.nslabs;
    int ks = slab * slab_k;
    int ke = ks + slab_k;
    if (ke > rd.k) {
        ke = rd.k;
    }
    int c2 = tile / rd.nslabs;
    int ccmax = c2 * 512 + 512;
    if (ccmax > rd.n) {
        ccmax = rd.n;
    }
    for (int cc = c2 * 512 + tid; cc < ccmax; cc += 512) {
        gemv_phase1(rd, cc, ks, ke, slab);
    }
}

// 017-d: stage_p2_gu_d with WC-wide down phase-1 — the phase-2 part (silu
// stripe writes) is byte-identical to stage_p2_gu_d; only the phase-1
// column ownership widens: each down tile now covers 512*WC columns and
// thread tid owns the WC consecutive columns [tid*WC, tid*WC+WC) (host
// tile count tiles_gu_d = tiles_w(rd.n, rd.nslabs, 512*WC)). Each column
// is still computed exactly once with the identical k-walk.
template <int WC>
__device__ __forceinline__ void stage_p2_gu_d_wc(const LayerFusedConst* c,
                                                 const PlanRow* table,
                                                 int tile, int tid) {
    const PlanRow rg = table[4];
    const PlanRow ru = table[5];
    const PlanRow rd = table[6];

    const int slab_k = (rd.k + rd.nslabs - 1) / rd.nslabs;
    int c1 = (tile % rd.nslabs) / (512 / slab_k);
    int col = c1 * 512 + tid;
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

    int slab = tile % rd.nslabs;
    int ks = slab * slab_k;
    int ke = ks + slab_k;
    if (ke > rd.k) {
        ke = rd.k;
    }
    int c2 = tile / rd.nslabs;
    int col0 = c2 * (512 * WC) + tid * WC;
    if (col0 < rd.n) {
        int wc = rd.n - col0;
        if (wc > WC) {
            wc = WC;
        }
        gemv_phase1_wc<WC>(rd, col0, ks, ke, slab, wc);
    }
}

// ---------------------------------------------------------------------------
// Stage 0 (layer 0 only): gather + attn rms(0) — both redundant across
// blocks (every block copies the embed row and computes the full rms row
// with the exact rms_norm_row_f16 arithmetic, threads 0..255), so the
// p1_qkv stage reads its own block's xn bits (all writers are identical —
// no cross-block ordering needed).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void stage_gather_rms0(const LayerFusedConst* c,
                                                  unsigned int token, int tid) {
    __shared__ float s_sh[256];
    // gather_row's grid is ceil(n/256) blocks of 256 threads; here the
    // same byte copy runs on all 512 threads (stride 512, identical
    // destination bytes — the rms pass below re-reads x, so no
    // cross-block ordering matters; the values are block-local).
    for (int i = tid; i < c->h; i += 512) {
        c->x[i] = c->embed[(size_t)token * c->h + i];
    }
    __syncthreads();
    float s = 0.f;
    if (tid < 256) {
        for (int i = tid; i < c->h; i += 256) {
            float v = hbits_to_f32(c->x[i]);
            s += v * v;
        }
        s_sh[tid] = s;
    }
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (tid < st) {
            s_sh[tid] += s_sh[tid + st];
        }
        __syncthreads();
    }
    if (tid == 0) {
        float mean_sq = s_sh[0] / (float)c->h;
        s_sh[0] = rsqrtf(mean_sq + c->eps);
    }
    __syncthreads();
    float rstd = s_sh[0];
    if (tid < 256) {
        for (int i = tid; i < c->h; i += 256) {
            float v = hbits_to_f32(c->x[i]) * rstd * hbits_to_f32(c->wnorm0[i]);
            c->xn[i] = f32_to_hbits(v);
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void stage_ts_mark(const LayerFusedConst* c, int li,
                                              int stage) {
    if (blockIdx.x == 0 && threadIdx.x == 0 && c->stage_ts != 0) {
        c->stage_ts[(size_t)li * 9 + stage] = (unsigned int)clock64();
    }
}

// ---------------------------------------------------------------------------
// The layer kernel: 8 stages, 7 grid barriers, grid = G co-resident
// blocks (host-computed), 512 threads.
//
// Partial-participant barriers (S1-10b): every barrier guards a specific
// (producer set -> consumer set) dependency, and the producer/consumer
// block sets are always the prefix {bx < P} of the grid. Blocks outside a
// barrier's participant prefix neither produce nor consume anything the
// barrier protects — they SKIP the barrier and race ahead to the next
// stage, which itself waits on the barrier that guards its own inputs:
//   - bar0 (all G):  p1_qkv partials (every block >= 1 tile) -> p2_qkv.
//   - bar1 (P1):     p2_qkv q/k/v (blocks < cq+2ck) -> flash q rows (blocks
//                    < q_heads). P1 = max(cq+2ck, q_heads) — the consumers
//                    that are also producers arrive after their stage work.
//   - bar2 (P2):     flash attn (blocks < q_heads) -> p1_o tiles (blocks <
//                    tiles_o). P2 = max(q_heads, tiles_o).
//   - bar3 (P3):     p1_o partials (blocks < tiles_o) -> add_rms(o) (block
//                    0, a p1_o producer itself). P3 = tiles_o.
//   - bar4 (P4):     xn written by block 0 (add_rms(o)) -> p1_gu tiles
//                    (blocks < tiles_gu). P4 = tiles_gu; block 0 arrives
//                    last (after its add_rms), the others spin.
//   - bar5 (P5):     p1_gu partials (blocks < tiles_gu) -> p2_gu_d tiles
//                    (blocks < tiles_gu_d). P5 = max(tiles_gu, tiles_gu_d).
//   - bar6 (P6):     p2_gu_d partials (blocks < tiles_gu_d) -> add_rms(down)
//                    (block 0). P6 = tiles_gu_d.
// Blocks with no tile in a stage (bx >= the stage's tile count) skip the
// stage AND its barrier — they reach the next barrier's participant prefix
// only if that stage is theirs, otherwise they exit. The release of every
// partial barrier therefore still orders exactly the writes its consumers
// read. The skipped early-exit blocks hold no locks — co-residency is only
// required while the kernel runs, and blocks may retire at any time.
// All P values are clamped to G (grid-stride stages: every block then
// participates).
//
// S1-11 widening: the widened stages (p1_qkv, p2_qkv, p1_gu) run PAIRED
// tw-col tiles (one adjacent pair per block), so their producer sets are
// {bx < tiles/2} — a subset of the prefix the P formulas compute from the
// widened tile counts, which the host uploads per width (the participant
// sets therefore extend automatically; with W=2 and Qwen3-0.6B all the
// qkv/gu tile counts exceed the grid, so bar4/bar5 become full-grid).
//
// 017-d: with WC > 1 the p1 stages' tiles are 256*WC columns wide (see
// p1_tiles<W, WC>) and stage 7's tiles 512*WC wide — the producer sets
// shrink accordingly, and since every producer set stays a prefix of the
// grid and the P formulas are min(count, G) upper bounds, the barrier
// participant sets extend automatically (p1 producers {bx < count/2}
// are still a subset of the {bx < count} prefix; the excess prefix
// blocks arrive without work). WC only changes which columns each
// thread owns — never the per-column arithmetic.
// ---------------------------------------------------------------------------
template <int W, int WC>
__device__ __forceinline__ void layer_fused_body(
    const LayerFusedConst* c,
    const PlanRow* table,       // this layer's 7 rows
    const unsigned short* wnext,  // next attn_norm / final_norm
    const unsigned short* wffn,   // this layer's ffn_norm
    const unsigned short* wq,     // this layer's q head-norm weights
    const unsigned short* wk,     // this layer's k head-norm weights
    volatile unsigned int* bar,   // [2 * N_BARRIERS] u32 (zeroed once)
    int li, int n_layers, unsigned int pos, unsigned int token) {
    const int tid = threadIdx.x;
    const int bx = blockIdx.x;
    const int G = gridDim.x;
    // 10 barrier slots (2 * N_BARRIERS u32): index k guards stage k -> k+1
    // (bar0..bar6 full-grid/participant, bar7/bar8 the add_rms split
    // barriers, bar9 spare).
    volatile unsigned int* bcnt = bar;
    volatile unsigned int* bgen = bar + 10;
    const int tw = W == 1 ? 512 : 256 * WC;
    // Stage-2 (p2_qkv) keeps its own 256-col (W=2) / 512-col (W=1) tiles
    // regardless of WC — phase-2 only sums per-column partials, so its
    // tile width is independent of phase-1's.
    const int cq = W == 1 ? (c->nqk + 511) / 512 : (c->nqk + 255) / 256;
    const int ck = W == 1 ? (c->kvk + 511) / 512 : (c->kvk + 255) / 256;
    const int P1m = (cq + 2 * ck) < c->q_heads ? c->q_heads : (cq + 2 * ck);
    const int P1 = P1m < G ? P1m : G;
    const int P2m = c->q_heads < c->tiles_o ? c->tiles_o : c->q_heads;
    const int P2 = P2m < G ? P2m : G;
    const int P4 = c->tiles_gu < G ? c->tiles_gu : G;
    const int P5m = c->tiles_gu < c->tiles_gu_d ? c->tiles_gu_d : c->tiles_gu;
    const int P5 = P5m < G ? P5m : G;

    stage_ts_mark(c, li, 0);
    if (li == 0) {
        stage_gather_rms0(c, token, tid);
    }
    // 1. phase-1 of the q/k/v plans (reads xn)
    p1_tiles<W, WC>(c, table, 0, 3, c->tiles_qkv, tw, bx, G, tid);
    stage_ts_mark(c, li, 1);
    grid_barrier(&bcnt[0], &bgen[0], G);
    // 2. q/k/v phase-2 + casts + head-norm + rope (cq + 2*ck tiles)
    p2qkv_tiles<W>(c, table, cq + 2 * ck, bx, G, tid, pos, wq, wk);
    stage_ts_mark(c, li, 2);
    if (bx < P1) {
        grid_barrier(&bcnt[1], &bgen[1], P1);
    }
    // 3. fused flash (kv write + attention): one tile per q head
    for (int h = bx; h < c->q_heads; h += G) {
        stage_flash(c, h, tid, li);
    }
    stage_ts_mark(c, li, 3);
    if (bx < P2) {
        grid_barrier(&bcnt[2], &bgen[2], P2);
    }
    // 4. o phase-1 (reads the full attention row)
    p1_tiles_plain(c, table, 3, 1, c->tiles_o, bx, G, tid);
    stage_ts_mark(c, li, 4);
    grid_barrier(&bcnt[3], &bgen[3], G);
    // 5. o phase-2 + residual add (all blocks) + ffn rms (block 0)
    stage_add_columns(table[3].partials, c->x, c->h, c->nslabs_o, bx, G, tid);
    grid_barrier(&bcnt[7], &bgen[7], G);
    if (bx == 0) {
        stage_rms_out(c->x, c->xn, wffn, c->h, c->eps, tid);
    }
    stage_ts_mark(c, li, 5);
    if (bx < P4) {
        grid_barrier(&bcnt[4], &bgen[4], P4);
    }
    // 6. gate/up phase-1 (reads xn = p2_o's ffn-normed x)
    p1_tiles<W, WC>(c, table, 4, 2, c->tiles_gu, tw, bx, G, tid);
    stage_ts_mark(c, li, 6);
    if (bx < P5) {
        grid_barrier(&bcnt[5], &bgen[5], P5);
    }
    // 7. gate/up phase-2 + swiglu + down phase-1 (one tile per down
    //    (col-block, slab) — the split's full down phase-1 grid; 017-d:
    //    WC > 1 widens each tile to 512*WC columns)
    if constexpr (WC > 1) {
        for (int t = bx; t < c->tiles_gu_d; t += G) {
            stage_p2_gu_d_wc<WC>(c, table, t, tid);
        }
    } else {
        for (int t = bx; t < c->tiles_gu_d; t += G) {
            stage_p2_gu_d(c, table, t, tid);
        }
    }
    stage_ts_mark(c, li, 7);
    grid_barrier(&bcnt[6], &bgen[6], G);
    // 8. down phase-2 + residual add (all blocks) + next attn rms (block 0)
    stage_add_columns(table[6].partials, c->x, c->h, c->nslabs_d, bx, G, tid);
    grid_barrier(&bcnt[8], &bgen[8], G);
    if (bx == 0) {
        stage_rms_out(c->x, c->xn, wnext, c->h, c->eps, tid);
    }
    stage_ts_mark(c, li, 8);
}

// W=1 entry: the S1-10 kernel verbatim (REINFER_FUSED_BW=off).
extern "C" __global__ void __launch_bounds__(512) decode_step_layer_fused(
    const LayerFusedConst* __restrict__ c,
    const PlanRow* __restrict__ table,       // this layer's 7 rows
    const unsigned short* __restrict__ wnext,  // next attn_norm / final_norm
    const unsigned short* __restrict__ wffn,   // this layer's ffn_norm
    const unsigned short* __restrict__ wq,     // this layer's q head-norm weights
    const unsigned short* __restrict__ wk,     // this layer's k head-norm weights
    volatile unsigned int* __restrict__ bar,   // [2 * N_BARRIERS] u32 (zeroed once)
    int li, int n_layers, unsigned int pos, unsigned int token) {
    layer_fused_body<1, 1>(c, table, wnext, wffn, wq, wk, bar, li, n_layers,
                           pos, token);
}

// W=2 entry (S1-11 block-width): __launch_bounds__(512, 2) forces 64
// registers/thread so two blocks are co-resident per SM (grid = 2x the
// W=1 grid; the host gates it by the occupancy query).
extern "C" __global__ void __launch_bounds__(512, 2) decode_step_layer_fused_bw2(
    const LayerFusedConst* __restrict__ c,
    const PlanRow* __restrict__ table,       // this layer's 7 rows
    const unsigned short* __restrict__ wnext,  // next attn_norm / final_norm
    const unsigned short* __restrict__ wffn,   // this layer's ffn_norm
    const unsigned short* __restrict__ wq,     // this layer's q head-norm weights
    const unsigned short* __restrict__ wk,     // this layer's k head-norm weights
    volatile unsigned int* __restrict__ bar,   // [2 * N_BARRIERS] u32 (zeroed once)
    int li, int n_layers, unsigned int pos, unsigned int token) {
    layer_fused_body<2, 1>(c, table, wnext, wffn, wq, wk, bar, li, n_layers,
                           pos, token);
}

// 017-d WC=2 entry: W=2 pairing plus 2 consecutive columns per thread
// (LDG.32 b loads; tiles 512/1024 cols wide). Still __launch_bounds__
// (512, 2) — 64 registers/thread, two blocks per SM.
extern "C" __global__ void __launch_bounds__(512, 2) decode_step_layer_fused_bw2_wc2(
    const LayerFusedConst* __restrict__ c,
    const PlanRow* __restrict__ table,       // this layer's 7 rows
    const unsigned short* __restrict__ wnext,  // next attn_norm / final_norm
    const unsigned short* __restrict__ wffn,   // this layer's ffn_norm
    const unsigned short* __restrict__ wq,     // this layer's q head-norm weights
    const unsigned short* __restrict__ wk,     // this layer's k head-norm weights
    volatile unsigned int* __restrict__ bar,   // [2 * N_BARRIERS] u32 (zeroed once)
    int li, int n_layers, unsigned int pos, unsigned int token) {
    layer_fused_body<2, 2>(c, table, wnext, wffn, wq, wk, bar, li, n_layers,
                           pos, token);
}

// 017-d WC=4 entry: 4 consecutive columns per thread (LDG.64 b loads;
// tiles 1024/2048 cols wide).
extern "C" __global__ void __launch_bounds__(512, 2) decode_step_layer_fused_bw2_wc4(
    const LayerFusedConst* __restrict__ c,
    const PlanRow* __restrict__ table,       // this layer's 7 rows
    const unsigned short* __restrict__ wnext,  // next attn_norm / final_norm
    const unsigned short* __restrict__ wffn,   // this layer's ffn_norm
    const unsigned short* __restrict__ wq,     // this layer's q head-norm weights
    const unsigned short* __restrict__ wk,     // this layer's k head-norm weights
    volatile unsigned int* __restrict__ bar,   // [2 * N_BARRIERS] u32 (zeroed once)
    int li, int n_layers, unsigned int pos, unsigned int token) {
    layer_fused_body<2, 4>(c, table, wnext, wffn, wq, wk, bar, li, n_layers,
                           pos, token);
}
