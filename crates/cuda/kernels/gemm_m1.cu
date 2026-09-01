// JIT m=1 GEMM (decode-path gemv): C[1 x n] = A[1 x k] x B[k x n].
//
// The decode step's projection GEMMs (q/k/v/o/gate/up/down per layer plus
// lm_head) all run with m = 1: a single activation row times the weight
// matrix. cuBLAS COMPUTE_32F delivers ~15% SM efficiency on these skinny
// shapes (S1-1 profile: qkv 2.74 + ffn 3.76 + o 1.49 + lm_head 0.43 =
// 8.42 ms/step of GEMM-family cost), while the same bytes are bandwidth-
// bound work: every plan is a pure dot-product row, one C value per output
// column. This kernel pair replaces cublas for exactly those shapes (see
// `Jgemm::matches` in gemm.rs — m == 1, f16 in, f32 out, row-major
// [k x n] B with ld = n, i.e. `GemmPlan::row_major_f16` with m == 1).
//
// Semantics (both kernels, phase 2 sums the phase-1 slab partials):
//   c[c] = sum_k f32(a[k]) * f32(b[k*n + c])   (a: [k] f16 row; b: [k*n] f16
//   row-major; c: [n] f32). f16 -> f32 via the hardware F2F instruction
//   (exact IEEE conversion for finite values; the decode activations/weights
//   are finite, and the judged quantity is the fp32 accumulation vs the
//   cublas COMPUTE_32F referee at the 014 D7 tolerance tier — drift from
//   accumulation-order differences is expected at ~1e-6..1e-5 rel, recorded,
//   not bit-identical).
//
// Parallelism: phase 1 splits k into nslabs slabs across the grid, because
// a thread-per-column kernel with grid = ceil(n/256) has only 4..12 blocks
// for the decode shapes (o n=1024, qkv n=1536, ffn n=3072) — far too few
// threads in flight to cover DRAM latency (measured ~3x SLOWER than cublas
// on those shapes; lm_head n=151936 was already bandwidth-bound at parity
// with cublas). grid1 = ncols * nslabs (nslabs chosen in Jgemm::launch so
// the grid has ~96 blocks, capped by k/32), block = 256; one thread owns
// one output column within its (column-block, slab) tile and walks the
// slab's k range in fixed stride-4 steps, accumulating the four consecutive
// k positions into four independent accumulators (ILP over the four
// positions; per-position guards keep every k >= 1 correct), then a
// fixed-order tree (acc0+acc1)+(acc2+acc3) -> partials[slab*n + col]
// (s-major layout: at one slab a warp's 32 columns are 128 contiguous
// bytes — one coalesced line for every partials read/write).
// Phase 2 (`gemv_m1_f16f32_reduce`) sums the nslabs partials per column in
// ascending slab order into c[col]. No atomics, no warp shuffles — every
// (col, slab) is computed by exactly one thread in a fixed order, so the
// result is deterministic (bit-identical across repeated launches).
//
// Why not stride-32 slicing like fused_q8_dot: there the 32 lanes of a
// warp each own one k residue and a butterfly reduction merges them; here
// a thread owns an entire column (the engine's B is row-major [k x n], so
// per-instruction coalescing requires consecutive threads to read
// consecutive COLUMNS at the same k). No cross-thread reduction exists in
// this design, so every thread must cover all of its k slab itself.
//
// Loads: read-only paths via __ldg (LDG.E.CONSTANT). Within a warp the 32
// threads at one k position touch b[k*n + c0 .. c0+31] — 64 contiguous
// bytes (coalesced read); the a[k] row is warp-uniform and stays in L1.
// The kernel is bandwidth-bound (decode m=1 plans stream ~700 MB/step);
// scalar 2-byte loads with full coalescing keep the LSU ~10x under the
// DRAM ceiling on sm_120a (4 warps x 4 pipes), so __half2 wide loads would
// not change the wall time and are not used (B column index parity would
// also misalign half2 for odd columns).

#include <cuda_fp16.h>

// Phase 1: per-(column-block, k-slab) partial dot products.
// grid = ncols * nslabs (linearized: bx / nslabs = column block,
// bx % nslabs = slab), block = 256. Writes partials[slab*n + col]
// (s-major, see the header).
extern "C" __global__ void gemv_m1_f16f32(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    float* __restrict__ partials,
    int n,
    int k,
    int nslabs) {
    int col = (blockIdx.x / nslabs) * 256 + threadIdx.x;
    if (col >= n) {
        return;
    }
    int slab = blockIdx.x % nslabs;
    int slab_k = (k + nslabs - 1) / nslabs;  // ceil; last slab is shorter
    int ks = slab * slab_k;
    int ke = ks + slab_k;
    if (ke > k) {
        ke = k;
    }
    // Four independent accumulators over four consecutive k positions
    // (ILP: independent FMA chains, loads can run ahead). Fixed order:
    // acc_i accumulates k = k0 + i, k0 ascending by 4.
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (int k0 = ks; k0 < ke; k0 += 4) {
        acc0 += __half2float(__ldg(&a[k0])) *
                __half2float(__ldg(&b[(size_t)k0 * n + col]));
        if (k0 + 1 < ke) {
            acc1 += __half2float(__ldg(&a[k0 + 1])) *
                    __half2float(__ldg(&b[(size_t)(k0 + 1) * n + col]));
        }
        if (k0 + 2 < ke) {
            acc2 += __half2float(__ldg(&a[k0 + 2])) *
                    __half2float(__ldg(&b[(size_t)(k0 + 2) * n + col]));
        }
        if (k0 + 3 < ke) {
            acc3 += __half2float(__ldg(&a[k0 + 3])) *
                    __half2float(__ldg(&b[(size_t)(k0 + 3) * n + col]));
        }
    }
    // Fixed-order combination (deterministic; different from the cublas
    // blocked reduction — the D7 drift record).
    partials[(size_t)slab * n + col] = (acc0 + acc1) + (acc2 + acc3);
}

// Phase 2: per-column reduction of the slab partials (ascending slab
// order — fixed, deterministic). grid = ceil(n/256), block = 256.
extern "C" __global__ void gemv_m1_f16f32_reduce(
    const float* __restrict__ partials,
    float* __restrict__ c,
    int n,
    int nslabs) {
    int col = blockIdx.x * 256 + threadIdx.x;
    if (col >= n) {
        return;
    }
    const float* p = partials + col;  // s-major: slab s at p[(size_t)s * n]
    float acc = 0.0f;
    for (int s = 0; s < nslabs; ++s) {
        acc += p[(size_t)s * n];
    }
    c[col] = acc;
}
