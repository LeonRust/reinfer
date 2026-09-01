// Batched-prefill FMHA kernel (specs/006 D2 Jit(fmha)).
//
// Own thin wrapper around the vendored flash-attn v2.8.3 forward kernel
// (see crates/cuda/vendor/fmha/README.md and version.json for provenance):
// flash_fwd_kernel.h is instantiated through a single exported kernel
// below, compiled by JitCache with the per-arch gencode flags.
//
// Design (one line): compute_attn<Flash_fwd_kernel_traits<128,128,128,4,
// false,false,cutlass::half_t>, no-dropout, causal, no-local, no-alibi,
// Is_even_MN, Is_even_K=true, no-softcap, no-return-softmax> over a
// contiguous [S,B,nqk] Q/K/V/O (fed via affine strides, no transposes),
// GQA via h_k, rotary_dim=0 (RoPE applied by the engine before the call),
// q pre-scaled by 1/sqrt(d) with scale_softmax=1.0 so the score math
// matches the per-token decode path.
//
// The upstream params struct (csrc/flash_attn/src/flash.h) depends on
// torch; this file mirrors it field-for-field with the torch-free philox
// state from the vendored philox_unpack.cuh shim.

#include "flash_fwd_kernel.h"  // vendored; -I <vendor>/fmha/headers

// ---------------------------------------------------------------------------
// Params (mirror of upstream Flash_fwd_params; index_t = int64_t)
// ---------------------------------------------------------------------------

namespace reinfer_fmha {

struct Flash_fwd_params {
    using index_t = int64_t;

    // QKV matrices.
    void* __restrict__ q_ptr;
    void* __restrict__ k_ptr;
    void* __restrict__ v_ptr;

    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;

    // Number of heads; GQA: h_k < h, h_h_k_ratio = h / h_k.
    int h, h_k;
    int h_h_k_ratio;

    // O matrix.
    void* __restrict__ o_ptr;
    void* __restrict__ oaccum_ptr;
    index_t o_batch_stride;
    index_t o_row_stride;
    index_t o_head_stride;

    // P matrix (Return_softmax only; null here).
    void* __restrict__ p_ptr;

    // Softmax LSE.
    void* __restrict__ softmax_lse_ptr;
    void* __restrict__ softmax_lseaccum_ptr;

    // Dimensions.
    int b, seqlen_q, seqlen_k, seqlen_knew, d, seqlen_q_rounded,
        seqlen_k_rounded, d_rounded, rotary_dim, total_q;

    // Scaling factors.
    float scale_softmax;
    float scale_softmax_log2;

    // Varlen offsets (null for fixed-length batches).
    int* __restrict__ cu_seqlens_q;
    int* __restrict__ cu_seqlens_k;
    int* __restrict__ leftpad_k;
    int* __restrict__ seqused_k;
    int* __restrict__ blockmask;

    // K_new / V_new (append-KV; null here).
    void* __restrict__ knew_ptr;
    void* __restrict__ vnew_ptr;
    index_t knew_batch_stride;
    index_t vnew_batch_stride;
    index_t knew_row_stride;
    index_t vnew_row_stride;
    index_t knew_head_stride;
    index_t vnew_head_stride;

    // Rotary (null; rotary_dim = 0 skips the branch in the kernel).
    void* __restrict__ rotary_cos_ptr;
    void* __restrict__ rotary_sin_ptr;

    // KV-cache batch remap (null).
    int* __restrict__ cache_batch_idx;

    // Paged KV (null; prefill reads fresh K/V, not the page store).
    int* __restrict__ block_table;
    index_t block_table_batch_stride;
    int page_block_size;

    // Dropout (disabled at compile time; kept for layout fidelity).
    float p_dropout;
    uint8_t p_dropout_in_uint8_t;
    float rp_dropout;
    float scale_softmax_rp_dropout;

    // Local window / softcap (unused: Is_local=false, Is_softcap=false).
    int window_size_left, window_size_right;
    float softcap;

    // RNG state (torch-free stand-in, see vendored philox_unpack.cuh).
    at::cuda::philox::PhiloxCudaState philox_args;

    // Only touched when Is_dropout; kept for layout fidelity.
    uint64_t* rng_state;

    bool is_bf16;
    bool is_causal;

    // Varlen bookkeeping (unused: fixed-length batches).
    bool is_seqlens_k_cumulative;
    bool is_rotary_interleaved;

    int num_splits;  // Split-KV (unused: single split).

    // Alibi (unused: Has_alibi=false).
    void* __restrict__ alibi_slopes_ptr;
    index_t alibi_slopes_batch_stride;

    // LSE layout (both false: LSE written as [b, h, seqlen_q] f32).
    bool unpadded_lse;
    bool seqlenq_ngroups_swapped;
};

}  // namespace reinfer_fmha

// ---------------------------------------------------------------------------
// Exported kernels
// ---------------------------------------------------------------------------

// Is_even_MN (seqlen % block_m == 0) and Is_even_K (seqlen % 64 == 0) are
// selected per launch shape; the loader picks the matching symbol.
//
// Variant sets (S1-7 FMHA heuristics; same math, different tile geometry —
// see fmha.rs pick() for the selection data):
//   v0 128x128x128,  4 warps — baseline (the original prefill kernel)
//   v1 128x128x128,  8 warps — more threads per CTA (256), same smem
//   v2 128x64x128,   4 warps — half-size KV tile, 64 KiB smem (more
//                              concurrent CTAs per SM)
//   v3 256x128x128,  8 warps — double Q block, 128 KiB smem, half the CTAs
#define REINFER_FMHA_KERNEL(NAME, BM, BN, NWARPS, IS_EVEN_MN, IS_EVEN_K)       \
    extern "C" __global__ void NAME(const reinfer_fmha::Flash_fwd_params params) { \
        flash::compute_attn<                                                   \
            Flash_fwd_kernel_traits<BM, BN, 128, NWARPS, false, false,         \
                                    cutlass::half_t>,                          \
            /*Is_dropout=*/false, /*Is_causal=*/true, /*Is_local=*/false,      \
            /*Has_alibi=*/false, /*Is_even_MN=*/IS_EVEN_MN,                    \
            /*Is_even_K=*/IS_EVEN_K, /*Is_softcap=*/false,                     \
            /*Return_softmax=*/false>(params);                                 \
    }

REINFER_FMHA_KERNEL(fmha_v0_mn_even_k_even, 128, 128, 4, true, true)
REINFER_FMHA_KERNEL(fmha_v0_mn_even_k_odd, 128, 128, 4, true, false)
REINFER_FMHA_KERNEL(fmha_v0_mn_odd_k_even, 128, 128, 4, false, true)
REINFER_FMHA_KERNEL(fmha_v0_mn_odd_k_odd, 128, 128, 4, false, false)

REINFER_FMHA_KERNEL(fmha_v1_mn_even_k_even, 128, 128, 8, true, true)
REINFER_FMHA_KERNEL(fmha_v1_mn_even_k_odd, 128, 128, 8, true, false)
REINFER_FMHA_KERNEL(fmha_v1_mn_odd_k_even, 128, 128, 8, false, true)
REINFER_FMHA_KERNEL(fmha_v1_mn_odd_k_odd, 128, 128, 8, false, false)

REINFER_FMHA_KERNEL(fmha_v2_mn_even_k_even, 128, 64, 4, true, true)
REINFER_FMHA_KERNEL(fmha_v2_mn_even_k_odd, 128, 64, 4, true, false)
REINFER_FMHA_KERNEL(fmha_v2_mn_odd_k_even, 128, 64, 4, false, true)
REINFER_FMHA_KERNEL(fmha_v2_mn_odd_k_odd, 128, 64, 4, false, false)

REINFER_FMHA_KERNEL(fmha_v3_mn_even_k_even, 256, 128, 8, true, true)
REINFER_FMHA_KERNEL(fmha_v3_mn_even_k_odd, 256, 128, 8, true, false)
REINFER_FMHA_KERNEL(fmha_v3_mn_odd_k_even, 256, 128, 8, false, true)
REINFER_FMHA_KERNEL(fmha_v3_mn_odd_k_odd, 256, 128, 8, false, false)
