// Fused Q8_0 dequant-dot decode kernel (006 T6; sm90+ decode gate
// precondition per specs/006-cuda-perf/spec.md: the 0.85x llama.cpp CUDA
// decode gate is only judged after this kernel lands).
//
// Semantics (aligned with the existing "dequant -> fp16 device buffer ->
// GEMM" path, 003 D3 / 014 D4, with CUBLAS_COMPUTE_32F accumulation as the
// engine dense path, engine.rs gemm1):
//
//   out[n] = sum_k f32(y_nk) * f32(x_k)
//   y_nk   = RNE_f16(f32(q_nk) * f32(f16(scale_nk)))   -- in-register dequant
//   x_k    = activation row, f16 bits (decode: M = 1)
//
// y_nk is bit-exact with dequant_q8_0 (single f32 multiply, same
// half_bits_to_f32) followed by cast_f32_to_f16 (single RNE rounding): the
// Q8_0 block format is 2-byte fp16 scale (little endian) + 32 int8
// (QK8_0 = 32, 34 B/block). Accumulation is fp32 (matches the 003 dense
// 32F-acc gate tier; D7 table: f16-in/f32-out rel 1e-4 + atol 1e-6).
// Output is the per-layer EP f32 result; residual/add belong to engine
// integration.
//
// Layout (unified 8-column tile per thread block): one block = 8 output
// columns x full K; 256 threads = 8 warps; warp w owns output column
// n0 + w. Thread t of warp w accumulates element positions k = t, t+32,
// ... (fixed stride-32 order over K), then a fixed 5-step butterfly warp
// reduction writes out[n0 + w]. Deterministic: no atomics, fixed tree
// order; no arch-specific instructions (portable; gencode per target arch).

// fp16 -> fp32 bit construction (same semantics as crates/gguf f16_to_f32
// and llama.cpp ggml_fp16_to_fp32): integer-only expansion, subnormal
// normalize, NaN payload preserved, inf passthrough. Verified against
// dequant_kernels.cu (014 T5 0-ulp gate); __half2float is not used (hardware
// conversion normalizes NaN payloads, breaking bit-exactness on adversarial
// random scale bits).
__device__ __forceinline__ float hbits_to_f32(unsigned short h) {
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

// f32 -> fp16 bits, RNE (same semantics as dense_kernels.cu f32_to_hbits
// and engine.rs f32_to_f16_bits): software rounding, inf/NaN truncation,
// subnormal flush at the -10 exponent boundary.
__device__ __forceinline__ unsigned short f32_to_hbits(float f) {
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

// One thread block per 8 output columns; grid = ceil(n / 8), block = 256
// threads (8 warps). k must be a multiple of 32 (Q8_0 block count).
// w: row-major [n x k] Q8_0 blob, 34 B per 32-element block.
// x: f16 bits, k elements. out: f32, n elements.
extern "C" __global__ void fused_q8_dot(
    const unsigned short* __restrict__ x,
    const unsigned char* __restrict__ w,
    float* __restrict__ out,
    int n,
    int k) {
    int kb = k >> 5;  // Q8_0 blocks per row (caller guarantees k % 32 == 0)
    if (kb <= 0) {
        return;
    }
    int n0 = blockIdx.x * 8;
    int wcol = threadIdx.x >> 5;  // warp -> output column within the tile
    int lane = threadIdx.x & 31;  // element position within a Q8_0 block
    int col = n0 + wcol;
    float acc = 0.0f;
    if (col < n) {
        const unsigned char* wrow = w + (size_t)col * (size_t)kb * 34;
        for (int b = 0; b < kb; ++b) {
            const unsigned char* blk = wrow + (size_t)b * 34;
            unsigned short bits = (unsigned short)blk[0] | ((unsigned short)blk[1] << 8);
            float d = hbits_to_f32(bits);
            signed char q = (signed char)blk[2 + lane];
            // In-register dequant: f32 single multiply -> single RNE rounding
            // to fp16; the f16 value is never written back to memory.
            unsigned short y = f32_to_hbits((float)q * d);
            acc += hbits_to_f32(y) * hbits_to_f32(x[(size_t)b * 32 + lane]);
        }
    }
    // Fixed 5-step butterfly reduction (deterministic order, no atomics).
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, off);
    }
    if (lane == 0 && col < n) {
        out[col] = acc;
    }
}
