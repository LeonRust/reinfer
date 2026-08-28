// Q8_0 dequant kernel (014 T5). Block layout: 2-byte fp16 scale (RNE on
// read) + 32 int8 quantized values (QK8_0 = 32, 34 B/block). Semantics:
// `y = f32(q) * f32(f16(scale))` single-multiply — no FMA-ized `q*d+0.0f`
// writing, no double intermediate. Must be bit-exact vs
// crates/gguf::codes::dequantize_q8_0 (single-source semantics, 014 T2
// golden gate).
#include <math.h>

// ---------------------------------------------------------------------------
// fp16 -> fp32 bit construction (same semantics as crates/gguf f16_to_f32 and
// llama.cpp ggml_fp16_to_fp32): integer-only expansion — subnormal normalize,
// NaN payload preserved (no hardware quieting), inf passthrough. __half2float
// is *not* used: hardware conversion normalizes NaN payloads (observed
// 0x7fffffff vs reference 0xfff3e000 on cc 12.0), breaking the 0-ulp gate on
// adversarial random scale bits.
// ---------------------------------------------------------------------------
__device__ float half_bits_to_f32(unsigned short h) {
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

// ---------------------------------------------------------------------------
// One thread block per Q8_0 block; 34 raw bytes -> 32 f32 values.
// ---------------------------------------------------------------------------
extern "C" __global__ void dequant_q8_0(const unsigned char* __restrict__ blob,
                                        float* __restrict__ out,
                                        int nblocks) {
    int b = blockIdx.x;
    if (b >= nblocks) {
        return;
    }
    const unsigned char* blk = blob + (size_t)b * 34;
    unsigned short bits = (unsigned short)blk[0] | ((unsigned short)blk[1] << 8);
    float d = half_bits_to_f32(bits);
    const signed char* qs = reinterpret_cast<const signed char*>(blk + 2);
    for (int i = threadIdx.x; i < 32; i += blockDim.x) {
        out[(size_t)b * 32 + i] = (float)qs[i] * d;  // single multiply
    }
}
