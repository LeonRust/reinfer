// S2-B: batch-decode companion kernels (specs/005 S2-B — B requests x 1
// token merged forward).
//
// The batch decode step runs the projections as GEMMs with m = B (the rows
// of the B requests share the model weights) and the per-request attention
// as a per-request loop. The row-wise kernels below mirror the
// single-request kernels' math per element:
// - rope_neox_batch_f16 == rope_heads_f16 per (request, head) row, with the
//   rope position taken per request from a device array (the single-request
//   kernel takes one position; batch requests each carry their own pos).
//
// f16 bit helpers duplicated from dense_kernels.cu (same semantics:
// f32_to_hbits = RNE bit construction, hbits_to_f32 = software expand).

__device__ __forceinline__ float hbits_to_f32(unsigned short h) {
    unsigned int sign = (unsigned int)(h >> 15) << 31;
    unsigned int exp = (h >> 10) & 0x1f;
    unsigned int man = h & 0x03ff;
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

// S2-B+: batched KV-slot write — grid = B blocks (256 threads; the per-token
// KV row is kv_heads*d <= 1024 f16), replacing the B per-request kv_write
// launches of the per-request attention loop with one launch. Request b
// writes its (k_rows, v_rows) row to the slot (phys_b, off_b) of its pool:
//   phys_b = pages[(b*n_layer + li)*pp] + pos[b]/block_len (the identity
//   page table's page[0] = the layer base — the write slot the engine's
//   per-request kv_write computed as base_pages + li*pp + lp),
//   off_b  = pos[b] % block_len.
// The copy is byte-identical to kv_write_row (dense_kernels.cu): k at
// kv[slot + c], v at kv[v_base + slot + c], slot = (phys*block_len + off)
// * per_tok, v_base = total_pages*block_len*per_tok. pp is derived as
// max_kv / block_len (engine contract: pp * block_len == max_kv).
extern "C" __global__ void kv_write_batch_f16(
    const unsigned short* __restrict__ k_rows,   // [B][per_tok] f16
    const unsigned short* __restrict__ v_rows,   // [B][per_tok] f16
    const unsigned short* const* __restrict__ kv,       // [B] pool K-region bases
    const unsigned int* __restrict__ pages,      // [B][n_layer][pp] identity tables
    const unsigned int* __restrict__ pos,        // [B] per-request positions
    int B, int block_len, int kv_heads, int d, int total_pages,
    int n_layer, int li, int max_kv) {
    int b = blockIdx.x;
    if (b >= B) {
        return;
    }
    int pp = max_kv / block_len;
    int lp = pos[b] / block_len;
    int off = pos[b] % block_len;
    unsigned int phys = pages[((size_t)b * n_layer + li) * pp] + lp;
    int per_tok = kv_heads * d;
    const size_t slot = ((size_t)phys * block_len + off) * per_tok;
    const size_t v_base = (size_t)total_pages * block_len * per_tok;
    const unsigned short* kr = k_rows + (size_t)b * per_tok;
    const unsigned short* vr = v_rows + (size_t)b * per_tok;
    unsigned short* kvb = (unsigned short*)kv[b];
    for (int c = threadIdx.x; c < per_tok; c += blockDim.x) {
        kvb[slot + c] = kr[c];
        kvb[v_base + slot + c] = vr[c];
    }
}

// Batch half-split RoPE (NEOX) with optional element scale — one block per
// head row (grid = B*heads, block 256; half <= 256 by contract). Row r ->
// request b = r/heads, head h = r%heads; the position comes from the
// device array `pos` (per-request, unlike the single-request kernel's
// scalar position). The theta formula and the f16 double-rounding with
// scale are identical to rope_heads_f16 (dense_kernels.cu): for scale=1.0
// (k heads) the f16 rounding is idempotent, so the result is bit-identical
// to plain rope.
extern "C" __global__ void rope_neox_batch_f16(
    unsigned short* __restrict__ x,
    const unsigned int* __restrict__ pos,  // [B] per-request positions
    int B, int heads, int half, float eta, float scale) {
    int row = blockIdx.x;
    if (row >= B * heads) {
        return;
    }
    int p = threadIdx.x;
    if (p >= half) {
        return;
    }
    int b = row / heads;
    unsigned short* xr = x + (size_t)row * 2 * half;
    // Same theta formula as rope_heads_f16 (full-dimension frequency
    // denominator): theta = pos[b] * eta^(-2p/(2*half)).
    float theta = (float)pos[b] * powf(eta, -2.f * (float)p / (2.f * (float)half));
    float c = cosf(theta), s = sinf(theta);
    float a = hbits_to_f32(xr[p]);
    float bv = hbits_to_f32(xr[p + half]);
    float v1 = a * c - bv * s;
    float v2 = a * s + bv * c;
    xr[p] = f32_to_hbits(hbits_to_f32(f32_to_hbits(v1)) * scale);
    xr[p + half] = f32_to_hbits(hbits_to_f32(f32_to_hbits(v2)) * scale);
}
