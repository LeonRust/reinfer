// Batched-prefill companion kernels (006 T1).
//
// Row-wise variants of the dense per-token kernels (dense_kernels.cu) with
// the SAME math per row/token — bit-identical outputs by construction:
// - rms_norm_rows_f16  == rms_norm_row_f16 per row (grid = rows, block 256)
// - rope_neox_rows_f16 == rope_neox_f16 per (row = s*heads + h), pos = s
// - kv_write_seq_rows  == kv_write_row per token (page layout identical)
//
// This is what makes the FMHA prefill path pin-aligned with the per-token
// path on the same input (specs/006 T1 determinism requirement).
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

// 逐行 RMSNorm（grid = rows，block = 256；每行与 rms_norm_row_f16 相同）。
extern "C" __global__ void rms_norm_rows_f16(const unsigned short* __restrict__ x,
                                             unsigned short* __restrict__ out,
                                             const unsigned short* __restrict__ w,
                                             int rows, int n, float eps) {
    int row = blockIdx.x;
    const unsigned short* xr = x + (size_t)row * n;
    unsigned short* orr = out + (size_t)row * n;
    int tid = threadIdx.x;
    __shared__ float s_sh[256];
    float s = 0.f;
    for (int i = tid; i < n; i += 256) {
        float v = hbits_to_f32(xr[i]);
        s += v * v;
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
    for (int i = tid; i < n; i += 256) {
        float v = hbits_to_f32(xr[i]) * rstd * hbits_to_f32(w[i]);
        orr[i] = f32_to_hbits(v);
    }
}

// 批 RoPE（行主序 [seqlen × heads] 输入；row r → s = r/heads, h = r%heads，
// pos = s；每行与 rope_neox_f16 相同数学）。
//
// S1-7: rows-per-CTA batching — with half < 256 (d=128 → half=64) the old
// launch left 3/4 of the block idle; `rows_per_cta` rows share a block
// (thread t → row r = cta*rows_per_cta + t/half, pair p = t%half). With
// rows_per_cta = 256/half every thread is busy. The row math is fully
// per-element self-contained (no smem, no cross-thread reduction), so the
// element results are bit-identical for any rows_per_cta — only the
// CTA/thread that produces each element moves.
extern "C" __global__ void rope_neox_rows_f16(unsigned short* __restrict__ x,
                                              int half, int heads, int seqlen,
                                              float eta, int rows_per_cta) {
    int r = blockIdx.x * rows_per_cta + threadIdx.x / half;
    int p = threadIdx.x % half;
    if (r >= seqlen * heads || p >= half) {
        return;
    }
    int s = r / heads;
    unsigned short* xr = x + (size_t)r * 2 * half;
    float theta = (float)s * powf(eta, -2.f * (float)p / (2.f * (float)half));
    float c = cosf(theta), sn = sinf(theta);
    float a = hbits_to_f32(xr[p]);
    float b = hbits_to_f32(xr[p + half]);
    xr[p] = f32_to_hbits(a * c - b * sn);
    xr[p + half] = f32_to_hbits(a * sn + b * c);
}

// 批 embed 行拷贝（grid = rows，block = 256；每行与 gather_row 相同）。
extern "C" __global__ void gather_rows_f16(const unsigned short* __restrict__ src,
                                           unsigned short* __restrict__ dst,
                                           const unsigned int* __restrict__ toks,
                                           int rows, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * n) {
        return;
    }
    int row = i / n;
    int col = i % n;
    dst[i] = src[(size_t)toks[row] * n + col];
}

// 批 KV 写：token s → 页 (s/block_len, s%block_len)；与 kv_write_row 逐
// token 同地址同值（页序 = 逐 token 路径的 li*pp + s/32——page_base 由
// 调用方传 li*pp，与 kv_write_row 的显式 phys 参数语义一致）。
// grid = seqlen × per_tok/256；block = 256。
extern "C" __global__ void kv_write_seq_rows(const unsigned short* __restrict__ k_rows,
                                             const unsigned short* __restrict__ v_rows,
                                             unsigned short* __restrict__ kv,
                                             int seqlen, int block_len,
                                             int kv_heads, int d, int page_base,
                                             int total_pages) {
    int per_tok = kv_heads * d;
    int blocks_per_tok = (per_tok + 255) / 256;
    int b = blockIdx.x;
    int s = b / blocks_per_tok;
    if (s >= seqlen) {
        return;
    }
    int chunk = b % blocks_per_tok;
    int i = chunk * 256 + threadIdx.x;
    if (i >= per_tok) {
        return;
    }
    int phys = page_base + s / block_len;
    int off = s % block_len;
    int kh = i / d;
    int di = i % d;
    int tok_base = ((phys * block_len + off) * kv_heads + kh) * d + di;
    size_t k_region = (size_t)total_pages * block_len * per_tok;
    kv[tok_base] = k_rows[(size_t)s * per_tok + i];
    kv[k_region + tok_base] = v_rows[(size_t)s * per_tok + i];
}

// S1-7 fused QKV: single cast from the fused [s x (nqk+2kvk)] f32 GEMM
// output into three contiguous per-section f16 buffers ([s*nqk] for q,
// [s*kvk] for k and v) — the exact layout the separated 3x-GEMM path
// produces, so every downstream kernel (rms_heads / rope / scale / FMHA /
// kv_write) is layout-agnostic. Element conversion is bit-identical to
// cast_f32_to_f16 (same truncation path above; no round bit in either).
// grid = (ceil((nqk+2kvk)/256), s); block = 256.
extern "C" __global__ void cast_split_qkv_f16(const float* __restrict__ c,
                                              unsigned short* __restrict__ q,
                                              unsigned short* __restrict__ k,
                                              unsigned short* __restrict__ v,
                                              int nqk, int kvk) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int n = nqk + 2 * kvk;
    if (col >= n) {
        return;
    }
    int row = blockIdx.y;
    const float* cr = c + (size_t)row * n;
    if (col < nqk) {
        q[(size_t)row * nqk + col] = f32_to_hbits(cr[col]);
    } else if (col < nqk + kvk) {
        k[(size_t)row * kvk + (col - nqk)] = f32_to_hbits(cr[col]);
    } else {
        v[(size_t)row * kvk + (col - nqk - kvk)] = f32_to_hbits(cr[col]);
    }
}
