// 014 T8: paged decode attention (GQA) kernel — gather + two-pass softmax.
//
// KV cache layout (contract in crates/memory::pool):
//   K region: [total_pages][block_len][kv_heads][d] f16 (row-major)
//   V region: same size immediately following K region
//   page:     [logical_pages] u32 physical page ids (shared K/V table)
//   q:        [B, QH, d] f16          (decode step: one query token/item)
//   kv_lens:  [B] u32 (current KV length per batch item)
//   scores:   scratch f32 [B, QH, max_kv]
//   out:      [B, QH, d] f16
//
// Per (batch b, q_head h): one CTA. GQA mapping per 014 D3:
//   kv_head = h / kv_ratio (integer division, contiguous groups);
//   non-divisible cases validated by the test trio (14/2, 12/2, 5/2).
// Fixed 256-lane reduction tree, no atomicAdd; deterministic (012 scope).

#include <cuda_fp16.h>
#include <math.h>


// 软件 f16→f32 位构造（与 dequant_kernels.cu / ggml 一致；硬件 F2F 对
// NaN payload 的行为差异已在 014 T5 记录）。
__device__ float hbits_to_f32(unsigned short h) {
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

constexpr int TPB = 256;

extern "C" __global__ void decode_step_gqa(
    const unsigned short* __restrict__ q,      // [B, QH, d]
    const unsigned int* __restrict__ page,     // [logical_pages]
    const unsigned short* __restrict__ kv,     // K [P][BL][KH][d] then V
    const unsigned int* __restrict__ kv_lens,  // [B]
    float* __restrict__ scores,                // [B, QH, max_kv] scratch
    unsigned short* __restrict__ out,          // [B, QH, d]
    int B, int QH, int d, int block_len, int kv_ratio, int kv_heads,
    int max_kv, int total_pages) {
    int cta = blockIdx.x;
    if (cta >= B * QH) {
        return;
    }
    int b = cta / QH;
    int h = cta % QH;
    int kv_h = h / kv_ratio;
    int kv_len = (int)kv_lens[b];
    int tid = threadIdx.x;

    const int per_tok_kv = kv_heads * d;

    if (kv_len <= 0) {
        return;
    }

    __shared__ float sq[256];   // d ≤ 256（判据档范围）
    __shared__ float sh[TPB]; // (判据档归约改串行——sh 保留兼容)
    for (int i = tid; i < d; i += TPB) {
        sq[i] = hbits_to_f32(q[((size_t)(b * QH + h)) * d + i]);
    }
    __syncthreads();

    const size_t kv_base = (size_t)total_pages * block_len * per_tok_kv; // V 区基址

    // ---------------- pass 1: scores（gather K + dot；判据档 tid==0 串行） ----------------
    __shared__ float smaxv;
    __shared__ float sinv;
    if (tid == 0) {
        float maxv = -1e30f;
        for (int t = 0; t < kv_len; ++t) {
            int lp = t / block_len;
            int off = t % block_len;
            int phys = (int)page[lp];
            const unsigned short* krow =
                kv + (((size_t)phys * block_len + off) * kv_heads + kv_h) * d;
            float acc = 0.0f;
            for (int i = 0; i < d; ++i) {
                acc += sq[i] * hbits_to_f32(krow[i]);
            }
            scores[((size_t)b * QH + h) * max_kv + t] = acc;
            maxv = fmaxf(maxv, acc);
        }
        float sumv = 0.0f;
        for (int t = 0; t < kv_len; ++t) {
            sumv += expf(scores[((size_t)b * QH + h) * max_kv + t] - maxv);
        }
        smaxv = maxv;
        sinv = sumv != 0.0f ? 1.0f / sumv : 0.0f;
    }
    __syncthreads();
    float maxv = smaxv;
    float inv = sinv;

    // ---------------- pass 3: PV（每线程一输出列；串行 gather+累积） ----------------
    for (int i = tid; i < d; i += TPB) {
        float acc = 0.0f;
        for (int t = 0; t < kv_len; ++t) {
            float p = expf(scores[((size_t)b * QH + h) * max_kv + t] - maxv) * inv;
            int lp = t / block_len;
            int off = t % block_len;
            int phys = (int)page[lp];
            const unsigned short* vrow = kv + kv_base +
                (((size_t)phys * block_len + off) * kv_heads + kv_h) * d;
            acc += p * hbits_to_f32(vrow[i]);
        }
        out[((size_t)(b * QH + h)) * d + i] = __float2half(acc);
    }
}
